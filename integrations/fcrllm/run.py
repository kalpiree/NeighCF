from __future__ import annotations
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from data import DataError, batches, cosine_multiplier, digest, file_digest, load_bundle, prediction_record, read_rows, indexed_rows, sequence_values, training_negative, validate_features, versions, write_json
from evaluate import evaluate_records
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resume import RestartManager, runtime_context
NATIVE_FILES = ('models/FlipClassroom.py', 'models/Dualview_model.py', 'models/SASRec.py', 'models/GRU4Rec.py', 'models/BaseModel.py', 'models/utils.py')

def source_hashes(source):
    return {name: file_digest(source / 'FCRLLM' / name) for name in NATIVE_FILES}

def native_args(args):
    return SimpleNamespace(dataset='canonical', model_name='fcrllm_sasrec', encoder='sasrec', hidden_size=64, trm_num=2, num_heads=1, num_layers=1, dropout_rate=args.dropout, max_len=args.max_history, freeze=True, gamma_fc=args.gamma_fc, beta_hopfield=args.beta_hopfield, alpha_hopfield=args.alpha_hopfield, gamma_update=args.gamma_update, num_hopblock=args.num_hopblock, hyb_alpha=args.hyb_alpha)

def create_model(args, bundle):
    sys.path.insert(0, str((args.source / 'FCRLLM').resolve()))
    dual = importlib.import_module('models.Dualview_model')
    official = importlib.import_module('models.FlipClassroom')
    expected_path = (args.source / 'FCRLLM/models/FlipClassroom.py').resolve()
    if Path(official.__file__).resolve() != expected_path:
        raise DataError('Python imported a different models package')
    original = dual._load_pretrained_embedding

    def artifact_loader(path, pad_size=1):
        filename = {'itm_emb_np.pkl': 'item_embeddings.npy', 'pca64_itm_emb_np.pkl': 'pca64.npy'}.get(Path(path).name)
        if filename is None:
            raise DataError('unexpected native embedding request')
        array = np.load(args.features / filename, mmap_mode='r', allow_pickle=False)
        expected_dim = 64 if filename == 'pca64.npy' else args.feature_manifest['shape'][1]
        if array.shape != (len(bundle['items']), expected_dim):
            raise DataError('native item embedding dimensions differ')
        result = torch.zeros((len(array) + 1 + pad_size, array.shape[1]), dtype=torch.float32)
        for start in range(0, len(array), 4096):
            stop = min(start + 4096, len(array))
            result[start + 1:stop + 1] = torch.from_numpy(np.array(array[start:stop], dtype=np.float32, copy=True))
        return result
    dual._load_pretrained_embedding = artifact_loader
    try:
        model = official.FlipClassEnsemble(0, len(bundle['items']), args.device, native_args(args)).to(args.device)
    finally:
        dual._load_pretrained_embedding = original
    return model

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_tensors(rows, args, bundle, epoch=None):
    sequences, positions = zip(*(sequence_values(row, bundle, args.max_history) for row in rows))
    result = {'seq': torch.tensor(sequences, dtype=torch.long, device=args.device), 'positions': torch.tensor(positions, dtype=torch.long, device=args.device)}
    if epoch is not None:
        positive = torch.zeros_like(result['seq'])
        negative = torch.zeros_like(result['seq'])
        positive[:, -1] = torch.tensor([bundle['item_to_index'][r['target_id']] for r in rows], device=args.device)
        negative[:, -1] = torch.tensor([training_negative(r, bundle, args.seed, epoch) for r in rows], device=args.device)
        result.update(pos=positive, neg=negative)
    return result

def score(model, rows, args, bundle, manager=None, stage=None, checkpoint=None):
    model.eval()
    groups = list(batches(len(rows), args.eval_batch_size))
    journal = manager.scoring(stage, [[rows[i]['query_id'] for i in indices] for indices in groups]) if manager else None
    records = journal.records(require_complete=False) if journal else []
    with torch.inference_mode():
        for batch_index, indices in enumerate(groups):
            if journal and batch_index < journal.completed_batches:
                continue
            selected = [rows[i] for i in indices]
            values = make_tensors(selected, args, bundle)
            values['item_indices'] = torch.tensor([[bundle['item_to_index'][i] for i in row['candidate_ids']] for row in selected], device=args.device)
            scores = model.predict(**values).float().cpu().tolist()
            current = [prediction_record(row, value) for row, value in zip(selected, scores)]
            records.extend(current)
            if journal:
                journal.append(batch_index, current)
                checkpoint()
    return journal.records() if journal else records

def persist(prefix, records, manifest, candidates):
    with Path(str(prefix) + '.predictions.jsonl').open('w') as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    result, ranks = evaluate_records(records, manifest, candidates)
    write_json(str(prefix) + '.metrics.json', result)
    write_json(str(prefix) + '.ranks.json', ranks)
    return result

def run_manifest(args, bundle):
    return {'model': 'FCRLLM-Llama2-controlled', 'dataset': args.dataset_name or args.data.name, 'seed': args.seed, 'protocol_sha256': bundle['protocol_sha256'], 'run_kind': args.command, 'diagnostic': args.command != 'train', 'runtime': {'device': args.device, 'threads': args.threads, 'versions': versions()}, 'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in {'feature_manifest', 'resume', 'checkpoint_seconds'}}, 'comparison_scope': 'reconstruction of published NeighCF protocol; original frozen splits unavailable', 'method_variant': 'native FCRLLM FlipClassEnsemble with controlled semantic-encoder adaptation', 'semantic_encoder': args.feature_manifest['encoder'], 'feature_manifest_sha256': file_digest(args.features / 'manifest.json'), 'expected_candidates': args.expected_candidates, 'feature_protocol_rebind': {'enabled': args.feature_protocol_rebind, 'feature_protocol_sha256': args.feature_manifest['contract']['protocol_sha256'], 'evaluation_protocol_sha256': bundle['protocol_sha256'], 'candidate_derivation': bundle['split_metadata'].get('candidate_derivation')}, 'source_file_sha256': bundle['source_file_sha256'], 'mapping_sha256': bundle['mapping_sha256'], 'native_source_sha256': source_hashes(args.source), 'integration_source_sha256': {name: file_digest(Path(__file__).parent / name) for name in ('data.py', 'features.py', 'run.py')}, 'shared_source_sha256': {name: file_digest(Path(__file__).resolve().parents[2] / name) for name in ('integrations/resume.py', 'evaluate.py', 'adapters/llmsrec_data.py')}, 'architecture': vars(native_args(args)), 'history_limit': args.max_history, 'full_input_history_retained': True, 'training': {'epochs': args.epochs, 'batch_size': args.batch_size, 'optimizer': 'Adam', 'lr': args.lr, 'weight_decay': args.weight_decay, 'schedule': 'linear warmup then cosine to zero', 'warmup_ratio': args.warmup_ratio, 'objective': 'native BCE at canonical final target plus native bidirectional full-batch Hopfield MSE', 'microbatch_accumulation': False, 'final_partial_batch': 'retained at actual size', 'negative_sampling': 'one deterministic negative per query/epoch, excludes full observed history and target only', 'budget': 'fixed budget; no convergence claim'}, 'selection_metric': args.selection_metric, 'adaptations': ['local Llama2 title embeddings replace native external embedding service when controlled feature mode is selected', 'training-only PCA fit; held-out-only catalog items projected without refitting', 'canonical chronological 80/10/10 multiple queries replace native leave-one-out data loader', 'one target loss per canonical query; full native alignment at every sequence position including native padding behavior', f'fixed {args.expected_candidates} evaluation candidates replace native resampling; validation-only checkpoint selection', 'five-epoch common budget and warmup/cosine replace native 200-epoch example', 'native unused similar-user loader artifact is omitted; no forward/loss neighbor input is used'], 'source_paper': 'https://doi.org/10.1145/3774904.3792565', 'source_repository': 'https://github.com/Byungmoon-Heo/FCRLLM'}

def train(args, bundle, manifest):
    rows = indexed_rows(bundle, 'train')
    valid = indexed_rows(bundle, 'valid')
    if args.command == 'pilot':
        valid = valid[:args.pilot_valid_queries]
    epochs = 1 if args.command == 'pilot' else args.epochs
    steps_per_epoch = math.ceil(len(rows) / args.batch_size)
    if args.command == 'pilot':
        steps_per_epoch = min(steps_per_epoch, args.pilot_steps)
    total_steps = epochs * steps_per_epoch
    warmup = min(total_steps, math.ceil(total_steps * args.warmup_ratio))
    model = create_model(args, bundle)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = args.output / 'best.pt'
    state = {'phase': 'train', 'epoch': 1, 'next_batch': 0, 'step': 0, 'best': None, 'selected_epoch': None, 'losses': 0.0, 'example_count': 0, 'actual_batch_sizes': [], 'training_queries_seen': [], 'training_query_updates': 0, 'epoch_events': []}
    contract = {'manifest': manifest, 'runtime': runtime_context(args.device)}
    with RestartManager(args.output, contract, resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:

        def save_model(path):
            torch.save({'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()}, 'manifest_sha256': digest(manifest), 'epoch': state['epoch']}, path)

        def restore_model(path):
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['manifest_sha256'] != digest(manifest):
                raise DataError('restart model belongs to a different run')
            model.load_state_dict(saved['state_dict'], strict=True)
        restored_state = manager.restore(restore_model, optimizer)
        if restored_state is not None:
            state = restored_state
        actual_batch_sizes = set(state['actual_batch_sizes'])
        training_queries_seen = set(state['training_queries_seen'])

        def snapshot(force=False):
            state['actual_batch_sizes'] = sorted(actual_batch_sizes)
            state['training_queries_seen'] = sorted(training_queries_seen)
            names = ('best.pt', 'checkpoint.json', 'valid-best.predictions.jsonl', 'valid-best.metrics.json', 'valid-best.ranks.json', 'progress.jsonl', 'completion.json', 'test.predictions.jsonl', 'test.metrics.json', 'test.ranks.json')
            artifacts = {name: args.output / name for name in names if (args.output / name).is_file()}
            return manager.checkpoint(state, save_model, optimizer, artifacts=artifacts, force=force)
        if restored_state is None:
            snapshot(force=True)
        if state['phase'] == 'complete':
            completion = json.loads((args.output / 'completion.json').read_text())
            if completion.get('status') != 'complete' or completion.get('checkpoint_sha256') != file_digest(checkpoint_path):
                raise DataError('completed restart receipt no longer matches selected model')
            print(json.dumps(completion), flush=True)
            return
        for epoch in range(state['epoch'], epochs + 1):
            if state['phase'] not in {'train', 'validation'}:
                break
            state['epoch'] = epoch
            if not 0 <= state['next_batch'] <= steps_per_epoch or state['step'] != (epoch - 1) * steps_per_epoch + state['next_batch']:
                raise DataError('restart optimizer step differs from deterministic epoch batch cursor')
            model.train()
            if state['phase'] == 'train':
                order = batches(len(rows), args.batch_size, args.seed + epoch)
                for batch_index, indices in enumerate(order):
                    if batch_index < state['next_batch']:
                        continue
                    if args.command == 'pilot' and batch_index >= args.pilot_steps:
                        break
                    batch = [rows[i] for i in indices]
                    lr = args.lr * cosine_multiplier(state['step'], total_steps, warmup)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**make_tensors(batch, args, bundle, epoch))
                    if not torch.isfinite(loss):
                        raise DataError('native FCRLLM loss is nonfinite')
                    loss.backward()
                    gradient_count = 0
                    for name, value in model.named_parameters():
                        if value.grad is not None:
                            gradient_count += 1
                            if not torch.isfinite(value.grad).all():
                                raise DataError(f'nonfinite gradient: {name}')
                    if not gradient_count:
                        raise DataError('native objective produced no gradients')
                    optimizer.step()
                    actual_batch_sizes.add(len(batch))
                    training_queries_seen.update((row['query_id'] for row in batch))
                    state['training_query_updates'] += len(batch)
                    state['losses'] += loss.item() * len(batch)
                    state['example_count'] += len(batch)
                    state['step'] += 1
                    state['next_batch'] = batch_index + 1
                    if batch_index == 0 or (batch_index + 1) % 25 == 0:
                        print(json.dumps({'epoch': epoch, 'batch': batch_index + 1, 'batches': steps_per_epoch, 'loss': loss.item(), 'lr': lr, 'actual_batch_size': len(batch)}), flush=True)
                    snapshot()
                state['phase'] = 'validation'
                snapshot(force=True)
            records = score(model, valid, args, bundle, manager, f'validation-epoch-{epoch}', snapshot)
            result, _ = evaluate_records(records, manifest, bundle['expected_candidates'])
            value = result['metrics'][args.selection_metric]
            event = {'epoch': epoch, 'loss': state['losses'] / state['example_count'], 'training_queries': state['example_count'], 'validation': result['metrics']}
            state['epoch_events'].append(event)
            (args.output / 'progress.jsonl').write_text(''.join((json.dumps(item) + '\n' for item in state['epoch_events'])))
            print(json.dumps(event), flush=True)
            if state['best'] is None or value > state['best']:
                state['best'], state['selected_epoch'] = (value, epoch)
                save_model(checkpoint_path)
                write_json(args.output / 'checkpoint.json', {'checkpoint_sha256': file_digest(checkpoint_path), 'manifest_sha256': digest(manifest), 'selected_epoch': epoch, 'validation_value': value, 'selection_metric': args.selection_metric})
                persist(args.output / 'valid-best', records, manifest, bundle['expected_candidates'])
            state.update(epoch=epoch + 1, next_batch=0, losses=0.0, example_count=0, phase='train' if epoch < epochs else 'restore')
            snapshot(force=True)
        if state['step'] != total_steps:
            raise DataError('training did not complete the declared optimizer-step budget')
        if args.command == 'train' and (len(training_queries_seen) != len(rows) or state['training_query_updates'] != epochs * len(rows)):
            raise DataError('training did not cover every query once per epoch')
        validate_features(bundle, args.features, args.feature_protocol_rebind)
        if source_hashes(args.source) != manifest['native_source_sha256']:
            raise DataError('native source changed during training')
        if any((file_digest(bundle['path'] / name) != value for name, value in bundle['source_file_sha256'].items())):
            raise DataError('frozen input files changed during training')
        saved_meta = json.loads((args.output / 'checkpoint.json').read_text())
        if saved_meta['checkpoint_sha256'] != file_digest(checkpoint_path):
            raise DataError('checkpoint hash mismatch')
        seed_all(args.seed + 1000003)
        restored = create_model(args, bundle)
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if saved['manifest_sha256'] != digest(manifest):
            raise DataError('checkpoint belongs to a different run')
        restored.load_state_dict(saved['state_dict'], strict=True)
        model = restored
        del saved
        state['phase'] = 'restore'
        snapshot(force=True)
        checked = score(restored, valid, args, bundle, manager, 'selected-validation-replay', snapshot)
        selected_records = [json.loads(line) for line in (args.output / 'valid-best.predictions.jsonl').read_text().splitlines() if line]
        previous = {row['query_id']: row for row in selected_records}
        restore_max_abs_error = 0.0
        for row in checked:
            old = previous[row['query_id']]
            if row['candidate_ids'] != old['candidate_ids'] or not np.allclose(row['scores'], old['scores'], atol=1e-05, rtol=0.0001):
                raise DataError('fresh checkpoint restoration changed candidate scores')
            restore_max_abs_error = max(restore_max_abs_error, float(np.max(np.abs(np.asarray(row['scores']) - np.asarray(old['scores'])))))
        completion = {'status': 'complete', 'run_kind': args.command, 'diagnostic': args.command != 'train', 'selected_epoch': state['selected_epoch'], 'validation_value': state['best'], 'optimizer_steps': state['step'], 'fresh_checkpoint_restore_verified': True, 'checkpoint_sha256': file_digest(checkpoint_path), 'restore_checked_queries': len(checked), 'restore_max_abs_error': restore_max_abs_error, 'restore_atol': 1e-05, 'restore_rtol': 0.0001, 'training_batch_sizes': sorted(actual_batch_sizes), 'training_queries': len(training_queries_seen), 'training_query_updates': state['training_query_updates'], 'validation_queries': len(valid), 'test_evaluations': 0, 'training_budget': 'fixed epochs, not a convergence claim', 'restart_supported': True}
        if args.command == 'train':
            state['phase'] = 'test'
            snapshot(force=True)
            test = indexed_rows(bundle, 'test')
            records = score(restored, test, args, bundle, manager, 'test', snapshot)
            persist(args.output / 'test', records, manifest, bundle['expected_candidates'])
            completion.update(test_evaluations=1, test_queries=len(records))
        if str(args.device).startswith('cuda'):
            completion['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
        write_json(args.output / 'completion.json', completion)
        state['phase'] = 'complete'
        snapshot(force=True)
        print(json.dumps(completion), flush=True)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit', 'pilot', 'train'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='resume only a matching durable checkpoint')
    p.add_argument('--checkpoint-seconds', type=float, default=600.0)
    p.add_argument('--dataset-name')
    p.add_argument('--protocol-sha256')
    p.add_argument('--expected-candidates', type=int, default=51)
    p.add_argument('--feature-protocol-rebind', action='store_true', help='reuse exact unchanged training/catalog features from the declared candidate-derivation parent protocol')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--max-history', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--num-hopblock', type=int, default=1)
    p.add_argument('--gamma-fc', type=float, default=1.0)
    p.add_argument('--beta-hopfield', type=float, default=0.3)
    p.add_argument('--alpha-hopfield', type=float, default=0.5)
    p.add_argument('--gamma-update', type=float, default=0.7)
    p.add_argument('--hyb-alpha', type=float, default=0.5)
    p.add_argument('--selection-metric', choices=('NDCG@10', 'HR@10'), default='NDCG@10')
    p.add_argument('--pilot-steps', type=int, default=2)
    p.add_argument('--pilot-valid-queries', type=int, default=16)
    return p

def main():
    args = parser().parse_args()
    if args.output.exists() and (not args.resume):
        raise DataError('output directory exists; use a new path or explicit verified --resume')
    if args.resume and (not args.output.is_dir() or args.command == 'audit'):
        raise DataError('resume requires an existing training/pilot output')
    if any((x < 1 for x in (args.epochs, args.batch_size, args.eval_batch_size, args.max_history, args.threads, args.pilot_steps, args.pilot_valid_queries))) or not 0 <= args.warmup_ratio < 1:
        raise DataError('invalid training settings')
    if args.lr <= 0 or not 0 <= args.dropout < 1 or (not 0 <= args.hyb_alpha <= 1):
        raise DataError('invalid optimizer/model settings')
    args.source, args.features, args.data = (args.source.resolve(), args.features.resolve(), args.data.resolve())
    if args.expected_candidates < 2:
        raise DataError('evaluation needs at least one positive and one negative')
    bundle = load_bundle(args.data, args.protocol_sha256, args.expected_candidates)
    args.feature_manifest = validate_features(bundle, args.features, args.feature_protocol_rebind)
    manifest = run_manifest(args, bundle)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if json.loads((args.output / 'run.metadata.json').read_text()) != manifest:
            raise DataError('run manifest changed; refusing resume')
    else:
        write_json(args.output / 'run.metadata.json', manifest)
    if args.command == 'audit':
        ids = set()
        counts = {}
        for split in ('train', 'valid', 'test'):
            count = 0
            for row in read_rows(bundle, split):
                if row['query_id'] in ids:
                    raise DataError('query ID reused across splits')
                ids.add(row['query_id'])
                count += 1
            counts[split] = count
        write_json(args.output / 'audit.json', {'status': 'passed', 'queries': counts, 'items': len(bundle['items'])})
        print(json.dumps({'status': 'passed', 'queries': counts}), flush=True)
        return
    global torch, np
    import torch
    import numpy as np
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    train(args, bundle, manifest)
if __name__ == '__main__':
    main()
