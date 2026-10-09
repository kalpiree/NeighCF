import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

from adapters.llmsrec_data import ExcludedCatalog, index_queries, load_canonical, prediction_record
from evaluate import evaluate_records
from integrations.resume import RestartManager, _atomic_copy, _atomic_json, _sync_dir, digest, file_digest, runtime_context
from integrations.traditional.source import CLASSES, COMMIT, FILES, URL, model_class, verified_sources


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('value must be positive')
    return number


def protocol(metadata, explicit):
    value = explicit or metadata.get('protocol_sha256')
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('provide a lowercase common protocol SHA256 in splits.json or --protocol-sha256')
    if explicit and metadata.get('protocol_sha256') not in (None, explicit):
        raise ValueError('explicit protocol SHA256 differs from splits.json')
    return value


def output_path(path, *inputs):
    absolute = Path(path).absolute()
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ValueError('output path must not traverse symlinks')
    resolved = absolute.resolve()
    package = Path(__file__).resolve().parents[2]
    for source in (package, *map(lambda p: Path(p).resolve(), inputs)):
        if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise ValueError('outputs must be separate from code, source and dataset directories')
    return resolved


def data_bundle(args):
    items, rows, metadata, hashes = load_canonical(args.data, args.expected_candidates)
    mappings, indexed = index_queries(items, rows, args.max_length)
    if args.model == 'caser':
        trained = rows['train'].user_ids
        if not (rows['valid'].user_ids | rows['test'].user_ids) <= trained:
            raise ValueError('Caser evaluation users must have training queries')
    return {'items': items, 'rows': indexed, 'mappings': mappings, 'hashes': hashes,
            'protocol_sha256': protocol(metadata, args.protocol_sha256)}


def settings(args, bundle):
    return {'item_size': len(bundle['items']) + 1,
            'num_users': len(bundle['mappings']['user_to_index']) + 1,
            'hidden_size': args.dimension, 'gru_hidden_size': args.dimension,
            'batch_size': args.batch_size, 'max_seq_length': args.max_length,
            'num_hidden_layers': 2, 'num_attention_heads': 2, 'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.5, 'attention_probs_dropout_prob': 0.5,
            'initializer_range': 0.02, 'nh': 8, 'nv': 4, 'reg_weight': 1e-4}


def seed_all(seed, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable')
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def tensors(rows, device):
    import torch
    sequences = torch.tensor([r['sequence_item_indices'] for r in rows], device=device)
    users = torch.tensor([r['user_index'] for r in rows], device=device)
    return sequences, users


def negatives(rows, item_count):
    result = []
    for row in rows:
        excluded = {i - 1 for i in row['history_item_indices']}
        excluded.add(row['target_item_index'] - 1)
        catalog = ExcludedCatalog(range(1, item_count + 1), excluded)
        if not len(catalog):
            raise ValueError('no unobserved training negative remains for ' + row['source']['query_id'])
        result.append(catalog[random.randrange(len(catalog))])
    return result


def score(model, rows, batch_size, device):
    import torch
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = [rows[i] for i in range(start, min(start + batch_size, len(rows)))]
            sequences, users = tensors(batch, device)
            hidden = model.predict(sequences, users)[:, -1, :]
            candidates = torch.tensor([r['candidate_item_indices'] for r in batch], device=device)
            scores = (model.item_embeddings(candidates) * hidden.unsqueeze(1)).sum(-1)
            for row, values in zip(batch, scores.cpu().tolist()):
                yield prediction_record(row, values)


def metadata(args, bundle):
    return {'model': CLASSES[args.model].removesuffix('Model'), 'dataset': args.dataset,
            'seed': args.seed, 'protocol_sha256': bundle['protocol_sha256'],
            'implementation': 'BSARec native model and loss with canonical-query adapter v1'}


def save_tensor(path, value):
    import torch
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_dir(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def train(args, bundle):
    import torch
    if args.dimension % 2:
        raise ValueError('dimension must be divisible by the two native attention heads')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning rate must be finite and positive')
    out = output_path(args.output, args.source, args.data)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError('new training requires an empty output directory')
    seed_all(args.seed, args.device)
    native_settings = settings(args, bundle)
    model = model_class(args.source, args.model)(SimpleNamespace(**native_settings)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    package = Path(__file__).resolve().parents[2]
    contract = {
        'kind': 'traditional-canonical-v1', 'native_source': {'url': URL, 'commit': COMMIT, 'files': FILES},
        'adapter_source_sha256': {name: file_digest(package / name) for name in (
            'integrations/traditional/run.py', 'integrations/traditional/source.py',
            'integrations/resume.py', 'adapters/llmsrec_data.py', 'evaluate.py')},
        'dataset_sha256': bundle['hashes'], 'id_map_sha256': digest(bundle['mappings']),
        'metadata': metadata(args, bundle), 'model': args.model, 'native_settings': native_settings,
        'epochs': args.epochs, 'learning_rate': args.learning_rate, 'scheduler': None,
        'expected_candidates': args.expected_candidates, 'runtime': runtime_context(args.device),
        'training_protocol': 'one canonical train query, one uniform negative excluding full history and target',
        'selection': 'maximum validation NDCG@10; earliest epoch wins ties',
    }
    with RestartManager(out, contract, resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:
        save = lambda path: torch.save(model.state_dict(), path)
        restore = lambda path: model.load_state_dict(torch.load(path, map_location=args.device, weights_only=True))
        progress = manager.restore(restore, optimizer) or {
            'phase': 'train', 'epoch': 0, 'cursor': 0, 'steps': 0,
            'best_epoch': None, 'best_ndcg10': None, 'validation': [],
        }
        def checkpoint(force=False):
            artifacts = {'best.pt': out / 'best.pt'} if progress['best_epoch'] is not None else {}
            if progress['phase'] == 'complete':
                artifacts['evaluation.json'] = out / 'evaluation.json'
                artifacts['predictions.jsonl'] = out / 'predictions.jsonl'
            manager.checkpoint(progress, save, optimizer, artifacts=artifacts, force=force)
        if not args.resume:
            _atomic_json(out / 'run.json', contract)
            checkpoint(True)
        if progress['phase'] == 'complete':
            return json.loads((out / 'evaluation.json').read_text())
        train_rows = bundle['rows']['train']
        while progress['epoch'] < args.epochs:
            epoch = progress['epoch']
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(train_rows), generator=generator).tolist()
            model.train()
            for start in range(progress['cursor'], len(order), args.batch_size):
                batch = [train_rows[i] for i in order[start:start + args.batch_size]]
                sequences, users = tensors(batch, args.device)
                targets = torch.tensor([r['target_item_index'] for r in batch], device=args.device)
                negative = torch.tensor(negatives(batch, len(bundle['items'])), device=args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.calculate_loss(sequences, targets, negative, None, users)
                if not torch.isfinite(loss):
                    raise ValueError('non-finite native training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('non-finite native gradient')
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError('non-finite native parameter')
                progress.update(cursor=min(start + args.batch_size, len(order)), steps=progress['steps'] + 1)
                checkpoint()
            checkpoint(True)
            result, _ = evaluate_records(score(model, bundle['rows']['valid'], args.batch_size, args.device),
                                         metadata(args, bundle), args.expected_candidates)
            value = result['metrics']['NDCG@10']
            progress['validation'].append({'epoch': epoch + 1, 'metrics': result['metrics']})
            if progress['best_ndcg10'] is None or value > progress['best_ndcg10']:
                save_tensor(out / 'best.pt', model.state_dict())
                progress.update(best_ndcg10=value, best_epoch=epoch + 1)
            progress.update(epoch=epoch + 1, cursor=0)
            checkpoint(True)
        last = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        restore(out / 'best.pt')
        result = write_evaluation(out, model, bundle, args)
        model.load_state_dict(last)
        progress['phase'] = 'complete'
        checkpoint(True)
        return result


def write_evaluation(out, model, bundle, args):
    records = score(model, bundle['rows']['test'], args.batch_size, args.device)
    descriptor, temporary = tempfile.mkstemp(prefix='.predictions-', dir=out)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            def emitted():
                for row in records:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                    yield row
            result, _ = evaluate_records(emitted(), metadata(args, bundle), args.expected_candidates)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out / 'predictions.jsonl')
        _sync_dir(out)
        result['inputs'] = {'predictions_sha256': file_digest(out / 'predictions.jsonl'),
                            'selected_model_sha256': file_digest(out / 'best.pt')}
        _atomic_json(out / 'evaluation.json', result)
        return result
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def evaluate(args, bundle):
    import torch
    run = Path(args.run).resolve()
    out = output_path(args.output, args.data, args.source, run)
    if out.exists():
        raise ValueError('evaluation output must be a new directory')
    identity = json.loads((run / '.restart/identity.json').read_text())
    contract = identity['contract']
    if identity['sha256'] != digest(contract) or contract['kind'] != 'traditional-canonical-v1':
        raise ValueError('invalid run contract')
    if (contract['dataset_sha256'] != bundle['hashes'] or contract['model'] != args.model
            or contract['metadata'] != metadata(args, bundle)
            or contract['native_settings'] != settings(args, bundle)
            or contract['expected_candidates'] != args.expected_candidates):
        raise ValueError('evaluation arguments differ from the saved run')
    seed_all(args.seed, args.device)
    package = Path(__file__).resolve().parents[2]
    if (contract['runtime'] != runtime_context(args.device)
            or any(file_digest(package / name) != expected
                   for name, expected in contract['adapter_source_sha256'].items())):
        raise ValueError('saved runtime or adapter source differs')
    with (run / '.restart/lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        pointer = json.loads((run / '.restart/latest.json').read_text())
        name = pointer['generation']
        if Path(name).name != name or not name.startswith('generation-'):
            raise ValueError('invalid generation pointer')
        generation = run / '.restart' / name
        manifest_path = generation / 'manifest.json'
        if file_digest(manifest_path) != pointer['manifest_sha256']:
            raise ValueError('checkpoint manifest checksum differs')
        manifest = json.loads(manifest_path.read_text())
        if manifest['contract_sha256'] != identity['sha256']:
            raise ValueError('checkpoint contract differs')
        for relative, expected in manifest['files'].items():
            path = generation / relative
            if not path.resolve().is_relative_to(generation.resolve()) or path.is_symlink() or file_digest(path) != expected:
                raise ValueError('checkpoint artifact checksum differs')
        progress = json.loads((generation / 'progress.json').read_text())
        if progress['phase'] != 'complete' or 'artifacts/best.pt' not in manifest['files']:
            raise ValueError('standalone evaluation requires a completed run')
        model = model_class(args.source, args.model)(SimpleNamespace(**contract['native_settings'])).to(args.device)
        best = generation / 'artifacts/best.pt'
        model.load_state_dict(torch.load(best, map_location=args.device, weights_only=True))
        out.mkdir(parents=True)
        _atomic_copy(best, out / 'best.pt')
        return write_evaluation(out, model, bundle, args)


def parser():
    command = argparse.ArgumentParser()
    command.add_argument('command', choices=('audit', 'train', 'evaluate'))
    command.add_argument('--model', required=True, choices=tuple(CLASSES))
    command.add_argument('--source', required=True, type=Path)
    command.add_argument('--data', required=True, type=Path)
    command.add_argument('--dataset', required=True)
    command.add_argument('--output', type=Path)
    command.add_argument('--run', type=Path)
    command.add_argument('--protocol-sha256')
    command.add_argument('--expected-candidates', type=positive, default=100)
    command.add_argument('--max-length', type=positive, default=50)
    command.add_argument('--dimension', type=positive, default=64)
    command.add_argument('--batch-size', type=positive, default=256)
    command.add_argument('--learning-rate', type=float, default=0.001)
    command.add_argument('--epochs', type=positive)
    command.add_argument('--seed', type=int, default=42)
    command.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    command.add_argument('--checkpoint-seconds', type=float, default=600)
    command.add_argument('--resume', action='store_true')
    return command


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.command != 'audit' and args.output is None:
        arguments.error('--output is required')
    if args.command == 'train' and args.epochs is None:
        arguments.error('--epochs is required; no epoch budget is inferred')
    if args.command == 'evaluate' and args.run is None:
        arguments.error('--run is required')
    if not 0 <= args.seed < 2 ** 32:
        arguments.error('--seed must be in [0, 2**32)')
    verified_sources(args.source)
    bundle = data_bundle(args)
    if args.command == 'audit':
        result = {'native_source': {'url': URL, 'commit': COMMIT, 'files': FILES},
                  'dataset_sha256': bundle['hashes'], 'protocol_sha256': bundle['protocol_sha256'],
                  'queries': {key: len(value) for key, value in bundle['rows'].items()},
                  'native_settings': settings(args, bundle)}
    elif args.command == 'train':
        result = train(args, bundle)
    else:
        result = evaluate(args, bundle)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
