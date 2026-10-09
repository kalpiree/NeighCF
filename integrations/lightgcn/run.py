import argparse
import contextlib
import fcntl
import hashlib
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
from integrations.lightgcn.source import COMMIT, FILES, URL, model_class, sampler, verified_sources


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
    import numpy as np
    items, rows, metadata, hashes = load_canonical(args.data, args.expected_candidates)
    protocol_sha256 = protocol(metadata, args.protocol_sha256)
    manifest_path = Path(args.data) / 'manifest.json'
    if metadata.get('manifest', 'manifest.json') != 'manifest.json' or manifest_path.is_symlink():
        raise ValueError('canonical protocol manifest binding mismatch')
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != metadata.get('protocol_sha256') or manifest_sha256 != protocol_sha256:
        raise ValueError('canonical protocol manifest binding mismatch')
    manifest = json.loads(manifest_bytes)
    for name, actual in hashes.items():
        if name != 'splits.json' and manifest['output_file_sha256'].get(name) != actual:
            raise ValueError('canonical file differs from manifest: ' + name)
    counts = manifest['protocol']['candidates']
    if counts.get('evaluation_count', counts.get('count')) != args.expected_candidates:
        raise ValueError('evaluation candidate count differs from canonical protocol')
    hashes['manifest.json'] = manifest_sha256
    if metadata.get('dataset') not in (None, args.dataset):
        raise ValueError('dataset label differs from canonical metadata')
    mappings, indexed = index_queries(items, rows, 1)
    users = len(mappings['user_to_index'])
    positives = [set() for _ in range(users)]
    for row in indexed['train']:
        positives[row['user_index'] - 1].update(i - 1 for i in row['history_item_indices'])
        positives[row['user_index'] - 1].add(row['target_item_index'] - 1)
    if any(not p for p in positives):
        raise ValueError('LightGCN evaluation users require train-only graph edges')
    if any(len(p) == len(items) for p in positives):
        raise ValueError('no unobserved training negative remains')
    edges = [(u, i) for u, pos in enumerate(positives) for i in sorted(pos)]
    return {'items': items, 'rows': indexed, 'mappings': mappings, 'hashes': hashes,
            'protocol_sha256': protocol_sha256, 'edges': edges,
            'positives': [np.array(sorted(p), dtype=np.int64) for p in positives]}


def settings(args, bundle):
    return {'latent_dim_rec': args.dimension, 'lightGCN_n_layers': args.layers,
            'keep_prob': 1.0, 'dropout': 0, 'A_split': False, 'pretrain': 0,
            'batch_size': args.batch_size, 'weight_decay': args.weight_decay}


def native_dataset(bundle, device):
    import torch
    users, items = len(bundle['positives']), len(bundle['items'])
    left = [u for u, i in bundle['edges']]
    right = [users + i for u, i in bundle['edges']]
    indices = torch.tensor([left + right, right + left], dtype=torch.long)
    degree = torch.bincount(indices[0], minlength=users + items).to(torch.float64)
    values = (degree[indices[0]] * degree[indices[1]]).rsqrt().float()
    graph = torch.sparse_coo_tensor(indices, values, (users + items, users + items)).coalesce().to(device)
    return SimpleNamespace(n_users=users, m_items=items, allPos=bundle['positives'],
                           trainDataSize=len(bundle['edges']), getSparseGraph=lambda: graph)


def create_model(args, bundle):
    return model_class(args.source)(settings(args, bundle), native_dataset(bundle, args.device)).to(args.device)


def epoch_samples(args, bundle, epoch):
    sample, rng = sampler(args.source, (args.seed + epoch) % (2 ** 32))
    dataset = SimpleNamespace(n_users=len(bundle['positives']), m_items=len(bundle['items']),
                              allPos=bundle['positives'], trainDataSize=len(bundle['edges']))
    samples = sample(dataset)
    rng.shuffle(samples)
    return samples


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


def score(model, rows, batch_size, device):
    import torch
    model.eval()
    with torch.inference_mode():
        all_users, all_items = model.computer()
        for start in range(0, len(rows), batch_size):
            batch = [rows[i] for i in range(start, min(start + batch_size, len(rows)))]
            users = torch.tensor([r['user_index'] - 1 for r in batch], device=device)
            candidates = torch.tensor([[i - 1 for i in r['candidate_item_indices']] for r in batch], device=device)
            scores = model.f((all_items[candidates] * all_users[users].unsqueeze(1)).sum(-1))
            for row, values in zip(batch, scores.cpu().tolist()):
                yield prediction_record(row, values)


def metadata(args, bundle):
    return {'model': 'LightGCN', 'dataset': args.dataset,
            'seed': args.seed, 'protocol_sha256': bundle['protocol_sha256'],
            'implementation': 'official LightGCN native model, BPR and sampler; train-only graph and fixed-candidate adapter v1'}


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


def finite_optimizer_state(optimizer):
    import torch
    def check(value):
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():
                raise ValueError('non-finite native optimizer state')
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('non-finite native optimizer state')
    check(optimizer.state)


def train(args, bundle):
    import torch
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('weight decay must be finite and nonnegative')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning rate must be finite and positive')
    out = output_path(args.output, args.source, args.data)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError('new training requires an empty output directory')
    seed_all(args.seed, args.device)
    native_settings = settings(args, bundle)
    model = create_model(args, bundle)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    package = Path(__file__).resolve().parents[2]
    contract = {
        'kind': 'lightgcn-canonical-v1', 'native_source': {'url': URL, 'commit': COMMIT, 'files': FILES},
        'adapter_source_sha256': {name: file_digest(package / name) for name in (
            'integrations/lightgcn/run.py', 'integrations/lightgcn/source.py',
            'integrations/resume.py', 'adapters/llmsrec_data.py', 'evaluate.py')},
        'dataset_sha256': bundle['hashes'], 'id_map_sha256': digest(bundle['mappings']),
        'metadata': metadata(args, bundle), 'model': 'lightgcn', 'train_graph_sha256': digest(bundle['edges']), 'native_settings': native_settings,
        'epochs': args.epochs, 'learning_rate': args.learning_rate, 'scheduler': None,
        'expected_candidates': args.expected_candidates, 'runtime': runtime_context(args.device),
        'training_protocol': 'deduplicated train histories and targets only; native uniform user/positive/unobserved-negative sampling; seed+epoch local NumPy stream',
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
        while progress['epoch'] < args.epochs:
            epoch = progress['epoch']
            order = epoch_samples(args, bundle, epoch)
            model.train()
            for start in range(progress['cursor'], len(order), args.batch_size):
                batch = torch.tensor(order[start:start + args.batch_size], device=args.device)
                users, targets, negative = batch[:, 0], batch[:, 1], batch[:, 2]
                optimizer.zero_grad(set_to_none=True)
                ranking_loss, regularization = model.bpr_loss(users, targets, negative)
                loss = ranking_loss + args.weight_decay * regularization
                if not torch.isfinite(loss):
                    raise ValueError('non-finite native training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('non-finite native gradient')
                optimizer.step()
                finite_optimizer_state(optimizer)
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
    if identity['sha256'] != digest(contract) or contract['kind'] != 'lightgcn-canonical-v1':
        raise ValueError('invalid run contract')
    if (contract['dataset_sha256'] != bundle['hashes'] or contract['model'] != 'lightgcn'
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
        model = create_model(args, bundle)
        best = generation / 'artifacts/best.pt'
        model.load_state_dict(torch.load(best, map_location=args.device, weights_only=True))
        out.mkdir(parents=True)
        _atomic_copy(best, out / 'best.pt')
        return write_evaluation(out, model, bundle, args)


def parser():
    command = argparse.ArgumentParser()
    command.add_argument('command', choices=('audit', 'train', 'evaluate'))
    command.add_argument('--source', required=True, type=Path)
    command.add_argument('--data', required=True, type=Path)
    command.add_argument('--dataset', required=True)
    command.add_argument('--output', type=Path)
    command.add_argument('--run', type=Path)
    command.add_argument('--protocol-sha256')
    command.add_argument('--expected-candidates', type=positive, default=51)
    command.add_argument('--layers', type=positive, default=3)
    command.add_argument('--weight-decay', type=float, default=1e-4)
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
                  'native_settings': settings(args, bundle), 'train_graph_sha256': digest(bundle['edges']), 'train_edges': len(bundle['edges'])}
    elif args.command == 'train':
        result = train(args, bundle)
    else:
        result = evaluate(args, bundle)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
