import argparse
import json
import math
from pathlib import Path
import random
from evaluate import evaluate_records
from integrations.resume import RestartManager, _atomic_json, _rng_state, _restore_rng, digest, file_digest, runtime_context
from integrations.traditional.run import output_path, save_tensor, seed_all
from .common import record, source_identity


def finite_model(runtime, optimizer=None):
    import torch
    for name, value in runtime.model.named_parameters():
        if not torch.isfinite(value).all() or value.grad is not None and not torch.isfinite(value.grad).all():
            raise FloatingPointError('nonfinite model state: ' + name)
    for name, value in runtime.model.named_buffers():
        if not torch.isfinite(value).all():
            raise FloatingPointError('nonfinite model buffer: ' + name)
    if optimizer is not None:
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise FloatingPointError('nonfinite optimizer state: ' + name)


def metadata(args, bundle, runtime):
    return {'model': runtime.name, 'dataset': bundle['metadata']['dataset'], 'seed': args.seed, 'protocol_sha256': bundle['metadata']['protocol_sha256'], 'implementation': runtime.description}


def score(runtime, rows):
    import torch
    runtime.model.eval()
    with torch.inference_mode():
        for row in rows:
            values = runtime.scores(row)
            if not torch.isfinite(values).all():
                raise FloatingPointError('nonfinite native candidate score')
            yield record(row, values.detach().float().cpu().tolist())


def validate_training_arguments(args):
    import torch
    for name in ('epochs', 'batch_size', 'expected_candidates', 'training_candidates'):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(name + ' must be a positive integer')
    if type(args.seed) is not int or not 0 <= args.seed < 2 ** 32:
        raise ValueError('seed must fit the native NumPy seed range')
    for name in ('learning_rate', 'weight_decay', 'warmup_ratio', 'checkpoint_seconds'):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + ' must be finite')
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1 or args.checkpoint_seconds < 0:
        raise ValueError('invalid learning rate, weight decay, warmup ratio or checkpoint interval')
    if args.optimizer not in ('adam', 'adamw') or args.schedule not in ('constant', 'warmup_cosine'):
        raise ValueError('unsupported optimizer or schedule')
    torch.device(args.device)


def run_training(args, bundle, runtime):
    import torch
    validate_training_arguments(args)
    if runtime.training_size < 1:
        raise ValueError('training requires at least one native example')
    out = output_path(args.output, args.source, args.data, *runtime.input_paths)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError('new run output must be empty')
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    optimizer_type = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    optimizer = optimizer_type(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.schedule == 'warmup_cosine':
        if not 0 <= args.warmup_ratio < 1:
            raise ValueError('warmup ratio must be in [0,1)')
        steps = math.ceil(runtime.training_size / args.batch_size) * args.epochs
        warmup = int(steps * args.warmup_ratio)
        def scale(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            return .5 * (1 + math.cos(math.pi * min(1., max(0., (step - warmup) / max(1, steps - warmup)))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    contract = {'kind': 'native-baseline-canonical-v1', 'native': runtime.contract, 'adapter_sha256': source_identity(), 'dataset_sha256': bundle['hashes'], 'metadata': metadata(args, bundle, runtime), 'epochs': args.epochs, 'batch_size': args.batch_size, 'optimizer': args.optimizer, 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'schedule': args.schedule, 'warmup_ratio': args.warmup_ratio, 'runtime': runtime_context(args.device) | {'cpu_threads': torch.get_num_threads(), 'cpu_interop_threads': torch.get_num_interop_threads()}, 'training_candidates': bundle['training_candidates'], 'expected_candidates': bundle['expected_candidates'], 'selection': 'validation NDCG@10, earliest epoch on ties'}
    with RestartManager(out, contract, resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:
        save = lambda path: torch.save(runtime.model.state_dict(), path)
        restore = lambda path: runtime.model.load_state_dict(torch.load(path, map_location=args.device, weights_only=True), strict=True)
        progress = manager.restore(restore, optimizer, scheduler) or {'phase': 'train', 'epoch': 0, 'cursor': 0, 'updates': 0, 'best_metric': None, 'best_epoch': None, 'validation': []}
        finite_model(runtime, optimizer)
        def checkpoint(force=False):
            artifacts = {'best.pt': out / 'best.pt'} if progress['best_epoch'] is not None else {}
            if progress['phase'] == 'complete':
                artifacts.update({'evaluation.json': out / 'evaluation.json', 'predictions.jsonl': out / 'predictions.jsonl'})
            manager.checkpoint(progress, save, optimizer, scheduler, artifacts, force=force)
        if not args.resume:
            _atomic_json(out / 'run.json', contract)
            checkpoint(True)
        if progress['phase'] == 'complete':
            return json.loads((out / 'evaluation.json').read_text())
        while progress['epoch'] < args.epochs:
            order = list(range(runtime.training_size))
            random.Random(args.seed + progress['epoch']).shuffle(order)
            runtime.model.train()
            for cursor in range(progress['cursor'], len(order), args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                indices = order[cursor:cursor + args.batch_size]
                loss = runtime.loss(indices)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite native loss')
                loss.backward()
                finite_model(runtime, optimizer)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                finite_model(runtime, optimizer)
                progress.update(cursor=min(cursor + args.batch_size, len(order)), updates=progress['updates'] + 1)
                checkpoint()
            checkpoint(True)
            rng = _rng_state()
            try:
                result, _ = evaluate_records(score(runtime, bundle['rows']['valid']), metadata(args, bundle, runtime), bundle['expected_candidates'])
            finally:
                _restore_rng(rng)
            metric = result['metrics']['NDCG@10']
            progress['validation'].append({'epoch': progress['epoch'] + 1, 'metrics': result['metrics']})
            if progress['best_metric'] is None or metric > progress['best_metric']:
                save_tensor(out / 'best.pt', runtime.model.state_dict())
                progress.update(best_metric=metric, best_epoch=progress['epoch'] + 1)
            progress.update(epoch=progress['epoch'] + 1, cursor=0)
            checkpoint(True)
        last = {key: value.detach().cpu().clone() for key, value in runtime.model.state_dict().items()}
        rng = _rng_state()
        latest = manager._read_latest()
        if file_digest(out / 'best.pt') != latest['manifest']['files']['artifacts/best.pt']:
            raise ValueError('selected best model differs from full-state snapshot')
        restore(out / 'best.pt')
        journal = manager.scoring('selected-test', [[row['query_id']] for row in bundle['rows']['test']])
        complete = journal.completed_batches
        try:
            for index, row in enumerate(bundle['rows']['test']):
                if index >= complete:
                    journal.append(index, list(score(runtime, [row])))
                if manager.stop_requested:
                    runtime.model.load_state_dict(last, strict=True)
                    _restore_rng(rng)
                    checkpoint(True)
            records = list(journal.records())
            result, _ = evaluate_records(records, metadata(args, bundle, runtime), bundle['expected_candidates'])
            temporary = out / '.predictions.pending'
            with temporary.open('w') as stream:
                for row in records:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
                __import__('os').fsync(stream.fileno())
            __import__('os').replace(temporary, out / 'predictions.jsonl')
            result['inputs'] = {'selected_model_sha256': file_digest(out / 'best.pt'), 'predictions_sha256': file_digest(out / 'predictions.jsonl')}
            _atomic_json(out / 'evaluation.json', result)
        finally:
            runtime.model.load_state_dict(last, strict=True)
            _restore_rng(rng)
        progress['phase'] = 'complete'
        checkpoint(True)
        return result


def add_arguments(parser):
    parser.add_argument('command', choices=('audit', 'train'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-candidates', type=int, default=51)
    parser.add_argument('--training-candidates', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, required=True)
    parser.add_argument('--optimizer', choices=('adam', 'adamw'), required=True)
    parser.add_argument('--schedule', choices=('constant', 'warmup_cosine'), required=True)
    parser.add_argument('--warmup-ratio', type=float, default=.1)
    parser.add_argument('--checkpoint-seconds', type=float, default=600)
    parser.add_argument('--resume', action='store_true')
    return parser
