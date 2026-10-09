from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import importlib.metadata
import sys
import numpy as np
from .config import finite, positive_int
from .prompting import build_prompt
from .retrieval import canonical, digest, validate_query
from .scoring import CandidateLikelihoodScorer


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def preflight_output(output, protected=()):
    path = Path(os.path.abspath(output))
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError('output path or ancestor is a symlink')
    if path.exists() and not path.is_dir():
        raise ValueError('output must be a directory')
    protected_paths = [Path(__file__).resolve().parents[2], *[Path(item) for item in protected]]
    for source in protected_paths:
        source = source.resolve()
        source = source if source.is_dir() else source.parent
        if path == source or source in path.parents or path in source.parents:
            raise ValueError('output must be disjoint from input and source directories')
    if path.exists() and any(item.is_symlink() for item in path.iterdir()):
        raise ValueError('output contains symlink entries')
    return path


def runtime_identity(device):
    import torch
    versions = {}
    for name in ('transformers', 'tokenizers', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'torch': torch.__version__, 'python': sys.version, 'libraries': versions, 'device': str(device), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'cuda_devices': [{'name': torch.cuda.get_device_name(index), 'capability': torch.cuda.get_device_capability(index)} for index in range(torch.cuda.device_count())], 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32}


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted(Path(__file__).parent.glob('*.py')), root / 'evaluate.py']
    return {str(path.relative_to(root)): file_sha256(path) for path in paths}


def previous_path(path):
    return path.with_name(path.stem + '.previous' + path.suffix)


def checkpoint_receipt(path):
    return path.with_name(path.name + '.sha256.json')


def verified_checkpoint_path(path, allow_fallback=True):
    path = Path(path)
    errors = []
    for candidate in ((path, previous_path(path)) if allow_fallback else (path,)):
        try:
            receipt_path = checkpoint_receipt(candidate)
            if candidate.is_symlink() or receipt_path.is_symlink():
                raise ValueError('checkpoint paths cannot be symlinks')
            receipt = json.loads(receipt_path.read_text())
            if candidate.stat().st_size != receipt['size'] or file_sha256(candidate) != receipt['sha256']:
                raise ValueError('checkpoint byte checksum mismatch')
            return candidate
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    raise ValueError(f'no verified checkpoint or fallback: {errors}')


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    betas: list
    epsilon: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    backbone_mode: str
    train_gate: bool
    selection_metric: str
    checkpoint_every_updates: int
    max_gradient_norm: float | None

    def __post_init__(self):
        for name in ('epochs', 'batch_size', 'checkpoint_every_updates'):
            positive_int(getattr(self, name), name)
        for name in ('learning_rate', 'epsilon'):
            finite(getattr(self, name), name, 0)
            if getattr(self, name) == 0:
                raise ValueError(f'{name} must be positive')
        finite(self.weight_decay, 'weight_decay', 0)
        finite(self.warmup_ratio, 'warmup_ratio', 0, 1)
        if not isinstance(self.betas, list) or len(self.betas) != 2:
            raise ValueError('betas must contain exactly two numbers')
        for value in self.betas:
            finite(value, 'beta', 0, 1)
            if value == 1:
                raise ValueError('Adam beta cannot equal 1')
        if type(self.seed) is not int or type(self.train_gate) is not bool:
            raise ValueError('seed must be integer and train_gate boolean')
        if self.backbone_mode not in {'gate_only', 'full_model'}:
            raise ValueError('backbone_mode must be gate_only or full_model')
        if self.selection_metric not in {'HR@1', 'HR@5', 'HR@10', 'NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR'}:
            raise ValueError('unknown selection metric')
        if self.max_gradient_norm is not None:
            finite(self.max_gradient_norm, 'max_gradient_norm', 0)
            if self.max_gradient_norm == 0:
                raise ValueError('max_gradient_norm must be positive')

    @classmethod
    def from_dict(cls, raw):
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    def to_dict(self):
        return asdict(self)


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_checkpoint(path, value):
    import torch
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {'sha256': file_sha256(temporary), 'size': Path(temporary).stat().st_size}
        if path.exists():
            verified = verified_checkpoint_path(path, allow_fallback=True)
            if verified == path:
                previous = previous_path(path)
                previous_temporary = previous.with_name('.' + previous.name + '.pending')
                if previous_temporary.exists():
                    if previous_temporary.is_symlink():
                        raise ValueError('unexpected symlink checkpoint temporary')
                    previous_temporary.unlink()
                os.link(path, previous_temporary)
                os.replace(previous_temporary, previous)
                atomic_json(checkpoint_receipt(previous), json.loads(checkpoint_receipt(path).read_text()))
        os.replace(temporary, path)
        atomic_json(checkpoint_receipt(path), receipt)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state():
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG topology differs')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def model_identity(model, tokenizer):
    result = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        result.update(canonical([key, str(tensor.dtype), list(tensor.shape)]))
        result.update(tensor.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    tokenizer_identity = {'class': type(tokenizer).__qualname__, 'vocabulary': tokenizer.get_vocab(), 'special_tokens_map': getattr(tokenizer, 'special_tokens_map', {}), 'backend': tokenizer.backend_tokenizer.to_str() if hasattr(tokenizer, 'backend_tokenizer') else None}
    model_config = model.config.to_dict() if hasattr(model, 'config') else None
    return {'model_sha256': result.hexdigest(), 'tokenizer_sha256': digest(tokenizer_identity), 'model_class': type(model).__qualname__, 'model_config': model_config}


def metrics(records):
    if not records:
        raise ValueError('evaluation requires nonempty queries')
    result = {key: 0.0 for key in ('HR@1', 'HR@5', 'HR@10', 'NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR')}
    ranks = []
    for record in records:
        order = sorted(zip(record['candidate_ids'], record['scores']), key=lambda pair: (-pair[1], pair[0]))
        rank = next(index for index, (item, _) in enumerate(order, 1) if item == record['target_id'])
        ranks.append({'query_id': record['query_id'], 'user_id': record['user_id'], 'target_id': record['target_id'], 'target_rank': rank, 'candidate_count': len(order), 'has_score_ties': len({score for _, score in order}) != len(order)})
        for cutoff in (1, 5, 10):
            result[f'HR@{cutoff}'] += float(rank <= cutoff)
            result[f'NDCG@{cutoff}'] += 1 / math.log2(rank + 1) if rank <= cutoff else 0
        result['MRR'] += 1 / rank
    return {key: value / len(records) for key, value in result.items()}, ranks


def selected_neighbors(retriever, query, variant, mode, expected_candidate_count=None):
    return retriever.retrieve(query, variant=variant, soft_context=mode == 'soft_embedding_gate' and variant in {'full', 'global_pool'}, expected_candidate_count=expected_candidate_count)


def evaluate(model, tokenizer, gate, config, retriever, queries, device, variant):
    import torch
    state = rng_state()
    training = model.training
    scorer = CandidateLikelihoodScorer(model, tokenizer, config, device, gate)
    records = []
    model.eval()
    try:
        with torch.no_grad():
            for query in queries:
                found = selected_neighbors(retriever, query, variant, config.mode)
                prompt = build_prompt(query, retriever.items, found, config)
                scores = scorer.score(prompt)
                records.append({key: query[key] for key in ('query_id', 'user_id', 'target_id', 'candidate_ids')} | {'scores': scores})
        summary, ranks = metrics(records)
        return summary, ranks, records
    finally:
        model.train(training)
        restore_rng(state)


def check_finite_state(model, gate, optimizer=None):
    import torch
    for prefix, module in (('model', model), ('gate', gate)):
        if module is None:
            continue
        for key, parameter in module.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f'nonfinite {prefix} parameter {key}')
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f'nonfinite {prefix} gradient {key}')
    if optimizer is not None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise FloatingPointError(f'nonfinite optimizer state {key}')


def load_checkpoint(path, device='cpu'):
    import torch
    path = Path(path)
    verified = verified_checkpoint_path(path, allow_fallback=path.name.startswith('latest'))
    checkpoint = torch.load(verified, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != 'neighcf-reconstruction-v1':
        raise ValueError('not a NeighCF reconstruction checkpoint')
    if digest(checkpoint['contract']) != checkpoint['contract_sha256']:
        raise ValueError('checkpoint contract digest mismatch')
    return checkpoint


def restore_for_scoring(checkpoint_path, config, retriever, model, tokenizer, device, variant):
    from .soft_gate import SoftNeighborEmbeddingGate
    checkpoint = load_checkpoint(checkpoint_path, device)
    contract = checkpoint['contract']
    if contract['config'] != config.to_dict() or contract['retrieval_source'] != retriever.source_identity or contract['variant'] != variant:
        raise ValueError('checkpoint config, retrieval sources or variant differ')
    if contract['initial_model'] != model_identity(model, tokenizer):
        raise ValueError('local initial model or tokenizer differs from training')
    current_sources = source_identity()
    if contract['source_sha256'] != current_sources or contract['runtime'] != runtime_identity(device):
        raise ValueError('checkpoint scoring source or runtime differs')
    gate = SoftNeighborEmbeddingGate(config).to(device) if variant in {'full', 'global_pool'} else None
    if checkpoint['model'] is not None:
        model.load_state_dict(checkpoint['model'], strict=True)
    if gate is not None:
        gate.load_state_dict(checkpoint['gate'], strict=True)
    check_finite_state(model, gate)
    return gate, checkpoint


def train(config, training_config, retriever, train_queries, valid_queries, model, tokenizer, output, device='cpu', variant='full', resume=False, test_queries=None, max_updates=None, metadata=None):
    import fcntl
    import torch
    from .soft_gate import SoftNeighborEmbeddingGate
    from evaluate import validate_metadata, evaluate_records
    cfg, tc = config, training_config
    metadata = validate_metadata(metadata)
    if metadata['seed'] != tc.seed:
        raise ValueError('metadata seed differs from training seed')
    if max_updates is not None:
        positive_int(max_updates, 'max_updates')
    if cfg.mode != 'soft_embedding_gate':
        raise ValueError('training requires explicit soft_embedding_gate reconstruction mode')
    if variant not in {'full', 'global_pool', 'fixed_neighbors', 'without_collaborative'}:
        raise ValueError('unknown training variant')
    if tc.backbone_mode == 'gate_only' and (not tc.train_gate or variant not in {'full', 'global_pool'}):
        raise ValueError('this gate-only configuration has no trainable likelihood path')
    train_queries, valid_queries = list(train_queries), list(valid_queries)
    test_queries = list(test_queries) if test_queries is not None else None
    for split_index, rows in enumerate((train_queries, valid_queries, test_queries)):
        if rows is None:
            continue
        if not rows or len({row['query_id'] for row in rows}) != len(rows):
            raise ValueError('each supplied split must be nonempty with unique query IDs')
        for query in rows:
            validate_query(query, retriever.items, cfg.training_candidate_count if split_index == 0 else cfg.expected_candidate_count)
    if set(row['query_id'] for row in train_queries) & set(row['query_id'] for row in valid_queries):
        raise ValueError('train and validation query IDs overlap')
    if test_queries is not None and {row['query_id'] for row in test_queries} & {row['query_id'] for row in train_queries + valid_queries}:
        raise ValueError('test query IDs overlap training or validation')
    output = preflight_output(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume and any(path.name != '.training.lock' for path in output.iterdir()):
            raise ValueError('new training output must be empty')
        model.to(device)
        contract = {'implementation': 'documented_reference_reconstruction', 'metadata': metadata, 'config': cfg.to_dict(), 'training_config': tc.to_dict(), 'variant': variant, 'retrieval_source': retriever.source_identity, 'train_queries': digest(train_queries), 'valid_queries': digest(valid_queries), 'test_queries': digest(test_queries), 'initial_model': model_identity(model, tokenizer), 'runtime': runtime_identity(device), 'source_sha256': source_identity()}
        contract_sha = digest(contract)
        random.seed(tc.seed)
        np.random.seed(tc.seed % (2 ** 32))
        torch.manual_seed(tc.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tc.seed)
        gate = SoftNeighborEmbeddingGate(cfg).to(device) if variant in {'full', 'global_pool'} else None
        for parameter in model.parameters():
            parameter.requires_grad_(tc.backbone_mode == 'full_model')
        if gate is not None:
            for parameter in gate.parameters():
                parameter.requires_grad_(tc.train_gate)
        parameters = [parameter for module in (model, gate) if module is not None for parameter in module.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=tc.learning_rate, betas=tuple(tc.betas), eps=tc.epsilon, weight_decay=tc.weight_decay)
        batches_per_epoch = math.ceil(len(train_queries) / tc.batch_size)
        total_updates = batches_per_epoch * tc.epochs
        warmup_steps = int(total_updates * tc.warmup_ratio)
        def schedule(step):
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1, max(0, progress))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        progress = {'epoch': 0, 'next_batch': 0, 'updates': 0, 'best_metric': None, 'best_epoch': None, 'history': [], 'status': 'training'}
        if resume:
            saved = load_checkpoint(output / 'latest.pt', device)
            if saved['contract_sha256'] != contract_sha:
                raise ValueError('resume contract differs: data, sources, model, runtime, config and variant must match')
            if saved['model'] is not None:
                model.load_state_dict(saved['model'], strict=True)
            if gate is not None:
                gate.load_state_dict(saved['gate'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            progress = saved['progress']
            restore_rng(saved['rng'])
            if (output / 'completion.json').is_file():
                completion = json.loads((output / 'completion.json').read_text())
                if completion['contract_sha256'] != contract_sha or progress['status'] != 'complete':
                    raise ValueError('completion marker does not match restored checkpoint')
                expected_artifacts = {'best.pt', 'latest.pt'} | ({'test.metrics.json', 'test.ranks.jsonl', 'test.scores.jsonl'} if test_queries is not None else set())
                if set(completion['artifacts']) != expected_artifacts or completion['selected_epoch'] != progress['best_epoch'] or completion['selection_value'] != progress['best_metric']:
                    raise ValueError('completion artifact set or selected model differs')
                for name, expected in completion['artifacts'].items():
                    if Path(name).name != name or file_sha256(output / name) != expected:
                        raise ValueError('completed artifact integrity failure')
                return completion
        else:
            atomic_json(output / 'manifest.json', {'contract': contract, 'contract_sha256': contract_sha})
        check_finite_state(model, gate, optimizer)
        scorer = CandidateLikelihoodScorer(model, tokenizer, cfg, device, gate)
        def save(name):
            check_finite_state(model, gate, optimizer)
            checkpoint = {'format': 'neighcf-reconstruction-v1', 'contract': contract, 'contract_sha256': contract_sha, 'model': model.state_dict() if tc.backbone_mode == 'full_model' else None, 'gate': gate.state_dict() if gate is not None else None, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'progress': progress}
            atomic_checkpoint(output / name, checkpoint)
            atomic_json(output / 'progress.json', progress)
        if not resume:
            save('latest.pt')
        starting_updates = progress['updates']
        while progress['epoch'] < tc.epochs:
            epoch = progress['epoch']
            order = list(range(len(train_queries)))
            random.Random(tc.seed + epoch).shuffle(order)
            for batch in range(progress['next_batch'], batches_per_epoch):
                indices = order[batch * tc.batch_size:(batch + 1) * tc.batch_size]
                optimizer.zero_grad(set_to_none=True)
                model.train()
                total_loss = 0.0
                for index in indices:
                    query = train_queries[index]
                    found = selected_neighbors(retriever, query, variant, cfg.mode, cfg.training_candidate_count)
                    prompt = build_prompt(query, retriever.items, found, cfg, cfg.training_candidate_count)
                    scores, weights = scorer.tensor_scores(prompt)
                    target = torch.tensor([query['candidate_ids'].index(query['target_id'])], device=device)
                    loss = torch.nn.functional.cross_entropy(scores[None, :], target)
                    if weights is not None:
                        pool_weights = gate(found['pool_scores'], query['user_id'], len(query['history']))
                        loss = loss + cfg.gate_eta * pool_weights.sum()
                    if not torch.isfinite(loss) or not loss.requires_grad:
                        raise FloatingPointError('loss is nonfinite or has no trainable gradient path')
                    (loss / len(indices)).backward()
                    total_loss += float(loss.detach()) / len(indices)
                check_finite_state(model, gate, optimizer)
                if tc.max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, tc.max_gradient_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                check_finite_state(model, gate, optimizer)
                progress.update(next_batch=batch + 1, updates=progress['updates'] + 1, last_loss=total_loss)
                if progress['updates'] % tc.checkpoint_every_updates == 0:
                    save('latest.pt')
                if max_updates is not None and progress['updates'] - starting_updates >= max_updates:
                    save('latest.pt')
                    return {'status': 'paused_at_optimizer_boundary', 'progress': progress, 'contract_sha256': contract_sha}
            validation, _, _ = evaluate(model, tokenizer, gate, cfg, retriever, valid_queries, device, variant)
            value = validation[tc.selection_metric]
            improved = progress['best_metric'] is None or value > progress['best_metric']
            progress['history'].append({'epoch': epoch, 'validation': validation})
            progress.update(epoch=epoch + 1, next_batch=0)
            if improved:
                progress.update(best_metric=value, best_epoch=epoch)
                save('best.pt')
            save('latest.pt')
        progress['status'] = 'complete'
        save('latest.pt')
        best = load_checkpoint(output / 'best.pt', device)
        if best['contract_sha256'] != contract_sha or best['progress']['best_epoch'] != progress['best_epoch'] or best['progress']['best_metric'] != progress['best_metric']:
            raise ValueError('selected checkpoint does not match validated training progress')
        if best['model'] is not None:
            model.load_state_dict(best['model'], strict=True)
        if gate is not None:
            gate.load_state_dict(best['gate'], strict=True)
        completion = {'status': 'complete', 'contract_sha256': contract_sha, 'selected_epoch': best['progress']['best_epoch'], 'selection_metric': tc.selection_metric, 'selection_value': best['progress']['best_metric'], 'test_evaluated': test_queries is not None}
        if test_queries is not None:
            summary, ranks, records = evaluate(model, tokenizer, gate, cfg, retriever, test_queries, device, variant)
            common_result, ranks = evaluate_records(records, metadata, expected_candidates=cfg.expected_candidate_count)
            common_result['reconstruction'] = {'contract_sha256': contract_sha, 'selected_epoch': completion['selected_epoch'], 'variant': variant}
            atomic_json(output / 'test.metrics.json', common_result)
            for name, rows in (('test.ranks.jsonl', ranks), ('test.scores.jsonl', records)):
                with (output / name).open('wb') as stream:
                    for row in rows:
                        stream.write(canonical(row) + b'\n')
        completion['artifacts'] = {path.name: file_sha256(path) for path in output.iterdir() if path.name in {'best.pt', 'latest.pt', 'test.metrics.json', 'test.ranks.jsonl', 'test.scores.jsonl'}}
        atomic_json(output / 'completion.json', completion)
        return completion
