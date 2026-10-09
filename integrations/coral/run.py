import argparse
from dataclasses import asdict
import fcntl
import importlib.metadata
import json
from pathlib import Path
import sys
import torch

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate import evaluate_records
from integrations.coral.core import Config, CoRAL, Ratings
from integrations.coral.storage import Checkpoints, LocalYesProbability, ProbabilityCache
from integrations.drdt.run import atomic_json, load_data, output_path, read_json, rows, sha
from integrations.neighcf.retrieval import canonical, digest, timestamp
from integrations.resume import runtime_context


def evaluate(engine, queries, probability, metadata):
    predictions = []
    for query in queries:
        scores = [engine.episode(query['user_id'], candidate, timestamp(query), probability)[0] for candidate in query['candidate_ids']]
        predictions.append({key: query[key] for key in ('query_id', 'user_id', 'candidate_ids', 'target_id')} | {'scores': scores})
    metrics, ranks = evaluate_records(predictions, metadata, expected_candidates=engine.config.expected_candidates)
    return metrics, ranks, predictions


def eligible_episodes(engine):
    eligible = []
    for index, row in enumerate(engine.ratings.records):
        matrix = engine.ratings.before(row['user_id'], row['item_id'], row['timestamp'])
        if any(user != row['user_id'] for user, _ in matrix):
            eligible.append(index)
    return eligible


def validate_ratings_binding(records, data):
    data = Path(data)
    manifest = read_json(data / 'manifest.json')
    path = data / 'train_interactions.jsonl'
    if sha(path) != manifest['output_file_sha256'].get('train_interactions.jsonl'):
        raise ValueError('canonical training interaction hash mismatch')
    observed = {}
    for row in rows(path):
        if row.get('split') != 'train':
            raise ValueError('canonical rating source contains a held-out interaction')
        identity = (str(row['user_id']), str(row['item_id']), timestamp(row))
        if identity in observed:
            raise ValueError('canonical training interaction identity is duplicated')
        observed[identity] = row.get('rating')
    for row in records:
        identity = (row['user_id'], row['item_id'], row['timestamp'])
        if identity not in observed or observed[identity] is None or observed[identity] != row['rating']:
            raise ValueError('CoRAL rating must match an explicitly observed rating in canonical training interactions')


def train(engine, valid, test, probability, output, contract, resume=False, stop_after_episodes=None):
    if {row['query_id'] for row in valid} & {row['query_id'] for row in test}:
        raise ValueError('validation and test query identities overlap')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = engine.config
    eligible = eligible_episodes(engine)
    if len(eligible) < config.episodes_per_epoch:
        raise ValueError('episodes_per_epoch exceeds eligible observed training ratings')
    store = Checkpoints(output, contract)
    contract_path = output / 'contract.json'
    if contract_path.exists():
        if not resume or canonical(read_json(contract_path)) != canonical(contract):
            raise ValueError('exact training contract and --resume are required')
        checkpoint = store.load()
        engine.restore(checkpoint['engine'])
        progress = checkpoint['progress']
    else:
        if any(path.name != '.lock' for path in output.iterdir()):
            raise ValueError('new training output must be empty')
        atomic_json(contract_path, contract)
        progress = {'phase': 'train', 'epoch': 0, 'next_episode': 0, 'order': [], 'best_metric': None,
                    'best_epoch': None, 'best_generation': None, 'validation': [], 'completed_episodes': 0}
        progress = store.save(engine, progress)
    if progress['phase'] == 'complete':
        completion = read_json(output / 'completion.json')
        if completion.get('contract_sha256') != digest(contract) or set(completion.get('artifacts', {})) != {'test.metrics.json', 'test.ranks.json', 'test.scores.json'}:
            raise ValueError('completion receipt identity or artifact inventory mismatch')
        selected = {'status': 'complete', 'selected_epoch': progress['best_epoch'],
                    'selected_generation': progress['best_generation'], 'validation_NDCG@10': progress['best_metric']}
        if canonical({key: completion.get(key) for key in selected}) != canonical(selected):
            raise ValueError('completion selection differs from checkpoint progress')
        for name, expected in completion['artifacts'].items():
            if sha(output / name) != expected:
                raise ValueError('completed test artifact checksum mismatch')
        return read_json(output / 'test.metrics.json')
    entered = engine.episodes
    while progress['epoch'] < config.epochs:
        if not progress['order']:
            progress['order'] = engine.random.sample(eligible, config.episodes_per_epoch)
            progress = store.save(engine, progress)
        while progress['next_episode'] < len(progress['order']) and engine.updates < config.max_updates:
            row = engine.ratings.records[progress['order'][progress['next_episode']]]
            engine.episode(row['user_id'], row['item_id'], row['timestamp'], probability, label=row['label'], train=True)
            progress['next_episode'] += 1
            progress['completed_episodes'] += 1
            progress = store.save(engine, progress)
            if stop_after_episodes is not None and engine.episodes - entered >= stop_after_episodes:
                return {'status': 'paused', 'completed_episodes': progress['completed_episodes']}
        metrics, _, _ = evaluate(engine, valid, probability, contract['metadata'])
        value = metrics['metrics']['NDCG@10']
        selected = progress['best_metric'] is None or value > progress['best_metric']
        if selected:
            progress['best_metric'], progress['best_epoch'] = value, progress['epoch']
        progress['validation'].append({'epoch': progress['epoch'], 'NDCG@10': value})
        progress['epoch'] += 1
        if engine.updates >= config.max_updates:
            progress['epoch'] = config.epochs
        progress['next_episode'], progress['order'] = 0, []
        progress = store.save(engine, progress, select_best=selected)
    latest = engine.snapshot()
    best = store.read_generation(progress['best_generation'])
    engine.restore(best['engine'])
    metrics, ranks, predictions = evaluate(engine, test, probability, contract['metadata'])
    engine.restore(latest)
    for name, value in (('test.metrics.json', metrics), ('test.ranks.json', ranks), ('test.scores.json', predictions)):
        atomic_json(output / name, value)
    atomic_json(output / 'completion.json', {'status': 'complete', 'contract_sha256': digest(contract),
                'selected_epoch': progress['best_epoch'], 'validation_NDCG@10': progress['best_metric'],
                'selected_generation': progress['best_generation'],
                'artifacts': {name: sha(output / name) for name in ('test.metrics.json', 'test.ranks.json', 'test.scores.json')}})
    progress['phase'] = 'complete'
    store.save(engine, progress)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('audit', 'train', 'score'))
    for name in ('config', 'data', 'ratings', 'embeddings', 'embedding-provenance'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model-path')
    parser.add_argument('--output')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    parser.add_argument('--max-prompt-tokens', type=int, default=3072)
    parser.add_argument('--max-backend-calls', type=int)
    args = parser.parse_args(argv)
    config = Config.from_dict(read_json(args.config))
    items, _, valid, splits = load_data(args.data, 'valid', config.expected_candidates)
    _, _, test, _ = load_data(args.data, 'test', config.expected_candidates)
    if {row['query_id'] for row in valid} & {row['query_id'] for row in test}:
        raise ValueError('validation and test query identities overlap')
    ratings = Ratings(items, rows(args.ratings), config.rating_threshold)
    validate_ratings_binding(ratings.records, args.data)
    embeddings = read_json(args.embeddings)
    if set(embeddings) != {'users', 'items'}:
        raise ValueError('embeddings must have exactly users and items mappings')
    provenance = read_json(args.embedding_provenance)
    required = {'fit_split': 'train', 'protocol_sha256': splits['protocol_sha256'],
                'ratings_sha256': sha(args.ratings), 'embeddings_sha256': sha(args.embeddings)}
    if any(provenance.get(key) != value for key, value in required.items()) or not isinstance(provenance.get('source_method'), str) or not provenance['source_method'].strip():
        raise ValueError('external embedding provenance must bind observed training ratings, embedding bytes and canonical protocol')
    engine = CoRAL(config, ratings, embeddings['users'], embeddings['items'])
    for query in valid + test:
        if query['user_id'] not in engine.user_index:
            raise ValueError('evaluation user has no supplied training-only embedding')
        for item in query['candidate_ids']:
            matrix = ratings.before(query['user_id'], item, timestamp(query))
            if not any(user != query['user_id'] for user, _ in matrix):
                raise ValueError('evaluation query lacks strictly earlier collaborative rating evidence')
    eligible = eligible_episodes(engine)
    if len(eligible) < config.episodes_per_epoch:
        raise ValueError('episodes_per_epoch exceeds eligible observed training ratings')
    if args.command == 'audit':
        print(json.dumps({'status': 'data_ready', 'eligible_training_ratings': len(eligible),
                          'excluded_initial_ratings_without_collaborative_past': len(ratings.records) - len(eligible),
                          'validation_queries': len(valid), 'test_queries': len(test),
                          'maximum_uncached_calls': config.epochs * config.episodes_per_epoch * (config.train_steps + 1) +
                              (config.epochs * len(valid) + len(test)) * config.expected_candidates * (config.eval_steps + 1),
                          'implementation': 'explicit_primary_paper_ddpg_reconstruction'}))
        return
    if not args.model_path or not args.output or args.max_backend_calls is None or args.max_backend_calls < 1:
        raise ValueError('train/score require local --model-path, --output and positive --max-backend-calls')
    if args.command == 'score' and not args.checkpoint:
        raise ValueError('score requires a completed or selected training checkpoint directory')
    protected = [args.config, args.data, args.ratings, args.embeddings, args.embedding_provenance, args.model_path]
    if args.checkpoint:
        protected.append(args.checkpoint)
    output = output_path(args.output, protected)
    if output.exists():
        if any(path.is_symlink() for path in output.rglob('*')):
            raise ValueError('output contains symlinks')
        if not args.resume and any(output.iterdir()):
            raise ValueError('new output must be empty')
        if args.command == 'score' and any(output.iterdir()):
            raise ValueError('standalone scoring requires a new output directory')
    package = Path(__file__).resolve().parents[2]
    names = ['integrations/coral/core.py', 'integrations/coral/storage.py', 'integrations/coral/run.py',
             'integrations/drdt/run.py', 'integrations/drdt/source.py', 'integrations/drdt/engine.py',
             'integrations/neighcf/retrieval.py', 'integrations/zeroshot/backends.py', 'integrations/resume.py', 'evaluate.py']
    model_files = {}
    for path in sorted(Path(args.model_path).rglob('*')):
        if path.is_symlink():
            raise ValueError('model requires materialized files, not symlinks')
        if path.is_file():
            model_files[path.relative_to(args.model_path).as_posix()] = sha(path)
    if not model_files:
        raise ValueError('local model snapshot is empty')
    contract = {'kind': 'coral-ddpg-rating-reconstruction-v1', 'metadata': {'model': 'CoRAL-DDPG-reconstruction',
                'dataset': splits['dataset'], 'seed': config.seed, 'protocol_sha256': splits['protocol_sha256']},
                'config': asdict(config), 'source_sha256': {name: sha(package / name) for name in names},
                'data_sha256': {name: sha(Path(args.data) / name) for name in ('items.json', 'train.jsonl', 'valid.jsonl', 'test.jsonl', 'train_interactions.jsonl', 'manifest.json', 'splits.json')},
                'ratings_sha256': sha(args.ratings), 'embeddings_sha256': sha(args.embeddings),
                'embedding_provenance': provenance, 'model_files': model_files, 'device': args.device, 'dtype': args.dtype,
                'max_prompt_tokens': args.max_prompt_tokens,
                'runtime': runtime_context(args.device) | {'cpu_threads': torch.get_num_threads(), 'cpu_interop_threads': torch.get_num_interop_threads()},
                'versions': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'transformers', 'tokenizers')},
                'paper': 'https://arxiv.org/abs/2403.06447',
                'adaptations': ['canonical candidate ranking replaces native rating classification evaluation',
                                'explicit observed ratings; latest strictly earlier record per user/item; target pair excluded from evidence',
                                'external frozen train-partition user/item embeddings; training-partition fit is not a per-query temporal embedding fit',
                                'MLP widths, OU process, replay budget and optimization values explicitly configured',
                                'state transition MLP learns by critic TD gradients on recomputed replay histories with its own Polyak target',
                                'separate training/evaluation retrieval limits; training discrepancy early stop; optimizer warmup counts collected transitions',
                                'max_updates caps Adam update pairs, finishing the current episode before validation; explicit epoch/episode budget',
                                'repeated retrieval allowed; target user excluded from collaborative pool',
                                'validation NDCG@10 selects policy; test evaluated once after selection',
                                'local frozen LLM next-token Yes probability; no binary Yes/No renormalization']}
    if args.command == 'score':
        old_contract = read_json(Path(args.checkpoint) / 'contract.json')
        if canonical(contract) != canonical(old_contract):
            raise ValueError('scoring inputs/runtime must match the trained policy contract')
    backend = LocalYesProbability(args.model_path, config.yes_token_text, args.device, args.dtype, args.max_prompt_tokens)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probability = None
        def score(messages):
            nonlocal probability
            if probability is None:
                probability = ProbabilityCache(output / 'probabilities', digest(contract), backend, args.max_backend_calls)
            return probability(messages)
        if args.command == 'train':
            train(engine, valid, test, score, output, contract, args.resume)
        else:
            checkpoint_dir = Path(args.checkpoint)
            with (checkpoint_dir / '.lock').open('r+b') as source_lock:
                fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                checkpoint = Checkpoints(checkpoint_dir, contract).load(best=True)
                engine.restore(checkpoint['engine'])
            metrics, ranks, predictions = evaluate(engine, test, score, contract['metadata'])
            for name, value in (('metrics.json', metrics), ('ranks.json', ranks), ('scores.json', predictions)):
                atomic_json(output / name, value)
            atomic_json(output / 'manifest.json', {'contract': contract, 'selected_generation': checkpoint['generation']})


if __name__ == '__main__':
    main()
