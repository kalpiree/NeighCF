import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate import evaluate_records
from integrations.drdt.engine import DOMAINS, rank_query, select_demonstration
from integrations.drdt.source import COMMIT, FILES, load_methods
from integrations.neighcf.retrieval import canonical, digest, validate_query


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError('output symlinks are forbidden')
    with temporary.open('wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def output_path(path, inputs):
    value = Path(path).absolute()
    if any(parent.is_symlink() for parent in (value, *value.parents)):
        raise ValueError('output symlink ancestors are forbidden')
    value = value.resolve()
    protected = [Path(__file__).resolve().parents[2], *[Path(item).resolve() for item in inputs]]
    if any(value == item or value in item.parents or item in value.parents for item in protected):
        raise ValueError('output must be disjoint from source and input paths')
    return value


def load_data(data, split, expected_candidates):
    data = Path(data)
    splits = read_json(data / 'splits.json')
    if splits.get('manifest', 'manifest.json') != 'manifest.json' or sha(data / 'manifest.json') != splits.get('protocol_sha256'):
        raise ValueError('canonical protocol manifest binding mismatch')
    manifest = read_json(data / 'manifest.json')
    names = ['items.json', 'train.jsonl', split + '.jsonl']
    for name in names:
        if sha(data / name) != manifest['output_file_sha256'].get(name):
            raise ValueError('canonical data bytes differ from protocol manifest')
    counts = manifest['protocol']['candidates']
    if counts.get('evaluation_count', counts['count']) != expected_candidates:
        raise ValueError('evaluation candidate count differs from canonical protocol')
    items, training, queries = read_json(data / 'items.json'), rows(data / 'train.jsonl'), rows(data / (split + '.jsonl'))
    if not isinstance(items, dict) or not items or any(not isinstance(key, str) or not key or not isinstance(value, dict) or not isinstance(value.get('title'), str) or not value['title'].strip() for key, value in items.items()):
        raise ValueError('catalog must map string IDs to nonempty titles')
    for values, count in ((training, counts.get('training_count', counts['count'])), (queries, expected_candidates)):
        if not values or len({row['query_id'] for row in values}) != len(values):
            raise ValueError('split must have nonempty unique query IDs')
        for row in values:
            validate_query(row, items, count)
    if set(row['query_id'] for row in training) & set(row['query_id'] for row in queries):
        raise ValueError('training and evaluation query IDs overlap')
    return items, training, queries, splits


class CallJournal:
    def __init__(self, output, backend, contract_sha256, max_calls, retry_uncertain=False, expected_model=None):
        self.output, self.backend, self.contract_sha256 = Path(output), backend, contract_sha256
        self.max_calls, self.retry_uncertain, self.expected_model = max_calls, retry_uncertain, expected_model
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('max_calls must be a positive integer')

    def for_query(self, query_id):
        key, index = hashlib.sha256(query_id.encode()).hexdigest(), [0]

        def complete(messages):
            name = f'call-{key}-{index[0]:03d}'
            index[0] += 1
            path, pending = self.output / (name + '.json'), self.output / (name + '.pending')
            identity = {'contract_sha256': self.contract_sha256, 'messages_sha256': digest(messages)}
            if path.exists():
                value = read_json(path)
                if value.get('identity') != identity or value.get('response_sha256') != digest(value['response']):
                    raise ValueError('saved DRDT call identity or response checksum mismatch')
                if self.expected_model and value['response'].get('model') != self.expected_model:
                    raise ValueError('saved DRDT response model differs from the pinned model ID')
                return value['response']
            if pending.exists() and (read_json(pending) != identity or not self.retry_uncertain):
                raise ValueError('uncertain DRDT call requires matching inputs and explicit --retry-uncertain')
            budget_path = self.output / 'call-budget.json'
            budget = read_json(budget_path) if budget_path.exists() else {'contract_sha256': self.contract_sha256, 'attempts': 0}
            if budget.get('contract_sha256') != self.contract_sha256 or type(budget.get('attempts')) is not int or budget['attempts'] < 0:
                raise ValueError('DRDT call budget identity mismatch')
            if budget['attempts'] >= self.max_calls:
                raise ValueError('DRDT maximum backend call budget reached')
            budget['attempts'] += 1
            atomic_json(budget_path, budget)
            atomic_json(pending, identity)
            result = self.backend(messages)
            if not isinstance(result, dict) or not isinstance(result.get('text'), str) or result.get('finish_reason') != 'stop':
                raise ValueError('DRDT backend did not return a normally stopped text response')
            if self.expected_model and result.get('model') != self.expected_model:
                raise ValueError('DRDT API response model differs from the pinned model ID')
            atomic_json(path, {'identity': identity, 'response': result, 'response_sha256': digest(result)})
            return result

        return complete


def run_queries(output, contract, queries, compute, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in output.iterdir()):
        raise ValueError('output symlinks are forbidden')
    allowed = {'contract.json', '.lock', 'metrics.json', 'ranks.json', 'completion.json', 'call-budget.json', 'call-budget.json.tmp'}
    if any(path.name not in allowed and not (path.name.startswith(('query-', 'call-')) and path.suffix in {'.json', '.tmp', '.pending'}) for path in output.iterdir()):
        raise ValueError('output contains unrecognized files')
    with (output / '.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract_path = output / 'contract.json'
        if contract_path.exists():
            if not resume or canonical(read_json(contract_path)) != canonical(contract):
                raise ValueError('resume requires the exact existing contract')
        else:
            if any(path.name != '.lock' for path in output.iterdir()):
                raise ValueError('new output must be empty')
            atomic_json(contract_path, contract)
        saved = sorted(output.glob('query-*.json'))
        if len(saved) > len(queries) or [p.name for p in saved] != [f'query-{index:08d}.json' for index in range(len(saved))]:
            raise ValueError('query journal is not a contiguous prefix')
        previous, records = digest(contract), []
        for index, query in enumerate(queries):
            path = output / f'query-{index:08d}.json'
            if path.exists():
                envelope = read_json(path)
                payload = envelope['payload']
                if envelope.get('previous_sha256') != previous or envelope.get('payload_sha256') != digest(payload):
                    raise ValueError('query journal chain or payload checksum mismatch')
            else:
                payload = compute(query)
                envelope = {'previous_sha256': previous, 'payload': payload, 'payload_sha256': digest(payload)}
                atomic_json(path, envelope)
            if any(payload.get(key) != query[key] for key in ('query_id', 'user_id', 'candidate_ids', 'target_id')):
                raise ValueError('journal query or candidate identity mismatch')
            records.append({key: payload[key] for key in ('query_id', 'user_id', 'candidate_ids', 'target_id', 'scores')})
            previous = digest(envelope)
        metrics, ranks = evaluate_records(records, contract['metadata'], expected_candidates=contract['expected_candidates'])
        atomic_json(output / 'metrics.json', metrics)
        atomic_json(output / 'ranks.json', ranks)
        atomic_json(output / 'completion.json', {'status': 'complete', 'queries': len(queries), 'journal_sha256': previous,
                                              'contract_sha256': digest(contract), 'metrics_sha256': sha(output / 'metrics.json'),
                                              'ranks_sha256': sha(output / 'ranks.json')})
        return metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('audit', 'score'))
    for name in ('data', 'source', 'domain'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', choices=('valid', 'test'), default='test')
    parser.add_argument('--expected-candidates', type=int, default=51)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reflection-steps', type=int, default=3)
    parser.add_argument('--history-limit', type=int, default=20)
    parser.add_argument('--output')
    parser.add_argument('--backend', choices=('local', 'api'), default='local')
    parser.add_argument('--model-path')
    parser.add_argument('--api-model')
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32', choices=('float32', 'float16', 'bfloat16'))
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--max-prompt-tokens', type=int, default=3072)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-api-calls', type=int)
    parser.add_argument('--retry-uncertain', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.reflection_steps <= 20 or not args.reflection_steps < args.history_limit <= 1000:
        raise ValueError('reflection_steps must be1..20 and history_limit must exceed it and be at most1000')
    if min(args.max_new_tokens, args.max_prompt_tokens) < 1:
        raise ValueError('prompt and generation token limits must be positive')
    if args.domain not in DOMAINS:
        raise ValueError('domain must be one of native DRDT: ' + ', '.join(DOMAINS))
    import numpy as np
    load_methods(args.source, np.random.RandomState(args.seed))
    items, training, queries, splits = load_data(args.data, args.split, args.expected_candidates)
    for query in queries:
        select_demonstration(query, training, items, args.expected_candidates)
        if len(query['history'][-args.history_limit:]) <= args.reflection_steps:
            raise ValueError('query history is too short for configured reflection steps')
    if args.command == 'audit':
        print(json.dumps({'status': 'ready', 'queries': len(queries), 'native_commit': COMMIT,
                          'calls_per_query': 2 * args.reflection_steps + 2, 'domain': args.domain}))
        return
    if not args.output:
        raise ValueError('score requires --output')
    if args.backend == 'local' and not args.model_path:
        raise ValueError('local scoring requires --model-path')
    if args.backend == 'api' and (not args.allow_api or not args.api_model):
        raise ValueError('API scoring requires --api-model and explicit --allow-api')
    if args.backend == 'api' and (args.max_api_calls is None or args.max_api_calls < 1):
        raise ValueError('API scoring requires a positive explicit --max-api-calls')
    if args.backend == 'api' and not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY is required before API scoring can start')
    output = output_path(args.output, [args.data, args.source] + ([args.model_path] if args.model_path else []))
    package = Path(__file__).resolve().parents[2]
    sources = ['integrations/drdt/source.py', 'integrations/drdt/engine.py', 'integrations/drdt/run.py',
               'integrations/neighcf/retrieval.py', 'integrations/zeroshot/backends.py', 'evaluate.py']
    model_files = {}
    if args.model_path:
        for path in sorted(Path(args.model_path).rglob('*')):
            if path.is_symlink():
                raise ValueError('model path contains symlinks; use a materialized snapshot')
            if path.is_file():
                model_files[path.relative_to(args.model_path).as_posix()] = sha(path)
        if not model_files:
            raise ValueError('model snapshot is empty')
    contract = {'kind': 'drdt-native-canonical-v1', 'metadata': {'model': 'DRDT-native-canonical', 'dataset': splits['dataset'],
                'seed': args.seed, 'protocol_sha256': splits['protocol_sha256']}, 'split': args.split,
                'expected_candidates': args.expected_candidates, 'domain': args.domain, 'reflection_steps': args.reflection_steps,
                'history_limit': args.history_limit, 'native_commit': COMMIT, 'native_source_sha256': FILES,
                'source_sha256': {name: sha(package / name) for name in sources},
                'inputs': {name: sha(Path(args.data) / name) for name in ('items.json', 'train.jsonl', args.split + '.jsonl', 'splits.json', 'manifest.json')},
                'backend': args.backend, 'model_files': model_files, 'api_model': args.api_model,
                'device': args.device, 'dtype': args.dtype, 'max_new_tokens': args.max_new_tokens,
                'max_prompt_tokens': args.max_prompt_tokens,
                'numpy_version': importlib.metadata.version('numpy'),
                'adaptations': ['canonical frozen evaluation candidates', 'strictly earlier other-user training demonstration with matching history endpoint; lexical query-ID tie break',
                                'demonstration retains the first required training negatives before native target insertion and shuffle',
                                'strict complete candidate ranking replaces native substring matching', 'per-query seeded native sampling; completed-query journal resume',
                                'configured backend replaces native fixed GPT model IDs',
                                'artists-native-books-adaptation-v1 changes only static purchase/book preference wording; reflection mechanics unchanged']}
    if args.backend == 'local':
        contract['runtime_versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers')}
    from integrations.zeroshot.backends import LocalChat, OpenAIChat
    backend = (LocalChat(args.model_path, device=args.device, dtype=args.dtype, max_new_tokens=args.max_new_tokens,
                         max_prompt_tokens=args.max_prompt_tokens) if args.backend == 'local'
               else OpenAIChat(args.api_model, max_new_tokens=args.max_new_tokens))
    journal = CallJournal(output, backend, digest(contract), args.max_api_calls or len(queries) * (2 * args.reflection_steps + 2),
                          args.retry_uncertain, args.api_model if args.backend == 'api' else None)
    compute = lambda query: rank_query(args.source, items, training, query, journal.for_query(query['query_id']), domain=args.domain, seed=args.seed,
                                      reflection_steps=args.reflection_steps, history_limit=args.history_limit,
                                      expected_candidates=args.expected_candidates, model_name=args.api_model or 'local')
    run_queries(output, contract, queries, compute, args.resume)


if __name__ == '__main__':
    main()
