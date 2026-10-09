import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile

from adapters.llmsrec_data import load_canonical, prediction_record
from evaluate import evaluate_records, validate_metadata
from integrations.neighcf.retrieval import validate_query

ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic(path, value):
    path = Path(path)
    descriptor, pending = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        Path(pending).unlink(missing_ok=True)


def load_data(path, expected=51):
    path = Path(path)
    items, rows, splits, hashes = load_canonical(path, expected)
    manifest = json.loads((path / 'manifest.json').read_text())
    if splits.get('protocol_sha256') != file_digest(path / 'manifest.json'):
        raise ValueError('Canonical manifest SHA mismatch')
    for name in ('items.json', 'train.jsonl', 'valid.jsonl', 'test.jsonl'):
        if manifest.get('output_file_sha256', {}).get(name) != hashes[name]:
            raise ValueError('Canonical file SHA mismatch: ' + name)
    counts = manifest['protocol']['candidates']
    if counts.get('evaluation_count', counts['count']) != expected:
        raise ValueError('Evaluation candidate protocol mismatch')
    for split in ('valid', 'test'):
        for row in rows[split]:
            validate_query(row, items, expected)
    hashes['manifest.json'] = file_digest(path / 'manifest.json')
    return items, rows, splits, hashes


def messages_for(query, items, history_limit=0):
    if type(history_limit) is not int or history_limit < 0:
        raise ValueError('history_limit must be nonnegative')
    history = query['history'][-history_limit:] if history_limit else query['history']
    content = {'history': [items[item]['title'] for item in history],
               'candidates': [{'candidate': index, 'title': items[item]['title']}
                              for index, item in enumerate(query['candidate_ids'], 1)]}
    return [{'role': 'system', 'content': 'Rank candidate items by how likely the user is to consume each next. Use the chronological history as evidence. Return only a JSON array containing every candidate number exactly once, most likely first.'},
            {'role': 'user', 'content': canonical(content)}]


def parse_ranking(text, candidate_ids):
    try:
        order = json.loads(text)
    except (ValueError, TypeError) as error:
        raise ValueError('Completion must be a JSON ranking array') from error
    if not isinstance(order, list) or any(type(value) is not int for value in order):
        raise ValueError('Ranking must contain integer candidate numbers')
    if len(order) != len(candidate_ids) or set(order) != set(range(1, len(candidate_ids) + 1)):
        raise ValueError('Ranking must be an exact permutation; missing or repeated candidates are refused')
    positions = {candidate: rank for rank, candidate in enumerate(order)}
    return [float(-positions[index]) for index in range(1, len(candidate_ids) + 1)]


def validate_response(response, requested_model):
    if not isinstance(response, dict) or response.get('finish_reason') != 'stop':
        raise ValueError('Generation must finish normally without truncation')
    if not isinstance(response.get('text'), str) or not isinstance(response.get('model'), str):
        raise ValueError('Response requires text and recorded model identity')
    actual = response['model']
    if requested_model.startswith('gpt-4'):
        if requested_model == 'gpt-4':
            valid = actual == 'gpt-4' or actual.startswith('gpt-4-')
        else:
            valid = actual == requested_model
        if not valid:
            raise ValueError('Response model differs from requested GPT-4 model')


def source_identity():
    names = ['integrations/zeroshot/core.py', 'integrations/zeroshot/backends.py',
             'integrations/zeroshot/run.py', 'adapters/llmsrec_data.py', 'evaluate.py',
             'integrations/neighcf/retrieval.py', 'integrations/neighcf/config.py']
    return {name: file_digest(ROOT / name) for name in names}


def local_identity(path):
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in {'.json', '.model', '.safetensors', '.bin', '.txt'})
    if not (path / 'config.json').is_file() or not any(p.suffix in {'.bin', '.safetensors'} for p in files):
        raise ValueError('Local model configuration and weights are required')
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != 'llama' or config.get('hidden_size') != 4096 or config.get('num_hidden_layers') != 32 or config.get('vocab_size') != 32000:
        raise ValueError('LLaMA2 baseline requires a local 7B Llama architecture checkpoint')
    return {str(p.relative_to(path)): file_digest(p) for p in files}


def run_queries(items, queries, metadata, settings, output, backend, resume=False, request_budget=None,
                retry_uncertain=False, stop_after=None):
    metadata = validate_metadata(metadata)
    queries = list(queries)
    if not queries or len({row['query_id'] for row in queries}) != len(queries):
        raise ValueError('Queries must be nonempty and unique')
    for row in queries:
        validate_query(row, items, settings['expected_candidates'])
    if request_budget is not None and (type(request_budget) is not int or request_budget < 1):
        raise ValueError('request_budget must be a positive integer')
    from integrations.neighcf.training import preflight_output
    output = preflight_output(output)
    contract = {'metadata': metadata, 'settings': settings, 'items_sha256': digest(items),
                'queries_sha256': digest(queries), 'source_sha256': source_identity()}
    identity = {'contract': contract, 'sha256': digest(contract)}
    existed = output.exists()
    if resume and not existed:
        raise ValueError('Resume requires an existing output')
    if not resume and existed and any(output.iterdir()):
        raise ValueError('New output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / '.lock', os.O_CREAT | os.O_RDWR, 0o600)
    handlers = {}
    stopped = [False]
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'manifest.json'
        if resume:
            if json.loads(path.read_text()) != identity:
                raise ValueError('Resume contract changed')
        else:
            atomic(path, identity)
        journal = output / 'journal'
        if journal.is_symlink():
            raise ValueError('Journal cannot be a symlink')
        journal.mkdir(exist_ok=True)
        if any(p.is_symlink() for p in journal.iterdir()):
            raise ValueError('Journal entries cannot be symlinks')
        for signum in (signal.SIGTERM, signal.SIGUSR1):
            handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
        records, calls = [], 0
        for query in queries:
            messages = messages_for(query, items, settings['history_limit'])
            request_identity = digest({'messages': messages, 'settings': settings})
            name = digest(query['query_id'])
            response_path = journal / (name + '.json')
            pending = journal / (name + '.pending.json')
            if response_path.exists():
                saved = json.loads(response_path.read_text())
                content = saved['content']
                if saved['sha256'] != digest(content) or content['request_sha256'] != request_identity or content['query_sha256'] != digest(query):
                    raise ValueError('Journal response integrity or request mismatch')
                response = content['response']
            else:
                if stopped[0] or stop_after is not None and calls >= stop_after or request_budget is not None and calls >= request_budget:
                    atomic(output / 'progress.json', {'status': 'paused', 'completed': len(records), 'total': len(queries)})
                    return {'status': 'paused', 'completed': len(records), 'calls_this_invocation': calls}
                if pending.exists() and not retry_uncertain:
                    raise ValueError('Uncertain previous request; inspect it before explicitly allowing retry')
                atomic(pending, {'request_sha256': request_identity, 'query_id': query['query_id']})
                response = backend(messages)
                calls += 1
                content = {'query_sha256': digest(query), 'request_sha256': request_identity, 'response': response}
                atomic(response_path, {'content': content, 'sha256': digest(content)})
                pending.unlink(missing_ok=True)
            validate_response(response, settings['model_id'])
            scores = parse_ranking(response['text'], query['candidate_ids'])
            records.append(prediction_record({'source': query}, scores))
        metrics, ranks = evaluate_records(records, metadata, settings['expected_candidates'])
        metrics['inference'] = {'contract_sha256': identity['sha256'], 'ranking': 'zero_shot_generated_full_permutation',
                                'prompt_policy': 'numbered_candidates_chronological_titles', 'target_visible_to_model': False}
        artifacts = {'metrics.json': metrics, 'ranks.json': ranks, 'predictions.json': records}
        completed = output / 'completion.json'
        if completed.exists():
            old = json.loads(completed.read_text())
            if old.get('contract_sha256') != identity['sha256'] or set(old.get('artifacts', {})) != set(artifacts):
                raise ValueError('Completion contract or artifact inventory mismatch')
            for name, expected in old['artifacts'].items():
                if file_digest(output / name) != expected:
                    raise ValueError('Completed artifact changed')
            return {'status': 'complete', 'completed': len(records), 'calls_this_invocation': calls}
        for name, value in artifacts.items():
            atomic(output / name, value)
        atomic(output / 'completion.json', {'status': 'complete', 'contract_sha256': identity['sha256'],
                'queries': len(records), 'artifacts': {name: file_digest(output / name) for name in artifacts}})
        atomic(output / 'progress.json', {'status': 'complete', 'completed': len(records), 'total': len(records)})
        return {'status': 'complete', 'completed': len(records), 'calls_this_invocation': calls}
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        os.close(descriptor)
