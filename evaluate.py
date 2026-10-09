from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable
CUTOFFS = (1, 5, 10)
METRIC_NAMES = tuple((f'HR@{k}' for k in CUTOFFS)) + tuple((f'NDCG@{k}' for k in CUTOFFS)) + ('MRR',)
SCHEMA_VERSION = 1

class EvaluationError(ValueError):
    pass

def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f'{label} must be a nonempty string')
    return value

def _sha256_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EvaluationError(f'{label} must be a 64-character SHA-256 hex digest')
    if any((character not in '0123456789abcdefABCDEF' for character in value)):
        raise EvaluationError(f'{label} must be a 64-character SHA-256 hex digest')
    return value.lower()

def validate_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise EvaluationError('run metadata must be a JSON object')
    checked = dict(metadata)
    for field in ('model', 'dataset'):
        checked[field] = _nonempty_string(metadata.get(field), field)
    seed = metadata.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationError('seed must be an integer')
    checked['protocol_sha256'] = _sha256_string(metadata.get('protocol_sha256'), 'protocol_sha256')
    return checked

def metrics_from_rank(rank: int) -> dict[str, float]:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise EvaluationError('rank must be a positive integer')
    values = {f'HR@{k}': float(rank <= k) for k in CUTOFFS}
    values.update({f'NDCG@{k}': 1.0 / math.log2(rank + 1) if rank <= k else 0.0 for k in CUTOFFS})
    values['MRR'] = 1.0 / rank
    return values

def _validate_metrics(metrics: Any, label: str='metrics') -> None:
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
        raise EvaluationError(f"{label} must contain exactly {', '.join(METRIC_NAMES)}")
    for name in METRIC_NAMES:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise EvaluationError(f'{label}.{name} must be a finite number in [0, 1]')
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise EvaluationError(f'{label}.{name} must be a finite number in [0, 1]')
    tolerance = 1e-12
    if abs(metrics['HR@1'] - metrics['NDCG@1']) > tolerance:
        raise EvaluationError(f'{label}: NDCG@1 must equal HR@1')
    for family in ('HR', 'NDCG'):
        if any((metrics[f'{family}@{a}'] > metrics[f'{family}@{b}'] + tolerance for a, b in zip(CUTOFFS, CUTOFFS[1:]))):
            raise EvaluationError(f'{label}: {family} must not decrease with cutoff')
    for cutoff in (5, 10):
        upper = metrics['HR@1'] + (metrics[f'HR@{cutoff}'] - metrics['HR@1']) / math.log2(3)
        if metrics[f'NDCG@{cutoff}'] > upper + tolerance:
            raise EvaluationError(f'{label}: NDCG@{cutoff} exceeds the single-positive bound')
    if metrics['MRR'] + tolerance < metrics['HR@1']:
        raise EvaluationError(f'{label}: full-candidate MRR must be at least HR@1')

def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)

def evaluate_records(records: Iterable[Any], metadata: Any, expected_candidates: int=100) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checked_metadata = validate_metadata(metadata)
    if isinstance(expected_candidates, bool) or not isinstance(expected_candidates, int) or expected_candidates < 1:
        raise EvaluationError('expected_candidates must be a positive integer')
    seen_queries: set[str] = set()
    users: set[str] = set()
    audit: list[dict[str, Any]] = []
    evaluation_set: list[dict[str, Any]] = []
    for row_number, record in enumerate(records, start=1):
        prefix = f'prediction row {row_number}'
        if not isinstance(record, dict):
            raise EvaluationError(f'{prefix} must be a JSON object')
        query_id = _nonempty_string(record.get('query_id'), f'{prefix}.query_id')
        if query_id in seen_queries:
            raise EvaluationError(f'{prefix}: duplicate query_id {query_id!r}')
        user_id = _nonempty_string(record.get('user_id'), f'{prefix}.user_id')
        candidate_ids = record.get('candidate_ids')
        if not isinstance(candidate_ids, list) or len(candidate_ids) != expected_candidates:
            raise EvaluationError(f'{prefix}.candidate_ids must contain exactly {expected_candidates} items')
        for index, item_id in enumerate(candidate_ids):
            _nonempty_string(item_id, f'{prefix}.candidate_ids[{index}]')
        if len(set(candidate_ids)) != len(candidate_ids):
            raise EvaluationError(f'{prefix}: duplicate candidate item IDs')
        target_id = _nonempty_string(record.get('target_id'), f'{prefix}.target_id')
        if target_id not in candidate_ids:
            raise EvaluationError(f'{prefix}: target_id must occur exactly once in candidate_ids')
        raw_scores = record.get('scores')
        if not isinstance(raw_scores, list) or len(raw_scores) != len(candidate_ids):
            raise EvaluationError(f'{prefix}.scores must match candidate_ids in length')
        scores: list[float] = []
        for index, value in enumerate(raw_scores):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvaluationError(f'{prefix}.scores[{index}] must be a finite number')
            try:
                score = float(value)
            except (OverflowError, ValueError) as exc:
                raise EvaluationError(f'{prefix}.scores[{index}] must be finite') from exc
            if not math.isfinite(score):
                raise EvaluationError(f'{prefix}.scores[{index}] must be finite')
            scores.append(score)
        ranking = sorted(zip(candidate_ids, scores), key=lambda item: (-item[1], item[0]))
        rank = next((index for index, (item_id, _) in enumerate(ranking, 1) if item_id == target_id))
        audit.append({'query_id': query_id, 'user_id': user_id, 'target_id': target_id, 'target_rank': rank, 'candidate_count': len(candidate_ids), 'has_score_ties': len(set(scores)) != len(scores)})
        evaluation_set.append({'query_id': query_id, 'user_id': user_id, 'target_id': target_id, 'candidate_ids': sorted(candidate_ids)})
        seen_queries.add(query_id)
        users.add(user_id)
    if not audit:
        raise EvaluationError('predictions contain no query rows')
    per_query = [metrics_from_rank(record['target_rank']) for record in audit]
    metrics = {name: math.fsum((row[name] for row in per_query)) / len(audit) for name in METRIC_NAMES}
    _validate_metrics(metrics)
    evaluation_set.sort(key=lambda row: row['query_id'])
    evaluation_set_sha256 = hashlib.sha256((_canonical_json(evaluation_set) + '\n').encode('utf-8')).hexdigest()
    result = {'schema_version': SCHEMA_VERSION, 'kind': 'evaluation', 'metadata': checked_metadata, 'evaluation': {'expected_candidates': expected_candidates, 'positive_count': 1, 'aggregation': 'query_macro', 'tie_break': 'score_descending_then_item_id_ascending', 'mrr_scope': 'all_candidates'}, 'evaluation_set_sha256': evaluation_set_sha256, 'counts': {'queries': len(audit), 'users': len(users), 'tied_queries': sum((row['has_score_ties'] for row in audit))}, 'metrics': metrics}
    return (result, audit)

def _reject_json_constant(value: str) -> None:
    raise EvaluationError(f'nonstandard JSON numeric constant {value!r} is not allowed')

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise EvaluationError(f'invalid JSON in {path}: {exc}') from exc

def _read_jsonl(path: Path) -> Iterable[Any]:
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise EvaluationError(f'{path}:{line_number}: blank JSONL rows are not allowed')
            try:
                yield json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise EvaluationError(f'invalid JSON at {path}:{line_number}: {exc}') from exc

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def evaluate_files(predictions: Path, metadata: Path, expected_candidates: int=100) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result, audit = evaluate_records(_read_jsonl(predictions), _read_json(metadata), expected_candidates)
    result['inputs'] = {'predictions_file': str(predictions), 'predictions_sha256': _file_sha256(predictions), 'metadata_file': str(metadata), 'metadata_sha256': _file_sha256(metadata)}
    return (result, audit)

def aggregate_results(results: Iterable[Any]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    seeds: set[int] = set()
    reference: tuple[Any, ...] | None = None
    for index, result in enumerate(results, 1):
        if not isinstance(result, dict) or result.get('kind') != 'evaluation':
            raise EvaluationError(f'result {index} is not an evaluation result')
        if result.get('schema_version') != SCHEMA_VERSION:
            raise EvaluationError(f'result {index} has an unsupported schema_version')
        metadata = validate_metadata(result.get('metadata'))
        fingerprint = _sha256_string(result.get('evaluation_set_sha256'), f'result {index}.evaluation_set_sha256')
        evaluation = result.get('evaluation')
        counts = result.get('counts')
        if not isinstance(evaluation, dict) or not isinstance(counts, dict):
            raise EvaluationError(f'result {index} lacks evaluation settings or counts')
        if any((isinstance(counts.get(key), bool) or not isinstance(counts.get(key), int) or counts[key] < 1 for key in ('queries', 'users'))):
            raise EvaluationError(f'result {index} has invalid query/user counts')
        if counts['users'] > counts['queries']:
            raise EvaluationError(f'result {index} has more users than queries')
        _validate_metrics(result.get('metrics'), f'result {index}.metrics')
        group = (metadata['model'], metadata['dataset'], metadata['protocol_sha256'], _canonical_json(evaluation), fingerprint, counts['queries'], counts['users'])
        if reference is not None and group != reference:
            raise EvaluationError('cannot aggregate differing models, datasets, protocol fingerprints, evaluation settings, or evaluation sets')
        reference = group
        if metadata['seed'] in seeds:
            raise EvaluationError(f"cannot aggregate duplicate seed {metadata['seed']}")
        seeds.add(metadata['seed'])
        checked.append(result)
    if not checked:
        raise EvaluationError('at least one evaluation result is required')
    first = checked[0]
    summary = {}
    for name in METRIC_NAMES:
        values = [result['metrics'][name] for result in checked]
        summary[name] = {'mean': statistics.mean(values), 'sample_stddev': statistics.stdev(values) if len(values) > 1 else None}
    return {'schema_version': SCHEMA_VERSION, 'kind': 'aggregate', 'model': first['metadata']['model'], 'dataset': first['metadata']['dataset'], 'protocol_sha256': first['metadata']['protocol_sha256'].lower(), 'evaluation': first['evaluation'], 'evaluation_set_sha256': first['evaluation_set_sha256'].lower(), 'run_count': len(checked), 'seeds': sorted(seeds), 'metrics': summary, 'significance_test': None, 'interpretation': 'Mean and sample standard deviation across seeds; no significance claim.', 'runs': [{'seed': result['metadata']['seed'], 'counts': result['counts'], 'metrics': result['metrics']} for result in sorted(checked, key=lambda result: result['metadata']['seed'])]}

def _write_exclusive(artifacts: list[tuple[Path, str]]) -> None:
    resolved = [path.absolute() for path, _ in artifacts]
    if len(set(resolved)) != len(resolved):
        raise EvaluationError('output paths must be distinct')
    for path, _ in artifacts:
        if os.path.lexists(path):
            raise EvaluationError(f'refusing to overwrite existing output: {path}')
    created: list[Path] = []
    try:
        for path, content in artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                created.append(path)
                stream.write(content)
    except (OSError, UnicodeError):
        for path in created:
            path.unlink(missing_ok=True)
        raise

def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n'

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluate = commands.add_parser('evaluate', help='evaluate one prediction JSONL file')
    evaluate.add_argument('predictions', type=Path)
    evaluate.add_argument('--metadata', required=True, type=Path)
    evaluate.add_argument('--output', required=True, type=Path)
    evaluate.add_argument('--expected-candidates', type=int, default=100)
    evaluate.add_argument('--ranks-output', type=Path, help='default: OUTPUT stem followed by .ranks.jsonl')
    aggregate = commands.add_parser('aggregate', help='summarize independent seeds of one model')
    aggregate.add_argument('results', type=Path, nargs='+')
    aggregate.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'evaluate':
            result, audit = evaluate_files(args.predictions, args.metadata, args.expected_candidates)
            ranks_path = args.ranks_output or args.output.with_suffix('.ranks.jsonl')
            ranks_text = ''.join((_canonical_json(row) + '\n' for row in audit))
            result['audit'] = {'ranks_file': str(ranks_path), 'ranks_sha256': hashlib.sha256(ranks_text.encode('utf-8')).hexdigest()}
            _write_exclusive([(args.output, _pretty_json(result)), (ranks_path, ranks_text)])
            print(f"Evaluated {result['counts']['queries']} queries; results: {args.output}; audit ranks: {ranks_path}")
        else:
            summary = aggregate_results((_read_json(path) for path in args.results))
            summary['input_results'] = [{'file': str(path), 'sha256': _file_sha256(path)} for path in args.results]
            _write_exclusive([(args.output, _pretty_json(summary))])
            print(f"Aggregated {summary['run_count']} runs: {args.output}")
    except (EvaluationError, OSError, UnicodeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
