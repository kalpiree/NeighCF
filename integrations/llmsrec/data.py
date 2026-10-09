from __future__ import annotations
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))
from adapters.llmsrec_data import DataError, ExcludedCatalog, index_queries, load_canonical, prediction_record
MARKERS = ('[PAD]', '[UserRep]', '[HistoryEmb]', '[UserOut]', '[ItemOut]', '[CLS]', '</s>')

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def evaluation_contract(expected_candidates):
    if type(expected_candidates) is not int or expected_candidates < 2:
        raise DataError('evaluation requires one true item and at least one negative')
    return {'candidate_count': expected_candidates, 'negative_count': expected_candidates - 1, 'positive_count': 1, 'published_manuscript_candidate_count': 100, 'origin': 'user_override_50_negatives' if expected_candidates == 51 else 'published_manuscript' if expected_candidates == 100 else 'explicit_cli_configuration', 'changes_manuscript_candidate_count': expected_candidates != 100}

def load_data(path, expected_candidates=51, max_length=128, omit_dates=False):
    evaluation_contract(expected_candidates)

    def validate_query(query, split, number):
        if not query['history']:
            raise DataError(f"{query['query_id']}: empty histories are unsupported")
        if not omit_dates:
            timestamps = query.get('history_timestamps')
            if not isinstance(timestamps, list) or len(timestamps) != len(query['history']):
                raise DataError(f"{query['query_id']}: history_timestamps must align with history; provide actual millisecond event times or explicitly use --omit-dates")
            for timestamp in timestamps:
                utc_day(timestamp)
            if any((a > b for a, b in zip(timestamps, timestamps[1:]))):
                raise DataError(f"{query['query_id']}: history timestamps are not chronological")
    items, raw, split_metadata, hashes = load_canonical(Path(path), expected_candidates, row_validator=validate_query)
    for item_id, metadata in items.items():
        if any((token in metadata['title'] for token in MARKERS)):
            raise DataError(f'reserved model marker in title for {item_id!r}')
    if len(raw['train']) < 2:
        raise DataError('LLM-SRec requires at least two training queries for pairwise uniformity')
    mapping, indexed = index_queries(items, raw, max_length)
    return {'items': items, 'rows': indexed, 'mapping': mapping, 'mapping_sha256': digest(mapping), 'source_file_sha256': hashes, 'source_split_metadata': split_metadata, 'max_length': max_length, 'expected_candidates': expected_candidates, 'omit_dates': omit_dates, 'catalog': range(1, len(items) + 1)}

def utc_day(timestamp):
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or (not math.isfinite(timestamp)):
        raise DataError('timestamps must be finite milliseconds since Unix epoch')
    try:
        return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime('%Y-%m-%d')
    except (ValueError, OverflowError, OSError) as exc:
        raise DataError('timestamp is outside supported calendar range') from exc

def batch_indices(count, batch_size, *, seed=None, min_size=1):
    if count < min_size or batch_size < min_size:
        raise DataError(f'count and batch_size must be at least {min_size}')
    indices = list(range(count))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    batches = [indices[i:i + batch_size] for i in range(0, count, batch_size)]
    if len(batches) > 1 and len(batches[-1]) < min_size:
        batches[-2].extend(batches.pop())
    return batches

def cosine_multiplier(step, total_steps, warmup_steps):
    if total_steps < 1 or not 0 <= warmup_steps <= total_steps:
        raise DataError('invalid warmup/cosine step counts')
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def training_candidates(query, catalog, seed, epoch, count=4):
    banned = set(query['history_item_indices']) | {query['target_item_index']}
    if isinstance(catalog, range) and catalog.start == 1 and (catalog.step == 1):
        available = ExcludedCatalog(catalog, [item - 1 for item in banned if item in catalog])
    else:
        available = [item for item in catalog if item not in banned]
    if len(available) < count - 1:
        raise DataError(f"{query['source']['query_id']}: fewer than {count - 1} eligible training negatives")
    key = digest([seed, epoch, query['source']['query_id'], 'training-negatives'])
    return [query['target_item_index']] + random.Random(key).sample(available, count - 1)

def item_prompt(bundle, item_index):
    original = bundle['mapping']['index_to_item'][item_index]
    title = bundle['items'][original]['title']
    return 'The item title and item embedding are as follows: "' + title + '"[HistoryEmb], then generate item representation token:[ItemOut]'

def user_prompt(bundle, query):
    source = query['source']
    selected_ids = query['history_item_indices'][-10:]
    selected_titles = [bundle['items'][item]['title'] for item in source['history'][-10:]]
    selected_times = source.get('history_timestamps', [])[-10:]
    parts = []
    for position, title in enumerate(selected_titles, 1):
        text = f'Item No.{position}, '
        if not bundle['omit_dates']:
            text += f'Time: {utc_day(selected_times[position - 1])}, '
        parts.append(text + '"' + title + '"[HistoryEmb]')
    return ('This user has made a series of purchases in the following order: ' + ','.join(parts) + '. Based on this sequence of purchases, generate user representation token:[UserOut]', selected_ids)

def teacher_contract(bundle):
    return {'mapping_sha256': bundle['mapping_sha256'], 'train_sha256': bundle['source_file_sha256']['train.jsonl'], 'max_length': bundle['max_length'], 'hidden_units': 64, 'item_count': len(bundle['items']), 'user_count': len(bundle['mapping']['user_to_index']), 'objective': 'official SASRec BCE, canonical last-position target once per query'}

def validate_teacher_contract(bundle, metadata):
    expected = teacher_contract(bundle)
    if metadata.get('contract') != expected:
        raise DataError('SASRec teacher metadata does not match the exact item/user map, training split, or sequence settings')

def protocol_hash(bundle, explicit=None):
    value = explicit or bundle['source_split_metadata'].get('protocol_sha256')
    if not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value.lower())):
        raise DataError('provide the common --protocol-sha256 or splits.json protocol_sha256')
    return value.lower()

def verify_restored_validation(score, queries, expected, *, atol=0.0001, rtol=0.001):
    if not queries or len(expected) != len(queries):
        raise DataError('saved validation records do not cover the full scoring context')
    restored = score(queries)
    if len(restored) != len(expected):
        raise DataError('restored validation records do not cover every selected query')
    max_error, candidate_count, seen = (0.0, 0, set())
    for query, previous, current in zip(queries, expected, restored):
        query_id = query['source']['query_id']
        if query_id in seen:
            raise DataError('duplicate query in restored validation context')
        seen.add(query_id)
        if previous.get('query_id') != query_id or current.get('query_id') != query_id:
            raise DataError('restored validation query identity or order changed')
        for field in ('user_id', 'target_id', 'candidate_ids'):
            if current.get(field) != previous.get(field):
                raise DataError(f'{query_id}: restored validation {field} changed')
        count = len(previous['candidate_ids'])
        if not count or len(previous['scores']) != count or len(current['scores']) != count:
            raise DataError(f'{query_id}: restored candidate score count changed')
        for before, after in zip(previous['scores'], current['scores']):
            if isinstance(before, bool) or isinstance(after, bool) or (not isinstance(before, (int, float))) or (not isinstance(after, (int, float))) or (not math.isfinite(before)) or (not math.isfinite(after)):
                raise DataError(f'{query_id}: nonfinite or invalid restored validation score')
            difference = abs(before - after)
            max_error = max(max_error, difference)
            if difference > atol + rtol * abs(before):
                raise DataError(f'{query_id}: fresh checkpoint restoration changed a candidate score beyond tolerance')
        candidate_count += count
    return {'restore_validation_queries': len(queries), 'restore_candidate_scores': candidate_count, 'restore_max_absolute_score_error': max_error, 'restore_scoring_context': 'complete selected validation query list with identical item/user batching'}
