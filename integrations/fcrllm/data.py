from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import sys
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))
from adapters.llmsrec_data import DataError, JsonlRows

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')

def versions():
    result = {'python': sys.version}
    for name in ('torch', 'numpy', 'scikit-learn', 'transformers', 'accelerate', 'sentencepiece'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result

def load_bundle(path, protocol=None, expected_candidates=100):
    path = Path(path)
    items = json.loads((path / 'items.json').read_text())
    if not isinstance(items, dict) or not items:
        raise DataError('items.json must be a nonempty item metadata mapping')
    for item, meta in items.items():
        if not isinstance(item, str) or not item or (not isinstance(meta, dict)) or (not isinstance(meta.get('title'), str)) or (not meta['title'].strip()):
            raise DataError('all item IDs and titles must be nonempty strings')
    metadata = json.loads((path / 'splits.json').read_text())
    if metadata.get('expected_candidates', expected_candidates) != expected_candidates:
        raise DataError('explicit evaluation candidate count differs from frozen bundle')
    recorded = metadata.get('protocol_sha256')
    protocol = protocol or recorded
    if not isinstance(protocol, str) or not re.fullmatch('[a-fA-F0-9]{64}', protocol):
        raise DataError('common protocol SHA256 required')
    if recorded and protocol.lower() != recorded.lower():
        raise DataError('explicit protocol differs from frozen bundle')
    names = ['items.json', 'splits.json', 'train.jsonl', 'valid.jsonl', 'test.jsonl']
    ids = sorted(items)
    return {'path': path, 'items': items, 'item_ids': ids, 'item_to_index': {item: i for i, item in enumerate(ids, 1)}, 'mapping_sha256': digest(ids), 'protocol_sha256': protocol.lower(), 'source_file_sha256': {name: file_digest(path / name) for name in names}, 'split_metadata': metadata, 'expected_candidates': expected_candidates}

def validate_row(row, bundle, split, number, seen):
    prefix = f'{split}:{number}'
    for field in ('query_id', 'user_id', 'target_id'):
        if not isinstance(row.get(field), str) or not row[field]:
            raise DataError(f'{prefix}: {field} must be a string')
    if row['query_id'] in seen:
        raise DataError(f'{prefix}: duplicate query ID')
    seen.add(row['query_id'])
    history = row.get('history')
    if not isinstance(history, list) or not history or any((item not in bundle['items'] for item in history)):
        raise DataError(f'{prefix}: history must be nonempty and use catalog items')
    if row['target_id'] not in bundle['items'] or row['target_id'] in history:
        raise DataError(f'{prefix}: target is unknown or occurs in its history')
    if 'history_timestamps' in row:
        times = row['history_timestamps']
        if len(times) != len(history) or times != sorted(times):
            raise DataError(f'{prefix}: history timestamps do not match chronological history')
        if 'timestamp' in row and any((t >= row['timestamp'] for t in times)):
            raise DataError(f'{prefix}: history reaches query time')
    if split != 'train':
        c = row.get('candidate_ids')
        if not isinstance(c, list) or len(c) != bundle['expected_candidates'] or len(set(c)) != len(c) or (c.count(row['target_id']) != 1):
            raise DataError(f'{prefix}: invalid frozen candidates')
        if any((item not in bundle['items'] for item in c)) or set(c) & set(history):
            raise DataError(f'{prefix}: candidate outside catalog or in full history')

def indexed_rows(bundle, split):
    if split not in ('train', 'valid', 'test'):
        raise DataError('unknown split')
    seen = set()
    rows = JsonlRows.scan(bundle['path'] / f'{split}.jsonl', validator=lambda row, number: validate_row(row, bundle, split, number, seen))
    if not seen:
        raise DataError(f'{split} split is empty')
    if rows.source_sha256 != bundle['source_file_sha256'][f'{split}.jsonl']:
        raise DataError('canonical split changed while indexing')
    return rows

def read_rows(bundle, split):
    if split not in ('train', 'valid', 'test'):
        raise DataError('unknown split')
    seen = set()
    with (bundle['path'] / f'{split}.jsonl').open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            validate_row(row, bundle, split, number, seen)
            yield row
    if not seen:
        raise DataError(f'{split} split is empty')

def train_item_ids(bundle):
    seen = set()
    for row in read_rows(bundle, 'train'):
        seen.update(row['history'])
        seen.add(row['target_id'])
    return sorted(seen)

def feature_contract(bundle, train_ids):
    return {'protocol_sha256': bundle['protocol_sha256'], 'mapping_sha256': bundle['mapping_sha256'], 'items_sha256': bundle['source_file_sha256']['items.json'], 'train_sha256': bundle['source_file_sha256']['train.jsonl'], 'pca_fit_item_ids_sha256': digest(train_ids), 'pca_fit_items': len(train_ids), 'pca_dimension': 64}

def validate_features(bundle, path, allow_protocol_rebind=False):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    expected = feature_contract(bundle, train_item_ids(bundle))
    actual = manifest.get('contract', {})
    if actual != expected:
        parent = bundle['split_metadata'].get('candidate_derivation', {}).get('source_protocol_sha256')
        only_protocol = {k: v for k, v in actual.items() if k != 'protocol_sha256'} == {k: v for k, v in expected.items() if k != 'protocol_sha256'}
        if not (allow_protocol_rebind and parent == actual.get('protocol_sha256') and only_protocol):
            raise DataError('features have different catalog, training data, PCA fit population, or protocol; explicit candidate-only protocol rebind requires exact parent provenance')
    if manifest.get('item_ids') != bundle['item_ids'] or manifest.get('status') != 'complete':
        raise DataError('features must be complete with exact sorted item row order')
    for name in ('item_embeddings.npy', 'pca64.npy', 'pca_model.npz'):
        if manifest.get('file_sha256', {}).get(name) != file_digest(path / name):
            raise DataError(f'feature artifact hash mismatch: {name}')
    return manifest

def training_negative(row, bundle, seed, epoch):
    banned = set(row['history']) | {row['target_id']}
    if len(banned) >= len(bundle['item_ids']):
        raise DataError('no eligible training negative')
    rng = random.Random(digest([seed, epoch, row['query_id'], 'fcrllm-negative']))
    while True:
        candidate = bundle['item_ids'][rng.randrange(len(bundle['item_ids']))]
        if candidate not in banned:
            return bundle['item_to_index'][candidate]

def sequence_values(row, bundle, max_history):
    if max_history < 1:
        raise DataError('history limit must be positive')
    history = [bundle['item_to_index'][x] for x in row['history'][-max_history:]]
    pad = [0] * (max_history - len(history))
    return (pad + history, pad + list(range(1, len(history) + 1)))

def batches(count, batch_size, seed=None):
    if count < 1 or batch_size < 1:
        raise DataError('positive query count and batch size required')
    indices = list(range(count))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    for start in range(0, count, batch_size):
        yield indices[start:start + batch_size]

def cosine_multiplier(step, total, warmup):
    if total < 1 or not 0 <= step < total or (not 0 <= warmup <= total):
        raise DataError('invalid scheduler step counts')
    if warmup and step < warmup:
        return (step + 1) / warmup
    if step == total - 1:
        return 0.0
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    return 0.5 * (1 + math.cos(math.pi * progress))

def prediction_record(row, scores):
    if len(scores) != len(row['candidate_ids']) or not all((math.isfinite(x) for x in scores)):
        raise DataError('one finite score per frozen candidate required')
    return {k: row[k] for k in ('query_id', 'user_id', 'target_id', 'candidate_ids')} | {'scores': scores}
