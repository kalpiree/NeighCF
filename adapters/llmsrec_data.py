from __future__ import annotations
import argparse
from array import array
from bisect import bisect_right
from collections.abc import Sequence as RowSequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
SPLITS = ('train', 'valid', 'test')
SCHEMA_VERSION = 1

class DataError(ValueError):
    pass

def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DataError(f'{label} must be a positive integer')
    return value

def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataError(f'{label} must be a nonempty string')
    return value

def _no_constant(value: str) -> None:
    raise DataError(f'nonstandard JSON constant {value!r} is not allowed')

def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f'duplicate JSON key {key!r}')
        result[key] = value
    return result

def _loads(text: str, label: str) -> Any:
    try:
        return json.loads(text, parse_constant=_no_constant, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise DataError(f'{label}: {exc}') from exc

def _json(value: Any, pretty: bool=False) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2 if pretty else None, separators=None if pretty else (',', ':')) + '\n'

def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def _id_list(value: Any, items: dict[str, Any], label: str) -> list[str]:
    if not isinstance(value, list):
        raise DataError(f'{label} must be an array of item ID strings')
    for position, item_id in enumerate(value):
        _string(item_id, f'{label}[{position}]')
        if item_id not in items:
            raise DataError(f'{label}[{position}]: unknown item ID {item_id!r}')
    return value

class ExcludedCatalog(RowSequence):

    def __init__(self, catalog, excluded_positions):
        self.catalog = catalog
        self.excluded = sorted(set(excluded_positions))
        if any((type(i) is not int or not 0 <= i < len(catalog) for i in self.excluded)):
            raise DataError('excluded catalog positions must address actual items')
        self.length = len(catalog) - len(self.excluded)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        lo, hi = (index, index + len(self.excluded))
        while lo < hi:
            middle = (lo + hi) // 2
            if middle - bisect_right(self.excluded, middle) < index:
                lo = middle + 1
            else:
                hi = middle
        return self.catalog[lo]

class RowView(RowSequence):

    def __init__(self, rows, indices):
        self.rows, self.indices = (rows, indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RowView(self.rows, self.indices[key])
        return self.rows[self.indices[key]]

    @property
    def query_ids(self):
        identities = getattr(self.rows, 'query_ids', None)
        return None if identities is None else RowView(identities, self.indices)

class JsonlRows(RowSequence):

    def __init__(self, path, offsets, source_sha256=None):
        self._fd, self._pid = (None, None)
        self.path = Path(path)
        self.offsets = offsets if isinstance(offsets, array) else array('Q', offsets)
        self.source_sha256 = source_sha256
        self._signature = self._stat_signature(self.path.stat())
        self.user_ids, self.query_ids = (None, None)

    @staticmethod
    def _stat_signature(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    @classmethod
    def scan(cls, path, validator=None):
        path = Path(path)
        offsets = array('Q', [0])
        hashed = hashlib.sha256()
        initial = cls._stat_signature(path.stat())
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                hashed.update(line)
                if not line.strip():
                    raise DataError(f'{path.name}:{number}: blank JSONL rows are not allowed')
                row = _loads(line.decode('utf-8'), f'{path.name}:{number}')
                if validator is not None:
                    validator(row, number)
                offsets.append(stream.tell())
        if initial != cls._stat_signature(path.stat()):
            raise DataError('canonical input changed while it was being indexed')
        return cls(path, offsets, hashed.hexdigest())

    def __len__(self):
        return len(self.offsets) - 1

    def _descriptor(self):
        if self._fd is None or self._pid != os.getpid():
            self.close()
            self._fd, self._pid = (os.open(self.path, os.O_RDONLY), os.getpid())
        if self._stat_signature(os.fstat(self._fd)) != self._signature or self._stat_signature(self.path.stat()) != self._signature:
            raise DataError('canonical JSONL input changed after validation')
        return self._fd

    def __getitem__(self, index):
        if isinstance(index, slice):
            return RowView(self, range(len(self))[index])
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start, end = (self.offsets[index], self.offsets[index + 1])
        data = os.pread(self._descriptor(), end - start, start)
        if len(data) != end - start:
            raise DataError('canonical row was truncated after validation')
        return _loads(data.decode('utf-8'), f'{self.path.name}:{index + 1}')

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        return {**self.__dict__, '_fd': None, '_pid': None}

class MappedRows(RowSequence):

    def __init__(self, rows, transform):
        self.rows, self.transform = (rows, transform)
        self.user_ids = getattr(rows, 'user_ids', None)
        self.query_ids = getattr(rows, 'query_ids', None)
        self.source_sha256 = getattr(rows, 'source_sha256', None)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return RowView(self, range(len(self))[index])
        return self.transform(self.rows[index])

    def __iter__(self):
        for row in self.rows:
            yield self.transform(row)

def load_canonical(source: Path, expected_candidates: int=100, *, row_validator=None) -> tuple[dict[str, Any], dict[str, RowSequence], dict[str, Any], dict[str, str]]:
    _positive_int(expected_candidates, 'expected_candidates')
    source = Path(source)
    contents = {name: (source / name).read_bytes() for name in ('items.json', 'splits.json')}
    texts = {name: value.decode('utf-8') for name, value in contents.items()}
    items = _loads(texts['items.json'], 'items.json')
    if not isinstance(items, dict) or not items:
        raise DataError('items.json must be a nonempty object mapping item IDs to metadata')
    for item_id, metadata in items.items():
        _string(item_id, 'items.json item ID')
        if not isinstance(metadata, dict):
            raise DataError(f'items.json[{item_id!r}] must be a metadata object')
        _string(metadata.get('title'), f'items.json[{item_id!r}].title')
    split_metadata = _loads(texts['splits.json'], 'splits.json')
    if not isinstance(split_metadata, dict):
        raise DataError('splits.json must be an object')
    rows = {}
    hashes = {name: _sha256(data) for name, data in contents.items()}
    query_ids: set[str] = set()
    for split in SPLITS:
        users, identities = (set(), [])

        def validate(row, line_number):
            label = f'{split}.jsonl:{line_number}'
            if not isinstance(row, dict):
                raise DataError(f'{label}: query must be an object')
            query_id = _string(row.get('query_id'), f'{label}.query_id')
            if query_id in query_ids:
                raise DataError(f'{label}: duplicate query_id {query_id!r} across dataset')
            query_ids.add(query_id)
            users.add(_string(row.get('user_id'), f'{label}.user_id'))
            identities.append(query_id)
            _id_list(row.get('history'), items, f'{label}.history')
            target = _string(row.get('target_id'), f'{label}.target_id')
            if target not in items:
                raise DataError(f'{label}: unknown target item ID {target!r}')
            if 'candidate_ids' in row or split != 'train':
                candidates = _id_list(row.get('candidate_ids'), items, f'{label}.candidate_ids')
                if split != 'train' and len(candidates) != expected_candidates:
                    raise DataError(f'{label}: expected {expected_candidates} candidates, found {len(candidates)}')
                if len(candidates) != len(set(candidates)):
                    raise DataError(f'{label}: duplicate candidate item IDs')
                if target not in candidates:
                    raise DataError(f'{label}: target must occur exactly once in candidate_ids')
            if row_validator is not None:
                row_validator(row, split, line_number)
        rows[split] = JsonlRows.scan(source / f'{split}.jsonl', validate)
        rows[split].user_ids, rows[split].query_ids = (frozenset(users), tuple(identities))
        hashes[f'{split}.jsonl'] = rows[split].source_sha256
        if not rows[split]:
            raise DataError(f'{split}.jsonl contains no queries')
    return (items, rows, split_metadata, hashes)

def index_queries(items: dict[str, Any], rows: dict[str, RowSequence], max_length: int=128) -> tuple[dict[str, Any], dict[str, RowSequence]]:
    _positive_int(max_length, 'max_length')
    item_ids = sorted(items)
    user_ids = sorted(set().union(*(rows[split].user_ids if getattr(rows[split], 'user_ids', None) is not None else {row['user_id'] for row in rows[split]} for split in SPLITS)))
    item_to_index = {item: index for index, item in enumerate(item_ids, 1)}
    user_to_index = {user: index for index, user in enumerate(user_ids, 1)}
    mappings = {'schema_version': SCHEMA_VERSION, 'padding_index': 0, 'item_to_index': item_to_index, 'user_to_index': user_to_index, 'index_to_item': [None] + item_ids, 'index_to_user': [None] + user_ids}
    indexed: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:

        def convert(row):
            history = [item_to_index[item] for item in row['history']]
            retained = history[-max_length:]
            converted = {'source': row, 'user_index': user_to_index[row['user_id']], 'history_item_indices': history, 'sequence_item_indices': [0] * (max_length - len(retained)) + retained, 'target_item_index': item_to_index[row['target_id']], 'history_truncated': len(history) > max_length}
            if 'candidate_ids' in row:
                converted['candidate_item_indices'] = [item_to_index[item] for item in row['candidate_ids']]
            return converted
        indexed[split] = MappedRows(rows[split], convert)
    return (mappings, indexed)

def prediction_record(indexed_query: dict[str, Any], scores: Sequence[float]) -> dict[str, Any]:
    source = indexed_query['source']
    candidates = source.get('candidate_ids')
    if not isinstance(candidates, list) or not candidates:
        raise DataError('query has no candidate_ids to score')
    if len(candidates) != len(set(candidates)) or source.get('target_id') not in candidates:
        raise DataError('prediction source has invalid candidate IDs/target')
    if len(scores) != len(candidates):
        raise DataError('scores must have the same length/order as candidate_ids')
    values = []
    for value in scores:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise DataError('scores must be finite numbers; convert tensors/arrays to Python lists')
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise DataError('scores must be finite numbers') from exc
        if not math.isfinite(numeric):
            raise DataError('scores must be finite numbers')
        values.append(numeric)
    return {'query_id': source['query_id'], 'user_id': source['user_id'], 'target_id': source['target_id'], 'candidate_ids': list(candidates), 'scores': values}

def export_dataset(source: Path, output: Path, max_length: int=128, expected_candidates: int=100) -> dict[str, Any]:
    _positive_int(max_length, 'max_length')
    source, output = (Path(source), Path(output))
    items, rows, split_metadata, source_hashes = load_canonical(source, expected_candidates)
    mappings, indexed = index_queries(items, rows, max_length)
    item_metadata = {str(mappings['item_to_index'][item_id]): {'item_id': item_id, 'metadata': metadata} for item_id, metadata in items.items()}
    files = {'id_maps.json': _json(mappings, True), 'items.indexed.json': _json(item_metadata, True)}
    for split in SPLITS:
        files[f'{split}.indexed.jsonl'] = ''.join((_json(row) for row in indexed[split]))
    manifest = {'schema_version': SCHEMA_VERSION, 'model': 'LLM-SRec', 'integration_status': 'data_export_only', 'requires_runtime_adapter': True, 'source_directory': str(source.resolve()), 'source_file_sha256': source_hashes, 'source_bundle_sha256': _sha256(_json(source_hashes).encode('utf-8')), 'source_split_metadata': split_metadata, 'item_count': len(items), 'user_count': len(mappings['user_to_index']), 'sequence_policy': {'max_length': max_length, 'padding_index': 0, 'padding_side': 'left', 'truncate_side': 'left', 'full_source_history_retained': True}, 'evaluation_candidates': expected_candidates, 'candidate_policy': 'preserve IDs and order; never sample or insert target', 'timestamp_policy': 'no timestamps invented; optional source fields retained unmodified', 'temporal_leakage_checked': False, 'counts': {split: {'queries': len(indexed[split]), 'truncated_histories': sum((row['history_truncated'] for row in indexed[split])), 'empty_histories': sum((not row['history_item_indices'] for row in indexed[split]))} for split in SPLITS}, 'output_file_sha256': {name: _sha256(content.encode('utf-8')) for name, content in files.items()}}
    files['manifest.json'] = _json(manifest, True)
    if os.path.lexists(output):
        raise DataError(f'refusing to overwrite existing output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        for name, content in files.items():
            with (output / name).open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(content)
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='canonical dataset directory')
    parser.add_argument('output', type=Path, help='new indexed-data directory')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--expected-candidates', type=int, default=100)
    args = parser.parse_args(argv)
    try:
        manifest = export_dataset(args.source, args.output, args.max_length, args.expected_candidates)
    except (DataError, OSError, UnicodeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(f'Exported indexed query data to {args.output}. Training and fixed-candidate scoring still require an LLM-SRec runtime adapter.')
    print('Query counts: ' + ', '.join((f"{split}={manifest['counts'][split]['queries']}" for split in SPLITS)))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
