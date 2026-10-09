import argparse
from bisect import bisect_right
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.llmsrec_data import JsonlRows, MappedRows

class LlaraDataError(ValueError):
    pass

class ExcludedCatalog(Sequence):

    def __init__(self, catalog, excluded_positions):
        self.catalog = catalog
        self.excluded = sorted(set(excluded_positions))
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

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LlaraDataError(f'duplicate JSON key: {key!r}')
        result[key] = value
    return result

def _parse(text, location):
    try:
        return json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise LlaraDataError(f'{location}: {exc}') from exc

def _identifier(value, field):
    if not isinstance(value, str) or not value.strip():
        raise LlaraDataError(f'{field} must be a nonempty string')
    return value

class LlaraData:

    def __init__(self, data_dir, stage, max_history=None, train_candidate_count=None, train_candidate_seed=None, sep=', ', *, loaded=None):
        self.data_dir = Path(data_dir)
        self.stage = {'val': 'valid'}.get(stage, stage)
        if self.stage not in {'train', 'valid', 'test'}:
            raise LlaraDataError('stage must be train, valid (or val), or test')
        for name, value in (('max_history', max_history), ('train_candidate_count', train_candidate_count)):
            if value is not None and (type(value) is not int or value < 1):
                raise LlaraDataError(f'{name} must be a positive integer or None')
        if train_candidate_seed is not None and type(train_candidate_seed) is not int:
            raise LlaraDataError('train_candidate_seed must be an integer or None')
        if (train_candidate_count is None) != (train_candidate_seed is None):
            raise LlaraDataError('training sampling requires both count and seed')
        if not isinstance(sep, str):
            raise LlaraDataError('sep must be a string')
        self.max_history = max_history
        self.sep = sep
        self.train_candidate_count = train_candidate_count
        self.train_candidate_seed = train_candidate_seed
        self.source_sha256 = {}

        def read(name):
            path = self.data_dir / name
            try:
                raw = path.read_bytes()
                decoded = raw.decode('utf-8')
            except (OSError, UnicodeError) as exc:
                raise LlaraDataError(f'cannot read {path}: {exc}') from exc
            self.source_sha256[name] = hashlib.sha256(raw).hexdigest()
            return decoded
        if loaded is None:
            items = _parse(read('items.json'), 'items.json')
            split_metadata = _parse(read('splits.json'), 'splits.json')
        else:
            items, loaded_rows, split_metadata, hashes = loaded
            self.source_sha256 = {name: hashes[name] for name in ('items.json', 'splits.json', f'{self.stage}.jsonl')}
        if not isinstance(items, dict) or not items:
            raise LlaraDataError('items.json must be a nonempty object')
        self.item_id2name = {}
        reserved = ('[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]')
        for item_id, metadata in items.items():
            _identifier(item_id, 'catalog item ID')
            if not isinstance(metadata, dict):
                raise LlaraDataError(f'item {item_id!r} metadata must be an object')
            title = _identifier(metadata.get('title'), f'item {item_id!r} title')
            if any((marker in title for marker in reserved)):
                raise LlaraDataError(f'item {item_id!r} title contains a reserved LLaRA marker')
            self.item_id2name[item_id] = title
        self.index_to_item = tuple(sorted(items))
        self.item_to_index = {item: index for index, item in enumerate(self.index_to_item)}
        self.padding_item_id = len(self.index_to_item)
        self.item_count = len(self.index_to_item)
        self.splits_metadata = split_metadata
        if not isinstance(self.splits_metadata, dict):
            raise LlaraDataError('splits.json must contain an object')
        split_name = f'{self.stage}.jsonl'
        if loaded is None:
            offsets, source_hash = ([0], hashlib.sha256())
            with (self.data_dir / split_name).open('rb') as stream:
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    source_hash.update(line)
                    if line.strip():
                        offsets.append(stream.tell())
            self.source_sha256[split_name] = source_hash.hexdigest()
            raw_rows = JsonlRows(self.data_dir / split_name, offsets, source_sha256=source_hash.hexdigest())
        else:
            raw_rows = loaded_rows[self.stage]
        self._rows = MappedRows(raw_rows, self._validate_row)
        query_ids = set()
        max_history_seen, minimum_candidates, maximum_candidates = (0, None, 0)
        truncated, generated = (0, 0)
        for line_number, row in enumerate(raw_rows, 1):
            location = f'{split_name}:{line_number}'
            if not isinstance(row, dict):
                raise LlaraDataError(f'{location}: row must be an object')
            try:
                normalized = self._validate_row(row)
            except LlaraDataError as exc:
                raise LlaraDataError(f'{location}: {exc}') from exc
            if normalized['query_id'] in query_ids:
                raise LlaraDataError(f"{location}: duplicate query_id {normalized['query_id']!r}")
            query_ids.add(normalized['query_id'])
            history_length, candidate_count = (len(normalized['history']), len(normalized['candidate_ids']))
            max_history_seen = max(max_history_seen, history_length)
            minimum_candidates = candidate_count if minimum_candidates is None else min(minimum_candidates, candidate_count)
            maximum_candidates = max(maximum_candidates, candidate_count)
            truncated += bool(max_history and history_length > max_history)
            generated += normalized['training_candidates_generated']
        if not self._rows:
            raise LlaraDataError(f'{split_name} contains no examples')
        self.sequence_width = max_history or max_history_seen
        self.candidate_width = maximum_candidates
        self._summary_counts = {'candidate_count_min': minimum_candidates, 'truncated_histories': truncated, 'training_candidates_generated': generated}

    def _items(self, values, field, allow_empty=False):
        if not isinstance(values, list) or (not values and (not allow_empty)):
            raise LlaraDataError(f"{field} must be a {('possibly empty ' if allow_empty else 'nonempty ')}list")
        for item in values:
            _identifier(item, field)
            if item not in self.item_to_index:
                raise LlaraDataError(f'unknown item ID {item!r} in {field}')
        return tuple(values)

    def _validate_row(self, row):
        query_id = _identifier(row.get('query_id'), 'query_id')
        user_id = _identifier(row.get('user_id'), 'user_id')
        history = self._items(row.get('history'), 'history')
        target_id = _identifier(row.get('target_id'), 'target_id')
        if target_id not in self.item_to_index:
            raise LlaraDataError(f'unknown target_id {target_id!r}')
        generated = False
        if self.stage == 'train' and self.train_candidate_count is not None or 'candidate_ids' not in row:
            if self.stage != 'train':
                raise LlaraDataError('evaluation candidate_ids are required; re-sampling is forbidden')
            if self.train_candidate_count is None:
                raise LlaraDataError('training candidate_ids missing; set explicit training count and seed')
            excluded = set(history) | {target_id}
            pool = ExcludedCatalog(self.index_to_item, (self.item_to_index[item] for item in excluded))
            count = self.train_candidate_count - 1
            if count > len(pool):
                raise LlaraDataError('insufficient unseen items for requested training candidates')
            seed_material = json.dumps([self.train_candidate_seed, query_id], ensure_ascii=False).encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest(), 'big'))
            candidates = rng.sample(pool, count) + [target_id]
            rng.shuffle(candidates)
            generated = True
        else:
            candidates = row['candidate_ids']
        candidates = self._items(candidates, 'candidate_ids')
        if len(set(candidates)) != len(candidates):
            raise LlaraDataError('candidate_ids contain duplicates')
        if target_id not in candidates:
            raise LlaraDataError('target_id is absent from candidate_ids')
        return dict(query_id=query_id, user_id=user_id, history=history, target_id=target_id, candidate_ids=candidates, training_candidates_generated=generated)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        row = self._rows[index]
        history = row['history'][-self.sequence_width:]
        candidates = row['candidate_ids']
        seq = [self.item_to_index[item] for item in history]
        seq += [self.padding_item_id] * (self.sequence_width - len(seq))
        cans = [self.item_to_index[item] for item in candidates]
        cans += [self.padding_item_id] * (self.candidate_width - len(cans))
        seq_names = [self.item_id2name[item] for item in history]
        cans_names = [self.item_id2name[item] for item in candidates]
        title = self.item_id2name[row['target_id']]
        return {'seq': seq, 'seq_name': seq_names, 'len_seq': len(history), 'seq_str': self.sep.join(seq_names), 'cans': cans, 'cans_name': cans_names, 'cans_str': self.sep.join(cans_names), 'len_cans': len(candidates), 'item_id': self.item_to_index[row['target_id']], 'item_name': title, 'correct_answer': title, 'query_id': row['query_id'], 'user_id': row['user_id'], 'history_ids': list(history), 'candidate_ids': list(candidates), 'target_id': row['target_id'], 'original_history_length': len(row['history']), 'training_candidates_generated': row['training_candidates_generated']}

    def mapping_manifest(self):
        mapping = {'index_to_item': list(self.index_to_item), 'padding_item_id': self.padding_item_id}
        encoded = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return {**mapping, 'mapping_sha256': hashlib.sha256(encoded.encode()).hexdigest()}

    def summary(self):
        return {'stage': self.stage, 'examples': len(self), 'item_count': self.item_count, 'padding_item_id': self.padding_item_id, 'sequence_width': self.sequence_width, 'candidate_count_min': self._summary_counts['candidate_count_min'], 'candidate_count_max': self.candidate_width, 'truncated_histories': self._summary_counts['truncated_histories'], 'training_candidates_generated': self._summary_counts['training_candidates_generated'], 'source_sha256': dict(self.source_sha256), 'mapping_sha256': self.mapping_manifest()['mapping_sha256'], 'integration_status': 'data_adapter_only'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--stage', choices=('train', 'valid', 'val', 'test'), required=True)
    parser.add_argument('--max-history', type=int)
    parser.add_argument('--train-candidate-count', type=int)
    parser.add_argument('--train-candidate-seed', type=int)
    args = parser.parse_args()
    try:
        dataset = LlaraData(**vars(args))
    except LlaraDataError as exc:
        parser.error(str(exc))
    print(json.dumps(dataset.summary(), indent=2))
if __name__ == '__main__':
    main()
