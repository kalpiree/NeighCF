from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
SPLITS = ('train', 'valid', 'test')
PROTOCOL_VERSION = 'ml100k-published-protocol-reconstruction-v1'

class PreparationError(ValueError):
    pass

@dataclass(frozen=True)
class Event:
    user: str
    item: str
    rating: int
    timestamp: int

def encode(value, pretty=False):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2 if pretty else None, separators=None if pretty else (',', ':')) + '\n').encode('utf-8')

def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def positive_int(value, label):
    if type(value) is not int or value < 1:
        raise PreparationError(f'{label} must be a positive integer')
    return value

def read_raw(raw_dir):
    raw_dir = Path(raw_dir)
    titles = {}
    for number, line in enumerate((raw_dir / 'u.item').read_text(encoding='latin-1').splitlines(), 1):
        fields = line.split('|')
        if len(fields) < 2:
            raise PreparationError(f'u.item:{number}: expected ID and title')
        try:
            item = str(positive_int(int(fields[0]), 'item ID'))
        except ValueError as exc:
            raise PreparationError(f'u.item:{number}: invalid item ID') from exc
        if item in titles or not fields[1].strip():
            raise PreparationError(f'u.item:{number}: duplicate ID or empty title')
        titles[item] = {'title': fields[1]}
    if not titles:
        raise PreparationError('u.item is empty')
    grouped, seen = (defaultdict(list), set())
    for number, line in enumerate((raw_dir / 'u.data').read_text(encoding='ascii').splitlines(), 1):
        fields = line.split('\t')
        if len(fields) != 4:
            raise PreparationError(f'u.data:{number}: expected four tab-separated fields')
        try:
            user, item, rating, timestamp = map(int, fields)
            positive_int(user, 'user ID')
            positive_int(item, 'item ID')
            positive_int(timestamp, 'Unix timestamp')
            if rating not in range(1, 6):
                raise PreparationError('rating must be from 1 through 5')
        except ValueError as exc:
            raise PreparationError(f'u.data:{number}: invalid numeric field: {exc}') from exc
        user, item = (str(user), str(item))
        if item not in titles:
            raise PreparationError(f'u.data:{number}: item {item} lacks metadata')
        if (user, item) in seen:
            raise PreparationError(f'u.data:{number}: repeated user-item pair is unsupported')
        seen.add((user, item))
        grouped[user].append(Event(user, item, rating, timestamp))
    if not grouped:
        raise PreparationError('u.data is empty')
    return (titles, {user: sorted(grouped[user], key=lambda e: (e.timestamp, int(e.item))) for user in sorted(grouped, key=int)})

def split_endpoints(count):
    return (count * 8 // 10, count * 9 // 10)

def split_at(index, endpoints):
    return 'train' if index < endpoints[0] else 'valid' if index < endpoints[1] else 'test'

def filter_core(users, minimum):
    active = users
    removed, rounds = ([], [])
    while True:
        item_counts = Counter((event.item for events in active.values() for event in events))
        next_active = {}
        this_round = []
        for user, events in active.items():
            for event in events:
                low_user, low_item = (len(events) < minimum, item_counts[event.item] < minimum)
                if low_user or low_item:
                    this_round.append({'user_id': user, 'item_id': event.item, 'rating': event.rating, 'timestamp_unix_s': event.timestamp, 'filter_round': len(rounds) + 1, 'user_below_minimum': low_user, 'item_below_minimum': low_item})
                else:
                    next_active.setdefault(user, []).append(event)
        if not this_round:
            return (active, removed, rounds)
        rounds.append({'round': len(rounds) + 1, 'removed_interactions': len(this_round), 'remaining_users': len(next_active), 'remaining_interactions': sum(map(len, next_active.values()))})
        removed.extend(this_round)
        active = next_active

def sample_candidates(pool, target, seed, query_id, negative_count):

    def draws(namespace):
        key = hashlib.sha256(encode([PROTOCOL_VERSION, seed, query_id, namespace])).digest()
        counter = 0
        while True:
            yield int.from_bytes(hashlib.sha256(key + counter.to_bytes(8, 'big')).digest(), 'big')
            counter += 1

    def below(generator, bound):
        limit = (1 << 256) - (1 << 256) % bound
        while True:
            value = next(generator)
            if value < limit:
                return value % bound
    remaining, negatives = (list(pool), [])
    generator = draws('negative')
    for _ in range(negative_count):
        index = below(generator, len(remaining))
        negatives.append(remaining[index])
        remaining[index] = remaining[-1]
        remaining.pop()
    candidates = negatives + [target]
    generator = draws('order')
    for index in range(len(candidates) - 1, 0, -1):
        other = below(generator, index + 1)
        candidates[index], candidates[other] = (candidates[other], candidates[index])
    return candidates

def prepare(raw_dir, output, seed=20260930, max_history=None, negative_count=99, minimum_interactions=5):
    if type(seed) is not int:
        raise PreparationError('seed must be an integer')
    if max_history is not None:
        positive_int(max_history, 'max_history')
    positive_int(negative_count, 'negative_count')
    positive_int(minimum_interactions, 'minimum_interactions')
    raw_dir, output = (Path(raw_dir), Path(output))
    if os.path.lexists(output):
        raise PreparationError(f'refusing to overwrite existing output: {output}')
    titles, raw_users = read_raw(raw_dir)
    users, filtered_out, filter_rounds = filter_core(raw_users, minimum_interactions)
    raw_hashes = {name: digest(raw_dir / name) for name in ('u.data', 'u.item')}
    endpoints = {user: split_endpoints(len(events)) for user, events in users.items()}
    first_train = {}
    training_counts = Counter()
    raw_split_counts = Counter()
    for user, events in users.items():
        for index, event in enumerate(events):
            split = split_at(index, endpoints[user])
            raw_split_counts[split] += 1
            if split == 'train':
                training_counts[event.item] += 1
                first_train[event.item] = min(first_train.get(event.item, event.timestamp), event.timestamp)
    all_items = {event.item for events in users.values() for event in events}
    catalog = {item: titles[item] for item in sorted(all_items, key=int)}
    if len(catalog) < negative_count + 2:
        raise PreparationError('preprocessed catalog cannot supply target, nonempty history and negatives')
    settings = {'protocol_version': PROTOCOL_VERSION, 'status': 'reconstruction_from_published_protocol_not_verified_original_frozen_splits', 'dataset': 'MovieLens-100K', 'source': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip', 'source_readme': 'https://files.grouplens.org/datasets/movielens/ml-100k-README.txt', 'feedback': 'all ratings 1-5 are implicit interaction events; no rating threshold', 'paper_specified': {'subset': 'MovieLens-100K', 'minimum_user_and_item_interactions': 5, 'chronological_split': '8:1:1', 'evaluation': 'one positive and 99 random negatives'}, 'filtering': {'minimum_user_and_item_interactions': minimum_interactions, 'implementation_assumption': 'simultaneous iterative user/item k-core before splitting; paper does not specify iteration/order', 'scope': 'full dataset, as published preprocessing; depends on held-out activity and is not a temporal backtest', 'duplicate_user_item_pairs': 'rejected'}, 'split': 'per-user chronological; boundaries floor(0.8*n), floor(0.9*n), n', 'split_tie_break': 'numeric item ID ascending within equal Unix-second timestamp', 'query_unit': 'each eligible event with nonempty strictly earlier history', 'history': {'max_items': max_history, 'order': 'timestamp then numeric item ID', 'visibility': "preprocessed user's events with timestamp < target timestamp", 'evaluation': 'rolling observed history; prior valid/test events may enter later histories', 'ties': 'all same-second events excluded from history and observed-past exclusions'}, 'catalog': {'membership': 'all items surviving published preprocessing; closed-catalog assumption', 'metadata': 'original u.item title only; no held-out labels or aggregate statistics', 'availability': 'no invented release-date or first-observation eligibility constraints', 'cold_targets': 'retained even without training interactions; baselines must score every frozen candidate'}, 'candidates': {'count': negative_count + 1, 'negative_count': negative_count, 'seed': seed, 'sampler': 'SHA-256 counter draws with rejection sampling and partial Fisher-Yates; separate order namespace', 'exclusions': 'entire strictly earlier observed user prefix plus target, before history truncation', 'future_interactions': 'conditional on frozen preprocessed catalog, never consulted per query; a future positive may be a sampled unobserved negative', 'same_second_interactions': 'unobserved at query time, so eligible as negatives', 'insufficient_pool': 'exclude and audit query; never shrink candidate set', 'applies_to': 'train, valid and test; frozen once and shared across all model seeds'}, 'timestamp': {'source_unit': 'Unix seconds', 'query_fields': 'timestamp/history_timestamps are milliseconds = source seconds * 1000', 'audit_field': 'timestamp_unix_s in interactions.jsonl and every query'}, 'offline_training': 'all training prefixes may be used to fit a frozen model; no global wall-clock chronology guarantee', 'neighbor_context': 'offline collaborator index must use train_interactions.jsonl only; never valid/test labels', 'metric_contract': {'averaging': 'query_macro', 'cutoffs': [1, 5, 10], 'mrr': 'full candidate list', 'tie_break': 'score descending then canonical string item ID ascending'}}
    exclusions, query_counts = ({split: Counter() for split in SPLITS}, Counter())
    query_users = {split: set() for split in SPLITS}
    query_ranges = {split: [None, None] for split in SPLITS}
    history_stats = {split: Counter() for split in SPLITS}
    total_same_second_past = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    handles = {}
    try:
        filenames = [f'{split}.jsonl' for split in SPLITS] + ['interactions.jsonl', 'train_interactions.jsonl', 'excluded_queries.jsonl', 'filtered_interactions.jsonl']
        handles = {name: (output / name).open('xb') for name in filenames}
        (output / 'items.json').write_bytes(encode(catalog, True))
        (output / 'catalog.json').write_bytes(encode({item: {'first_train_timestamp_unix_s': first_train.get(item), 'training_interactions': training_counts[item]} for item in sorted(catalog, key=int)}, True))
        for removed in filtered_out:
            handles['filtered_interactions.jsonl'].write(encode(removed))
        for user, events in users.items():
            past, cursor = ([], 0)
            for index, event in enumerate(events):
                while cursor < index and events[cursor].timestamp < event.timestamp:
                    past.append(events[cursor])
                    cursor += 1
                split = split_at(index, endpoints[user])
                query_id = f'ml100k:u{user}:i{event.item}:t{event.timestamp}'
                audit = {'query_id': query_id, 'user_id': user, 'item_id': event.item, 'rating': event.rating, 'timestamp_unix_s': event.timestamp, 'split': split, 'user_event_index': index}
                handles['interactions.jsonl'].write(encode(audit))
                if split == 'train':
                    handles['train_interactions.jsonl'].write(encode(audit))
                visible = [previous for previous in past if previous.item in catalog]
                observed = {previous.item for previous in past}
                pool = [item for item in catalog if item != event.item and item not in observed]
                reason = None
                if not visible:
                    reason = 'empty_strictly_earlier_catalog_history'
                elif len(pool) < negative_count:
                    reason = 'insufficient_eligible_negatives'
                if reason:
                    exclusions[split][reason] += 1
                    handles['excluded_queries.jsonl'].write(encode({**audit, 'reason': reason, 'eligible_negative_count': len(pool)}))
                    continue
                retained = visible[-max_history:] if max_history else visible
                candidates = sample_candidates(pool, event.item, seed, query_id, negative_count)
                tied_past = index - len(past)
                total_same_second_past += tied_past
                row = {'query_id': query_id, 'user_id': user, 'history': [previous.item for previous in retained], 'history_timestamps': [previous.timestamp * 1000 for previous in retained], 'target_id': event.item, 'candidate_ids': candidates, 'timestamp': event.timestamp * 1000, 'timestamp_unix_s': event.timestamp, 'user_event_index': index, 'history_length_before_truncation': len(visible), 'observed_past_count': len(past), 'same_second_predecessors_excluded': tied_past}
                handles[f'{split}.jsonl'].write(encode(row))
                query_counts[split] += 1
                query_users[split].add(user)
                low, high = query_ranges[split]
                query_ranges[split] = [min(low, event.timestamp) if low is not None else event.timestamp, max(high, event.timestamp) if high is not None else event.timestamp]
                history_stats[split]['items'] += len(retained)
                history_stats[split]['truncated_queries'] += len(retained) < len(visible)
                history_stats[split]['max_items'] = max(history_stats[split]['max_items'], len(retained))
        for stream in handles.values():
            stream.close()
        for split in SPLITS:
            if query_counts[split] < 1:
                raise PreparationError(f'{split} has no eligible queries; cannot freeze a usable comparison')
        stats = {'raw': {'users': len(raw_users), 'metadata_items': len(titles), 'observed_items': len({e.item for seq in raw_users.values() for e in seq}), 'interactions': sum(map(len, raw_users.values()))}, 'preprocessed': {'users': len(users), 'items': len(catalog), 'interactions': sum(map(len, users.values())), 'split_interactions': dict(raw_split_counts)}, 'paper_table_counts': {'users': 943, 'items': 1682, 'interactions': 100000}, 'filter_rounds': filter_rounds, 'filtered_interactions': len(filtered_out), 'catalog_items': len(catalog), 'training_observed_catalog_items': len(first_train), 'catalog_items_without_training_interactions': len(set(catalog) - set(first_train)), 'query_counts': {s: query_counts[s] for s in SPLITS}, 'query_user_counts': {s: len(query_users[s]) for s in SPLITS}, 'query_timestamp_ranges_unix_s': query_ranges, 'excluded_query_counts': {s: dict(exclusions[s]) for s in SPLITS}, 'history': {s: {**dict(history_stats[s]), 'mean_items': history_stats[s]['items'] / query_counts[s]} for s in SPLITS}, 'same_second_predecessors_excluded_across_retained_queries': total_same_second_past}
        output_names = filenames + ['items.json', 'catalog.json']
        manifest = {'schema_version': 1, 'protocol': settings, 'raw_file_sha256': raw_hashes, 'statistics': stats, 'output_file_sha256': {name: digest(output / name) for name in output_names}, 'fingerprint_rule': 'protocol_sha256 = SHA256 of exact manifest.json bytes; splits.json is a derived pointer and excluded to avoid circular hashing'}
        manifest_bytes = encode(manifest, True)
        (output / 'manifest.json').write_bytes(manifest_bytes)
        protocol_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        split_metadata = {'protocol_sha256': protocol_sha256, 'manifest': 'manifest.json', 'dataset': 'MovieLens-100K', 'protocol_version': PROTOCOL_VERSION, 'history_max_items': max_history, 'timestamp_unit': 'milliseconds', 'query_counts': stats['query_counts'], 'raw_split_interactions': dict(raw_split_counts), 'split_endpoints': {u: {'train_end_exclusive': endpoints[u][0], 'valid_end_exclusive': endpoints[u][1], 'event_count': len(users[u])} for u in users}}
        (output / 'splits.json').write_bytes(encode(split_metadata, True))
        return {'protocol_sha256': protocol_sha256, **stats}
    except BaseException:
        for stream in handles.values():
            stream.close()
        shutil.rmtree(output)
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_dir', type=Path, help='official ml-100k folder containing u.data and u.item')
    parser.add_argument('output', type=Path, help='new output folder (must not already exist)')
    parser.add_argument('--seed', type=int, default=20260930, help='fixed candidate seed; keep unchanged across model seeds')
    parser.add_argument('--max-history', type=int, help='keep most recent N strictly earlier items; default preserves full history')
    parser.add_argument('--negative-count', type=int, default=99, help='default99; changing this defines a different protocol')
    parser.add_argument('--minimum-interactions', type=int, default=5, help='published minimum for users AND items; iterative k-core')
    args = parser.parse_args(argv)
    try:
        result = prepare(**vars(args))
    except (PreparationError, OSError, UnicodeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
