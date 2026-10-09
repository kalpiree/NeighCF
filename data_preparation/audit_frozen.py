import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

class AuditError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise AuditError(message)

def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            result.update(data)
    return result.hexdigest()

def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f'{path.name}:{number}: blank row')
            yield json.loads(line)

def audit(directory, raw_dir=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    splits = json.loads((directory / 'splits.json').read_text())
    fingerprint = sha256(directory / 'manifest.json')
    require(splits['protocol_sha256'] == fingerprint, 'protocol fingerprint mismatch')
    for name, expected in manifest['output_file_sha256'].items():
        require(sha256(directory / name) == expected, f'hash mismatch: {name}')
    events, by_id, item_counts = (defaultdict(list), {}, Counter())
    for event in read_rows(directory / 'interactions.jsonl'):
        require(event['query_id'] not in by_id, 'duplicate interaction ID')
        by_id[event['query_id']] = event
        events[event['user_id']].append(event)
        item_counts[event['item_id']] += 1
    minimum = manifest['protocol']['filtering']['minimum_user_and_item_interactions']
    require(all((count >= minimum for count in item_counts.values())), 'item frequency filter not satisfied')
    require(all((len(sequence) >= minimum for sequence in events.values())), 'user frequency filter not satisfied')
    catalog = json.loads((directory / 'items.json').read_text())
    catalog_audit = json.loads((directory / 'catalog.json').read_text())
    require(set(item_counts) == set(catalog) == set(catalog_audit), 'closed catalog differs from preprocessed items')
    training_ids, timestamps = (set(), {})
    first_train, training_counts = ({}, Counter())
    for user, sequence in events.items():
        require(sequence == sorted(sequence, key=lambda e: (e['timestamp_unix_s'], int(e['item_id']))), 'event order mismatch')
        require(len({e['item_id'] for e in sequence}) == len(sequence), 'repeated user-item pair')
        timestamps[user] = [e['timestamp_unix_s'] for e in sequence]
        n = len(sequence)
        for index, event in enumerate(sequence):
            expected = 'train' if index < n * 8 // 10 else 'valid' if index < n * 9 // 10 else 'test'
            require(event['split'] == expected, 'event violates chronological 80/10/10 boundary')
            require(event['user_event_index'] == index, 'event index mismatch')
            if expected == 'train':
                training_ids.add(event['query_id'])
                item = event['item_id']
                training_counts[item] += 1
                first_train[item] = min(first_train.get(item, event['timestamp_unix_s']), event['timestamp_unix_s'])
    train_rows = list(read_rows(directory / 'train_interactions.jsonl'))
    require(len(train_rows) == len(training_ids), 'training interaction count mismatch')
    require({e['query_id'] for e in train_rows} == training_ids, 'nontraining labels in train_interactions')
    require(all((e == by_id[e['query_id']] for e in train_rows)), 'training interactions changed')
    for item in catalog:
        require(catalog_audit[item]['training_interactions'] == training_counts[item], 'training frequency mismatch')
        require(catalog_audit[item]['first_train_timestamp_unix_s'] == first_train.get(item), 'first training timestamp mismatch')
    seen, counts, excluded_counts = (set(), Counter(), Counter())
    maximum = manifest['protocol']['history']['max_items']
    candidate_count = manifest['protocol']['candidates']['count']
    training_candidate_count = manifest['protocol']['candidates'].get('training_count', candidate_count)
    cohort_candidate_count = manifest['protocol']['candidates'].get('cohort_eligibility_count', candidate_count)
    cold_queries = Counter()
    for split in ('train', 'valid', 'test'):
        for query in read_rows(directory / f'{split}.jsonl'):
            query_id = query['query_id']
            require(query_id in by_id and query_id not in seen, 'missing/duplicate query ID')
            seen.add(query_id)
            event = by_id[query_id]
            require(event['split'] == split, 'query in wrong split')
            require(query['user_id'] == event['user_id'], 'query user changed')
            require(query['target_id'] == event['item_id'], 'target differs from actual interaction')
            require(query['timestamp_unix_s'] == event['timestamp_unix_s'], 'raw query timestamp changed')
            require(query['timestamp'] == event['timestamp_unix_s'] * 1000, 'query milliseconds conversion changed')
            user, timestamp = (event['user_id'], event['timestamp_unix_s'])
            stop = bisect_left(timestamps[user], timestamp)
            past = events[user][:stop]
            history = past[-maximum:] if maximum else past
            require(bool(history), 'query has empty history')
            require(query['history'] == [e['item_id'] for e in history], 'history is not the permitted chronological prefix')
            require(query['history_timestamps'] == [e['timestamp_unix_s'] * 1000 for e in history], 'history timestamps changed')
            candidates = query['candidate_ids']
            expected_candidates = training_candidate_count if split == 'train' else candidate_count
            require(len(candidates) == len(set(candidates)) == expected_candidates, 'candidate count or uniqueness mismatch')
            require(query['target_id'] in candidates, 'positive absent')
            require(set(candidates) <= set(catalog), 'candidate outside closed catalog')
            require(not set(candidates) & {e['item_id'] for e in past}, 'negative or target leaks observed past')
            require(query['history_length_before_truncation'] == len(past), 'pretruncation count mismatch')
            require(query['observed_past_count'] == len(past), 'observed past count mismatch')
            require(query['same_second_predecessors_excluded'] == event['user_event_index'] - stop, 'same-second audit mismatch')
            counts[split] += 1
            cold_queries[split] += event['item_id'] not in first_train
    for excluded in read_rows(directory / 'excluded_queries.jsonl'):
        query_id = excluded['query_id']
        require(query_id in by_id and query_id not in seen, 'excluded event missing/duplicate')
        seen.add(query_id)
        event = by_id[query_id]
        require(all((excluded[key] == value for key, value in event.items())), 'excluded audit changed source interaction')
        stop = bisect_left(timestamps[event['user_id']], event['timestamp_unix_s'])
        reason = 'empty_strictly_earlier_catalog_history' if stop == 0 else 'insufficient_eligible_negatives'
        require(excluded['reason'] == reason, 'unexplained query exclusion')
        expected_pool = len(catalog) - stop - 1
        require(excluded['eligible_negative_count'] == expected_pool, 'excluded negative count mismatch')
        if stop:
            require(expected_pool < cohort_candidate_count - 1, 'eligible query was excluded')
        excluded_counts[excluded['split']] += 1
    require(seen == set(by_id), 'interactions lost between retained/excluded queries')
    require(dict(counts) == manifest['statistics']['query_counts'] == splits['query_counts'], 'manifest query counts mismatch')
    raw_verified = False
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        for name, expected in manifest['raw_file_sha256'].items():
            require(sha256(raw_dir / name) == expected, f'raw hash mismatch: {name}')
        original = {}
        for line in (raw_dir / 'u.data').read_text().splitlines():
            user, item, rating, timestamp = line.split('\t')
            key = (str(int(user)), str(int(item)))
            require(key not in original, 'raw repeated pair')
            original[key] = (int(rating), int(timestamp))
        audited = {}
        for event in list(by_id.values()) + list(read_rows(directory / 'filtered_interactions.jsonl')):
            key = (event['user_id'], event['item_id'])
            require(key not in audited, 'event retained and filtered or duplicated')
            audited[key] = (event['rating'], event['timestamp_unix_s'])
        require(original == audited, 'raw interactions/timestamps not fully accounted for')
        raw_titles = {}
        for line in (raw_dir / 'u.item').read_text(encoding='latin-1').splitlines():
            item, title, *_ = line.split('|')
            raw_titles[str(int(item))] = title
        require(all((meta == {'title': raw_titles[item]} for item, meta in catalog.items())), 'catalog metadata changed or added held-out labels')
        raw_verified = True
    return {'status': 'passed', 'protocol_sha256': fingerprint, 'query_counts': dict(counts), 'excluded_query_counts': dict(excluded_counts), 'cold_target_query_counts': dict(cold_queries), 'preprocessed_events': len(by_id), 'catalog_items': len(catalog), 'raw_timestamps_and_interactions_verified': raw_verified, 'global_temporal_backtest': False, 'verified': ['frozen file hashes', 'chronological per-user 80/10/10 split', 'strictly earlier histories', 'all observed past excluded from negatives', 'unique fixed candidate count', 'target present once', 'no valid/test labels in train_interactions', 'closed catalog including cold items', 'every preprocessed event retained or exclusion-audited']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--raw-dir', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.directory, args.raw_dir), sort_keys=True, indent=2))
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f'audit failed: {exc}', file=sys.stderr)
        return 2
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
