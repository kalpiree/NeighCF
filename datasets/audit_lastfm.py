from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

class AuditError(ValueError):
    pass

def require(value, message):
    if not value:
        raise AuditError(message)

def hashed(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)

def put(bits, number):
    bits[(number - 1) // 8] |= 1 << (number - 1) % 8

def has(bits, number):
    return bool(bits[(number - 1) // 8] & 1 << (number - 1) % 8)

def independent_source(line):
    values = line.rstrip('\r\n').split('\t')
    if len(values) != 6:
        return ('malformed_six_field_tsv', None)
    user, raw, artist, name = values[:4]
    if not user.strip():
        return ('missing_user_id', None)
    if re.fullmatch('[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}', artist) is None:
        return ('missing_or_invalid_artist_mbid', None)
    if not name.strip():
        return ('missing_artist_name', None)
    try:
        parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.strftime('%Y-%m-%dT%H:%M:%SZ') != raw:
            raise ValueError()
        stamp = int((parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()) * 1000
    except ValueError:
        return ('missing_or_invalid_utc_timestamp', None)
    return (None, (user, 'mbid:' + artist.lower(), name.strip(), stamp, raw))

def sample_from_spec(catalog, observed, target, seed, query):
    generators = {}
    counters = {}
    for purpose in ('negatives', 'order'):
        payload = json.dumps(['lastfm-1k-celma-artist-first-listen-v1', seed, query, purpose], sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n'
        generators[purpose] = hashlib.sha256(payload.encode()).digest()
        counters[purpose] = 0

    def bounded(purpose, limit):
        while True:
            counter = counters[purpose]
            counters[purpose] += 1
            value = int.from_bytes(hashlib.sha256(generators[purpose] + counter.to_bytes(8, 'big')).digest(), 'big')
            if value < (1 << 256) - (1 << 256) % limit:
                return value % limit
    chosen = []
    used = observed | {target}
    while len(chosen) < 99:
        value = catalog[bounded('negatives', len(catalog))]
        if value not in used:
            chosen.append(value)
            used.add(value)
    chosen.append(target)
    for i in range(99, 0, -1):
        j = bounded('order', i + 1)
        chosen[i], chosen[j] = (chosen[j], chosen[i])
    return chosen

def audit(directory, source_file):
    directory, source_file = (Path(directory), Path(source_file))
    manifest = json.loads((directory / 'manifest.json').read_text())
    split_meta = json.loads((directory / 'splits.json').read_text())
    fingerprint = hashed(directory / 'manifest.json')
    require(split_meta['protocol_sha256'] == fingerprint, 'protocol fingerprint mismatch')
    cfg = manifest['protocol']
    require(cfg['dataset'] == 'LastFM-1K-Celma' and cfg['candidates']['count'] == 100, 'wrong dataset/parent protocol')
    for name, expected in manifest['output_file_sha256'].items():
        require(Path(name).name == name and hashed(directory / name) == expected, 'frozen file checksum differs: ' + name)
    require(manifest['raw_file_sha256'] == {source_file.name: hashed(source_file)}, 'raw SHA256 differs')
    require(manifest['raw_file_md5'] == {source_file.name: hashed(source_file, 'md5')}, 'raw MD5 differs')
    source = manifest['source']
    if not cfg['synthetic_fixture']:
        require(manifest['raw_file_md5'][source_file.name] == source['events_md5'], 'official event MD5 differs')
    size = manifest['statistics']['source_lines']
    require(type(size) is int and 0 < size <= source['published_source_lines'], 'invalid source line count')
    mask_size = (size + 7) // 8
    valid_bits = bytearray(mask_size)
    expected_exclusions = defaultdict(lambda: bytearray(mask_size))
    first, names, raw_users = ({}, {}, set())
    observed_lines = 0
    with source_file.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            require(number <= size, 'source has unexpected additional lines')
            observed_lines = number
            raw_users.add(line.split('\t', 1)[0])
            reason, parsed = independent_source(line)
            if reason:
                put(expected_exclusions[reason], number)
                continue
            put(valid_bits, number)
            user, item, name, stamp, raw = parsed
            names[item] = min(names.get(item, name), name)
            pair, value = ((user, item), (stamp, number, raw))
            if pair not in first or value[:2] < first[pair][:2]:
                first[pair] = value
    require(observed_lines == size, 'source line count differs')
    if not cfg['synthetic_fixture']:
        require(size == source['published_source_lines'] and len(raw_users) == source['published_source_users'], 'official source totals differ')
    counts = Counter((user for user, item in first))
    eligible = {user for user, count in counts.items() if count >= cfg['eligibility']['minimum_distinct_artists_per_user']}
    first_bits, retained_bits = (bytearray(mask_size), bytearray(mask_size))
    expected = {}
    for pair, value in first.items():
        put(first_bits, value[1])
        if pair[0] in eligible:
            expected[pair] = value
            put(retained_bits, value[1])
        else:
            put(expected_exclusions['user_below_distinct_artist_minimum'], value[1])
    duplicate = bytearray((a & ~b for a, b in zip(valid_bits, first_bits)))
    if any(duplicate):
        expected_exclusions['later_or_duplicate_user_artist_listen'] = duplicate
    actual_exclusions = defaultdict(lambda: bytearray(mask_size))
    covered = bytearray(retained_bits)
    exclusions_count = 0
    for row in rows(directory / 'source_exclusions.jsonl.gz'):
        line = row['source_line']
        require(type(line) is int and 1 <= line <= size and (not has(covered, line)), 'raw source duplicated/unaccountable exclusion')
        put(covered, line)
        put(actual_exclusions[row['reason']], line)
        exclusions_count += 1
    expected_exclusions = {k: v for k, v in expected_exclusions.items() if any(v)}
    require(dict(actual_exclusions) == expected_exclusions, 'raw exclusion classification differs from independent source pass')
    require(exclusions_count + len(expected) == size, 'not every raw line is accounted for exactly once')
    grouped = defaultdict(list)
    actual_pairs = set()
    for event in rows(directory / 'interactions.jsonl'):
        pair = (event['user_id'], event['item_id'])
        require(pair in expected and pair not in actual_pairs, 'retained user-artist pair differs or repeats')
        actual_pairs.add(pair)
        require((event['timestamp'], event['source_line'], event['timestamp_raw']) == expected[pair], 'event is not the earliest real source listen')
        require(event['query_id'] == f"LastFM-1K-Celma:first-artist-line:{event['source_line']}", 'query provenance differs')
        require(event['timestamp_field'] == 'scrobble_utc' and event['timestamp_precision'] == 'second', 'source clock mislabeled')
        grouped[pair[0]].append(event)
    require(actual_pairs == set(expected), 'retained cohort does not match all eligible source pairs')
    items = json.loads((directory / 'items.json').read_text())
    item_set = set(items)
    sorted_items = sorted(items)
    expected_items = {item for user, item in expected}
    require(set(items) == expected_items, 'catalog differs from retained source artists')
    for item, meta in items.items():
        require(meta == {'title': names[item], 'artist_mbid': item[5:]}, 'artist ID/name is not the recorded source metadata')
    streams = {split: iter(rows(directory / (split + '.jsonl'))) for split in ('train', 'valid', 'test')}
    excluded_stream = iter(rows(directory / 'excluded_queries.jsonl'))
    training_stream = iter(rows(directory / 'train_interactions.jsonl'))
    query_counts, training_counts, first_train = (Counter(), Counter(), {})
    for user in sorted(grouped):
        sequence = grouped[user]
        require(sequence == sorted(sequence, key=lambda e: (e['timestamp'], e['item_id'])), 'interaction chronology differs')
        n = len(sequence)
        require(split_meta['split_endpoints'][user] == {'train_end_exclusive': n * 8 // 10, 'valid_end_exclusive': n * 9 // 10, 'event_count': n}, 'split endpoint differs')
        for i, event in enumerate(sequence):
            split = 'train' if i < n * 8 // 10 else 'valid' if i < n * 9 // 10 else 'test'
            require(event['split'] == split and event['user_event_index'] == i, 'per-user chronological split differs')
            past = [previous for previous in sequence[:i] if previous['timestamp'] < event['timestamp']]
            seen = {previous['item_id'] for previous in past}
            if split == 'train':
                require(next(training_stream, None) == event, 'training interactions include missing/held-out event')
                training_counts[event['item_id']] += 1
                first_train[event['item_id']] = min(first_train.get(event['item_id'], event['timestamp']), event['timestamp'])
            pool = len(items) - len(seen) - 1
            reason = 'empty_strictly_earlier_history' if not past else 'insufficient_eligible_negatives' if pool < 99 else None
            if reason:
                require(next(excluded_stream, None) == {**event, 'reason': reason, 'eligible_negative_count': pool}, 'query exclusion differs')
                continue
            row = next(streams[split], None)
            require(row is not None, 'missing canonical query')
            for name in ('query_id', 'user_id', 'timestamp', 'timestamp_raw', 'timestamp_field', 'timestamp_precision', 'source_line', 'user_event_index'):
                require(row[name] == event[name], 'query identity/source clock differs')
            history = past[-cfg['history']['max_items']:] if cfg['history']['max_items'] else past
            require(row['history'] == [e['item_id'] for e in history] and row['history_timestamps'] == [e['timestamp'] for e in history], 'history has future/equal-time events or lost source identity')
            require(row['target_id'] == event['item_id'] and row['history_length_before_truncation'] == len(past), 'target/history count differs')
            require(row['observed_past_count'] == len(past) and row['same_time_predecessors_excluded'] == i - len(past), 'history visibility counters differ')
            candidates = row['candidate_ids']
            require(len(candidates) == 100 and len(set(candidates)) == 100 and (candidates.count(row['target_id']) == 1), 'candidate cardinality/positive differs')
            require(set(candidates) <= item_set and (not set(candidates) & seen), 'candidate leaks observed artist or unknown item')
            require(candidates == sample_from_spec(sorted_items, seen, row['target_id'], cfg['candidates']['seed'], row['query_id']), 'candidate order/sample differs from frozen random specification')
            query_counts[split] += 1
    require(all((next(stream, None) is None for stream in streams.values())) and next(excluded_stream, None) is None and (next(training_stream, None) is None), 'unexpected extra query/training rows')
    require(dict(query_counts) == split_meta['query_counts'] == manifest['statistics']['query_counts'], 'query totals differ')
    require(set(split_meta['split_endpoints']) == eligible, 'split user cohort differs')
    catalog_stats = json.loads((directory / 'catalog.json').read_text())
    require(catalog_stats == {item: {'training_interactions': training_counts[item], 'first_train_timestamp': first_train.get(item)} for item in items}, 'catalog statistics include held-out interactions')
    return {'status': 'passed', 'dataset': 'LastFM-1K-Celma', 'protocol_sha256': fingerprint, 'source_lines_accounted': size, 'retained_first_artist_events': len(expected), 'users': len(eligible), 'catalog_items': len(items), 'query_counts': dict(query_counts), 'real_source_timestamps_verified': True, 'full_raw_exclusion_accounting_verified': True, 'original_manuscript_cohort': False, 'synthetic_fixture': cfg['synthetic_fixture']}
