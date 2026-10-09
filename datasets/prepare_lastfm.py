from __future__ import annotations
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
VERSION = 'lastfm-1k-celma-artist-first-listen-v1'
DATASET = 'LastFM-1K-Celma'
SOURCE_PATH = Path(__file__).with_name('lastfm-source.json')
MBID = re.compile('[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')

class LastFMError(ValueError):
    pass

def encoded(value, pretty=False):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2 if pretty else None, separators=None if pretty else (',', ':')) + '\n').encode()

def hash_file(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()

def parse(line):
    fields = line.rstrip('\r\n').split('\t')
    if len(fields) != 6:
        raise LastFMError('malformed_six_field_tsv')
    user, raw_time, artist, name, _, _ = fields
    if not user.strip():
        raise LastFMError('missing_user_id')
    if not MBID.fullmatch(artist):
        raise LastFMError('missing_or_invalid_artist_mbid')
    if not name.strip():
        raise LastFMError('missing_artist_name')
    try:
        stamp = datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise LastFMError('missing_or_invalid_utc_timestamp') from exc
    return (user, 'mbid:' + artist.lower(), name.strip(), int(stamp.timestamp()) * 1000, raw_time)

def candidates(catalog, seen, target, seed, query, count=100):
    if len(catalog) - len(seen) - 1 < count - 1:
        raise LastFMError('insufficient_negative_pool')

    def draws(namespace):
        key = hashlib.sha256(encoded([VERSION, seed, query, namespace])).digest()
        for index in itertools.count():
            yield int.from_bytes(hashlib.sha256(key + index.to_bytes(8, 'big')).digest(), 'big')

    def below(values, n):
        cutoff = (1 << 256) - (1 << 256) % n
        while True:
            value = next(values)
            if value < cutoff:
                return value % n
    result, selected, values = ([], set(), draws('negatives'))
    while len(result) < count - 1:
        item = catalog[below(values, len(catalog))]
        if item != target and item not in seen and (item not in selected):
            selected.add(item)
            result.append(item)
    result.append(target)
    values = draws('order')
    for i in range(len(result) - 1, 0, -1):
        j = below(values, i + 1)
        result[i], result[j] = (result[j], result[i])
    return result

def prepare(events, output, source_receipt, *, minimum_artists=10, max_history=None, seed=20260930, fixture=False):
    events, output = (Path(events), Path(output))
    if minimum_artists < 1 or (max_history is not None and max_history < 1):
        raise LastFMError('history/user limits must be positive')
    source = json.loads(SOURCE_PATH.read_text())
    raw_sha, raw_md5 = (hash_file(events), hash_file(events, 'md5'))
    if not fixture and (raw_md5 != source['events_md5'] or source_receipt.get('archive_md5') != source['archive_md5'] or source_receipt.get('events_sha256') != raw_sha or (source_receipt.get('status') != 'verified')):
        raise LastFMError('official source archive/event checksum receipt is required')
    if output.exists():
        raise LastFMError('output must be new; previous stages are never overwritten')
    output.mkdir(parents=True)
    database = output / '.staging.sqlite3'
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE raw_events (source_line INTEGER PRIMARY KEY,user_id TEXT,item_id TEXT,timestamp INTEGER,timestamp_raw TEXT)')
    stats, artists, raw_users = (Counter(), {}, set())
    try:
        with ExitStack() as stack:

            def writer(name, compressed=False):
                raw = stack.enter_context((output / name).open('xb'))
                return stack.enter_context(gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='')) if compressed else raw
            exclusions = writer('source_exclusions.jsonl.gz', True)
            pending = []
            with events.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    stats['source_lines'] += 1
                    raw_users.add(line.split('\t', 1)[0])
                    try:
                        user, item, title, stamp, raw_time = parse(line)
                    except LastFMError as exc:
                        reason = str(exc)
                        stats['source_excluded:' + reason] += 1
                        exclusions.write(encoded({'source_line': line_number, 'reason': reason}))
                        continue
                    if item in artists and artists[item] != title:
                        stats['artist_name_variant_events'] += 1
                    artists[item] = min(artists.get(item, title), title)
                    pending.append((line_number, user, item, stamp, raw_time))
                    if len(pending) == 50000:
                        db.executemany('INSERT INTO raw_events VALUES (?,?,?,?,?)', pending)
                        db.commit()
                        pending.clear()
                    if line_number % 1000000 == 0:
                        print(json.dumps({'event': 'source_ingestion', 'lines': line_number}), file=sys.stderr, flush=True)
            db.executemany('INSERT INTO raw_events VALUES (?,?,?,?,?)', pending)
            db.commit()
            if not fixture and (stats['source_lines'] != source['published_source_lines'] or len(raw_users) != source['published_source_users']):
                raise LastFMError('source statistics differ from the primary source')
            db.execute('CREATE INDEX user_artist_time ON raw_events(user_id,item_id,timestamp,source_line)')
            db.execute('CREATE TABLE ranked AS SELECT *,ROW_NUMBER() OVER(PARTITION BY user_id,item_id ORDER BY timestamp,source_line) AS occurrence FROM raw_events')
            for event in db.execute('SELECT source_line FROM ranked WHERE occurrence>1 ORDER BY source_line'):
                stats['source_excluded:later_or_duplicate_user_artist_listen'] += 1
                exclusions.write(encoded({'source_line': event[0], 'reason': 'later_or_duplicate_user_artist_listen'}))
            users = [row[0] for row in db.execute('SELECT user_id FROM ranked WHERE occurrence=1 GROUP BY user_id HAVING COUNT(*)>=? ORDER BY user_id', (minimum_artists,))]
            db.execute('CREATE TABLE users (user_id TEXT PRIMARY KEY)')
            db.executemany('INSERT INTO users VALUES (?)', ((user,) for user in users))
            for event in db.execute('SELECT source_line FROM ranked WHERE occurrence=1 AND user_id NOT IN (SELECT user_id FROM users) ORDER BY source_line'):
                stats['source_excluded:user_below_distinct_artist_minimum'] += 1
                exclusions.write(encoded({'source_line': event[0], 'reason': 'user_below_distinct_artist_minimum'}))
            db.execute('CREATE TABLE events AS SELECT source_line,user_id,item_id,timestamp,timestamp_raw FROM ranked WHERE occurrence=1 AND user_id IN (SELECT user_id FROM users)')
            db.execute('CREATE INDEX events_user_time ON events(user_id,timestamp,item_id)')
            catalog = [row[0] for row in db.execute('SELECT DISTINCT item_id FROM events ORDER BY item_id')]
            if len(catalog) < 101:
                raise LastFMError('fewer than101 retained artists; cannot freeze100-candidate parent')
            items = {item: {'title': artists[item], 'artist_mbid': item[5:]} for item in catalog}
            (output / 'items.json').write_bytes(encoded(items, True))
            streams = {name: writer(name) for name in ('interactions.jsonl', 'train_interactions.jsonl', 'excluded_queries.jsonl', 'train.jsonl', 'valid.jsonl', 'test.jsonl')}
            query_counts, split_counts, first_train, train_counts, boundaries = (Counter(), Counter(), {}, Counter(), {})
            for user, group in itertools.groupby(db.execute('SELECT * FROM events ORDER BY user_id,timestamp,item_id'), key=lambda row: row['user_id']):
                sequence = list(group)
                n, cursor, past, seen = (len(sequence), 0, [], set())
                train_end, valid_end = (8 * n // 10, 9 * n // 10)
                boundaries[user] = {'train_end_exclusive': train_end, 'valid_end_exclusive': valid_end, 'event_count': n}
                for index, event in enumerate(sequence):
                    while cursor < index and sequence[cursor]['timestamp'] < event['timestamp']:
                        past.append(sequence[cursor])
                        seen.add(sequence[cursor]['item_id'])
                        cursor += 1
                    split = 'train' if index < train_end else 'valid' if index < valid_end else 'test'
                    query = f"LastFM-1K-Celma:first-artist-line:{event['source_line']}"
                    record = {'query_id': query, 'user_id': user, 'item_id': event['item_id'], 'timestamp': event['timestamp'], 'timestamp_raw': event['timestamp_raw'], 'timestamp_field': 'scrobble_utc', 'timestamp_precision': 'second', 'source_line': event['source_line'], 'split': split, 'user_event_index': index}
                    streams['interactions.jsonl'].write(encoded(record))
                    split_counts[split] += 1
                    if split == 'train':
                        streams['train_interactions.jsonl'].write(encoded(record))
                        train_counts[event['item_id']] += 1
                        first_train[event['item_id']] = min(first_train.get(event['item_id'], event['timestamp']), event['timestamp'])
                    pool = len(catalog) - len(seen) - 1
                    reason = 'empty_strictly_earlier_history' if not past else 'insufficient_eligible_negatives' if pool < 99 else None
                    if reason:
                        streams['excluded_queries.jsonl'].write(encoded({**record, 'reason': reason, 'eligible_negative_count': pool}))
                        continue
                    history = past[-max_history:] if max_history else past
                    row = {'query_id': query, 'user_id': user, 'target_id': event['item_id'], 'history': [row['item_id'] for row in history], 'history_timestamps': [row['timestamp'] for row in history], 'candidate_ids': candidates(catalog, seen, event['item_id'], seed, query), 'timestamp': event['timestamp'], 'timestamp_raw': event['timestamp_raw'], 'timestamp_field': 'scrobble_utc', 'timestamp_precision': 'second', 'source_line': event['source_line'], 'user_event_index': index, 'history_length_before_truncation': len(past), 'observed_past_count': len(past), 'same_time_predecessors_excluded': index - len(past)}
                    streams[split + '.jsonl'].write(encoded(row))
                    query_counts[split] += 1
            if any((query_counts[split] == 0 for split in ('train', 'valid', 'test'))):
                raise LastFMError('an evaluation/training split has no eligible queries')
            (output / 'catalog.json').write_bytes(encoded({item: {'training_interactions': train_counts[item], 'first_train_timestamp': first_train.get(item)} for item in catalog}, True))
            stats.update(users=len(users), catalog_items=len(catalog), retained_first_artist_events=sum(split_counts.values()), source_users=len(raw_users))
        db.close()
        db = None
        database.unlink()
        protocol = {'version': VERSION, 'dataset': DATASET, 'status': 'distinct_timestamped_benchmark_not_original_paper_cohort', 'synthetic_fixture': fixture, 'published_source_users': source['published_source_users'], 'manuscript_users': 1220, 'source_fields': source['source_fields'], 'eligibility': {'minimum_distinct_artists_per_user': minimum_artists, 'artist_metadata': 'valid MusicBrainz artist UUID and nonempty source artist name; no inferred IDs or metadata', 'artist_title': 'lexicographically smallest stripped observed source name per MBID; MBID identity retained', 'repeat_listens': 'earliest valid user-artist event, ties resolved by source line; explicit novel-artist adaptation', 'session_interpretation': 'minimum10 distinct artist first listens by default; source has scrobbles, no explicit listening-session IDs', 'user_sampling': 'none; all eligible source users', 'item_frequency_filter': 'none'}, 'split': 'per-user chronological floor(.8*n),floor(.9*n),n; equal times ordered by canonical MBID', 'history': {'max_items': max_history, 'visibility': 'strictly earlier source timestamps only; same-time block excluded', 'evaluation': 'rolling observed history including earlier validation/test events'}, 'timestamp': {'unit': 'milliseconds', 'precision': 'second', 'source': 'recorded UTC scrobble time; never imputed or replaced by tagging/signup time'}, 'catalog': 'all retained source artists; closed catalog; artist names only; no demographic or review inputs', 'candidates': {'count': 100, 'seed': seed, 'sampling': 'SHA256 rejection draws over catalog without replacement; separately shuffled order', 'exclude': 'all strictly earlier artists and target, before history truncation', 'future_positives': 'not consulted'}, 'metrics': {'averaging': 'query_macro', 'tie_break': 'score descending then string item ID ascending'}, 'source_urls': [source['record_url'], source['required_attribution_url']], 'license': source['license']}
        files = sorted((path.name for path in output.iterdir() if path.is_file()))
        manifest = {'schema_version': 1, 'protocol': protocol, 'source': source, 'raw_file_sha256': {events.name: raw_sha}, 'raw_file_md5': {events.name: raw_md5}, 'source_archive_sha256': source_receipt.get('archive_sha256'), 'preparation_source_sha256': {'prepare_lastfm.py': hash_file(__file__), 'lastfm-source.json': hash_file(SOURCE_PATH)}, 'statistics': {**dict(stats), 'query_counts': dict(query_counts), 'split_interactions': dict(split_counts)}, 'output_file_sha256': {name: hash_file(output / name) for name in files}}
        (output / 'manifest.json').write_bytes(encoded(manifest, True))
        fingerprint = hash_file(output / 'manifest.json')
        (output / 'splits.json').write_bytes(encoded({'dataset': DATASET, 'protocol_sha256': fingerprint, 'manifest': 'manifest.json', 'query_counts': dict(query_counts), 'timestamp_unit': 'milliseconds', 'history_max_items': max_history, 'split_endpoints': boundaries}, True))
        return {'status': 'prepared_pending_independent_audit', 'dataset': DATASET, 'protocol_sha256': fingerprint, **manifest['statistics']}
    finally:
        if db is not None:
            db.close()
