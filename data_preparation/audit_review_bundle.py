import argparse
import ast
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

class AuditError(ValueError):
    pass

def check(condition, message):
    if not condition:
        raise AuditError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            check(bool(line.strip()), f'{path.name}:{number}: blank row')
            yield json.loads(line)

def source_ms(value):
    if len(value) == 10:
        return (round(datetime.strptime(value, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000), 'day')
    try:
        parsed = datetime.strptime(value, '%a %b %d %H:%M:%S %z %Y')
    except ValueError:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    check(parsed.tzinfo is not None and parsed.microsecond % 1000 == 0, 'invalid source date timezone/precision')
    return (round(parsed.timestamp() * 1000), 'millisecond' if parsed.microsecond else 'second')

def audit(directory, reviews=None, metadata=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    split_meta = json.loads((directory / 'splits.json').read_text())
    protocol, stats = (manifest['protocol'], manifest['statistics'])
    candidate_count = protocol['candidates']['count']
    training_candidate_count = protocol['candidates'].get('training_count', candidate_count)
    cohort_candidate_count = protocol['candidates'].get('cohort_eligibility_count', candidate_count)
    fingerprint = digest(directory / 'manifest.json')
    check(split_meta['protocol_sha256'] == fingerprint, 'protocol hash mismatch')
    for name, expected in manifest['output_file_sha256'].items():
        check(digest(directory / name) == expected, f'file hash mismatch: {name}')
    for name, path in (('reviews', reviews), ('metadata', metadata)):
        if path is not None:
            check(digest(path) == manifest['raw_file_sha256'][name], f'raw {name} hash mismatch')
    catalog = json.loads((directory / 'items.json').read_text())
    catalog_set = set(catalog)
    metadata_items_verified = 0
    if metadata is not None:
        matched = set()
        opener = gzip.open if str(metadata).endswith('.gz') else open
        with opener(metadata, 'rt', encoding='utf-8') as stream:
            for text in stream:
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    record = ast.literal_eval(text)
                item = str(record.get('id' if protocol['dataset'] == 'steam' else 'book_id'))
                if item not in catalog_set:
                    continue
                title = record.get('title')
                if protocol['dataset'] == 'steam' and (not isinstance(title, str) or not title.strip()):
                    title = record.get('app_name')
                check(catalog[item] == {'title': title}, 'catalog title differs from original metadata or contains extra labels')
                if protocol['dataset'] == 'goodreads-history':
                    history_shelf = protocol['eligibility'].get('goodreads_history_shelf')
                    if history_shelf is None:
                        rule = protocol['eligibility']['goodreads_genre_rule']
                        history_shelf = ast.literal_eval(rule.split('name ', 1)[1].rsplit(' with count>0', 1)[0])
                    valid_shelf = False
                    for shelf in record.get('popular_shelves', []):
                        if isinstance(shelf, dict) and isinstance(shelf.get('name'), str) and (shelf['name'].strip().casefold() == history_shelf.casefold()):
                            try:
                                valid_shelf |= int(shelf.get('count', 0)) > 0
                            except (ValueError, TypeError):
                                pass
                    check(valid_shelf, 'non-History book entered History catalog')
                    required = protocol['eligibility']['goodreads_required_fields']
                    if 'authors' in required:
                        check(isinstance(record.get('authors'), list) and bool(record['authors']) and all((isinstance(a, dict) and a.get('author_id') is not None and str(a['author_id']).strip() for a in record['authors'])), 'incomplete author metadata')
                    if 'publication_year' in required:
                        check(1 <= int(record.get('publication_year', 0)) <= 9999, 'incomplete publication year')
                    if 'description' in required:
                        check(isinstance(record.get('description'), str) and bool(record['description'].strip()), 'incomplete description')
                matched.add(item)
        check(matched == catalog_set, 'catalog item lacks original metadata')
        metadata_items_verified = len(matched)
    minimum = protocol['eligibility']['minimum_user_interactions']
    maximum = protocol['history']['max_items']
    source_counts = stats['source_and_filter_counts']
    review_seen = bytearray((source_counts['reviews_input'] + 7) // 8)
    source_accounted = 0

    def mark_source(line):
        nonlocal source_accounted
        check(type(line) is int and 1 <= line <= source_counts['reviews_input'], 'source line out of range')
        index, bit = divmod(line - 1, 8)
        mask = 1 << bit
        check(not review_seen[index] & mask, 'source review duplicated across retained/excluded records')
        review_seen[index] |= mask
        source_accounted += 1
    names = ('train', 'valid', 'test', 'excluded_queries', 'train_interactions')
    groups = {name: iter(itertools.groupby(rows(directory / f'{name}.jsonl'), key=lambda q: q['user_id'])) for name in names}
    pending = {name: next(group, None) for name, group in groups.items()}
    query_counts, split_counts, query_exclusions, observed_items = (Counter(), Counter(), Counter(), set())
    training_counts, first_train, precision_counts = (Counter(), {}, Counter())
    previous_user, user_count, event_count = (None, 0, 0)
    with tempfile.TemporaryDirectory(prefix='review-bundle-audit-') as temp:
        db = sqlite3.connect(Path(temp) / 'source.sqlite3')
        try:
            db.execute('CREATE TABLE retained(source_line INTEGER PRIMARY KEY,user_id TEXT,item_id TEXT,timestamp_raw TEXT)')
            for user, group in itertools.groupby(rows(directory / 'interactions.jsonl'), key=lambda e: e['user_id']):
                check(previous_user is None or user > previous_user, 'users are not unique/in lexical order')
                previous_user, user_count = (user, user_count + 1)
                events = list(group)
                check(len(events) >= minimum, 'user below configured minimum')
                check(events == sorted(events, key=lambda e: (e['timestamp'], e['item_id'])), 'events not chronological')
                check(len({e['item_id'] for e in events}) == len(events), 'duplicate user-item event')
                times = [e['timestamp'] for e in events]
                by_id = {e['query_id']: e for e in events}
                check(len(by_id) == len(events), 'duplicate event query ID')
                n, seen = (len(events), set())
                expected_boundaries = {'train_end_exclusive': n * 8 // 10, 'valid_end_exclusive': n * 9 // 10, 'event_count': n}
                check(split_meta['split_endpoints'][user] == expected_boundaries, 'split boundary metadata mismatch')
                for index, event in enumerate(events):
                    split = 'train' if index < n * 8 // 10 else 'valid' if index < n * 9 // 10 else 'test'
                    check(event['split'] == split and event['user_event_index'] == index, 'event split/index mismatch')
                    check((event['timestamp'], event['timestamp_precision']) == source_ms(event['timestamp_raw']), 'source timestamp changed')
                    check(event['timestamp_field'] == protocol['source_fields']['date'], 'timestamp source field mismatch')
                    check(event['item_id'] in catalog_set, 'event outside catalog')
                    mark_source(event['source_line'])
                    observed_items.add(event['item_id'])
                    split_counts[split] += 1
                    precision_counts[event['timestamp_precision']] += 1
                    if split == 'train':
                        item = event['item_id']
                        training_counts[item] += 1
                        first_train[item] = min(first_train.get(item, event['timestamp']), event['timestamp'])
                if reviews is not None:
                    db.executemany('INSERT INTO retained VALUES(?,?,?,?)', [(e['source_line'], user, e['item_id'], e['timestamp_raw']) for e in events])
                for name in names:
                    matching = pending[name] is not None and pending[name][0] == user
                    selected = list(pending[name][1]) if matching else []
                    if matching:
                        pending[name] = next(groups[name], None)
                    check(pending[name] is None or pending[name][0] > user, f'out of order/unrecognized user in {name}')
                    if name == 'train_interactions':
                        check(selected == [e for e in events if e['split'] == 'train'], 'nontraining/altered events in training data')
                        continue
                    for query in selected:
                        qid = query['query_id']
                        check(qid in by_id and qid not in seen, 'unknown/duplicate query')
                        seen.add(qid)
                        event = by_id[qid]
                        stop = bisect_left(times, event['timestamp'])
                        past = events[:stop]
                        if name == 'excluded_queries':
                            check(all((query[key] == value for key, value in event.items())), 'excluded event modified')
                            reason = 'empty_strictly_earlier_history' if not past else 'insufficient_eligible_negatives'
                            pool = len(catalog) - len(past) - 1
                            check(query['reason'] == reason and query['eligible_negative_count'] == pool, 'incorrect exclusion reason')
                            check(not past or pool < cohort_candidate_count - 1, 'eligible query excluded')
                            query_exclusions[event['split'] + ':' + reason] += 1
                            continue
                        check(event['split'] == name and event['item_id'] == query['target_id'], 'wrong split/target')
                        for key in ('timestamp', 'timestamp_raw', 'timestamp_precision', 'timestamp_field', 'source_line', 'user_event_index'):
                            check(query[key] == event[key], f'query source field changed: {key}')
                        history = past[-maximum:] if maximum else past
                        check(bool(history), 'empty query history')
                        check(query['history'] == [e['item_id'] for e in history], 'history is not strictly earlier observed prefix')
                        check(query['history_timestamps'] == [e['timestamp'] for e in history], 'history timestamps changed')
                        candidates = query['candidate_ids']
                        expected_candidates = training_candidate_count if name == 'train' else candidate_count
                        check(len(candidates) == len(set(candidates)) == expected_candidates, 'candidate count or uniqueness mismatch')
                        check(set(candidates) <= catalog_set and query['target_id'] in candidates, 'unknown candidate/missing target')
                        check(not set(candidates) & {e['item_id'] for e in past}, 'observed-past item sampled as a negative')
                        check(query['observed_past_count'] == query['history_length_before_truncation'] == len(past), 'past count mismatch')
                        check(query['same_time_predecessors_excluded'] == event['user_event_index'] - len(past), 'same-time count mismatch')
                        query_counts[name] += 1
                check(seen == set(by_id), 'event lost between queries/exclusions')
                event_count += len(events)
            check(all((value is None for value in pending.values())), 'queries contain unrecognized users')
            for excluded in rows(directory / 'source_exclusions.jsonl.gz'):
                if excluded['source'] == 'reviews':
                    mark_source(excluded['source_line'])
            check(source_accounted == source_counts['reviews_input'], 'raw source reviews not fully accounted')
            check(observed_items == catalog_set, 'closed catalog differs from retained events')
            check(dict(query_counts) == stats['query_counts'] == split_meta['query_counts'], 'query counts disagree')
            check(dict(split_counts) == stats['split_interactions'], 'split event counts disagree')
            check(dict(query_exclusions) == stats['excluded_queries'], 'query exclusion counts disagree')
            check(dict(precision_counts) == stats['timestamp_precision_counts'], 'timestamp precision counts disagree')
            check(user_count == source_counts['users_selected'] and event_count == source_counts['reviews_preprocessed'], 'preprocessed counts disagree')
            catalog_audit = json.loads((directory / 'catalog.json').read_text())
            check(set(catalog_audit) == catalog_set, 'catalog audit item set differs')
            for item in catalog:
                check(catalog_audit[item] == {'training_interactions': training_counts[item], 'first_train_timestamp': first_train.get(item)}, 'catalog training statistics mismatch')
            verified_raw_records = 0
            if reviews is not None:
                expected = iter(db.execute('SELECT source_line,user_id,item_id,timestamp_raw FROM retained ORDER BY source_line'))
                wanted = next(expected, None)
                opener = gzip.open if str(reviews).endswith('.gz') else open
                with opener(reviews, 'rt', encoding='utf-8') as stream:
                    for line_number, text in enumerate(stream, 1):
                        if wanted is None or line_number != wanted[0]:
                            continue
                        try:
                            source = json.loads(text)
                        except json.JSONDecodeError:
                            source = ast.literal_eval(text)
                        user_field = protocol['source_fields']['user']
                        raw_user = source.get(user_field)
                        if protocol['dataset'] == 'steam':
                            if raw_user is None or not str(raw_user).strip():
                                user_field, raw_user = ('username', source.get('username'))
                            raw_user = user_field + ':' + str(raw_user)
                        check(str(raw_user) == wanted[1], 'user differs from raw source')
                        check(str(source[protocol['source_fields']['item']]) == wanted[2], 'item differs from raw source')
                        check(source[protocol['source_fields']['date']] == wanted[3], 'timestamp string differs from raw source')
                        verified_raw_records += 1
                        wanted = next(expected, None)
                check(wanted is None and verified_raw_records == event_count, 'not all retained rows verified against source')
        finally:
            db.close()
    return {'status': 'passed', 'protocol_sha256': fingerprint, 'users': user_count, 'catalog_items': len(catalog), 'preprocessed_events': event_count, 'query_counts': dict(query_counts), 'source_reviews_accounted': source_accounted, 'raw_retained_records_verified': verified_raw_records, 'raw_metadata_hash_verified': metadata is not None, 'raw_metadata_items_verified': metadata_items_verified, 'global_temporal_backtest': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(**vars(args)), indent=2, sort_keys=True))
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f'audit failed: {exc}', file=sys.stderr)
        return 2
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
