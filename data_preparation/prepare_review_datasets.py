from __future__ import annotations
import argparse
import ast
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
import shutil
import sqlite3
import sys
VERSION = 'review-datasets-published-protocol-reconstruction-v1'
SPLITS = ('train', 'valid', 'test')

class DataError(ValueError):
    pass

def encoded(value, pretty=False):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2 if pretty else None, separators=None if pretty else (',', ':')) + '\n').encode()

def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()

def records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise DataError(f'{path}:{line_number}: empty record')
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    raise ValueError('record is not an object')
            except (ValueError, SyntaxError) as exc:
                raise DataError(f'{path}:{line_number}: invalid record: {exc}') from exc
            yield (line_number, record)

def identifier(value):
    if isinstance(value, str) and value.strip():
        return value
    if type(value) is int and value > 0:
        return str(value)
    return None

def parse_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        raise DataError('missing timestamp')
    value = value.strip()
    if re.fullmatch('\\d{4}-\\d{2}-\\d{2}', value):
        parsed = datetime.strptime(value, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        precision = 'day'
    else:
        try:
            parsed = datetime.strptime(value, '%a %b %d %H:%M:%S %z %Y')
        except ValueError:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise DataError('timestamp has no timezone')
        precision = 'second'
        if parsed.microsecond:
            if parsed.microsecond % 1000:
                raise DataError('submillisecond timestamp cannot be represented losslessly')
            precision = 'millisecond'
    return (round(parsed.timestamp() * 1000), precision)

def metadata_record(record, dataset, history_shelf, book_required):
    item = identifier(record.get('id' if dataset == 'steam' else 'book_id'))
    if item is None:
        return (None, None, 'missing_item_id')
    title = record.get('title')
    if dataset == 'steam' and (not isinstance(title, str) or not title.strip()):
        title = record.get('app_name')
    if not isinstance(title, str) or not title.strip():
        return (item, None, 'missing_title')
    if dataset == 'goodreads-history':
        shelves = record.get('popular_shelves')
        is_history = False
        for shelf in shelves if isinstance(shelves, list) else []:
            if not isinstance(shelf, dict) or not isinstance(shelf.get('name'), str) or shelf['name'].strip().casefold() != history_shelf.casefold():
                continue
            try:
                is_history |= int(shelf.get('count', 0)) > 0
            except (ValueError, TypeError):
                pass
        if not is_history:
            return (item, None, 'not_exact_history_shelf')
        if 'authors' in book_required:
            authors = record.get('authors')
            if not isinstance(authors, list) or not authors or any((not isinstance(a, dict) or not identifier(a.get('author_id')) for a in authors)):
                return (item, None, 'incomplete_authors')
        if 'publication_year' in book_required:
            try:
                year = int(record.get('publication_year', ''))
                if not 1 <= year <= 9999:
                    raise ValueError()
            except (ValueError, TypeError):
                return (item, None, 'incomplete_publication_year')
        if 'description' in book_required and (not isinstance(record.get('description'), str) or not record['description'].strip()):
            return (item, None, 'incomplete_description')
    return (item, title, None)

def query_candidates(catalog_ids, observed, target, seed, query_id, count=100):
    if len(catalog_ids) - len(observed) - (target not in observed) < count - 1:
        raise DataError('insufficient negative pool')

    def generator(namespace):
        key = hashlib.sha256(encoded([VERSION, seed, query_id, namespace])).digest()
        index = 0
        while True:
            yield int.from_bytes(hashlib.sha256(key + index.to_bytes(8, 'big')).digest(), 'big')
            index += 1

    def below(draws, bound):
        limit = (1 << 256) - (1 << 256) % bound
        while True:
            value = next(draws)
            if value < limit:
                return value % bound
    draws, selected = (generator('negatives'), set())
    candidates = []
    while len(candidates) < count - 1:
        item = catalog_ids[below(draws, len(catalog_ids))]
        if item != target and item not in observed and (item not in selected):
            selected.add(item)
            candidates.append(item)
    candidates.append(target)
    draws = generator('order')
    for i in range(len(candidates) - 1, 0, -1):
        j = below(draws, i + 1)
        candidates[i], candidates[j] = (candidates[j], candidates[i])
    return candidates

def prepare(dataset, reviews, metadata, output, seed=20260930, max_history=None, max_users=None, user_sample_seed=20260930, history_shelf='history', book_required='title,authors', date_field=None, user_field=None, item_field=None, text_field=None, steam_user_fallback='username', minimum_user_interactions=None, require_review_text=False):
    if dataset not in {'steam', 'goodreads-history'}:
        raise DataError('dataset must be steam or goodreads-history')
    for name, value in (('max_history', max_history), ('max_users', max_users), ('minimum_user_interactions', minimum_user_interactions)):
        if value is not None and (type(value) is not int or value < 1):
            raise DataError(f'{name} must be a positive integer')
    if type(seed) is not int or type(user_sample_seed) is not int:
        raise DataError('seeds must be integers')
    if not isinstance(history_shelf, str) or not history_shelf.strip():
        raise DataError('history_shelf must be nonempty')
    required = tuple(sorted(set(book_required.split(','))))
    if not set(required) <= {'title', 'authors', 'publication_year', 'description'} or 'title' not in required:
        raise DataError('book_required must include title and use title,authors,publication_year,description only')
    minimum = minimum_user_interactions or (20 if dataset == 'steam' else 40)
    user_field = user_field or ('username' if dataset == 'steam' else 'user_id')
    item_field = item_field or ('product_id' if dataset == 'steam' else 'book_id')
    date_field = date_field or ('date' if dataset == 'steam' else 'date_added')
    text_field = text_field or ('text' if dataset == 'steam' else 'review_text')
    if steam_user_fallback not in {'username', 'none'}:
        raise DataError('steam_user_fallback must be username or none')
    output, reviews, metadata = (Path(output), Path(reviews), Path(metadata))
    if os.path.lexists(output):
        raise DataError(f'refusing to overwrite output {output}')
    source_hashes = {'reviews': file_hash(reviews), 'metadata': file_hash(metadata)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    db = None
    try:
        with ExitStack() as stack:

            def writer(name, compressed=False):
                raw = stack.enter_context((output / name).open('xb'))
                return stack.enter_context(gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='')) if compressed else raw
            exclusions = writer('source_exclusions.jsonl.gz', True)
            counts = Counter()
            catalog_metadata = {}
            for line, record in records(metadata):
                counts['metadata_input'] += 1
                item, title, reason = metadata_record(record, dataset, history_shelf, required)
                if reason:
                    counts['metadata_excluded:' + reason] += 1
                    exclusions.write(encoded({'source': 'metadata', 'source_line': line, 'item_id': item, 'reason': reason}))
                else:
                    if item in catalog_metadata:
                        if catalog_metadata[item]['title'] != title:
                            raise DataError(f'metadata line {line}: conflicting titles for duplicate item ID {item!r}')
                        counts['metadata_excluded:duplicate_identical_eligible_metadata'] += 1
                        exclusions.write(encoded({'source': 'metadata', 'source_line': line, 'item_id': item, 'reason': 'duplicate_identical_eligible_metadata'}))
                        continue
                    catalog_metadata[item] = {'title': title}
                    counts['metadata_eligible'] += 1
            if not catalog_metadata:
                raise DataError('metadata eligibility rules retained no items')
            print(f"Metadata complete: {len(catalog_metadata)} eligible of {counts['metadata_input']} records", file=sys.stderr, flush=True)
            db_path = output / '.staging.sqlite3'
            db = sqlite3.connect(db_path)
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE raw_events (source_line INTEGER PRIMARY KEY, user_id TEXT, item_id TEXT, timestamp INTEGER, timestamp_raw TEXT, precision TEXT, rating TEXT)')
            batch = []
            for line, record in records(reviews):
                counts['reviews_input'] += 1
                if counts['reviews_input'] % 250000 == 0:
                    print(f"Review stream: {counts['reviews_input']} source records read", file=sys.stderr, flush=True)
                user, item = (identifier(record.get(user_field)), identifier(record.get(item_field)))
                if dataset == 'steam':
                    identity_field = user_field
                    if user is None and steam_user_fallback == 'username':
                        user = identifier(record.get('username'))
                        identity_field = 'username'
                    if user is not None:
                        user = identity_field + ':' + user
                        counts['steam_identity_source:' + identity_field] += 1
                raw_date = record.get(date_field)
                reason = None
                if user is None or item is None:
                    reason = 'missing_user_or_item_id'
                elif item not in catalog_metadata:
                    reason = 'ineligible_or_missing_item_metadata'
                elif require_review_text and (not isinstance(record.get(text_field), str) or not record[text_field].strip()):
                    reason = 'missing_review_text'
                else:
                    try:
                        timestamp, precision = parse_timestamp(raw_date)
                    except (ValueError, TypeError, OverflowError):
                        reason = 'missing_or_invalid_timestamp'
                if reason:
                    counts['reviews_excluded:' + reason] += 1
                    exclusions.write(encoded({'source': 'reviews', 'source_line': line, 'user_id': user, 'item_id': item, 'reason': reason, 'timestamp_raw': raw_date}))
                    continue
                rating = json.dumps(record.get('rating'), allow_nan=False)
                batch.append((line, user, item, timestamp, raw_date, precision, rating))
                if len(batch) >= 5000:
                    db.executemany('INSERT INTO raw_events VALUES (?,?,?,?,?,?,?)', batch)
                    db.commit()
                    batch.clear()
            if batch:
                db.executemany('INSERT INTO raw_events VALUES (?,?,?,?,?,?,?)', batch)
                db.commit()
            print(f"Review stream complete: {counts['reviews_input']} records; resolving duplicates and user eligibility", file=sys.stderr, flush=True)
            db.row_factory = sqlite3.Row
            db.execute('CREATE INDEX raw_user_item_time ON raw_events(user_id,item_id,timestamp,source_line)')
            db.execute('CREATE TABLE ranked AS SELECT *,ROW_NUMBER() OVER(PARTITION BY user_id,item_id ORDER BY timestamp,source_line) AS occurrence FROM raw_events')
            for event in db.execute('SELECT * FROM ranked WHERE occurrence>1 ORDER BY source_line'):
                counts['reviews_excluded:duplicate_user_item_later_occurrence'] += 1
                exclusions.write(encoded({'source': 'reviews', 'source_line': event['source_line'], 'user_id': event['user_id'], 'item_id': event['item_id'], 'timestamp_raw': event['timestamp_raw'], 'reason': 'duplicate_user_item_later_occurrence'}))
            eligible_users = [row[0] for row in db.execute('SELECT user_id FROM ranked WHERE occurrence=1 GROUP BY user_id HAVING COUNT(*)>=? ORDER BY user_id', (minimum,))]
            counts['users_eligible_before_sampling'] = len(eligible_users)
            print(f'Eligible users after deduplication and minimum {minimum}: {len(eligible_users)}', file=sys.stderr, flush=True)
            selected_users = set(eligible_users)
            if max_users is not None and len(eligible_users) > max_users:
                selected_users = set(sorted(eligible_users, key=lambda u: (hashlib.sha256(encoded([VERSION, user_sample_seed, u])).digest(), u))[:max_users])
            counts['users_selected'] = len(selected_users)
            print(f'Selected users: {len(selected_users)}; max_users={max_users}, user_sample_seed={user_sample_seed}', file=sys.stderr, flush=True)
            eligible_users = set(eligible_users)
            db.execute('CREATE TABLE selected_users (user_id TEXT PRIMARY KEY)')
            db.executemany('INSERT INTO selected_users VALUES (?)', ((u,) for u in sorted(selected_users)))
            for event in db.execute('SELECT * FROM ranked WHERE occurrence=1 AND user_id NOT IN (SELECT user_id FROM selected_users) ORDER BY source_line'):
                reason = 'user_not_in_deterministic_sample' if event['user_id'] in eligible_users else 'user_below_minimum_interactions'
                counts['reviews_excluded:' + reason] += 1
                exclusions.write(encoded({'source': 'reviews', 'source_line': event['source_line'], 'user_id': event['user_id'], 'item_id': event['item_id'], 'timestamp_raw': event['timestamp_raw'], 'reason': reason}))
            db.execute('CREATE TABLE events AS SELECT source_line,user_id,item_id,timestamp,timestamp_raw,precision,rating FROM ranked WHERE occurrence=1 AND user_id IN (SELECT user_id FROM selected_users)')
            db.execute('CREATE INDEX events_user_time ON events(user_id,timestamp,item_id)')
            counts['reviews_preprocessed'] = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
            item_ids = [row[0] for row in db.execute('SELECT DISTINCT item_id FROM events ORDER BY item_id')]
            if len(item_ids) < 101:
                raise DataError('fewer than 101 retained catalog items; cannot form history+target+99 negatives')
            items = {item: catalog_metadata[item] for item in item_ids}
            print(f"Freezing {counts['reviews_preprocessed']} interactions and {len(items)} items", file=sys.stderr, flush=True)
            (output / 'items.json').write_bytes(encoded(items, True))
            streams = {name: writer(name) for name in ['interactions.jsonl', 'train_interactions.jsonl', 'excluded_queries.jsonl'] + [f'{s}.jsonl' for s in SPLITS]}
            split_counts, query_counts, query_user_counts, query_exclusions = (Counter(), Counter(), Counter(), Counter())
            first_train, training_counts, boundaries = ({}, Counter(), {})
            date_precisions, cold_targets = (Counter(), Counter())
            sequence_rows = db.execute('SELECT * FROM events ORDER BY user_id,timestamp,item_id')
            for user, group in itertools.groupby(sequence_rows, key=lambda e: e['user_id']):
                sequence = list(group)
                n, cursor, past, observed = (len(sequence), 0, [], set())
                train_end, valid_end = (n * 8 // 10, n * 9 // 10)
                boundaries[user] = {'train_end_exclusive': train_end, 'valid_end_exclusive': valid_end, 'event_count': n}
                user_splits = set()
                for index, event in enumerate(sequence):
                    while cursor < index and sequence[cursor]['timestamp'] < event['timestamp']:
                        past.append(sequence[cursor])
                        observed.add(sequence[cursor]['item_id'])
                        cursor += 1
                    split = 'train' if index < train_end else 'valid' if index < valid_end else 'test'
                    query_id = f"{dataset}:review-line:{event['source_line']}"
                    audit = {'query_id': query_id, 'user_id': user, 'item_id': event['item_id'], 'timestamp': event['timestamp'], 'timestamp_raw': event['timestamp_raw'], 'timestamp_precision': event['precision'], 'timestamp_field': date_field, 'rating': json.loads(event['rating']), 'source_line': event['source_line'], 'split': split, 'user_event_index': index}
                    streams['interactions.jsonl'].write(encoded(audit))
                    split_counts[split] += 1
                    date_precisions[event['precision']] += 1
                    if split == 'train':
                        streams['train_interactions.jsonl'].write(encoded(audit))
                        training_counts[event['item_id']] += 1
                        first_train[event['item_id']] = min(first_train.get(event['item_id'], event['timestamp']), event['timestamp'])
                    pool_size = len(items) - len(observed) - 1
                    reason = 'empty_strictly_earlier_history' if not past else 'insufficient_eligible_negatives' if pool_size < 99 else None
                    if reason:
                        query_exclusions[f'{split}:{reason}'] += 1
                        streams['excluded_queries.jsonl'].write(encoded({**audit, 'reason': reason, 'eligible_negative_count': pool_size}))
                        continue
                    history = past[-max_history:] if max_history else past
                    row = {'query_id': query_id, 'user_id': user, 'target_id': event['item_id'], 'history': [e['item_id'] for e in history], 'history_timestamps': [e['timestamp'] for e in history], 'candidate_ids': query_candidates(item_ids, observed, event['item_id'], seed, query_id), 'timestamp': event['timestamp'], 'timestamp_raw': event['timestamp_raw'], 'timestamp_precision': event['precision'], 'timestamp_field': date_field, 'source_line': event['source_line'], 'user_event_index': index, 'history_length_before_truncation': len(past), 'observed_past_count': len(past), 'same_time_predecessors_excluded': index - len(past)}
                    streams[f'{split}.jsonl'].write(encoded(row))
                    query_counts[split] += 1
                    user_splits.add(split)
                for split in user_splits:
                    query_user_counts[split] += 1
            if any((query_counts[s] == 0 for s in SPLITS)):
                raise DataError('at least one split has no eligible queries; data is unusable for comparison')
            (output / 'catalog.json').write_bytes(encoded({item: {'training_interactions': training_counts[item], 'first_train_timestamp': first_train.get(item)} for item in item_ids}, True))
            db.close()
            db = None
            db_path.unlink()
        protocol = {'version': VERSION, 'dataset': dataset, 'status': 'reconstruction_from_published_protocol_not_verified_original_frozen_splits', 'published_requirements': {'minimum_user_interactions': 20 if dataset == 'steam' else 40, 'chronological_split': '8:1:1', 'candidates': 100, 'goodreads_genre': 'History' if dataset == 'goodreads-history' else None}, 'source_fields': {'user': user_field, 'item': item_field, 'date': date_field, 'review_text': text_field}, 'eligibility': {'minimum_user_interactions': minimum, 'require_nonempty_review_text': require_review_text, 'metadata': 'nonempty title; Steam may fall back to app_name', 'metadata_duplicates': 'repeat eligible item ID accepted only when title exactly matches; duplicate source row audited; conflicting titles fail', 'goodreads_required_fields': list(required) if dataset == 'goodreads-history' else None, 'goodreads_history_shelf': history_shelf if dataset == 'goodreads-history' else None, 'goodreads_genre_rule': f'popular_shelves contains exact casefolded name {history_shelf!r} with count>0' if dataset == 'goodreads-history' else None, 'filter_order': 'metadata/genre, nonempty review text, valid timestamp, earliest user-item deduplication, user minimum, optional fixed user sample', 'max_users': max_users, 'user_sample_seed': user_sample_seed, 'steam_user_fallback': steam_user_fallback if dataset == 'steam' else None, 'steam_identity_policy': 'default uses username consistently for all rows because raw user_id is mixed/missing; username uniqueness is not guaranteed by source. Canonical IDs prefixed with actual field; an explicit alternate user_field plus fallback never merges field namespaces' if dataset == 'steam' else None, 'user_sampling': 'SHA256 priorities over eligible user IDs; all eligible users when max_users is null', 'missing_values': 'excluded with source line and reason; timestamps never imputed'}, 'split': 'per-user floor(.8*n),floor(.9*n),n after filtering; equal-time order lexical canonical item ID', 'history': {'max_items': max_history, 'visibility': 'strictly earlier timestamps only; same-time block excluded', 'evaluation': 'rolling observed user history; earlier validation/test events visible for subsequent queries'}, 'timestamp': {'unit': 'milliseconds', 'original_string': 'retained in every interaction/query', 'day_precision': 'midnight UTC encodes an observed calendar day, not an observed time of day; same-day events are simultaneous', 'goodreads_clock_meaning': 'date_added is shelf addition for the reviewed book, not independently verified review publication' if dataset == 'goodreads-history' and date_field == 'date_added' else 'chosen source field; meaning recorded by field name'}, 'catalog': 'all items retained after preprocessing; closed catalog; title only; no warm-target exclusions', 'candidates': {'count': 100, 'seed': seed, 'sampling': 'SHA256 counter draws; unbiased rejection over full catalog, without replacement; separate shuffle namespace', 'exclude': 'all strictly earlier observed items plus target, before history truncation', 'future_positives': 'not consulted per query; may be sampled as currently unobserved negatives', 'insufficient_pool_or_empty_history': 'query excluded and audited, interaction preserved'}, 'training_context': 'fit model and collaborator indexes from train_interactions.jsonl only; closed metadata catalog contains no held-out review text or label statistics', 'temporal_scope': 'offline per-user split, not a globally chronological wall-clock backtest; full-data filtering is an explicit preprocessing choice', 'metrics': {'averaging': 'query_macro', 'cutoffs': [1, 5, 10], 'mrr': 'full candidate list', 'tie_break': 'score descending then string item ID ascending'}, 'source_urls': {'steam': 'https://github.com/kang205/SASRec#steam-dataset', 'goodreads-history': 'https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html'}[dataset]}
        stats = {'source_and_filter_counts': dict(counts), 'catalog_items': len(items), 'training_observed_catalog_items': len(first_train), 'catalog_items_without_training_interactions': len(items) - len(first_train), 'split_interactions': dict(split_counts), 'query_counts': dict(query_counts), 'query_user_counts': dict(query_user_counts), 'excluded_queries': dict(query_exclusions), 'timestamp_precision_counts': dict(date_precisions), 'manuscript_table_counts': {'users': 11938, 'items': 3581, 'interactions': 274726} if dataset == 'steam' else {'users': 1120, 'items': 2359, 'interactions': 73637}}
        output_names = sorted((p.name for p in output.iterdir() if p.is_file()))
        manifest = {'schema_version': 1, 'protocol': protocol, 'raw_file_sha256': source_hashes, 'raw_filenames': {'reviews': reviews.name, 'metadata': metadata.name}, 'statistics': stats, 'output_file_sha256': {name: file_hash(output / name) for name in output_names}, 'fingerprint_rule': 'SHA256(exact manifest.json bytes); splits.json is a derived pointer excluded to avoid circular hashing'}
        (output / 'manifest.json').write_bytes(encoded(manifest, True))
        fingerprint = file_hash(output / 'manifest.json')
        (output / 'splits.json').write_bytes(encoded({'dataset': dataset, 'protocol_sha256': fingerprint, 'manifest': 'manifest.json', 'query_counts': dict(query_counts), 'timestamp_unit': 'milliseconds', 'history_max_items': max_history, 'split_endpoints': boundaries}, True))
        return {'protocol_sha256': fingerprint, **stats}
    except BaseException:
        if db is not None:
            db.close()
        shutil.rmtree(output)
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('steam', 'goodreads-history'), required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--max-history', type=int)
    parser.add_argument('--max-users', type=int)
    parser.add_argument('--user-sample-seed', type=int, default=20260930)
    parser.add_argument('--history-shelf', default='history')
    parser.add_argument('--book-required', default='title,authors')
    parser.add_argument('--date-field')
    parser.add_argument('--user-field')
    parser.add_argument('--item-field')
    parser.add_argument('--text-field')
    parser.add_argument('--steam-user-fallback', choices=('username', 'none'), default='username')
    parser.add_argument('--minimum-user-interactions', type=int)
    parser.add_argument('--allow-empty-review-text', dest='require_review_text', action='store_false', default=False)
    parser.add_argument('--require-review-text', dest='require_review_text', action='store_true', help='optional stricter variant; default retains review records regardless of text content')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(**vars(args)), sort_keys=True, indent=2))
    except (DataError, OSError, ValueError, sqlite3.Error) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
