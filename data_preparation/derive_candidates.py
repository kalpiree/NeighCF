from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil

class DerivationError(ValueError):
    pass

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)

def reduced_candidates(row, expected, source_expected, split='valid', seed=42):
    candidates, target = (row['candidate_ids'], row['target_id'])
    if len(candidates) != source_expected or len(set(candidates)) != len(candidates) or candidates.count(target) != 1:
        raise DerivationError('source candidate count/uniqueness/positive violation')
    if not 2 <= expected <= source_expected:
        raise DerivationError('derived candidate count must be between 2 and source count')
    negatives = [item for item in candidates if item != target]
    material = json.dumps([seed, split, row['query_id'], 'candidate-override-v2'], separators=(',', ':'), ensure_ascii=True)
    query_seed = int(hashlib.sha256(material.encode()).hexdigest(), 16)
    rng = random.Random(query_seed)
    selected = [target] + rng.sample(negatives, expected - 1)
    rng.shuffle(selected)
    return selected

def verified_manifest(source):
    source = Path(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    splits = json.loads((source / 'splits.json').read_text())
    if sha256(source / 'manifest.json') != splits['protocol_sha256']:
        raise DerivationError('source manifest/protocol mismatch')
    for name, expected in manifest['output_file_sha256'].items():
        if Path(name).name != name or sha256(source / name) != expected:
            raise DerivationError(f'source file hash mismatch: {name}')
    return (manifest, splits)

def derive(source, output, expected_candidates=51, seed=42):
    source, output = (Path(source).resolve(), Path(output).resolve())
    if output.exists() or source == output or source in output.parents:
        raise DerivationError('output must be a new directory outside the source bundle')
    original, split_meta = verified_manifest(source)
    original_count = original['protocol']['candidates']['count']
    if not 2 <= expected_candidates < original_count:
        raise DerivationError('new evaluation count must be smaller than source count and at least 2')
    output.mkdir(parents=True)
    changed = {'valid.jsonl', 'test.jsonl'}
    for name in original['output_file_sha256']:
        if name not in changed:
            shutil.copyfile(source / name, output / name)
    counts = {}
    for split in ('valid', 'test'):
        count = 0
        with (output / f'{split}.jsonl').open('w') as stream:
            for row in rows(source / f'{split}.jsonl'):
                row['candidate_ids'] = reduced_candidates(row, expected_candidates, original_count, split, seed)
                stream.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
                count += 1
        counts[split] = count
    derivation = {'version': 'candidate-sample-shuffle-v2', 'seed': seed, 'source_protocol_sha256': split_meta['protocol_sha256'], 'source_manifest_sha256': sha256(source / 'manifest.json'), 'source_splits_sha256': sha256(source / 'splits.json'), 'source_output_file_sha256': original['output_file_sha256'], 'source_candidates': original_count, 'expected_candidates': expected_candidates, 'selection_rule': 'Python random.Random(query_seed).sample(all saved negatives in saved order, expected_candidates-1); prepend target; shuffle the complete selected list using the same RNG', 'query_seed_rule': "int(SHA256(UTF8(json.dumps([seed,split,query_id,'candidate-override-v2'],separators=(',',':'),ensure_ascii=True))).hexdigest(),16)", 'randomness': 'deterministic per-query uniform sampling without replacement and full candidate shuffle; seed42 by default', 'scope': 'user-requested evaluation override; training, catalog, histories, timestamps, query identities and source cohort unchanged', 'query_counts': counts}
    manifest = copy.deepcopy(original)
    manifest['candidate_derivation'] = derivation
    cfg = manifest['protocol']['candidates']
    cfg.update(count=expected_candidates, evaluation_count=expected_candidates, training_count=original['protocol']['candidates'].get('training_count', original_count), cohort_eligibility_count=original['protocol']['candidates'].get('cohort_eligibility_count', original_count), derivation_selection_rule=derivation['selection_rule'])
    manifest['protocol']['evaluation_override'] = 'user requested 50 negatives plus one positive' if expected_candidates == 51 else 'explicit user-selected evaluation candidate count'
    manifest['output_file_sha256'] = {name: sha256(output / name) for name in original['output_file_sha256']}
    (output / 'manifest.json').write_bytes(encoded(manifest))
    splits = copy.deepcopy(split_meta)
    splits.update(protocol_sha256=sha256(output / 'manifest.json'), expected_candidates=expected_candidates, candidate_derivation=derivation)
    (output / 'splits.json').write_bytes(encoded(splits))
    return audit(source, output)

def audit(source, derived):
    source, derived = (Path(source), Path(derived))
    old, old_splits = verified_manifest(source)
    new, new_splits = verified_manifest(derived)
    relation = new.get('candidate_derivation', {})
    if relation != new_splits.get('candidate_derivation') or relation.get('source_protocol_sha256') != old_splits['protocol_sha256']:
        raise DerivationError('candidate derivation parent mismatch')
    if relation.get('source_output_file_sha256') != old['output_file_sha256'] or relation.get('source_splits_sha256') != sha256(source / 'splits.json'):
        raise DerivationError('derivation parent hash inventory changed')
    if set(new['output_file_sha256']) != set(old['output_file_sha256']):
        raise DerivationError('derived file inventory differs')
    counts = {}
    for name, old_hash in old['output_file_sha256'].items():
        if name not in ('valid.jsonl', 'test.jsonl'):
            if new['output_file_sha256'][name] != old_hash:
                raise DerivationError(f'non-evaluation file changed: {name}')
            continue
        count = 0
        for before, after in itertools.zip_longest(rows(source / name), rows(derived / name)):
            if before is None or after is None:
                raise DerivationError('evaluation query count changed')
            c, target = (before['candidate_ids'], before['target_id'])
            if len(c) != relation['source_candidates'] or len(set(c)) != len(c) or c.count(target) != 1:
                raise DerivationError('invalid parent candidates')
            key = json.dumps([relation['seed'], name[:-6], before['query_id'], 'candidate-override-v2'], separators=(',', ':'), ensure_ascii=True)
            rng = random.Random(int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest(), 'big'))
            expected = [target]
            expected.extend(rng.sample([item for item in c if item != target], relation['expected_candidates'] - 1))
            rng.shuffle(expected)
            if after.get('candidate_ids') != expected:
                raise DerivationError('derived candidates violate recorded selection rule')
            if {k: v for k, v in before.items() if k != 'candidate_ids'} != {k: v for k, v in after.items() if k != 'candidate_ids'}:
                raise DerivationError('evaluation history, target, identity, timestamp or metadata changed')
            count += 1
        counts[name[:-6]] = count
    if counts != relation['query_counts'] or new_splits['protocol_sha256'] == old_splits['protocol_sha256']:
        raise DerivationError('derivation counts or new protocol identity invalid')
    return {'status': 'passed', 'source_protocol_sha256': old_splits['protocol_sha256'], 'protocol_sha256': new_splits['protocol_sha256'], 'expected_candidates': relation['expected_candidates'], 'query_counts': counts, 'unchanged_training_and_catalog_verified': True, 'unchanged_histories_timestamps_targets_query_ids_verified': True}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('derive', 'audit'))
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-candidates', type=int, default=51)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    result = derive(args.source, args.output, args.expected_candidates, args.seed) if args.command == 'derive' else audit(args.source, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
