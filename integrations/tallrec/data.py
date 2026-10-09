import hashlib
import json
import random

from adapters.llmsrec_data import ExcludedCatalog, load_canonical
from integrations.resume import digest
from integrations.traditional.run import protocol

POLICY = 'implicit-history-positive-one-unobserved-negative-v1'


def bundle(args):
    items, rows, metadata, hashes = load_canonical(args.data, args.expected_candidates)
    if metadata.get('dataset') not in (None, args.dataset):
        raise ValueError('dataset label differs from canonical metadata')
    return {'items': items, 'rows': rows, 'hashes': hashes,
            'protocol_sha256': protocol(metadata, args.protocol_sha256)}


def example(items, row, candidate, positive, domain):
    preference = ', '.join('"' + items[item]['title'] + '"' for item in row['history'][-10:])
    target = '"' + items[candidate]['title'] + '"'
    return {'instruction': f'Given the user\'s preference and unpreference, identify whether the user will like the target {domain} by answering "Yes." or "No.".',
            'input': f'User Preference: {preference}\nUser Unpreference: \nWhether the user will like the target {domain} {target}?',
            'output': 'Yes.' if positive else 'No.'}


def binary_records(data, split, seed, domain):
    catalog = sorted(data['items'])
    positions = {item: i for i, item in enumerate(catalog)}
    for row in data['rows'][split]:
        if split == 'train':
            excluded = {positions[item] for item in row['history']}
            excluded.add(positions[row['target_id']])
            eligible = ExcludedCatalog(catalog, excluded)
            if not len(eligible):
                raise ValueError('no unobserved negative for query ' + row['query_id'])
            rng = random.Random(int(digest({'seed': seed, 'query_id': row['query_id']}), 16))
            candidates = [row['target_id'], eligible[rng.randrange(len(eligible))]]
        else:
            candidates = row['candidate_ids']
        for candidate in candidates:
            yield example(data['items'], row, candidate, candidate == row['target_id'], domain)


def export_json(path, records):
    from integrations.resume import _sync_dir
    import os
    count = 0
    with path.open('x') as stream:
        stream.write('[')
        for row in records:
            stream.write((',' if count else '') + json.dumps(row, ensure_ascii=False, sort_keys=True))
            count += 1
        stream.write(']\n')
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)
    return count


def check_tokenizer(tokenizer):
    if tokenizer.encode('Yes.', add_special_tokens=False)[0] != 8241 or tokenizer.encode('No.', add_special_tokens=False)[0] != 3782:
        raise ValueError('native TALLRec requires LLaMA Yes/No token IDs 8241/3782')
    if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
        raise ValueError('native TALLRec requires LLaMA BOS/EOS 1/2')


def validate_prompt_tokens(tokenizer, prompt_function, examples, cutoff):
    for index, row in enumerate(examples):
        tokens = tokenizer(prompt_function(row), truncation=False, padding=False)['input_ids']
        selected = [token for token in tokens if token in (8241, 3782)]
        expected = [8241, 3782, 8241 if row['output'] == 'Yes.' else 3782]
        if selected != expected:
            raise ValueError(f'native validation token-position assumption fails at example {index}')
        if len(tokens) >= cutoff:
            raise ValueError(f'prompt exceeds native cutoff before complete answer at example {index}')


def native_yes_score(logits):
    import torch
    if logits.ndim != 2 or logits.shape[1] <= 8241 or not torch.isfinite(logits).all():
        raise ValueError('finite batch-by-vocabulary logits required')
    probabilities = logits.softmax(dim=-1)
    return probabilities[:, [8241, 3782]].float().softmax(dim=-1)[:, 0]
