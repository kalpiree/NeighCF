import json
from pathlib import Path
import random
import numpy as np
from integrations.resume import digest, file_digest
from .source import COMMIT


class Assets:
    def __init__(self, directory, bundle, seed, max_len):
        self.directory = Path(directory).resolve()
        self.seed = seed
        self.max_len = max_len
        path = self.directory / 'manifest.json'
        self.manifest = json.loads(path.read_text())
        m = self.manifest
        if m.get('schema') != 'llmcf-authentic-augmentation-v1' or m.get('source_commit') != COMMIT or m.get('protocol_sha256') != bundle['metadata']['protocol_sha256']:
            raise ValueError('augmentation source/protocol/schema mismatch')
        for key in ('generator_model', 'generator_revision', 'embedding_model', 'embedding_revision', 'generation_description'):
            if not isinstance(m.get(key), str) or not m[key].strip():
                raise ValueError('authentic augmentation provenance is required: ' + key)
        self.hashes = {'manifest.json': file_digest(path)}
        for name in ('bank.jsonl', 'cot_embeddings.npy', 'item_embeddings.npy'):
            p = self.directory / name
            if p.is_symlink() or not p.is_file() or file_digest(p) != m.get('sha256', {}).get(name):
                raise ValueError('augmentation file missing or hash differs: ' + name)
            self.hashes[name] = m['sha256'][name]
        ids = m.get('item_ids')
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(bundle['items']):
            raise ValueError('item embedding IDs must exactly cover canonical catalog')
        self.item = np.load(self.directory / 'item_embeddings.npy', allow_pickle=False)
        self.cot = np.load(self.directory / 'cot_embeddings.npy', allow_pickle=False)
        for name, value, count in (('item', self.item, len(ids)), ('cot', self.cot, None)):
            if value.ndim != 2 or value.shape[1] != 768 or count is not None and value.shape[0] != count or value.dtype.kind != 'f' or not np.isfinite(value).all() or np.any(np.linalg.norm(value, axis=1) == 0):
                raise ValueError('finite nonzero 768D augmentation features required: ' + name)
        self.item = self.item.astype(np.float32)
        self.cot = self.cot.astype(np.float32)
        if not np.isfinite(self.item).all() or not np.isfinite(self.cot).all():
            raise ValueError('augmentation features overflow native float32')
        self.item_index = {item: i for i, item in enumerate(ids)}
        train = {row['query_id']: row for row in bundle['rows']['train']}
        self.bank = []
        seen = set()
        with (self.directory / 'bank.jsonl').open() as stream:
            for line in stream:
                value = json.loads(line)
                example = value.get('example_id')
                query = train.get(value.get('source_query_id'))
                if not isinstance(example, str) or not example or example in seen or query is None:
                    raise ValueError('demonstration must name a unique example from canonical training')
                candidate = value.get('candidate_id')
                if candidate not in query['candidate_ids'] or type(value.get('label')) is not int or value['label'] != int(candidate == query['target_id']):
                    raise ValueError('demonstration candidate/label differs from its training query')
                cot_index = value.get('cot_index')
                if type(cot_index) is not int or not 0 <= cot_index < len(self.cot) or not isinstance(value.get('cot_text'), str) or not value['cot_text'].strip():
                    raise ValueError('demonstration requires its real CoT text and aligned embedding index')
                seen.add(example)
                self.bank.append(value | {'query': query})
        if not self.bank or len({x['cot_index'] for x in self.bank}) != len(self.bank) or len(self.cot) != len(self.bank):
            raise ValueError('each demonstration requires one distinct CoT embedding')
        self.bank.sort(key=lambda x: x['example_id'])
        self.features = np.stack([self.history_feature(row['query']) for row in self.bank])

    def history_feature(self, query):
        return self.item[[self.item_index[x] for x in query['history']]].mean(axis=0)

    def retrieve(self, query):
        scores = self.features @ self.history_feature(query)
        chosen = []
        for label in (1, 0):
            eligible = [i for i, row in enumerate(self.bank) if row['label'] == label and row['source_query_id'] != query['query_id'] and (row['query']['user_id'] != query['user_id'] or row['query']['timestamp'] < query['timestamp'])]
            eligible.sort(key=lambda i: (-float(scores[i]), self.bank[i]['example_id']))
            if len(eligible) < 2:
                raise ValueError('insufficient two-positive/two-negative causal training demonstrations for ' + query['query_id'])
            chosen.extend(eligible[:2])
        random.Random(int(digest(['llmcf-demo-order-v1', self.seed, query['query_id']]), 16)).shuffle(chosen)
        return [self.bank[i] for i in chosen]
