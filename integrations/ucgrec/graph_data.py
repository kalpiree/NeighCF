from __future__ import annotations
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

class GraphError(ValueError):
    pass

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()
DATASET_POLICIES = {'movielens_native_v1': {'loader': 'data/movielens_data.py', 'class': 'MovielensData', 'prompt': 'prompt/movie.txt', 'long_term': 'semantic_top3', 'item_noun': 'movie', 'adaptation': 'native MovieLens top-k policy'}, 'steam_native_v1': {'loader': 'data/steam_data.py', 'class': 'SteamData', 'prompt': 'prompt/game.txt', 'long_term': 'random_half_unsorted', 'item_noun': 'game', 'adaptation': 'native Steam top-k policy'}, 'goodreads_semantic_adaptation_v1': {'loader': 'data/movielens_data.py', 'class': 'MovielensData', 'prompt': 'prompt/movie.txt', 'long_term': 'semantic_top3', 'item_noun': 'book', 'adaptation': 'controlled Goodreads adaptation: MovieLens semantic view selection and book wording; Goodreads is not a native source dataset'}, 'lastfm_artist_semantic_adaptation_v1': {'loader': 'data/movielens_data.py', 'class': 'MovielensData', 'prompt': 'prompt/movie.txt', 'long_term': 'semantic_top3', 'item_noun': 'artist', 'adaptation': 'controlled LastFM artist adaptation: MovieLens semantic view selection with artist descriptions and listening-history wording; not an official LastFM recipe'}}

def dataset_policy(name):
    if name not in DATASET_POLICIES:
        raise GraphError('an explicit supported dataset policy is required')
    return {'version': name, **DATASET_POLICIES[name]}

def recommendation_prompt(source, policy_name):
    policy = dataset_policy(policy_name)
    text = (Path(source) / policy['prompt']).read_text().strip()
    if policy_name == 'goodreads_semantic_adaptation_v1':
        text = text.replace('viewing history', 'reading history')
    if policy_name == 'lastfm_artist_semantic_adaptation_v1':
        text = text.replace('viewing history', 'listening history')
    markers = ('[LongTermSeq]', '[ShortTermSeq]', '[HistoryHere]', '[UserOut]')
    if any((text.count(marker) != 1 for marker in markers)):
        raise GraphError('native recommendation prompt lacks required unique graph/output markers')
    return text

def timestamp_milliseconds(row, declared_unit=None):
    values = []
    for field, multiplier in (('timestamp_unix_s', 1000), ('timestamp_unix_ms', 1)):
        if field in row:
            if type(row[field]) is not int:
                raise GraphError('event timestamp must be an exact integer in its declared unit')
            values.append(row[field] * multiplier)
    if 'timestamp' in row:
        if declared_unit not in {'milliseconds', 'seconds'} or type(row['timestamp']) is not int:
            raise GraphError('generic timestamp requires explicit seconds/milliseconds metadata and an integer value')
        values.append(row['timestamp'] * (1000 if declared_unit == 'seconds' else 1))
    if not values or any((value != values[0] for value in values)):
        raise GraphError('missing or inconsistent event timestamps')
    return values[0]

def build_global_graph(path, mapping, indexed_splits, timestamp_unit=None):
    grouped, event_ids = (defaultdict(dict), set())
    item_map = mapping['item_to_index']
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get('split') != 'train':
                raise GraphError(f'training graph event {line_number} is not marked train')
            user, item = (row.get('user_id'), row.get('item_id'))
            index, timestamp = (row.get('user_event_index'), timestamp_milliseconds(row, timestamp_unit))
            if not isinstance(user, str) or item not in item_map or type(index) is not int or (index < 0) or (type(timestamp) is not int):
                raise GraphError(f'malformed graph event {line_number}')
            query_id = row.get('query_id')
            if not isinstance(query_id, str) or query_id in event_ids or index in grouped[user]:
                raise GraphError('duplicate/missing event identity in training graph source')
            event_ids.add(query_id)
            grouped[user][index] = (timestamp, item_map[item] - 1, query_id)
    if not grouped:
        raise GraphError('empty training graph event file')
    for split in ('valid', 'test'):
        if any((row['source']['query_id'] in event_ids for row in indexed_splits[split])):
            raise GraphError('held-out target event is present in training graph')
    for row in indexed_splits['train']:
        source = row['source']
        if source['query_id'] not in event_ids:
            raise GraphError('canonical training target missing from graph event table')
    incoming, counts = ({i: [] for i in range(len(item_map))}, Counter())
    edge_counts = Counter()
    for user in sorted(grouped):
        events = grouped[user]
        ordered = sorted(events)
        if ordered != list(range(len(ordered))):
            raise GraphError(f'training events for {user} are not a complete chronological prefix')
        rows = [events[index] for index in ordered]
        if any((a[0] > b[0] for a, b in zip(rows, rows[1:]))):
            raise GraphError(f'training event timestamps for {user} are not chronological')
        for left, right in zip(rows, rows[1:]):
            src, dst = (left[1], right[1])
            counts[src] += 1
            counts[dst] += 1
            edge_counts[src, dst] += 1
            if src not in incoming[dst]:
                incoming[dst].append(src)
    total = sum(counts.values())
    popularity = {item: counts[item] / total if total else 0.0 for item in incoming}
    return {'incoming': incoming, 'popularity': popularity, 'training_events_sha256': sha256(path), 'events': len(event_ids), 'users': len(grouped), 'unique_edges': len(edge_counts), 'transition_occurrences': sum(edge_counts.values()), 'policy': 'pinned source: directed unique edges, endpoint-frequency node popularity; train events only', 'timestamp_normalization': 'explicit integer seconds multiplied by1000; integer milliseconds preserved losslessly', 'input_timestamp_unit': timestamp_unit}

class GraphSurface:

    def __init__(self, graph):
        self.graph = graph

    def in_edges(self, node):
        return [(source, node) for source in self.graph['incoming'].get(node, [])]

    def nodes(self, data=None):
        if data != 'weight':
            raise GraphError('unexpected source graph API')
        return self.graph['popularity']

def source_functions(path, class_name, names, namespace):
    tree = ast.parse(Path(path).read_text())
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name))
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != set(names):
        raise GraphError('required method is absent from pinned source')
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}

class QueryGraphs:

    def __init__(self, source, bundle, graph, features, evaluation_seed=20260930, policy_name='movielens_native_v1'):
        import numpy as np
        import torch
        self.np, self.torch = (np, torch)
        self.bundle, self.graph, self.features = (bundle, graph, features)
        self.padding_item_id = len(bundle['items'])
        self.max_len = 10
        self.global_graph = GraphSurface(graph)
        self.evaluation_seed = evaluation_seed
        self.policy = dataset_policy(policy_name)
        if self.policy['long_term'] == 'semantic_top3' and features is None:
            raise GraphError('semantic long-term selection requires proper content features')
        self.methods = source_functions(Path(source) / self.policy['loader'], self.policy['class'], ('niche_walk', 'build_graph', 'local_graph'), {'np': np, 'random': random})

    def sample(self, query, *, training_seed, epoch=None):
        np, torch = (self.np, self.torch)
        history = [index - 1 for index in query['history_item_indices'][-10:]]
        if not history:
            raise GraphError('empty histories are unsupported')
        short_k = 3 if len(history) > 3 else max(1, len(history) // 2)
        key = [self.bundle['source_split_metadata'].get('protocol_sha256'), query['source']['query_id'], self.evaluation_seed if epoch is None else training_seed, 'evaluation' if epoch is None else epoch]
        seed = int(hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8], 16)
        py_state, np_state = (random.getstate(), np.random.get_state())
        try:
            random.seed(seed)
            np.random.seed(seed)
            walks = self.methods['niche_walk'](self, history, len(history))
            if self.policy['long_term'] == 'random_half_unsorted':
                long_positions = random.sample(range(len(history)), max(1, len(history) // 2))
            elif len(history) == 1:
                long_positions = [0]
            else:
                embeddings = self.features[history]
                sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
                means = (sim - torch.eye(len(history))).sum(1) / (len(history) - 1)
                long_positions = sorted(torch.topk(means, short_k).indices.tolist())
            long_term = [history[index] for index in long_positions]
            short_term = history[-short_k:]
            g_adj, g_weight = self.methods['build_graph'](self, walks + [history, long_term, short_term])
            views = [sequence + [self.padding_item_id] * (10 - len(sequence)) for sequence in (history, long_term, short_term)]
            local, aliases, items = self.methods['local_graph'](self, views)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        g_adj[g_weight.sum(axis=1) == 0] = self.padding_item_id
        return {'seq': views[0], 'lengths': [len(history), len(long_term), len(short_term)], 'views': [history, long_term, short_term], 'g_adj': g_adj, 'g_weight': g_weight, 'local': local, 'aliases': aliases, 'items': items, 'graph_seed': seed}

def load_features(path, manifest_path, bundle, expected_policy=None):
    import torch
    metadata = json.loads(Path(manifest_path).read_text())
    if metadata.get('features_sha256') != sha256(path):
        raise GraphError('text-feature file hash differs from its manifest')
    if metadata.get('feature_kind') not in {'official_release', 'llama3_descriptions_sbert_reconstruction'}:
        raise GraphError('UCGRec needs official features or Llama3-description/SBERT features; raw-title substitutions are unsupported')
    if not metadata.get('provenance'):
        raise GraphError('text-feature provenance is required')
    expected_items_hash = bundle.get('source_file_sha256', {}).get('items.json')
    if not expected_items_hash or metadata.get('items_sha256') != expected_items_hash:
        raise GraphError('text features are not bound to the current item metadata bytes')
    if metadata['feature_kind'] == 'llama3_descriptions_sbert_reconstruction':
        if expected_policy and metadata.get('dataset_policy') != expected_policy:
            raise GraphError('description dataset policy differs from recommendation policy')
        descriptions_path = Path(manifest_path).parent / 'descriptions.jsonl'
        if metadata.get('descriptions_sha256') != sha256(descriptions_path):
            raise GraphError('generated descriptions differ from feature provenance')
        descriptions = [json.loads(line) for line in descriptions_path.read_text().splitlines() if line]
        if [row.get('item_id') for row in descriptions] != metadata.get('item_ids'):
            raise GraphError('description item order differs from feature matrix rows')
        for row in descriptions:
            item = bundle['items'].get(row['item_id'])
            if item is None or row.get('title') != item['title'] or (not isinstance(row.get('description'), str)) or (not row['description'].strip()):
                raise GraphError('description title/content differs from current item metadata')
    ids = metadata.get('item_ids')
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or any((not isinstance(item, str) for item in ids)):
        raise GraphError('feature manifest requires unique original item IDs in matrix row order')
    features = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] != len(ids):
        raise GraphError('feature matrix dimensions do not match the manifest')
    if not torch.isfinite(features).all() or features.shape[1] < 1:
        raise GraphError('text features must be finite nonempty vectors')
    row_by_id = {item: index for index, item in enumerate(ids)}
    original_ids = bundle['mapping']['index_to_item'][1:]
    if any((item not in row_by_id for item in original_ids)):
        raise GraphError('text features do not cover the complete common catalog')
    matrix = torch.stack([features[row_by_id[item]].float() for item in original_ids])
    if (matrix.norm(dim=1) == 0).any():
        raise GraphError('real catalog items cannot have zero content features')
    return (torch.cat([matrix, torch.zeros(1, matrix.shape[1])]), metadata)
