from bisect import bisect_left
import hashlib
import json
import math
import time
import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def timestamp(row):
    if 'timestamp' in row:
        value = row['timestamp']
        if 'timestamp_unix_s' in row and value != row['timestamp_unix_s'] * 1000:
            raise ValueError('timestamp units disagree')
    elif 'timestamp_unix_s' in row:
        value = row['timestamp_unix_s'] * 1000
    else:
        raise ValueError('timestamp or timestamp_unix_s is required')
    if type(value) is not int or value < 0:
        raise ValueError('timestamp must be a nonnegative integer in milliseconds')
    return value


def validate_query(query, items, expected_candidate_count=None):
    for field in ('query_id', 'user_id', 'target_id'):
        if not isinstance(query.get(field), str) or not query[field]:
            raise ValueError(f'{field} must be a nonempty string')
    for field in ('history', 'candidate_ids'):
        values = query.get(field)
        if not isinstance(values, list) or not values or any(value not in items for value in values):
            raise ValueError(f'{field} must be nonempty catalog item IDs')
    if len(set(query['candidate_ids'])) != len(query['candidate_ids']) or query['target_id'] not in query['candidate_ids']:
        raise ValueError('candidate IDs must be unique and contain target')
    if expected_candidate_count is not None and len(query['candidate_ids']) != expected_candidate_count:
        raise ValueError('candidate count differs from configured protocol')
    times = query.get('history_timestamps')
    if not isinstance(times, list) or len(times) != len(query['history']):
        raise ValueError('history_timestamps must align with history')
    now = timestamp(query)
    if any(type(value) is not int or value < 0 or value >= now for value in times) or times != sorted(times):
        raise ValueError('history must be chronological and strictly earlier than query')


class NeighCFRetriever:
    def __init__(self, items, item_embeddings, centres, events, config):
        self.config = config
        self.items = items
        if not isinstance(items, dict) or not items:
            raise ValueError('items must be a nonempty object')
        for key, value in items.items():
            if not isinstance(key, str) or not key or not isinstance(value, dict) or not isinstance(value.get('title'), str) or not value['title'].strip():
                raise ValueError('catalog requires string IDs and nonempty titles')
        if set(item_embeddings) != set(items):
            raise ValueError('frozen item embeddings must exactly cover catalog')
        self.item_ids = sorted(items)
        self.item_positions = {item: pos for pos, item in enumerate(self.item_ids)}
        embeddings = np.asarray([item_embeddings[item] for item in self.item_ids], dtype=np.float64)
        centres_array = np.asarray(centres, dtype=np.float64)
        if embeddings.ndim != 2 or centres_array.ndim != 2 or embeddings.shape[1] != centres_array.shape[1] or not embeddings.shape[1] or not len(centres_array):
            raise ValueError('embeddings and centres need matching nonempty feature dimensions')
        if not np.isfinite(embeddings).all() or not np.isfinite(centres_array).all():
            raise ValueError('embeddings and centres must be finite')
        self.cluster_count = len(centres_array)
        if config.active_user_clusters > self.cluster_count or (config.index_top_clusters or 0) > self.cluster_count:
            raise ValueError('cluster selection cannot exceed supplied centres')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            distances = ((embeddings[:, None, :] - centres_array[None, :, :]) ** 2).sum(axis=-1)
            logits = -distances / (config.sigma ** 2)
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            self.memberships = probabilities / probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(self.memberships).all():
            raise ValueError('nonfinite memberships')
        self.user_events = {}
        normalized_events = []
        seen = set()
        for event in events:
            if event.get('split') != 'train':
                raise ValueError('neighbor event inputs must explicitly declare split=train')
            user, item, moment = event.get('user_id'), event.get('item_id'), timestamp(event)
            if not isinstance(user, str) or not user or item not in items:
                raise ValueError('invalid neighbor event user or item')
            identity = (user, moment, item)
            if identity in seen:
                raise ValueError('duplicate neighbor event; deduplicate explicitly before retrieval')
            seen.add(identity)
            self.user_events.setdefault(user, []).append((moment, item))
            normalized_events.append(identity)
        for user, history in self.user_events.items():
            history.sort(key=lambda pair: (pair[0], pair[1]))
        self.event_times = {user: [pair[0] for pair in history] for user, history in self.user_events.items()}
        self.source_identity = digest({'items': items, 'embeddings': item_embeddings, 'centres': centres_array.tolist(), 'events': sorted(normalized_events)})

    def profile(self, history):
        return self.memberships[[self.item_positions[item] for item in history]].mean(axis=0)

    def threshold(self, query):
        cfg = self.config
        if cfg.threshold_policy == 'fixed':
            return cfg.threshold_value
        if cfg.threshold_policy == 'per_user':
            if query['user_id'] not in cfg.user_thresholds:
                raise ValueError('missing explicit per-user threshold')
            return cfg.user_thresholds[query['user_id']]
        value = cfg.activity_threshold_intercept + cfg.activity_threshold_slope * math.log1p(len(query['history']))
        return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))

    def retrieve(self, query, variant='full', soft_context=False, expected_candidate_count=None):
        start = time.perf_counter()
        expected = self.config.expected_candidate_count if expected_candidate_count is None else expected_candidate_count
        if expected not in {self.config.training_candidate_count, self.config.expected_candidate_count}:
            raise ValueError('candidate count is not one of the pinned training/evaluation protocols')
        validate_query(query, self.items, expected)
        if variant not in {'full', 'without_collaborative', 'fixed_neighbors', 'global_pool'}:
            raise ValueError('unknown retrieval variant')
        cfg = self.config
        now = timestamp(query)
        target = self.profile(query['history'])
        top = np.argsort(-target, kind='stable')[:cfg.active_user_clusters]
        candidate_profiles = self.memberships[[self.item_positions[item] for item in query['candidate_ids']]]
        active = sorted(set(top.tolist()) | set(np.flatnonzero((candidate_profiles > cfg.candidate_cluster_threshold).any(axis=0)).tolist()))
        eligible = []
        insufficient = 0
        for user in sorted(self.user_events):
            if user == query['user_id']:
                continue
            end = bisect_left(self.event_times[user], now)
            if end < 2:
                insufficient += 1
                continue
            sequence = self.user_events[user][:end]
            profile = self.profile([item for _, item in sequence])
            eligible.append((user, sequence, profile))
        pool = []
        for user, sequence, profile in eligible:
            if variant == 'global_pool' or cfg.index_policy == 'literal_positive':
                included = True
            elif cfg.index_policy == 'user_top_clusters':
                included = bool(set(np.argsort(-profile, kind='stable')[:cfg.index_top_clusters].tolist()) & set(active))
            else:
                included = bool((profile[active] > cfg.index_mass_threshold).any())
            if included:
                left, right = target[active], profile[active]
                denominator = np.linalg.norm(left) * np.linalg.norm(right)
                cosine = float(np.dot(left, right) / denominator) if denominator else 0.0
                overlap = float(np.minimum(left, right).sum())
                similarity = math.sqrt(max(0.0, min(1.0, cosine)) * max(0.0, min(1.0, overlap)))
                pool.append({'user_id': user, 'score': similarity, 'cosine': cosine, 'shared_mass': overlap, 'history': [item for _, item in sequence[:-1]][-cfg.neighbor_history_limit:], 'history_timestamps': [time for time, _ in sequence[:-1]][-cfg.neighbor_history_limit:], 'next_item_id': sequence[-1][1], 'next_timestamp': sequence[-1][0], 'profile_event_count': len(sequence)})
        pool.sort(key=lambda row: (-row['score'], row['user_id']))
        tau = self.threshold(query)
        if variant == 'without_collaborative':
            neighbors = []
        elif soft_context:
            neighbors = pool[:cfg.soft_context_k]
        elif variant == 'fixed_neighbors':
            neighbors = pool[:cfg.fixed_neighbor_k]
        elif cfg.neighbor_k_mode == 'topk_override':
            neighbors = pool[:cfg.neighbor_k]
        else:
            neighbors = [row for row in pool if row['score'] >= tau]
            if cfg.neighbor_k_mode == 'after_threshold':
                neighbors = neighbors[:cfg.neighbor_k]
        return {'neighbors': neighbors, 'pool_scores': [row['score'] for row in pool], 'active_clusters': active, 'threshold': tau, 'variant': variant, 'soft_context': soft_context, 'eligible_users': len(eligible), 'pool_users': len(pool), 'insufficient_causal_history_users': insufficient, 'index_policy': cfg.index_policy, 'literal_index_is_full_pool': cfg.index_policy == 'literal_positive', 'selection': 'fixed_score_ranked_context_before_learned_gate' if soft_context else 'hard_text_selection', 'retrieval_seconds': time.perf_counter() - start, 'source_identity': self.source_identity}
