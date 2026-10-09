from dataclasses import dataclass
import json
from .retrieval import validate_query, timestamp


@dataclass(frozen=True)
class PreparedPrompt:
    text: str
    neighbor_spans: tuple
    neighbor_scores: tuple
    candidate_ids: tuple
    candidate_titles: tuple
    query_id: str
    user_id: str
    history_length: int


def build_prompt(query, items, neighbors, config, expected_candidate_count=None):
    expected = config.expected_candidate_count if expected_candidate_count is None else expected_candidate_count
    if expected not in {config.training_candidate_count, config.expected_candidate_count}:
        raise ValueError('prompt candidate count differs from pinned protocols')
    validate_query(query, items, expected)
    if isinstance(neighbors, dict):
        neighbors = neighbors['neighbors']
    text = config.instruction + '\n\nSimilar users:\n'
    spans, scores = [], []
    for position, neighbor in enumerate(neighbors, 1):
        if neighbor['next_timestamp'] >= timestamp(query) or any(value >= timestamp(query) for value in neighbor['history_timestamps']):
            raise ValueError('neighbor prompt contains noncausal evidence')
        history = [items[item]['title'] for item in neighbor['history']]
        target = items[neighbor['next_item_id']]['title']
        segment = f'User {position} consumed {json.dumps(history, ensure_ascii=False)}; then chose {json.dumps(target, ensure_ascii=False)}.\n'
        spans.append((len(text), len(text) + len(segment)))
        scores.append(neighbor['score'])
        text += segment
    history = query['history'][-config.target_history_limit:] if config.target_history_limit else query['history']
    text += '\nTarget user history: ' + json.dumps([items[item]['title'] for item in history], ensure_ascii=False)
    titles = tuple(items[item]['title'] for item in query['candidate_ids'])
    text += '\nCandidates: ' + json.dumps(titles, ensure_ascii=False)
    text += '\n' + config.response_prefix
    return PreparedPrompt(text, tuple(spans), tuple(scores), tuple(query['candidate_ids']), titles, query['query_id'], query['user_id'], len(query['history']))
