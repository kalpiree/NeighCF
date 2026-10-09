import copy
import hashlib
import re
from types import SimpleNamespace
import numpy as np

from integrations.drdt.source import load_methods
from integrations.neighcf.retrieval import timestamp, validate_query


DOMAINS = ('ml-1m', 'Games', 'Books', 'Magazine', 'Luxury', 'artists-native-books-adaptation-v1')


def query_seed(seed, query_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{query_id}'.encode()).digest()[:4], 'big')


def select_demonstration(query, training, items, expected_candidates):
    validate_query(query, items, expected_candidates)
    eligible = []
    for row in training:
        validate_query(row, items)
        if row.get('split') not in (None, 'train'):
            raise ValueError('demonstration rows must be training examples')
        if row['user_id'] == query['user_id'] or timestamp(row) >= timestamp(query):
            continue
        if row['history'][-1] == query['history'][-1] and len(row['candidate_ids']) >= expected_candidates:
            eligible.append(row)
    if not eligible:
        raise ValueError('no strictly earlier other-user training demonstration with the same final history item')
    return min(eligible, key=lambda row: (row['query_id'], timestamp(row)))


def parse_ranking(text, candidate_ids, items):
    if not isinstance(text, str):
        raise ValueError('ranking response must be text')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != len(candidate_ids):
        raise ValueError('DRDT response must rank every candidate exactly once')
    order = []
    for line in lines:
        matches = [index for index, item in enumerate(candidate_ids) if line == items[item]['title']]
        match = re.fullmatch(r'(\d+)(?:[.)]\s*(.*))?', line)
        if match:
            position = int(match.group(1))
            suffix = match.group(2)
            if position >= len(candidate_ids) or (suffix and suffix != items[candidate_ids[position]]['title']):
                raise ValueError('DRDT response candidate index/title mismatch')
            matches = [position]
        if len(matches) != 1 or matches[0] in order:
            raise ValueError('DRDT response has unknown, ambiguous or repeated candidates')
        order.append(matches[0])
    scores = [0.0] * len(candidate_ids)
    for rank, position in enumerate(order):
        scores[position] = float(len(candidate_ids) - rank)
    return scores


def rank_query(source, items, training, query, complete, *, domain, seed, reflection_steps=3,
               history_limit=20, expected_candidates=51, model_name='configured-backend'):
    if domain not in DOMAINS:
        raise ValueError('native DRDT has no prompt for this domain; provide a separately qualified domain adaptation')
    if type(reflection_steps) is not int or not 1 <= reflection_steps <= 20:
        raise ValueError('reflection_steps must be in 1..20')
    if type(history_limit) is not int or not reflection_steps < history_limit <= 1000:
        raise ValueError('history_limit must exceed reflection_steps and be at most1000')
    validate_query(query, items, expected_candidates)
    history = query['history'][-history_limit:]
    if len(history) <= reflection_steps:
        raise ValueError('DRDT needs a nonempty prefix before its reflection steps')
    demo = select_demonstration(query, training, items, expected_candidates)
    rng = np.random.RandomState(query_seed(seed, query['query_id']))
    artist_adaptation = domain == 'artists-native-books-adaptation-v1'
    methods = load_methods(source, rng, artist_adaptation=artist_adaptation)
    ids = sorted(items)
    positions = {item: index + 1 for index, item in enumerate(ids)}
    calls = []
    call_limit = 2 * reflection_steps + 2

    def create(*, model, messages):
        if len(calls) >= call_limit:
            raise ValueError('native DRDT exceeded its configured call budget')
        request = copy.deepcopy(messages)
        value = complete(request)
        result = {'text': value} if isinstance(value, str) else value
        if not isinstance(result, dict) or not isinstance(result.get('text'), str) or not result['text'].strip():
            raise ValueError('DRDT backend returned empty or malformed text')
        if result.get('finish_reason', 'stop') != 'stop':
            raise ValueError('DRDT response ended without a normal stop')
        calls.append({'messages': request, 'response': result})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result['text']))])

    native = SimpleNamespace(reflex_step=reflection_steps, recall_budget=expected_candidates,
                             model_name=model_name, model_name_long=model_name,
                             item_text=['[PAD]'] + [items[item]['title'] for item in ids],
                             client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    negatives = [item for item in demo['candidate_ids'] if item != demo['target_id']][:expected_candidates - 1]
    example_input = (demo['user_id'], [[positions[item]] for item in demo['history'][-history_limit:]],
                     min(history_limit, len(demo['history'])), positions[demo['target_id']],
                     [positions[item] for item in negatives])
    example = methods['get_batch_examples'](native, [[example_input]], 0)[0]
    candidates = [f'{index}. {items[item]["title"]}' for index, item in enumerate(query['candidate_ids'])]
    result, transcript = methods['inference_COT_reflex_long'](
        native, 'Books' if artist_adaptation else domain, [items[item]['title'] for item in history], candidates, example)
    if len(calls) != call_limit or transcript[-1]['content'] != result['rank']:
        raise ValueError('native DRDT conversation did not complete its exact reflection budget')
    return {'query_id': query['query_id'], 'user_id': query['user_id'], 'candidate_ids': query['candidate_ids'],
            'target_id': query['target_id'], 'scores': parse_ranking(result['rank'], query['candidate_ids'], items),
            'demonstration_query_id': demo['query_id'], 'calls': calls,
            'native_random_seed': query_seed(seed, query['query_id'])}
