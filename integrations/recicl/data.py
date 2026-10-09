import json


def title(items, item):
    value = items[item]['title']
    if '###' in value or '<' in value or '>' in value:
        raise ValueError('title conflicts with native prompt delimiter/placeholder syntax')
    return json.dumps(value, ensure_ascii=False)


def history_title(items, history):
    return ', '.join(title(items, item) for item in history) if history else 'unkow'


def prompt_template(count, item_kind):
    pieces = ['Recommend a ' + item_kind + ' using the user\'s interaction history. Answer Yes or No.']
    for index in range(1, count + 1):
        pieces.append('#Example ' + str(index) + '\nHistory: <ItemTitleList' + str(index) + '>\nQuestion: Would the user like <TargetItemTitle' + str(index) + '>?\nAnswer: <Response' + str(index) + '>')
    index = count + 1
    pieces.append('#Current query\nHistory: <ItemTitleList' + str(index) + '>\nQuestion: Would the user like <TargetItemTitle' + str(index) + '>?\nAnswer:')
    return '\n\n'.join(pieces)


def segments(row, candidate, label, demonstrations, history_length):
    history = row['history']
    times = row['history_timestamps']
    if not history or len(times) != len(history) or any(t >= row['timestamp'] for t in times) or times != sorted(times):
        raise ValueError('strictly prior aligned history required')
    past = []
    for index in range(1, len(history)):
        earlier = [history[j] for j in range(index) if times[j] < times[index]]
        if earlier:
            past.append({'history': earlier[-history_length:], 'candidate': history[index], 'label': 1, 'timestamp': times[index]})
    return past[-demonstrations:] + [{'history': history[-history_length:], 'candidate': candidate, 'label': label, 'timestamp': row['timestamp']}] if demonstrations else [{'history': history[-history_length:], 'candidate': candidate, 'label': label, 'timestamp': row['timestamp']}]


def samples(row, candidate, label, items, demonstrations, history_length, item_kind, device):
    import torch
    values = segments(row, candidate, label, demonstrations, history_length)
    template = prompt_template(len(values) - 1, item_kind)
    sample = {'UserID': torch.zeros((1, len(values)), dtype=torch.long, device=device), 'InteractedItemTitles': ['###'.join(history_title(items, x['history']) for x in values)], 'TargetItemTitle': ['###'.join(title(items, x['candidate']) for x in values)], 'InteractedNum': torch.tensor([[len(x['history']) for x in values]], device=device), 'label': torch.tensor([[x['label'] for x in values]], device=device)}
    rendered = template
    for index, value in enumerate(values, 1):
        rendered = rendered.replace('<ItemTitleList' + str(index) + '>', history_title(items, value['history'])).replace('<TargetItemTitle' + str(index) + '>', title(items, value['candidate']))
        if index < len(values):
            rendered = rendered.replace('<Response' + str(index) + '>', 'Yes' if value['label'] else 'No')
    return sample, template, rendered
