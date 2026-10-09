import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))
from adapters.llara_data import LlaraData, LlaraDataError
from adapters.llmsrec_data import load_canonical, index_queries

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def evaluation_contract(expected_candidates):
    if type(expected_candidates) is not int or expected_candidates < 2:
        raise LlaraDataError('evaluation requires one true item and at least one negative')
    return {'candidate_count': expected_candidates, 'negative_count': expected_candidates - 1, 'positive_count': 1, 'published_manuscript_candidate_count': 100, 'origin': 'user_override_50_negatives' if expected_candidates == 51 else 'published_manuscript' if expected_candidates == 100 else 'explicit_cli_configuration', 'changes_manuscript_candidate_count': expected_candidates != 100}

def load_bundle(path, seed=42, max_history=10, teacher_max_length=128, train_candidates=20, expected_candidates=51):
    evaluation_contract(expected_candidates)
    items, rows, metadata, hashes = load_canonical(Path(path), expected_candidates)
    mapping, indexed = index_queries(items, rows, teacher_max_length)
    del indexed
    datasets = {split: LlaraData(path, split, max_history=max_history, train_candidate_count=train_candidates, train_candidate_seed=seed, loaded=(items, rows, metadata, hashes)) for split in ('train', 'valid', 'test')}
    marker_strings = ('[PAD]', '[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]', '<s>', '</s>')
    if any((marker in value['title'] for value in items.values() for marker in marker_strings)):
        raise LlaraDataError('catalog contains a reserved tokenizer/embedding marker')
    protocol = metadata.get('protocol_sha256')
    if not isinstance(protocol, str) or not re.fullmatch('[0-9a-fA-F]{64}', protocol):
        raise LlaraDataError('splits.json requires the common protocol_sha256')
    teacher_contract = {'mapping_sha256': digest(mapping), 'train_sha256': hashes['train.jsonl'], 'max_length': teacher_max_length, 'hidden_units': 64, 'item_count': len(items), 'user_count': len(mapping['user_to_index']), 'objective': 'official SASRec BCE, canonical last-position target once per query'}
    return {'datasets': datasets, 'items': items, 'source_sha256': hashes, 'protocol_sha256': protocol.lower(), 'teacher_contract': teacher_contract, 'mapping': datasets['train'].mapping_manifest(), 'split_metadata': metadata, 'expected_candidates': expected_candidates}

def teacher_row_order(bundle):
    count = len(bundle['items'])
    return list(range(1, count + 1)) + [0]

def validate_teacher(bundle, saved, *, require_final=False):
    if saved.get('contract') != bundle['teacher_contract']:
        raise LlaraDataError('teacher catalog/user map, split, dimension or history contract differs')
    if saved.get('validation_sha256') != bundle['source_sha256']['valid.jsonl']:
        raise LlaraDataError('teacher was selected on another validation split')
    if saved.get('protocol_sha256') != bundle['protocol_sha256']:
        raise LlaraDataError('teacher protocol differs')
    if require_final and saved.get('diagnostic', True):
        raise LlaraDataError('full training requires a completed validation-selected teacher')

def batches(size, batch_size, seed=None):
    if size < 1 or batch_size < 1:
        raise LlaraDataError('batch size and dataset size must be positive')
    indices = list(range(size))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    return [indices[start:start + batch_size] for start in range(0, size, batch_size)]

def training_plan(seed, step, total_steps, templates):
    if total_steps < 1 or not 0 <= step < total_steps:
        raise LlaraDataError('invalid curriculum step')
    rng = random.Random(digest([seed, step, 'llara-curriculum']))
    return {'template_index': rng.randrange(len(templates)), 'hybrid': rng.random() < (step + 1) / total_steps, 'hybrid_probability': (step + 1) / total_steps}

def render_prompt(sample, template, hybrid=True):
    if template.count('[HistoryHere]') != 1 or template.count('[CansHere]') != 1:
        raise LlaraDataError('template must have one history and one candidate placeholder')
    if any((marker in template for marker in ('[ItemEmb]', '[HistoryEmb]', '[CansEmb]', '[PH]'))):
        raise LlaraDataError('template must not supply embedding markers or target placeholders')
    history_marker, candidate_marker = ('[HistoryEmb]', '[CansEmb]') if hybrid else ('[PH]', '[PH]')
    history = ', '.join((title + ' ' + history_marker for title in sample['seq_name']))
    candidates = ', '.join((title + ' ' + candidate_marker for title in sample['cans_name']))
    template = re.sub('following\\s+\\d+\\s+', f"following {sample['len_cans']} ", template)
    return template.replace('[HistoryHere]', history).replace('[CansHere]', candidates)

def token_example(tokenizer, sample, template, hybrid, answer=None, max_tokens=2048):
    prompt = render_prompt(sample, template, hybrid)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_text = sample['correct_answer'] if answer is None else answer
    answer_ids = tokenizer.encode(answer_text + '\n', add_special_tokens=False)
    if not prompt_ids or not answer_ids or len(prompt_ids) + len(answer_ids) > max_tokens:
        raise LlaraDataError(f"{sample['query_id']}: empty or over-budget token sequence; no silent truncation")
    return {'sample': sample, 'prompt_ids': prompt_ids, 'answer_ids': answer_ids, 'input_ids': prompt_ids + answer_ids, 'labels': [-100] * len(prompt_ids) + answer_ids, 'answer_tokens': len(answer_ids), 'hybrid': hybrid}

def prediction_record(sample, scores):
    if len(scores) != sample['len_cans']:
        raise LlaraDataError('one score per real frozen candidate is required')
    if any((type(score) not in (float, int) or not math.isfinite(score) for score in scores)):
        raise LlaraDataError('candidate scores must be finite numbers')
    return {key: sample[key] for key in ('query_id', 'user_id', 'candidate_ids', 'target_id')} | {'scores': list(scores)}

def cosine_lr(step, total_steps, warmup_steps, peak, start, minimum):
    if total_steps < 1 or not 0 <= step < total_steps or (not 0 <= warmup_steps <= total_steps):
        raise LlaraDataError('invalid scheduler step')
    if warmup_steps and step < warmup_steps:
        return start + (peak - start) * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * min(1, max(0, progress))))

def loss_scale(answer_token_count, optimizer_batch_token_count):
    if not 0 < answer_token_count <= optimizer_batch_token_count:
        raise LlaraDataError('invalid causal-loss denominator')
    return answer_token_count / optimizer_batch_token_count
