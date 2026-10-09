from __future__ import annotations
import hashlib
import json
POLICY = 'ucgrec-single-final-object-brace-v1'
MAX_NEW_TOKENS = 512
EOT_ID = 128009
EXPECTED_TERMINATORS = (128009, 128001)
JSON_WHITESPACE = ' \t\r\n'

class FormatRefused(ValueError):
    pass

def _require(condition, message):
    if not condition:
        raise FormatRefused(message)

def _text_sha(text):
    try:
        return hashlib.sha256(text.encode('utf-8', errors='strict')).hexdigest()
    except UnicodeEncodeError as exc:
        raise FormatRefused('invalid Unicode scalar in text') from exc

def _skip_ws(text, offset):
    while offset < len(text) and text[offset] in JSON_WHITESPACE:
        offset += 1
    return offset

def _one_complete_string(text):
    decoder = json.JSONDecoder(strict=True)
    pos = _skip_ws(text, 0)
    _require(pos < len(text) and text[pos] == '{', 'single JSON object required')
    pos = _skip_ws(text, pos + 1)
    _require(pos < len(text) and text[pos] == '"', 'quoted description key required')
    try:
        key, pos = decoder.raw_decode(text, pos)
    except json.JSONDecodeError as exc:
        raise FormatRefused('malformed JSON key') from exc
    _require(key == 'description', 'sole description key required')
    pos = _skip_ws(text, pos)
    _require(pos < len(text) and text[pos] == ':', 'description colon required')
    pos = _skip_ws(text, pos + 1)
    _require(pos < len(text) and text[pos] == '"', 'complete JSON string value required')
    try:
        description, end = decoder.raw_decode(text, pos)
    except json.JSONDecodeError as exc:
        raise FormatRefused('incomplete or malformed JSON string value') from exc
    _require(isinstance(description, str) and bool(description.strip()), 'nonempty description required')
    _require(len(description.strip().split()) <= 200, 'description exceeds 200 words')
    _text_sha(description)
    tail = _skip_ws(text, end)
    if tail == len(text):
        return (description, True)
    _require(text[tail] == '}' and _skip_ws(text, tail + 1) == len(text), 'only one missing final object brace is eligible; extra content refused')
    return (description, False)

def normalize_description(raw, *, generated_token_ids, decode, max_new_tokens=MAX_NEW_TOKENS, terminator_ids=EXPECTED_TERMINATORS):
    _require(type(raw) is str, 'raw generation must be a string')
    _require(type(max_new_tokens) is int and max_new_tokens == MAX_NEW_TOKENS, 'qualified 512-token generation cap required')
    _require(tuple(terminator_ids) == EXPECTED_TERMINATORS, 'qualified original terminators required')
    _require(isinstance(generated_token_ids, (list, tuple)) and 0 < len(generated_token_ids) <= max_new_tokens and all((type(token) is int and 0 <= token < 128256 for token in generated_token_ids)), 'bounded original Llama3 token IDs required')
    ids = list(generated_token_ids)
    decoded = decode(ids, skip_special_tokens=True)
    _require(type(decoded) is str and decoded.strip() == raw, 'raw generation differs from the same token sequence decoded by the pinned tokenizer')
    original_hash = _text_sha(raw)
    description, missing_brace = _one_complete_string(raw)
    if missing_brace:
        _require(ids[-1] == EOT_ID and len(ids) < max_new_tokens, 'missing-brace correction requires qualified end-of-turn before token cap')
    normalized = raw + '}' if missing_brace else raw
    parsed = json.loads(normalized)
    _require(type(parsed) is dict and set(parsed) == {'description'} and (parsed['description'] == description), 'normalized strict JSON or exact description equality failed')
    _require(normalized == raw or normalized == raw + '}', 'nonterminal normalization refused')
    description_hash = _text_sha(description)
    normalized_description_hash = _text_sha(parsed['description'])
    _require(description_hash == normalized_description_hash, 'description text changed')
    return {'raw_generation': raw, 'normalized_generation': normalized, 'description_verbatim': description, 'strict_parser_description': description.strip(), 'receipt': {'policy': POLICY, 'action': 'append_one_final_object_brace' if missing_brace else 'unchanged_valid_json', 'appended_text': '}' if missing_brace else '', 'raw_sha256': original_hash, 'normalized_sha256': _text_sha(normalized), 'description_sha256_before': description_hash, 'description_sha256_after': normalized_description_hash, 'strict_parser_description_sha256': _text_sha(description.strip()), 'generated_token_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(), 'generated_token_count': len(ids), 'final_token_id': ids[-1], 'max_new_tokens': max_new_tokens, 'terminator_ids': list(terminator_ids), 'raw_decode_verified': True, 'description_word_count': len(description.strip().split())}}
