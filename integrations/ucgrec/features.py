import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from graph_data import sha256, dataset_policy, DATASET_POLICIES
from normalization import normalize_description, POLICY, EOT_ID, MAX_NEW_TOKENS, EXPECTED_TERMINATORS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resume import RestartManager, digest, runtime_context

def model_manifest(path):
    return {str(file.relative_to(path)): sha256(file) for file in sorted(path.rglob('*')) if file.is_file() and file.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt', '.pt'}}

def parse_description(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError('generated response must be a JSON object') from exc
    if not isinstance(value, dict) or set(value) != {'description'}:
        raise ValueError('generated response must have only a description field')
    description = value['description']
    if not isinstance(description, str) or not description.strip():
        raise ValueError('generated description must be a nonempty string')
    description = description.strip()
    if len(description.split()) > 200:
        raise ValueError("generated description exceeds the paper's 200-word limit")
    return description

def normalization_contract():
    return {'policy': POLICY, 'source_sha256': sha256(Path(__file__).with_name('normalization.py')), 'qualified_eot_id': EOT_ID, 'max_new_tokens': MAX_NEW_TOKENS, 'terminator_ids': list(EXPECTED_TERMINATORS), 'raw_decode_rule': 'tokenizer.decode(ids, skip_special_tokens=True).strip()', 'description_policy': 'preserve exact JSON string; original parser strip unchanged'}

def exact_json(left, right):
    return digest(left) == digest(right)

def validate_normalization_tokenizer(tokenizer):
    if not tokenizer.chat_template or tokenizer.get_vocab().get('<|eot_id|>') != EOT_ID:
        raise ValueError('format policy requires the pinned Llama3 chat end-of-turn token')

def validate_description_record(row, tokenizer, *, expected_item_id=None, expected_title=None):
    validate_normalization_tokenizer(tokenizer)
    if not isinstance(row, dict):
        raise ValueError('description journal row must be an object')
    required = {'item_id', 'title', 'description', 'raw_generation', 'normalized_generation', 'generated_token_ids', 'normalization_receipt'}
    if not required <= row.keys() or (expected_item_id is not None and row['item_id'] != expected_item_id) or (expected_title is not None and row['title'] != expected_title) or ('query_id' in row and row['query_id'] != row['item_id']):
        raise ValueError('description journal identity or format fields differ')
    result = normalize_description(row['raw_generation'], generated_token_ids=row['generated_token_ids'], decode=tokenizer.decode)
    if row['normalized_generation'] != result['normalized_generation'] or not exact_json(row['normalization_receipt'], result['receipt']):
        raise ValueError('description normalization text or receipt differs')
    if row['description'] != parse_description(result['normalized_generation']):
        raise ValueError('description differs from unchanged strict parser')
    return result['receipt']['action']

def validate_description_records(rows, tokenizer, items, item_ids, *, require_complete=True):
    if len(rows) > len(item_ids) or (require_complete and len(rows) != len(item_ids)):
        raise ValueError('description journal catalog coverage differs')
    counts = {'unchanged_valid_json': 0, 'append_one_final_object_brace': 0}
    for row, item in zip(rows, item_ids):
        action = validate_description_record(row, tokenizer, expected_item_id=item, expected_title=items[item]['title'])
        counts[action] += 1
    return counts

def generation_terminators(tokenizer, model):
    values = []
    for eos in (tokenizer.eos_token_id, getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)):
        if eos is None:
            continue
        values.extend(eos if isinstance(eos, (list, tuple)) else [eos])
    if tokenizer.chat_template:
        vocabulary = tokenizer.get_vocab()
        eot = vocabulary.get('<|eot_id|>')
        if type(eot) is not int:
            raise ValueError('Llama3 chat tokenizer must declare its actual <|eot_id|> token')
        values.append(eot)
    if not values or any((type(value) is not int or value < 0 for value in values)):
        raise ValueError('the staged tokenizer/model must declare valid generation terminator IDs')
    return list(dict.fromkeys(values))

def declared_model_source(args, prefix):
    model_id, revision = (getattr(args, prefix + '_id'), getattr(args, prefix + '_revision'))
    if bool(model_id) != bool(revision) or (revision and (not re.fullmatch('[0-9a-f]{40}', revision))):
        raise ValueError('a declared model source requires both its repository ID and exact 40-character commit')
    return None if not model_id else {'repository_id': model_id, 'revision': revision, 'verification': 'declared acquisition identity; the staged asset receipt verifies origin and local file hashes bind runtime bytes'}

def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def atomic_json(path, value):
    atomic_write(path, lambda stream: stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()))

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--description-model-path', type=Path, required=True)
    p.add_argument('--encoder-path', type=Path, required=True)
    p.add_argument('--description-model-id')
    p.add_argument('--description-model-revision')
    p.add_argument('--encoder-model-id')
    p.add_argument('--encoder-model-revision')
    p.add_argument('--description-prompt-file', type=Path, required=True)
    p.add_argument('--prompt-origin', required=True, choices=('paper_fig2_transcription', 'documented_reconstruction'))
    p.add_argument('--dataset-policy', required=True, choices=tuple(DATASET_POLICIES))
    p.add_argument('--prompt-template-version', default='dataset-description-v1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--encoder-batch-size', type=int, default=32)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-seconds', type=float, default=600.0)
    return p

def main():
    p = parser()
    args = p.parse_args()
    if args.output.exists() and (not args.resume):
        p.error('output already exists; preserve it or explicitly resume its matching checkpoint')
    if args.resume and (not args.output.is_dir()):
        p.error('resume requires an existing feature output')
    if min(args.max_new_tokens, args.encoder_batch_size) < 1 or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0:
        p.error('positive generation/batch/checkpoint settings are required')
    if args.max_new_tokens != MAX_NEW_TOKENS or args.seed != 42:
        p.error('this explicit format policy requires the qualified seed42 and 512-token generation cap')
    policy = dataset_policy(args.dataset_policy)
    sources = {'description_model': declared_model_source(args, 'description_model'), 'encoder_model': declared_model_source(args, 'encoder_model')}
    config = json.loads((args.description_model_path / 'config.json').read_text())
    if (config.get('model_type'), config.get('hidden_size'), config.get('num_hidden_layers'), config.get('vocab_size')) != ('llama', 4096, 32, 128256):
        p.error('description generator must be the supplied local Llama3-8B architecture')
    if not args.encoder_path.is_dir():
        p.error('local Sentence-BERT checkpoint directory is required')
    template = args.description_prompt_file.read_text()
    if template.count('{title}') != 1 or not template.strip():
        p.error('description prompt must contain exactly one {title} placeholder')
    if policy['item_noun'] not in template.lower():
        p.error("description prompt must explicitly describe the selected dataset's item type")
    if args.dataset_policy != 'movielens_native_v1' and args.prompt_origin == 'paper_fig2_transcription':
        p.error('nonmovie prompts are controlled dataset adaptations, not the movie Figure2 transcription')
    items_path = args.data / 'items.json'
    items = json.loads(items_path.read_text())
    if not isinstance(items, dict) or not items or any((not isinstance(item, str) or not isinstance(value, dict) or (not isinstance(value.get('title'), str)) or (not value['title'].strip()) for item, value in items.items())):
        p.error('items must map original item IDs to nonempty titles')
    import torch
    import numpy as np
    description_files, encoder_files = (model_manifest(args.description_model_path), model_manifest(args.encoder_path))
    if not any((Path(name).suffix in {'.bin', '.safetensors', '.pt'} for name in description_files)) or not any((Path(name).suffix in {'.bin', '.safetensors', '.pt'} for name in encoder_files)):
        p.error('both local checkpoints must contain actual model weights')
    scientific_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in {'resume', 'checkpoint_seconds'}}
    import importlib.metadata
    package_versions = {}
    for name in ('torch', 'transformers', 'sentence-transformers', 'numpy'):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    format_contract = normalization_contract()
    contract = {'kind': 'ucgrec-feature-restart-format-v1', 'arguments': scientific_args, 'items_sha256': sha256(items_path), 'item_ids': sorted(items), 'dataset_policy': policy, 'description_model_files': description_files, 'encoder_files': encoder_files, 'declared_model_sources': sources, 'prompt_sha256': sha256(args.description_prompt_file), 'runtime': runtime_context(args.device), 'package_versions': package_versions, 'format_normalization': format_contract, 'source_sha256': {name: sha256(Path(__file__).parent / name) for name in ('features.py', 'graph_data.py', 'normalization.py')}, 'restart_helper_sha256': sha256(Path(__file__).parent.parent / 'resume.py')}
    reference = {'frozen_model_references_sha256': digest(contract)}
    item_ids = sorted(items)
    with RestartManager(args.output, contract, resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:

        def save_reference(path):
            torch.save(reference, path)

        def restore_reference(path):
            if torch.load(path, map_location='cpu', weights_only=True) != reference:
                raise ValueError('frozen description/SBERT checkpoint references changed')
        state = manager.restore(restore_reference)
        if state is None:
            torch.manual_seed(args.seed)
            state = {'phase': 'describe', 'provenance': None, 'descriptions_sha256': None}

        def boundary(force=False):
            return manager.checkpoint(state, save_reference, force=force)
        if not args.resume:
            boundary(force=True)
        if state['phase'] == 'complete':
            metadata = json.loads((args.output / 'features.manifest.json').read_text())
            if digest(metadata) != state['manifest_digest'] or sha256(args.output / 'features.pt') != metadata['features_sha256'] or sha256(args.output / 'descriptions.jsonl') != metadata['descriptions_sha256']:
                raise ValueError('completed feature artifacts changed')
            if not exact_json(metadata.get('format_normalization'), format_contract) or metadata.get('source_sha256') != contract['source_sha256'] or (not exact_json(metadata.get('provenance'), state['provenance'])) or (not exact_json(state['provenance'].get('format_normalization'), format_contract)):
                raise ValueError('completed feature format/source provenance differs')
            if not exact_json(json.loads((args.output / 'generation.manifest.json').read_text()), state['provenance']):
                raise ValueError('completed generation provenance changed')
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.description_model_path), local_files_only=True)
            rows = [json.loads(line) for line in (args.output / 'descriptions.jsonl').read_text().splitlines() if line]
            counts = validate_description_records(rows, tokenizer, items, item_ids)
            if not exact_json(metadata.get('normalization_counts'), counts) or not exact_json(state.get('normalization_counts'), counts):
                raise ValueError('completed normalization counts differ')
            print('Completed feature artifacts verified.', flush=True)
            return
        journal = manager.scoring('descriptions', [[item] for item in item_ids])
        tokenizer = None
        if state['phase'] == 'describe':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.description_model_path), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(str(args.description_model_path), local_files_only=True, torch_dtype=torch.bfloat16, device_map=args.device)
            model.eval().requires_grad_(False)
            terminators = generation_terminators(tokenizer, model)
            validate_normalization_tokenizer(tokenizer)
            if terminators != list(EXPECTED_TERMINATORS):
                raise ValueError('format policy requires the original verified terminator order')
            if args.resume:
                manager.restore(restore_reference)
            provenance = {'paper': 'https://aclanthology.org/2026.findings-acl.175.pdf', 'description_model_path': str(args.description_model_path.resolve()), 'description_model_files': description_files, 'encoder_path': str(args.encoder_path.resolve()), 'encoder_files': encoder_files, 'declared_model_sources': sources, 'prompt': template, 'prompt_origin': args.prompt_origin, 'prompt_sha256': sha256(args.description_prompt_file), 'prompt_template_version': args.prompt_template_version, 'dataset_policy': policy, 'prompt_reference': 'Figure2 movie response format; dataset noun adaptation explicitly recorded', 'response_contract': {'format': 'JSON with a single description field', 'maximum_words': 200, 'word_count_rule': 'whitespace-separated tokens', 'invalid_response_policy': POLICY + '; otherwise fail without repair'}, 'format_normalization': format_contract, 'format_qualification': {'probe_job': 30907, 'observed_eot_id': EOT_ID, 'observed_token_counts': {'ml100k': 177, 'goodreads': 173, 'lastfm-1k-celma': 164}}, 'decoding': {'do_sample': False, 'max_new_tokens': args.max_new_tokens, 'seed': args.seed, 'eos_token_ids': terminators, 'pad_token_id': terminators[0], 'stop_policy': 'union of staged tokenizer/model EOS and verified Llama3 <|eot_id|> when chat template is used', 'tokenizer_add_special_tokens': not bool(tokenizer.chat_template)}, 'uses_checkpoint_chat_template': bool(tokenizer.chat_template), 'encoder_batch_policy': 'global SentenceTransformer text-length descending sort, fixed original batches, restore catalog order', 'encoder_batch_size': args.encoder_batch_size, 'unrecovered_paper_settings': ['exact SBERT checkpoint', 'description decoding settings', 'exact Llama3 variant/revision']}
            if state['provenance'] is not None and (not exact_json(state['provenance'], provenance)):
                raise ValueError('restored generation provenance differs')
            provenance_path = args.output / 'generation.manifest.json'
            if provenance_path.exists():
                if not exact_json(json.loads(provenance_path.read_text()), provenance):
                    raise ValueError('saved generation provenance changed')
            elif state['provenance'] is not None:
                raise ValueError('saved generation provenance is missing')
            else:
                atomic_json(provenance_path, provenance)
            state['provenance'] = provenance
            state['normalization_counts'] = validate_description_records(journal.records(require_complete=False), tokenizer, items, item_ids, require_complete=False)
            boundary(force=True)
            for index in range(journal.completed_batches, len(item_ids)):
                item = item_ids[index]
                prompt = template.replace('{title}', items[item]['title'])
                if tokenizer.chat_template:
                    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=not bool(tokenizer.chat_template)).to(args.device)
                with torch.inference_mode():
                    generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=terminators[0], eos_token_id=terminators)
                generated_ids = generated[0, encoded['input_ids'].shape[1]:].tolist()
                text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                try:
                    normalized = normalize_description(text, generated_token_ids=generated_ids, decode=tokenizer.decode, max_new_tokens=args.max_new_tokens, terminator_ids=terminators)
                    description = parse_description(normalized['normalized_generation'])
                except ValueError as exc:
                    atomic_json(args.output / 'invalid_generation.json', {'item_id': item, 'raw_generation': text, 'generated_token_ids': generated_ids, 'format_normalization': format_contract, 'error': str(exc)})
                    raise RuntimeError(f'invalid generated description for {item}: {exc}') from exc
                journal.append(index, [{'query_id': item, 'item_id': item, 'title': items[item]['title'], 'description': description, 'raw_generation': text, 'normalized_generation': normalized['normalized_generation'], 'generated_token_ids': generated_ids, 'normalization_receipt': normalized['receipt']}])
                state['normalization_counts'][normalized['receipt']['action']] += 1
                boundary()
                print(f'description {index + 1}/{len(item_ids)}', flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            records = journal.records()
            atomic_write(args.output / 'descriptions.jsonl', lambda stream: stream.write(''.join((json.dumps({key: value for key, value in row.items() if key != 'query_id'}, ensure_ascii=False) + '\n' for row in records)).encode()))
            state.update(phase='encode', descriptions_sha256=sha256(args.output / 'descriptions.jsonl'))
            boundary(force=True)
        if sha256(args.output / 'descriptions.jsonl') != state['descriptions_sha256']:
            raise ValueError('completed descriptions changed before encoding')
        records = journal.records()
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.description_model_path), local_files_only=True)
        counts = validate_description_records(records, tokenizer, items, item_ids)
        if not exact_json(counts, state.get('normalization_counts')) or not exact_json(state['provenance'].get('format_normalization'), format_contract):
            raise ValueError('restored normalization counts or policy differs')
        if not exact_json(json.loads((args.output / 'generation.manifest.json').read_text()), state['provenance']):
            raise ValueError('saved generation provenance differs before encoding')
        saved_rows = [json.loads(line) for line in (args.output / 'descriptions.jsonl').read_text().splitlines() if line]
        if not exact_json(saved_rows, [{key: value for key, value in row.items() if key != 'query_id'} for row in records]):
            raise ValueError('saved descriptions differ from verified token/normalization journal')
        del tokenizer
        descriptions = [row['description'] for row in records]
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(args.encoder_path), device=args.device, local_files_only=True)
        encoder.eval()
        if not callable(getattr(encoder, '_text_length', None)):
            raise ValueError('SentenceTransformer encoder lacks the required recorded text-length batching interface')
        order = np.argsort([-encoder._text_length(text) for text in descriptions]).tolist()
        batches = [order[start:start + args.encoder_batch_size] for start in range(0, len(order), args.encoder_batch_size)]
        vectors_journal = manager.scoring('sbert-' + state['descriptions_sha256'], [[item_ids[i] for i in batch] for batch in batches])
        if args.resume:
            manager.restore(restore_reference)
        boundary(force=True)
        for batch_index in range(vectors_journal.completed_batches, len(batches)):
            indices = batches[batch_index]
            vectors = encoder.encode([descriptions[i] for i in indices], batch_size=args.encoder_batch_size, convert_to_tensor=True, normalize_embeddings=False, show_progress_bar=False).float().cpu()
            if vectors.ndim != 2 or vectors.shape[0] != len(indices) or (not torch.isfinite(vectors).all()) or (vectors.norm(dim=1) == 0).any():
                raise RuntimeError('invalid Sentence-BERT feature vectors')
            vectors_journal.append(batch_index, [{'query_id': item_ids[i], 'vector': vector.tolist()} for i, vector in zip(indices, vectors)])
            boundary()
        encoded_by_id = {row['query_id']: row['vector'] for row in vectors_journal.records()}
        vectors = torch.tensor([encoded_by_id[item] for item in item_ids], dtype=torch.float32)
        if vectors.ndim != 2 or not torch.isfinite(vectors).all() or (vectors.norm(dim=1) == 0).any():
            raise RuntimeError('invalid complete feature matrix')
        atomic_write(args.output / 'features.pt', lambda stream: torch.save(vectors, stream))
        metadata = {'status': 'complete', 'feature_kind': 'llama3_descriptions_sbert_reconstruction', 'item_ids': item_ids, 'dataset_policy': args.dataset_policy, 'features_sha256': sha256(args.output / 'features.pt'), 'descriptions_sha256': sha256(args.output / 'descriptions.jsonl'), 'items_sha256': sha256(items_path), 'provenance': state['provenance'], 'source_sha256': contract['source_sha256'], 'format_normalization': format_contract, 'normalization_counts': counts, 'restart_helper_sha256': contract['restart_helper_sha256']}
        atomic_json(args.output / 'features.manifest.json', metadata)
        state.update(phase='complete', manifest_digest=digest(metadata))
        boundary(force=True)
        print('Completed Llama3 descriptions + Sentence-BERT features.', flush=True)
if __name__ == '__main__':
    main()
