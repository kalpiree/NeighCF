import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from data import LlaraDataError, batches, cosine_lr, digest, evaluation_contract, file_digest, load_bundle, loss_scale, prediction_record, teacher_row_order, token_example, training_plan, validate_teacher
from evaluate import evaluate_records
from integrations.resume import RestartManager, runtime_context

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')

def write_jsonl(path, records):
    with Path(path).open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prompt_path(args):
    return Path(__file__).parent / 'prompts/book.txt' if args.prompt_name == 'book.txt' else args.source / 'prompt' / args.prompt_name

def prompt_provenance(args):
    return {'requested': args.prompt_name, 'upstream_base': 'prompt/movie.txt' if args.prompt_name == 'book.txt' else 'prompt/' + args.prompt_name, 'template_sha256': file_digest(prompt_path(args)), 'adaptation': 'movie-to-book domain adaptation: movie→book and watched/watch→read; original three-template structure retained' if args.prompt_name == 'book.txt' else None}

def source_hashes(args):
    return {str(path.relative_to(args.source)): file_digest(path) for path in (args.source / 'model/mlp_projector.py', args.source / 'model/model_interface.py', args.source / 'data/data_interface.py', args.source / 'recommender/A_SASRec_final_bce_llm.py', args.source / 'prompt' / ('movie.txt' if args.prompt_name == 'book.txt' else args.prompt_name))}

def base_manifest(path):
    if path is None or not path.is_dir():
        raise LlaraDataError('--model-path requires a local Llama-2-7b-hf snapshot')
    config = json.loads((path / 'config.json').read_text())
    expected = {'model_type': 'llama', 'hidden_size': 4096, 'num_hidden_layers': 32, 'vocab_size': 32000, 'num_attention_heads': 32}
    if any((config.get(key) != value for key, value in expected.items())):
        raise LlaraDataError('backbone architecture is not the declared Llama-2-7B')
    files = sorted((p for p in path.rglob('*') if p.is_file() and p.suffix in {'.safetensors', '.bin', '.json', '.model'}))
    if not any((p.suffix in {'.bin', '.safetensors'} for p in files)):
        raise LlaraDataError('local model snapshot has no weights')
    return {str(p.relative_to(path)): {'bytes': p.stat().st_size, 'sha256': file_digest(p)} for p in files}

def integration_hashes():
    root = Path(__file__).parent
    return {name: file_digest(root / name) for name in ('run.py', 'data.py', 'prompts/book.txt')}

def shared_source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: file_digest(root / name) for name in ('integrations/resume.py', 'evaluate.py', 'adapters/llara_data.py', 'adapters/llmsrec_data.py')}

def metadata(args, bundle):
    return {'model': 'LLaRA', 'dataset': args.dataset_name or args.data.name, 'seed': args.seed, 'protocol_sha256': bundle['protocol_sha256'], 'run_kind': args.command, 'backbone': 'meta-llama/Llama-2-7b-hf', 'evaluation_protocol': evaluation_contract(args.expected_candidates), 'precision': 'BF16 frozen backbone; FP32 trainable projector/LoRA where PEFT casts', 'ranking_adaptation': 'sum conditional title-plus-newline log likelihood over frozen candidates', 'teacher_variant': 'shared validation-selected SASRec; item embeddings remapped by original catalog IDs', 'source_file_sha256': bundle['source_sha256'], 'mapping': bundle['mapping'], 'teacher_checkpoint_sha256': file_digest(args.teacher) if args.teacher is not None else None, 'upstream_source_sha256': source_hashes(args), 'prompt_provenance': prompt_provenance(args), 'integration_sha256': integration_hashes(), 'shared_source_sha256': shared_source_hashes(), 'history_limit': args.max_history, 'max_tokens': args.max_tokens, 'effective_batch_size': args.effective_batch_size, 'microbatch_size': args.microbatch_size, 'epochs_requested': args.epochs, 'train_candidates': args.train_candidates, 'observed_training_candidate_count': {'min': bundle['datasets']['train'].summary()['candidate_count_min'], 'max': bundle['datasets']['train'].candidate_width}, 'training_candidate_policy': 'explicit native count generated per query/seed from full-history-excluded lexical catalog; canonical training candidates are not model input; validation/test candidates preserved', 'curriculum': 'linear hybrid probability, one deterministic draw per optimizer batch', 'learning_rate': args.lr, 'warmup_ratio': args.warmup_ratio, 'minimum_learning_rate': args.min_lr, 'warmup_start_lr': args.warmup_start_lr, 'loss_aggregation': 'mean causal cross entropy over all answer tokens in optimizer batch', 'comparison_scope': 'reconstructed NeighCF protocol with user override to 50 evaluation negatives; original split files unavailable' if args.expected_candidates == 51 else 'reconstructed NeighCF protocol with explicitly configured evaluation candidates; original split files unavailable'}

class Runtime:

    def __init__(self, args, bundle, manifest):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        from peft import LoraConfig, TaskType, get_peft_model
        self.args, self.bundle, self.manifest = (args, bundle, manifest)
        self.device = torch.device(args.device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise LlaraDataError('LLaRA runtime requires a CUDA GPU')
        torch.cuda.set_device(self.device)
        if not torch.cuda.is_bf16_supported():
            raise LlaraDataError('BF16 unavailable; refusing an implicit precision change')
        self.tokenizer = LlamaTokenizer.from_pretrained(str(args.model_path), local_files_only=True, use_fast=False)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]']})
        self.tokenizer.padding_side = 'right'
        self.tokenizer_hash = digest(self.tokenizer.get_vocab())
        base = LlamaForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={'': str(self.device)}, attn_implementation='sdpa')
        self.original_vocab = base.config.vocab_size
        base.resize_token_embeddings(len(self.tokenizer))
        base.config.use_cache = False
        base.config.pad_token_id = self.tokenizer.pad_token_id
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
        self.llm = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
        official = load_module('llara_official_projector', args.source / 'model/mlp_projector.py')
        self.projector = official.MlpProjector(rec_size=64, llm_size=4096).to(self.device)
        self.item_embeddings, self.teacher_metadata = self.load_teacher()
        self.marker_ids = {token: self.tokenizer.convert_tokens_to_ids(token) for token in ('[HistoryEmb]', '[CansEmb]', '[ItemEmb]', '[PH]')}
        self.named_trainable = {'llm.' + name: p for name, p in self.llm.named_parameters() if p.requires_grad}
        if not self.named_trainable or any(('lora_' not in name for name in self.named_trainable)):
            raise RuntimeError('backbone freeze/LoRA trainability differs from declared configuration')
        self.named_trainable.update({'projector.' + name: p for name, p in self.projector.named_parameters()})
        prompt_text = prompt_path(args).read_text()
        if args.prompt_name == 'book.txt':
            original = (args.source / 'prompt/movie.txt').read_text()
            if prompt_text != original.replace('watched', 'read').replace('watch', 'read').replace('movie', 'book'):
                raise LlaraDataError('book template differs from its declared exact domain adaptation')
        self.templates = [line for line in prompt_text.splitlines() if line.strip()]
        if not self.templates or not 0 <= args.eval_template < len(self.templates):
            raise LlaraDataError('evaluation template index is unavailable')

    def load_teacher(self):
        if self.args.teacher is None:
            raise LlaraDataError("--teacher requires the shared integration's teacher.pth")
        saved = json.loads(self.args.teacher.with_name('teacher.metadata.json').read_text())
        validate_teacher(self.bundle, saved, require_final=self.args.command in {'train', 'score'})
        if saved.get('native_checkpoint_sha256') != file_digest(self.args.teacher):
            raise LlaraDataError('teacher checkpoint hash differs from its metadata')
        if self.args.command in {'train', 'score'}:
            completed = json.loads(self.args.teacher.with_name('teacher.completion.json').read_text())
            if completed.get('diagnostic', True) or completed.get('status') != 'complete' or completed.get('native_checkpoint_sha256') != saved['native_checkpoint_sha256']:
                raise LlaraDataError('teacher training/tuning is not complete')
        config, state = torch.load(self.args.teacher, map_location='cpu', weights_only=False)
        weight = state.get('item_emb.weight')
        if weight is None or tuple(weight.shape) != (len(self.bundle['items']) + 1, 64):
            raise LlaraDataError('teacher item embedding shape differs from the validated catalog')
        if config['item_num'] != len(self.bundle['items']) or config['args'].hidden_units != 64:
            raise LlaraDataError('teacher native configuration is incompatible')
        remapped = weight[teacher_row_order(self.bundle)].clone()
        return (torch.nn.Embedding.from_pretrained(remapped, freeze=True, padding_idx=len(self.bundle['items'])).to(self.device), saved)

    def set_training(self, training):
        self.llm.train(training)
        self.projector.train(training)
        self.item_embeddings.eval()

    def wrapped(self, ids, examples):
        embeds = self.llm.get_input_embeddings()(ids).clone()
        for row_index, example in enumerate(examples):
            sample = example['sample']
            if (ids[row_index] == self.marker_ids['[ItemEmb]']).any():
                raise LlaraDataError('target-item placeholder is forbidden in recommendation input')
            for marker, key, length_key in (('[HistoryEmb]', 'seq', 'len_seq'), ('[CansEmb]', 'cans', 'len_cans')):
                positions = (ids[row_index] == self.marker_ids[marker]).nonzero().flatten()
                expected = sample[length_key] if example['hybrid'] else 0
                if len(positions) != expected:
                    raise LlaraDataError('prompt marker count differs from retained item count')
                if expected:
                    item_ids = torch.tensor(sample[key][:expected], device=self.device, dtype=torch.long)
                    projected = self.projector(self.item_embeddings(item_ids))
                    embeds[row_index, positions] = projected.to(embeds.dtype)
        return embeds

    def forward_examples(self, examples):
        width = max((len(example['input_ids']) for example in examples))
        ids = torch.full((len(examples), width), self.tokenizer.pad_token_id, device=self.device, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for row, example in enumerate(examples):
            size = len(example['input_ids'])
            ids[row, :size] = torch.tensor(example['input_ids'], device=self.device)
            labels[row, :size] = torch.tensor(example['labels'], device=self.device)
            mask[row, :size] = 1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.llm(inputs_embeds=self.wrapped(ids, examples), attention_mask=mask, labels=labels, use_cache=False, return_dict=True)

    def examples(self, samples, template_index=0, hybrid=True):
        return [token_example(self.tokenizer, sample, self.templates[template_index], hybrid, max_tokens=self.args.max_tokens) for sample in samples]

    def full_likelihood(self, sample, answer):
        example = token_example(self.tokenizer, sample, self.templates[self.args.eval_template], True, answer=answer, max_tokens=self.args.max_tokens)
        with torch.no_grad():
            result = self.forward_examples([example])
            score = -float(result.loss.float()) * example['answer_tokens']
        return score

    def score_one(self, sample, verify_cache=False):
        from transformers.cache_utils import DynamicCache
        self.set_training(False)
        examples = [token_example(self.tokenizer, sample, self.templates[self.args.eval_template], True, answer=title, max_tokens=self.args.max_tokens) for title in sample['cans_name']]
        first = examples[0]
        prompt = torch.tensor([first['prompt_ids']], dtype=torch.long, device=self.device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            prefix = self.llm(inputs_embeds=self.wrapped(prompt, [first]), attention_mask=torch.ones_like(prompt), use_cache=True, return_dict=True)
            first_logits = prefix.logits[0, -1].float()
            raw_cache = prefix.past_key_values
            legacy = raw_cache.to_legacy_cache() if hasattr(raw_cache, 'to_legacy_cache') else raw_cache
            if not isinstance(legacy, tuple):
                raise RuntimeError('unsupported Transformers cache representation')
            del prefix, raw_cache
            scores = []
            first_log_probs = torch.log_softmax(first_logits, dim=-1)
            for example in examples:
                answer = example['answer_ids']
                score = first_log_probs[answer[0]]
                if len(answer) > 1:
                    continuation = torch.tensor([answer[:-1]], device=self.device, dtype=torch.long)
                    cache = DynamicCache.from_legacy_cache(legacy)
                    mask = torch.ones((1, prompt.shape[1] + continuation.shape[1]), device=self.device, dtype=torch.long)
                    output = self.llm(input_ids=continuation, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
                    targets = torch.tensor(answer[1:], device=self.device, dtype=torch.long)
                    score = score - torch.nn.functional.cross_entropy(output.logits[0].float(), targets, reduction='sum')
                    del output, cache
                scores.append(float(score))
            del legacy
        if verify_cache:
            for index in (0, len(scores) - 1):
                full = self.full_likelihood(sample, sample['cans_name'][index])
                if not math.isclose(scores[index], full, rel_tol=self.args.cache_rtol, abs_tol=self.args.cache_atol):
                    raise RuntimeError(f'cached/full likelihood mismatch: cached={scores[index]}, full={full}')
        return prediction_record(sample, scores)

    def score(self, samples, verify_cache=False, *, journal=None, checkpoint=None):
        completed = journal.completed_batches if journal is not None else 0
        records = journal.records(require_complete=False) if journal is not None else []
        if completed > len(samples) or len(records) != completed:
            raise LlaraDataError('scoring journal differs from original query batches')
        for index in range(completed, len(samples)):
            record = self.score_one(samples[index], verify_cache=verify_cache and index == completed)
            if journal is not None:
                journal.append(index, [record])
            records.append(record)
            if checkpoint is not None:
                checkpoint()
            if index == completed or (index + 1) % 100 == 0:
                print(json.dumps({'event': 'ranking', 'queries_scored': index + 1, 'queries_total': len(samples)}), flush=True)
        return records

    def contract(self):
        return {'mapping': self.bundle['mapping'], 'source_sha256': self.bundle['source_sha256'], 'protocol_sha256': self.bundle['protocol_sha256'], 'backbone_manifest': self.manifest, 'teacher_sha256': file_digest(self.args.teacher), 'tokenizer_sha256': self.tokenizer_hash, 'upstream_source_sha256': source_hashes(self.args), 'prompt_provenance': prompt_provenance(self.args), 'max_history': self.args.max_history, 'integration_sha256': integration_hashes(), 'shared_source_sha256': shared_source_hashes(), 'max_tokens': self.args.max_tokens, 'eval_template': self.args.eval_template, 'evaluation_protocol': evaluation_contract(self.args.expected_candidates), 'lora': [self.args.lora_r, self.args.lora_alpha, self.args.lora_dropout]}

    def save(self, path, epoch):
        torch.save({'contract': self.contract(), 'epoch': epoch, 'trainable': {name: p.detach().cpu() for name, p in self.named_trainable.items()}, 'extra_input_rows': self.llm.get_input_embeddings().weight[self.original_vocab:].detach().cpu(), 'extra_output_rows': self.llm.get_output_embeddings().weight[self.original_vocab:].detach().cpu()}, path)

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['contract'] != self.contract() or state['trainable'].keys() != self.named_trainable.keys():
            raise LlaraDataError('checkpoint provenance or trainable tensors differ')
        with torch.no_grad():
            for name, parameter in self.named_trainable.items():
                parameter.copy_(state['trainable'][name].to(parameter.device))
            self.llm.get_input_embeddings().weight[self.original_vocab:].copy_(state['extra_input_rows'].to(self.device))
            self.llm.get_output_embeddings().weight[self.original_vocab:].copy_(state['extra_output_rows'].to(self.device))

def persist_scores(path, records, args, bundle):
    result, ranks = evaluate_records(records, metadata(args, bundle), bundle['expected_candidates'])
    write_jsonl(path.with_suffix('.predictions.jsonl'), records)
    write_json(path.with_suffix('.metadata.json'), result['metadata'])
    write_json(path.with_suffix('.metrics.json'), result)
    write_jsonl(path.with_suffix('.ranks.jsonl'), ranks)
    return result

def restart_contract(args, bundle, model):
    operational = {'data', 'source', 'output', 'model_path', 'teacher', 'checkpoint', 'device', 'resume', 'checkpoint_seconds'}
    parameters = {key: value for key, value in vars(args).items() if key not in operational}
    return {'schema': 'llara-restart-v1', 'model_contract': model.contract(), 'parameters': parameters, 'runtime': runtime_context(args.device), 'libraries': {'transformers': __import__('transformers').__version__, 'peft': __import__('peft').__version__}, 'teacher_metadata_sha256': file_digest(args.teacher.with_name('teacher.metadata.json')), 'teacher_completion_sha256': file_digest(args.teacher.with_name('teacher.completion.json')) if args.teacher.with_name('teacher.completion.json').exists() else None, 'scoring_checkpoint_sha256': file_digest(args.checkpoint) if args.checkpoint is not None else None}

def epoch_batches(args, train_size, epoch):
    order = batches(train_size, args.effective_batch_size, seed=args.seed + epoch)
    return [order[0][:args.pilot_examples]] if args.command == 'pilot' else order

def snapshot_artifacts(output, progress, score_split):
    names = []
    if progress['best'] is not None:
        names.extend(['best.pt', 'valid-best.predictions.jsonl', 'valid-best.metadata.json', 'valid-best.metrics.json', 'valid-best.ranks.jsonl'])
    if progress['phase'] == 'complete':
        kind = progress['completion']['run_kind']
        prefix = 'test' if kind == 'train' else score_split if kind == 'score' else None
        if prefix is not None:
            names.extend((prefix + suffix for suffix in ('.predictions.jsonl', '.metadata.json', '.metrics.json', '.ranks.jsonl')))
    return {name: output / name for name in names}

def gpu_run(args, bundle):
    manifest = base_manifest(args.model_path)
    model = Runtime(args, bundle, manifest)
    context = {'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'peft': __import__('peft').__version__, 'gpu': torch.cuda.get_device_name(args.device), 'trainable_parameters': sum((p.numel() for p in model.named_trainable.values()))}
    with RestartManager(args.output, restart_contract(args, bundle, model), resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:
        write_json(args.output / 'backbone.manifest.json', manifest)
        write_json(args.output / 'runtime.json', context)
        if args.command == 'score' and args.checkpoint is None:
            raise LlaraDataError('score requires --checkpoint')
        saved_progress = manager.peek_progress()
        needs_optimizer = args.command != 'score' and (saved_progress is None or saved_progress['phase'] in {'train', 'validation', 'fresh_restore'})
        optimizer = torch.optim.Adam(model.named_trainable.values(), lr=args.lr, weight_decay=args.weight_decay) if needs_optimizer else None
        progress = manager.restore(model.restore, optimizer=optimizer)
        if progress is None:
            if args.command == 'score':
                model.restore(args.checkpoint)
            progress = {'phase': 'score' if args.command == 'score' else 'train', 'epoch': 1, 'next_batch': 0, 'epoch_order': epoch_batches(args, len(bundle['datasets']['train']), 1) if args.command != 'score' else None, 'global_step': 0, 'best': None, 'selected_epoch': None, 'training_events': [], 'train_examples': 0, 'answer_tokens': 0, 'elapsed_seconds': 0.0, 'peak_allocated_bytes': 0, 'lr_scheduler': {'kind': 'explicit_cosine', 'completed_steps': 0}}
        started, elapsed_before = (time.monotonic(), progress['elapsed_seconds'])
        torch.cuda.reset_peak_memory_stats(args.device)

        def checkpoint(*, force=False):
            progress['elapsed_seconds'] = elapsed_before + time.monotonic() - started
            progress['peak_allocated_bytes'] = max(progress['peak_allocated_bytes'], torch.cuda.max_memory_allocated(args.device))
            return manager.checkpoint(progress, lambda path: model.save(path, progress['epoch']), optimizer=optimizer, artifacts=snapshot_artifacts(args.output, progress, args.score_split), force=force)
        if progress['phase'] == 'complete':
            write_json(args.output / 'completion.json', progress['completion'])
            print(json.dumps(progress['completion']), flush=True)
            return
        write_jsonl(args.output / 'train-progress.jsonl', progress['training_events'])
        checkpoint(force=True)
        if args.command == 'score':
            samples = [bundle['datasets'][args.score_split][i] for i in range(len(bundle['datasets'][args.score_split]))]
            journal = manager.scoring('score-' + args.score_split, [[s['query_id']] for s in samples])
            records = model.score(samples, verify_cache=True, journal=journal, checkpoint=checkpoint)
            persist_scores(args.output / args.score_split, records, args, bundle)
            progress['phase'] = 'complete'
            progress['completion'] = {'status': 'complete', 'run_kind': 'score', 'queries': len(records), 'cache_full_likelihood_verified': True}
            checkpoint(force=True)
            write_json(args.output / 'completion.json', progress['completion'])
            return
        train = bundle['datasets']['train']
        validation = bundle['datasets']['valid']
        valid_samples = [validation[i] for i in range(min(len(validation), args.pilot_valid_queries) if args.command == 'pilot' else len(validation))]
        total_epochs = args.epochs if args.command == 'train' else 1
        total_steps = args.epochs * len(batches(len(train), args.effective_batch_size)) if args.command == 'train' else 1
        warmup = min(total_steps, math.ceil(total_steps * args.warmup_ratio))
        best_checkpoint = args.output / 'best.pt'
        while progress['phase'] in {'train', 'validation'}:
            epoch = progress['epoch']
            order = epoch_batches(args, len(train), epoch)
            if progress['epoch_order'] is None:
                progress['epoch_order'] = order
            if progress['epoch_order'] != order or not 0 <= progress['next_batch'] <= len(order):
                raise LlaraDataError('saved epoch permutation/cursor differs')
            expected_step = (epoch - 1) * len(order) + progress['next_batch']
            if progress['global_step'] != expected_step or len(progress['training_events']) != expected_step:
                raise LlaraDataError('saved optimizer step differs from epoch permutation/cursor')
            if progress['lr_scheduler']['completed_steps'] != progress['global_step']:
                raise LlaraDataError('saved cosine scheduler position differs')
            if progress['phase'] == 'train':
                for batch_index in range(progress['next_batch'], len(order)):
                    indices = order[batch_index]
                    model.set_training(True)
                    global_step = progress['global_step']
                    plan = training_plan(args.seed, global_step, total_steps, model.templates)
                    if args.command == 'pilot':
                        plan['hybrid'] = True
                    examples = model.examples([train[i] for i in indices], plan['template_index'], plan['hybrid'])
                    denominator = sum((example['answer_tokens'] for example in examples))
                    lr = cosine_lr(global_step, total_steps, warmup, args.lr, args.warmup_start_lr, args.min_lr)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    for micro in batches(len(examples), args.microbatch_size):
                        current = [examples[i] for i in micro]
                        scale = loss_scale(sum((example['answer_tokens'] for example in current)), denominator)
                        output = model.forward_examples(current)
                        loss = output.loss.float() * scale
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite causal training loss')
                        loss.backward()
                        total_loss += float(loss.detach())
                        del output, loss
                    gradients = {name: parameter.grad for name, parameter in model.named_trainable.items() if parameter.grad is not None}
                    if not gradients or any((not torch.isfinite(gradient).all() for gradient in gradients.values())):
                        raise RuntimeError('missing or nonfinite trainable gradients')
                    if plan['hybrid'] and (not any((name.startswith('projector.') and bool(gradient.abs().max() > 0) for name, gradient in gradients.items()))):
                        raise RuntimeError('hybrid training produced no projector gradient')
                    if not any(('lora_' in name and bool(gradient.abs().max() > 0) for name, gradient in gradients.items())):
                        raise RuntimeError('training produced no LoRA gradient')
                    optimizer.step()
                    progress['global_step'] += 1
                    progress['next_batch'] = batch_index + 1
                    progress['lr_scheduler']['completed_steps'] = progress['global_step']
                    progress['train_examples'] += len(examples)
                    progress['answer_tokens'] += denominator
                    event = {'epoch': epoch, 'step': progress['global_step'], 'loss': total_loss, 'examples': len(examples), 'answer_tokens': denominator, 'lr': lr, **plan, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device)}
                    progress['training_events'].append(event)
                    with (args.output / 'train-progress.jsonl').open('a') as stream:
                        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
                    if batch_index == 0 or (batch_index + 1) % 10 == 0:
                        print(json.dumps(event), flush=True)
                    checkpoint()
                progress['phase'] = 'validation'
                checkpoint(force=True)
            journal = manager.scoring(f'valid-epoch-{epoch}', [[s['query_id']] for s in valid_samples])
            records = model.score(valid_samples, verify_cache=True, journal=journal, checkpoint=checkpoint)
            result, _ = evaluate_records(records, metadata(args, bundle), bundle['expected_candidates'])
            metric = result['metrics'][args.selection_metric]
            print(json.dumps({'epoch': epoch, 'validation': result['metrics']}), flush=True)
            if progress['best'] is None or metric > progress['best']:
                progress['best'], progress['selected_epoch'] = (metric, epoch)
                model.save(best_checkpoint, epoch)
                persist_scores(args.output / 'valid-best', records, args, bundle)
                progress['best_checkpoint_sha256'] = file_digest(best_checkpoint)
            if epoch == total_epochs:
                progress['phase'] = 'fresh_restore'
            else:
                progress.update(phase='train', epoch=epoch + 1, next_batch=0, epoch_order=epoch_batches(args, len(train), epoch + 1))
            checkpoint(force=True)
        if file_digest(best_checkpoint) != progress['best_checkpoint_sha256']:
            raise LlaraDataError('selected checkpoint differs from durable best artifact')
        selected_records = [json.loads(line) for line in (args.output / 'valid-best.predictions.jsonl').read_text().splitlines()]
        if len(selected_records) != len(valid_samples):
            raise LlaraDataError('selected validation records differ from original query count')
        del optimizer, model
        optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        seed_all(args.seed + 1000003)
        model = Runtime(args, bundle, manifest)
        model.restore(best_checkpoint)
        restored_record = model.score_one(valid_samples[0], verify_cache=True)
        expected = selected_records[0]
        if restored_record['candidate_ids'] != expected['candidate_ids'] or any((not math.isclose(a, b, rel_tol=0.0001, abs_tol=0.001) for a, b in zip(restored_record['scores'], expected['scores']))):
            raise RuntimeError('fresh checkpoint restoration changed fixed candidate scores')
        progress['phase'] = 'test' if args.command == 'train' else 'finish'
        checkpoint(force=True)
        completed = {'status': 'complete', 'run_kind': args.command, 'optimizer_steps': progress['global_step'], 'evaluation_protocol': evaluation_contract(args.expected_candidates), 'fresh_restore_verified': True, 'cache_full_likelihood_verified': True, 'cache_atol': args.cache_atol, 'cache_rtol': args.cache_rtol, 'checkpoint_sha256': file_digest(best_checkpoint), 'selected_validation_value': progress['best'], 'selected_epoch': progress['selected_epoch'], 'validation_queries': len(valid_samples), 'seed': args.seed, 'train_examples': progress['train_examples'], 'answer_tokens': progress['answer_tokens']}
        if args.command == 'train':
            test = bundle['datasets']['test']
            samples = [test[i] for i in range(len(test))]
            journal = manager.scoring('test-selected', [[s['query_id']] for s in samples])
            records = model.score(samples, verify_cache=True, journal=journal, checkpoint=checkpoint)
            persist_scores(args.output / 'test', records, args, bundle)
            completed['test_queries'] = len(records)
        else:
            completed['limitation'] = 'one diagnostic optimizer batch and a validation subset; not a baseline result'
        completed['seconds'] = elapsed_before + time.monotonic() - started
        completed['gpu_peak_allocated_bytes'] = max(progress['peak_allocated_bytes'], torch.cuda.max_memory_allocated(args.device))
        progress['phase'], progress['completion'] = ('complete', completed)
        checkpoint(force=True)
        write_json(args.output / 'completion.json', completed)
        print(json.dumps(completed), flush=True)

def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('audit', 'pilot', 'train', 'score'))
    result.add_argument('--data', type=Path, required=True)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--resume', action='store_true', help='resume only an existing checkpoint with the exact scientific contract')
    result.add_argument('--checkpoint-seconds', type=float, default=600, help='operational periodic checkpoint interval; signal checkpoints occur at the next safe boundary')
    result.add_argument('--dataset-name')
    result.add_argument('--model-path', type=Path)
    result.add_argument('--teacher', type=Path)
    result.add_argument('--checkpoint', type=Path)
    result.add_argument('--score-split', choices=('valid', 'test'), default='test')
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--max-history', type=int, default=10)
    result.add_argument('--teacher-max-length', type=int, default=128)
    result.add_argument('--train-candidates', type=int, default=20)
    result.add_argument('--expected-candidates', type=int, default=51, help='fixed evaluation total including the true item; default 51 is the user override to 50 negatives')
    result.add_argument('--max-tokens', type=int, default=2048)
    result.add_argument('--prompt-name', default='movie.txt', choices=('movie.txt', 'game.txt', 'artist.txt', 'book.txt'))
    result.add_argument('--eval-template', type=int, default=0)
    result.add_argument('--epochs', type=int, default=5)
    result.add_argument('--effective-batch-size', type=int, default=128)
    result.add_argument('--microbatch-size', type=int, default=1)
    result.add_argument('--lr', type=float, default=0.0008)
    result.add_argument('--warmup-start-lr', type=float, default=8e-06)
    result.add_argument('--min-lr', type=float, default=8e-06)
    result.add_argument('--warmup-ratio', type=float, default=0.05)
    result.add_argument('--weight-decay', type=float, default=1e-05)
    result.add_argument('--lora-r', type=int, default=8)
    result.add_argument('--lora-alpha', type=int, default=32)
    result.add_argument('--lora-dropout', type=float, default=0.1)
    result.add_argument('--selection-metric', choices=('HR@10', 'NDCG@10'), default='NDCG@10')
    result.add_argument('--pilot-examples', type=int, default=2)
    result.add_argument('--pilot-valid-queries', type=int, default=1)
    result.add_argument('--cache-atol', type=float, default=0.1)
    result.add_argument('--cache-rtol', type=float, default=0.005)
    return result

def main():
    global torch
    args = parser().parse_args()
    evaluation_contract(args.expected_candidates)
    for field in ('max_history', 'teacher_max_length', 'train_candidates', 'expected_candidates', 'max_tokens', 'epochs', 'effective_batch_size', 'microbatch_size', 'pilot_examples', 'pilot_valid_queries', 'lora_r', 'lora_alpha'):
        if getattr(args, field) < 1:
            raise LlaraDataError(f'{field} must be positive')
    if not 0 <= args.warmup_ratio <= 1 or not 0 <= args.lora_dropout < 1:
        raise LlaraDataError('invalid warmup ratio or LoRA dropout')
    if args.command == 'train' and (args.epochs != 5 or args.effective_batch_size != 128):
        raise LlaraDataError('full training requires five epochs and effective batch size 128; evaluation count is explicitly configured')
    if args.pilot_examples > args.effective_batch_size:
        raise LlaraDataError('pilot examples cannot exceed one optimizer batch')
    bundle = load_bundle(args.data, args.seed, args.max_history, args.teacher_max_length, args.train_candidates, args.expected_candidates)
    info = metadata(args, bundle)
    if args.checkpoint_seconds <= 0 or not math.isfinite(args.checkpoint_seconds):
        raise LlaraDataError('checkpoint interval must be finite and positive')
    if args.resume:
        if args.command == 'audit' or not args.output.is_dir():
            raise LlaraDataError('--resume requires an existing GPU run output directory')
        if json.loads((args.output / 'run.metadata.json').read_text()) != info:
            raise LlaraDataError('resume metadata differs; refusing to overwrite the existing run')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / 'run.metadata.json', info)
        write_json(args.output / 'data.audit.json', {split: dataset.summary() for split, dataset in bundle['datasets'].items()})
    if args.command == 'audit':
        print(json.dumps({'status': 'data_audited', 'examples': {k: len(v) for k, v in bundle['datasets'].items()}, 'model_execution': False}), flush=True)
        return
    import torch
    seed_all(args.seed)
    gpu_run(args, bundle)
if __name__ == '__main__':
    main()
