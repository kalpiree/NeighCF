from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from data import DataError, PACKAGE_ROOT, batch_indices, cosine_multiplier, digest, evaluation_contract, file_digest, item_prompt, load_data, prediction_record, protocol_hash, teacher_contract, training_candidates, user_prompt, validate_teacher_contract, verify_restored_validation
from evaluate import evaluate_records
from gradient_cache import backward_cached
from integrations.resume import RestartManager, RestartError, runtime_context

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def write_jsonl(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')

def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def upstream_fingerprint(source):
    files = ('SeqRec/sasrec/model.py', 'models/seqllm4rec.py', 'models/seqllm_model.py')
    return {name: file_digest(source / name) for name in files}

def integration_fingerprint():
    directory = Path(__file__).resolve().parent
    return {name: file_digest(directory / name) for name in ('run.py', 'data.py', 'gradient_cache.py')}

def metadata(args, bundle, model):
    return {'model': model, 'dataset': args.dataset_name or args.data.name, 'seed': args.seed, 'protocol_sha256': protocol_hash(bundle, args.protocol_sha256), 'comparison_scope': 'reconstructed NeighCF protocol with user override to 50 evaluation negatives; original split files unavailable' if args.expected_candidates == 51 else 'reconstruction of NeighCF experimental protocol with explicitly configured evaluation candidates; original split files unavailable', 'evaluation_protocol': evaluation_contract(args.expected_candidates), 'mapping_sha256': bundle['mapping_sha256'], 'source_file_sha256': bundle['source_file_sha256'], 'upstream_source_sha256': upstream_fingerprint(args.source), 'integration_source_sha256': integration_fingerprint(), 'shared_source_sha256': {name: file_digest(PACKAGE_ROOT / name) for name in ('integrations/resume.py', 'evaluate.py', 'adapters/llmsrec_data.py')}, 'teacher_history_limit': args.max_length, 'llm_history_limit': 10, 'date_policy': 'explicit date-free adaptation' if args.omit_dates else 'real per-event UTC dates', 'backbone': 'meta-llama/Llama-2-7b-hf', 'method_variant': 'controlled-backbone LLM-SRec (native paper uses Llama-3.2-3B-Instruct)', 'run_kind': args.command}

def selected_rows(rows, indices):
    return [rows[i] for i in indices]

def evaluate_predictions(records, args, bundle, model):
    return evaluate_records(records, metadata(args, bundle, model), args.expected_candidates)

def create_teacher(args, bundle):
    source_module = module_from_file('llmsrec_native_sasrec', args.source / 'SeqRec/sasrec/model.py')
    native = argparse.Namespace(device=args.device, hidden_units=64, maxlen=args.max_length, dropout_rate=0.1, num_blocks=2, num_heads=1, nn_parameter=False)
    teacher = source_module.SASRec(len(bundle['mapping']['user_to_index']), len(bundle['items']), native).to(args.device)
    for parameter in teacher.parameters():
        if parameter.ndim > 1:
            torch.nn.init.xavier_normal_(parameter)
    return teacher

def sequence_array(queries):
    return np.asarray([query['sequence_item_indices'] for query in queries], dtype=np.int64)

def query_batches(queries, size):
    identities = getattr(queries, 'query_ids', None)
    return [[identities[i] if identities is not None else queries[i]['source']['query_id'] for i in indices] for indices in batch_indices(len(queries), size)]

def existing_artifacts(output, names):
    return {name: output / name for name in names if (output / name).exists()}

def publish_events(output, events):
    write_jsonl(output / 'progress.jsonl', events)

def teacher_scores(teacher, queries, args, bundle, *, manager=None, stage=None, boundary=None):
    teacher.eval()
    journal = manager.scoring(stage, query_batches(queries, args.eval_batch_size)) if manager else None
    records = journal.records(require_complete=False) if journal else []
    with torch.no_grad():
        for index, indices in enumerate(batch_indices(len(queries), args.eval_batch_size)):
            if journal and index < journal.completed_batches:
                continue
            batch = selected_rows(queries, indices)
            candidates = np.asarray([row['candidate_item_indices'] for row in batch])
            scores = teacher.predict(None, sequence_array(batch), candidates).float().cpu().tolist()
            rows = [prediction_record(row, score) for row, score in zip(batch, scores)]
            records.extend(rows)
            if journal:
                journal.append(index, rows)
            if boundary:
                boundary()
    return journal.records() if journal else records

def train_teacher(args, bundle, manager):
    bce = torch.nn.BCEWithLogitsLoss()
    output = args.output
    checkpoint_path = output / 'teacher.pth'
    seed_all(args.seed)
    teacher = create_teacher(args, bundle)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=args.teacher_lr, betas=(0.9, 0.98), weight_decay=args.teacher_l2_grid[0])

    def save_model(path):
        torch.save([teacher.kwargs, {key: value.detach().cpu() for key, value in teacher.state_dict().items()}], path)

    def restore_model(path):
        _, state = torch.load(path, map_location='cpu', weights_only=False)
        teacher.load_state_dict(state, strict=True)
    progress = manager.restore(restore_model, optimizer=optimizer)
    if progress is None:
        progress = {'phase': 'train', 'trial': 0, 'epoch': 1, 'next_batch': 0, 'global_step': 0, 'best': None, 'trial_best': None, 'trial_stale': 0, 'losses': [], 'events': []}

    def boundary(force=False):
        manager.checkpoint(progress, save_model, optimizer=optimizer, artifacts=existing_artifacts(output, ['teacher.pth', 'teacher.metadata.json']), force=force)
    if not args.resume:
        boundary(force=True)
    publish_events(output, progress['events'])
    while progress['trial'] < len(args.teacher_l2_grid):
        trial, epoch = (progress['trial'], progress['epoch'])
        regularization = args.teacher_l2_grid[trial]
        if progress['phase'] == 'train':
            teacher.train()
            batches = batch_indices(len(bundle['rows']['train']), args.teacher_batch_size, seed=args.seed + epoch)
            for batch_index in range(progress['next_batch'], len(batches)):
                batch = selected_rows(bundle['rows']['train'], batches[batch_index])
                sequences = sequence_array(batch)
                positive = np.zeros_like(sequences)
                negative = np.zeros_like(sequences)
                for i, query in enumerate(batch):
                    samples = training_candidates(query, bundle['catalog'], args.seed, epoch, count=2)
                    positive[i, -1], negative[i, -1] = samples
                optimizer.zero_grad(set_to_none=True)
                pos, neg = teacher(None, sequences, positive, negative)
                loss = bce(pos[:, -1], torch.ones_like(pos[:, -1])) + bce(neg[:, -1], torch.zeros_like(neg[:, -1]))
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite SASRec training loss')
                loss.backward()
                check_gradients(dict(teacher.named_parameters()))
                optimizer.step()
                progress['losses'].append(loss.item())
                progress['next_batch'] = batch_index + 1
                progress['global_step'] += 1
                boundary()
            progress['phase'] = 'valid'
            boundary(force=True)
        records = teacher_scores(teacher, bundle['rows']['valid'], args, bundle, manager=manager, stage=f'teacher-trial-{trial}-epoch-{epoch}-valid', boundary=boundary)
        result, _ = evaluate_predictions(records, args, bundle, 'SASRec-teacher')
        value = result['metrics'][args.selection_metric]
        event = {'epoch': epoch, 'l2': regularization, 'loss': sum(progress['losses']) / len(progress['losses']), 'validation': result['metrics']}
        progress['events'].append(event)
        publish_events(output, progress['events'])
        print(json.dumps(event), flush=True)
        if progress['best'] is None or value > progress['best']:
            progress['best'] = value
            save_model(checkpoint_path)
            write_json(output / 'teacher.metadata.json', {'contract': teacher_contract(bundle), 'selected_epoch': epoch, 'validation_metric': args.selection_metric, 'validation_value': value, 'validation_sha256': bundle['source_file_sha256']['valid.jsonl'], 'evaluation_protocol': evaluation_contract(args.expected_candidates), 'native_checkpoint_sha256': file_digest(checkpoint_path), 'upstream_source_sha256': upstream_fingerprint(args.source), 'seed': args.seed, 'epochs_requested': args.teacher_epochs, 'selected_l2': regularization, 'l2_grid': args.teacher_l2_grid, 'run_kind': 'diagnostic' if args.teacher_diagnostic else 'validation_selected_teacher', 'diagnostic': args.teacher_diagnostic, 'early_stopping_patience': args.teacher_patience, 'l2_implementation': 'Adam weight_decay on all teacher parameters', 'protocol_sha256': protocol_hash(bundle, args.protocol_sha256)})
        if progress['trial_best'] is None or value > progress['trial_best']:
            progress['trial_best'], progress['trial_stale'] = (value, 0)
        else:
            progress['trial_stale'] += 1
        trial_done = epoch >= args.teacher_epochs or (args.teacher_patience and progress['trial_stale'] >= args.teacher_patience)
        if trial_done:
            progress.update(trial=trial + 1, epoch=1, next_batch=0, trial_best=None, trial_stale=0, losses=[], phase='train')
            if progress['trial'] < len(args.teacher_l2_grid):
                seed_all(args.seed)
                teacher = create_teacher(args, bundle)
                optimizer = torch.optim.Adam(teacher.parameters(), lr=args.teacher_lr, betas=(0.9, 0.98), weight_decay=args.teacher_l2_grid[progress['trial']])
            else:
                progress['phase'] = 'complete'
        else:
            progress.update(epoch=epoch + 1, next_batch=0, losses=[], phase='train')
        boundary(force=True)
    write_json(output / 'teacher.completion.json', {'status': 'complete', 'diagnostic': args.teacher_diagnostic, 'evaluation_protocol': evaluation_contract(args.expected_candidates), 'run_kind': 'diagnostic' if args.teacher_diagnostic else 'validation_selected_teacher', 'native_checkpoint_sha256': file_digest(checkpoint_path), 'l2_grid': args.teacher_l2_grid, 'maximum_epochs_per_trial': args.teacher_epochs, 'early_stopping_patience': args.teacher_patience})
    print(f'Saved validation-selected teacher: {checkpoint_path}', flush=True)

def load_teacher(args, bundle):
    if args.teacher is None:
        raise DataError("--teacher must point to this integration's teacher.pth")
    sidecar = args.teacher.with_name('teacher.metadata.json')
    saved = json.loads(sidecar.read_text())
    validate_teacher_contract(bundle, saved)
    if args.command == 'train' and saved.get('diagnostic', True):
        raise DataError('full LLM training requires a validation-selected teacher, not a diagnostic teacher checkpoint')
    if args.command == 'train':
        completion = json.loads(args.teacher.with_name('teacher.completion.json').read_text())
        if completion.get('status') != 'complete' or completion.get('diagnostic', True) or completion.get('native_checkpoint_sha256') != saved['native_checkpoint_sha256']:
            raise DataError('teacher training/tuning has not completed with this checkpoint')
    if saved['native_checkpoint_sha256'] != file_digest(args.teacher):
        raise DataError('teacher checkpoint hash differs from its metadata')
    if saved['validation_sha256'] != bundle['source_file_sha256']['valid.jsonl']:
        raise DataError('teacher was selected using a different validation split')
    if saved['protocol_sha256'] != protocol_hash(bundle, args.protocol_sha256):
        raise DataError('teacher checkpoint belongs to a different declared experimental protocol')
    if saved['upstream_source_sha256']['SeqRec/sasrec/model.py'] != upstream_fingerprint(args.source)['SeqRec/sasrec/model.py']:
        raise DataError('native SASRec source changed since teacher training')
    teacher = create_teacher(args, bundle)
    kwargs, state = torch.load(args.teacher, map_location='cpu', weights_only=False)
    if kwargs['item_num'] != len(bundle['items']) or kwargs['args'].hidden_units != 64:
        raise DataError('native SASRec configuration differs from the validated contract')
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher

def check_gradients(named):
    found = False
    for name, parameter in named.items():
        if parameter.grad is not None:
            found = True
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f'nonfinite gradient in {name}')
    if not found:
        raise RuntimeError('backward produced no trainable gradients')

def local_model_manifest(path):
    if not path or not path.is_dir():
        raise DataError('--model-path must name a local Llama-2-7b-hf snapshot')
    config = json.loads((path / 'config.json').read_text())
    if (config.get('model_type'), config.get('hidden_size'), config.get('num_hidden_layers'), config.get('vocab_size')) != ('llama', 4096, 32, 32000):
        raise DataError('local checkpoint does not have the manuscript Llama-2-7B architecture')
    files = sorted((file for file in path.rglob('*') if file.is_file() and file.suffix in {'.safetensors', '.bin', '.json', '.model', '.tiktoken'}))
    if not any((file.suffix in {'.safetensors', '.bin'} for file in files)):
        raise DataError('no local model weight files found')
    return {str(file.relative_to(path)): {'bytes': file.stat().st_size, 'sha256': file_digest(file)} for file in files}

class Runtime:

    def __init__(self, args, bundle, base_manifest):
        self.args, self.bundle, self.base_manifest = (args, bundle, base_manifest)
        self.device = args.device
        self.teacher = load_teacher(args, bundle)
        official = module_from_file('llmsrec_native_llm', args.source / 'models/seqllm4rec.py')
        original_model_loader = official.AutoModelForCausalLM
        original_tokenizer_loader = official.AutoTokenizer

        class LocalModelLoader:

            @staticmethod
            def from_pretrained(_model_id, **kwargs):
                return original_model_loader.from_pretrained(str(args.model_path), local_files_only=True, **kwargs)

        class LocalTokenizerLoader:

            @staticmethod
            def from_pretrained(_model_id, **kwargs):
                return original_tokenizer_loader.from_pretrained(str(args.model_path), local_files_only=True, **kwargs)
        official.AutoModelForCausalLM = LocalModelLoader
        official.AutoTokenizer = LocalTokenizerLoader
        native_args = SimpleNamespace(nn_parameter=False, token=False)
        self.llm = official.llm4rec(device=args.device, llm_model='llama-3b', args=native_args)
        for name, child in self.llm.named_children():
            if name != 'llm_model':
                child.to(args.device)
        hidden = self.llm.llm_model.config.hidden_size
        self.projection = torch.nn.Sequential(torch.nn.Linear(64, hidden), torch.nn.LayerNorm(hidden), torch.nn.LeakyReLU(), torch.nn.Linear(hidden, hidden)).to(args.device)
        torch.nn.init.xavier_normal_(self.projection[0].weight)
        torch.nn.init.xavier_normal_(self.projection[3].weight)
        if any((p.requires_grad for p in self.llm.llm_model.parameters())):
            raise RuntimeError('the LLM backbone unexpectedly has trainable parameters')
        self.original_vocab = json.loads((args.model_path / 'config.json').read_text())['vocab_size']
        self.tokenizer_hash = digest(self.llm.llm_tokenizer.get_vocab())
        self.named_trainable = {'llm.' + name: parameter for name, parameter in self.llm.named_parameters() if parameter.requires_grad}
        self.named_trainable.update({'projection.' + name: p for name, p in self.projection.named_parameters() if p.requires_grad})

    def set_training(self, value):
        self.llm.train(value)
        self.projection.train(value)
        self.teacher.eval()

    def project(self, ids):
        with torch.no_grad():
            embeddings = self.teacher.item_emb(torch.tensor(ids, dtype=torch.long, device=self.device))
        return self.projection(embeddings)

    def check_texts(self, texts, marker, expected_history):
        tokenizer = self.llm.llm_tokenizer
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False)['input_ids']
        marker_id = tokenizer.convert_tokens_to_ids(marker)
        history_id = tokenizer.convert_tokens_to_ids('[HistoryEmb]')
        for ids, count in zip(encoded, expected_history):
            if len(ids) > 1024:
                raise DataError('prompt exceeds native 1024-token budget; refusing silent marker truncation')
            if ids.count(marker_id) != 1 or ids.count(history_id) != count:
                raise DataError('model marker positions do not match actual item counts')

    def training_loss(self, queries, epoch):
        if len(queries) < 2:
            raise DataError('pairwise uniformity requires batch size >= 2')
        prompts, histories, candidates, candidate_texts = ([], [], [], [])
        for query in queries:
            text, history = user_prompt(self.bundle, query)
            prompts.append(text)
            histories.append(history)
            ids = training_candidates(query, self.bundle['catalog'], self.args.seed, epoch)
            candidates.extend(ids)
            candidate_texts.extend((item_prompt(self.bundle, item) for item in ids))
        self.check_texts(prompts, '[UserOut]', list(map(len, histories)))
        self.check_texts(candidate_texts, '[ItemOut]', [1] * len(candidate_texts))
        with torch.no_grad():
            teacher_features = self.teacher(None, sequence_array(queries), None, None, mode='log_only')
        samples = {'text_input': prompts, 'log_emb': teacher_features, 'candidates_pos': candidate_texts, 'interact': [self.project(history) for history in histories], 'candidate_embs': self.project(candidates)}
        with torch.autocast('cuda', dtype=torch.float16):
            return self.llm(samples, mode=0)

    def training_features(self, queries, epoch):
        prompts_and_ids = [user_prompt(self.bundle, query) for query in queries]
        candidate_ids = [item for query in queries for item in training_candidates(query, self.bundle['catalog'], self.args.seed, epoch)]
        item_reps = self.encode([item_prompt(self.bundle, item) for item in candidate_ids], candidate_ids, '[ItemOut]')
        user_reps = self.encode([x[0] for x in prompts_and_ids], [x[1] for x in prompts_and_ids], '[UserOut]')
        with torch.no_grad():
            teacher_features = self.teacher(None, sequence_array(queries), None, None, mode='log_only')
        teacher_reps = self.llm.pred_user_CF2(teacher_features)
        return (user_reps, item_reps, teacher_reps)

    def full_batch_objective(self, outputs):
        user_reps, item_reps, teacher_reps = outputs
        rec_loss = self.llm.rec_loss(user_reps, item_reps)
        user_normalized = torch.nn.functional.normalize(user_reps, p=2, dim=1)
        teacher_normalized = torch.nn.functional.normalize(teacher_reps, p=2, dim=1)
        match_loss = self.llm.mse(user_normalized, teacher_normalized)
        match_loss = match_loss + self.llm.uniformity(user_normalized) + self.llm.uniformity(teacher_normalized)
        return (rec_loss + match_loss, {'rec_loss': float(rec_loss.detach()), 'match_loss': float(match_loss.detach())})

    def encode(self, texts, item_ids, marker):
        history_counts = [len(ids) for ids in item_ids] if marker == '[UserOut]' else [1] * len(texts)
        self.check_texts(texts, marker, history_counts)
        tokens = self.llm.llm_tokenizer(texts, return_tensors='pt', padding='longest', truncation=False).to(self.device)
        embeds = self.llm.llm_model.get_input_embeddings()(tokens['input_ids'])
        if marker == '[UserOut]':
            projected = [self.project(ids) for ids in item_ids]
            embeds = self.llm.replace_out_token_all(tokens, embeds, token=['[UserOut]', '[HistoryEmb]'], embs={'[HistoryEmb]': projected})
            head = self.llm.pred_user
        else:
            projected = self.project(item_ids)
            embeds = self.llm.replace_out_token_all_infer(tokens, embeds, token=['[ItemOut]', '[HistoryEmb]'], embs={'[HistoryEmb]': projected})
            head = self.llm.pred_item
        with torch.autocast('cuda', dtype=torch.float16):
            output = self.llm.llm_model(inputs_embeds=embeds, output_hidden_states=True, use_cache=False)
            positions = self.llm.get_embeddings(tokens, marker)
            representations = torch.cat([output.hidden_states[-1][i, ids].mean(0, keepdim=True) for i, ids in enumerate(positions)])
            result = head(representations)
        if not torch.isfinite(result).all():
            raise RuntimeError('nonfinite LLM representation')
        return result

    def score(self, queries, *, manager=None, stage=None, boundary=None):
        self.set_training(False)
        wanted = sorted({item for query in queries for item in query['candidate_item_indices']})
        item_batches = batch_indices(len(wanted), self.args.item_batch_size)
        items = manager.scoring(stage + '-items', [[f'item-{wanted[i]}' for i in indices] for indices in item_batches]) if manager else None
        users = manager.scoring(stage + '-users', query_batches(queries, self.args.eval_batch_size)) if manager else None
        cache = {}
        if items:
            for row in items.records(require_complete=False):
                cache[row['item_id']] = torch.tensor(row['representation'], dtype=getattr(torch, row['dtype']))
        with torch.no_grad():
            for index, indices in enumerate(item_batches):
                if items and index < items.completed_batches:
                    continue
                ids = [wanted[i] for i in indices]
                representations = self.encode([item_prompt(self.bundle, item) for item in ids], ids, '[ItemOut]').cpu()
                cache.update(zip(ids, representations))
                if items:
                    items.append(index, [{'query_id': f'item-{item}', 'item_id': item, 'dtype': str(rep.dtype).removeprefix('torch.'), 'representation': rep.tolist()} for item, rep in zip(ids, representations)])
                if boundary:
                    boundary()
            records = users.records(require_complete=False) if users else []
            for index, indices in enumerate(batch_indices(len(queries), self.args.eval_batch_size)):
                if users and index < users.completed_batches:
                    continue
                batch = selected_rows(queries, indices)
                prompts_and_ids = [user_prompt(self.bundle, query) for query in batch]
                representations = self.encode([x[0] for x in prompts_and_ids], [x[1] for x in prompts_and_ids], '[UserOut]')
                rows = []
                for query, user_rep in zip(batch, representations):
                    candidates = torch.stack([cache[item] for item in query['candidate_item_indices']]).to(self.device)
                    with torch.autocast('cuda', dtype=torch.float16):
                        scores = torch.matmul(candidates, user_rep).float().cpu().tolist()
                    rows.append(prediction_record(query, scores))
                records.extend(rows)
                if users:
                    users.append(index, rows)
                if boundary:
                    boundary()
        return users.records() if users else records

    def save(self, path, epoch):
        state = {name: parameter.detach().cpu() for name, parameter in self.named_trainable.items()}
        embeddings = self.llm.llm_model.get_input_embeddings().weight
        output = self.llm.llm_model.get_output_embeddings().weight
        torch.save({'trainable': state, 'extra_input_rows': embeddings[self.original_vocab:].detach().cpu(), 'extra_output_rows': output[self.original_vocab:].detach().cpu(), 'tokenizer_sha256': self.tokenizer_hash, 'base_model_manifest': self.base_manifest, 'teacher_sha256': file_digest(self.args.teacher), 'contract': teacher_contract(self.bundle), 'epoch': epoch}, path)

    def restore(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['contract'] != teacher_contract(self.bundle) or saved['base_model_manifest'] != self.base_manifest or saved['tokenizer_sha256'] != self.tokenizer_hash or (saved['teacher_sha256'] != file_digest(self.args.teacher)):
            raise DataError('checkpoint provenance differs from dataset, teacher, tokenizer, or backbone')
        if saved['trainable'].keys() != self.named_trainable.keys():
            raise DataError('checkpoint trainable parameter names do not match')
        with torch.no_grad():
            for name, parameter in self.named_trainable.items():
                parameter.copy_(saved['trainable'][name].to(parameter.device))
            self.llm.llm_model.get_input_embeddings().weight[self.original_vocab:].copy_(saved['extra_input_rows'].to(self.device))
            self.llm.llm_model.get_output_embeddings().weight[self.original_vocab:].copy_(saved['extra_output_rows'].to(self.device))

def persist_scores(path, records, args, bundle):
    result, ranks = evaluate_predictions(records, args, bundle, 'LLM-SRec')
    write_jsonl(path.with_suffix('.predictions.jsonl'), records)
    write_json(path.with_suffix('.metadata.json'), result['metadata'])
    write_json(path.with_suffix('.metrics.json'), result)
    write_jsonl(path.with_suffix('.ranks.jsonl'), ranks)
    return result
LLM_ARTIFACTS = ['best.pt', 'valid-best.predictions.jsonl', 'valid-best.metadata.json', 'valid-best.metrics.json', 'valid-best.ranks.jsonl', 'test.predictions.jsonl', 'test.metadata.json', 'test.metrics.json', 'test.ranks.jsonl']

def train_llm_epochs(args, bundle, model, optimizer, scheduler, manager):
    progress = manager.restore(model.restore, optimizer=optimizer, scheduler=scheduler)
    if progress is None:
        progress = {'phase': 'train', 'epoch': 1, 'next_batch': 0, 'global_step': 0, 'best': None, 'stale': 0, 'losses': [], 'events': []}

    def boundary(force=False):
        manager.checkpoint(progress, lambda path: model.save(path, progress['epoch']), optimizer=optimizer, scheduler=scheduler, artifacts=existing_artifacts(args.output, LLM_ARTIFACTS), force=force)
    if not args.resume:
        boundary(force=True)
    publish_events(args.output, progress['events'])
    valid = bundle['rows']['valid']
    if args.command == 'pilot':
        valid = valid[:args.pilot_valid_queries]
    while progress['phase'] in {'train', 'valid'}:
        epoch = progress['epoch']
        if progress['phase'] == 'train':
            model.set_training(True)
            batches = batch_indices(len(bundle['rows']['train']), args.effective_batch_size, seed=args.seed + epoch, min_size=2)
            if args.command == 'pilot':
                batches = [batches[0][:args.pilot_examples]]
            for batch_index in range(progress['next_batch'], len(batches)):
                batch = selected_rows(bundle['rows']['train'], batches[batch_index])
                microbatches = [selected_rows(batch, micro) for micro in batch_indices(len(batch), args.batch_size, min_size=2)]
                optimizer.zero_grad(set_to_none=True)
                loss, diagnostics = backward_cached(microbatches, lambda micro: model.training_features(micro, epoch), model.full_batch_objective, device=args.device)
                check_gradients(model.named_trainable)
                optimizer.step()
                scheduler.step()
                progress['losses'].append(loss)
                progress['next_batch'] = batch_index + 1
                progress['global_step'] += 1
                step = batch_index + 1
                if step == 1 or step % 20 == 0:
                    print(json.dumps({'epoch': epoch, 'step': step, 'steps': len(batches), 'loss': loss, **diagnostics, 'effective_batch_size': len(batch), 'microbatch_size': args.batch_size}), flush=True)
                boundary()
            progress['phase'] = 'valid'
            boundary(force=True)
        records = model.score(valid, manager=manager, stage=f'epoch-{epoch}-valid', boundary=boundary)
        result, _ = evaluate_predictions(records, args, bundle, 'LLM-SRec')
        metric = result['metrics'][args.selection_metric]
        peak = torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else 0
        event = {'epoch': epoch, 'mean_train_loss': sum(progress['losses']) / len(progress['losses']), 'validation': result['metrics'], 'gpu_peak_allocated_bytes': peak}
        progress['events'].append(event)
        publish_events(args.output, progress['events'])
        print(json.dumps(event), flush=True)
        if progress['best'] is None or metric > progress['best']:
            progress['best'], progress['stale'] = (metric, 0)
            model.save(args.output / 'best.pt', epoch)
            persist_scores(args.output / 'valid-best', records, args, bundle)
        else:
            progress['stale'] += 1
        done = epoch >= (1 if args.command == 'pilot' else args.epochs) or (args.patience and progress['stale'] >= args.patience)
        progress.update(phase='restore' if done else 'train', epoch=epoch + 1, next_batch=0, losses=[])
        boundary(force=True)
    return (progress, valid)

def llm_run(args, bundle, manager, base_manifest):
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise DataError('official 8-bit LLM-SRec runtime requires a CUDA GPU')
    write_json(args.output / 'backbone.manifest.json', base_manifest)
    model = Runtime(args, bundle, base_manifest)
    write_json(args.output / 'parameters.json', {'trainable': {name: parameter.numel() for name, parameter in model.named_trainable.items()}, 'trainable_total': sum((parameter.numel() for parameter in model.named_trainable.values())), 'frozen_backbone_parameters': sum((parameter.numel() for parameter in model.llm.llm_model.parameters())), 'frozen_teacher_parameters': sum((parameter.numel() for parameter in model.teacher.parameters())), 'backbone_manifest_sha256': digest(base_manifest), 'teacher_sha256': file_digest(args.teacher)})
    optimizer = torch.optim.Adam(model.named_trainable.values(), lr=args.lr, betas=(0.9, 0.98))
    group_count = len(batch_indices(len(bundle['rows']['train']), args.effective_batch_size, min_size=2))
    total_steps = args.epochs * group_count if args.command == 'train' else 1
    warmup_steps = min(total_steps, math.ceil(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_multiplier(step, total_steps, warmup_steps))
    torch.cuda.reset_peak_memory_stats(args.device)
    progress, valid = train_llm_epochs(args, bundle, model, optimizer, scheduler, manager)
    checkpoint = args.output / 'best.pt'
    best_records = [json.loads(line) for line in (args.output / 'valid-best.predictions.jsonl').read_text().splitlines()]
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    seed_all(args.seed + 1000003)
    restored = Runtime(args, bundle, base_manifest)
    restored.restore(checkpoint)

    def boundary(force=False):
        manager.checkpoint(progress, lambda path: restored.save(path, progress['epoch']), optimizer=optimizer, scheduler=scheduler, artifacts=existing_artifacts(args.output, LLM_ARTIFACTS), force=force)
    if progress['phase'] == 'restore':
        restoration = verify_restored_validation(lambda queries: restored.score(queries, manager=manager, stage='selected-validation-replay', boundary=boundary), valid, best_records)
        progress['restoration'] = restoration
        progress['phase'] = 'test' if args.command == 'train' else 'complete'
        boundary(force=True)
    if progress['phase'] == 'test':
        records = restored.score(bundle['rows']['test'], manager=manager, stage='final-test', boundary=boundary)
        persist_scores(args.output / 'test', records, args, bundle)
        progress['test_queries'] = len(records)
        progress['phase'] = 'complete'
        boundary(force=True)
    status = {'status': 'complete', 'run_kind': args.command, 'evaluation_protocol': evaluation_contract(args.expected_candidates), 'fresh_checkpoint_restore_verified': True, 'restore_atol': 0.0001, 'restore_rtol': 0.001, **progress['restoration'], 'selected_validation_value': progress['best'], 'selection_metric': args.selection_metric, 'checkpoint_sha256': file_digest(checkpoint), 'training_budget': 'five epochs by default; validation selects the checkpoint; no convergence claim', 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device)}
    if args.command == 'train':
        status['test_queries'] = progress['test_queries']
    else:
        status['limitation'] = 'one optimizer batch and a validation subset; infrastructure diagnostic only'
        status['pilot_examples'] = args.pilot_examples
    write_json(args.output / 'completion.json', status)
    print(json.dumps(status), flush=True)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit', 'teacher', 'pilot', 'train'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True, help='unmodified official LLM-SRec checkout')
    p.add_argument('--output', type=Path, required=True, help='new directory; existing paths refused')
    p.add_argument('--resume', action='store_true', help='resume only an identity-validated complete restart generation')
    p.add_argument('--checkpoint-seconds', type=float, default=600, help='periodic complete-batch checkpoint interval')
    p.add_argument('--dataset-name')
    p.add_argument('--protocol-sha256')
    p.add_argument('--expected-candidates', type=int, default=51, help='fixed evaluation total including the true item; default 51 is the user override to 50 negatives')
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--omit-dates', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--teacher-epochs', type=int, default=200)
    p.add_argument('--teacher-batch-size', type=int, default=256)
    p.add_argument('--teacher-lr', type=float, default=0.001)
    p.add_argument('--teacher-l2-grid', type=str, default='0,0.000001,0.00001,0.0001,0.001')
    p.add_argument('--teacher-patience', type=int, default=10)
    p.add_argument('--teacher-diagnostic', action='store_true', help='label a shortened teacher smoke run; cannot feed full LLM training')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--effective-batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=2)
    p.add_argument('--item-batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--selection-metric', choices=('HR@10', 'NDCG@10'), default='NDCG@10')
    p.add_argument('--patience', type=int, default=0, help='0 runs all 5 epochs; positive enables explicitly adapted early stopping')
    p.add_argument('--pilot-valid-queries', type=int, default=4)
    p.add_argument('--pilot-examples', type=int, default=2)
    return p

def main():
    global torch, np
    args = parser().parse_args()
    evaluation_contract(args.expected_candidates)
    for key in ('epochs', 'teacher_epochs', 'max_length', 'teacher_batch_size', 'eval_batch_size', 'item_batch_size', 'pilot_valid_queries'):
        if getattr(args, key) < 1:
            raise DataError(f"--{key.replace('_', '-')} must be positive")
    if args.batch_size < 2:
        raise DataError('--batch-size must be at least 2')
    if args.effective_batch_size < args.batch_size or not 2 <= args.pilot_examples <= args.effective_batch_size:
        raise DataError('effective batch must cover microbatch size, and pilot-examples must lie in [2, effective batch]')
    if args.patience < 0 or args.teacher_patience < 0 or (not 0 <= args.warmup_ratio <= 1):
        raise DataError('patience must be nonnegative and warmup-ratio must lie in [0,1]')
    args.teacher_l2_grid = [float(value) for value in args.teacher_l2_grid.split(',')]
    if not args.teacher_l2_grid or any((not math.isfinite(value) or value < 0 for value in args.teacher_l2_grid)):
        raise DataError('--teacher-l2-grid must be a comma-separated list of nonnegative finite values')
    args.source = args.source.resolve()
    bundle = load_data(args.data, args.expected_candidates, args.max_length, args.omit_dates)
    protocol_hash(bundle, args.protocol_sha256)
    for query in bundle['rows']['train']:
        training_candidates(query, bundle['catalog'], args.seed, 1)
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0:
        raise DataError('--checkpoint-seconds must be positive and finite')
    if args.resume:
        if args.command == 'audit' or not args.output.is_dir():
            raise DataError('--resume needs an existing non-audit run directory')
    elif os.path.lexists(args.output):
        raise DataError(f'refusing to overwrite existing run directory: {args.output}')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    manifest = metadata(args, bundle, 'LLM-SRec' if args.command != 'teacher' else 'SASRec-teacher')
    manifest.update({'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'counts': {split: len(rows) for split, rows in bundle['rows'].items()}, 'teacher_contract': teacher_contract(bundle), 'training_candidates': 'positive first + 3 distinct negatives excluding full history and target; query/epoch/seed deterministic', 'evaluation_candidates': 'source IDs and order exactly preserved', 'llm_optimizer': 'Adam betas=(0.9,0.98); linear warmup then cosine, per optimizer step', 'batch_loss': 'two-pass gradient cache computes native loss, including pairwise uniformity, on full effective batch; RNG replay checked', 'batch_tail_policy': 'retain every query; final partial effective batch allowed; singleton tail merged with preceding batch', 'precision': '8-bit frozen Llama2-7B with FP16 CUDA autocast; full representation loss in FP32', 'validation': 'complete fixed split each epoch; test once after best restore', 'upstream_adaptations': ['query-based last-position SASRec supervision', 'teacher held in eval mode', 'per-event timestamps', 'fixed-candidate scoring', 'strict 1024-token budget checks', 'stable validation and complete trainable/special-token checkpoint', 'singleton training tail merged', 'Llama2-7B shared backbone replaces native 3B', 'full-batch loss gradient cache with RNG replay']})
    manifest['choices_unspecified_by_manuscript'] = {'llm_lr': args.lr, 'warmup_ratio': args.warmup_ratio, 'teacher_max_epochs': args.teacher_epochs, 'teacher_validation_patience': args.teacher_patience, 'teacher_l2_grid': args.teacher_l2_grid, 'teacher_l2_scope': 'all parameters via Adam weight_decay', 'selection_metric': args.selection_metric, 'teacher_history_limit': args.max_length, 'llm_history_limit': 10, 'quantization': '8-bit frozen base follows LLM-SRec source', 'prompt_sampling': 'official LLM-SRec representation prompt is fixed; no invented dynamic prompt variants', 'llm_learning_objective_precision': 'FP32 losses on FP16/FP32 cached representations'}
    if not args.resume:
        write_json(args.output / 'run.manifest.json', manifest)
        write_json(args.output / 'id_maps.json', bundle['mapping'])
    if args.command == 'audit':
        print(json.dumps({'status': 'data audit passed', 'counts': manifest['counts']}), flush=True)
        return
    import torch
    import numpy as np
    import importlib.metadata
    versions = {}
    for name in ('torch', 'numpy', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'sentencepiece', 'safetensors'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    environment = {'python': sys.version, 'versions': versions, 'cuda': torch.version.cuda, 'device': args.device, 'gpu_name': torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') and torch.cuda.is_available() else None}
    seed_all(args.seed)
    base_manifest = local_model_manifest(args.model_path) if args.command != 'teacher' else None
    operational = {'resume', 'checkpoint_seconds', 'output', 'data', 'source', 'model_path', 'teacher'}
    contract = {'metadata': metadata(args, bundle, manifest['model']), 'arguments': {key: value for key, value in vars(args).items() if key not in operational}, 'runtime': runtime_context(args.device), 'environment': environment, 'backbone': base_manifest, 'teacher': {name: file_digest(args.teacher.with_name(name)) for name in ('teacher.pth', 'teacher.metadata.json', 'teacher.completion.json')} if args.teacher is not None else None}
    with RestartManager(args.output, contract, resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:
        if not args.resume:
            write_json(args.output / 'environment.json', environment)
        if args.command == 'teacher':
            train_teacher(args, bundle, manager)
        else:
            llm_run(args, bundle, manager, base_manifest)
if __name__ == '__main__':
    try:
        main()
    except (DataError, RestartError, OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(2)
