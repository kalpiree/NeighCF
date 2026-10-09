from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'llmsrec'))
import run as shared
from data import DataError, batch_indices, digest, file_digest, load_data, prediction_record, protocol_hash, teacher_contract, training_candidates, verify_restored_validation
from gradient_cache import backward_cached
sys.path.insert(0, str(HERE))
from graph_data import GraphError, QueryGraphs, build_global_graph, load_features, source_functions, dataset_policy, recommendation_prompt, DATASET_POLICIES
from integrations.resume import RestartManager, runtime_context

def ucg_fingerprint(source, policy_name):
    policy = dataset_policy(policy_name)
    return {name: file_digest(source / name) for name in ('model/model_interface.py', 'model/gce_gnn.py', 'model/mlp_projector.py', policy['loader'], policy['prompt'])}

def run_metadata(args, bundle):
    return {'model': 'UCGRec-topk', 'dataset': args.dataset_name or args.data.name, 'seed': args.seed, 'protocol_sha256': protocol_hash(bundle, args.protocol_sha256), 'run_kind': args.command, 'evaluation_candidates': args.expected_candidates, 'dataset_policy': dataset_policy(args.dataset_policy), 'feature_usage': 'unused by native Steam random long-term / recent short-term selection' if args.dataset_policy == 'steam_native_v1' else 'semantic content vectors select long-term history', 'recommendation_prompt_sha256': digest(recommendation_prompt(args.ucg_source, args.dataset_policy)), 'comparison_scope': 'reconstruction of NeighCF published experimental protocol; original split files unavailable', 'backbone': 'meta-llama/Llama-2-7b-hf', 'precision': 'BF16 LoRA; FP32 full-batch objective', 'mapping_sha256': bundle['mapping_sha256'], 'source_file_sha256': bundle['source_file_sha256'], 'ucgrec_source_sha256': ucg_fingerprint(args.ucg_source, args.dataset_policy), 'graph_events_sha256': bundle['global_graph']['training_events_sha256'], 'feature_file_sha256': file_digest(args.features) if args.features else None, 'feature_manifest_sha256': file_digest(args.feature_manifest) if args.feature_manifest else None, 'shared_source_sha256': {name: file_digest(HERE.parents[1] / name) for name in ('integrations/resume.py', 'evaluate.py', 'adapters/llmsrec_data.py')}, 'integration_sha256': {**{name: file_digest(HERE / name) for name in ('run.py', 'graph_data.py', 'features.py')}, **{'shared_llmsrec/' + name: file_digest(HERE.parent / 'llmsrec' / name) for name in ('run.py', 'data.py', 'gradient_cache.py')}}}

class Runtime(shared.Runtime):

    def __init__(self, args, bundle, base_manifest):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        from peft import LoraConfig, TaskType, get_peft_model
        self.args, self.bundle, self.base_manifest = (args, bundle, base_manifest)
        self.device = args.device
        self.teacher = shared.load_teacher(args, bundle)
        if args.dataset_policy == 'steam_native_v1' and args.features is None:
            features, feature_manifest = (None, {'feature_kind': 'unused_by_native_steam', 'dataset_policy': args.dataset_policy})
        else:
            features, feature_manifest = load_features(args.features, args.feature_manifest, bundle, args.dataset_policy)
        self.feature_manifest = feature_manifest
        self.graphs = QueryGraphs(args.ucg_source, bundle, bundle['global_graph'], features, args.graph_evaluation_seed, args.dataset_policy)
        self.tokenizer = LlamaTokenizer.from_pretrained(str(args.model_path), local_files_only=True, use_fast=False)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]', '[CansEmb]', '[GraphEmb]', '[LTGraphEmb]', '[STGraphEmb]', '[ItemOut]', '[UserOut]']})
        self.original_vocab = 32000
        backbone = LlamaForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16, device_map=args.device, attn_implementation='sdpa')
        backbone.resize_token_embeddings(len(self.tokenizer))
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        self.backbone = get_peft_model(backbone, config)
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.backbone.enable_input_require_grads()
        self.backbone.config.use_cache = False
        self.llm = SimpleNamespace(llm_model=self.backbone)
        gnn_module = shared.module_from_file('ucgrec_native_gnn', args.ucg_source / 'model/gce_gnn.py')
        projector_module = shared.module_from_file('ucgrec_native_projector', args.ucg_source / 'model/mlp_projector.py')
        self.graph_encoder = gnn_module.GCE_GNN(hiddenSize=64, hop=args.hop, ls_mode='LS').to(args.device)
        self.projector = projector_module.MlpProjector(rec_size=64, llm_size=4096).to(args.device)
        self.heads = torch.nn.ModuleDict({'user': torch.nn.Sequential(torch.nn.Linear(4096, 2048), torch.nn.LayerNorm(2048), torch.nn.LeakyReLU(), torch.nn.Linear(2048, 128)), 'item': torch.nn.Sequential(torch.nn.Linear(4096, 2048), torch.nn.LayerNorm(2048), torch.nn.LeakyReLU(), torch.nn.Linear(2048, 128)), 'auxiliary': torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.LayerNorm(128), torch.nn.GELU(), torch.nn.Linear(128, 128))}).to(args.device)
        for head in self.heads.values():
            torch.nn.init.xavier_normal_(head[0].weight)
            torch.nn.init.xavier_normal_(head[3].weight)
        self.native_losses = source_functions(args.ucg_source / 'model/model_interface.py', 'MInterface', ('rec_loss', 'uniformity'), {'torch': torch, 'F': torch.nn.functional})
        self.modules = {'backbone': self.backbone, 'graph_encoder': self.graph_encoder, 'projector': self.projector, 'heads': self.heads}
        self.named_trainable = {prefix + '.' + name: parameter for prefix, module in self.modules.items() for name, parameter in module.named_parameters() if parameter.requires_grad}
        self.tokenizer_hash = digest(self.tokenizer.get_vocab())
        self.prompt = recommendation_prompt(args.ucg_source, args.dataset_policy)
        self.padding = len(bundle['items'])

    def set_training(self, training):
        for module in self.modules.values():
            module.train(training)
        self.teacher.eval()

    def item_embedding(self, native_ids):
        ids = torch.as_tensor(native_ids, device=self.device, dtype=torch.long)
        canonical = torch.where(ids == self.padding, 0, ids + 1)
        return self.teacher.item_emb(canonical)

    def title(self, native_id):
        item = self.bundle['mapping']['index_to_item'][native_id + 1]
        return self.bundle['items'][item]['title']

    def graph_representation(self, graph):
        seq = torch.tensor(graph['seq'], device=self.device)
        hidden = self.item_embedding(seq)
        arguments = [graph['lengths'], torch.tensor(graph['items'], device=self.device), torch.tensor(graph['g_adj'], device=self.device), torch.tensor(graph['g_weight'], dtype=torch.float32, device=self.device), torch.tensor(graph['local'], device=self.device), torch.tensor(graph['aliases'], device=self.device), self.item_embedding]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                output = checkpoint(lambda h: self.graph_encoder(h, *arguments), hidden, use_reentrant=False, preserve_rng_state=True)
            else:
                output = self.graph_encoder(hidden, *arguments)
            return self.projector(output)

    def user_inputs(self, queries, epoch):
        texts, replacements, last_ids = ([], [], [])
        for query in queries:
            graph = self.graphs.sample(query, training_seed=self.args.seed, epoch=epoch)
            representation = self.graph_representation(graph)
            text = self.prompt
            for view, placeholder, marker in ((0, '[HistoryHere]', '[GraphEmb]'), (1, '[LongTermSeq]', '[LTGraphEmb]'), (2, '[ShortTermSeq]', '[STGraphEmb]')):
                text = text.replace(placeholder, ', '.join((self.title(item) + ' ' + marker for item in graph['views'][view])))
            texts.append(text)
            replacements.append({marker: representation[view, :graph['lengths'][view]] for view, marker in ((0, '[GraphEmb]'), (1, '[LTGraphEmb]'), (2, '[STGraphEmb]'))})
            last_ids.append(graph['seq'][-1])
        return (texts, replacements, last_ids)

    def item_inputs(self, native_ids):
        texts = ['The item title and item embedding are as follows: ' + self.title(item) + '[CansEmb], then generate item representation token:[ItemOut]' for item in native_ids]
        vectors = self.projector(self.item_embedding(native_ids))
        return (texts, [{'[CansEmb]': vector.unsqueeze(0)} for vector in vectors])

    def encode(self, texts, replacements, output_marker, head):
        token_rows = self.tokenizer(texts, truncation=False)['input_ids']
        if any((len(row) > 1024 for row in token_rows)):
            raise DataError('UCGRec prompt exceeds 1024 tokens; refusing silent truncation')
        tokens = self.tokenizer(texts, padding='longest', truncation=False, return_tensors='pt').to(self.device)
        embeds = self.backbone.get_input_embeddings()(tokens['input_ids']).clone()
        for index, mapping in enumerate(replacements):
            for marker, vectors in mapping.items():
                positions = (tokens['input_ids'][index] == self.tokenizer.convert_tokens_to_ids(marker)).nonzero().flatten()
                if len(positions) != len(vectors):
                    raise DataError('graph/item marker count differs from projected vectors')
                embeds[index, positions] = vectors.to(embeds.dtype)
        output_id = self.tokenizer.convert_tokens_to_ids(output_marker)
        positions = [(tokens['input_ids'][index] == output_id).nonzero().flatten() for index in range(len(texts))]
        if any((len(position) != 1 for position in positions)):
            raise DataError('exactly one output marker is required')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = self.backbone(inputs_embeds=embeds, output_hidden_states=True, use_cache=False)
            vectors = torch.cat([output.hidden_states[-1][index, position].mean(0, keepdim=True) for index, position in enumerate(positions)])
            return self.heads[head](vectors)

    def training_features(self, queries, epoch):
        candidates = [item - 1 for query in queries for item in training_candidates(query, self.bundle['catalog'], self.args.seed, epoch, count=5)]
        item_texts, item_replacements = self.item_inputs(candidates)
        items = self.encode(item_texts, item_replacements, '[ItemOut]', 'item')
        texts, replacements, last_ids = self.user_inputs(queries, epoch)
        users = self.encode(texts, replacements, '[UserOut]', 'user')
        auxiliary = self.heads['auxiliary'](self.item_embedding(last_ids))
        return (users, items, auxiliary)

    def full_batch_objective(self, outputs):
        users, items, auxiliary = outputs
        rec = self.native_losses['rec_loss'](self, users, items)
        users = torch.nn.functional.normalize(users, dim=1)
        auxiliary = torch.nn.functional.normalize(auxiliary, dim=1)
        match = torch.nn.functional.mse_loss(users, auxiliary)
        match = match + self.native_losses['uniformity'](self, users) + self.native_losses['uniformity'](self, auxiliary)
        return (rec + match, {'rec_loss': rec.detach().item(), 'match_loss': match.detach().item()})

    def score(self, queries, *, manager=None, stage=None, boundary=None):
        self.set_training(False)
        wanted = sorted({item - 1 for query in queries for item in query['candidate_item_indices']})
        item_groups = batch_indices(len(wanted), self.args.item_batch_size)
        item_journal = manager.scoring(stage + '-items', [[str(wanted[i]) for i in batch] for batch in item_groups]) if manager else None
        user_groups = batch_indices(len(queries), self.args.eval_batch_size)
        journal = manager.scoring(stage + '-users', [[queries[i]['source']['query_id'] for i in batch] for batch in user_groups]) if manager else None
        cache = {}
        if item_journal:
            for row in item_journal.records(require_complete=False):
                cache[int(row['query_id'])] = torch.tensor(row['vector'], dtype=getattr(torch, row['dtype']))
        records = journal.records(require_complete=False) if journal else []
        with torch.no_grad():
            for batch_index, indices in enumerate(item_groups):
                if item_journal and batch_index < item_journal.completed_batches:
                    continue
                items = [wanted[index] for index in indices]
                texts, replacements = self.item_inputs(items)
                vectors = self.encode(texts, replacements, '[ItemOut]', 'item').cpu()
                cache.update(zip(items, vectors))
                if item_journal:
                    item_journal.append(batch_index, [{'query_id': str(item), 'vector': vector.float().tolist(), 'dtype': str(vector.dtype).split('.')[-1]} for item, vector in zip(items, vectors)])
                    boundary()
            for batch_index, indices in enumerate(user_groups):
                if journal and batch_index < journal.completed_batches:
                    continue
                batch = [queries[index] for index in indices]
                texts, replacements, _ = self.user_inputs(batch, None)
                vectors = self.encode(texts, replacements, '[UserOut]', 'user')
                current = []
                for query, vector in zip(batch, vectors):
                    candidates = torch.stack([cache[item - 1] for item in query['candidate_item_indices']]).to(self.device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        scores = (candidates @ vector).float().cpu().tolist()
                    current.append(prediction_record(query, scores))
                records.extend(current)
                if journal:
                    journal.append(batch_index, current)
                    boundary()
        return journal.records() if journal else records

    def graph_contract(self):
        return {'graph_events_sha256': self.bundle['global_graph']['training_events_sha256'], 'features_sha256': file_digest(self.args.features) if self.args.features else None, 'feature_manifest_sha256': file_digest(self.args.feature_manifest) if self.args.feature_manifest else None, 'ucg_source': ucg_fingerprint(self.args.ucg_source, self.args.dataset_policy), 'hop': self.args.hop, 'evaluation_graph_seed': self.args.graph_evaluation_seed, 'dataset_policy': dataset_policy(self.args.dataset_policy), 'recommendation_prompt_sha256': digest(self.prompt)}

    def save(self, path, epoch):
        super().save(path, epoch)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        checkpoint['ucgrec_graph_contract'] = self.graph_contract()
        torch.save(checkpoint, path)

    def restore(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('ucgrec_graph_contract') != self.graph_contract():
            raise DataError('UCGRec graph, features, source, or hyperparameters changed since checkpoint')
        del saved
        super().restore(path)

def persist(path, records, args, bundle):
    from evaluate import evaluate_records
    result, ranks = evaluate_records(records, run_metadata(args, bundle), args.expected_candidates)
    shared.write_jsonl(path.with_suffix('.predictions.jsonl'), records)
    shared.write_json(path.with_suffix('.metadata.json'), result['metadata'])
    shared.write_json(path.with_suffix('.metrics.json'), result)
    shared.write_jsonl(path.with_suffix('.ranks.jsonl'), ranks)
    return result

def validate_training_controls(args):
    if args.command == 'train' and (args.epochs, args.effective_batch_size, args.expected_candidates) != (5, 128, 51):
        raise DataError('full UCGRec requires five epochs, effective batch128 and51 fixed candidates')

def package_versions():
    import importlib.metadata
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'numpy', 'accelerate'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions

def train_loop(args, bundle):
    validate_training_controls(args)
    base_manifest = shared.local_model_manifest(args.model_path)
    if args.command == 'train' and base_manifest != json.loads(args.pilot_receipt.with_name('backbone.manifest.json').read_text()):
        raise DataError('backbone files differ from the successful CUDA pilot')
    model = Runtime(args, bundle, base_manifest)
    optimizer = torch.optim.Adam([{'params': [p for name, p in model.named_trainable.items() if name.startswith('backbone.')], 'weight_decay': 0.0}, {'params': [p for name, p in model.named_trainable.items() if not name.startswith('backbone.')], 'weight_decay': 1e-05}], lr=args.lr, betas=(0.9, 0.999))
    groups_per_epoch = len(batch_indices(len(bundle['rows']['train']), args.effective_batch_size, min_size=2))
    total_epochs = args.epochs if args.command == 'train' else 1
    total_steps = total_epochs * groups_per_epoch if args.command == 'train' else 1
    from data import cosine_multiplier
    warmup = min(total_steps, math.ceil(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_multiplier(step, total_steps, warmup))
    contract = {'kind': 'ucgrec-restart-v1', 'metadata': run_metadata(args, bundle), 'base_manifest': base_manifest, 'teacher_sha256': file_digest(args.teacher), 'teacher_metadata_sha256': file_digest(args.teacher.with_name('teacher.metadata.json')), 'runtime': runtime_context(args.device), 'package_versions': package_versions(), 'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in {'resume', 'checkpoint_seconds'}}}
    valid = bundle['rows']['valid'] if args.command == 'train' else bundle['rows']['valid'][:args.pilot_valid_queries]
    checkpoint = args.output / 'best.pt'
    with RestartManager(args.output, contract, resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:
        shared.write_json(args.output / 'backbone.manifest.json', base_manifest)
        shared.write_json(args.output / 'parameters.json', {'trainable_total': sum((p.numel() for p in model.named_trainable.values())), 'trainable_by_name': {name: p.numel() for name, p in model.named_trainable.items()}, 'frozen_backbone_total': sum((p.numel() for p in model.backbone.parameters() if not p.requires_grad))})
        state = manager.restore(model.restore, optimizer, scheduler)
        if state is None:
            state = {'phase': 'train', 'epoch': 1, 'next_batch': 0, 'global_step': 0, 'losses': [], 'best': None, 'selected_epoch': None, 'largest_optimizer_batch': 0, 'events': [], 'train_examples': 0, 'peak_gpu_allocated_bytes': 0}
        artifact_names = ('best.pt', 'valid-best.predictions.jsonl', 'valid-best.metadata.json', 'valid-best.metrics.json', 'valid-best.ranks.jsonl', 'test.predictions.jsonl', 'test.metadata.json', 'test.metrics.json', 'test.ranks.jsonl', 'completion.json')

        def boundary(force=False):
            state['peak_gpu_allocated_bytes'] = max(state['peak_gpu_allocated_bytes'], torch.cuda.max_memory_allocated(args.device))
            return manager.checkpoint(state, lambda path: model.save(path, state['epoch']), optimizer, scheduler, artifacts=shared.existing_artifacts(args.output, artifact_names), force=force)
        if state['phase'] == 'complete':
            shared.write_json(args.output / 'completion.json', state['completion'])
            print(json.dumps(state['completion']), flush=True)
            return
        shared.publish_events(args.output, state['events'])
        boundary(force=True)
        torch.cuda.reset_peak_memory_stats(args.device)
        while state['phase'] in {'train', 'validation'}:
            epoch = state['epoch']
            groups = batch_indices(len(bundle['rows']['train']), args.effective_batch_size, seed=args.seed + epoch, min_size=2)
            if args.command == 'pilot':
                groups = [groups[0][:args.pilot_examples]]
            if not 0 <= state['next_batch'] <= len(groups) or state['global_step'] != (epoch - 1) * len(groups) + state['next_batch']:
                raise DataError('saved UCGRec optimizer position differs from deterministic batch cursor')
            if state['phase'] == 'train':
                model.set_training(True)
                for step in range(state['next_batch'], len(groups)):
                    batch = [bundle['rows']['train'][index] for index in groups[step]]
                    microbatches = [[batch[index] for index in micro] for micro in batch_indices(len(batch), args.batch_size)]
                    optimizer.zero_grad(set_to_none=True)
                    loss, details = backward_cached(microbatches, lambda micro: model.training_features(micro, epoch), model.full_batch_objective, device=args.device)
                    shared.check_gradients(model.named_trainable)
                    optimizer.step()
                    scheduler.step()
                    state['losses'].append(loss)
                    state['largest_optimizer_batch'] = max(state['largest_optimizer_batch'], len(batch))
                    state['train_examples'] += len(batch)
                    state['global_step'] += 1
                    state['next_batch'] = step + 1
                    if step == 0 or (step + 1) % 20 == 0:
                        print(json.dumps({'epoch': epoch, 'step': step + 1, 'steps': len(groups), 'effective_batch': len(batch), 'loss': loss, **details}), flush=True)
                    boundary()
                state['phase'] = 'validation'
                boundary(force=True)
            records = model.score(valid, manager=manager, stage=f'validation-epoch-{epoch}', boundary=boundary)
            from evaluate import evaluate_records
            result, _ = evaluate_records(records, run_metadata(args, bundle), args.expected_candidates)
            value = result['metrics'][args.selection_metric]
            event = {'epoch': epoch, 'mean_training_loss': sum(state['losses']) / len(state['losses']), 'validation': result['metrics'], 'peak_gpu_allocated_bytes': max(state['peak_gpu_allocated_bytes'], torch.cuda.max_memory_allocated(args.device))}
            state['events'].append(event)
            shared.publish_events(args.output, state['events'])
            print(json.dumps(event), flush=True)
            if state['best'] is None or value > state['best']:
                state['best'], state['selected_epoch'] = (value, epoch)
                model.save(checkpoint, epoch)
                persist(args.output / 'valid-best', records, args, bundle)
            state.update(epoch=epoch + 1, next_batch=0, losses=[], phase='train' if epoch < total_epochs else 'restore')
            boundary(force=True)
        if state['global_step'] != total_steps or (args.command == 'train' and state['train_examples'] != len(bundle['rows']['train']) * args.epochs):
            raise DataError('UCGRec training did not cover its declared complete budget')
        best_records = [json.loads(line) for line in (args.output / 'valid-best.predictions.jsonl').read_text().splitlines()]
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        shared.seed_all(args.seed + 1000003)
        model = Runtime(args, bundle, base_manifest)
        model.restore(checkpoint)
        if state['phase'] == 'restore':
            state['restoration'] = verify_restored_validation(lambda queries: model.score(queries, manager=manager, stage='selected-validation-replay', boundary=boundary), valid, best_records)
            state['phase'] = 'test' if args.command == 'train' else 'finish'
            boundary(force=True)
        if state['phase'] == 'test':
            records = model.score(bundle['rows']['test'], manager=manager, stage='final-test', boundary=boundary)
            persist(args.output / 'test', records, args, bundle)
            state['test_queries'] = len(records)
            state['phase'] = 'finish'
            boundary(force=True)
        completion = {'status': 'complete', 'run_kind': args.command, 'diagnostic': args.command == 'pilot', 'checkpoint_sha256': file_digest(checkpoint), 'fresh_checkpoint_restore_verified': True, 'largest_optimizer_batch': state['largest_optimizer_batch'], **state['restoration'], 'selected_validation_value': state['best'], 'selection_metric': args.selection_metric, 'selected_epoch': state['selected_epoch'], 'optimizer_steps': state['global_step'], 'train_examples': state['train_examples'], 'expected_candidates': args.expected_candidates, 'peak_gpu_allocated_bytes': max(state['peak_gpu_allocated_bytes'], torch.cuda.max_memory_allocated(args.device))}
        if args.command == 'train':
            completion['test_queries'] = state['test_queries']
        shared.write_json(args.output / 'completion.json', completion)
        state.update(phase='complete', completion=completion)
        boundary(force=True)
        print(json.dumps(completion), flush=True)

def main():
    global torch, np
    p = shared.parser()
    p.description = __doc__
    p.add_argument('--ucg-source', type=Path, required=True, help='official UCGRec_topk_ver directory')
    p.add_argument('--dataset-policy', choices=tuple(DATASET_POLICIES), required=True)
    p.add_argument('--train-events', type=Path)
    p.add_argument('--features', type=Path)
    p.add_argument('--feature-manifest', type=Path)
    p.add_argument('--graph-evaluation-seed', type=int, default=20260930)
    p.add_argument('--hop', type=int, default=4)
    p.add_argument('--pilot-receipt', type=Path, help='successful full-batch128 pilot completion.json, required by train')
    p.set_defaults(batch_size=1, item_batch_size=4, lr=0.0003, warmup_ratio=0.05)
    args = p.parse_args()
    validate_training_controls(args)
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0:
        p.error('checkpoint interval must be finite and positive')
    if args.command == 'teacher':
        p.error('use the shared llmsrec teacher command; this entry point trains UCGRec only')
    if args.batch_size < 1 or args.effective_batch_size < 2 or (not 2 <= args.pilot_examples <= args.effective_batch_size):
        p.error('microbatch>=1 and effective batch>=2 are required')
    if args.batch_size > args.effective_batch_size or any((getattr(args, key) < 1 for key in ('epochs', 'max_length', 'eval_batch_size', 'item_batch_size', 'pilot_valid_queries'))):
        p.error('positive sequence/batch/epoch sizes and microbatch<=effectivebatch are required')
    if args.patience != 0:
        p.error('UCGRec uses the declared fixed five-epoch budget; early stopping is unsupported')
    if not 0 <= args.warmup_ratio <= 1 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error('learning rate must be finite/positive and warmup-ratio must lie in [0,1]')
    if args.hop < 1 or args.hop > 4:
        p.error('hop must lie in [1,4]; MovieLens published/source configuration uses4')
    if bool(args.features) != bool(args.feature_manifest):
        p.error('features and feature-manifest must be supplied together')
    if args.command != 'audit' and args.dataset_policy != 'steam_native_v1' and (not args.features or not args.feature_manifest):
        p.error('UCGRec requires --features and --feature-manifest: official features or Llama3-generated descriptions encoded with SBERT')
    if args.command == 'train':
        if args.pilot_receipt is None:
            p.error('full training requires --pilot-receipt from successful full-batch128 CUDA pilot')
        receipt = json.loads(args.pilot_receipt.read_text())
        pilot_manifest = json.loads(args.pilot_receipt.with_name('run.manifest.json').read_text())
        if receipt.get('status') != 'complete' or receipt.get('run_kind') != 'pilot' or (not receipt.get('fresh_checkpoint_restore_verified')) or (receipt.get('largest_optimizer_batch') != 128) or (pilot_manifest['arguments'].get('pilot_examples') != args.effective_batch_size):
            p.error('pilot receipt does not prove the requested effective batch and fresh restoration')
    args.source, args.ucg_source = (args.source.resolve(), args.ucg_source.resolve())
    bundle = load_data(args.data, args.expected_candidates, args.max_length, omit_dates=True)
    for item in bundle['items'].values():
        if any((marker in item['title'] for marker in ('[CansEmb]', '[GraphEmb]', '[LTGraphEmb]', '[STGraphEmb]', '[PH]'))):
            raise DataError('reserved UCGRec marker in item title')
    protocol_hash(bundle, args.protocol_sha256)
    events = args.train_events or args.data / 'train_interactions.jsonl'
    timestamp_meta = bundle['source_split_metadata'].get('timestamp', {})
    timestamp_unit = timestamp_meta.get('unit') if isinstance(timestamp_meta, dict) else None
    timestamp_unit = timestamp_unit or bundle['source_split_metadata'].get('timestamp_unit')
    bundle['global_graph'] = build_global_graph(events, bundle['mapping'], bundle['rows'], timestamp_unit)
    if args.command == 'train':
        current = run_metadata(args, bundle)
        for field in ('model', 'protocol_sha256', 'mapping_sha256', 'source_file_sha256', 'ucgrec_source_sha256', 'graph_events_sha256', 'feature_file_sha256', 'feature_manifest_sha256', 'integration_sha256', 'shared_source_sha256', 'dataset_policy', 'recommendation_prompt_sha256', 'evaluation_candidates', 'feature_usage'):
            if current[field] != pilot_manifest.get(field):
                raise DataError(f'successful pilot differs from current {field}')
        previous_args = pilot_manifest['arguments']
        if args.hop != previous_args.get('hop') or args.graph_evaluation_seed != previous_args.get('graph_evaluation_seed'):
            raise DataError('graph configuration differs from successful pilot')
        for field in ('lr', 'warmup_ratio', 'max_length', 'expected_candidates', 'dataset_policy'):
            if getattr(args, field) != previous_args.get(field):
                raise DataError(f'scientific setting {field} differs from successful pilot')
        if any((getattr(args, field) > previous_args[field] for field in ('batch_size', 'item_batch_size', 'eval_batch_size'))):
            raise DataError('requested training/item/user batch exceeds CUDA-tested pilot sizes')
    for query in bundle['rows']['train']:
        training_candidates(query, bundle['catalog'], args.seed, 1, count=5)
    if args.output.exists() and (not args.resume):
        p.error('output already exists; use a new run directory or explicit verified resume')
    if args.resume and (args.command == 'audit' or not args.output.is_dir()):
        p.error('resume requires an existing pilot/train output')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest = run_metadata(args, bundle)
    manifest.update({'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in {'resume', 'checkpoint_seconds'}}, 'counts': {split: len(rows) for split, rows in bundle['rows'].items()}, 'source_quirks_preserved': ['incoming-edge walks choose highest node popularity with probability.7, random with.3', 'walk length11 including start, int(60/history_length) walks per item', 'original-history GNN context broadcast across all3views', 'auxiliary loss matches final raw right-padded item embedding, not SASRec sequence representation'], 'explicit_repairs': ['train-only explicit event graph', 'actual padding instead of phantom item0 neighbors', 'short histories retained with minimum1 long/short item', 'positive +4 negatives for training', 'fixed51candidates and fresh per-pass metrics', 'full-batch128 gradient-cache loss', 'complete checkpoint including frozen added token rows'], 'protocol_controls': 'Llama2-7B, five epochs, effective batch128, warmup+cosine, validation selection; five seeds required', 'full_batch_loss': 'source recommendation + MSE + two pairwise uniformity terms, including cross-microbatch pairs', 'paper': 'https://aclanthology.org/2026.findings-acl.175.pdf'})
    if args.resume:
        if json.loads((args.output / 'run.manifest.json').read_text()) != manifest:
            raise DataError('UCGRec run manifest changed; refusing resume')
    else:
        shared.write_json(args.output / 'run.manifest.json', manifest)
    shared.write_json(args.output / 'id_maps.json', bundle['mapping'])
    shared.write_json(args.output / 'graph.manifest.json', {key: value for key, value in bundle['global_graph'].items() if key not in ('incoming', 'popularity')})
    if args.command == 'audit':
        print(json.dumps({'status': 'data and train-only graph audit passed', 'counts': manifest['counts'], 'features_required_before_gpu': args.dataset_policy != 'steam_native_v1' and (not (args.features and args.feature_manifest))}), flush=True)
        return
    import torch
    import numpy as np
    shared.torch, shared.np = (torch, np)
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise DataError('UCGRec GPU pilot requires CUDA')
    torch.cuda.set_device(args.device)
    if not torch.cuda.is_bf16_supported():
        raise DataError('UCGRec requires BF16; refusing an implicit precision change')
    shared.write_json(args.output / 'environment.json', {'versions': package_versions(), 'python': sys.version, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(args.device)})
    shared.seed_all(args.seed)
    train_loop(args, bundle)
if __name__ == '__main__':
    try:
        main()
    except (DataError, GraphError, OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(2)
