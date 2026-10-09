import argparse
import copy
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from adapters.llmsrec_data import index_queries, prediction_record
from evaluate import evaluate_records
from integrations.lightgcn.run import finite_optimizer_state, output_path, save_tensor, seed_all
from integrations.morec.source import COMMIT, FILES, URL, model_class, verify
from integrations.neighcf.retrieval import timestamp, validate_query
from integrations.resume import RestartManager, _atomic_json, digest, file_digest, runtime_context
from integrations.zeroshot.core import load_data


def configuration(path):
    value = json.loads(Path(path).read_text())
    fields = {'epochs', 'batch_size', 'max_seq_len', 'embedding_dim', 'num_attention_heads',
              'transformer_block', 'drop_rate', 'learning_rate', 'encoder_learning_rate',
              'weight_decay', 'freeze_parameters_before', 'num_words_title'}
    if set(value) != fields:
        raise ValueError('MoRec configuration fields differ from the required schema')
    for key in ('epochs', 'batch_size', 'max_seq_len', 'embedding_dim', 'num_attention_heads', 'transformer_block', 'num_words_title'):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if type(value['freeze_parameters_before']) is not int or value['freeze_parameters_before'] < 0:
        raise ValueError('freeze_parameters_before must be nonnegative')
    for key in ('drop_rate', 'learning_rate', 'encoder_learning_rate', 'weight_decay'):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]):
            raise ValueError(key + ' must be finite')
    if not 0 <= value['drop_rate'] < 1 or value['weight_decay'] < 0 or min(value['learning_rate'], value['encoder_learning_rate']) <= 0:
        raise ValueError('Invalid dropout, decay or learning rate')
    if value['embedding_dim'] % value['num_attention_heads']:
        raise ValueError('Embedding dimension must divide evenly into heads')
    return value


def bundle(path, cfg, expected):
    items, rows, metadata, hashes = load_data(path, expected)
    mapping, indexed = index_queries(items, rows, cfg['max_seq_len'])
    selected = {}
    for row in indexed['train']:
        source = row['source']
        validate_query(source, items)
        key = (timestamp(source), source['query_id'])
        user = source['user_id']
        if user not in selected or key > selected[user][0]:
            selected[user] = (key, row)
    sequences = []
    for user in sorted(selected):
        row = selected[user][1]
        complete = row['history_item_indices'] + [row['target_item_index']]
        if len(set(complete)) >= len(items):
            raise ValueError('A training user has no unobserved negative item')
        sequences.append({'user_id': user, 'full': complete, 'sequence': complete[-cfg['max_seq_len'] - 1:]})
    if not sequences:
        raise ValueError('No training users')
    return {'items': items, 'rows': indexed, 'mapping': mapping, 'metadata': metadata,
            'hashes': hashes, 'training': sequences}


def tokenize_catalog(data, tokenizer, words):
    import torch
    ids = data['mapping']['index_to_item'][1:]
    encoding = tokenizer([data['items'][item]['title'] for item in ids], padding='max_length',
                         truncation=True, max_length=words, return_tensors='pt')
    if encoding['input_ids'].shape != (len(ids), words) or encoding['attention_mask'].shape != (len(ids), words):
        raise ValueError('Tokenizer must return aligned fixed-width title IDs and masks')
    tokens = torch.cat([encoding['input_ids'], encoding['attention_mask']], dim=1)
    return torch.cat([torch.zeros((1, 2 * words), dtype=torch.long), tokens.long()], dim=0)


def create_model(source, data, cfg, encoder):
    for index, (name, parameter) in enumerate(encoder.named_parameters()):
        parameter.requires_grad = index >= cfg['freeze_parameters_before'] and not name.startswith('pooler.')
    settings = SimpleNamespace(**cfg, word_embedding_dim=encoder.config.hidden_size,
        num_words_abstract=0, num_words_body=0, news_attributes=['title'], bert_model_load='bert-base-uncased')
    return model_class(source)(settings, len(data['items']), True, encoder)


def sample_batch(data, tokens, indices, cfg, seed, epoch, device):
    import torch
    packed, masks = [], []
    count = len(data['items'])
    for index in indices:
        row = data['training'][index]
        sequence = row['sequence']
        banned = set(row['full'])
        rng = random.Random(digest([seed, epoch, row['user_id']]))
        negatives = []
        for _ in range(len(sequence) - 1):
            value = rng.randint(1, count)
            while value in banned:
                value = rng.randint(1, count)
            negatives.append(value)
        padding = cfg['max_seq_len'] + 1 - len(sequence)
        positive = [0] * padding + sequence
        negative = [0] * padding + negatives + [0]
        packed.append(list(zip(positive, negative)))
        masks.append([0] * padding + [1] * (len(sequence) - 1))
    indices = torch.tensor(packed, dtype=torch.long)
    samples = tokens[indices].reshape(-1, tokens.shape[-1]).to(device)
    return samples, torch.tensor(masks, dtype=torch.float32, device=device)


def score(model, data, rows, tokens, batch_size, device):
    import torch
    model.eval()
    with torch.inference_mode():
        embeddings = torch.cat([model.bert_encoder(tokens[start:start + batch_size].to(device))
                                for start in range(0, len(tokens), batch_size)], dim=0)
        for start in range(0, len(rows), batch_size):
            selected = [rows[index] for index in range(start, min(start + batch_size, len(rows)))]
            histories = torch.tensor([row['sequence_item_indices'] for row in selected], dtype=torch.long, device=device)
            masks = (histories != 0).float()
            users = model.user_encoder(embeddings[histories], masks, device)[:, -1]
            candidates = torch.tensor([row['candidate_item_indices'] for row in selected], dtype=torch.long, device=device)
            values = (users.unsqueeze(1) * embeddings[candidates]).sum(-1).cpu().tolist()
            for query, scores in zip(selected, values):
                yield prediction_record(query, scores)


def model_files(path):
    path = Path(path)
    files = {str(p.relative_to(path)): file_digest(p) for p in sorted(path.rglob('*'))
             if p.is_file() and p.suffix in {'.json', '.txt', '.model', '.bin', '.safetensors'}}
    if 'config.json' not in files or not any(name.endswith(('.bin', '.safetensors')) for name in files):
        raise ValueError('A materialized local BERT configuration and weights are required')
    return files


def metadata(data, seed):
    return {'model': 'MoRec-text-BERT-SASRec-BCE', 'dataset': data['metadata']['dataset'],
            'seed': seed, 'protocol_sha256': data['metadata']['protocol_sha256']}


def train(model, data, tokens, cfg, output, seed, device, native_source, encoder_identity,
          resume=False, checkpoint_seconds=600, stop_after=None):
    import torch
    output = output_path(output, native_source)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('New training requires an empty output')
    groups = [{'params': [p for name, p in model.named_parameters() if p.requires_grad and 'bert_model' in name],
               'lr': cfg['encoder_learning_rate']},
              {'params': [p for name, p in model.named_parameters() if p.requires_grad and 'bert_model' not in name],
               'lr': cfg['learning_rate']}]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    root = Path(__file__).resolve().parents[2]
    names = ('integrations/morec/run.py', 'integrations/morec/source.py', 'integrations/resume.py',
             'integrations/zeroshot/core.py', 'integrations/lightgcn/run.py', 'adapters/llmsrec_data.py', 'evaluate.py')
    contract = {'kind': 'morec-text-native-canonical-v1', 'configuration': cfg, 'metadata': metadata(data, seed),
                'native_source': {'url': URL, 'commit': COMMIT, 'files': FILES},
                'source_sha256': {name: file_digest(root / name) for name in names},
                'data_sha256': data['hashes'], 'mapping_sha256': digest(data['mapping']),
                'training_sequences_sha256': digest(data['training']), 'encoder': encoder_identity,
                'title_tokens_sha256': digest(tokens.tolist()), 'runtime': runtime_context(device),
                'protocol': {'variant': 'original text end-to-end BERT/SASRec BCE',
                    'training': 'latest canonical train prefix per user; native shifted sequence BCE',
                    'negatives': 'uniform unseen items excluding complete selected training history and target',
                    'text': 'catalog titles, explicit fixed-width tokenizer truncation',
                    'selection': 'validation NDCG@10, earliest epoch on ties',
                    'evaluation': 'canonical frozen candidates, dot-product scores'}}
    with RestartManager(output, contract, resume=resume, interval_seconds=checkpoint_seconds) as manager:
        save = lambda path: torch.save(model.state_dict(), path)
        restore = lambda path: model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        progress = manager.restore(restore, optimizer) or {'phase': 'train', 'epoch': 0, 'cursor': 0,
                      'steps': 0, 'best_epoch': None, 'best_metric': None, 'validation': []}
        def checkpoint():
            artifacts = {'best.pt': output / 'best.pt'} if progress['best_epoch'] is not None else {}
            if progress['phase'] == 'complete':
                artifacts.update({name: output / name for name in ('metrics.json', 'ranks.json', 'predictions.json')})
            manager.checkpoint(progress, save, optimizer, artifacts=artifacts, force=True)
        if not resume:
            _atomic_json(output / 'run.json', contract)
            checkpoint()
        if progress['phase'] == 'complete':
            return json.loads((output / 'metrics.json').read_text())
        while progress['epoch'] < cfg['epochs']:
            order = list(range(len(data['training'])))
            random.Random(digest([seed, progress['epoch'], 'order'])).shuffle(order)
            model.train()
            for start in range(progress['cursor'], len(order), cfg['batch_size']):
                samples, masks = sample_batch(data, tokens, order[start:start + cfg['batch_size']], cfg, seed, progress['epoch'], device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(samples, masks, device)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite native MoRec BCE loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite native MoRec gradient')
                optimizer.step()
                finite_optimizer_state(optimizer)
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError('Nonfinite native MoRec parameter')
                progress.update(cursor=min(start + cfg['batch_size'], len(order)), steps=progress['steps'] + 1)
                artifacts = {'best.pt': output / 'best.pt'} if progress['best_epoch'] is not None else {}
                manager.checkpoint(progress, save, optimizer, artifacts=artifacts)
                if stop_after is not None and progress['steps'] >= stop_after:
                    checkpoint()
                    return {'status': 'paused', 'steps': progress['steps']}
            values, _ = evaluate_records(score(model, data, data['rows']['valid'], tokens, cfg['batch_size'], device),
                                         metadata(data, seed), len(data['rows']['valid'][0]['candidate_item_indices']))
            metric = values['metrics']['NDCG@10']
            progress['validation'].append(values['metrics'])
            if progress['best_metric'] is None or metric > progress['best_metric']:
                save_tensor(output / 'best.pt', model.state_dict())
                progress.update(best_metric=metric, best_epoch=progress['epoch'] + 1)
            progress.update(epoch=progress['epoch'] + 1, cursor=0)
            checkpoint()
        last = copy.deepcopy(model.state_dict())
        restore(output / 'best.pt')
        records = list(score(model, data, data['rows']['test'], tokens, cfg['batch_size'], device))
        metrics, ranks = evaluate_records(records, metadata(data, seed), len(records[0]['candidate_ids']))
        metrics['selection'] = {'epoch': progress['best_epoch'], 'validation_NDCG@10': progress['best_metric']}
        for name, value in (('metrics.json', metrics), ('ranks.json', ranks), ('predictions.json', records)):
            _atomic_json(output / name, value)
        model.load_state_dict(last)
        progress['phase'] = 'complete'
        checkpoint()
        return metrics


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('audit', 'train'))
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('example.config.json'))
    p.add_argument('--encoder-path', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--expected-candidates', type=int, default=51)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-seconds', type=float, default=600)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    cfg = configuration(args.config)
    verify(args.source)
    data = bundle(args.data, cfg, args.expected_candidates)
    if args.command == 'audit':
        print(json.dumps({'status': 'data_and_source_valid', 'variant': 'text-BERT-SASRec-original-BCE',
                          'training_users': len(data['training']), 'metadata': metadata(data, args.seed)}))
        return 0
    if not args.encoder_path or not args.output:
        p.error('Training requires --encoder-path and --output')
    output_path(args.output, args.data, args.source, args.encoder_path)
    encoder_identity = model_files(args.encoder_path)
    from transformers import AutoModel, AutoTokenizer
    seed_all(args.seed, args.device)
    encoder = AutoModel.from_pretrained(args.encoder_path, local_files_only=True, trust_remote_code=False)
    if encoder.config.model_type != 'bert':
        raise ValueError('This MoRec variant requires a BERT text encoder')
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, local_files_only=True, trust_remote_code=False)
    tokens = tokenize_catalog(data, tokenizer, cfg['num_words_title'])
    model = create_model(args.source, data, cfg, encoder).to(args.device)
    result = train(model, data, tokens, cfg, args.output, args.seed, args.device,
                   args.source, encoder_identity, args.resume, args.checkpoint_seconds)
    print(json.dumps({'status': 'complete', 'metrics': result['metrics']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
