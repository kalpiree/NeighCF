import json
import math
from pathlib import Path
import numpy as np
from integrations.resume import file_digest
from .assets import Assets
from .source import COMMIT, FILES, instantiate, source_bytes


def configuration(source, override=None):
    import yaml
    raw = source_bytes(source, COMMIT, FILES)
    config = yaml.safe_load(raw['retrieval_task/src_code/config/beauty/SASREC_aug.yaml'])
    if override is not None:
        values = json.loads(Path(override).read_text())
        if not isinstance(values, dict) or set(values) - set(config):
            raise ValueError('config overrides contain unknown native model keys')
        config.update(values)
    dim = config['hidden_dim']
    if dim != config['item_id_dim'] or dim != config['user_id_dim'] or dim != config['decoder_emb_dim'] or dim != config['aug_label_id_dim'] or config['cot_mapping_mlp_hidden_size'] != [768, dim] or config['ICL_history_mlp'] != [dim * 5, dim]:
        raise ValueError('incompatible native augmentation dimensions')
    if any(type(config[k]) is not int or config[k] <= 0 for k in ('max_len', 'hidden_dim', 'num_blocks', 'num_heads', 'decoder_nhead')) or dim % config['num_heads'] or dim % (2 * config['decoder_nhead']) or not 0 <= config['dropout_rate'] < 1 or not math.isfinite(config['reconstruct_weight']) or config['reconstruct_weight'] < 0:
        raise ValueError('invalid native model dimensions or probability')
    return config


class Runtime:
    name = 'LLM-CF-SASREC-aug-canonical'
    description = 'Pinned native SASREC_aug loss and CoT augmentation; canonical static candidate adaptation with genuine supplied CoT assets'

    def __init__(self, args, bundle, config, assets):
        import torch
        self.bundle = bundle
        self.config = config
        self.assets = assets
        self.device = args.device
        self.item_ids = {item: i + 1 for i, item in enumerate(sorted(bundle['items']))}
        users = sorted({row['user_id'] for rows in bundle['rows'].values() for row in rows})
        self.user_ids = {user: i + 1 for i, user in enumerate(users)}
        if not 1 <= args.num_neg <= len(self.item_ids):
            raise ValueError('native sampled-negative count must be between one and catalog size')
        self.model = instantiate(args.source, config, len(self.item_ids), len(self.user_ids), args.num_neg, args.device)
        self.model.set_emb_tab_for_aug_data(torch.from_numpy(assets.cot.astype(np.float32)), {self.item_ids[item]: assets.item[index] for item, index in assets.item_index.items()})
        self.model.to(args.device)
        self.training_size = len(bundle['rows']['train'])
        self.input_paths = [args.assets] + ([args.config] if args.config else [])
        self.contract = {'source_commit': COMMIT, 'source_files': FILES, 'native_model': 'SASREC_aug', 'config': config, 'assets_sha256': assets.hashes, 'num_neg': args.num_neg, 'input_mapping': 'sorted IDs starting at 1; right padded newest max_len history', 'retrieval': 'full-history mean item-text dot product; two positive and two negative; exclude current query and same-user equal/future queries; seeded query shuffle', 'train_objective': 'native catalog-sampled softmax plus native reconstruction weight; saved 100 candidate train pool retained but not substituted for native negatives', 'selection': 'canonical validation NDCG@10', 'config_origin': 'pinned Beauty SASREC_aug YAML with explicit optional JSON overrides'}

    def tensors(self, queries):
        import torch
        users = torch.tensor([self.user_ids[row['user_id']] for row in queries], device=self.device)
        histories = []
        for row in queries:
            values = [self.item_ids[x] for x in row['history'][-self.config['max_len']:]]
            histories.append(values + [0] * (self.config['max_len'] - len(values)))
        history = torch.tensor(histories, device=self.device)
        target = torch.tensor([self.item_ids[row['target_id']] for row in queries], device=self.device)
        demos = [self.assets.retrieve(row) for row in queries]
        augmented = []
        for position in range(4):
            selected = [rows[position] for rows in demos]
            demo_queries = [row['query'] | {'target_id': row['candidate_id']} for row in selected]
            u = torch.tensor([self.user_ids[row['user_id']] for row in demo_queries], device=self.device)
            h = []
            for row in demo_queries:
                values = [self.item_ids[x] for x in row['history'][-self.config['max_len']:]]
                h.append(values + [0] * (self.config['max_len'] - len(values)))
            augmented.append((u, torch.tensor(h, device=self.device), torch.tensor([self.item_ids[row['target_id']] for row in demo_queries], device=self.device), torch.tensor([row['cot_index'] for row in selected], device=self.device), torch.tensor([row['label'] for row in selected], device=self.device)))
        return ((users, history, target), augmented), target

    def loss(self, indices):
        x, y = self.tensors([self.bundle['rows']['train'][i] for i in indices])
        return self.model.train_step(x, y)

    def scores(self, row):
        import torch
        x, _ = self.tensors([row])
        hidden = self.model(x)
        embeddings = self.model.emb_tab_dict['item_id_emb_tab'](torch.tensor([self.item_ids[x] for x in row['candidate_ids']], device=self.device))
        return (hidden @ embeddings.T).squeeze(0)
