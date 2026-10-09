import contextlib
import importlib.metadata
from pathlib import Path
from integrations.llmcf.common import binary_negative
from integrations.resume import file_digest
from .source import COMMIT, FILES, native_methods
from .data import samples, title


def model_files(path):
    path = Path(path).resolve()
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError('local Hugging Face model checkpoint directory required')
    files = {}
    for child in sorted(path.rglob('*')):
        if child.is_symlink():
            raise ValueError('model checkpoint symlinks must be materialized before pinning')
        if child.is_file():
            files[str(child.relative_to(path))] = file_digest(child)
    if not any(name.endswith(('.safetensors', '.bin')) for name in files):
        raise ValueError('local model weights missing')
    return files


def load_backbone(path, dtype, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False, torch_dtype=getattr(torch, dtype)).to(device)
    return backbone, tokenizer


def wrap(source, backbone, tokenizer, max_txt_len):
    import torch
    methods = native_methods(source)
    class NativeNoCF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qwen_model = backbone
            self.llama_tokenizer = tokenizer
            self.rec_encoder = None
            self.use_lora = False
            self.incontext_learning = True
            self.low_resource = False
            self.proj_token_num = 1
            self.max_txt_len = max_txt_len
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            self.has_print_prompt = True
            self.has_pri_decode = True
            self.prompt_list_p = None
        def maybe_autocast(self):
            return contextlib.nullcontext()
    for name, method in methods.items():
        setattr(NativeNoCF, name, method)
    if not hasattr(backbone, 'model') or not hasattr(backbone.model, 'embed_tokens'):
        raise ValueError('native no-CF wrapper requires a Llama/Qwen style model.embed_tokens')
    for parameter in backbone.model.embed_tokens.parameters():
        parameter.requires_grad_(False)
    answers = [tokenizer(value, add_special_tokens=False).input_ids for value in ('Yes', 'No')]
    if any(len(value) != 1 for value in answers) or answers[0] == answers[1] or tokenizer.pad_token_id in [value[0] for value in answers]:
        raise ValueError('native raw Yes-logit objective requires distinct nonpadding single-token Yes and No')
    return NativeNoCF()


class Runtime:
    name = 'RecICL-noCF-canonical'
    description = 'Pinned RecICL no-CF native Yes-logit BCE; same-user recent positive demonstrations and frozen-negative binary canonical adaptation'

    def __init__(self, args, bundle, backbone=None, tokenizer=None, checkpoint_files=None):
        self.bundle = bundle
        self.args = args
        if args.demonstrations < 0 or args.history_length < 1 or args.max_txt_len < 1:
            raise ValueError('invalid history, demonstration or token limit')
        for item in bundle['items']:
            title(bundle['items'], item)
        if backbone is None:
            checkpoint_files = model_files(args.model_path)
            backbone, tokenizer = load_backbone(args.model_path, args.dtype, args.device)
        self.model = wrap(args.source, backbone, tokenizer, args.max_txt_len)
        self.tokenizer = tokenizer
        self.training_size = 2 * len(bundle['rows']['train'])
        self.input_paths = [args.model_path]
        versions = {}
        for package in ('transformers', 'tokenizers', 'safetensors', 'sentencepiece'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.contract = {'source_commit': COMMIT, 'source_files': FILES, 'variant': 'official-no-CF-native-methods', 'model_files_sha256': checkpoint_files, 'dtype': args.dtype, 'packages': versions, 'max_txt_len': args.max_txt_len, 'item_kind': args.item_kind, 'history_length': args.history_length, 'demonstrations': args.demonstrations, 'objective': 'native BCEWithLogits of raw first Yes vocabulary logit; input token embeddings frozen', 'canonical_adaptation': 'one positive and one deterministic saved negative per train query; prior observed-positive same-user examples, no repeat padding for short history; current test label is dummy zero; raw Yes candidate scores; static validation NDCG@10 rather than streaming AUC', 'prompt': 'adapter domain-named Example/History/Question/Answer template, four identical copies for native prompt sampling', 'autocast': 'disabled; explicit model dtype applies to native forward'}

    def prepare(self, row, candidate, label):
        sample, template, rendered = samples(row, candidate, label, self.bundle['items'], self.args.demonstrations, self.args.history_length, self.args.item_kind, self.args.device)
        ids = self.tokenizer(rendered, add_special_tokens=False).input_ids
        max_context = getattr(self.model.qwen_model.config, 'max_position_embeddings', None)
        if len(ids) > self.args.max_txt_len or max_context and len(ids) + 1 > max_context:
            raise ValueError('native prompt exceeds configured context; truncation is refused')
        self.model.incontext_sample_len = sample['label'].shape[1]
        self.model.prompt_list = [template] * 4
        self.model.prompt_list_p = None
        return sample

    def loss(self, indices):
        import torch
        values = []
        for index in indices:
            row = self.bundle['rows']['train'][index // 2]
            label = 1 - index % 2
            candidate = row['target_id'] if label else binary_negative(row, self.args.seed)
            sample = self.prepare(row, candidate, label)
            values.append(self.model.forward_v2(sample)['loss'])
        return torch.stack(values).mean()

    def scores(self, row):
        import torch
        values = []
        for candidate in row['candidate_ids']:
            sample = self.prepare(row, candidate, 0)
            values.append(self.model.generate_for_samples_v2(sample)['logits'][0])
        return torch.stack(values)
