import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import List

from integrations.resume import _atomic_json, digest, file_digest, runtime_context
from integrations.tallrec.data import check_tokenizer, validate_prompt_tokens
from integrations.tallrec.source import functions, prompts

VERSIONS = {'peft': '0.3.0', 'transformers': '4.28.0', 'bitsandbytes': '0.37.2'}


def tree_hashes(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('asset/checkpoint symlinks are unsupported')
        if path.is_file() and path.name != 'neighcf-checkpoint.json':
            result[str(path.relative_to(root))] = file_digest(path)
    if not result:
        raise ValueError('asset/checkpoint directory is empty')
    return result


def checkpoint_receipt(path, contract_sha256):
    path = Path(path)
    required = ('optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth', 'scaler.pt', 'training_args.bin')
    if any(not (path / name).is_file() for name in required):
        raise ValueError('full Trainer checkpoint with optimizer, scheduler, RNG, AMP scaler and training arguments is required')
    if not any((path / name).is_file() for name in ('pytorch_model.bin', 'adapter_model.bin')):
        raise ValueError('native adapter weights missing')
    return {'schema': 'tallrec-native-trainer-checkpoint-v1', 'contract_sha256': contract_sha256,
            'files': tree_hashes(path)}


def verify_checkpoint(path, contract_sha256):
    path = Path(path)
    receipt_path = path / 'neighcf-checkpoint.json'
    if receipt_path.is_symlink():
        raise ValueError('checkpoint receipt must not be a symlink')
    actual = checkpoint_receipt(path, contract_sha256)
    saved = json.loads(receipt_path.read_text())
    if digest(actual) != digest(saved):
        raise ValueError('native checkpoint integrity or contract differs')
    return actual


def load_native_adapter(model, checkpoint, contract_sha256, torch_module, setter):
    verify_checkpoint(checkpoint, contract_sha256)
    checkpoint = Path(checkpoint)
    path = checkpoint / 'pytorch_model.bin'
    if not path.is_file():
        path = checkpoint / 'adapter_model.bin'
    weights = torch_module.load(path, map_location='cpu', weights_only=True)
    setter(model, weights)


def compatible_setter(setter):
    def loaded(model, weights):
        setter(model, weights)
        return model
    return loaded


def compatible_arguments(constructor, train_examples=None):
    def configured(**kwargs):
        updates = None
        if train_examples is not None:
            batches = math.ceil(train_examples / kwargs['per_device_train_batch_size'])
            per_epoch = max(batches // kwargs['gradient_accumulation_steps'], 1)
            updates = math.ceil(per_epoch * kwargs['num_train_epochs'])
        for name in ('eval_steps', 'save_steps'):
            value = kwargs[name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid native validation cadence')
            kwargs[name] = min(math.ceil(value), updates) if updates is not None else math.ceil(value)
        return constructor(**kwargs)
    return configured


def runtime_versions():
    actual = {name: importlib.metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError('TALLRec requires a separate native dependency environment: ' + json.dumps(VERSIONS))
    return actual


def _launch(request):
    runtime_versions()
    import numpy as np
    import torch
    import transformers
    import peft
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score
    from transformers import LlamaForCausalLM, LlamaTokenizer, EarlyStoppingCallback
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('native launcher requires exactly one visible CUDA GPU and single process')
    contract = request['contract']
    runtime = {'native_packages': runtime_versions(), 'torch_runtime': runtime_context('cuda'),
               'datasets': importlib.metadata.version('datasets'), 'accelerate': importlib.metadata.version('accelerate')}
    runtime_path = Path(request['native_output']).parent / 'runtime.json'
    if runtime_path.exists():
        if digest(json.loads(runtime_path.read_text())) != digest(runtime):
            raise ValueError('native training runtime changed')
    else:
        _atomic_json(runtime_path, runtime)
    if digest(contract) != request['contract_sha256']:
        raise ValueError('request contract checksum differs')
    for name, expected in contract['adapter_source_sha256'].items():
        if file_digest(Path(__file__).resolve().parents[2] / name) != expected:
            raise ValueError('adapter source changed')
    for name in ('model_path', 'instruction_adapter'):
        if tree_hashes(contract[name]) != contract[name + '_sha256']:
            raise ValueError('external model or instruction adapter changed')
    for name, expected in request['prepared_sha256'].items():
        if file_digest(Path(request['prepared']) / name) != expected:
            raise ValueError('prepared canonical data changed')
    tokenizer = LlamaTokenizer.from_pretrained(contract['model_path'], local_files_only=True)
    check_tokenizer(tokenizer)
    training_prompt, _ = prompts(contract['source'])
    for name in ('train.json', 'valid.json'):
        rows = json.loads((Path(request['prepared']) / name).read_text())
        validate_prompt_tokens(tokenizer, training_prompt, rows, contract['cutoff_len'])
    resume = request.get('resume_checkpoint')
    if resume:
        verify_checkpoint(resume, request['contract_sha256'])
    transformers.set_seed(contract['seed'])
    parent = transformers.Trainer
    class NativeTrainer(parent):
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            load_native_adapter(model if model is not None else self.model, resume_from_checkpoint, request['contract_sha256'], torch, peft.set_peft_model_state_dict)
        def _load_best_model(self):
            load_native_adapter(self.model, self.state.best_model_checkpoint, request['contract_sha256'], torch, peft.set_peft_model_state_dict)
        def _save_checkpoint(self, model, trial, metrics=None):
            super()._save_checkpoint(model, trial, metrics)
            directory = Path(self.args.output_dir) / ('checkpoint-' + str(self.state.global_step))
            _atomic_json(directory / 'neighcf-checkpoint.json', checkpoint_receipt(directory, request['contract_sha256']))
        def train(self, *args, **kwargs):
            if resume:
                kwargs['resume_from_checkpoint'] = resume
            return super().train(*args, **kwargs)
    proxy = SimpleNamespace(Trainer=NativeTrainer, TrainingArguments=compatible_arguments(transformers.TrainingArguments, request['train_examples']),
                            DataCollatorForSeq2Seq=transformers.DataCollatorForSeq2Seq)
    namespace = {'os': os, 'sys': sys, 'List': List, 'np': np, 'torch': torch,
                 'transformers': proxy, 'load_dataset': load_dataset, 'EarlyStoppingCallback': EarlyStoppingCallback,
                 'LoraConfig': peft.LoraConfig, 'get_peft_model': peft.get_peft_model,
                 'get_peft_model_state_dict': peft.get_peft_model_state_dict,
                 'prepare_model_for_int8_training': peft.prepare_model_for_int8_training,
                 'set_peft_model_state_dict': compatible_setter(peft.set_peft_model_state_dict),
                 'LlamaForCausalLM': LlamaForCausalLM, 'LlamaTokenizer': LlamaTokenizer,
                 'roc_auc_score': roc_auc_score}
    loaded = functions(contract['source'], 'finetune_rec.py', ['train', 'generate_prompt'], namespace)
    _atomic_json(Path(request['native_output']).parent / 'launch.json', {'contract_sha256': request['contract_sha256'], 'status': 'native_training_started'})
    loaded['train'](base_model=contract['model_path'], train_data_path=str(Path(request['prepared']) / 'train.json'),
                    val_data_path=str(Path(request['prepared']) / 'valid.json'), output_dir=request['native_output'],
                    sample=request['train_examples'], seed=contract['seed'], batch_size=contract['batch_size'],
                    micro_batch_size=contract['micro_batch_size'], num_epochs=contract['epochs'],
                    learning_rate=contract['learning_rate'], cutoff_len=contract['cutoff_len'],
                    lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_target_modules=['q_proj', 'v_proj'],
                    train_on_inputs=True, group_by_length=True,
                    resume_from_checkpoint=resume or contract['instruction_adapter'])
    _atomic_json(Path(request['native_output']).parent / 'completion.json',
                 {'contract_sha256': request['contract_sha256'], 'phase': 'complete',
                  'selection': 'native validation AUC', 'native_output_sha256': tree_hashes(request['native_output'])})


def launch(request):
    from integrations.tallrec.run import run_lock
    from integrations.traditional.run import output_path
    contract = request['contract']
    output = Path(request['native_output']).parent
    output_path(output, contract['source'], contract['model_path'], contract['instruction_adapter'])
    with run_lock(output):
        return _launch(request)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    launch(json.loads(args.request.read_text()))


if __name__ == '__main__':
    main()
