import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from evaluate import evaluate_records
from integrations.resume import RestartManager, ScoreJournal, _atomic_json, digest, file_digest
from integrations.traditional.run import output_path, positive, seed_all
from integrations.tallrec.data import POLICY, binary_records, bundle, check_tokenizer, example, export_json, native_yes_score
from integrations.tallrec.native import tree_hashes, verify_checkpoint
from integrations.tallrec.source import COMMIT, FILES, URL, prompts, verified_sources


@contextlib.contextmanager
def run_lock(output):
    path = output / 'run.lock'
    if path.is_symlink():
        raise ValueError('run lock must not be a symlink')
    with path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_pins():
    root = Path(__file__).resolve().parents[2]
    names = ['integrations/tallrec/' + stem + '.py' for stem in ('run', 'data', 'source', 'native')]
    names += ['integrations/resume.py', 'integrations/traditional/run.py', 'adapters/llmsrec_data.py', 'evaluate.py']
    return {name: file_digest(root / name) for name in names}


def local_asset(path):
    path = Path(path).absolute()
    if not path.is_dir() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('asset must be an existing local nonsymlink directory')
    return path.resolve()


def contract(args, data):
    if args.adaptation != POLICY:
        raise ValueError('explicit --adaptation ' + POLICY + ' is required')
    model, instruction = local_asset(args.model_path), local_asset(args.instruction_adapter)
    config = json.loads((instruction / 'adapter_config.json').read_text())
    if (config.get('r'), config.get('lora_alpha'), config.get('lora_dropout')) != (8, 16, 0.05):
        raise ValueError('instruction adapter must match native rank8/alpha16/dropout0.05')
    if set(config.get('target_modules', [])) != {'q_proj', 'v_proj'} or config.get('bias') != 'none' or config.get('task_type') != 'CAUSAL_LM':
        raise ValueError('instruction adapter has incompatible native LoRA settings')
    if not (instruction / 'adapter_model.bin').is_file():
        raise ValueError('native PEFT0.3 instruction adapter_model.bin is required')
    if args.batch_size % args.micro_batch_size or args.micro_batch_size > args.batch_size:
        raise ValueError('batch size must be divisible by micro batch size')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning rate must be finite and positive')
    return {'kind': 'tallrec-native-canonical-v1', 'source': str(Path(args.source).resolve()),
            'source_identity': {'url': URL, 'commit': COMMIT, 'files': FILES},
            'adapter_source_sha256': source_pins(), 'data_sha256': data['hashes'],
            'protocol_sha256': data['protocol_sha256'], 'dataset': args.dataset,
            'model_path': str(model), 'model_path_sha256': tree_hashes(model),
            'instruction_adapter': str(instruction), 'instruction_adapter_sha256': tree_hashes(instruction),
            'seed': args.seed, 'domain': args.domain, 'adaptation': POLICY, 'history_length': 10,
            'expected_candidates': args.expected_candidates, 'epochs': args.epochs,
            'batch_size': args.batch_size, 'micro_batch_size': args.micro_batch_size,
            'learning_rate': args.learning_rate, 'cutoff_len': args.cutoff_len,
            'validation': 'native binary AUC over every saved canonical validation candidate',
            'compatibility': ['PEFT setter returns loaded model', 'ceil native eval/save interval and cap at the final optimizer update',
                              'verified full Trainer checkpoint resume override', 'PEFT-aware checkpoint and validation-best adapter restoration'],
            'scoring': 'native softmax(softmax(full vocabulary)[Yes8241,No3782]).Yes'}


def latest_checkpoint(native, expected):
    candidates = []
    for path in native.glob('checkpoint-*'):
        suffix = path.name.removeprefix('checkpoint-')
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise ValueError('no full native checkpoint is available for resume')
    path = max(candidates)[1]
    verify_checkpoint(path, expected)
    return path.resolve()


def prepare_request(args, data):
    specification = contract(args, data)
    out = output_path(args.output, args.data, args.source, args.model_path, args.instruction_adapter)
    prepared_only = out.exists() and (out / 'request.json').is_file() and not (out / 'native').exists() and not (out / 'launch.json').exists()
    if out.exists() and any(out.iterdir()) and not args.resume and not prepared_only:
        raise ValueError('new training requires an empty output directory or an unchanged unlaunched preparation')
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        identity_path = out / 'contract.json'
        if args.resume or prepared_only:
            old = json.loads(identity_path.read_text())
            if digest(old) != digest(specification):
                raise ValueError('training contract changed')
            request = json.loads((out / 'request.json').read_text())
            if (digest(request['contract']) != digest(specification) or request['contract_sha256'] != digest(specification)
                    or request['prepared'] != str(out / 'prepared') or request['native_output'] != str(out / 'native')):
                raise ValueError('prepared request differs from pinned training contract')
            for name, expected in request['prepared_sha256'].items():
                if file_digest(out / 'prepared' / name) != expected:
                    raise ValueError('prepared data changed')
            if (out / 'completion.json').exists():
                complete = json.loads((out / 'completion.json').read_text())
                if complete['contract_sha256'] != digest(specification) or complete['native_output_sha256'] != tree_hashes(out / 'native'):
                    raise ValueError('completed native output changed')
                return request, True
            request['resume_checkpoint'] = str(latest_checkpoint(out / 'native', digest(specification))) if args.resume else None
        else:
            prepared = out / 'prepared'
            prepared.mkdir()
            train_examples = export_json(prepared / 'train.json', binary_records(data, 'train', args.seed, args.domain))
            valid_examples = export_json(prepared / 'valid.json', binary_records(data, 'valid', args.seed, args.domain))
            _atomic_json(identity_path, specification)
            request = {'contract': specification, 'contract_sha256': digest(specification),
                       'prepared': str(prepared), 'native_output': str(out / 'native'),
                       'train_examples': train_examples, 'valid_examples': valid_examples,
                       'prepared_sha256': {name: file_digest(prepared / name) for name in ('train.json', 'valid.json')},
                       'resume_checkpoint': None}
        _atomic_json(out / 'request.json', request)
    return request, False


def train(args, data):
    request, complete = prepare_request(args, data)
    out = Path(request['native_output']).parent
    if not complete:
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]), PYTHONDONTWRITEBYTECODE='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                           HF_DATASETS_CACHE=str(out / 'cache'), WANDB_DISABLED='true', WANDB_MODE='disabled')
        subprocess.run([sys.executable, '-m', 'integrations.tallrec.native', '--request', str(out / 'request.json')],
                       cwd=out, env=environment, check=True)
    return json.loads((out / 'completion.json').read_text())


class Backend:
    def __init__(self, specification, adapter, device, micro_batch_size):
        import torch
        from transformers import LlamaForCausalLM, LlamaTokenizer
        from peft import PeftModel
        self.device, self.micro_batch_size = device, micro_batch_size
        self.tokenizer = LlamaTokenizer.from_pretrained(specification['model_path'], local_files_only=True)
        check_tokenizer(self.tokenizer)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = 0
        model = LlamaForCausalLM.from_pretrained(specification['model_path'], local_files_only=True,
                                               torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
        self.model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True).to(device).eval()
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        _, self.prompt = prompts(specification['source'])

    def scores(self, examples):
        import torch
        values = []
        for start in range(0, len(examples), self.micro_batch_size):
            rows = examples[start:start + self.micro_batch_size]
            texts = [self.prompt(row['instruction'], row['input']) for row in rows]
            tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=False)
            if tokens['input_ids'].shape[1] >= self.model.config.max_position_embeddings:
                raise ValueError('evaluation prompt exceeds native model context')
            with torch.inference_mode():
                outputs = self.model(**{name: value.to(self.device) for name, value in tokens.items()})
                values.extend(native_yes_score(outputs.logits[:, -1, :]).cpu().tolist())
        return values


def _evaluate(args, data, backend_factory=Backend):
    run = local_asset(args.run)
    specification = json.loads((run / 'contract.json').read_text())
    completed = json.loads((run / 'completion.json').read_text())
    if (specification['kind'] != 'tallrec-native-canonical-v1' or completed['phase'] != 'complete'
            or digest(specification) != completed['contract_sha256']):
        raise ValueError('completed TALLRec training contract required')
    if (specification['data_sha256'] != data['hashes'] or specification['dataset'] != args.dataset
            or specification['protocol_sha256'] != data['protocol_sha256']
            or specification['seed'] != args.seed or specification['domain'] != args.domain
            or specification['expected_candidates'] != args.expected_candidates
            or specification['adapter_source_sha256'] != source_pins()):
        raise ValueError('evaluation arguments, inputs or adapter source differ from training')
    verified_sources(specification['source'])
    for name in ('model_path', 'instruction_adapter'):
        if tree_hashes(specification[name]) != specification[name + '_sha256']:
            raise ValueError('model or instruction adapter changed')
    if tree_hashes(run / 'native') != completed['native_output_sha256']:
        raise ValueError('validation-selected native output changed')
    out = output_path(args.output, args.data, args.source, run, specification['model_path'], specification['instruction_adapter'])
    seed_all(args.seed, args.device)
    evaluation_contract = {'kind': 'tallrec-canonical-evaluation-v1', 'training_contract_sha256': digest(specification),
                           'completion_sha256': file_digest(run / 'completion.json'), 'source_sha256': source_pins(),
                           'device': args.device, 'micro_batch_size': args.micro_batch_size}
    with RestartManager(out, evaluation_contract, resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:
        save = lambda path: Path(path).write_bytes(b'TALLRec immutable evaluation model\n')
        restore = lambda path: None
        progress = manager.restore(restore) or {'phase': 'evaluate'}
        rows = data['rows']['test']
        journal = ScoreJournal(manager, 'test', [[row['query_id']] for row in rows])
        if not args.resume:
            manager.checkpoint(progress, save, force=True)
        backend = backend_factory(specification, run / 'native', args.device, args.micro_batch_size) if journal.completed_batches < len(rows) else None
        for index in range(journal.completed_batches, len(rows)):
            row = rows[index]
            samples = [example(data['items'], row, item, item == row['target_id'], args.domain) for item in row['candidate_ids']]
            scores = backend.scores(samples)
            if len(scores) != len(row['candidate_ids']) or any(not math.isfinite(value) for value in scores):
                raise ValueError('finite score required for every saved candidate')
            record = {name: row[name] for name in ('query_id', 'user_id', 'target_id', 'candidate_ids')}
            record['scores'] = scores
            journal.append(index, [record])
            manager.checkpoint({'phase': 'evaluate', 'queries': index + 1}, save)
        metadata = {'model': 'TALLRec', 'dataset': args.dataset, 'seed': args.seed,
                    'protocol_sha256': data['protocol_sha256'], 'implementation': POLICY,
                    'selection': specification['validation'], 'scoring': specification['scoring']}
        result, ranks = evaluate_records(journal.records(), metadata, args.expected_candidates)
        _atomic_json(out / 'evaluation.json', result)
        _atomic_json(out / 'ranks.json', ranks)
        _atomic_json(out / 'predictions.json', journal.records())
        manager.checkpoint({'phase': 'complete', 'queries': len(rows)}, save, force=True,
                           artifacts={name: out / name for name in ('evaluation.json', 'ranks.json', 'predictions.json')})
        return result


def evaluate(args, data, backend_factory=Backend):
    path = Path(args.run) / 'run.lock'
    if path.is_symlink():
        raise ValueError('run lock must not be a symlink')
    with path.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return _evaluate(args, data, backend_factory)


def parser():
    command = argparse.ArgumentParser()
    command.add_argument('command', choices=('audit', 'prepare', 'train', 'evaluate'))
    command.add_argument('--source', required=True, type=Path)
    command.add_argument('--data', required=True, type=Path)
    command.add_argument('--dataset', required=True)
    command.add_argument('--domain', required=True, choices=('movie', 'book', 'item', 'artist'))
    command.add_argument('--adaptation', choices=(POLICY,))
    command.add_argument('--model-path', type=Path)
    command.add_argument('--instruction-adapter', type=Path)
    command.add_argument('--output', type=Path)
    command.add_argument('--run', type=Path)
    command.add_argument('--protocol-sha256')
    command.add_argument('--expected-candidates', type=positive, default=51)
    command.add_argument('--epochs', type=positive, default=5)
    command.add_argument('--batch-size', type=positive, default=128)
    command.add_argument('--micro-batch-size', type=positive, default=4)
    command.add_argument('--learning-rate', type=float, default=1e-4)
    command.add_argument('--cutoff-len', type=positive, default=512)
    command.add_argument('--seed', type=int, default=42)
    command.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    command.add_argument('--checkpoint-seconds', type=float, default=600)
    command.add_argument('--resume', action='store_true')
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if not 0 <= args.seed < 2 ** 32:
        command.error('seed must be in [0, 2**32)')
    if args.command == 'train' and args.device != 'cuda':
        command.error('native TALLRec training requires exactly one visible CUDA GPU')
    if args.command != 'audit' and args.output is None:
        command.error('--output is required')
    if args.command in ('prepare', 'train') and (args.model_path is None or args.instruction_adapter is None):
        command.error('--model-path and --instruction-adapter are required; assets are never downloaded')
    if args.command == 'evaluate' and args.run is None:
        command.error('--run is required')
    verified_sources(args.source)
    data = bundle(args)
    if args.command == 'audit':
        result = {'source': {'url': URL, 'commit': COMMIT, 'files': FILES},
                  'data_sha256': data['hashes'], 'protocol_sha256': data['protocol_sha256'],
                  'queries': {key: len(value) for key, value in data['rows'].items()},
                  'required_adaptation': POLICY, 'native_validation': 'binary AUC',
                  'instruction_adapter_required': True, 'experiments_run': False}
    elif args.command == 'prepare':
        request, complete = prepare_request(args, data)
        result = {'prepared': request['prepared'], 'train_examples': request['train_examples'],
                  'valid_examples': request['valid_examples'], 'complete': complete, 'experiments_run': False}
    elif args.command == 'train':
        result = train(args, data)
    else:
        result = evaluate(args, data)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
