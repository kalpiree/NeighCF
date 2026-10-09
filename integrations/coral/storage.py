import hashlib
import json
import math
import os
from pathlib import Path
import torch

from integrations.drdt.run import atomic_json, read_json, sha
from integrations.neighcf.retrieval import digest


class Checkpoints:
    def __init__(self, directory, contract):
        self.directory, self.contract = Path(directory), contract
        self.contract_sha256 = digest(contract)
        self.active_generation = None

    def read_generation(self, generation):
        if type(generation) is not int or generation < 1:
            raise ValueError('invalid checkpoint generation')
        stem = self.directory / f'checkpoint-{generation:08d}'
        tensor, receipt_path = stem.with_suffix('.pt'), stem.with_suffix('.json')
        if tensor.is_symlink() or receipt_path.is_symlink():
            raise ValueError('checkpoint symlinks are forbidden')
        receipt = read_json(receipt_path)
        if receipt.get('generation') != generation or receipt.get('contract_sha256') != self.contract_sha256 or sha(tensor) != receipt.get('sha256'):
            raise ValueError('checkpoint checksum, generation or contract mismatch')
        value = torch.load(tensor, map_location='cpu', weights_only=False)
        if value.get('contract_sha256') != self.contract_sha256 or value.get('generation') != generation or value.get('progress', {}).get('best_generation') != receipt.get('best_generation'):
            raise ValueError('checkpoint payload identity mismatch')
        return value

    def load(self, best=False):
        pointer = read_json(self.directory / 'latest.json')
        if pointer.get('contract_sha256') != self.contract_sha256:
            raise ValueError('checkpoint pointer contract mismatch')
        errors = []
        for generation in (pointer.get('current'), pointer.get('previous')):
            if generation is None:
                continue
            try:
                value = self.read_generation(generation)
                self.active_generation = generation
                return self.read_generation(value['progress']['best_generation']) if best else value
            except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as error:
                errors.append(str(error))
                if best:
                    raise ValueError('selected best checkpoint is invalid') from error
        raise ValueError('no intact current or previous checkpoint: ' + '; '.join(errors))

    def save(self, engine, progress, select_best=False):
        existing = [int(path.stem.split('-')[1]) for path in self.directory.glob('checkpoint-*') if path.suffix in {'.pt', '.json', '.tmp'} and path.stem.split('-')[1].isdigit()]
        generation = max(existing, default=0) + 1
        pointer_path = self.directory / 'latest.json'
        previous = self.active_generation or (read_json(pointer_path)['current'] if pointer_path.exists() else None)
        progress = dict(progress)
        if select_best:
            progress['best_generation'] = generation
        stem = self.directory / f'checkpoint-{generation:08d}'
        tensor, temporary = stem.with_suffix('.pt'), stem.with_suffix('.tmp')
        if tensor.exists() or tensor.is_symlink() or temporary.is_symlink():
            raise ValueError('checkpoint target collision')
        with temporary.open('wb') as stream:
            torch.save({'contract_sha256': self.contract_sha256, 'generation': generation,
                        'engine': engine.snapshot(), 'progress': progress}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, tensor)
        atomic_json(stem.with_suffix('.json'), {'generation': generation, 'contract_sha256': self.contract_sha256,
                                              'sha256': sha(tensor), 'best_generation': progress.get('best_generation')})
        atomic_json(pointer_path, {'contract_sha256': self.contract_sha256, 'current': generation, 'previous': previous})
        self.active_generation = generation
        keep = {generation, previous, progress.get('best_generation')}
        if previous:
            keep.add(read_json(self.directory / f'checkpoint-{previous:08d}.json').get('best_generation'))
        for old in existing:
            if old not in keep:
                for suffix in ('.pt', '.json'):
                    (self.directory / f'checkpoint-{old:08d}{suffix}').unlink(missing_ok=True)
        return progress


class ProbabilityCache:
    def __init__(self, directory, contract_sha256, score, max_calls):
        self.directory, self.contract_sha256, self.score = Path(directory), contract_sha256, score
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('positive max_backend_calls is required')
        self.max_calls = max_calls
        self.directory.mkdir(exist_ok=True)

    def __call__(self, messages):
        name = digest(messages)
        path = self.directory / (name + '.json')
        if path.is_symlink():
            raise ValueError('probability cache symlinks are forbidden')
        if path.exists():
            value = read_json(path)
            if value.get('contract_sha256') != self.contract_sha256 or value.get('messages_sha256') != name or value.get('result_sha256') != digest(value['result']):
                raise ValueError('probability cache identity or checksum mismatch')
        else:
            budget_path = self.directory / 'budget.json'
            budget = read_json(budget_path) if budget_path.exists() else {'contract_sha256': self.contract_sha256, 'attempts': 0}
            if budget.get('contract_sha256') != self.contract_sha256 or type(budget.get('attempts')) is not int or not 0 <= budget['attempts'] < self.max_calls:
                raise ValueError('probability backend call budget reached or invalid')
            budget['attempts'] += 1
            atomic_json(budget_path, budget)
            result = self.score(messages)
            if not isinstance(result, dict) or isinstance(result.get('probability'), bool) or not isinstance(result.get('probability'), (int, float)) or not math.isfinite(result['probability']) or not 0 <= result['probability'] <= 1:
                raise ValueError('invalid Yes-probability result')
            value = {'contract_sha256': self.contract_sha256, 'messages_sha256': name,
                     'result': result, 'result_sha256': digest(result)}
            atomic_json(path, value)
        return value['result']['probability']


class LocalYesProbability:
    def __init__(self, model_path, yes_token_text, device='cpu', dtype='float32', max_prompt_tokens=3072):
        from integrations.zeroshot.backends import LocalChat
        self.backend = LocalChat(model_path, device=device, dtype=dtype, max_new_tokens=1, max_prompt_tokens=max_prompt_tokens)
        tokens = self.backend.tokenizer.encode(yes_token_text, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError('configured Yes text must encode as exactly one token')
        self.yes_token = tokens[0]

    def __call__(self, messages):
        backend, tokenizer = self.backend, self.backend.tokenizer
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        else:
            text = '\n\n'.join(message['role'].capitalize() + ': ' + message['content'] for message in messages) + '\n\nAssistant: '
            inputs = tokenizer(text, return_tensors='pt', add_special_tokens=True)
        count = inputs['input_ids'].shape[1]
        context = getattr(backend.model.config, 'max_position_embeddings', None)
        if count > backend.max_prompt_tokens or context and count > context:
            raise ValueError('CoRAL prompt exceeds local model context; no truncation applied')
        with torch.no_grad():
            logits = backend.model(**{key: value.to(backend.device) for key, value in inputs.items()}).logits[0, -1].float()
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite local Yes-token logits')
            probability = float(torch.softmax(logits, -1)[self.yes_token])
        return {'probability': probability, 'yes_token_id': self.yes_token, 'prompt_tokens': count}
