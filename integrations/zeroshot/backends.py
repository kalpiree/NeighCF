import json
import os
from pathlib import Path
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('API redirects are refused')


class OpenAIChat:
    def __init__(self, model_id, max_new_tokens=1024, timeout=120):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError('An explicit model ID is required')
        if type(max_new_tokens) is not int or max_new_tokens < 1 or timeout <= 0:
            raise ValueError('Token limit and timeout must be positive')
        self.model_id, self.max_new_tokens, self.timeout = model_id, max_new_tokens, timeout

    def request_body(self, messages):
        if not messages or any(message.get('role') not in {'system', 'user', 'assistant'} or not isinstance(message.get('content'), str) for message in messages):
            raise ValueError('Expected nonempty text chat messages')
        return {'model': self.model_id, 'messages': messages, 'temperature': 0,
                'max_tokens': self.max_new_tokens, 'n': 1, 'stream': False}

    def __call__(self, messages):
        body = self.request_body(messages)
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('OPENAI_API_KEY is required for explicit API execution')
        request = urllib.request.Request('https://api.openai.com/v1/chat/completions',
                    data=json.dumps(body).encode(), method='POST',
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        opener = urllib.request.build_opener(NoRedirect)
        try:
            with opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'OpenAI HTTP status {error.code}; request was not retried') from None
        except urllib.error.URLError:
            raise RuntimeError('OpenAI transport failure; request was not retried') from None
        choices = result.get('choices')
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('Expected exactly one API completion')
        choice = choices[0]
        message = choice.get('message', {})
        if message.get('refusal') or not isinstance(message.get('content'), str):
            raise ValueError('API response has no usable text completion')
        return {'text': message['content'], 'finish_reason': choice.get('finish_reason'),
                'model': result.get('model'), 'usage': result.get('usage'),
                'system_fingerprint': result.get('system_fingerprint'), 'request_id': result.get('id')}


class LocalChat:
    def __init__(self, model_path, device='cpu', dtype='float32', max_new_tokens=1024, max_prompt_tokens=3072):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = Path(model_path)
        if not path.is_dir() or not (path / 'config.json').is_file():
            raise ValueError('A local Hugging Face checkpoint directory is required')
        if dtype not in {'float32', 'float16', 'bfloat16'}:
            raise ValueError('Unsupported dtype')
        if min(max_new_tokens, max_prompt_tokens) < 1:
            raise ValueError('Token limits must be positive')
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                        trust_remote_code=False, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
        self.device, self.max_new_tokens, self.max_prompt_tokens = device, max_new_tokens, max_prompt_tokens
        self.model_id = str(path.name)

    def __call__(self, messages):
        import torch
        if not messages or any(message.get('role') not in {'system', 'user', 'assistant'} or not isinstance(message.get('content'), str) for message in messages):
            raise ValueError('Expected nonempty text chat messages')
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        else:
            text = '\n\n'.join(message['role'].capitalize() + ': ' + message['content'] for message in messages) + '\n\nAssistant: '
            inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
        length = inputs['input_ids'].shape[1]
        context = getattr(self.model.config, 'max_position_embeddings', None)
        if length > self.max_prompt_tokens or context and length + self.max_new_tokens > context:
            raise ValueError('Prompt and requested generation exceed configured context; no truncation applied')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError('An explicit EOS token is required')
        with torch.no_grad():
            tokens = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                        pad_token_id=eos, eos_token_id=eos)
        suffix = tokens[0, length:].tolist()
        stopped = bool(suffix) and suffix[-1] == eos
        return {'text': self.tokenizer.decode(suffix, skip_special_tokens=True),
                'finish_reason': 'stop' if stopped else 'length', 'model': self.model_id,
                'usage': {'prompt_tokens': length, 'completion_tokens': len(suffix)}}
