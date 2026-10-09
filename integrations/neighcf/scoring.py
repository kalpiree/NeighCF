import time


class CandidateLikelihoodScorer:
    def __init__(self, model, tokenizer, config, device='cpu', gate=None):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device, self.gate = device, gate
        self.last_measurements = {}

    def prepare(self, prompt):
        cfg = self.config
        encoded = self.tokenizer(prompt.text, add_special_tokens=True, return_offsets_mapping=self.gate is not None)
        prefix = encoded['input_ids']
        if not prefix or len(prefix) > cfg.max_prompt_tokens:
            raise ValueError('empty prompt or prompt exceeds explicit token limit; no silent truncation')
        owners = [-1] * len(prefix)
        if self.gate is not None:
            offsets = encoded.get('offset_mapping')
            if offsets is None or len(offsets) != len(prefix):
                raise ValueError('soft embedding gates require exact tokenizer offsets')
            for index, (left, right) in enumerate(offsets):
                matches = [number for number, (start, end) in enumerate(prompt.neighbor_spans) if right > left and left < end and right > start]
                if len(matches) > 1:
                    raise ValueError('a token overlaps multiple neighbor segments')
                if matches:
                    owners[index] = matches[0]
            if set(owners) - {-1} != set(range(len(prompt.neighbor_spans))):
                raise ValueError('every neighbor segment must map to at least one token')
        suffixes = []
        for title in prompt.candidate_titles:
            suffix = self.tokenizer(title + cfg.candidate_terminator, add_special_tokens=False)['input_ids']
            if not suffix or len(prefix) + len(suffix) > cfg.max_sequence_tokens:
                raise ValueError('candidate is empty or exceeds sequence token limit')
            suffixes.append(suffix)
        return prefix, owners, suffixes

    def tensor_scores(self, prompt):
        import torch
        prefix, owners, suffixes = self.prepare(prompt)
        start = time.perf_counter()
        gates = self.gate(prompt.neighbor_scores, prompt.user_id, prompt.history_length) if self.gate is not None else None
        results = []
        for suffix in suffixes:
            token_ids = torch.tensor([prefix + suffix], dtype=torch.long, device=self.device)
            mask = torch.ones_like(token_ids)
            if self.gate is None:
                output = self.model(input_ids=token_ids, attention_mask=mask)
            else:
                embeddings = self.model.get_input_embeddings()(token_ids)
                weights = [gates[owner] if owner >= 0 else embeddings.new_ones(()) for owner in owners]
                weights.extend(embeddings.new_ones(()) for _ in suffix)
                scaled = embeddings * torch.stack(weights).to(embeddings.dtype)[None, :, None]
                output = self.model(inputs_embeds=scaled, attention_mask=mask)
            logits = output.logits[:, len(prefix) - 1:len(prefix) + len(suffix) - 1, :].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite candidate logits')
            targets = token_ids[:, len(prefix):]
            values = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            score = values.sum() if self.config.likelihood_reduction == 'sum' else values.mean()
            results.append(score)
        scores = torch.stack(results)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('nonfinite candidate likelihood')
        self.last_measurements = {'prompt_tokens': len(prefix), 'candidate_suffix_tokens': [len(suffix) for suffix in suffixes], 'candidate_forward_passes': len(suffixes), 'scoring_seconds': time.perf_counter() - start, 'likelihood_definition': 'independently_tokenized_title_suffix_given_tokenized_prompt', 'gate_count': 0 if gates is None else len(gates)}
        return scores, gates

    def score(self, prompt):
        import torch
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                scores, _ = self.tensor_scores(prompt)
                return scores.detach().cpu().tolist()
        finally:
            self.model.train(was_training)


def load_local_model(model_path, device='cpu'):
    from pathlib import Path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(model_path).resolve()
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError('model-path must be an existing local Hugging Face checkpoint directory')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32)
    model.to(device)
    return model, tokenizer
