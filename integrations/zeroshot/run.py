import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrations.zeroshot.backends import LocalChat, OpenAIChat
from integrations.zeroshot.core import canonical, digest, load_data, local_identity, messages_for, run_queries


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('plan', 'export', 'run'))
    p.add_argument('--backend', choices=('llama2', 'gpt4'), required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--split', choices=('valid', 'test'), default='test')
    p.add_argument('--expected-candidates', type=int, default=51)
    p.add_argument('--dataset-name')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model-id')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    p.add_argument('--history-limit', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--max-prompt-tokens', type=int, default=3072)
    p.add_argument('--execute-api', action='store_true')
    p.add_argument('--max-requests', type=int)
    p.add_argument('--responses', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-uncertain', action='store_true')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.max_new_tokens, args.max_prompt_tokens) < 1 or args.history_limit < 0:
        p.error('Token limits must be positive and history limit nonnegative')
    if args.backend == 'gpt4':
        if not args.model_id or not (args.model_id == 'gpt-4' or args.model_id.startswith('gpt-4-')):
            p.error('Specify --model-id gpt-4 or an explicit gpt-4-* snapshot')
        if args.model_path:
            p.error('GPT-4 uses the API backend, not local model weights')
    elif args.execute_api or args.max_requests is not None:
        p.error('API execution flags apply only to GPT-4')
    model_id = args.model_id or 'meta-llama/Llama-2-7b-hf'
    items, rows, splits, hashes = load_data(args.data, args.expected_candidates)
    dataset = args.dataset_name or splits.get('dataset')
    if splits.get('dataset') and dataset != splits['dataset']:
        p.error('Dataset label differs from canonical metadata')
    metadata = {'model': 'LLaMA2-zero-shot' if args.backend == 'llama2' else 'GPT-4-zero-shot',
                'dataset': dataset, 'seed': args.seed, 'protocol_sha256': splits['protocol_sha256']}
    settings = {'backend': args.backend, 'model_id': model_id, 'split': args.split,
                'expected_candidates': args.expected_candidates, 'history_limit': args.history_limit,
                'max_new_tokens': args.max_new_tokens, 'max_prompt_tokens': args.max_prompt_tokens,
                'dtype': args.dtype if args.backend == 'llama2' else None,
                'device': args.device if args.backend == 'llama2' else None,
                'source_file_sha256': hashes, 'temperature': 0, 'prompt_version': 'zero-shot-ranking-v1'}
    if args.command == 'plan':
        print(json.dumps({'status': 'plan_only', 'metadata': metadata, 'settings': settings,
                          'queries': len(rows[args.split]), 'paid_api_calls': 0,
                          'requires_local_model': args.backend == 'llama2' and not args.responses,
                          'requires_explicit_api_execution': args.backend == 'gpt4' and not args.responses}, indent=2))
        return 0
    if not args.output:
        p.error('--output is required for export and run')
    from integrations.neighcf.training import preflight_output
    protected = [args.data] + [path for path in (args.model_path, args.responses) if path]
    output = preflight_output(args.output, protected)
    if args.command == 'export':
        if args.resume or args.execute_api or args.responses:
            p.error('Export does not accept execution or resume flags')
        if output.exists():
            p.error('Export output already exists')
        output.mkdir(parents=True)
        with (output / 'requests.jsonl').open('x') as stream:
            for query in rows[args.split]:
                messages = messages_for(query, items, args.history_limit)
                body = OpenAIChat(model_id, args.max_new_tokens).request_body(messages)
                row = {'query_id': query['query_id'], 'request_sha256': digest({'messages': messages, 'settings': settings}),
                       'body': body}
                stream.write(canonical(row) + '\n')
        (output / 'settings.json').write_text(canonical(settings) + '\n')
        print(json.dumps({'status': 'exported', 'queries': len(rows[args.split]), 'paid_api_calls': 0}))
        return 0
    if args.responses:
        if args.execute_api:
            p.error('Response replay and API execution are mutually exclusive')
        records = [json.loads(line) for line in args.responses.read_text().splitlines()]
        if len(records) != len(rows[args.split]) or len({row['query_id'] for row in records}) != len(records):
            p.error('Response replay must cover each query exactly once')
        by_query = {row['query_id']: row for row in records}
        by_request = {}
        for query in rows[args.split]:
            key = digest({'messages': messages_for(query, items, args.history_limit), 'settings': settings})
            record = by_query.get(query['query_id'])
            if not record or record.get('request_sha256') != key:
                p.error('Response replay request hash mismatch')
            if key in by_request and by_request[key] != record['response']:
                p.error('Identical replay requests contain contradictory responses')
            by_request[key] = record['response']
        def backend(messages):
            return by_request[digest({'messages': messages, 'settings': settings})]
    elif args.backend == 'gpt4':
        if not args.execute_api or args.max_requests is None or args.max_requests < 1:
            p.error('Paid execution requires --execute-api and a positive --max-requests budget')
        if not os.environ.get('OPENAI_API_KEY'):
            p.error('OPENAI_API_KEY is required for explicit API execution')
        backend = OpenAIChat(model_id, args.max_new_tokens)
    else:
        if not args.model_path:
            p.error('--model-path is required for local LLaMA2 inference')
        settings['local_model_sha256'] = local_identity(args.model_path)
        import torch
        torch.manual_seed(args.seed)
        backend = LocalChat(args.model_path, args.device, args.dtype, args.max_new_tokens, args.max_prompt_tokens)
    result = run_queries(items, rows[args.split], metadata, settings, output, backend, args.resume,
                         args.max_requests, args.retry_uncertain)
    print(json.dumps(result))
    return 75 if result['status'] == 'paused' else 0


if __name__ == '__main__':
    raise SystemExit(main())
