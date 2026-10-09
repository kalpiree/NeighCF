import argparse
import json
from pathlib import Path
from integrations.llmcf.common import canonical_bundle, source_bytes
from integrations.llmcf.engine import add_arguments, run_training, seed_all, validate_training_arguments
from .source import COMMIT, FILES
from .runtime import Runtime
from .data import title


def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser())
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--item-kind', choices=('movie', 'book', 'game', 'artist'), required=True)
    parser.add_argument('--demonstrations', type=int, default=4)
    parser.add_argument('--history-length', type=int, default=10)
    parser.add_argument('--max-txt-len', type=int, default=3072)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    args = parser.parse_args(argv)
    validate_training_arguments(args)
    bundle = canonical_bundle(args.data, args.expected_candidates, args.training_candidates)
    source_bytes(args.source, COMMIT, FILES)
    if args.demonstrations < 0 or args.history_length < 1 or args.max_txt_len < 1:
        raise ValueError('invalid context bounds')
    for item in bundle['items']:
        title(bundle['items'], item)
    if args.command == 'audit':
        print(json.dumps({'status': 'canonical_and_native_source_validated_no_model_execution', 'source_commit': COMMIT, 'rows': {key: len(value) for key, value in bundle['rows'].items()}, 'model_path_exists': args.model_path.is_dir(), 'variant': Runtime.description}, sort_keys=True))
        return
    if args.output is None:
        parser.error('train requires --output')
    seed_all(args.seed, args.device)
    result = run_training(args, bundle, Runtime(args, bundle))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
