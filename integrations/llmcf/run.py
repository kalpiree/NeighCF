import argparse
import json
from pathlib import Path
from .common import canonical_bundle
from .engine import add_arguments, run_training, seed_all, validate_training_arguments
from .runtime import Runtime, configuration
from .assets import Assets
from .source import COMMIT


def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser())
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--num-neg', type=int, required=True)
    args = parser.parse_args(argv)
    validate_training_arguments(args)
    bundle = canonical_bundle(args.data, args.expected_candidates, args.training_candidates)
    config = configuration(args.source, args.config)
    assets = Assets(args.assets, bundle, args.seed, config['max_len'])
    if not 1 <= args.num_neg <= len(bundle['items']):
        raise ValueError('invalid catalog sampled-negative count')
    for rows in bundle['rows'].values():
        for row in rows:
            assets.retrieve(row)
    if args.command == 'audit':
        print(json.dumps({'status': 'inputs_validated_no_model_execution', 'source_commit': COMMIT, 'assets_sha256': assets.hashes, 'rows': {key: len(value) for key, value in bundle['rows'].items()}, 'config': config}, sort_keys=True))
        return
    if args.output is None:
        parser.error('train requires --output')
    seed_all(args.seed, args.device)
    result = run_training(args, bundle, Runtime(args, bundle, config, assets))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
