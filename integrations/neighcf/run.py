import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrations.neighcf.config import Config
from integrations.neighcf.prompting import build_prompt
from integrations.neighcf.retrieval import NeighCFRetriever, canonical, validate_query, digest


def load_json(path):
    return json.loads(Path(path).read_text())


def load_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_retriever(args):
    config = Config.from_dict(load_json(args.config))
    retriever = NeighCFRetriever(load_json(args.items), load_json(args.embeddings), load_json(args.centres), load_rows(args.events), config)
    return config, retriever


def protocol_metadata(args, paths, seed):
    from integrations.neighcf.training import file_sha256
    from evaluate import validate_metadata
    split_sources = {}
    protocols, datasets = set(), set()
    config = Config.from_dict(load_json(args.config))
    for path in paths:
        splits = Path(path).parent / 'splits.json'
        if splits.is_file():
            value = load_json(splits)
            manifest_name = value.get('manifest', 'manifest.json')
            if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
                raise ValueError('canonical manifest pointer must name a file in the split directory')
            manifest_path = splits.parent / manifest_name
            if file_sha256(manifest_path) != value.get('protocol_sha256'):
                raise ValueError('canonical manifest does not match protocol SHA-256')
            manifest = load_json(manifest_path)
            inventory = manifest['output_file_sha256']
            for supplied in (path, args.items, args.events):
                expected = inventory.get(Path(supplied).name)
                if expected is None or file_sha256(supplied) != expected:
                    raise ValueError('supplied query/items/events do not match canonical manifest bytes')
            candidates = manifest['protocol']['candidates']
            evaluation_count = candidates.get('evaluation_count', candidates['count'])
            training_count = candidates.get('training_count', candidates['count'])
            if (training_count, evaluation_count) != (config.training_candidate_count, config.expected_candidate_count):
                raise ValueError('configured training/evaluation counts disagree with canonical manifest')
            if value.get('protocol_sha256'):
                protocols.add(value['protocol_sha256'])
            if value.get('dataset'):
                datasets.add(value['dataset'])
            split_sources[str(len(split_sources))] = file_sha256(splits)
    if len(protocols) > 1 or len(datasets) > 1:
        raise ValueError('query directories disagree on canonical protocol or dataset')
    protocol = args.protocol_sha256 or next(iter(protocols), None)
    dataset = args.dataset_name or next(iter(datasets), None)
    if protocols and protocol not in protocols:
        raise ValueError('explicit protocol disagrees with canonical splits')
    if datasets and dataset not in datasets:
        raise ValueError('explicit dataset name disagrees with canonical splits')
    metadata = {'model': 'NeighCF', 'dataset': dataset, 'seed': seed, 'protocol_sha256': protocol, 'implementation': 'documented_reference_reconstruction', 'variant': args.variant, 'canonical_split_sha256': split_sources, 'query_file_sha256': {str(index): file_sha256(path) for index, path in enumerate(paths)}}
    return validate_metadata(metadata)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'prompt', 'score', 'train'):
        command = commands.add_parser(name)
        for argument in ('config', 'items', 'embeddings', 'centres', 'events'):
            command.add_argument('--' + argument, required=True)
        command.add_argument('--variant', default='full', choices=('full', 'without_collaborative', 'fixed_neighbors', 'global_pool'))
        command.add_argument('--output', required=True)
        if name != 'train':
            command.add_argument('--queries', required=True)
        if name in {'score', 'train'}:
            command.add_argument('--model-path', required=True)
            command.add_argument('--device', default='cpu')
            command.add_argument('--dataset-name')
            command.add_argument('--protocol-sha256')
        if name == 'score':
            command.add_argument('--checkpoint')
            command.add_argument('--seed', type=int)
        if name == 'train':
            for argument in ('training-config', 'train', 'valid'):
                command.add_argument('--' + argument, required=True)
            command.add_argument('--test')
            command.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    from integrations.neighcf.training import preflight_output, file_sha256
    inputs = [getattr(args, name) for name in ('config', 'items', 'embeddings', 'centres', 'events', 'queries', 'train', 'valid', 'test', 'training_config', 'model_path', 'checkpoint') if getattr(args, name, None)]
    output = preflight_output(args.output, inputs)
    config, retriever = load_retriever(args)
    if args.command == 'train':
        from integrations.neighcf.scoring import load_local_model
        from integrations.neighcf.training import TrainingConfig, train
        training_config = TrainingConfig.from_dict(load_json(args.training_config))
        metadata = protocol_metadata(args, [args.train, args.valid] + ([args.test] if args.test else []), training_config.seed)
        model, tokenizer = load_local_model(args.model_path, args.device)
        train(config, training_config, retriever, load_rows(args.train), load_rows(args.valid), model, tokenizer, args.output, args.device, args.variant, args.resume, load_rows(args.test) if args.test else None, metadata=metadata)
        return
    rows = load_rows(args.queries)
    if not rows or len({row['query_id'] for row in rows}) != len(rows):
        raise ValueError('queries must be nonempty with unique query IDs')
    for query in rows:
        validate_query(query, retriever.items, config.expected_candidate_count)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    if args.command == 'score':
        from integrations.neighcf.scoring import load_local_model
        from integrations.neighcf.training import atomic_json, evaluate, restore_for_scoring
        from evaluate import evaluate_records
        if config.mode == 'soft_embedding_gate' and not args.checkpoint:
            raise ValueError('soft reconstruction scoring requires a trained/frozen checkpoint')
        manifest = load_json(Path(args.checkpoint).parent / 'manifest.json') if args.checkpoint else None
        if manifest and digest(manifest['contract']) != manifest['contract_sha256']:
            raise ValueError('run manifest contract digest mismatch')
        metadata = protocol_metadata(args, [args.queries], manifest['contract']['metadata']['seed'] if manifest else args.seed)
        model, tokenizer = load_local_model(args.model_path, args.device)
        gate, checkpoint = restore_for_scoring(args.checkpoint, config, retriever, model, tokenizer, args.device, args.variant) if args.checkpoint else (None, None)
        if checkpoint and checkpoint['contract_sha256'] != manifest['contract_sha256']:
            raise ValueError('checkpoint and run manifest differ')
        if checkpoint and any(metadata[key] != checkpoint['contract']['metadata'][key] for key in ('model', 'dataset', 'protocol_sha256')):
            raise ValueError('scoring protocol metadata differs from checkpoint')
        summary, ranks, records = evaluate(model, tokenizer, gate, config, retriever, rows, args.device, args.variant)
        common_result, ranks = evaluate_records(records, metadata, expected_candidates=config.expected_candidate_count)
        common_result['reconstruction'] = {'variant': args.variant, 'checkpoint_sha256': file_sha256(args.checkpoint) if args.checkpoint else None}
        atomic_json(output / 'metrics.json', common_result)
        for name, values in (('ranks.jsonl', ranks), ('scores.jsonl', records)):
            with (output / name).open('wb') as stream:
                for value in values:
                    stream.write(canonical(value) + b'\n')
    else:
        summaries = []
        with (output / (args.command + '.jsonl')).open('wb') as stream:
            for query in rows:
                found = retriever.retrieve(query, args.variant, soft_context=config.mode == 'soft_embedding_gate' and args.variant in {'full', 'global_pool'})
                value = {'query_id': query['query_id'], 'retrieval': found}
                if args.command == 'prompt':
                    prompt = build_prompt(query, retriever.items, found, config)
                    value['prompt'] = {'text': prompt.text, 'neighbor_spans': prompt.neighbor_spans, 'candidate_ids': prompt.candidate_ids, 'candidate_titles': prompt.candidate_titles}
                stream.write(canonical(value) + b'\n')
                summaries.append({'pool': found['pool_users'], 'neighbors': len(found['neighbors'])})
        (output / 'summary.json').write_bytes(canonical({'implementation': 'documented_reference_reconstruction', 'queries': len(rows), 'source_identity': retriever.source_identity, 'literal_positive_retains_full_eligible_pool': config.index_policy == 'literal_positive', 'counts': summaries}) + b'\n')


if __name__ == '__main__':
    main()
