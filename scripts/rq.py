import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('neighcf', 'llara', 'llmsrec', 'fcrllm', 'ucgrec', 'gru4rec', 'caser', 'sasrec')
SEEDS = (42, 43, 44, 45, 46)
M_VALUES = (4, 8, 12, 16, 20)
K_VALUES = (2, 4, 6, 8, 10)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def label(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9_-]+', value):
        raise ValueError('dataset names must contain only letters, numbers, underscores or hyphens')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load_project(path):
    path = Path(path).resolve()
    project = read(path)
    if project.get('schema_version') != 1 or not isinstance(project.get('datasets'), dict) or not project['datasets']:
        raise ValueError('project requires schema_version 1 and a nonempty datasets object')
    workspace = (path.parent / project['workspace']).resolve()
    relative = Path(project.get('output_root', 'runs'))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('output_root must stay relative to workspace')
    outputs = workspace / relative
    if any(p.is_symlink() for p in (outputs, *outputs.parents)):
        raise ValueError('output_root must not traverse symlinks')
    outputs = outputs.resolve()
    if outputs.is_relative_to(ROOT) or ROOT.is_relative_to(outputs):
        raise ValueError('workspace outputs must be outside the source release')
    if tuple(project.get('seeds', SEEDS)) != SEEDS:
        raise ValueError('the declared comparison uses seeds 42 through 46')
    if project.get('expected_candidates', 51) != 51:
        raise ValueError('this experiment matrix uses exactly 51 fixed evaluation candidates')
    if project.get('llm_epochs', 5) != 5 or project.get('effective_batch_size', 128) != 128:
        raise ValueError('the declared LLM comparison uses five epochs and effective batch 128')
    for name in project['datasets']:
        label(name)
        m = project['datasets'][name].get('default_m', 16)
        if type(m) is not int or m < 1:
            raise ValueError('default_m must be a positive integer')
    return project, workspace, outputs


def build_plan(path, rqs=(1, 2, 3, 4), datasets=None, models=None):
    project, workspace, outputs = load_project(path)
    selected = set(models or MODELS)
    if not selected <= set(MODELS):
        raise ValueError('unknown model filter')
    chosen = list(datasets or project['datasets'])
    if set(chosen) - project['datasets'].keys():
        raise ValueError('unknown dataset filter')
    if not set(rqs) <= {1, 2, 3, 4}:
        raise ValueError('unknown research question')
    base = read(ROOT / project.get('neighcf_config', 'integrations/neighcf/example.config.json'))
    training = read(ROOT / project.get('training_config', 'integrations/neighcf/example.training.json'))
    if type(base.get('soft_context_k')) is not int or base['soft_context_k'] < 1:
        raise ValueError('base soft_context_k must be a positive integer')
    tasks = {}
    def input_path(value):
        if value is None:
            return None
        value = value.replace('{seed}', str(seed)).replace('{dataset}', dataset)
        return str((workspace / value).resolve())
    def new_task(identifier, rq, dataset, model, seed, variant='full', file_output=False):
        if identifier in tasks:
            return tasks[identifier]
        output = outputs / identifier
        if file_output:
            output = output.with_suffix('.json')
        dataset_name = dataset
        canonical_dir = input_path(project['datasets'][dataset].get('canonical'))
        if canonical_dir and (Path(canonical_dir) / 'splits.json').is_file():
            dataset_name = read(Path(canonical_dir) / 'splits.json').get('dataset', dataset)
            if not isinstance(dataset_name, str) or not dataset_name:
                raise ValueError('canonical dataset label must be a nonempty string')
        task = {'id': identifier, 'rq': rq, 'dataset': dataset, 'dataset_label': dataset_name, 'model': model,
                'seed': seed, 'variant': variant, 'output': str(output), 'argv': [],
                'output_root': str(outputs), 'requires': [], 'depends_on': [], 'generated_files': {}, 'blocking': []}
        tasks[identifier] = task
        return task
    def asset(task, name, value, kind='file', **extra):
        resolved = input_path(value)
        entry = {'name': name, 'path': resolved, 'kind': kind, **extra}
        task['requires'].append(entry)
        if resolved is None:
            task['blocking'].append(name + ' must be configured')
        return resolved or '<missing:' + name + '>'
    def nc(task, data, variant, centres_m=None, context_k=None, fixed_tau=False):
        cfg, tc = copy.deepcopy(base), copy.deepcopy(training)
        cfg.update(training_candidate_count=100, expected_candidate_count=51)
        m = centres_m or data.get('default_m', 16)
        centres = data.get('centres_by_m', {}).get(str(m))
        if context_k is not None:
            cfg['soft_context_k'] = context_k
        tc.update(epochs=5, batch_size=128, seed=task['seed'], backbone_mode='full_model')
        if variant in ('without_collaborative', 'fixed_neighbors'):
            tc['train_gate'] = False
        if fixed_tau:
            threshold = data.get('rq3_threshold', {})
            value = threshold.get('value')
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                task['blocking'].append('RQ3 requires an explicit fixed tau strictly between zero and one')
            cfg.update(threshold_policy='fixed', threshold_value=value,
                       activity_threshold_intercept=None, activity_threshold_slope=None,
                       user_thresholds={}, gate_initial_threshold=value)
            tc['train_gate'] = False
            asset(task, 'validation-only threshold receipt', threshold.get('validation_receipt'),
                  'threshold_receipt', selected_tau=value,
                  valid_path=input_path(data['canonical']) + '/valid.jsonl',
                  split_metadata=input_path(data['canonical']) + '/splits.json')
        config_path = outputs / '.plans' / task['id'] / 'config.json'
        train_path = outputs / '.plans' / task['id'] / 'training.json'
        task['generated_files'][str(config_path)] = cfg
        task['generated_files'][str(train_path)] = tc
        common = ['--config', str(config_path), '--items', asset(task, 'items', data.get('items')),
                  '--embeddings', asset(task, 'item embeddings', data.get('embeddings')),
                  '--centres', asset(task, f'{m} cluster centres', centres, 'centres', rows=m),
                  '--events', asset(task, 'timestamped training events', data.get('events')),
                  '--variant', variant]
        return common, str(train_path)
    def nc_train(task, data, variant='full', centres_m=None, context_k=None, fixed_tau=False):
        common, train_config = nc(task, data, variant, centres_m, context_k, fixed_tau)
        canonical_dir = asset(task, 'canonical dataset', data.get('canonical'), 'canonical')
        task['argv'] = [sys.executable, '-m', 'integrations.neighcf.run', 'train', *common,
                        '--training-config', train_config,
                        '--train', canonical_dir + '/train.jsonl', '--valid', canonical_dir + '/valid.jsonl',
                        '--test', canonical_dir + '/test.jsonl',
                        '--model-path', asset(task, 'Llama-2 backbone', project.get('model_path'), 'directory'),
                        '--dataset-name', task['dataset_label'],
                        '--output', task['output'], '--device', project.get('device', 'cuda:0')]
    for dataset in chosen:
        data = project['datasets'][dataset]
        for seed in SEEDS:
            if 1 in rqs:
                for model in MODELS:
                    if model not in selected:
                        continue
                    task = new_task(f'rq1/{dataset}/{model}/seed-{seed}', 1, dataset, model, seed)
                    if model == 'neighcf':
                        nc_train(task, data)
                        continue
                    source_key = 'BSARec' if model in ('gru4rec', 'caser', 'sasrec') else {
                        'llara': 'LLaRA', 'llmsrec': 'LLM-SRec', 'fcrllm': 'FCRLLM', 'ucgrec': 'LLM-SRec'}[model]
                    common = ['--data', asset(task, 'canonical dataset', data.get('canonical'), 'canonical'),
                              '--source', asset(task, source_key + ' source', project.get('sources', {}).get(source_key), 'directory'),
                              '--output', task['output'], '--seed', str(seed), '--expected-candidates', '51']
                    if model in ('gru4rec', 'caser', 'sasrec'):
                        epochs = project.get('traditional_epochs')
                        if type(epochs) is not int or epochs < 1:
                            task['blocking'].append('traditional_epochs must be explicitly set to a positive integer')
                        task['argv'] = [sys.executable, '-m', 'integrations.traditional.run', 'train', *common,
                                        '--model', model, '--dataset', task['dataset_label'], '--epochs', str(epochs),
                                        '--batch-size', '256', '--dimension', '64', '--learning-rate', '0.001',
                                        '--device', 'cpu' if project.get('device') == 'cpu' else 'cuda']
                        continue
                    task['argv'] = [sys.executable, str(ROOT / 'integrations' / model / 'run.py'), 'train', *common,
                                    '--dataset-name', task['dataset_label'], '--epochs', '5', '--device', project.get('device', 'cuda:0')]
                    if model == 'fcrllm':
                        task['argv'] += ['--features', asset(task, 'FCRLLM feature bundle', data.get('fcrllm_features'), 'directory'),
                                         '--batch-size', '128']
                    else:
                        task['argv'] += ['--model-path', asset(task, 'Llama-2 backbone', project.get('model_path'), 'directory'),
                                         '--effective-batch-size', '128']
                    if model in ('llara', 'llmsrec', 'ucgrec'):
                        teacher_key = 'llmsrec' if model == 'ucgrec' else model
                        task['argv'] += ['--teacher', asset(task, model + ' teacher', data.get(teacher_key + '_teacher'), 'teacher')]
                    if model == 'llmsrec' and data.get('llmsrec_omit_dates', False):
                        task['argv'].append('--omit-dates')
                    if model == 'llara':
                        prompt = data.get('llara_prompt')
                        if prompt not in ('movie.txt', 'game.txt', 'artist.txt', 'book.txt'):
                            task['blocking'].append('an explicit supported llara_prompt is required')
                        task['argv'] += ['--prompt-name', str(prompt)]
                    if model == 'ucgrec':
                        policy = data.get('ucgrec_policy')
                        if policy not in ('steam_native_v1', 'movielens_native_v1', 'goodreads_semantic_adaptation_v1', 'lastfm_artist_semantic_adaptation_v1'):
                            task['blocking'].append('an explicit supported ucgrec_policy is required')
                        task['argv'] += ['--ucg-source', asset(task, 'UCGRec source', project.get('sources', {}).get('UCGRec'), 'directory'),
                                         '--dataset-policy', str(policy),
                                         '--train-events', asset(task, 'UCGRec timestamped events', data.get('events')),
                                         '--pilot-receipt', asset(task, 'full batch UCGRec pilot', data.get('ucgrec_pilot'), 'ucgrec_pilot')]
                        if policy != 'steam_native_v1':
                            task['argv'] += ['--features', asset(task, 'UCGRec semantic features', data.get('ucgrec_features')),
                                             '--feature-manifest', asset(task, 'UCGRec feature provenance', data.get('ucgrec_feature_manifest'))]
            if 'neighcf' not in selected:
                continue
            if 2 in rqs:
                for variant in ('full', 'without_collaborative', 'fixed_neighbors'):
                    task = new_task(f'rq2/{dataset}/{variant}/seed-{seed}', 2, dataset, 'neighcf', seed, variant)
                    nc_train(task, data, variant)
            if 3 in rqs:
                points = [(m, base['soft_context_k'], 'M') for m in M_VALUES]
                points += [(data.get('default_m', 16), k, 'K') for k in K_VALUES]
                for m, k, axis in points:
                    identifier = f'rq3/{dataset}/m-{m}-k-{k}/seed-{seed}'
                    if identifier in tasks:
                        tasks[identifier]['axes'].append(axis)
                        continue
                    task = new_task(identifier, 3, dataset, 'neighcf', seed)
                    task['axes'] = [axis]
                    nc_train(task, data, centres_m=m, context_k=k, fixed_tau=True)
                task = new_task(f'rq3/{dataset}/activity/seed-{seed}', 3, dataset, 'neighcf', seed, file_output=True)
                parent = f'rq1/{dataset}/neighcf/seed-{seed}'
                task['depends_on'] = [parent]
                ranks = str(outputs / parent / 'test.ranks.jsonl')
                threshold = project.get('activity_threshold')
                if type(threshold) is not int or threshold < 1:
                    task['blocking'].append('activity_threshold must be explicitly set to a positive history length')
                task['requires'] += [{'name': 'NeighCF test ranks', 'path': ranks, 'kind': 'file'},
                                     {'name': 'NeighCF completed test run', 'path': str(outputs / parent / 'completion.json'), 'kind': 'complete'}]
                canonical_dir = asset(task, 'canonical dataset', data.get('canonical'), 'canonical')
                task['argv'] = [sys.executable, 'scripts/analyze.py', 'activity', '--ranks', ranks,
                                '--queries', canonical_dir + '/test.jsonl', '--threshold', str(threshold),
                                '--activity-field', 'history_length_before_truncation', '--output', task['output']]
            if 4 in rqs:
                for variant in ('full', 'global_pool', 'fixed_neighbors'):
                    task = new_task(f'rq4/{dataset}/{variant}/trial-{seed}', 4, dataset, 'neighcf', seed, variant, True)
                    common, _ = nc(task, data, variant)
                    task['generated_files'] = {p: v for p, v in task['generated_files'].items() if p.endswith('/config.json')}
                    task['measurement_scope'] = 'retrieval_and_prompt_only; excludes model inference'
                    task['repetition'] = seed - 41
                    task['argv'] = [sys.executable, 'scripts/benchmark.py', *common,
                                    '--queries', asset(task, 'canonical test queries', str(Path(data['canonical']) / 'test.jsonl')),
                                    '--output', task['output'], '--warmup', '1']
    ordered = list(tasks.values())
    for task in ordered:
        for req in task['requires']:
            if req['path'] is None or (task['depends_on'] and req['name'].startswith('NeighCF ')):
                continue
            source = Path(req['path']).resolve()
            if source == outputs or source.is_relative_to(outputs) or outputs.is_relative_to(source):
                raise ValueError('output_root must be separate from source and input asset paths')
        task['missing_paths'] = [req['path'] for req in task['requires'] if req['path'] is not None and not Path(req['path']).exists()]
        task['asset_paths_present'] = not task['blocking'] and not task['missing_paths']
        task['preflight_required'] = True
    return {'schema_version': 1, 'execution': 'plan_only', 'workspace': str(workspace), 'output_root': str(outputs),
            'project_file': str(Path(path).resolve()), 'project_sha256': sha(path),
            'seeds': list(SEEDS), 'expected_candidates': 51, 'tasks': ordered}


def validate_requirement(req):
    path = req['path']
    if path is None:
        raise ValueError(req['name'] + ' is not configured')
    path = Path(path)
    kind = req['kind']
    if kind in ('directory', 'canonical'):
        if not path.is_dir():
            raise ValueError('missing directory: ' + str(path))
        if kind == 'canonical':
            for name in ('items.json', 'manifest.json', 'splits.json', 'train.jsonl', 'valid.jsonl', 'test.jsonl'):
                if not (path / name).is_file():
                    raise ValueError('missing canonical file: ' + str(path / name))
            splits, manifest = read(path / 'splits.json'), read(path / 'manifest.json')
            counts = manifest.get('protocol', {}).get('candidates', {})
            if (splits.get('protocol_sha256') != sha(path / 'manifest.json')
                    or splits.get('expected_candidates') != 51
                    or counts.get('count') != 51 or counts.get('evaluation_count') != 51
                    or counts.get('training_count') != 100):
                raise ValueError('canonical manifest must bind 100 training and 51 evaluation candidates')
        return
    if not path.is_file():
        raise ValueError('missing file: ' + str(path))
    if kind == 'centres':
        centres = read(path)
        if (not isinstance(centres, list) or len(centres) != req['rows'] or not centres
                or not isinstance(centres[0], list) or not centres[0]
                or any(not isinstance(row, list) or len(row) != len(centres[0]) for row in centres)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for row in centres for value in row)):
            raise ValueError('cluster centres do not match the declared M')
    elif kind == 'threshold_receipt':
        receipt = read(path)
        protocol = read(req['split_metadata']).get('protocol_sha256')
        if (receipt.get('split') != 'valid' or type(receipt.get('selected_tau')) not in (int, float)
                or receipt['selected_tau'] != req['selected_tau']
                or receipt.get('validation_sha256') != sha(req['valid_path'])
                or not isinstance(protocol, str) or not re.fullmatch('[0-9a-f]{64}', protocol)
                or receipt.get('protocol_sha256') != protocol):
            raise ValueError('RQ3 threshold must have matching validation-only selection provenance')
    elif kind == 'teacher':
        for name in ('teacher.metadata.json', 'teacher.completion.json'):
            if not path.with_name(name).is_file():
                raise ValueError('missing validation-selected teacher sidecar: ' + name)
    elif kind == 'ucgrec_pilot':
        receipt = read(path)
        manifest = read(path.with_name('run.manifest.json'))
        if (receipt.get('status') != 'complete' or receipt.get('run_kind') != 'pilot'
                or receipt.get('fresh_checkpoint_restore_verified') is not True
                or type(receipt.get('largest_optimizer_batch')) is not int
                or receipt['largest_optimizer_batch'] != 128
                or manifest.get('arguments', {}).get('pilot_examples') != 128):
            raise ValueError('UCGRec requires the successful full batch128 pilot and fresh restore receipt')
        if not path.with_name('backbone.manifest.json').is_file():
            raise ValueError('missing UCGRec pilot backbone manifest')
    elif kind == 'complete':
        receipt = read(path)
        if receipt.get('status') != 'complete' or receipt.get('test_evaluated') is not True:
            raise ValueError('activity analysis requires a completed selected-checkpoint test evaluation')


def validate_destinations(task):
    root = Path(task['output_root']).absolute()
    for raw in (task['output'], *task['generated_files']):
        path = Path(raw).absolute()
        if not path.is_relative_to(root):
            raise ValueError('planned output escapes the project output root')
        if any(ancestor.is_symlink() for ancestor in (path, *path.parents)):
            raise ValueError('planned output or generated configuration traverses a symlink')


def execute_task(task, resume=False):
    validate_destinations(task)
    if task['blocking']:
        raise ValueError('; '.join(task['blocking']))
    for req in task['requires']:
        validate_requirement(req)
    for path, value in task['generated_files'].items():
        validate_destinations(task)
        destination = Path(path)
        content = canonical(value) + '\n'
        if destination.exists():
            if destination.read_text() != content:
                raise ValueError('materialized experiment configuration changed: ' + str(destination))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('x') as stream:
                stream.write(content)
    argv = list(task['argv'])
    if resume:
        if task['rq'] == 4 or '/activity/' in task['id']:
            raise ValueError('resume applies only to training tasks')
        argv.append('--resume')
    output = Path(task['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(argv, cwd=ROOT, check=False)
    return result.returncode


def preflight_selection(tasks, resume=False):
    selected = {task['id'] for task in tasks}
    for task in tasks:
        validate_destinations(task)
        if task['blocking']:
            raise ValueError(task['id'] + ': ' + '; '.join(task['blocking']))
        if resume and (task['rq'] == 4 or '/activity/' in task['id']):
            raise ValueError('select training tasks individually when resuming')
        for req in task['requires']:
            if task['depends_on'] and set(task['depends_on']) <= selected and req['name'].startswith('NeighCF '):
                continue
            validate_requirement(req)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', choices=('plan', 'run'), default='plan')
    parser.add_argument('--project', type=Path, default=ROOT / 'configs/project.example.json')
    parser.add_argument('--rq', type=int, nargs='+', choices=(1, 2, 3, 4), default=[1, 2, 3, 4])
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--models', nargs='+', choices=MODELS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--id', nargs='+')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan(args.project, args.rq, args.datasets, args.models)
    if args.command == 'plan':
        if args.id or args.all or args.resume:
            parser.error('--id, --all and --resume require the explicit run command')
        text = json.dumps(plan, indent=2, allow_nan=False) + '\n'
        if args.output:
            with args.output.open('x') as stream:
                stream.write(text)
        else:
            print(text, end='')
        return 0
    if args.output:
        parser.error('--output writes a plan only')
    if bool(args.id) == args.all:
        parser.error('run requires exactly one of --id ID... or --all')
    known = {task['id'] for task in plan['tasks']}
    if args.id and set(args.id) - known:
        parser.error('unknown task ID')
    selected = [task for task in plan['tasks'] if args.all or task['id'] in args.id]
    preflight_selection(selected, args.resume)
    for task in selected:
        code = execute_task(task, args.resume)
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
