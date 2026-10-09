import ast
import hashlib
import json
from pathlib import Path
import subprocess
from adapters.llmsrec_data import load_canonical
from integrations.resume import digest, file_digest


def source_bytes(source, commit, files):
    source = Path(source).resolve()
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        raise ValueError('native source commit differs from the pinned revision')
    result = {}
    for name, expected in files.items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('native file escapes source checkout')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('native file hash differs: ' + name)
        result[name] = raw
    return result


def canonical_bundle(data, expected_candidates=51, training_candidates=100):
    data = Path(data).resolve()
    def validate(row, split, line):
        if not row['history']:
            raise ValueError('native baselines require a nonempty history')
        times = row.get('history_timestamps')
        if not isinstance(times, list) or len(times) != len(row['history']):
            raise ValueError('canonical histories require aligned timestamps')
        now = row.get('timestamp')
        if type(now) is not int or any(type(t) is not int or t >= now for t in times) or times != sorted(times):
            raise ValueError('history must be strictly prior to each query')
        if len(row.get('candidate_ids', [])) != (training_candidates if split == 'train' else expected_candidates):
            raise ValueError('training/evaluation frozen candidate count differs')
    items, rows, metadata, hashes = load_canonical(data, expected_candidates, row_validator=validate)
    manifest_path = data / metadata.get('manifest', 'manifest.json')
    if manifest_path.parent != data or file_digest(manifest_path) != metadata.get('protocol_sha256'):
        raise ValueError('canonical manifest/protocol mismatch')
    manifest = json.loads(manifest_path.read_text())
    for name, actual in hashes.items():
        if name != 'splits.json' and manifest['output_file_sha256'].get(name) != actual:
            raise ValueError('canonical file differs from manifest: ' + name)
    counts = manifest['protocol']['candidates']
    if counts.get('training_count', counts['count']) != training_candidates or counts.get('evaluation_count', counts['count']) != expected_candidates:
        raise ValueError('candidate protocol disagrees with manifest')
    if not isinstance(metadata.get('dataset'), str) or not metadata['dataset']:
        raise ValueError('canonical dataset label missing')
    hashes['manifest.json'] = file_digest(manifest_path)
    return {'items': items, 'rows': rows, 'metadata': metadata, 'hashes': hashes, 'expected_candidates': expected_candidates, 'training_candidates': training_candidates}


def source_identity(extra=()):
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / 'integrations/llmcf').glob('*.py')), *sorted((root / 'integrations/recicl').glob('*.py')), root / 'integrations/resume.py', root / 'adapters/llmsrec_data.py', root / 'evaluate.py', root / 'integrations/traditional/run.py', *[root / name for name in extra]]
    return {str(path.relative_to(root)): file_digest(path) for path in paths}


def record(row, scores):
    if len(scores) != len(row['candidate_ids']):
        raise ValueError('candidate score shape mismatch')
    return {key: row[key] for key in ('query_id', 'user_id', 'target_id', 'candidate_ids')} | {'scores': scores}


def binary_negative(row, seed):
    candidates = [item for item in row['candidate_ids'] if item != row['target_id']]
    if not candidates:
        raise ValueError('no frozen negative candidate is available')
    position = int(digest(['recicl-canonical-negative-v1', seed, row['query_id']]), 16) % len(candidates)
    return candidates[position]


def without_debug_import(raw, path):
    tree = ast.parse(raw, filename=str(path))
    if any(isinstance(node, ast.Name) and node.id == 'torchsnooper' for node in ast.walk(tree)):
        raise ValueError('debug import is used; cannot omit it')
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Import) and any(alias.name == 'torchsnooper' for alias in node.names))]
    return compile(ast.fix_missing_locations(tree), str(path), 'exec')
