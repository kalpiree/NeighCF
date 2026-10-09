from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from prepare_lastfm import LastFMError, SOURCE_PATH, hash_file, prepare
from audit_lastfm import audit

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name('.' + path.name + '.pending')
    with pending.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    sync_dir(path.parent)

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def sync_tree(path):
    for child in Path(path).rglob('*'):
        if child.is_symlink():
            raise LastFMError('symlink in frozen output')
        if child.is_file():
            with child.open('rb') as stream:
                os.fsync(stream.fileno())
    sync_dir(path)

def source_assets(raw, source):
    archive = raw / source['archive_filename']
    partial = archive.with_name(archive.name + '.partial')
    request_path = archive.with_name(archive.name + '.partial.request.json')
    request = {'url': source['archive_url'], 'md5': source['archive_md5']}
    if partial.is_symlink():
        raise LastFMError('partial archive must not be a symlink')
    if not archive.exists():
        if partial.exists() and (not request_path.exists() or json.loads(request_path.read_text()) != request):
            raise LastFMError('partial archive lacks matching source request')
        atomic(request_path, request)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '5', '--retry-max-time', '600', '--connect-timeout', '30', '--speed-time', '120', '--speed-limit', '1024', '--proto', '=https', '--proto-redir', '=https', '--continue-at', '-', '--output', str(partial), source['archive_url']], check=True)
        if partial.is_symlink() or hash_file(partial, 'md5') != source['archive_md5']:
            raise LastFMError('archive MD5 differs from the primary-source published checksum')
        with partial.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(partial, archive)
        sync_dir(raw)
    if archive.is_symlink() or hash_file(archive, 'md5') != source['archive_md5']:
        raise LastFMError('existing archive differs from published MD5')
    events = raw / source['events_filename']
    member_name = None
    if not events.exists():
        pending = events.with_name(events.name + '.partial')
        if pending.is_symlink():
            raise LastFMError('partial event TSV must not be a symlink')
        with tarfile.open(archive, 'r|gz') as packed:
            for member in packed:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts:
                    raise LastFMError('unsafe archive member path')
                if name.name != source['events_filename']:
                    continue
                if member_name is not None or not member.isfile():
                    raise LastFMError('event archive member must be one unique regular file')
                member_name = member.name
                with packed.extractfile(member) as stream, pending.open('wb') as output:
                    shutil.copyfileobj(stream, output, 1 << 20)
                    output.flush()
                    os.fsync(output.fileno())
        if member_name is None or hash_file(pending, 'md5') != source['events_md5']:
            raise LastFMError('event member missing or differs from published MD5')
        os.replace(pending, events)
        sync_dir(raw)
    if events.is_symlink() or hash_file(events, 'md5') != source['events_md5']:
        raise LastFMError('existing event TSV differs from published checksum')
    receipt = {'status': 'verified', 'dataset': source['dataset'], 'source_url': source['record_url'], 'archive_url': source['archive_url'], 'archive_md5': source['archive_md5'], 'archive_sha256': hash_file(archive), 'archive_bytes': archive.stat().st_size, 'events_md5': source['events_md5'], 'events_sha256': hash_file(events), 'events_bytes': events.stat().st_size, 'events_filename': events.name, 'event_member_observed_this_attempt': member_name, 'verified_utc': datetime.now(timezone.utc).isoformat(), 'license': source['license'], 'required_attribution_url': source['required_attribution_url']}
    atomic(raw / 'source.receipt.json', receipt)
    return (events, receipt)

def load_deriver(release):
    path = release / 'data_preparation/derive_candidates.py'
    spec = importlib.util.spec_from_file_location('lastfm_candidate_derivation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def execute(args):
    if not args.root.is_absolute() or not args.release.is_absolute():
        raise LastFMError('--root and --release must be absolute')
    root, release = (args.root.resolve(), args.release.resolve())
    source = json.loads(SOURCE_PATH.read_text())
    suffix = f'-h{args.max_history}' if args.max_history else ''
    if args.minimum_artists != 10:
        suffix += f'-u{args.minimum_artists}'
    name = 'lastfm-1k-celma' + suffix
    raw, state = (root / 'data/raw/lastfm-1k-celma', root / 'preparation' / name)
    parent, derived = (root / 'data/frozen' / name, root / 'frozen' / (name + '-51'))
    if args.plan:
        print(json.dumps({'dataset': source['dataset'], 'source': source, 'minimum_distinct_artists': args.minimum_artists, 'max_history': args.max_history, 'parent': str(parent), 'derived': str(derived), 'network_used': False}, indent=2))
        return
    if args.minimum_artists < 1 or (args.max_history is not None and args.max_history < 1):
        raise LastFMError('history/user limits must be positive')
    raw.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    with (raw / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completion = state / 'completion.json'
        if completion.exists():
            completion.unlink()
        events, receipt = source_assets(raw, source)
        stage = Path(tempfile.mkdtemp(prefix='attempt-', dir=state))
        candidate = parent if parent.exists() else stage / 'parent'
        if candidate != parent:
            summary = prepare(events, candidate, receipt, minimum_artists=args.minimum_artists, max_history=args.max_history)
            atomic(stage / 'prepare.json', summary)
        manifest = json.loads((candidate / 'manifest.json').read_text())
        expected_sources = {name: hash_file(Path(__file__).with_name(name)) for name in ('prepare_lastfm.py', 'lastfm-source.json')}
        if manifest['protocol']['synthetic_fixture'] or manifest['protocol']['history']['max_items'] != args.max_history or manifest['protocol']['eligibility']['minimum_distinct_artists_per_user'] != args.minimum_artists or (manifest['protocol']['candidates']['seed'] != 20260930) or (manifest['preparation_source_sha256'] != expected_sources):
            raise LastFMError('existing parent differs from requested real-data protocol/source')
        checked = audit(candidate, events)
        atomic(stage / 'parent.audit.json', checked)
        if candidate != parent:
            parent.parent.mkdir(parents=True, exist_ok=True)
            sync_tree(candidate)
            os.rename(candidate, parent)
            sync_dir(parent.parent)
        deriver = load_deriver(release)
        candidate = derived if derived.exists() else stage / 'derived'
        if candidate != derived:
            deriver.derive(parent, candidate, expected_candidates=51, seed=42)
        verified = deriver.audit(parent, candidate)
        relation = json.loads((candidate / 'manifest.json').read_text())['candidate_derivation']
        if verified.get('expected_candidates') != 51 or relation.get('seed') != 42 or verified.get('source_protocol_sha256') != checked['protocol_sha256']:
            raise LastFMError('existing candidate derivative differs from required51/seed42/parent')
        atomic(stage / 'candidate51.audit.json', verified)
        if candidate != derived:
            derived.parent.mkdir(parents=True, exist_ok=True)
            sync_tree(candidate)
            os.rename(candidate, derived)
            sync_dir(derived.parent)
        result = {'status': 'complete', 'dataset': 'LastFM-1K-Celma', 'original_manuscript_cohort': False, 'source_receipt_path': str(raw / 'source.receipt.json'), 'source_receipt_sha256': hash_file(raw / 'source.receipt.json'), 'parent_protocol_sha256': checked['protocol_sha256'], 'protocol_sha256': verified['protocol_sha256'], 'expected_candidates': 51, 'candidate_seed': 42, 'raw_source_verified': True, 'statistics': checked, 'frozen_path': str(derived), 'parent_path': str(parent), 'parent_audit_path': str(stage / 'parent.audit.json'), 'parent_audit_sha256': hash_file(stage / 'parent.audit.json'), 'derivation_audit_path': str(stage / 'candidate51.audit.json'), 'derivation_audit_sha256': hash_file(stage / 'candidate51.audit.json'), 'implementation_sha256': {name: hash_file(Path(__file__).with_name(name)) for name in ('run_lastfm.py', 'prepare_lastfm.py', 'audit_lastfm.py', 'lastfm-source.json')}, 'candidate_deriver_sha256': hash_file(release / 'data_preparation/derive_candidates.py'), 'source_url': source['record_url'], 'license': source['license'], 'required_attribution_url': source['required_attribution_url']}
        atomic(completion, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--minimum-artists', type=int, default=10)
    p.add_argument('--max-history', type=int, default=None, help='explicitly labeled variant only; default preserves full history')
    p.add_argument('--plan', action='store_true')
    return p
if __name__ == '__main__':
    try:
        execute(parser().parse_args())
    except (LastFMError, OSError, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        print('LastFM preparation failed: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
