from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import tempfile
import time
import uuid

class RestartError(RuntimeError):
    pass

class CheckpointExit(SystemExit):

    def __init__(self):
        super().__init__(75)

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(_json(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)

def _atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as target, Path(source).open('rb') as stream:
            shutil.copyfileobj(stream, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
        _sync_dir(destination.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)

def runtime_context(device='cpu'):
    import numpy as np
    import torch
    result = {'python': list(sys.version_info[:3]), 'numpy': np.__version__, 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda, 'device_type': torch.device(device).type, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if torch.device(device).type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        result['gpu'] = {'name': props.name, 'capability': [props.major, props.minor], 'memory': props.total_memory, 'cudnn': torch.backends.cudnn.version()}
    return result

def _rng_state():
    import numpy as np
    import torch
    name, keys, position, gaussian, cached = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [name, keys.tolist(), position, gaussian, cached], 'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def _restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value['python'])
    name, keys, position, gaussian, cached = value['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(value['torch_cpu'])
    if value['torch_cuda']:
        if not torch.cuda.is_available() or len(value['torch_cuda']) != torch.cuda.device_count():
            raise RestartError('CUDA RNG device count changed')
        torch.cuda.set_rng_state_all(value['torch_cuda'])

def _relative(name):
    path = Path(name)
    if path.is_absolute() or not path.parts or any((part in ('..', '.restart') for part in path.parts)):
        raise RestartError('artifact name must be a safe relative output path')
    return path

class RestartManager:

    def __init__(self, output, contract, *, resume=False, interval_seconds=600, install_signals=True):
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError('checkpoint interval must be finite and nonnegative')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.root = self.output / '.restart'
        self.root.mkdir(exist_ok=True)
        self.contract = json.loads(_json(contract))
        self.contract_sha256 = digest(self.contract)
        self.interval_seconds = interval_seconds
        self.stop_requested = False
        self.last_save_time = time.monotonic()
        self._handlers = {}
        self._ready_token = uuid.uuid4().hex
        self._fd = os.open(self.root / 'lock', os.O_CREAT | os.O_RDWR, 384)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._fd)
            self._fd = None
            raise RestartError('another worker already owns this output directory') from exc
        try:
            identity = self.root / 'identity.json'
            if identity.exists():
                saved = json.loads(identity.read_text())
                if saved != {'version': 1, 'contract': self.contract, 'sha256': self.contract_sha256}:
                    raise RestartError('experiment contract changed; refusing resume')
                if not resume:
                    raise RestartError('existing restart state requires explicit resume')
            elif resume:
                raise RestartError('resume requested without an initialized experiment')
            else:
                _atomic_json(identity, {'version': 1, 'contract': self.contract, 'sha256': self.contract_sha256})
            self._resume = resume
            self._latest = self._read_latest() if resume else None
            if resume and self._latest is None:
                raise RestartError('resume requested without a durable snapshot')
            with contextlib.suppress(FileNotFoundError):
                (self.root / 'continuation.json').unlink()
            if install_signals:
                for sig in (signal.SIGUSR1, signal.SIGTERM):
                    self._handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, self._request_stop)
                _atomic_json(self.root / 'worker-ready.json', {'pid': os.getpid(), 'contract_sha256': self.contract_sha256, 'instance': self._ready_token})
        except BaseException:
            self.close()
            raise

    def _request_stop(self, signum, frame):
        self.stop_requested = True

    def close(self):
        ready = self.root / 'worker-ready.json'
        if ready.is_file():
            with contextlib.suppress(OSError, ValueError):
                receipt = json.loads(ready.read_text())
                if receipt.get('pid') == os.getpid() and receipt.get('instance') == self._ready_token:
                    ready.unlink()
        for sig, handler in self._handlers.items():
            signal.signal(sig, handler)
        self._handlers = {}
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _assert_open(self):
        if self._fd is None:
            raise RestartError('restart manager is closed')

    def _read_latest(self):
        pointer = self.root / 'latest.json'
        if not pointer.exists():
            return None
        latest = json.loads(pointer.read_text())
        name = latest.get('generation', '')
        if not name.startswith('generation-') or Path(name).name != name:
            raise RestartError('invalid checkpoint generation pointer')
        directory = self.root / name
        manifest_path = directory / 'manifest.json'
        if file_digest(manifest_path) != latest.get('manifest_sha256'):
            raise RestartError('checkpoint manifest checksum mismatch')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('contract_sha256') != self.contract_sha256:
            raise RestartError('checkpoint belongs to a different experiment')
        for key, expected in manifest['files'].items():
            relative = _relative(key)
            path = directory / relative
            if path.is_symlink() or file_digest(path) != expected:
                raise RestartError('checkpoint file checksum mismatch: ' + key)
        if not {'model.pt', 'state.pt', 'progress.json'} <= manifest['files'].keys():
            raise RestartError('checkpoint lacks required state files')
        return {'directory': directory, 'manifest': manifest, 'pointer': latest}

    def save(self, progress, save_model, optimizer=None, scheduler=None, artifacts=None):
        import torch
        self._assert_open()
        _json(progress)
        generation = 'generation-' + uuid.uuid4().hex
        temporary = self.root / ('.pending-' + uuid.uuid4().hex)
        temporary.mkdir(mode=448)
        try:
            save_model(temporary / 'model.pt')
            state = {'optimizer': optimizer.state_dict() if optimizer is not None else None, 'scheduler': scheduler.state_dict() if scheduler is not None else None, 'rng': _rng_state()}
            torch.save(state, temporary / 'state.pt')
            (temporary / 'progress.json').write_text(_json(progress) + '\n')
            names = []
            for name, source in (artifacts or {}).items():
                relative = _relative(name)
                target = temporary / 'artifacts' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not Path(source).is_file() or Path(source).is_symlink():
                    raise RestartError('checkpoint artifact must be an existing regular file: ' + str(source))
                shutil.copyfile(source, target)
                names.append(str(relative))
            paths = [path for path in temporary.rglob('*') if path.is_file()]
            files = {}
            for path in paths:
                if path.is_symlink():
                    raise RestartError('checkpoint file cannot be a symlink')
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
                files[str(path.relative_to(temporary))] = file_digest(path)
            manifest = {'version': 1, 'contract_sha256': self.contract_sha256, 'files': files, 'artifacts': names, 'created_unix': time.time(), 'optimizer': optimizer is not None, 'scheduler': scheduler is not None}
            _atomic_json(temporary / 'manifest.json', manifest)
            for directory in sorted((p for p in temporary.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                _sync_dir(directory)
            _sync_dir(temporary)
            completed = self.root / generation
            os.replace(temporary, completed)
            _sync_dir(self.root)
            pointer = {'version': 1, 'generation': generation, 'manifest_sha256': file_digest(completed / 'manifest.json')}
            _atomic_json(self.root / 'latest.json', pointer)
            self._latest = self._read_latest()
            self.last_save_time = time.monotonic()
            generations = sorted(self.root.glob('generation-*'), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in generations[2:]:
                if old != completed:
                    shutil.rmtree(old)
            return pointer
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def restore(self, restore_model, optimizer=None, scheduler=None):
        import torch
        self._assert_open()
        if not self._resume:
            return None
        latest = self._read_latest()
        manifest, directory = (latest['manifest'], latest['directory'])
        if manifest['optimizer'] != (optimizer is not None) or manifest['scheduler'] != (scheduler is not None):
            raise RestartError('optimizer/scheduler presence differs from snapshot')
        state = torch.load(directory / 'state.pt', map_location='cpu', weights_only=True)
        progress = json.loads((directory / 'progress.json').read_text())
        restore_model(directory / 'model.pt')
        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        for name in manifest['artifacts']:
            relative = _relative(name)
            _atomic_copy(directory / 'artifacts' / relative, self.output / relative)
        _restore_rng(state['rng'])
        self.last_save_time = time.monotonic()
        return progress

    def peek_progress(self):
        self._assert_open()
        latest = self._read_latest()
        if latest is None:
            return None
        return json.loads((latest['directory'] / 'progress.json').read_text())

    def checkpoint(self, progress, save_model, optimizer=None, scheduler=None, artifacts=None, force=False):
        self._assert_open()
        due = force or self.stop_requested or time.monotonic() - self.last_save_time >= self.interval_seconds
        if not due:
            return False
        pointer = self.save(progress, save_model, optimizer, scheduler, artifacts)
        if self.stop_requested:
            self._read_latest()
            _atomic_json(self.root / 'continuation.json', {'status': 'checkpointed', 'exit_code': 75, 'contract_sha256': self.contract_sha256, **pointer})
            raise CheckpointExit()
        return True

    def scoring(self, stage, batches):
        self._assert_open()
        return ScoreJournal(self, stage, batches)

class ScoreJournal:

    def __init__(self, manager, stage, batches):
        if not isinstance(stage, str) or not stage:
            raise ValueError('scoring stage must be a nonempty stable identifier')
        self.manager = manager
        self.batches = [[str(query) for query in batch] for batch in batches]
        flat = [query for batch in self.batches for query in batch]
        if not flat or any((not batch for batch in self.batches)) or len(flat) != len(set(flat)):
            raise RestartError('evaluation batches must contain unique nonempty query IDs')
        self.root = manager.root / 'scores' / digest(stage)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = {'version': 1, 'stage': stage, 'contract_sha256': manager.contract_sha256, 'batches_sha256': digest(self.batches), 'batch_count': len(self.batches)}
        path = self.root / 'index.json'
        if path.exists():
            self.index = json.loads(path.read_text())
            if self.index.get('identity') != self.identity:
                raise RestartError("evaluation stage's original query batching changed")
        else:
            self.index = {'identity': self.identity, 'completed_batches': 0, 'chain_sha256': digest([])}
            _atomic_json(path, self.index)
        self._validate()

    def _validate(self):
        count = self.index.get('completed_batches')
        if not isinstance(count, int) or not 0 <= count <= len(self.batches):
            raise RestartError('evaluation journal contains invalid batch count')
        chain = digest([])
        for i in range(count):
            path = self.root / f'batch-{i:08d}.json'
            chunk = json.loads(path.read_text())
            if not isinstance(chunk, dict) or chunk.get('batch') != i or chunk.get('records_sha256') != digest(chunk.get('records')):
                raise RestartError('evaluation chunk missing, corrupted or out of order')
            self._check_records(i, chunk['records'])
            chain = digest([chain, file_digest(path)])
        if chain != self.index.get('chain_sha256'):
            raise RestartError('evaluation hash chain is corrupted')

    def _check_records(self, batch_index, records):
        if [str(row.get('query_id')) for row in records] != self.batches[batch_index]:
            raise RestartError('evaluation predictions differ from original query order')
        _json(records)

    @property
    def completed_batches(self):
        return self.index['completed_batches']

    def append(self, batch_index, records):
        self.manager._assert_open()
        if batch_index != self.completed_batches or batch_index >= len(self.batches):
            raise RestartError('evaluation chunks must append exactly the next complete batch')
        self._check_records(batch_index, records)
        path = self.root / f'batch-{batch_index:08d}.json'
        _atomic_json(path, {'batch': batch_index, 'records': records, 'records_sha256': digest(records)})
        updated = {'identity': self.identity, 'completed_batches': batch_index + 1, 'chain_sha256': digest([self.index['chain_sha256'], file_digest(path)])}
        _atomic_json(self.root / 'index.json', updated)
        self.index = updated

    def records(self, *, require_complete=True):
        self._validate()
        if require_complete and self.completed_batches != len(self.batches):
            raise RestartError('evaluation is incomplete; no final result can be returned')
        return [record for i in range(self.completed_batches) for record in json.loads((self.root / f'batch-{i:08d}.json').read_text())['records']]
