from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from data import DataError, digest, feature_contract, file_digest, load_bundle, train_item_ids, versions, write_json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resume import RestartManager, runtime_context

def masked_mean(hidden, attention_mask):
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    if (mask.sum(dim=1) == 0).any():
        raise DataError('cannot pool an empty token sequence')
    return (hidden.float() * mask.float()).sum(dim=1) / mask.float().sum(dim=1)

def title_text(title):
    return 'The item title is: ' + title

def token_batch(tokenizer, texts, max_tokens):
    result = tokenizer(texts, padding=True, truncation=False, add_special_tokens=True, return_tensors='pt')
    lengths = result['attention_mask'].sum(dim=1).tolist()
    if not lengths or min(lengths) < 1 or max(lengths) > max_tokens:
        raise DataError(f'item text token length {max(lengths, default=0)} exceeds {max_tokens}; no silent truncation')
    return (result, lengths)

def local_manifest(path):
    if not path or not path.is_dir():
        raise DataError('--model-path must be an existing local Llama-2-7b-hf snapshot')
    config = json.loads((path / 'config.json').read_text())
    if (config.get('model_type'), config.get('hidden_size'), config.get('num_hidden_layers'), config.get('vocab_size')) != ('llama', 4096, 32, 32000):
        raise DataError('local model architecture differs from Llama-2-7B')
    files = sorted((p for p in path.rglob('*') if p.is_file() and p.suffix in {'.safetensors', '.bin', '.json', '.model', '.tiktoken'}))
    if not any((p.suffix in {'.safetensors', '.bin'} for p in files)):
        raise DataError('local model has no weights')
    return {str(p.relative_to(path)): {'bytes': p.stat().st_size, 'sha256': file_digest(p)} for p in files}

def fit_pca(embeddings, fit_rows, seed=42):
    import numpy as np
    from sklearn.decomposition import PCA
    if len(fit_rows) < 64 or embeddings.shape[1] < 64:
        raise DataError('PCA64 requires at least 64 observed training items and embedding dimensions')
    pca = PCA(n_components=64, svd_solver='randomized', random_state=seed)
    pca.fit(np.asarray(embeddings[fit_rows], dtype=np.float32))
    transformed = np.empty((len(embeddings), 64), dtype=np.float32)
    for start in range(0, len(embeddings), 4096):
        transformed[start:start + 4096] = pca.transform(embeddings[start:start + 4096])
    return (transformed, pca)

def atomic_artifact(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.partial-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def atomic_json(path, value):
    atomic_artifact(path, lambda stream: stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()))

class BinaryChunks:

    def __init__(self, root, name, batches, width):
        self.root = Path(root) / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.batches = batches
        self.width = width

    @staticmethod
    def empty():
        return {'next': 0, 'chain': digest([])}

    def array(self, index):
        import numpy as np
        return np.load(self.root / f'batch-{index:08d}.npy', allow_pickle=False)

    def verify(self, progress):
        import numpy as np
        if not isinstance(progress.get('next'), int) or not 0 <= progress['next'] <= len(self.batches):
            raise DataError('invalid feature chunk cursor')
        chain = digest([])
        for index in range(progress['next']):
            receipt_path = self.root / f'batch-{index:08d}.json'
            receipt = json.loads(receipt_path.read_text())
            array_path = self.root / f'batch-{index:08d}.npy'
            if receipt.get('batch') != index or receipt.get('ids_sha256') != digest(self.batches[index]) or receipt.get('array_sha256') != file_digest(array_path):
                raise DataError('feature partial chunk checksum/order differs')
            value = self.array(index)
            if value.shape != (len(self.batches[index]), self.width) or value.dtype != np.float32 or (not np.isfinite(value).all()):
                raise DataError('invalid feature partial chunk values')
            chain = digest([chain, file_digest(receipt_path)])
        if chain != progress.get('chain'):
            raise DataError('feature partial chunk chain differs from durable snapshot')

    def append(self, progress, index, values, extra=None):
        import numpy as np
        if index != progress['next'] or index >= len(self.batches):
            raise DataError('feature batches must append exactly the next original batch')
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(self.batches[index]), self.width) or not np.isfinite(values).all():
            raise DataError('invalid new feature chunk')
        path = self.root / f'batch-{index:08d}.npy'
        atomic_artifact(path, lambda stream: np.save(stream, values, allow_pickle=False))
        receipt_path = self.root / f'batch-{index:08d}.json'
        atomic_json(receipt_path, {'batch': index, 'ids_sha256': digest(self.batches[index]), 'array_sha256': file_digest(path), 'extra': extra})
        return {'next': index + 1, 'chain': digest([progress['chain'], file_digest(receipt_path)])}

    def assemble(self, progress, target):
        import numpy as np
        self.verify(progress)
        if progress['next'] != len(self.batches):
            raise DataError('cannot publish incomplete feature matrix')
        target = Path(target)
        partial = target.with_name('.' + target.name + '.assembling')
        result = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(sum(map(len, self.batches)), self.width))
        start = 0
        for index, ids in enumerate(self.batches):
            result[start:start + len(ids)] = self.array(index)
            start += len(ids)
        result.flush()
        del result
        with partial.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(partial, target)
        fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

def pca_fit_worker(embedding_path, output_path, fit_rows_path, seed, threads):
    import numpy as np
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    embeddings = np.load(embedding_path, mmap_mode='r', allow_pickle=False)
    fit_rows = json.loads(Path(fit_rows_path).read_text())
    if len(fit_rows) < 64 or embeddings.shape[1] < 64:
        raise DataError('PCA64 requires at least64 training rows and dimensions')
    with threadpool_limits(limits=threads):
        pca = PCA(n_components=64, svd_solver='randomized', random_state=seed)
        pca.fit(np.asarray(embeddings[fit_rows], dtype=np.float32))
    atomic_artifact(output_path, lambda stream: np.savez(stream, components=pca.components_, mean=pca.mean_, explained_variance=pca.explained_variance_, fit_rows=np.asarray(fit_rows)))

def run_pca_fit(args, manager, snapshot, fit_rows):
    rows_path = manager.root / 'pca-fit-rows.json'
    atomic_json(rows_path, fit_rows)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    command = [sys.executable, str(Path(__file__).resolve()), '_pca_fit_worker', str(args.output / 'item_embeddings.npy'), str(args.output / 'pca_model.npz'), str(rows_path), str(args.pca_seed), str(args.threads)]
    process = subprocess.Popen(command, env=env)
    try:
        while process.poll() is None:
            if manager.stop_requested:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                snapshot(force=True)
            time.sleep(0.2)
        if process.returncode != 0:
            raise DataError(f'PCA fit child failed with exit {process.returncode}')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-path', type=Path)
    p.add_argument('--native-embeddings', type=Path, help='optional trusted local .npy artifact instead of Llama2 generation')
    p.add_argument('--native-manifest', type=Path, help='required with --native-embeddings; encoder, item_ids, embeddings_sha256')
    p.add_argument('--protocol-sha256')
    p.add_argument('--expected-candidates', type=int, default=51)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=('float16', 'bfloat16', 'float32'), default='float16')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--pca-seed', type=int, default=42)
    p.add_argument('--resume', action='store_true', help='resume only a matching durable feature snapshot')
    p.add_argument('--checkpoint-seconds', type=float, default=600.0)
    return p

def main():
    args = parser().parse_args()
    if args.output.exists() and (not args.resume):
        raise DataError('output already exists; features are immutable without explicit verified resume')
    if args.resume and (not args.output.is_dir()):
        raise DataError('resume requires an existing feature output')
    if args.batch_size < 1 or args.max_tokens < 1 or args.threads < 1:
        raise DataError('batch size, token budget and threads must be positive')
    if bool(args.native_embeddings) != bool(args.native_manifest) or bool(args.model_path) == bool(args.native_embeddings):
        raise DataError('choose --model-path OR both --native-embeddings and --native-manifest')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    bundle = load_bundle(args.data, args.protocol_sha256, args.expected_candidates)
    training_ids = train_item_ids(bundle)
    if len(training_ids) < 64:
        raise DataError('fewer than64 train-observed items')
    fit_rows = [bundle['item_to_index'][i] - 1 for i in training_ids]
    scientific_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in {'resume', 'checkpoint_seconds'}}
    model_checkpoint = local_manifest(args.model_path) if args.model_path else None
    native, native_sha = (None, None)
    width = 4096
    if args.native_embeddings:
        native = json.loads(args.native_manifest.read_text())
        native_sha = file_digest(args.native_embeddings)
        if native.get('item_ids') != bundle['item_ids'] or native.get('embeddings_sha256') != native_sha or (not native.get('encoder')):
            raise DataError('native embedding manifest must pin encoder, exact item order and source SHA256')
        native_array = np.load(args.native_embeddings, mmap_mode='r', allow_pickle=False)
        if native_array.ndim != 2 or len(native_array) != len(bundle['items']) or any((not np.isfinite(native_array[start:start + 4096]).all() for start in range(0, len(native_array), 4096))):
            raise DataError('invalid native embedding matrix')
        width = native_array.shape[1]
    root = Path(__file__).resolve().parents[2]
    contract = {'kind': 'fcrllm-feature-restart-v1', 'feature_contract': feature_contract(bundle, training_ids), 'data_sha256': bundle['source_file_sha256'], 'arguments': scientific_args, 'model_checkpoint': model_checkpoint, 'native_sha256': native_sha, 'native_manifest': native, 'runtime': runtime_context('cpu' if args.native_embeddings else args.device), 'versions': versions(), 'source_sha256': {name: file_digest(root / name) for name in ('integrations/fcrllm/features.py', 'integrations/fcrllm/data.py', 'integrations/resume.py', 'adapters/llmsrec_data.py')}}
    reference = {'frozen_reference_sha256': digest(contract)}
    with RestartManager(args.output, contract, resume=args.resume, interval_seconds=args.checkpoint_seconds) as manager:

        def save_reference(path):
            torch.save(reference, path)

        def restore_reference(path):
            if torch.load(path, map_location='cpu', weights_only=True) != reference:
                raise DataError('frozen encoder reference changed')
        state = manager.restore(restore_reference)
        if state is None:
            state = {'phase': 'encode', 'embedding_chunks': BinaryChunks.empty(), 'pca_chunks': BinaryChunks.empty(), 'token_min': None, 'token_max': None, 'width': width, 'encoder': None}

        def snapshot(force=False):
            return manager.checkpoint(state, save_reference, force=force)
        if not args.resume:
            snapshot(force=True)
        batches = [bundle['item_ids'][start:start + args.batch_size] for start in range(0, len(bundle['items']), args.batch_size)]
        chunks = BinaryChunks(manager.root, 'embedding-chunks', batches, width)
        chunks.verify(state['embedding_chunks'])
        if state['phase'] == 'complete':
            saved = json.loads((args.output / 'manifest.json').read_text())
            if saved.get('status') != 'complete' or digest(saved) != state['manifest_digest'] or any((file_digest(args.output / name) != value for name, value in saved['file_sha256'].items())):
                raise DataError('completed feature artifacts changed or are incomplete')
            print(json.dumps({'status': 'complete', 'items': len(bundle['items']), 'manifest_sha256': file_digest(args.output / 'manifest.json')}), flush=True)
            return
        if state['phase'] == 'encode':
            if args.model_path:
                from transformers import AutoModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
                tokenizer.pad_token, tokenizer.padding_side = (tokenizer.eos_token, 'right')
                model = AutoModel.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True)
                if args.max_tokens > model.config.max_position_embeddings or model.config.hidden_size != width:
                    raise DataError('token limit or hidden size differs from encoder contract')
                model.to(args.device).eval().requires_grad_(False)
                if args.resume:
                    manager.restore(restore_reference)
                state['encoder'] = {'mode': 'controlled_Llama2_7B_title_embedding_adaptation', 'model': 'meta-llama/Llama-2-7b-hf', 'checkpoint_file_sha256': model_checkpoint, 'checkpoint_manifest_sha256': digest(model_checkpoint), 'pooling': 'final hidden-state attention-mask mean, including BOS; float32 accumulation; right padding EOS excluded', 'text_template': 'The item title is: {title}', 'metadata_fields': ['title'], 'dtype': args.dtype, 'max_tokens': args.max_tokens, 'truncation': 'forbidden', 'tokenizer_special_tokens': tokenizer.special_tokens_map, 'limitation': 'native LLM-ESR preprocessing uses text-embedding-ada-002; semantic encoder intentionally replaced'}
            else:
                state['encoder'] = {'mode': 'explicit_native_embedding_artifact', 'source_manifest': native, 'source_manifest_sha256': file_digest(args.native_manifest)}
            snapshot(force=True)
            with torch.inference_mode():
                for index in range(state['embedding_chunks']['next'], len(batches)):
                    ids = batches[index]
                    if args.model_path:
                        tokens, sizes = token_batch(tokenizer, [title_text(bundle['items'][i]['title']) for i in ids], args.max_tokens)
                        tokens = {key: value.to(args.device) for key, value in tokens.items()}
                        hidden = model(**tokens, use_cache=False, return_dict=True).last_hidden_state
                        pooled = masked_mean(hidden, tokens['attention_mask'])
                        values = pooled.cpu().numpy()
                        state['token_min'] = min(sizes) if state['token_min'] is None else min(state['token_min'], min(sizes))
                        state['token_max'] = max(sizes) if state['token_max'] is None else max(state['token_max'], max(sizes))
                        del hidden, pooled, tokens
                    else:
                        start = index * args.batch_size
                        values, sizes = (native_array[start:start + len(ids)], None)
                    state['embedding_chunks'] = chunks.append(state['embedding_chunks'], index, values, sizes)
                    if index == 0 or index % 50 == 0:
                        print(json.dumps({'stage': 'item_embeddings', 'completed': min((index + 1) * args.batch_size, len(bundle['items'])), 'total': len(bundle['items'])}), flush=True)
                    snapshot()
            if args.model_path:
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            state['phase'] = 'assemble'
            snapshot(force=True)
        if state['phase'] == 'assemble':
            chunks.assemble(state['embedding_chunks'], args.output / 'item_embeddings.npy')
            state['embedding_sha256'] = file_digest(args.output / 'item_embeddings.npy')
            state['phase'] = 'pca_fit'
            snapshot(force=True)
        if file_digest(args.output / 'item_embeddings.npy') != state['embedding_sha256']:
            raise DataError('assembled embedding artifact changed')
        if state['phase'] == 'pca_fit':
            snapshot(force=True)
            run_pca_fit(args, manager, snapshot, fit_rows)
            state['pca_model_sha256'] = file_digest(args.output / 'pca_model.npz')
            state['phase'] = 'pca_transform'
            snapshot(force=True)
        if file_digest(args.output / 'pca_model.npz') != state['pca_model_sha256']:
            raise DataError('fitted PCA artifact changed')
        if state['phase'] == 'pca_transform':
            from sklearn.decomposition import PCA
            from threadpoolctl import threadpool_limits
            saved = np.load(args.output / 'pca_model.npz', allow_pickle=False)
            if not np.array_equal(saved['fit_rows'], np.asarray(fit_rows)):
                raise DataError('PCA fit population changed')
            pca = PCA(n_components=64, svd_solver='randomized', random_state=args.pca_seed)
            pca.components_, pca.mean_, pca.explained_variance_ = (saved['components'], saved['mean'], saved['explained_variance'])
            pca.n_features_in_ = width
            embeddings = np.load(args.output / 'item_embeddings.npy', mmap_mode='r', allow_pickle=False)
            projection_batches = [bundle['item_ids'][start:start + 4096] for start in range(0, len(bundle['items']), 4096)]
            projected = BinaryChunks(manager.root, 'pca-chunks', projection_batches, 64)
            projected.verify(state['pca_chunks'])
            with threadpool_limits(limits=args.threads):
                for index in range(state['pca_chunks']['next'], len(projection_batches)):
                    start = index * 4096
                    values = pca.transform(embeddings[start:start + 4096])
                    state['pca_chunks'] = projected.append(state['pca_chunks'], index, values)
                    snapshot()
            projected.assemble(state['pca_chunks'], args.output / 'pca64.npy')
            state['pca64_sha256'] = file_digest(args.output / 'pca64.npy')
            state['phase'] = 'publish'
            snapshot(force=True)
        if file_digest(args.output / 'pca64.npy') != state['pca64_sha256']:
            raise DataError('PCA projection artifact changed')
        names = ('item_embeddings.npy', 'pca64.npy', 'pca_model.npz')
        manifest = {'status': 'complete', 'contract': feature_contract(bundle, training_ids), 'runtime': {'device': args.device, 'threads': args.threads, 'versions': versions()}, 'arguments': scientific_args, 'item_ids': bundle['item_ids'], 'encoder': state['encoder'], 'shape': [len(bundle['items']), width], 'pca_seed': args.pca_seed, 'pca_policy': 'fit only train-observed item metadata; project all closed-catalog items, including cold items', 'pca_fit_item_ids': training_ids, 'source_file_sha256': bundle['source_file_sha256'], 'file_sha256': {name: file_digest(args.output / name) for name in names}, 'integration_source_sha256': {name: file_digest(Path(__file__).parent / name) for name in ('data.py', 'features.py')}, 'token_lengths': {'minimum': state['token_min'], 'maximum': state['token_max']} if state['token_min'] is not None else None}
        atomic_json(args.output / 'manifest.json', manifest)
        state.update(phase='complete', manifest_digest=digest(manifest))
        snapshot(force=True)
        print(json.dumps({'status': 'complete', 'items': len(bundle['items']), 'train_fit_items': len(training_ids), 'manifest_sha256': file_digest(args.output / 'manifest.json')}), flush=True)
if __name__ == '__main__':
    if len(sys.argv) == 7 and sys.argv[1] == '_pca_fit_worker':
        pca_fit_worker(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    else:
        main()
