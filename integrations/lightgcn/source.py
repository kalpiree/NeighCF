import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import types

COMMIT = '947ca2b3b1d2d3545b114145710cb06c4e57b3d2'
URL = 'https://github.com/gusye1234/LightGCN-PyTorch.git'
FILES = {'code/model.py': '987351a5071065cddc4acda8fdaeda127a5e180a8716cf63132bf76c34dde32e', 'code/utils.py': '829e31d7ad3d5d0654524d7b8d23cea11950d5e4cde5517f8ef9c91d149101a8', 'code/dataloader.py': '093419d07f7a475fb8c410c753b276a6fc7ac1de195f417a6d6c8d66559475e7', 'code/Procedure.py': '60fda65c3adee9d8ffca225c21ff8da5d80ebedc1997673c0ce95417d2035e8b'}


def verified_sources(source):
    source = Path(source).resolve()
    commit = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if commit != COMMIT:
        raise ValueError('LightGCN checkout does not match pinned commit')
    contents = {}
    for name, expected in FILES.items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('source must be a regular file inside the checkout')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('source checksum differs: ' + name)
        contents[name] = data
    return contents


def model_class(source):
    contents = verified_sources(source)
    saved = {name: sys.modules.get(name) for name in ('world', 'dataloader')}
    try:
        world = types.ModuleType('world')
        world.cprint = lambda value: None
        dataloader = types.ModuleType('dataloader')
        dataloader.BasicDataset = object
        sys.modules.update(world=world, dataloader=dataloader)
        module = types.ModuleType('_neighcf_pinned_lightgcn')
        path = Path(source).resolve() / 'code/model.py'
        module.__file__ = str(path)
        exec(compile(contents['code/model.py'], str(path), 'exec'), module.__dict__)
        return module.LightGCN
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def sampler(source, seed):
    import numpy as np
    data = verified_sources(source)['code/utils.py']
    tree = ast.parse(data)
    node, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'UniformSample_original_python']
    rng = np.random.RandomState(seed)
    namespace = {'np': types.SimpleNamespace(random=rng, array=np.array), 'time': time.time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'pinned-native-sampler', 'exec'), namespace)
    return namespace['UniformSample_original_python'], rng
