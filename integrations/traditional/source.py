import hashlib
from pathlib import Path
import subprocess
import sys
import types

COMMIT = 'c80bdc0f1a889b78fe9a27e2058389cc84caea84'
URL = 'https://github.com/yehjin-shin/BSARec.git'
FILES = {
    'src/model/_modules.py': '3cd495cb69f1cee1a4dcee6b0918e4bbf5852fd6ceff9ac796a0fece9ec1a59c',
    'src/model/_abstract_model.py': 'bf8e4fec891aa264c8c9403cb5f8af62d731f39e5426850b2f5179beb0076be6',
    'src/model/gru4rec.py': 'dfa5c2e55deaaf5e805b2436b7049e8ee92e4dfe735b3896591c501bf26162cb',
    'src/model/caser.py': '5cfdbeb315883e60c41a3aefccfff0b5c5cf941220359d95f58cfbb7aae03add',
    'src/model/sasrec.py': '65fef2068014192ef9e04efada4797b91852c52fc5ea0c1fa03b7feed19b1cee',
}
CLASSES = {'gru4rec': 'GRU4RecModel', 'caser': 'CaserModel', 'sasrec': 'SASRecModel'}


def verified_sources(source):
    source = Path(source).resolve()
    commit = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != COMMIT:
        raise ValueError('BSARec checkout does not match the pinned commit')
    contents = {}
    for name, expected in FILES.items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('native source must be a regular file inside its checkout')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('native source checksum differs: ' + name)
        contents[name] = data
    return contents


def model_class(source, model):
    if model not in CLASSES:
        raise ValueError('unknown traditional model')
    contents = verified_sources(source)
    saved = {name: module for name, module in sys.modules.items()
             if name == 'model' or name.startswith('model.')}
    for name in saved:
        del sys.modules[name]
    try:
        package = types.ModuleType('model')
        package.__path__ = []
        sys.modules['model'] = package
        for stem in ('_modules', '_abstract_model', model):
            name = 'model.' + stem
            path = Path(source).resolve() / 'src/model' / (stem + '.py')
            module = types.ModuleType(name)
            module.__file__ = str(path)
            module.__package__ = 'model'
            sys.modules[name] = module
            exec(compile(contents['src/model/' + stem + '.py'], str(path), 'exec'), module.__dict__)
        return getattr(module, CLASSES[model])
    finally:
        for name in list(sys.modules):
            if name == 'model' or name.startswith('model.'):
                del sys.modules[name]
        sys.modules.update(saved)
