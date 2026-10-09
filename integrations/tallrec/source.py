import ast
import hashlib
from pathlib import Path
import subprocess

URL = 'https://github.com/SAI990323/TALLRec.git'
COMMIT = 'c1db29ce6501bce32cc8c3e343c4eb14155beeb1'
FILES = {'finetune_rec.py': '83dd50cd5c74549afca068e75786641ce2d692b4ddaac1c74afd5ed51f3a8138', 'evaluate.py': '5f6e865ae7df290a2bb8e2a60341942c094e39bcdc1ccc859bc1b0a905c35652', 'preprocess_movie.py': '6dc5ee303d791a1f6d91758650ba5bfa53536a8753b2bb59400db9495249655a', 'preprocess_book.py': '30feb03f9a7827c1bba0d7581ea29b506a42852e2d7ef4612a32c6ec767c623b', 'requirements.txt': 'f7ad93d6f93c10b8f64484df158fadfcee506c4f9c92aa2336df400559d6fd58', 'LICENSE': 'df825c8e037af50d35015f3dc941f6565a7c28c3bbd8d4b85e5da4b494612024', 'shell/instruct_7B.sh': '85f185a99deb825b50641f5e937eb4dd8faa9edda95a4afde1ec8b2bd72e00a0'}


def verified_sources(source):
    source = Path(source).resolve()
    commit = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if commit != COMMIT:
        raise ValueError('TALLRec checkout does not match pinned commit')
    result = {}
    for name, expected in FILES.items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError('source must remain inside the pinned checkout')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('source checksum differs: ' + name)
        result[name] = data
    return result


def functions(source, filename, names, namespace=None):
    contents = verified_sources(source)
    tree = ast.parse(contents[filename])
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError('required native function missing')
    namespace = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(Path(source) / filename), 'exec'), namespace)
    return namespace


def prompts(source):
    training = functions(source, 'finetune_rec.py', ['generate_prompt'])['generate_prompt']
    evaluation = functions(source, 'evaluate.py', ['generate_prompt'])['generate_prompt']
    return training, evaluation
