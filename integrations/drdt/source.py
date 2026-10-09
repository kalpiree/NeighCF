import ast
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace


COMMIT = '928eee5d3571022a3da661243cc73d715625b8b8'
FILES = {'src/model/llm_gpt.py': '10a85b5362088094467a3bf2bc041aeb0b6679e78fc4d25a825a9b75dbff449d'}


def load_methods(source, rng, artist_adaptation=False):
    root = Path(source)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('DRDT source must be a real checkout directory')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise ValueError('DRDT checkout revision differs from pinned source')
    path = root / 'src/model/llm_gpt.py'
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError('DRDT source symlinks are not accepted')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != FILES['src/model/llm_gpt.py']:
        raise ValueError('DRDT native source hash mismatch')
    tree = ast.parse(data)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LLM_GPT']
    if len(classes) != 1:
        raise ValueError('DRDT native class is missing or duplicated')
    names = {'inference_COT_reflex_long', 'get_batch_examples'}
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise ValueError('DRDT native methods are missing')
    if artist_adaptation:
        class ArtistPrompt(ast.NodeTransformer):
            def visit_JoinedStr(self, node):
                for value in node.values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        for before, after in [('purchased', 'listened to'), ('purchase', 'listen to'), ('Books', 'artists'),
                                              ('author style', 'musical style'), ('price, publish date', 'genre, release date')]:
                            value.value = value.value.replace(before, after)
                return node

            def visit_Assign(self, node):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'products' and isinstance(node.value, ast.Name) and node.value.id == 'dataset_name':
                    node.value = ast.copy_location(ast.Constant(value='artists'), node.value)
                return self.generic_visit(node)

        methods = [ast.fix_missing_locations(ArtistPrompt().visit(node)) if node.name == 'inference_COT_reflex_long' else node for node in methods]
    namespace = {'np': SimpleNamespace(random=rng)}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}
