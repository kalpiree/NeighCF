import ast
import random
import numpy as np
from integrations.llmcf.common import source_bytes

COMMIT = '67706a6dc49c9ca01072d0379e300c9227034330'
FILES = {'minigpt4/models/minigpt4rec_v2.py': 'bbf35f23e99d36fd1174a2ba16d4b200246f44ddf8c9d419cb2cc49f076c71a2'}
METHODS = ('encode_recdata_v2', 'recprompt_wrap_v2', 'prompt_based_encode_v2', 'prompt_with_p', 'forward_v2', 'generate_for_samples_v2')


def native_methods(source):
    import torch
    raw = source_bytes(source, COMMIT, FILES)
    filename = next(iter(FILES))
    tree = ast.parse(raw[filename], filename=filename)
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MiniGPT4Rec_v2')
    functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_ids_order')]
    functions.extend(next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == name) for name in METHODS)
    namespace = {'torch': torch, 'nn': torch.nn, 'np': np, 'random': random}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), filename, 'exec'), namespace)
    return {name: namespace[name] for name in METHODS}
