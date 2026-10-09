import sys
import types
from collections import namedtuple
from .common import source_bytes, without_debug_import

COMMIT = '441e2e969347112462eb28783840e88063d98694'
FILES = {
    'retrieval_task/src_code/models/base.py': '6bc16846bf2d6ace00a0203cc43a209dc21a2b35b54da164c2b2cd2b3a8e3913',
    'retrieval_task/src_code/models/SASREC.py': '897ab0cdbf7d2a84cd1dded8633e9cb85a5aabb1c01e7420511a3381d7fe9484',
    'retrieval_task/src_code/models/DeepFM.py': '1ec35ec6a8941c6bc9a60dea73f1b3413b6c18d796e531071fdb8cc845fbc6a2',
    'retrieval_task/src_code/models/transformer_decoder.py': '53dd241f1f31f8bb4cc07877b05b3aae12c858b9dffa8e902ca8c3c2ee521a80',
    'retrieval_task/src_code/config/beauty/SASREC_aug.yaml': 'dc00b234c7a89e06154c80ba834282726c127910d61e3199862a3116b8c37fdb',
    'LLaMA-Factory/generate/retrieval_ICL_retrieval.py': 'bc91d0070532aef1b883d7a9398a7383a3682c6a17ecd1613d86baaf65752159',
}
SparseFeat = namedtuple('SparseFeat', 'name vocabulary_size embedding_dim')
VarLenSparseFeat = namedtuple('VarLenSparseFeat', 'name sparsefeat maxlen')
DenseFeat = namedtuple('DenseFeat', 'name embedding_dim')


def instantiate(source, config, item_count, user_count, num_neg, device):
    raw = source_bytes(source, COMMIT, FILES)
    package = '_neighcf_pinned_llmcf'
    if any(key == package or key.startswith(package + '.') for key in sys.modules):
        raise RuntimeError('native import namespace already active')
    parent = types.ModuleType(package)
    parent.__path__ = []
    sys.modules[package] = parent
    try:
        for name in ('transformer_decoder', 'base', 'DeepFM', 'SASREC'):
            path = 'retrieval_task/src_code/models/' + name + '.py'
            module = types.ModuleType(package + '.' + name)
            module.__package__ = package
            module.__file__ = str(source / path)
            sys.modules[module.__name__] = module
            exec(without_debug_import(raw[path], module.__file__), module.__dict__)
        inputs = [SparseFeat('user_id', user_count + 1, config['user_id_dim']), VarLenSparseFeat('history_item_id', 'item_id', config['max_len']), SparseFeat('item_id', item_count + 1, config['item_id_dim'])]
        init = {'workspace': '', 'name': 'SASREC_aug', 'device': device, 'patience': 2, 'model_config': config, 'num_neg': num_neg, 'input_data': inputs, 'input_aug_data': inputs + [DenseFeat('cot_id', 768), SparseFeat('label_id', 2, config['aug_label_id_dim'])], 'num_aug': 4}
        return sys.modules[package + '.SASREC'].SASREC_aug(init).to(device)
    finally:
        for name in list(sys.modules):
            if name == package or name.startswith(package + '.'):
                del sys.modules[name]
