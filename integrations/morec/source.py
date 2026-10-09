import importlib.util
from pathlib import Path
import sys
from integrations.resume import file_digest

URL = "https://github.com/westlake-repl/IDvs.MoRec.git"
COMMIT = "ce372cfc523c52458fa0267c9834d1be488e026a"
FILES = {'bce_text/main-end2end/model/__init__.py': 'a95e432efdbaa10c5534739a3378029e8ca295003f707c0aed736cbf25aa80f2', 'bce_text/main-end2end/model/model.py': 'dbd24c3847c10174142f6a62fd12ca2a7820c16a32462f17512bf4666cfd9dad', 'bce_text/main-end2end/model/encoders.py': '66a9af8a45fa322cb2156b7c5533270db66c01b2341341146d6e1473505ec9f7', 'bce_text/main-end2end/model/modules.py': 'd1488ed68bcc7a65ca80e30ef7d78592b8c84d36c7ce30be182a85cc2da49a42', 'bce_text/main-end2end/data_utils/dataset.py': '375e3eb16943c7ae6d3ee08ecbcc1644a3fface46f4c79ae9518a226b52929b0'}


def verify(source):
    source = Path(source)
    for name, expected in FILES.items():
        if file_digest(source / name) != expected:
            raise ValueError("MoRec native source SHA mismatch: " + name)
    return source


def model_class(source):
    source = verify(source)
    name = "_neighcf_morec_native_" + COMMIT
    if name not in sys.modules:
        directory = source / "bce_text/main-end2end/model"
        spec = importlib.util.spec_from_file_location(name, directory / "__init__.py", submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].Model
