import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def check(dependencies=False, gpu=False):
    errors = []
    upstream = json.loads((ROOT / 'third_party/UPSTREAM.json').read_text())
    excluded = {'.venv', '__pycache__', '.git'} | {entry['destination'] for entry in upstream['sources']}
    sources = [path for path in ROOT.rglob("*.py")
               if not any(part in excluded for part in path.relative_to(ROOT).parts)]
    for path in sources:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            errors.append(str(error))
    configurations = list((ROOT / "configs").glob("*.json"))
    for path in configurations:
        try:
            json.loads(path.read_text())
        except ValueError as error:
            errors.append(f"{path.name}: {error}")
    result = {"python": sys.version.split()[0], "source_files": len(sources),
              "config_files": len(configurations), "source_syntax_valid": not errors,
              "dependencies_checked": dependencies, "gpu_checked": gpu}
    if dependencies:
        result["dependencies"] = {}
        for name in ("torch", "numpy", "transformers", "peft", "accelerate", "sentencepiece", "scikit-learn", "sentence-transformers"):
            module = {"scikit-learn": "sklearn", "sentence-transformers": "sentence_transformers"}.get(name, name)
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing dependency: {name}")
                continue
            try:
                completed = subprocess.run([sys.executable, "-B", "-c", f"import {module}"],
                                           text=True, capture_output=True, timeout=30)
                imported = completed.returncode == 0
            except subprocess.TimeoutExpired:
                imported = False
            result["dependencies"][name] = {"version": version, "imported": imported}
            if not imported:
                errors.append(f"Cannot import {name}; check interpreter architecture and installed binary wheels")
    if gpu:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if not result["cuda_available"]:
            errors.append("No CUDA device available")
        else:
            result["cuda_version"] = torch.version.cuda
            result["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    result["errors"] = errors
    result["ok"] = not errors
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependencies", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    result = check(args.dependencies, args.gpu)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
