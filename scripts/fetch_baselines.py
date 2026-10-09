from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile


class SourceError(ValueError):
    pass


def git(directory, *arguments):
    result = subprocess.run(["git", "-C", str(directory), *arguments], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def relative_path(value):
    if not isinstance(value, str):
        raise SourceError("source path must be a string")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise SourceError("invalid relative source path")
    return path


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != 1 or not isinstance(manifest.get("sources"), list):
        raise SourceError("unsupported upstream manifest")
    names, destinations = set(), set()
    for entry in manifest["sources"]:
        name, destination = entry["name"], entry["destination"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names or destination in destinations:
            raise SourceError("duplicate or malformed upstream name")
        if str(relative_path(destination)) != name:
            raise SourceError("destination must equal its source name")
        if not re.fullmatch(r"[0-9a-f]{40}", entry["commit"]):
            raise SourceError("upstream commit must be a full immutable hash")
        if not re.fullmatch(r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", entry["url"]):
            raise SourceError("upstream URL must be an explicit GitHub HTTPS repository")
        if not entry.get("required_files"):
            raise SourceError("missing required upstream source hashes")
        for relative, digest in entry["required_files"].items():
            relative_path(relative)
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise SourceError("invalid upstream file hash")
        for license in entry["licenses"]:
            relative_path(license["path"])
            relative_path(license["bundled_copy"])
            if not re.fullmatch(r"[0-9a-f]{64}", license["sha256"]):
                raise SourceError("invalid upstream license hash")
        names.add(name)
        destinations.add(destination)
    return manifest


def verify(entry, directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise SourceError("source directory is missing or is a symlink")
    if Path(git(directory, "rev-parse", "--show-toplevel")).resolve() != directory.resolve():
        raise SourceError("source is not an independent repository")
    if git(directory, "rev-parse", "HEAD") != entry["commit"]:
        raise SourceError("upstream commit differs from its pin")
    if git(directory, "status", "--porcelain", "--untracked-files=all"):
        raise SourceError("upstream checkout is modified or contains untracked files")
    hashes = dict(entry["required_files"])
    hashes.update({value["path"]: value["sha256"] for value in entry["licenses"]})
    for relative, expected in hashes.items():
        target = directory / relative
        if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(directory.resolve()):
            raise SourceError("required source file is missing or escapes its checkout")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise SourceError("upstream file differs from its pin: " + relative)
    return {"name": entry["name"], "commit": entry["commit"], "directory": str(directory.resolve()), "status": "verified"}


def fetch(entry, parent):
    parent = Path(parent).resolve()
    destination = parent / entry["destination"]
    if destination.exists() or destination.is_symlink():
        return verify(entry, destination)
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + entry["name"] + "-", dir=parent))
    try:
        git(temporary, "init", "--quiet")
        git(temporary, "remote", "add", "origin", entry["url"])
        git(temporary, "fetch", "--quiet", "--depth=1", "origin", entry["commit"])
        git(temporary, "checkout", "--quiet", "--detach", entry["commit"])
        verify(entry, temporary)
        if destination.exists() or destination.is_symlink():
            raise SourceError("destination appeared while fetching")
        os.rename(temporary, destination)
        return verify(entry, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parser():
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description="Inspect, verify or explicitly fetch the pinned baseline sources without model weights or datasets.")
    result.add_argument("--manifest", type=Path, default=root / "third_party/UPSTREAM.json")
    result.add_argument("--directory", type=Path, default=root / "third_party")
    result.add_argument("--only", nargs="+", default=[])
    group = result.add_mutually_exclusive_group()
    group.add_argument("--fetch", action="store_true")
    group.add_argument("--verify", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    known = {entry["name"] for entry in manifest["sources"]}
    if set(args.only) - known:
        raise SourceError("unknown requested baseline")
    entries = [entry for entry in manifest["sources"] if not args.only or entry["name"] in args.only]
    results = []
    for entry in entries:
        if args.fetch:
            results.append(fetch(entry, args.directory))
        elif args.verify:
            results.append(verify(entry, args.directory / entry["destination"]))
        else:
            results.append({"name": entry["name"], "url": entry["url"], "commit": entry["commit"], "destination": str(args.directory / entry["destination"]), "license_status": entry["license_status"], "status": "plan_only"})
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
