import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fit_centres(values, clusters, seed, iterations=100, tolerance=1e-6):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all()
            or type(clusters) is not int or not 1 <= clusters <= len(values)
            or type(iterations) is not int or iterations < 1
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Invalid training vectors or clustering settings")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(values)))]
    distances = ((values - values[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < clusters:
        available = np.ones(len(values), dtype=bool)
        available[chosen] = False
        weights = np.where(available, distances, 0.0)
        total = weights.sum()
        next_index = int(rng.choice(len(values), p=weights / total)) if total > 0 else int(np.flatnonzero(available)[0])
        chosen.append(next_index)
        distances = np.minimum(distances, ((values - values[next_index]) ** 2).sum(axis=1))
    centres = values[chosen].copy()
    converged = False
    for step in range(iterations):
        assignments = np.empty(len(values), dtype=np.int64)
        nearest = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), 2048):
            block = values[start:start + 2048]
            squared = ((block[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            assignments[start:start + len(block)] = squared.argmin(axis=1)
            nearest[start:start + len(block)] = squared.min(axis=1)
        updated = np.empty_like(centres)
        for index in range(clusters):
            members = values[assignments == index]
            if len(members):
                updated[index] = members.mean(axis=0)
            else:
                replacement = int(np.argmax(nearest))
                updated[index] = values[replacement]
                nearest[replacement] = -1
        shift = float(np.max(np.linalg.norm(updated - centres, axis=1)))
        centres = updated
        if shift <= tolerance:
            converged = True
            break
    return centres, {"iterations": step + 1, "converged": converged,
                     "maximum_centre_shift": shift}


def prepare(source, train_events, output, clusters, seed, iterations=100):
    source, train_events, output = Path(source), Path(train_events), Path(output)
    if output.exists():
        raise ValueError("Output already exists")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    embeddings_path = source / "item_embeddings.npy"
    expected = manifest.get("file_sha256", {}).get("item_embeddings.npy")
    if not expected or digest_file(embeddings_path) != expected:
        raise ValueError("Item embedding hash differs from the feature manifest")
    item_ids = manifest.get("item_ids")
    if (not isinstance(item_ids, list) or not item_ids or len(set(item_ids)) != len(item_ids)
            or any(not isinstance(item, str) or not item for item in item_ids)):
        raise ValueError("Invalid item identity mapping")
    values = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or values.shape[0] != len(item_ids) or not values.shape[1]:
        raise ValueError("Embedding shape differs from the item mapping")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite item embeddings")
    positions = {item: index for index, item in enumerate(item_ids)}
    training_ids = set()
    with train_events.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "train" or row.get("item_id") not in positions:
                raise ValueError("Cluster fitting requires catalog-matched training events only")
            training_ids.add(row["item_id"])
    training_ids = sorted(training_ids)
    centres, fit = fit_centres(values[[positions[item] for item in training_ids]], clusters, seed, iterations)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".neighcf-features-", dir=output.parent))
    try:
        embedding_output = temporary / "embeddings.json"
        with embedding_output.open("w") as stream:
            stream.write("{")
            for index, item in enumerate(item_ids):
                if index:
                    stream.write(",\n")
                stream.write(json.dumps(item) + ":" + json.dumps(values[index].tolist(), allow_nan=False))
            stream.write("}\n")
        (temporary / "centres.json").write_text(json.dumps(centres.tolist(), allow_nan=False) + "\n")
        receipt = {
            "implementation": "reference_reconstruction", "clustering": "seeded_kmeans_plus_plus_lloyd",
            "clusters": clusters, "seed": seed, "max_iterations": iterations,
            "fit_item_ids": training_ids, "fit_scope": "unique_training_observed_items_only",
            "item_encoder": manifest.get("encoder"), "fit": fit,
            "source_sha256": {"feature_manifest": digest_file(manifest_path),
                              "item_embeddings": expected, "train_events": digest_file(train_events)},
            "output_sha256": {name: digest_file(temporary / name) for name in ("embeddings.json", "centres.json")},
        }
        (temporary / "manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.rename(output)
        return receipt
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcr-features", type=Path, required=True)
    parser.add_argument("--train-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    try:
        receipt = prepare(args.fcr_features, args.train_events, args.output, args.clusters, args.seed, args.iterations)
        print(json.dumps({"clusters": receipt["clusters"], "training_items": len(receipt["fit_item_ids"]), "fit": receipt["fit"]}))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
