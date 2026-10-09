import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.neighcf.config import Config
from integrations.neighcf.prompting import build_prompt
from integrations.neighcf.retrieval import NeighCFRetriever


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def measure(retriever, queries, config, variant, device="cpu", scorer=None, warmup=1):
    if type(warmup) is not int or warmup < 0 or not queries:
        raise ValueError("Nonempty queries and a nonnegative warmup count are required")
    if len({query["query_id"] for query in queries}) != len(queries):
        raise ValueError("Duplicate benchmark query IDs")
    scope = "causal_retrieval_and_prompt" if scorer is None else "causal_retrieval_prompt_and_candidate_likelihood"

    def one(query):
        synchronize(device)
        start = time.perf_counter()
        found = retriever.retrieve(query, variant=variant,
                                   soft_context=config.mode == "soft_embedding_gate" and variant in {"full", "global_pool"})
        synchronize(device)
        retrieved = time.perf_counter()
        prompt = build_prompt(query, retriever.items, found, config)
        synchronize(device)
        prompted = time.perf_counter()
        row = {"query_id": query["query_id"], "measurement_scope": scope,
               "implementation_variant": f"{config.mode}:{config.index_policy}:{variant}",
               "device": str(device), "retrieval_seconds": retrieved - start,
               "prompt_seconds": prompted - retrieved, "eligible_users": found["eligible_users"],
               "pool_users": found["pool_users"], "neighbors": len(found["neighbors"]),
               "literal_index_is_full_pool": found["literal_index_is_full_pool"],
               "candidate_count": len(query["candidate_ids"]),
               "source_identity": retriever.source_identity}
        if scorer is not None:
            scorer.score(prompt)
            synchronize(device)
            row["inference_seconds"] = time.perf_counter() - prompted
            row["prompt_tokens"] = scorer.last_measurements["prompt_tokens"]
            row["candidate_forward_passes"] = scorer.last_measurements["candidate_forward_passes"]
        return row

    for index in range(warmup):
        one(queries[index % len(queries)])
    return [one(query) for query in queries]


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "items", "embeddings", "centres", "events", "queries", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant", choices=("full", "global_pool", "fixed_neighbors"), default="full")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-variant", choices=("full", "global_pool", "fixed_neighbors"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    try:
        from integrations.neighcf.training import preflight_output
        protected = [args.config, args.items, args.embeddings, args.centres, args.events, args.queries]
        protected.extend(path for path in (args.model_path, args.checkpoint) if path is not None)
        preflight_output(args.output.parent, protected)
        if args.output.is_symlink():
            raise ValueError("Output cannot be a symlink")
        if args.output.exists():
            raise ValueError("Output already exists")
        if bool(args.model_path) != bool(args.checkpoint):
            raise ValueError("Inference timing requires both --model-path and --checkpoint")
        if not args.model_path and args.device != "cpu":
            raise ValueError("Retrieval-only timing uses CPU; provide a model for GPU timing")
        cfg = Config.from_dict(json.loads(args.config.read_text()))
        items = json.loads(args.items.read_text())
        retriever = NeighCFRetriever(items, json.loads(args.embeddings.read_text()),
                                    json.loads(args.centres.read_text()), read_rows(args.events), cfg)
        queries = read_rows(args.queries)
        scorer = None
        checkpoint_variant = args.checkpoint_variant or args.variant
        if args.model_path:
            from integrations.neighcf.scoring import CandidateLikelihoodScorer, load_local_model
            from integrations.neighcf.training import restore_for_scoring
            model, tokenizer = load_local_model(args.model_path, args.device)
            gate, checkpoint = restore_for_scoring(args.checkpoint, cfg, retriever, model, tokenizer, args.device, checkpoint_variant)
            if args.variant == "fixed_neighbors":
                gate = None
            scorer = CandidateLikelihoodScorer(model, tokenizer, cfg, args.device, gate)
        rows = measure(retriever, queries, cfg, args.variant, args.device, scorer, args.warmup)
        config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
        with args.output.open("x") as stream:
            for row in rows:
                row.update(config_sha256=config_sha, warmup_queries=args.warmup,
                           checkpoint_variant=checkpoint_variant if args.checkpoint else None,
                           excludes=["model_loading", "feature_preparation", "initial_membership_computation"],
                           includes=["causal_neighbor_profile_construction", "pool_filtering", "similarity", "prompt_rendering"])
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
