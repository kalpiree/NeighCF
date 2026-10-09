import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_records
from integrations.neighcf.config import Config
from integrations.neighcf.prompting import build_prompt
from integrations.neighcf.retrieval import NeighCFRetriever


def main():
    import torch
    from integrations.neighcf.soft_gate import SoftNeighborEmbeddingGate
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "integrations/neighcf/example.config.json").read_text())
    cfg = Config.from_dict(raw)
    items = {str(index): {"title": f"Item {index}"} for index in range(60)}
    embeddings = {item: [int(item) % 5 / 5, int(item) % 7 / 7] for item in items}
    events = [dict(user_id=user, item_id=str(index), timestamp=1000 * (index + 1), split="train")
              for user in ("neighbor-a", "neighbor-b") for index in range(3)]
    query = {"query_id": "synthetic-query", "user_id": "target-user", "history": ["0", "1"],
             "history_timestamps": [1000, 2000], "timestamp": 10000,
             "candidate_ids": [str(index) for index in range(5, 56)], "target_id": "5"}
    retriever = NeighCFRetriever(items, embeddings, [[0., 0.], [1., 1.]], events, cfg)
    found = retriever.retrieve(query, soft_context=True)
    assert len(found["neighbors"]) == 2 and found["literal_index_is_full_pool"]
    prompt = build_prompt(query, items, found, cfg)
    assert tuple(query["candidate_ids"]) == prompt.candidate_ids
    gate = SoftNeighborEmbeddingGate(cfg)
    weights = gate(prompt.neighbor_scores, prompt.user_id, prompt.history_length)
    weights.sum().backward()
    assert torch.isfinite(weights).all() and torch.isfinite(gate.parameters_raw.grad).all()
    assert gate.parameters_raw.grad.abs().sum() > 0
    scores = [float(-index) for index in range(51)]
    prediction = {key: query[key] for key in ("query_id", "user_id", "candidate_ids", "target_id")}
    result, ranks = evaluate_records([prediction | {"scores": scores}],
                                    {"model": "synthetic", "dataset": "synthetic", "seed": 42,
                                     "protocol_sha256": "0" * 64}, 51)
    assert ranks[0]["target_rank"] == 1 and result["metrics"]["MRR"] == 1.0
    print(json.dumps({"status": "passed", "scope": "synthetic CPU setup check", "queries": 1,
                      "candidates": 51, "neighbors": 2, "artifacts_written": False}))


if __name__ == "__main__":
    main()
