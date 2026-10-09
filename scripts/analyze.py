import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import metrics_from_rank


def read_rows(path):
    path = Path(path)
    with path.open() as stream:
        if path.suffix == ".json":
            rows = json.load(stream)
            if not isinstance(rows, list):
                raise ValueError("Expected a JSON array of records")
            yield from rows
        else:
            for line in stream:
                if not line.strip():
                    raise ValueError("Blank JSONL record")
                yield json.loads(line)


def unique_rows(rows, label):
    result = {}
    for row in rows:
        key = row.get("query_id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"{label}: missing or duplicate query_id")
        result[key] = row
    if not result:
        raise ValueError(f"{label}: no records")
    return result


def activity(ranks, queries, threshold, field="history_length_before_truncation"):
    if type(threshold) is not int or threshold < 1:
        raise ValueError("Activity threshold must be a positive integer")
    ranks = unique_rows(ranks, "ranks")
    queries = unique_rows(queries, "queries")
    if ranks.keys() != queries.keys():
        raise ValueError("Ranks must cover the exact query set")
    groups = {"cold": [], "warm": []}
    for identity, row in ranks.items():
        query = queries[identity]
        count = query.get(field)
        if type(count) is not int or count < 0:
            raise ValueError(f"{identity}: missing or invalid activity field {field}")
        if row.get("user_id") != query.get("user_id") or row.get("target_id") != query.get("target_id"):
            raise ValueError(f"{identity}: rank/query identity mismatch")
        candidates = query.get("candidate_ids")
        rank = row.get("target_rank")
        if (not isinstance(candidates, list) or len(candidates) != len(set(candidates))
                or query["target_id"] not in candidates
                or row.get("candidate_count") != len(candidates)
                or type(rank) is not int or not 1 <= rank <= len(candidates)):
            raise ValueError(f"{identity}: invalid candidate/rank contract")
        groups["cold" if count < threshold else "warm"].append(row)
    result = {"activity_threshold": threshold, "activity_field": field,
              "aggregation": "query_macro", "groups": {}}
    for name, rows in groups.items():
        values = [metrics_from_rank(row["target_rank"]) for row in rows]
        result["groups"][name] = {
            "queries": len(rows), "users": len({row["user_id"] for row in rows}),
            "metrics": {key: math.fsum(v[key] for v in values) / len(values)
                        for key in values[0]} if values else None,
        }
    return result


def timing(rows):
    rows = list(rows)
    unique_rows(rows, "timing records")
    signatures = {(row.get("measurement_scope"), row.get("implementation_variant"),
                   row.get("device")) for row in rows}
    if len(signatures) != 1 or any(not isinstance(v, str) or not v for v in next(iter(signatures))):
        raise ValueError("Timing records must share one explicit scope, variant and device")
    result = dict(zip(("measurement_scope", "implementation_variant", "device"), next(iter(signatures))))
    result["queries"] = len(rows)
    for key in ("retrieval_seconds", "prompt_seconds", "inference_seconds"):
        if all(key not in row for row in rows):
            continue
        values = [row.get(key) for row in rows]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid or inconsistent {key}")
        ordered = sorted(values)
        position = (len(ordered) - 1) * 0.95
        low, high = math.floor(position), math.ceil(position)
        p95 = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        result[key] = {"mean": statistics.mean(values), "median": statistics.median(values),
                       "p95": p95, "total": math.fsum(values)}
    if "retrieval_seconds" not in result:
        raise ValueError("Missing retrieval timing")
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("activity")
    p.add_argument("--ranks", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--threshold", type=int, required=True)
    p.add_argument("--activity-field", default="history_length_before_truncation")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("timing")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "activity":
            result = activity(read_rows(args.ranks), read_rows(args.queries), args.threshold, args.activity_field)
        else:
            result = timing(read_rows(args.records))
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
