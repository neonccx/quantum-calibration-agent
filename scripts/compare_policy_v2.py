"""Compare complete paired evaluations without treating missing records as successes."""
import argparse
import hashlib
import json
from pathlib import Path


def read_evaluation(directory):
    config = json.loads((directory / "config.json").read_text())
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    ids = config["selected_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicated selected IDs")
    if len(rows) != len(ids) or {row["id"] for row in rows} != set(ids):
        raise ValueError("Evaluation is incomplete or contains duplicate/unexpected predictions")
    scores = json.loads((directory / "controller_score.json").read_text())["records"]
    if len(scores) != len(ids) or {row["id"] for row in scores} != set(ids):
        raise ValueError("Controller scores are incomplete or duplicated")
    indexed = {row["id"]: dict(row) for row in rows}
    for score in scores:
        indexed[score["id"]]["controller_executable"] = score["controller_executable"]
    for row in indexed.values():
        for key in ("valid", "next_tool_correct", "arguments_correct", "controller_executable"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"Non-boolean evaluation flag: {key}")
    return config, indexed


def compare(baseline, adapted):
    before, a = read_evaluation(baseline)
    after, b = read_evaluation(adapted)
    for key in ("protocol", "model", "test_sha256", "tokenizer_sha256", "selected_ids"):
        if before.get(key) != after.get(key) or key not in before:
            raise ValueError(f"Unmatched comparison input: {key}")
    if before.get("adapter") is not None or not after.get("adapter"):
        raise ValueError("Expected an unmodified baseline and an explicit adapted checkpoint")
    ids = before["selected_ids"]
    for identity in ids:
        if a[identity]["expected"] != b[identity]["expected"]:
            raise ValueError("Reference answers differ")
    metrics = {}
    for key in ("valid", "next_tool_correct", "arguments_correct", "controller_executable"):
        improved = [i for i in ids if not a[i][key] and b[i][key]]
        regressed = [i for i in ids if a[i][key] and not b[i][key]]
        metrics[key] = dict(baseline=sum(a[i][key] for i in ids)/len(ids),
                            adapted=sum(b[i][key] for i in ids)/len(ids),
                            delta_percentage_points=100*(len(improved)-len(regressed))/len(ids),
                            improved_ids=improved, regressed_ids=regressed)
    by_action = {}
    for action in sorted({a[i]["expected"]["next_tool"] for i in ids}):
        group = [i for i in ids if a[i]["expected"]["next_tool"] == action]
        by_action[action] = dict(count=len(group),
            baseline_correct=sum(a[i]["next_tool_correct"] for i in group),
            adapted_correct=sum(b[i]["next_tool_correct"] for i in group))
    return dict(scope="Paired frozen-context diagnostic; does not measure closed-loop acceptance",
                sample_count=len(ids), adapter=after["adapter"], metrics=metrics, by_action=by_action,
                test_sha256=before["test_sha256"],
                input_sha256={f"{arm}/{name}": hashlib.sha256((directory/name).read_bytes()).hexdigest()
                    for arm, directory in (("baseline", baseline), ("adapted", adapted))
                    for name in ("config.json", "predictions.jsonl", "controller_score.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.adapted)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("scope", "sample_count", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
