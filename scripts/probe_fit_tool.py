"""Bounded real-model regression probe on previously failing diagnostic contexts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.policies import HuggingFacePolicy
from qmagent.contracts import parse_decision
from score_policy_predictions import executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists")
    rows = {r["id"]: r for r in map(json.loads, args.test_file.read_text().splitlines())}
    selected = ["164192409bab01f813be-05", "d81880fddf6ee235689d-09"]
    policy = HuggingFacePolicy(args.model, args.adapter, trust_remote_code=True, decode_backend="hf", fit_update_tool=True)
    records = []
    for identity in selected:
        context = json.loads(rows[identity]["messages"][1]["content"])
        try:
            action = parse_decision(policy.decide(context))
            ok, error = executable(context, action)
            record = dict(id=identity, action=action, controller_executable=ok, error=error,
                          generation=policy.last_generation)
        except (ValueError, RuntimeError) as exc:
            record = dict(id=identity, controller_executable=False, error=str(exc), generation=policy.last_generation)
        records.append(record)
        print(json.dumps(record), flush=True)
    with args.output.open("x") as output:
        json.dump(dict(scope="Known-failure regression probe; not an unbiased test metric", records=records), output, indent=2)


if __name__ == "__main__":
    main()
