#!/usr/bin/env python3
"""Rescore frozen predictions against actual controller prerequisites and scan limits."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.runtime import AgentRunner
from qmagent.contracts import TOOLS, parse_decision

def executable(context, prediction):
    controller = AgentRunner(None, None)
    controller.state = context["state"]
    controller.observation = context["observation"]
    controller.completed = {TOOLS.index(name) for name in context["completed_stages"]}
    controller.passes = context["consecutive_iq_passes"]
    try:
        action = parse_decision(prediction)
        controller._candidate(action)
        name = action["next_tool"]
        budget = context["budget"]
        if name in TOOLS and (budget["remaining_experiments"] <= 0 or
                budget["tool_counts"].get(name, 0) >= budget["max_calls_per_tool"]):
            raise ValueError("Experiment budget exhausted")
        return True, None
    except (KeyError, TypeError, ValueError) as exc:
        return False, str(exc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {r["id"]: r for r in map(json.loads, args.test_file.read_text().splitlines())}
    records = []
    for record in map(json.loads, args.predictions.read_text().splitlines()):
        context = json.loads(rows[record["id"]]["messages"][1]["content"])
        ok, error = executable(context, record.get("prediction"))
        records.append({"id": record["id"], "controller_executable": ok, "rejection": error})
    report = {"scope": "Offline controller validation, NOT measured closed-loop success", "sample_count": len(records),
        "controller_executable_rate": sum(r["controller_executable"] for r in records)/len(records), "records": records}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
