"""Re-serialize existing executed trajectories with explicit fit application targets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.parameter_tools import calculate_fit_updates
from qmagent.protocol import TOOLS_SCHEMA, FIT_TOOL_SCHEMA, policy_messages, parse_call


def transform(row):
    context = json.loads(row["messages"][1]["content"])
    action = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    call = {"name": "calibration.step", "arguments": action}
    try:
        calculated = calculate_fit_updates(context["observation"], context["state"])["updates"]
    except (ValueError, KeyError, TypeError):
        calculated = None
    if calculated and action["parameter_action"]["updates"] == calculated and action["next_tool"].startswith("sq."):
        call = {"name": "calibration.step_from_fit", "arguments": {
            "next_tool": action["next_tool"], "scan": action["parameter_action"]["scan"], "reason": action["reason"]}}
    # Every transformed target must resolve to exactly the original executed action.
    resolved = parse_call("<tool_call>"+json.dumps(call)+"</tool_call>", context, allow_fit_tool=True)
    if resolved != action:
        raise ValueError("Transformation changed the recorded controller action")
    result = dict(row)
    result["messages"] = policy_messages(context, fit_update_tool=True)+[
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]}]
    result["tools"] = TOOLS_SCHEMA+[FIT_TOOL_SCHEMA]
    return result, call["name"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source/"manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"protocol": "calibration-step-fit-0.1", "training_ready": False,
              "pending": ["installed-tokenizer loss-boundary audit", "model selection after protocol probe"],
              "scope": "Format revision of existing executed actions; not new independent experiments",
              "source_manifest_sha256": hashlib.sha256((args.source/"manifest.json").read_bytes()).hexdigest(),
              "split_sha256": {}, "counts": {}}
    for split in ("train", "validation", "test", "ood"):
        source = args.source/(split+".jsonl")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["split_sha256"][split]:
            raise ValueError("Frozen source split changed")
        counts = Counter()
        target = args.output/(split+".jsonl")
        with target.open("x") as output:
            for line in raw.splitlines():
                row, kind = transform(json.loads(line))
                output.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+"\n")
                counts[kind] += 1
        report["counts"][split] = dict(counts)
        report["split_sha256"][split] = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
