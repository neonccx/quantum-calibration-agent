#!/usr/bin/env python3
"""Frozen-context native tool-call evaluation, distinct from closed-loop success."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.policies import HuggingFacePolicy
from qmagent.protocol import PROTOCOL_VERSION, TOOLS_SCHEMA, parse_call, policy_messages


def select(rows, limit):
    if not limit or limit >= len(rows):
        return rows
    # Round-robin across target action classes, in frozen file order.
    groups = {}
    for row in rows:
        name = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["next_tool"]
        groups.setdefault(name, []).append(row)
    picked = []
    while len(picked) < limit:
        for key in sorted(groups):
            if groups[key] and len(picked) < limit:
                picked.append(groups[key].pop(0))
    return picked


def argument_match(actual, expected):
    if actual["next_tool"] != expected["next_tool"]:
        return False
    for field in ("updates", "scan"):
        a, b = actual["parameter_action"][field], expected["parameter_action"][field]
        if set(a) != set(b):
            return False
        for key in a:
            tolerance = 1000. if key.endswith("_hz") else .01 if key.endswith("_us") else 1e-4
            if abs(a[key]-b[key]) > tolerance:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--trust-remote-code", action="store_true", help="Explicitly trust the already installed model implementation")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-profile", choices=("minimal", "skill"), default="skill")
    parser.add_argument("--resume", action="store_true",
                        help="Append to a verified prediction prefix after interruption")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    raw = args.test_file.read_bytes()
    selected = select([json.loads(line) for line in raw.splitlines()], args.limit)
    if not selected:
        raise ValueError("Empty evaluation")
    first_context = json.loads(selected[0]["messages"][1]["content"])
    system_prompt = policy_messages(first_context, prompt_profile=args.prompt_profile)[0]["content"]
    config = {"protocol": PROTOCOL_VERSION, "model": args.model, "adapter": args.adapter,
        "test_sha256": hashlib.sha256(raw).hexdigest(), "selected_ids": [row["id"] for row in selected],
        "scope": "frozen-context imitation metrics; not closed-loop or hardware success",
        "sampling": "all rows" if not args.limit else "round-robin action-stratified subset",
        "batch_size": args.batch_size,
        "prompt_profile": args.prompt_profile,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((Path(args.model)/"tokenizer_config.json").read_bytes()).hexdigest()}
    records = []
    if args.resume:
        config_path = args.output / "config.json"
        predictions_path = args.output / "predictions.jsonl"
        if not config_path.is_file() or not predictions_path.is_file():
            raise ValueError("--resume requires config.json and predictions.jsonl")
        recorded_config = json.loads(config_path.read_text())
        immutable_fields = ("protocol", "model", "adapter", "test_sha256", "selected_ids",
                            "batch_size", "prompt_profile", "system_prompt_sha256",
                            "tokenizer_sha256")
        mismatches = [field for field in immutable_fields
                      if recorded_config.get(field) != config.get(field)]
        if mismatches:
            raise ValueError("Resume config mismatch: " + ", ".join(mismatches))
        records = [json.loads(line) for line in predictions_path.read_text().splitlines()]
        expected_prefix = config["selected_ids"][:len(records)]
        if [record.get("id") for record in records] != expected_prefix:
            raise ValueError("Existing predictions are not the selected-ID prefix")
        if len(records) >= len(selected):
            raise ValueError("Evaluation already has all selected predictions")
        with (args.output / "resume_history.jsonl").open("a") as history:
            history.write(json.dumps({
                "resumed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "completed_prefix": len(records),
                "remaining": len(selected) - len(records),
            }) + "\n")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"config.json").write_text(json.dumps(config, indent=2))
    policy = HuggingFacePolicy(args.model, args.adapter, max_new_tokens=512, decode_backend="hf",
                              trust_remote_code=args.trust_remote_code, prompt_profile=args.prompt_profile)
    generation_seconds = sum(record.get("seconds", 0.0) for record in records)
    stream_mode = "a" if args.resume else "x"
    with (args.output/"predictions.jsonl").open(stream_mode) as stream:
        for offset in range(len(records), len(selected), args.batch_size):
            batch = selected[offset:offset+args.batch_size]
            contexts = [json.loads(row["messages"][1]["content"]) for row in batch]
            message_batches = [policy_messages(context, prompt_profile=args.prompt_profile)
                               for context in contexts]
            # Contexts are already compacted: reconstruction must be idempotent.
            before = time.monotonic()
            raw_outputs, generation_details = policy._generate_batch(
                message_batches, policy.max_new_tokens, tools=TOOLS_SCHEMA)
            batch_seconds = time.monotonic()-before
            generation_seconds += batch_seconds
            for row, context, raw, details in zip(batch, contexts, raw_outputs, generation_details):
                expected = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
                try:
                    actual = parse_call(raw, context=context)
                    record = {"id": row["id"], "valid": True, "prediction": actual,
                        "next_tool_correct": actual["next_tool"] == expected["next_tool"],
                        "arguments_correct": argument_match(actual, expected)}
                except (ValueError, RuntimeError) as exc:
                    record = {"id": row["id"], "valid": False, "error": str(exc),
                              "next_tool_correct": False, "arguments_correct": False}
                record.update(expected=expected, seconds=details["amortized_seconds"],
                              generation=details)
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                print(f"{len(records)}/{len(selected)} valid={record['valid']} next={record['next_tool_correct']}", flush=True)
    metrics = {"sample_count": len(records), "scope": config["scope"],
        **{key+"_rate": sum(record[key] for record in records)/len(records)
           for key in ("valid", "next_tool_correct", "arguments_correct")},
        "mean_seconds": sum(r["seconds"] for r in records)/len(records),
        "wall_seconds": generation_seconds,
        "samples_per_second": len(records)/generation_seconds,
        "batch_size": args.batch_size,
        "target_counts": dict(Counter(r["expected"]["next_tool"] for r in records)),
        "dataset_sha256": config["test_sha256"], "adapter": args.adapter}
    (args.output/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
