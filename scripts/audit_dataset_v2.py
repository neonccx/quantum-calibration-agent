#!/usr/bin/env python3
"""Verify artifacts, device isolation and exact assistant-only native loss boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.protocol import parse_call, policy_messages

def audit(dataset, model=None):
    manifest = json.loads((dataset/"manifest.json").read_text())
    groups, counts, lengths, loss_lengths = {}, {}, [], []
    tokenizer = None
    if model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=False)
    for split in ("train", "validation", "test", "ood"):
        raw = (dataset/(split+".jsonl")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest["split_sha256"][split]
        groups[split], counts[split] = set(), 0
        for line in raw.splitlines():
            row = json.loads(line)
            groups[split].add(row["device_id"])
            counts[split] += 1
            context = json.loads(row["messages"][1]["content"])
            assert policy_messages(context) == row["messages"][:-1], "Online/training prompt mismatch"
            visible = json.dumps(row["messages"])
            assert not any(key in visible for key in ('"_truth"', '"ground_truth"', '"recommended_update"', '"prepared_state": [', '"measurement":'))
            obs = context.get("observation")
            if obs:
                assert (dataset/"artifacts"/(obs["raw_reference"]["sha256"]+".json")).is_file()
            if tokenizer:
                options = dict(tokenize=False, enable_thinking=False, preserve_thinking=False,
                               tools=row["tools"], tool_call_format="json")
                prompt = tokenizer.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, **options)
                full = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, **options)
                a = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                b = tokenizer(full, add_special_tokens=False)["input_ids"]
                assert b[:len(a)] == a and len(b) > len(a), "Assistant loss boundary mismatch"
                answer = tokenizer.decode(b[len(a):], skip_special_tokens=True).strip()
                assert parse_call(answer) == row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
                lengths.append(len(b))
                loss_lengths.append(len(b)-len(a))
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a < b)
    artifacts = list((dataset/"artifacts").glob("*.json"))
    for path in artifacts:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == path.stem, "Artifact content/hash mismatch"
    return {"all_passed": True, "sample_counts": counts, "artifact_count": len(artifacts),
        "device_counts": {k: len(v) for k, v in groups.items()}, "tokenizer_checked": bool(tokenizer),
        "max_tokens": max(lengths) if lengths else None,
        "min_assistant_loss_tokens": min(loss_lengths) if loss_lengths else None,
        "max_assistant_loss_tokens": max(loss_lengths) if loss_lengths else None,
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() if tokenizer else None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
