"""Audit new function-call targets using the installed tokenizer without loading weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.protocol import parse_call, policy_messages


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit output exists")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True, use_fast=False)
    manifest = json.loads((args.dataset/"manifest.json").read_text())
    count, maximum, minimum_target = 0, 0, 10**9
    for split in ("train", "validation", "test", "ood"):
        raw = (args.dataset/(split+".jsonl")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["split_sha256"][split]:
            raise ValueError("Changed split")
        for line in raw.splitlines():
            row = json.loads(line)
            context = json.loads(row["messages"][1]["content"])
            if row["messages"][:-1] != policy_messages(context, fit_update_tool=True):
                raise ValueError("Runtime and training prompts differ")
            options = dict(tools=row["tools"], tool_call_format="json", enable_thinking=False, preserve_thinking=False)
            prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True, **options)
            full = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False, **options)
            if full[:len(prompt)] != prompt or len(full) <= len(prompt):
                raise ValueError("Invalid assistant loss boundary")
            suffix = tokenizer.decode(full[len(prompt):], skip_special_tokens=True).strip()
            parsed = parse_call(suffix, context, allow_fit_tool=True)
            target = row["messages"][-1]["tool_calls"][0]["function"]
            if parsed != parse_call("<tool_call>"+json.dumps(target)+"</tool_call>", context, True):
                raise ValueError("Decoded target changed controller action")
            count += 1
            maximum = max(maximum, len(full))
            minimum_target = min(minimum_target, len(full)-len(prompt))
        print(f"Audited {split}; total={count}", flush=True)
    report = dict(all_passed=True, tokenizer_checked=True, samples=count, max_tokens=maximum,
                  minimum_target_tokens=minimum_target, protocol=manifest["protocol"],
                  tokenizer_sha256=hashlib.sha256((Path(args.model)/"tokenizer_config.json").read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256((args.dataset/"manifest.json").read_bytes()).hexdigest())
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
