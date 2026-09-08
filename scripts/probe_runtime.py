#!/usr/bin/env python3
"""Load a local checkpoint and run one bounded inference compatibility probe."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qmagent.policies import HuggingFacePolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    policy = HuggingFacePolicy(args.model, max_new_tokens=args.max_new_tokens,
                               trust_remote_code=True, decode_backend="hf")
    response = policy._generate([
        {"role": "system", "content": "Reply briefly in English."},
        {"role": "user", "content": "Say ready."},
    ], args.max_new_tokens)
    print(json.dumps({"text": response, "generation": policy.last_generation}, indent=2))


if __name__ == "__main__":
    main()
