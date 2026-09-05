"""Verify streaming preserves token IDs on both supported model decode paths."""
import argparse
from pathlib import Path

from qmagent.policies import HuggingFacePolicy
from qmagent.storage import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    policy = HuggingFacePolicy(args.model, trust_remote_code=True)
    results = []
    for backend in ("cuda_graph", "hf"):
        policy.decode_backend = backend
        chunks = []
        messages = [{"role": "user", "content": "请用中文一句话说明量子比特是什么。"}]
        streamed = policy.chat(messages, {}, 64, on_text=chunks.append)
        stream_stats = dict(policy.last_generation)
        complete = policy.chat(messages, {}, 64)
        stats = dict(policy.last_generation)
        case = {"backend": backend, "streamed": streamed, "complete": complete,
                "chunks": chunks, "stream_stats": stream_stats, "nonstream_stats": stats,
                "same_token_ids": stream_stats["output_token_sha256"] == stats["output_token_sha256"]}
        results.append(case)
        atomic_json(args.output / "results.json", results, replace=True)
        assert case["same_token_ids"] and streamed == complete == "".join(chunks).strip()
        assert len(chunks) > 1 and stream_stats["decode_backend"] == backend
        print(f"PASS {backend}: {len(chunks)} chunks; identical token IDs", flush=True)


if __name__ == "__main__":
    main()
