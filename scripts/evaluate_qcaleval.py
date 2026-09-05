#!/usr/bin/env python3
"""Run a frozen QCalEval snapshot against an OpenAI-compatible VLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path

from qmagent.qcaleval import (
    QUESTION_NAMES,
    canonical_json,
    image_content_blocks,
    score_reproducible_subset,
    sha256_bytes,
    validate_snapshot,
)


def request_completion(api_base, api_key, payload, timeout):
    request = urllib.request.Request(
        api_base,
        data=canonical_json(payload),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        status = response.status
    elapsed = time.perf_counter() - started
    decoded = json.loads(raw)
    message = decoded["choices"][0]["message"]
    content = message.get("content")
    if content is None:
        content = message.get("reasoning") or ""
    if isinstance(content, list):
        content = " ".join(block.get("text", "") for block in content if block.get("type") == "text")
    return str(content).strip(), decoded, status, elapsed


def validate_api_base(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API URL must use http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API URL must not contain credentials, query parameters, or fragments")
    if not parsed.path.rstrip("/").endswith("/v1/chat/completions"):
        raise ValueError("API URL must be the exact /v1/chat/completions endpoint")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("unencrypted HTTP is allowed only for a loopback endpoint")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--api-base", required=True, type=validate_api_base, help="Exact OpenAI-compatible /v1/chat/completions URL")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-key-env", default="QCAL_VLM_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write a request plan without network calls")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite existing result: {args.output}")

    snapshot = validate_snapshot(args.snapshot)
    entries = snapshot["entries"][: args.limit] if args.limit else snapshot["entries"]
    if not entries:
        parser.error("limit selected no entries")
    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        parser.error(f"set {args.api_key_env}; secrets are never accepted as command arguments")

    results = []
    for entry in entries:
        image_blocks = image_content_blocks(args.snapshot, entry)
        responses = {}
        for label, name in QUESTION_NAMES.items():
            prompt = entry["prompts"][label]
            payload = {
                "model": args.model_id,
                "messages": [{"role": "user", "content": image_blocks + [{"type": "text", "text": prompt}]}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            }
            if args.no_think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            request_sha = sha256_bytes(canonical_json(payload))
            record = {"question": label, "prompt": prompt, "request_sha256": request_sha}
            if args.dry_run:
                record.update({"answer": None, "error": "dry_run", "latency_seconds": None, "raw_response": None})
            else:
                try:
                    answer, raw, status, latency = request_completion(args.api_base, api_key, payload, args.timeout)
                    record.update({"answer": answer, "error": None, "http_status": status, "latency_seconds": latency, "raw_response": raw})
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError) as exc:
                    record.update({"answer": None, "error": f"{type(exc).__name__}: {exc}", "latency_seconds": None, "raw_response": None})
            responses[name] = record
        scored = None if args.dry_run else score_reproducible_subset(entry, responses)
        results.append({"id": entry["id"], "experiment_type": entry["experiment_type"], "responses": responses, "reproducible_subset": scored})

    aggregates = {}
    if not args.dry_run:
        for label in QUESTION_NAMES:
            values = [row["reproducible_subset"][label]["score"] for row in results]
            values = [value for value in values if value is not None]
            aggregates[label] = {"mean": round(sum(values) / len(values), 1), "count": len(values)} if values else {"mean": None, "count": 0}
    output = {
        "schema_version": "qmagent.qcaleval.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run" if args.dry_run else "zero_shot",
        "model_id": args.model_id,
        "api_base": args.api_base,
        "snapshot_path": str(args.snapshot.resolve()),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "source": snapshot["source"],
        "scoring_scope": "Q2/Q4/Q5/Q6 programmatic subset; not an official QCalEval overall score",
        "aggregate": aggregates,
        "results": results,
    }
    output["run_sha256"] = sha256_bytes(canonical_json(output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(results)} entries, mode={output['mode']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
