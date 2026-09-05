"""Wait for ordered jobs, then run the preregistered B0/F0 minimal-prompt ablation."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ACTIVE = {"running", "waiting_for_frozen_evaluation", "probing", "closed_loop"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-status", type=Path, required=True)
    parser.add_argument("--fit-status", type=Path, required=True)
    parser.add_argument("--skill-results", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026090500)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    state = {
        "pid": os.getpid(),
        "status": "waiting_for_ordered_jobs",
        "prompt_profile": "minimal",
        "comparison_contract": "same checkpoint/adapter, native tool schema, public context, controller, simulator and seeds; system instruction only changes",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "src" / "qmagent").glob("*.py"))
        },
        "stages": [],
    }

    def save():
        temporary = args.output / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    def run(name, command, allowed_failure=False, cwd=None):
        item = {"name": name, "command": list(map(str, command)), "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state["stages"].append(item)
        state["status"] = "running"
        state["current_stage"] = name
        save()
        with (args.output / f"{name}.log").open("x") as log:
            code = subprocess.run(list(map(str, command)), cwd=cwd or root, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        item.update(returncode=code, finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        if code and not allowed_failure:
            raise RuntimeError(f"stage {name} failed: {code}")

    save()
    try:
        deadline = time.monotonic() + 72 * 3600
        for dependency in (args.evaluation_status, args.fit_status):
            while True:
                dependency_state = json.loads(dependency.read_text())
                if dependency_state.get("status") not in ACTIVE:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("ordered dependency did not finish within 72 hours")
                pid = dependency_state.get("pid")
                if pid:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError as exc:
                        raise RuntimeError(f"dependency stopped without terminal status: {dependency}") from exc
                time.sleep(30)

        launcher = ["bash", args.launcher]
        environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1")
        for split in ("test", "ood"):
            for arm in ("base", "sft"):
                output = args.output / f"{split}_{arm}"
                extra = ["--adapter", args.adapter] if arm == "sft" else []
                run(f"{split}_{arm}", launcher + [root / "scripts/evaluate_policy_v2.py",
                    "--model", args.model, "--trust-remote-code", "--prompt-profile", "minimal",
                    "--test-file", args.dataset / f"{split}.jsonl", "--output", output] + extra)
                run(f"{split}_{arm}_controller", launcher + [root / "scripts/score_policy_predictions.py",
                    "--test-file", args.dataset / f"{split}.jsonl", "--predictions", output / "predictions.jsonl",
                    "--output", output / "controller_score.json"])
            run(f"{split}_base_prompt_comparison", launcher + [root / "scripts/compare_policy_v2.py",
                "--baseline", args.skill_results / f"{split}_base", "--adapted", args.output / f"{split}_base",
                "--output", args.output / f"{split}_base_prompt_comparison.json"])
            run(f"{split}_sft_prompt_comparison", launcher + [root / "scripts/compare_policy_v2.py",
                "--baseline", args.skill_results / f"{split}_sft", "--adapted", args.output / f"{split}_sft",
                "--output", args.output / f"{split}_sft_prompt_comparison.json"])
            run(f"{split}_minimal_training_comparison", launcher + [root / "scripts/compare_policy_v2.py",
                "--baseline", args.output / f"{split}_base", "--adapted", args.output / f"{split}_sft",
                "--output", args.output / f"{split}_minimal_training_comparison.json"])

        for arm in ("base", "sft"):
            extra = ["--adapter", args.adapter] if arm == "sft" else []
            run(f"fresh_{arm}", launcher + ["-m", "qmagent.cli", "run", "--policy", "hf",
                "--decode-backend", "hf", "--prompt-profile", "minimal", "--model", args.model,
                "--trust-remote-code", "--backend", "physical", "--seed", args.seed,
                "--episodes", args.episodes, "--request-timeout", "300", "--max-steps", "30",
                "--max-tool-calls", "8", "--output-dir", args.output / f"fresh_{arm}"] + extra,
                allowed_failure=True, cwd=root)
        state["status"] = "completed_with_failures" if any(item["returncode"] for item in state["stages"]) else "completed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
