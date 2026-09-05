"""One-shot ordered experiment: wait for the frozen evaluation, then probe fit calls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-status", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"pid": os.getpid(), "status": "waiting_for_frozen_evaluation"}
    def save():
        temporary = args.output/"status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(args.output/"status.json")
    save()
    root = Path(__file__).resolve().parents[1]
    try:
        deadline = time.monotonic()+48*3600
        while True:
            prior = json.loads(args.evaluation_status.read_text())
            if prior["status"] != "running":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Frozen evaluation did not finish within 48 hours")
            try:
                os.kill(prior["pid"], 0)
            except ProcessLookupError:
                raise RuntimeError("Frozen evaluation stopped without a final status")
            time.sleep(30)
        state["status"] = "probing"
        save()
        with (args.output/"probe.log").open("x") as log:
            subprocess.run(["bash", args.launcher, str(root/"scripts/probe_fit_tool.py"),
                "--model", args.model, "--adapter", args.adapter, "--test-file", args.test_file,
                "--output", str(args.output/"probe.json")], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=1800)
        probe = json.loads((args.output/"probe.json").read_text())
        if not all(r["controller_executable"] and "calibration.step_from_fit" in (r.get("generation") or {}).get("native_call", "") for r in probe["records"]):
            raise RuntimeError("Model did not execute the explicit fit tool correctly in every regression case")
        state["status"] = "closed_loop"
        save()
        with (args.output/"closed_loop.log").open("x") as log:
            result = subprocess.run(["bash", args.launcher, "-m", "qmagent.cli", "run", "--policy", "hf",
                "--model", args.model, "--adapter", args.adapter, "--trust-remote-code", "--fit-update-tool",
                "--decode-backend", "hf", "--request-timeout", "300", "--backend", "physical",
                "--seed", "2026090500", "--episodes", "3", "--output-dir", str(args.output/"closed_loop")],
                cwd=root, env=dict(os.environ, PYTHONPATH=str(root/"src")), stdout=log, stderr=subprocess.STDOUT,
                timeout=7200)
        state.update(status="completed" if result.returncode == 0 else "completed_with_failures", returncode=result.returncode)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
