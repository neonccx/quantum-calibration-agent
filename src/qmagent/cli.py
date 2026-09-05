"""Run a synthetic closed loop. Never connects to SSH, QICK or lab instruments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from . import __version__


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def batch_main(argv=None) -> int:
    import numpy as np
    import scipy
    from .contracts import IQ_THRESHOLDS
    from .policies import RulePolicy
    from .model_worker import ManagedPolicy
    from .settings import Settings
    from .runtime import AgentRunner
    from .backends import make_backend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("rule", "hf"), default="rule")
    parser.add_argument("--decode-backend", choices=("auto", "hf", "cuda_graph"), default="auto")
    parser.add_argument("--prompt-profile", choices=("minimal", "skill"), default="skill",
                        help="Calibration instruction arm: minimal B0/F0 or full skill B1/F1")
    parser.add_argument("--fit-update-tool", action="store_true", help="Enable explicit deterministic fit-update function calls (experimental protocol)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--backend", choices=("physical", "legacy"), default="physical")
    parser.add_argument("--simulation-profile", choices=("nominal", "drift", "quasistatic", "ambiguous"), default="nominal")
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--max-input-tokens", type=int, default=20480)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=180, help="HF request timeout in seconds, including loading")
    parser.add_argument("--trust-remote-code", action="store_true", help="Opt in to executing code from the local model directory")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_dict({key: value for key, value in vars(args).items()
            if key in Settings.__dataclass_fields__} | {
                "model": str(Path(args.model).expanduser().resolve()) if args.model else None,
                "adapter": str(Path(args.adapter).expanduser().resolve()) if args.adapter else None})
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.episodes <= 1000:
        parser.error("episodes must be between 1 and 1000")
    if args.policy == "hf" and not args.model:
        parser.error("--policy hf requires --model (local directory)")
    if args.policy == "rule" and (args.model or args.adapter or args.trust_remote_code):
        parser.error("Model options require --policy hf")
    if args.seed < 0 or not np.isfinite(args.noise_scale) or args.noise_scale < 0:
        parser.error("seed/noise-scale must be nonnegative and noise-scale finite")
    if not 1 <= args.max_steps <= 1000 or not 1 <= args.max_tool_calls <= 100:
        parser.error("Invalid experiment budget")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Output directory is not empty; refusing to overwrite a previous run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    dump(args.output_dir / "run_config.json", config | {
        "version": __version__, "backend": make_backend(settings).backend_name, "synthetic": True,
        "iq_thresholds": IQ_THRESHOLDS, "required_iq_passes": 2, "source_sha256": source_hashes,
        "hf_attention_policy": "sdpa with cudnn disabled; native flash/efficient/math dispatch",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "dataset_relationship": "new online synthetic episodes; not v1.1 held-out test qubits",
    })
    summaries = []
    policy = None
    try:
        policy = RulePolicy() if args.policy == "rule" else ManagedPolicy(settings)
        for episode in range(args.episodes):
            backend = make_backend(settings, args.seed + episode)
            episode_dir = args.output_dir / f"episode_{episode:04d}"
            episode_dir.mkdir()
            with (episode_dir / "trajectory.jsonl").open("x", encoding="utf-8") as handle:
                def log_event(event):
                    handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    if event["event"] == "experiment":
                        obs = event["observation"]
                        print(f"episode={episode} step={event['step']} tool={event['tool']} "
                              f"reliable={obs['quality']['reliable']} iq_passes={event['consecutive_iq_passes']}", flush=True)
                outcome = AgentRunner(backend, policy, args.max_steps, args.max_tool_calls, log_event).run()
            outcome.pop("events")
            outcome.update(episode=episode, seed=args.seed + episode)
            dump(episode_dir / "result.json", outcome)
            summaries.append(outcome)
            print(f"episode={episode} status={outcome['status']} experiments={outcome['experiment_count']}", flush=True)
    except Exception as exc:
        dump(args.output_dir / "error.json", {"error": str(exc), "type": type(exc).__name__})
        raise
    except KeyboardInterrupt:
        dump(args.output_dir / "error.json", {"error": "Interrupted by user", "type": "KeyboardInterrupt"})
        return 130
    finally:
        if policy is not None and hasattr(policy, "close"):
            policy.close()
    statuses = {name: sum(item["status"] == name for item in summaries)
                for name in ("accepted", "escalated", "budget_exhausted", "invalid_action", "tool_error", "policy_error")}
    successful = [item["experiment_count"] for item in summaries if item["status"] == "accepted"]
    aggregate = {"policy": args.policy, "synthetic": True, "episodes": len(summaries), "statuses": statuses,
                 "acceptance_rate": statuses["accepted"] / len(summaries),
                 "mean_experiments_success": float(np.mean(successful)) if successful else None,
                 "episode_results": summaries}
    dump(args.output_dir / "summary.json", aggregate)
    print(json.dumps({key: value for key, value in aggregate.items() if key != "episode_results"}, indent=2))
    # Scientific calibration failures are valid outcomes; software/contract errors are not.
    return 1 if statuses["invalid_action"] or statuses["tool_error"] or statuses["policy_error"] else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        return batch_main(argv[1:])
    if argv and argv[0] == "rpc":
        from .rpc import main as rpc_main
        return rpc_main(argv[1:])
    # Preserve v0.1 automation commands with --output-dir.
    if any(item == "--output-dir" or item.startswith("--output-dir=") for item in argv):
        return batch_main(argv)
    from .terminal import main as terminal_main
    return terminal_main(argv)


if __name__ == "__main__":
    sys.exit(main())
