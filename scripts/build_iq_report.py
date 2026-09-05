"""Re-analyze an existing synthetic batch episode without re-running experiments."""

import argparse
import json
from pathlib import Path

from qmagent.iq_report import from_trajectory, gates_equivalent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.episode_dir.parent / "run_config.json").read_text())
    result = json.loads((args.episode_dir / "result.json").read_text())
    if config.get("synthetic") is not True:
        raise ValueError("This exporter currently supports recorded synthetic episodes only")
    if args.output.exists():
        raise ValueError("Choose a new output directory; existing reports are preserved")
    report = from_trajectory(args.episode_dir / "trajectory.jsonl", args.output,
                             policy="hf + LoRA" if config.get("adapter") else config["policy"], status=result["status"],
                             consecutive_passes=result["consecutive_iq_passes"])
    if report is None:
        print("No recorded IQ experiment; no IQ image generated.")
        return
    if not gates_equivalent(report["gate"], result["final_iq_gate"]):
        raise ValueError("Recomputed IQ gate differs from original result; do not use as acceptance evidence")
    print(args.output / "iq_report.png")


if __name__ == "__main__":
    main()
