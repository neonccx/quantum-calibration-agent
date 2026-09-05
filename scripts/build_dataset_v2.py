#!/usr/bin/env python3
"""Execute grouped-device calibration trajectories; export native next-action SFT.

The controller's atomic calibration.step calls acquisition then registered analysis.
This dataset teaches action selection, not imitation of numerical fitting or code execution.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.physical_backend import PhysicalSimulator
from qmagent.policies import RulePolicy
from qmagent.protocol import TOOLS_SCHEMA, PROTOCOL_VERSION, policy_messages, assistant_call
from qmagent.runtime import AgentRunner
from qmagent.storage import digest, encoded
from qmagent.contracts import decision


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, allow_nan=False, indent=2)
        handle.write("\n")


class RecordingBackend(PhysicalSimulator):
    def __init__(self, *args, artifact_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_dir = artifact_dir

    def acquire(self, *args):
        raw = super().acquire(*args)
        # digest uses canonical JSON; store those exact bytes so file SHA matches reference.
        path = self.artifact_dir / (digest(raw)+".json")
        payload = encoded(raw)
        if hashlib.sha256(payload).hexdigest() != digest(raw):
            raise RuntimeError("Canonical artifact encoding mismatch")
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(payload)
        return raw


class Teacher(RulePolicy):
    def __init__(self, narrow=False):
        self.narrow = narrow

    def decide(self, context):
        if not context["budget"]["remaining_experiments"]:
            return decision("ESCALATE_HARDWARE_REVIEW", reason="Experiment budget exhausted; do not claim successful calibration")
        action = super().decide(context)
        if context["observation"] is None and self.narrow:
            action["parameter_action"]["scan"] = {"frequency_span_hz": 4e6}
        name = action["next_tool"]
        if context["budget"]["tool_counts"].get(name, 0) >= context["budget"]["max_calls_per_tool"]:
            return decision("ESCALATE_HARDWARE_REVIEW", reason="Per-experiment retry budget exhausted; preserve failed evidence")
        return action


def build(output, devices=64):
    if devices < 16 or devices % 8:
        raise ValueError("devices must be a multiple of eight, at least sixteen")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output/"artifacts"
    artifacts.mkdir()
    rows = {name: [] for name in ("train", "validation", "test", "ood")}
    episodes, targets, outcomes = [], Counter(), Counter()
    for device_index in range(devices+devices//4):
        split = ("train" if device_index < devices*3//4 else "validation" if device_index < devices*7//8
                 else "test" if device_index < devices else "ood")
        seed = 2026090400+device_index
        device_id = hashlib.sha256(f"device:{seed}".encode()).hexdigest()[:16]
        variants = [("nominal", 1., False), ("nominal", 8., False), ("ambiguous", 1., False), ("nominal", 1., True)]
        if split == "ood":
            variants = [("drift", 1., False), ("quasistatic", 1., False), ("nominal", 30., False)]
        for replica, (profile, noise, narrow) in enumerate(variants):
            episode_id = hashlib.sha256(f"episode:{seed}:{replica}".encode()).hexdigest()[:20]
            backend = RecordingBackend(seed, noise, profile, artifact_dir=artifacts)
            runner = AgentRunner(backend, Teacher(narrow), max_steps=24, max_tool_calls=7)
            transcript, sample_ids = [], []
            while runner.status == "active":
                context = runner.context()
                messages = policy_messages(context)
                action = runner.plan()
                if action is None:
                    break
                answer = assistant_call(action)
                sample_id = f"{episode_id}-{len(sample_ids):02d}"
                runner.step()
                if runner.status in ("invalid_action", "policy_error", "tool_error"):
                    raise RuntimeError(f"Invalid teacher trajectory: {episode_id}: {runner.reason}")
                rows[split].append({"id": sample_id, "task": "next_action", "device_id": device_id,
                    "episode_id": episode_id, "protocol_version": PROTOCOL_VERSION,
                    "tools": TOOLS_SCHEMA, "messages": messages+[answer]})
                sample_ids.append(sample_id)
                targets[action["next_tool"]] += 1
                transcript.extend([{"role": "user", "content": messages[1]["content"]}, answer,
                    {"role": "tool", "name": "calibration.step", "content": json.dumps({
                        "status": runner.status, "context": policy_messages(runner.context())[1]["content"]})}])
            result = runner.result()
            events = result.pop("events")
            write(output/"trajectories"/(episode_id+".json"), {"device_id": device_id, "split": split,
                "messages": transcript, "sample_ids": sample_ids, "result": result})
            # Full measurements are separately content-addressed; audit events retain the executed decisions.
            write(output/"audit"/(episode_id+".json"), events)
            write(output/"evaluator_only"/(episode_id+".json"), {"seed": seed, "profile": profile,
                "noise_scale": noise, "narrow_initial_scan": narrow, "truth": backend._truth})
            outcomes[split+":"+runner.status] += 1
            episodes.append({"episode_id": episode_id, "device_id": device_id, "split": split,
                             "status": runner.status, "samples": len(sample_ids)})
        print(f"device {device_index+1}/{devices+devices//4} {split}", flush=True)
    for split, samples in rows.items():
        with (output/(split+".jsonl")).open("x") as handle:
            for row in samples:
                handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+"\n")
    groups = {split: {row["device_id"] for row in data} for split, data in rows.items()}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a < b)
    manifest = {"schema": PROTOCOL_VERSION, "synthetic": True, "teacher": "observable_only_rule_policy",
        "training_ready": True, "release_scope": "research_sft_candidate_not_hardware_validated",
        "data_origin": "executed reduced physics simulator; not experimental measurements",
        "loss_scope": "one next assistant native call per sample; preceding context fully masked",
        "context_strategy": "same bounded controller state/history as online inference; full trajectories retained for audit",
        "device_counts": {k: len(v) for k, v in groups.items()}, "sample_counts": {k: len(v) for k, v in rows.items()},
        "outcomes": dict(outcomes), "target_counts": dict(targets), "episodes": episodes,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).parents[1]/"src/qmagent").glob("*.py")},
        "split_sha256": {k: hashlib.sha256((output/(k+".jsonl")).read_bytes()).hexdigest() for k in rows},
        "limitations": ["Rule-teacher imitation is not proof of optimal actions", "No leakage or multilevel transmon model",
            "OOD drift/quasistatic/very-high-noise profiles excluded from train", "Independent aliasing and stale-calibration curricula remain future additions"]}
    write(output/"manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("sample_counts", "outcomes", "target_counts")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=64)
    args = parser.parse_args()
    build(args.output, args.devices)
