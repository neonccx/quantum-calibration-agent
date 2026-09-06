#!/usr/bin/env python3
"""Build an immutable v3 next-action dataset with ZPA2D and XEB-style stages."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.contracts import TOOLS, decision
from qmagent.physical_backend import PhysicalSimulator
from qmagent.policies import RulePolicy
from qmagent.protocol import TOOLS_SCHEMA, PROTOCOL_VERSION, assistant_call, policy_messages
from qmagent.runtime import AgentRunner
from qmagent.storage import digest, encoded


def write_json(path, value):
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
        payload = encoded(raw)
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != digest(raw):
            raise RuntimeError("Canonical acquisition digest mismatch")
        path = self.artifact_dir/(checksum+".json")
        if not path.exists():
            path.write_bytes(payload)
        return raw


class V3Teacher(RulePolicy):
    """Observable-only controller teacher; never reads simulator truth."""
    def __init__(self, narrow=False):
        self.narrow = narrow

    def decide(self, context):
        if not context["budget"]["remaining_experiments"]:
            return decision("ESCALATE_HARDWARE_REVIEW", reason="Experiment budget exhausted; preserve the failed evidence")
        action = super().decide(context)
        if context["observation"] is None and self.narrow:
            action["parameter_action"]["scan"] = {"frequency_span_hz": 4e6}
        if action["next_tool"] == "sq.s21_zpa2d" and not action["parameter_action"]["scan"]:
            action["parameter_action"]["scan"] = {"zpa_points": 21, "frequency_points": 161}
        if action["next_tool"] == "sq.xeb" and not action["parameter_action"]["scan"]:
            action["parameter_action"]["scan"] = {"max_depth": 96, "depth_points": 12,
                                                    "circuits_per_depth": 64}
        name = action["next_tool"]
        if name in TOOLS and context["budget"]["tool_counts"].get(name, 0) >= context["budget"]["max_calls_per_tool"]:
            return decision("ESCALATE_HARDWARE_REVIEW", reason="Per-tool retry budget exhausted; preserve the failed evidence")
        return action


def build(output, devices=32):
    if devices < 16 or devices % 8:
        raise ValueError("devices must be a multiple of eight and at least sixteen")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output/"artifacts"
    artifacts.mkdir()
    rows = {name: [] for name in ("train", "validation", "test", "ood")}
    episodes, targets, outcomes = [], Counter(), Counter()
    total_devices = devices+devices//4
    for device_index in range(total_devices):
        split = ("train" if device_index < devices*3//4 else
                 "validation" if device_index < devices*7//8 else
                 "test" if device_index < devices else "ood")
        seed = 2026090600+device_index
        device_id = hashlib.sha256(f"v3-device:{seed}".encode()).hexdigest()[:16]
        variants = [("nominal", 1., False), ("nominal", 8., False),
                    ("ambiguous", 1., False), ("nominal", 1., True)]
        if split == "ood":
            variants = [("drift", 1., False), ("quasistatic", 1., False),
                        ("nominal", 30., False), ("flux_edge", 1., False),
                        ("low_xeb", .5, False)]
        for replica, (profile, noise, narrow) in enumerate(variants):
            episode_id = hashlib.sha256(f"v3-episode:{seed}:{replica}".encode()).hexdigest()[:20]
            backend = RecordingBackend(seed, noise, profile, artifact_dir=artifacts)
            runner = AgentRunner(backend, V3Teacher(narrow), max_steps=32, max_tool_calls=8)
            transcript, sample_ids = [], []
            while runner.status == "active":
                messages = policy_messages(runner.context())
                action = runner.plan()
                if action is None:
                    break
                answer = assistant_call(action)
                sample_id = f"{episode_id}-{len(sample_ids):02d}"
                runner.step()
                if runner.status in {"invalid_action", "policy_error", "tool_error"}:
                    raise RuntimeError(f"Invalid v3 teacher trajectory {episode_id}: {runner.reason}")
                rows[split].append({"id": sample_id, "task": "next_action_v3", "device_id": device_id,
                    "episode_id": episode_id, "protocol_version": PROTOCOL_VERSION,
                    "tools": TOOLS_SCHEMA, "messages": messages+[answer]})
                sample_ids.append(sample_id)
                targets[action["next_tool"]] += 1
                transcript.extend([{"role": "user", "content": messages[1]["content"]}, answer,
                    {"role": "tool", "name": "calibration.step", "content": json.dumps({
                        "status": runner.status, "context": policy_messages(runner.context())[1]["content"]})}])
            result = runner.result()
            events = result.pop("events")
            write_json(output/"trajectories"/(episode_id+".json"), {"device_id": device_id,
                "split": split, "messages": transcript, "sample_ids": sample_ids, "result": result})
            write_json(output/"audit"/(episode_id+".json"), events)
            # Evaluator-only truth is isolated from JSONL/model messages and audited below.
            write_json(output/"evaluator_only"/(episode_id+".json"), {"seed": seed, "profile": profile,
                "noise_scale": noise, "narrow_initial_scan": narrow, "truth": backend._truth})
            outcomes[f"{split}:{runner.status}"] += 1
            episodes.append({"episode_id": episode_id, "device_id": device_id, "split": split,
                             "status": runner.status, "samples": len(sample_ids)})
        print(f"device {device_index+1}/{total_devices} {split}", flush=True)
    for split, samples in rows.items():
        with (output/(split+".jsonl")).open("x", encoding="utf-8") as handle:
            for row in samples:
                handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+"\n")
    groups = {split: {row["device_id"] for row in samples} for split, samples in rows.items()}
    if any(groups[a]&groups[b] for a in groups for b in groups if a < b):
        raise RuntimeError("Device leakage across splits")
    manifest = {"schema": PROTOCOL_VERSION, "runtime_schema": "runtime-0.2", "synthetic": True,
        "physics_version": PhysicalSimulator.backend_name, "teacher": "observable_only_rule_policy_v3",
        "training_ready": True, "release_scope": "single_qubit_simulation_research_not_hardware_validated",
        "data_origin": "executed reduced flux-transmon/cQED/XEB-style simulator; not experimental measurements",
        "loss_scope": "one next assistant native call per sample; preceding context fully masked",
        "device_counts": {key: len(value) for key, value in groups.items()},
        "sample_counts": {key: len(value) for key, value in rows.items()},
        "outcomes": dict(outcomes), "target_counts": dict(targets), "episodes": episodes,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).parents[1]/"src/qmagent").glob("*.py")},
        "split_sha256": {key: hashlib.sha256((output/(key+".jsonl")).read_bytes()).hexdigest() for key in rows},
        "sources": ["https://doi.org/10.1103/PhysRevA.76.042319",
                    "https://doi.org/10.1038/s41586-019-1666-5",
                    "https://doi.org/10.1063/1.5089550"],
        "limitations": ["Synthetic single-qubit XEB-style proxy, not a multiqubit XEB benchmark",
            "Standard single-qubit randomized benchmarking is more conventional",
            "Rule-teacher imitation is not proof of optimal hardware actions",
            "Evaluator-only truth is never model input", "No coupler or two-qubit physics"]}
    write_json(output/"manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("device_counts", "sample_counts", "outcomes", "target_counts")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=32)
    args = parser.parse_args()
    build(args.output, args.devices)
