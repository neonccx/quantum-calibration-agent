#!/usr/bin/env python3
"""Fail-closed integrity, shape, split and simulator-truth leakage audit for dataset_v3."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from qmagent.contracts import TOOLS
from qmagent.protocol import PROTOCOL_VERSION


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(dataset):
    manifest = json.loads((dataset/"manifest.json").read_text())
    if manifest["schema"] != PROTOCOL_VERSION or manifest["runtime_schema"] != "runtime-0.2":
        raise ValueError("Unexpected v3 protocol/runtime schema")
    split_devices, ids, targets, references = {}, set(), Counter(), set()
    forbidden = {"sweet_zpa", "junction_asymmetry", "ej_sum_hz", "xeb_cycle_fidelity", "_truth"}
    for split in ("train", "validation", "test", "ood"):
        path = dataset/(split+".jsonl")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["split_sha256"][split]:
            raise ValueError(f"{split} checksum mismatch")
        rows = load_lines(path)
        if len(rows) != manifest["sample_counts"][split]:
            raise ValueError(f"{split} count mismatch")
        split_devices[split] = {row["device_id"] for row in rows}
        for row in rows:
            if row["id"] in ids:
                raise ValueError("Duplicate sample id")
            ids.add(row["id"])
            if row["protocol_version"] != PROTOCOL_VERSION or row["task"] != "next_action_v3":
                raise ValueError("Mixed protocol/task row")
            encoded = json.dumps(row, sort_keys=True)
            if any(name in encoded for name in forbidden):
                raise ValueError("Simulator truth leaked into model-visible row")
            call = row["messages"][-1]["tool_calls"][0]["function"]
            target = call["arguments"]["next_tool"]
            targets[target] += 1
            context = json.loads(row["messages"][-2]["content"])
            observation = context.get("observation")
            if observation:
                if "measurement" in observation or "sweep" in observation:
                    raise ValueError("Raw arrays leaked into a prompt")
                reference = observation.get("raw_reference", {}).get("sha256")
                if reference:
                    references.add(reference)
    for left in split_devices:
        for right in split_devices:
            if left < right and split_devices[left]&split_devices[right]:
                raise ValueError(f"Device leakage: {left}/{right}")
    artifacts = list((dataset/"artifacts").glob("*.json"))
    for path in artifacts:
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
            raise ValueError(f"Artifact content/hash mismatch: {path.name}")
        raw = json.loads(path.read_text())
        i, q = raw["measurement"]["i"], raw["measurement"]["q"]
        if raw["tool"] == "sq.s21_zpa2d":
            expected = (len(raw["sweep"]["zpa_values"]), len(raw["sweep"]["frequency_values_hz"]))
            if len(i) != expected[0] or len(q) != expected[0] or any(len(row) != expected[1] for row in i+q):
                raise ValueError("Malformed ZPA2D artifact shape")
        elif len(i) != len(q):
            raise ValueError("Malformed aligned I/Q artifact")
    missing = references-{path.stem for path in artifacts}
    if missing:
        raise ValueError(f"Missing referenced artifacts: {len(missing)}")
    if not {"sq.s21_zpa2d", "sq.xeb"} <= set(targets):
        raise ValueError("New v3 tools are absent from native-call targets")
    report = {"protocol": PROTOCOL_VERSION, "samples": len(ids), "devices": {k: len(v) for k, v in split_devices.items()},
              "targets": dict(targets), "artifact_count": len(artifacts), "referenced_artifacts": len(references),
              "device_leakage": False, "truth_leakage": False, "artifact_hashes_valid": True,
              "zpa2d_shapes_valid": True, "new_tools_present": True}
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    audit(args.dataset)
