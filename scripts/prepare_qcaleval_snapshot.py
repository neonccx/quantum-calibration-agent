#!/usr/bin/env python3
"""Freeze the official QCalEval test split into a content-addressed snapshot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from qmagent.qcaleval import QUESTION_NAMES, snapshot_digest, sha256_file


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "entry"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nvidia/QCalEval")
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face dataset commit SHA")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", args.revision):
        parser.error("revision must be a full immutable commit SHA, not a branch or tag")

    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        parser.error(f"install the pinned QCalEval preparation dependencies first: {exc}")

    dataset = load_dataset(args.dataset, split="test", revision=args.revision)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    config_path = hf_hub_download(
        repo_id=args.dataset,
        filename="experiment_config.json",
        repo_type="dataset",
        revision=args.revision,
    )
    experiment_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    type_to_family = {
        experiment_type: family
        for family, config in experiment_config.items()
        for experiment_type in config.get("q6_status_mapping", {})
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = args.output_dir / "images"
    image_dir.mkdir()
    entries = []
    for row_index, row in enumerate(dataset):
        entry_id = str(row["id"])
        images = []
        for image_index, image in enumerate(row["images"]):
            if image is None:
                continue
            relative = Path("images") / f"{row_index:04d}_{safe_component(entry_id)}_{image_index}.png"
            image.convert("RGBA").save(args.output_dir / relative, format="PNG")
            images.append({
                "path": relative.as_posix(),
                "mime_type": "image/png",
                "sha256": sha256_file(args.output_dir / relative),
                "source_image_id": row.get("image_ids", [None] * len(row["images"]))[image_index],
            })
        experiment_type = str(row["experiment_type"])
        family = type_to_family.get(experiment_type)
        entries.append({
            "id": entry_id,
            "experiment_type": experiment_type,
            "images": images,
            "prompts": {label: row[f"q{index}_prompt"] for index, label in enumerate(QUESTION_NAMES, 1)},
            "ground_truth": {label: row[f"q{index}_answer"] for index, label in enumerate(QUESTION_NAMES, 1)},
            "q5_scoring": experiment_config.get(family, {}).get("q5_scoring", {}),
        })

    snapshot = {
        "schema_version": "qmagent.qcaleval.snapshot.v1",
        "source": {
            "dataset_id": args.dataset,
            "revision": args.revision,
            "split": "test",
            "license": "CC-BY-4.0",
            "official_evaluator_commit": "e9e9b9eb8b95f93e0db1c578e699fc212cf0bb84",
        },
        "entries": entries,
    }
    snapshot["snapshot_sha256"] = snapshot_digest(snapshot)
    path = args.output_dir / "snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path} ({len(entries)} entries, SHA-256 {snapshot['snapshot_sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
