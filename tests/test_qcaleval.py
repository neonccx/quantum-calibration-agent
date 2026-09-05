import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import importlib.util

from qmagent.qcaleval import (
    score_reproducible_subset,
    sha256_file,
    snapshot_digest,
    validate_snapshot,
)

spec = importlib.util.spec_from_file_location(
    "evaluate_qcaleval", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_qcaleval.py"
)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class QCalEvalSnapshotTests(unittest.TestCase):
    def make_snapshot(self, root: Path):
        image = root / "plot.png"
        image.write_bytes(b"fixture-image")
        entry = {
            "id": "ramsey_fixture",
            "experiment_type": "ramsey_success",
            "images": [{"path": "plot.png", "mime_type": "image/png", "sha256": sha256_file(image)}],
            "prompts": {f"Q{i}": f"question {i}" for i in range(1, 7)},
            "ground_truth": {
                "Q1": "{}",
                "Q2": "Classification: Expected behavior",
                "Q3": "reasoning",
                "Q4": "Assessment: Reliable",
                "Q5": '{"frequency_MHz": 2.0}',
                "Q6": "Status: SUCCESS",
            },
            "q5_scoring": {"frequency_MHz": {"type": "pct", "tol_full": 0.05, "tol_half": 0.1}},
        }
        snapshot = {
            "schema_version": "qmagent.qcaleval.snapshot.v1",
            "source": {"dataset_id": "fixture", "revision": "0123456789abcdef", "split": "test"},
            "entries": [entry],
        }
        snapshot["snapshot_sha256"] = snapshot_digest(snapshot)
        path = root / "snapshot.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        return path, entry

    def test_snapshot_hashes_manifest_and_images(self):
        with TemporaryDirectory() as directory:
            path, _ = self.make_snapshot(Path(directory))
            self.assertEqual(validate_snapshot(path)["source"]["dataset_id"], "fixture")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["entries"][0]["prompts"]["Q1"] = "tampered"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot SHA-256 mismatch"):
                validate_snapshot(path)

    def test_reproducible_subset_never_claims_q1_or_q3(self):
        with TemporaryDirectory() as directory:
            _, entry = self.make_snapshot(Path(directory))
            responses = {
                "experimental_conclusion": {"answer": "Classification: Expected behavior"},
                "fit_reliability": {"answer": "Assessment: Reliable"},
                "parameter_extraction": {"answer": '{"frequency_MHz": 2.08}'},
                "calibration_diagnosis": {"answer": "Status: SUCCESS"},
            }
            scores = score_reproducible_subset(entry, responses)
            self.assertIsNone(scores["Q1"]["score"])
            self.assertIsNone(scores["Q3"]["score"])
            self.assertEqual(scores["Q2"]["score"], 100.0)
            self.assertEqual(scores["Q4"]["score"], 100.0)
            self.assertEqual(scores["Q5"]["score"], 100.0)
            self.assertEqual(scores["Q6"]["score"], 100.0)

    def test_q6_official_alias(self):
        with TemporaryDirectory() as directory:
            _, entry = self.make_snapshot(Path(directory))
            entry["ground_truth"]["Q6"] = "Status: TOO_MANY_OSC"
            scores = score_reproducible_subset(
                entry,
                {
                    "experimental_conclusion": {"answer": ""},
                    "fit_reliability": {"answer": ""},
                    "parameter_extraction": {"answer": "{}"},
                    "calibration_diagnosis": {"answer": "Status: SAMPLING_TOO_COARSE"},
                },
            )
            self.assertEqual(scores["Q6"]["score"], 100.0)

    def test_api_endpoint_rejects_secret_bearing_or_remote_http_urls(self):
        self.assertEqual(
            evaluator.validate_api_base("http://127.0.0.1:8000/v1/chat/completions"),
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(
            evaluator.validate_api_base("https://vlm.example/v1/chat/completions"),
            "https://vlm.example/v1/chat/completions",
        )
        for value in (
            "http://vlm.example/v1/chat/completions",
            "https://token@vlm.example/v1/chat/completions",
            "https://vlm.example/v1/chat/completions?key=secret",
            "https://vlm.example/v1/models",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluator.validate_api_base(value)


if __name__ == "__main__":
    unittest.main()
