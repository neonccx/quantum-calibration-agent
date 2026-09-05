import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("compare_policy_v2", Path(__file__).resolve().parents[1]/"scripts/compare_policy_v2.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.before, self.after = [Path(self.temp.name)/name for name in ("base", "sft")]
        for directory, adapter, flags in ((self.before, None, [True, False]), (self.after, "adapter", [False, True])):
            directory.mkdir()
            (directory/"config.json").write_text(json.dumps(dict(protocol="v2", model="same", test_sha256="frozen",
                tokenizer_sha256="frozen", selected_ids=["a", "b"], adapter=adapter)))
            rows = [dict(id=identity, expected={"next_tool": "sq.s21"}, valid=True, next_tool_correct=flag,
                         arguments_correct=flag) for identity, flag in zip(("a", "b"), flags)]
            (directory/"predictions.jsonl").write_text("\n".join(map(json.dumps, rows)))
            (directory/"controller_score.json").write_text(json.dumps({"records": [
                dict(id=identity, controller_executable=flag) for identity, flag in zip(("a", "b"), flags)]}))

    def test_equal_aggregate_still_reports_regression(self):
        metric = comparison.compare(self.before, self.after)["metrics"]["next_tool_correct"]
        self.assertEqual(metric["delta_percentage_points"], 0)
        self.assertEqual(metric["improved_ids"], ["b"])
        self.assertEqual(metric["regressed_ids"], ["a"])

    def test_incomplete_run_rejected(self):
        path = self.after/"predictions.jsonl"
        path.write_text(path.read_text().splitlines()[0])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            comparison.compare(self.before, self.after)

    def test_changed_dataset_rejected(self):
        path = self.after/"config.json"
        config = json.loads(path.read_text())
        config["test_sha256"] = "changed"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "test_sha256"):
            comparison.compare(self.before, self.after)

    def test_prompt_only_comparison_requires_same_adapter_and_profiles(self):
        for directory, profile, prompt_hash in ((self.before, "skill", "full"),
                                                 (self.after, "minimal", "short")):
            config = json.loads((directory/"config.json").read_text())
            config.update(adapter="same-adapter", prompt_profile=profile,
                          system_prompt_sha256=prompt_hash)
            (directory/"config.json").write_text(json.dumps(config))
        report = comparison.compare(self.before, self.after, comparison="prompt")
        self.assertEqual(report["comparison"]["dimension"], "system_prompt")
        self.assertEqual(report["comparison"]["baseline_profile"], "skill")
        self.assertEqual(report["comparison"]["candidate_profile"], "minimal")

    def test_prompt_comparison_rejects_changed_weights(self):
        for directory, profile in ((self.before, "skill"), (self.after, "minimal")):
            config = json.loads((directory/"config.json").read_text())
            config.update(prompt_profile=profile, system_prompt_sha256=profile)
            (directory/"config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "identical checkpoint/adapter"):
            comparison.compare(self.before, self.after, comparison="prompt")
