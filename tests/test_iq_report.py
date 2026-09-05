import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest

from qmagent.contracts import DEFAULT_STATE
from qmagent.iq_report import analyze_iq, from_trajectory, render_iq_report
from qmagent.report_preview import save_preview
from qmagent.simulator import AnalyticSimulator, iq_gate
from qmagent.session import CalibrationSession
from qmagent.service import AgentService
from qmagent.settings import Settings


class IQReportTests(unittest.TestCase):
    def measurement(self):
        return AnalyticSimulator(20260904).measure("sq.iqraw", dict(DEFAULT_STATE), {})["measurement"]

    def test_same_controller_gate_and_no_state_error_claim(self):
        data = self.measurement()
        original = copy.deepcopy(data)
        report = analyze_iq(data)
        self.assertEqual(report["gate"], iq_gate(data))
        self.assertEqual(data, original)
        self.assertIsNone(report["state_preparation_error"])
        self.assertFalse(report["max_visibility_diagnostic"]["used_for_controller_acceptance"])

    def test_threshold_fit_does_not_use_held_out_shots(self):
        data = self.measurement()
        before = analyze_iq(data)
        changed = copy.deepcopy(data)
        for label in (0, 1):
            positions = [i for i, value in enumerate(changed["prepared_state"]) if value == label]
            for i in positions[1::2]:
                changed["i"][i] += 3
        after = analyze_iq(changed)
        self.assertEqual(before["max_visibility_diagnostic"]["threshold"],
                         after["max_visibility_diagnostic"]["threshold"])
        self.assertNotEqual(before["gate"]["metrics"], after["gate"]["metrics"])

    def test_no_iq_does_not_invent_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            path.write_text('{"event":"terminal","status":"invalid_action"}\n')
            self.assertIsNone(from_trajectory(path, Path(tmp)/"report", policy="hf", status="invalid_action"))
            self.assertFalse((Path(tmp)/"report").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plotting dependency")
    def test_plot_outputs_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            render_iq_report(self.measurement(), path, policy="rule", status="active", step=1)
            self.assertTrue((path/"iq_report.png").read_bytes().startswith(b"\x89PNG"))
            self.assertLess((path/"iq_report.png").stat().st_size, 1024*1024)
            self.assertIn("<svg", (path/"iq_report.svg").read_text())
            with self.assertRaises(ValueError):
                render_iq_report(self.measurement(), path, policy="rule", status="active")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plotting dependency")
    def test_session_report_transfers_preview_from_recorded_iq(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            session = CalibrationSession.create(home, Settings())
            service = AgentService(home, session)
            try:
                while session.runner.status == "active":
                    session.step()
                result = service.report()
                page = save_preview(result, home / "client")
                self.assertTrue(page.is_file())
                metrics = json.loads((Path(result["directory"])/"iq_metrics.json").read_text())
                self.assertEqual(metrics["gate"], session.runner.result()["final_iq_gate"])
                self.assertEqual(metrics["policy"], "rule")
            finally:
                service.close()

    def test_preview_integrity_paths_and_html_escaping(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = b"\x89PNG\r\n\x1a\nfixture"
            preview = {"name":"iq_report.png", "base64":base64.b64encode(data).decode(),
                       "sha256": hashlib.sha256(data).hexdigest()}
            result = {"preview": preview, "directory": "<script>bad</script>"}
            page = save_preview(result, Path(tmp))
            self.assertNotIn("<script>", page.read_text())
            self.assertIn("&lt;script&gt;", page.read_text())
            preview["name"] = "../outside.png"
            with self.assertRaises(ValueError):
                save_preview(result, Path(tmp))
            preview["name"] = "iq_report.png"
            preview["sha256"] = "wrong"
            with self.assertRaises(ValueError):
                save_preview(result, Path(tmp))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plotting dependency")
    def test_png_preview_survives_actual_stdio_rpc(self):
        from qmagent.remote import RemoteService
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            client = RemoteService(command=[sys.executable, "-m", "qmagent", "rpc", "--home", str(home)])
            try:
                token = client.prepare(mode="run")["token"]
                while not client.execute(token=token)["done"]:
                    pass
                result = client.report()
                self.assertEqual(result["status"]["status"], "accepted")
                page = save_preview(result, home / "downloads")
                self.assertTrue(page.is_file())
                self.assertTrue((page.parent / "iq_report.png").is_file())
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
