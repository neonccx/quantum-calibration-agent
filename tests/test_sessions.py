import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qmagent.contracts import DEFAULT_STATE, decision
from qmagent.policies import RulePolicy
from qmagent.runtime import AgentRunner
from qmagent.session import CalibrationSession, list_sessions, load_settings, save_settings
from qmagent.settings import Settings
from qmagent.simulator import AnalyticSimulator
from qmagent.storage import Journal, atomic_json, read_json, session_directory
from qmagent.terminal import Terminal, safe_text


class CountingPolicy(RulePolicy):
    def __init__(self):
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return super().decide(context)


class SessionFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)

    def create(self, **kwargs):
        session = CalibrationSession.create(self.home, Settings(), **kwargs)
        self.addCleanup(session.close)
        return session


class SessionTests(SessionFixture):
    def test_plan_cached_no_measurement_or_rng_advance(self):
        policy = CountingPolicy()
        session = self.create(policy=policy)
        before = session.runner.backend.checkpoint()
        plan = session.plan()
        self.assertEqual(session.plan(), plan)
        plan["next_tool"] = "FINISH"
        self.assertEqual(policy.calls, 1)
        self.assertEqual(session.runner.state, DEFAULT_STATE)
        self.assertEqual(session.runner.backend.checkpoint(), before)
        self.assertEqual(sum(session.runner.counts.values()), 0)
        session.step()
        self.assertEqual(policy.calls, 1)
        self.assertEqual(sum(session.runner.counts.values()), 1)

    def test_resume_mid_loop_exactly_matches_uninterrupted(self):
        session = self.create()
        for _ in range(5):
            session.step()
        session.plan()
        session_id = session.metadata["session_id"]
        checkpoint = session.runner.snapshot()
        session.close()
        resumed = CalibrationSession.resume(self.home, session_id)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.runner.snapshot(), checkpoint)
        while resumed.runner.status == "active":
            resumed.step()
        expected = AgentRunner(__import__('qmagent.physical_backend', fromlist=['PhysicalSimulator']).PhysicalSimulator(20260903), RulePolicy()).run()
        actual = resumed.runner.result()
        expected.pop("events")
        actual.pop("events")
        self.assertEqual(actual, expected)
        export = resumed.export(self.home / "export")
        trajectory = [json.loads(line) for line in (export / "trajectory.jsonl").read_text().splitlines()]
        experiments = [item for item in trajectory if item["event"] == "experiment"]
        self.assertEqual(len(experiments), actual["experiment_count"])
        self.assertEqual([item["step"] for item in experiments], list(range(1, len(experiments) + 1)))
        self.assertNotIn("backend_checkpoint", (export / "trajectory.jsonl").read_text())

    def test_exclusive_session_lock(self):
        session = self.create()
        with self.assertRaisesRegex(ValueError, "locked"):
            CalibrationSession.resume(self.home, session.metadata["session_id"])

    def test_path_traversal_and_symlink_rejected(self):
        for name in ("../other", "/tmp/x", "x/y", "", "x" * 81):
            with self.subTest(name=name), self.assertRaises(ValueError):
                session_directory(self.home, name)
        (self.home / "sessions").mkdir()
        (self.home / "sessions" / "linked").symlink_to(self.home, target_is_directory=True)
        with self.assertRaises(ValueError):
            session_directory(self.home, "linked")

    def test_metadata_edit_is_not_accepted(self):
        session = self.create()
        directory, session_id = session.directory, session.metadata["session_id"]
        session.close()
        metadata = read_json(directory / "session_config.json")
        metadata["settings"]["seed"] += 1
        atomic_json(directory / "session_config.json", metadata, replace=True)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            CalibrationSession.resume(self.home, session_id)

    def test_code_change_refuses_resume(self):
        session = self.create()
        session_id = session.metadata["session_id"]
        session.close()
        with patch("qmagent.session.source_hashes", return_value={}), self.assertRaisesRegex(ValueError, "version mismatch"):
            CalibrationSession.resume(self.home, session_id)

    def test_corrupted_record_refuses_resume(self):
        session = self.create()
        session.step()
        session_id, directory = session.metadata["session_id"], session.directory
        session.close()
        path = sorted((directory / "journal").glob("*.json"))[-1]
        value = read_json(path)
        value["body"]["snapshot"]["passes"] = 2
        atomic_json(path, value, replace=True)
        with self.assertRaisesRegex(ValueError, "checksum"):
            CalibrationSession.resume(self.home, session_id)

    def test_atomic_record_no_overwrite(self):
        path = self.home / "a.json"
        atomic_json(path, {"a": 1})
        with self.assertRaises(FileExistsError):
            atomic_json(path, {"a": 2})
        self.assertEqual(read_json(path), {"a": 1})

    def test_uncommitted_temp_record_ignored(self):
        session = self.create()
        (session.directory / "journal" / ".pending-test").write_text("partial")
        self.assertEqual(Journal(session.directory).recover()["snapshot"]["state"], DEFAULT_STATE)

    def test_persistence_failure_blocks_execution(self):
        session = self.create()
        with patch.object(session.journal, "append", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                session.plan()
        with self.assertRaises(RuntimeError):
            session.step()
        self.assertEqual(sum(session.runner.counts.values()), 0)

    def test_budget_persists(self):
        session = CalibrationSession.create(self.home, Settings(max_steps=1))
        session.step()
        session_id = session.metadata["session_id"]
        session.close()
        resumed = CalibrationSession.resume(self.home, session_id)
        self.addCleanup(resumed.close)
        resumed.step()
        self.assertEqual(resumed.runner.status, "budget_exhausted")
        self.assertEqual(sum(resumed.runner.counts.values()), 1)

    def test_terminal_session_cannot_reset_by_resuming(self):
        session = self.create()
        while session.runner.status == "active":
            session.step()
        count = sum(session.runner.counts.values())
        session_id = session.metadata["session_id"]
        session.close()
        resumed = CalibrationSession.resume(self.home, session_id)
        self.addCleanup(resumed.close)
        resumed.step()
        self.assertEqual(resumed.runner.status, "accepted")
        self.assertEqual(sum(resumed.runner.counts.values()), count)

    def test_interrupt_measurement_rolls_back_and_consumes_budget(self):
        session = self.create()
        before = session.runner.backend.checkpoint()
        def interrupt(*args):
            session.runner.backend.rng.random(10)
            raise KeyboardInterrupt
        with patch.object(session.runner.backend, "measure", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            session.step()
        self.assertEqual(session.runner.backend.checkpoint(), before)
        self.assertEqual(session.runner.state, DEFAULT_STATE)
        self.assertEqual(sum(session.runner.counts.values()), 1)
        self.assertEqual(session.runner.status, "active")
        self.assertEqual(Journal(session.directory).recover()["snapshot"]["counts"], {"sq.s21": 1})

    def test_listing_does_not_change_records(self):
        session = self.create()
        files = sorted(str(path) for path in self.home.rglob("*"))
        self.assertEqual(list_sessions(self.home)[0]["status"], "active")
        self.assertEqual(sorted(str(path) for path in self.home.rglob("*")), files)

    def test_export_refuses_existing_directory(self):
        session = self.create()
        with self.assertRaises(FileExistsError):
            session.export(self.home)

    def test_missing_backend_quality_rolls_back(self):
        session = self.create()
        before = session.runner.backend.checkpoint()
        original = session.runner.backend.measure
        def malformed(*args):
            result = original(*args)
            result.pop("quality")
            return result
        with patch.object(session.runner.backend, "measure", side_effect=malformed):
            session.step()
        self.assertEqual(session.runner.status, "tool_error")
        self.assertEqual(session.runner.state, DEFAULT_STATE)
        self.assertEqual(session.runner.backend.checkpoint(), before)

    def test_interrupt_during_commit_restores_all_state(self):
        session = self.create()
        before = session.runner.backend.checkpoint()
        def interrupt(event):
            if event["event"] == "experiment":
                raise KeyboardInterrupt
        session.runner.on_event = interrupt
        with self.assertRaises(KeyboardInterrupt):
            session.step()
        self.assertEqual(session.runner.backend.checkpoint(), before)
        self.assertEqual(session.runner.state, DEFAULT_STATE)
        self.assertIsNone(session.runner.observation)
        self.assertEqual(session.runner.history, [])
        self.assertEqual(sum(session.runner.counts.values()), 1)
        self.assertNotIn("experiment", [event["event"] for event in session.runner.events])


class TerminalTests(SessionFixture):
    def terminal(self, responses=()):
        output = []
        answers = iter(responses)
        return Terminal(self.home, self.create(), lambda prompt: next(answers), output.append), output

    def test_unknown_command_does_not_call_policy(self):
        terminal, output = self.terminal()
        with patch.object(terminal.session.policy, "decide", side_effect=AssertionError("Should not run")):
            terminal.dispatch("/shell rm -rf /somewhere")
        self.assertEqual(sum(terminal.session.runner.counts.values()), 0)
        self.assertIn("Unknown command", output[-1])

    def test_step_requires_confirmation(self):
        terminal, _ = self.terminal(["n", "y"])
        terminal.dispatch("/step")
        self.assertEqual(sum(terminal.session.runner.counts.values()), 0)
        terminal.dispatch("/step")
        self.assertEqual(sum(terminal.session.runner.counts.values()), 1)

    def test_run_cancellation_and_bounded_run(self):
        terminal, _ = self.terminal(["n", "y"])
        terminal.dispatch("/run")
        self.assertEqual(sum(terminal.session.runner.counts.values()), 0)
        terminal.dispatch("/run 3")
        self.assertEqual(sum(terminal.session.runner.counts.values()), 3)
        self.assertEqual(terminal.session.runner.status, "active")

    def test_status_alias_and_safe_terminal_output(self):
        terminal, output = self.terminal()
        terminal.dispatch("状态")
        terminal.say("\x1b]52;c;secret\x07\u202e")
        self.assertNotIn("\x1b", output[-1])
        self.assertNotIn("\u202e", output[-1])
        self.assertIn("\\u001b", output[-1])

    def test_invalid_command_args_fail_closed(self):
        terminal, _ = self.terminal()
        for command in ("/step yes", "/run 0", "/run 1001", "/resume ../other", "/status garbage"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                terminal.dispatch(command)
        self.assertEqual(sum(terminal.session.runner.counts.values()), 0)


class SettingsTests(unittest.TestCase):
    def test_reject_credentials_and_invalid_fields(self):
        for data in ({"api_key": "not-a-real-key"}, {"seed": True}, {"policy": "hf"},
                     {"model": "relative"}, {"request_timeout": 0}, {"noise_scale": float("nan")},
                     {"policy": "rule", "model": "/tmp/model"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                Settings.from_dict(data)

    def test_profile_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            settings = Settings(seed=123)
            save_settings(home, settings)
            self.assertEqual(load_settings(home), settings)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"seed":1,"seed":2}')
            with self.assertRaises(ValueError):
                read_json(path)
