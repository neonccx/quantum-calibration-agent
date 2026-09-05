import json
import signal
import time
import unittest

from qmagent.model_worker import ManagedPolicy
from qmagent.runtime import AgentRunner
from qmagent.settings import Settings
from qmagent.simulator import AnalyticSimulator


def echo_worker(connection, options):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while True:
            request = json.loads(connection.recv_bytes())
            if request.get("op") == "close":
                return
            connection.send_bytes(json.dumps({"ok": True, "answer": json.dumps(request["context"])}).encode())
    except EOFError:
        pass
    finally:
        connection.close()


def sleeping_worker(connection, options):
    time.sleep(30)


def error_worker(connection, options):
    connection.recv_bytes()
    connection.send_bytes(json.dumps({"ok": False, "error": "fake model load failure"}).encode())
    connection.close()


class WorkerTests(unittest.TestCase):
    def policy(self, worker, timeout=20):
        # Happy-path spawn imports NumPy/SciPy through this test module. On a
        # loaded shared server cold imports can exceed 3s; this is not a speed test.
        # The deadline/cleanup test below still explicitly uses one second.
        policy = ManagedPolicy(Settings(policy="hf", model="/tmp/test-only", request_timeout=timeout), worker)
        self.addCleanup(policy.close)
        return policy

    def test_public_context_roundtrip_reuses_process(self):
        policy = self.policy(echo_worker)
        self.assertEqual(json.loads(policy.decide({"state": {"x": 1}})), {"state": {"x": 1}})
        pid = policy.process.pid
        self.assertEqual(json.loads(policy.decide({"state": {"x": 2}})), {"state": {"x": 2}})
        self.assertEqual(policy.process.pid, pid)
        policy.close()
        self.assertIsNone(policy.process)

    def test_loading_timeout_kills_child_and_does_not_fallback(self):
        policy = self.policy(sleeping_worker, timeout=1)
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            policy.decide({"large": "x" * 500000})
        self.assertLess(time.monotonic() - start, 5)
        self.assertIsNone(policy.process)

    def test_chat_roundtrip_reuses_decision_worker(self):
        policy = self.policy(echo_worker)
        policy.decide({"stage": "plan"})
        pid = policy.process.pid
        reply = policy.chat([{"role": "user", "content": "Hello"}], {"stage": "chat"})
        self.assertEqual(json.loads(reply), {"stage": "chat"})
        self.assertEqual(policy.process.pid, pid)

    def test_model_error_is_policy_error_and_no_experiment(self):
        policy = self.policy(error_worker)
        outcome = AgentRunner(AnalyticSimulator(), policy).run()
        self.assertEqual(outcome["status"], "policy_error")
        self.assertEqual(outcome["experiment_count"], 0)
        self.assertIn("fake model load failure", outcome["reason"])

    def test_oversized_prompt_rejected(self):
        policy = self.policy(echo_worker)
        with self.assertRaises(ValueError):
            policy.decide({"text": "x" * (2 * 1024**2)})
        self.assertIsNone(policy.process)
