import json
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

from qmagent.conversation import ReplyStream, parse_reply
from qmagent.model_worker import ManagedPolicy
from qmagent.remote import RemoteService
from qmagent.settings import Settings
from qmagent.terminal import Terminal

ROOT = Path(__file__).resolve().parents[1]


def streaming_worker(connection, options):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    request = json.loads(connection.recv_bytes())
    connection.send_bytes(json.dumps({"event": "text", "text": "first 中"}).encode())
    marker = Path(request["context"]["ack"])
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    connection.send_bytes(json.dumps({"ok": marker.exists(), "answer": "first 中文",
                                     "error": "No streaming callback"}).encode())
    connection.recv_bytes()  # Graceful close.
    connection.close()


class StreamTests(unittest.TestCase):
    def test_default_budget_raised_and_still_bounded(self):
        self.assertEqual(Settings().chat_max_new_tokens, 2048)
        with self.assertRaises(ValueError):
            Settings.from_dict({"chat_max_new_tokens": 0})
        self.assertEqual(Settings.from_dict({"chat_max_new_tokens": 4096}).chat_max_new_tokens, 4096)

    def test_every_split_hides_proposal_and_preserves_unicode_whitespace(self):
        raw = '  你好 🌍\nUse x < y.\n\nYes!  <qm_action>{"command":"run","limit":1}</qm_action>'
        expected, proposal, _ = parse_reply(raw)
        self.assertEqual(proposal["command"], "run")
        for split in range(len(raw) + 1):
            chunks = []
            stream = ReplyStream(chunks.append)
            stream.feed(raw[:split])
            stream.feed(raw[split:])
            self.assertNotIn("<qm_action>", "".join(chunks))
            stream.finish(expected)
            self.assertEqual("".join(chunks), expected)
        chunks = []
        stream = ReplyStream(chunks.append)
        for char in raw:
            stream.feed(char)
        stream.finish(expected)
        self.assertEqual("".join(chunks), expected)

    def test_partial_marker_and_invalid_proposal_are_not_executable(self):
        for raw in ('Use <qm_', '<qm_action>{"command":"shell"}</qm_action>',
                    'Hello<qm_action>{bad', '  line\n\ntext  '):
            chunks = []
            stream = ReplyStream(chunks.append)
            for char in raw:
                stream.feed(char)
            final, proposal, _ = parse_reply(raw)
            stream.finish(final)
            self.assertIsNone(proposal)
            self.assertEqual("".join(chunks), final)

    def test_stream_size_and_final_consistency(self):
        stream = ReplyStream(lambda _: None)
        stream.feed("hello")
        self.assertFalse(stream.finish("different"))
        with self.assertRaises(ValueError):
            stream.feed("x" * 65536)

    def test_worker_callback_precedes_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "ack"
            policy = ManagedPolicy(Settings(request_timeout=20), streaming_worker)
            chunks = []
            def receive(text):
                self.assertTrue(policy.in_flight)
                chunks.append(text)
                marker.touch()
            try:
                result = policy.chat([{"role": "user", "content": "test"}], {"ack": str(marker)}, receive)
                self.assertEqual(chunks, ["first 中"])
                self.assertEqual(result, "first 中文")
                self.assertFalse(policy.in_flight)
            finally:
                policy.close()

    def connect(self, home):
        client = RemoteService(command=[sys.executable, str(ROOT / "tests/rpc_fixture.py"), str(home)])
        self.addCleanup(client.close)
        return client

    def test_real_rpc_stream_received_before_model_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            client = self.connect(home)
            chunks = []
            def receive(text):
                if not chunks:
                    self.assertFalse((home / "complete").exists())
                    (home / "ack").touch()
                chunks.append(text)
            result = client.ask("stream test", on_text=receive)
            self.assertEqual("".join(chunks), result["message"])
            self.assertGreaterEqual(len(chunks), 2)
            self.assertNotIn("qm_action", "".join(chunks))
            self.assertEqual(result["proposal"], {"command": "run", "limit": 1})
            self.assertEqual(client.status()["experiment_count"], 0)
            client.close()

    def test_cancel_from_stream_callback_drains_and_keeps_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.connect(Path(directory))
            sid = client.info()["session_id"]
            def cancel(text):
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                client.ask("cancel streaming", on_text=cancel)
            self.assertTrue(client.connected)
            self.assertEqual(client.status()["experiment_count"], 0)
            self.assertEqual(client.info()["conversation_turns"], 0)
            records = list((Path(directory) / "sessions" / sid).rglob("*.json*"))
            self.assertTrue(any('partial_assistant' in p.read_text() for p in records))
            client.close()

    def test_terminal_flushes_chunks_without_duplicate_final(self):
        class Service:
            def ask(self, text, on_text):
                on_text("你好 ")
                on_text("world!\x1b")
                return {"source": "model", "message": "你好 world!\x1b", "warning": None, "proposal": None}
        lines, writes = [], []
        terminal = Terminal(Path("."), service=Service(), output_fn=lines.append, write_fn=writes.append)
        terminal.ask("hello")
        self.assertEqual(writes, ["Nanbeige: ", "你好 ", "world!\\u001b", "\n"])
        self.assertFalse(any("world!" in line for line in lines))

    def test_terminal_labels_interrupted_partial(self):
        class Service:
            def ask(self, text, on_text):
                on_text("partial")
                raise TimeoutError("test")
        lines, writes = [], []
        terminal = Terminal(Path("."), service=Service(), output_fn=lines.append, write_fn=writes.append)
        with self.assertRaises(TimeoutError):
            terminal.ask("hello")
        self.assertTrue(any("incomplete" in line for line in lines))
        self.assertEqual(writes[-1], "\n")

    def test_terminal_preserves_canonical_answer_after_preview_difference(self):
        class Service:
            def ask(self, text, on_text):
                on_text("alpha  beta")
                return {"source": "model", "message": "alpha beta", "warning": None, "proposal": None}
        lines = []
        Terminal(Path("."), service=Service(), output_fn=lines.append).ask("test")
        self.assertIn("alpha beta", lines)

    def test_reconciled_service_saves_completed_chat(self):
        from qmagent.policies import RulePolicy
        from qmagent.service import AgentService
        from qmagent.session import CalibrationSession
        class Policy(RulePolicy):
            supports_streaming = True
            def chat(self, messages, context, on_text=None):
                if on_text:
                    on_text("alpha  beta")
                return "alpha beta"
        with tempfile.TemporaryDirectory() as tmp:
            session = CalibrationSession.create(Path(tmp), Settings(), policy=Policy())
            try:
                service = AgentService(Path(tmp), session)
                answer = service.ask("test", on_text=lambda _: None)
                self.assertEqual(answer["message"], "alpha beta")
                self.assertEqual(service.total_turns, 1)
                self.assertEqual(service.status()["experiment_count"], 0)
                self.assertTrue(any(e["event"] == "stream_reconciled" for e in session.runner.events))
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
