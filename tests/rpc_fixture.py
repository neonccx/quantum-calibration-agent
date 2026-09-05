"""Test-only cancellable chat; never selected by the production entry point."""
from pathlib import Path
import sys
import time

from qmagent.policies import RulePolicy
from qmagent.rpc import serve
from qmagent.service import AgentService
from qmagent.session import CalibrationSession
from qmagent.settings import Settings


class WaitingPolicy(RulePolicy):
    supports_streaming = True

    def chat(self, messages, context, on_text=None):
        if on_text:
            on_text("你好，streaming! ")
        (Path(sys.argv[1]) / "ready").touch()
        if messages[-1]["content"] == "stream test":
            deadline = time.monotonic() + 10
            while not (Path(sys.argv[1]) / "ack").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Client did not see text before completion")
                time.sleep(0.02)
            on_text("第二段。<qm_")
            on_text('action>{"command":"run","limit":1}</qm_action>')
            (Path(sys.argv[1]) / "complete").touch()
            return '你好，streaming! 第二段。<qm_action>{"command":"run","limit":1}</qm_action>'
        while True:
            time.sleep(0.02)


if __name__ == "__main__":
    home = Path(sys.argv[1])
    session = CalibrationSession.create(home, Settings(), policy=WaitingPolicy())
    serve(AgentService(home, session), sys.stdin.buffer, sys.stdout)
