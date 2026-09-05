"""Standard-library-only Mac client; all model/runtime work stays on the server."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import subprocess
import time

from .conversation import validate_message
from .rpc import MAX_FRAME, METHODS, decode_frame
from .storage import atomic_json, read_json


@dataclass(frozen=True)
class RemoteProfile:
    host: str
    project: str
    control_path: str | None = None
    home: str | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"host", "project", "control_path", "home"}:
            raise ValueError("Invalid remote profile; credentials/commands are not supported")
        if not {"host", "project"} <= set(value):
            raise ValueError("Missing remote host/project")
        result = cls(**value)
        if (not isinstance(result.host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,200}", result.host)
                or result.host.count("@") > 1):
            raise ValueError("Invalid SSH host/account")
        for name in ("project", "home"):
            path = getattr(result, name)
            if path is not None and (not isinstance(path, str) or not PurePosixPath(path).is_absolute()
                                     or any(ord(char) < 32 for char in path)):
                raise ValueError("Remote paths must be absolute, without control characters")
        if not result.project:
            raise ValueError("Missing remote project")
        if result.control_path is not None and (not isinstance(result.control_path, str) or not Path(result.control_path).is_absolute()):
            raise ValueError("Control socket path must be absolute")
        return result

    def command(self, session_id=None):
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
        if self.control_path:
            command += ["-S", self.control_path]
        remote_args = ["./qm-agent", "rpc"]
        if self.home:
            remote_args += ["--home", self.home]
        if session_id:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
                raise ValueError("Invalid session ID")
            remote_args += ["--session", session_id]
        return command + [self.host, "cd " + shlex.quote(self.project) + " && exec " + shlex.join(remote_args)]

    def login_command(self):
        if not self.control_path:
            raise ValueError("Configure --control-path for reusable interactive SSH login")
        return ["ssh", "-o", "ControlMaster=auto", "-o", "ControlPersist=3600",
                "-o", "ControlPath=" + self.control_path, self.host]


def save_remote(home, profile):
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(home / "remote.json", asdict(profile), replace=True)


def load_remote(home):
    path = home / "remote.json"
    if not path.exists():
        raise ValueError("Remote not configured; run qm-agent remote --host ... --project ... --control-path ...")
    return RemoteProfile.from_dict(read_json(path))


class RemoteService:
    remote = True

    def __init__(self, profile=None, session_id=None, command=None, on_wait=None):
        self.profile = profile
        self.process = subprocess.Popen(command or profile.command(session_id), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, bufsize=0, start_new_session=True)
        # The terminal's Ctrl-C belongs to this client, not the SSH subprocess.
        # Keep the channel alive long enough to deliver our scoped cancel frame.
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = bytearray()
        self.next_id = 0
        self.connected = True
        self.timeout = 240.0
        self.on_wait = on_wait
        try:
            self.cached_info = self.call("info")
            self.timeout = self.cached_info["settings"]["request_timeout"] + 60
        except BaseException:
            self._disconnect()
            raise

    def _send(self, value):
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if len(data) > MAX_FRAME:
            raise ValueError("Request too large")
        pending = memoryview(data)
        while pending:
            count = self.process.stdin.write(pending)
            if not count:
                raise ConnectionError("SSH input closed during request; do not automatically replay")
            pending = pending[count:]
        self.process.stdin.flush()

    def _receive(self, request_id, timeout, progress=True, on_text=None):
        start = time.monotonic()
        last_notice = 0
        text_size = 0
        while time.monotonic() - start < timeout:
            if b"\n" in self.buffer:
                line, _, remaining = self.buffer.partition(b"\n")
                self.buffer = bytearray(remaining)
                result = decode_frame(line)
                if result.get("id") != request_id:
                    raise RuntimeError("Unexpected response ID; refusing automatic replay")
                if "event" in result:
                    if (set(result) != {"id", "event", "text"} or result["event"] != "text"
                            or not isinstance(result["text"], str)):
                        raise ValueError("Invalid remote stream event")
                    text_size += len(result["text"])
                    if text_size > 65536:
                        raise ValueError("Remote text stream exceeds limit")
                    if on_text is not None:
                        on_text(result["text"])
                    last_notice = time.monotonic() - start
                    continue
                return result
            if self.selector.select(timeout=0.2):
                chunk = self.process.stdout.read(65536)
                if not chunk:
                    raise ConnectionError("SSH disconnected. The request may have been submitted; resume to inspect state before retrying.")
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_FRAME:
                    raise ValueError("Remote response exceeds limit")
            elapsed = time.monotonic() - start
            if progress and self.on_wait and elapsed - last_notice >= 20:
                self.on_wait(f"Server processing; waited {int(elapsed)} s; Ctrl-C cancels.")
                last_notice = elapsed
        raise TimeoutError("Remote request timed out; inspect saved state before retrying")

    def ask(self, text, on_text=None):
        options = {}
        if on_text is not None and self.cached_info.get("capabilities", {}).get("text_streaming"):
            options = {"stream": True, "on_text": on_text}
        return self.call("ask", text=text, **options)

    def call(self, method, on_text=None, **params):
        if not self.connected:
            raise ConnectionError("Connection closed; connect --resume <session-id>. Operations are not automatically replayed.")
        if method not in METHODS | {"close"}:
            raise ValueError("Unlisted remote method")
        if method == "ask":
            validate_message(params.get("text"))
        self.next_id += 1
        request_id = self.next_id
        try:
            self._send({"id": request_id, "method": method, "params": params})
            result = self._receive(request_id, self.timeout, on_text=on_text)
        except KeyboardInterrupt:
            try:
                self._send({"cancel_for": request_id})
                self._receive(request_id, 10, progress=False)
            except BaseException:
                self._disconnect()
            raise
        except (OSError, ValueError, TimeoutError, RuntimeError):
            self._disconnect()
            raise
        if result.get("ok") is not True:
            raise RuntimeError(result.get("error", "Remote Agent failed"))
        return result["result"]

    def __getattr__(self, name):
        if name in METHODS:
            return lambda **params: self.call(name, **params)
        raise AttributeError(name)

    def _disconnect(self):
        if not self.connected:
            return
        self.connected = False
        try:
            self.process.stdin.close()  # EOF triggers server-side cancellation and model cleanup.
        except OSError:
            pass
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.process.terminate()  # Our SSH channel only, not the user's ControlMaster.
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.process.stdout.close()
        self.selector.close()

    def close(self):
        if self.connected:
            try:
                self.call("close")
            except (OSError, ValueError, RuntimeError):
                pass
            finally:
                self._disconnect()
