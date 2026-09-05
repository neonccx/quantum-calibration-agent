"""Authenticated-by-SSH stdio service. No TCP port, HTTP endpoint or shell tool."""

import argparse
import _thread
import json
import os
import queue
import selectors
import signal
import sys
import threading

MAX_FRAME = 2 * 1024**2
METHODS = {"info", "status", "history", "why", "plan", "prepare", "execute", "cancel", "ask",
           "clear_chat", "pause", "report", "sessions", "new", "resume"}


def decode_frame(line):
    if len(line) > MAX_FRAME:
        raise ValueError("Protocol frame too large")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate protocol key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite protocol number")
    value = json.loads(line, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Expected protocol object")
    return value


def serve(service, incoming, outgoing):
    inbox = queue.Queue(maxsize=2)
    state = {"active": None, "disconnected": False, "cancelled": set()}
    guard = threading.Lock()
    stopped = threading.Event()

    def reader():
        selector = selectors.DefaultSelector()
        pending = bytearray()
        try:
            # Do not block a daemon thread inside Python's buffered stdin lock:
            # interpreter shutdown after SIGTERM would otherwise abort. Poll the
            # raw fd, so normal close/signals can stop and join the reader too.
            selector.register(incoming.fileno(), selectors.EVENT_READ)
            while not stopped.is_set():
                if not selector.select(timeout=0.2):
                    continue
                chunk = os.read(incoming.fileno(), 65536)
                if not chunk:
                    break
                pending.extend(chunk)
                while b"\n" in pending and not stopped.is_set():
                    line, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    message = decode_frame(line)
                    if set(message) == {"cancel_for"} and type(message["cancel_for"]) is int:
                        with guard:
                            if len(state["cancelled"]) < 16:
                                state["cancelled"].add(message["cancel_for"])
                            if state["active"] == message["cancel_for"]:
                                _thread.interrupt_main()
                        continue
                    while not stopped.is_set():
                        try:
                            inbox.put(message, timeout=0.2)
                            break
                        except queue.Full:
                            pass
                if len(pending) > MAX_FRAME:
                    raise ValueError("Protocol frame too large")
        except (ValueError, OSError, UnicodeError):
            pass
        finally:
            selector.close()
            with guard:
                state["disconnected"] = True
                if state["active"] is not None:
                    _thread.interrupt_main()
            try:
                inbox.put_nowait(None)
            except queue.Full:
                pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    def send(value):
        line = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        if len(line.encode()) > MAX_FRAME:
            raise ValueError("Protocol response too large; inspect full logs on server")
        outgoing.write(line)
        outgoing.flush()

    try:
        while not state["disconnected"]:
            request = inbox.get()
            if request is None:
                break
            request_id = request.get("id")
            try:
                if (set(request) != {"id", "method", "params"} or type(request_id) is not int
                        or request_id < 1 or not isinstance(request["params"], dict)):
                    raise ValueError("Invalid request envelope")
                if request["method"] == "close":
                    send({"id": request_id, "ok": True, "result": {"closing": True}})
                    break
                if request["method"] not in METHODS:
                    raise ValueError("Unlisted Agent method")
                with guard:
                    if request_id in state["cancelled"]:
                        state["cancelled"].discard(request_id)
                        raise ValueError("Request cancelled before execution")
                    state["active"] = request_id
                params = dict(request["params"])
                if "on_text" in params:
                    raise ValueError("Callbacks are server-owned")
                if request["method"] == "ask":
                    stream = params.pop("stream", False)
                    if type(stream) is not bool:
                        raise ValueError("stream must be boolean")
                    if stream:
                        params["on_text"] = lambda text: send({"id": request_id, "event": "text", "text": text})
                result = getattr(service, request["method"])(**params)
                send({"id": request_id, "ok": True, "result": result})
            except KeyboardInterrupt:
                try:
                    service.cancel()
                except (ValueError, OSError, RuntimeError):
                    pass
                if state["disconnected"]:
                    break
                send({"id": request_id, "ok": False, "error": "Cancelled; check saved session state before retrying"})
            except Exception as exc:
                if state["disconnected"]:
                    break
                send({"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                with guard:
                    state["active"] = None
                    state["cancelled"].discard(request_id)
    finally:
        stopped.set()
        thread.join(timeout=1)
        service.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home")
    parser.add_argument("--session")
    args = parser.parse_args(argv)
    # Preserve a dedicated protocol FD; imported libraries and spawned model
    # processes may write to fd 1, so redirect it to stderr before loading them.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from .service import AgentService
    from .session import CalibrationSession
    from .settings import app_home, load_settings
    home = app_home(args.home)
    session = (CalibrationSession.resume(home, args.session) if args.session
               else CalibrationSession.create(home, load_settings(home)))
    def shutdown(signum, frame):
        raise SystemExit(128 + signum)
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), shutdown)
    try:
        serve(AgentService(home, session), sys.stdin.buffer, output)
    except (KeyboardInterrupt, BrokenPipeError):
        return 130
    finally:
        output.close()
    return 0
