"""Lazy model subprocess with a real wall-clock timeout, including model loading.

The child receives only public context. Cancellation terminates our child, never an
external GPU process. The model's trusted local Python code is not OS-sandboxed.
"""

import json
import multiprocessing
import signal
import time


def _model_process(connection, options):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Parent owns Ctrl-C and cleanup.
    # Keep model/library progress out of an SSH stdio protocol stream.
    import os
    os.dup2(2, 1)
    try:
        from .policies import HuggingFacePolicy
        policy = HuggingFacePolicy(options["model"], options["adapter"], options["max_input_tokens"],
                                   options["max_new_tokens"], options["trust_remote_code"], options["decode_backend"],
                                   fit_update_tool=options.get("fit_update_tool", False),
                                   prompt_profile=options.get("prompt_profile", "skill"))
        while True:
            request = json.loads(connection.recv_bytes())
            if request["op"] == "close":
                return
            if request["op"] == "chat":
                def emit_text(text):
                    connection.send_bytes(json.dumps({"event": "text", "text": text}).encode())
                answer = policy.chat(request["messages"], request["context"], options["chat_max_new_tokens"],
                                     on_text=emit_text if request.get("stream") else None)
            elif request["op"] == "decide":
                answer = policy.decide(request["context"])
            else:
                raise ValueError("Unknown model request")
            connection.send_bytes(json.dumps({"ok": True, "answer": answer,
                "generation": policy.last_generation}, allow_nan=False).encode())
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send_bytes(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}).encode())
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class ManagedPolicy:
    supports_streaming = True

    def __init__(self, settings, worker_target=_model_process):
        self.options = settings.to_dict()
        self.timeout = settings.request_timeout
        self.worker_target = worker_target
        self.process = self.connection = None
        self.last_generation = None
        self.in_flight = False

    def decide(self, context):
        return self.request({"op": "decide", "context": context})

    def chat(self, messages, context, on_text=None):
        return self.request({"op": "chat", "messages": messages, "context": context,
                             "stream": on_text is not None}, on_text=on_text)

    def request(self, payload, on_text=None):
        started = time.monotonic()
        self.in_flight = True
        try:
            if self.process is None:
                spawn = multiprocessing.get_context("spawn")
                self.connection, child = spawn.Pipe(duplex=True)
                self.process = spawn.Process(target=self.worker_target, args=(child, self.options), daemon=True)
                self.process.start()
                child.close()
            request = json.dumps(payload, allow_nan=False).encode()
            # Public prompts are bounded before tokenization; avoid blocking on huge pipe writes.
            if len(request) > 2 * 1024**2:
                raise ValueError("Model context exceeds 2 MiB")
            # A dedicated sender thread prevents loading/pipe backpressure defeating the deadline.
            import threading
            errors = []
            connection = self.connection
            def send():
                try:
                    connection.send_bytes(request)
                except (OSError, EOFError) as exc:
                    errors.append(exc)
            sender = threading.Thread(target=send, daemon=True)
            sender.start()
            while time.monotonic() - started < self.timeout:
                if errors:
                    raise RuntimeError("Model worker disconnected") from errors[0]
                if self.connection.poll(0.1):
                    response = json.loads(self.connection.recv_bytes(1024 * 1024))
                    if response.get("event") == "text":
                        if not isinstance(response.get("text"), str) or on_text is None:
                            raise ValueError("Invalid model stream event")
                        on_text(response["text"])
                        continue  # Deltas do not finish the request or reset its deadline.
                    sender.join(timeout=0.5)
                    self.in_flight = False
                    if not response["ok"]:
                        raise RuntimeError(response["error"])
                    self.last_generation = response.get("generation")
                    return response["answer"]
                if not self.process.is_alive():
                    raise RuntimeError("Model worker exited without an answer")
            raise TimeoutError(f"Model request exceeded {self.timeout:g} seconds including initial loading; worker stopped without switching to a rule policy")
        except (Exception, KeyboardInterrupt):
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.pid is not None:
                if self.process.is_alive() and not self.in_flight and self.connection is not None:
                    try:
                        self.connection.send_bytes(b'{"op":"close"}')
                        self.process.join(timeout=2)
                    except (OSError, EOFError):
                        pass
                if self.process.is_alive():
                    self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.in_flight = False
