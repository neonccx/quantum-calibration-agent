"""Shared Agent application core for server-local UI and SSH RPC clients."""

import copy
from pathlib import Path
import secrets

from . import __version__
from .conversation import ReplyStream, explicit_intent, parse_reply, validate_message
from .session import CalibrationSession, list_sessions
from .settings import Settings, load_settings
from .storage import digest


class AgentService:
    def __init__(self, home, session):
        self.home, self.session = home, session
        self.ticket = None
        self._restore_chat()

    def _restore_chat(self):
        self.messages, self.total_turns = [], 0
        for record, _ in self.session.journal.records():
            for event in record["events"]:
                if event["event"] == "chat_clear":
                    self.messages, self.total_turns = [], 0
                if event["event"] == "chat":
                    self.messages.extend([{"role": "user", "content": event["user"]},
                                          {"role": "assistant", "content": event["assistant"]}])
                    self.messages = self.messages[-24:]
                    self.total_turns += 1

    def info(self):
        return {"version": __version__, "protocol": 1, "session_id": self.session.metadata["session_id"],
                "directory": str(self.session.directory), "settings": self.session.settings.to_dict(),
                "backend": "analytic simulator only", "conversation_turns": self.total_turns,
                "broken": self.session.broken, "capabilities": {"text_streaming": True}}

    def status(self):
        result = self.session.runner.result()
        result.pop("events")
        result["max_steps"] = self.session.settings.max_steps
        return result

    def history(self):
        return copy.deepcopy(self.session.runner.history)

    def why(self):
        action = self.session.runner.pending
        if action is None:
            for record, _ in self.session.journal.records():
                for event in record["events"]:
                    if event["event"] == "decision":
                        action = event["decision"]
        return {"reason": action["reason"] if action else "No decision yet. Request a preview of the next action.",
                "fit_result": (self.session.runner.observation or {}).get("fit_result", {})}

    def plan(self):
        self.ticket = None
        return {"plan": self.session.plan(), "status": self.status()}

    def prepare(self, mode="step", limit=None):
        """Issue a scoped challenge, with no execution or state writes to instruments."""
        self.ticket = None
        if mode not in {"step", "run"} or (limit is not None and (type(limit) is not int or not 1 <= limit <= 1000)):
            raise ValueError("Invalid execution request")
        if mode == "step" and limit is not None:
            raise ValueError("step does not accept a limit")
        if self.session.runner.status != "active":
            return {"token": None, "status": self.status()}
        plan = self.session.plan() if mode == "step" else None
        if mode == "step" and plan is None:
            return {"token": None, "status": self.status()}
        remaining = self.session.settings.max_steps - sum(self.session.runner.counts.values())
        allowance = 1 if mode == "step" else min(remaining, limit if limit is not None else remaining)
        token = secrets.token_urlsafe(24)
        self.ticket = {"token": token, "mode": mode, "remaining": allowance,
                       "state": digest(self.session.runner.snapshot())}
        return {"token": token, "mode": mode, "limit": allowance, "plan": plan, "status": self.status()}

    def execute(self, token):
        """A chat reply cannot call this: only the UI-confirmed capability can."""
        ticket = self.ticket
        if (not isinstance(token, str) or not ticket or not secrets.compare_digest(ticket["token"], token)
                or ticket["state"] != digest(self.session.runner.snapshot())):
            self.ticket = None
            raise ValueError("Execution confirmation expired; preview and confirm the plan again")
        before_count = sum(self.session.runner.counts.values())
        before_events = len(self.session.runner.events)
        try:
            action = self.session.plan()
            done = action is None
            if action is not None and ticket["remaining"] == 0 and action["next_tool"].startswith("sq."):
                done = True
            elif action is not None:
                self.session.step()
                ticket["remaining"] -= sum(self.session.runner.counts.values()) - before_count
                done = self.session.runner.status != "active" or ticket["mode"] == "step"
            ticket["state"] = digest(self.session.runner.snapshot())
            if done:
                self.ticket = None
            observations = [event for event in self.session.runner.events[before_events:] if event["event"] == "experiment"]
            last = observations[-1] if observations else None
            summary = ({"step": last["step"], "tool": last["tool"], "fit_result": last["observation"]["fit_result"],
                        "quality": last["observation"]["quality"], "iq_passes": last["consecutive_iq_passes"]} if last else None)
            return {"status": self.status(), "experiment": summary, "done": done, "remaining": ticket["remaining"]}
        except BaseException:
            self.ticket = None
            raise

    def cancel(self):
        self.ticket = None
        self.session.pause("Operation cancelled; no automatic retry")
        return self.status()

    def ask(self, text, on_text=None):
        validate_message(text)
        self.ticket = None
        proposal = explicit_intent(text)
        source, warning, generation = "controller", None, None
        if proposal and proposal["command"] == "status":
            status = self.status()
            message = (f"Current status: {status['status']}; executed {status['experiment_count']} simulated experiments; "
                       f"consecutive IQ passes {status['consecutive_iq_passes']}/2. "
                       f"Completed stages: {', '.join(status['completed_stages']) or 'none'}.")
            proposal = None
        elif proposal:
            message = "I will show the operation scope and wait for confirmation. Simulation only."
        elif not hasattr(self.session.policy, "chat"):
            source = "rule_notice"
            message = "RULE mode uses no language model and cannot answer open-ended questions. Configure HF or connect to a model server. This is a controller message."
        else:
            source = "model"
            stream = ReplyStream(on_text) if on_text is not None else None
            context = self.status()
            context.update(latest_fit=(self.session.runner.observation or {}).get("fit_result", {}),
                           history_turns_shown=len(self.messages) // 2, history_turns_total=self.total_turns)
            try:
                options = ({"on_text": stream.feed} if stream is not None and
                           getattr(self.session.policy, "supports_streaming", False) else {})
                raw = self.session.policy.chat(self.messages + [{"role": "user", "content": text}], context, **options)
                message, proposal, warning = parse_reply(raw)
                if stream is not None:
                    if not stream.finish(message):
                        self.session.runner.emit({"event": "stream_reconciled", "preview_chars": len(stream.emitted),
                                                  "final_chars": len(message)})
                        warning = (warning + "；" if warning else "") + "Streaming preview differed in formatting; the complete final response was preserved"
                generation = getattr(self.session.policy, "last_generation", None)
                if generation and generation.get("hit_output_limit"):
                    warning = (warning + "；" if warning else "") + (
                        f"Generated {generation.get('generated_tokens', '?')} tokens, reaching the configured limit; "
                        "ask 'continue' or increase --chat-max-new-tokens (maximum 4096)")
            except (Exception, KeyboardInterrupt) as exc:
                self.session.runner.emit({"event": "chat_error", "error": str(exc) or "Cancelled",
                    "user": text, "partial_assistant": stream.emitted if stream else "",
                    "partial_only": True})
                self.session.save()
                raise  # Chat failure never changes calibration status or executes a fallback.
        self.session.runner.emit({"event": "chat", "user": text, "assistant": message, "proposal": proposal,
                                  "source": source, "generation": generation, "warning": warning})
        self.session.save()
        self.messages.extend([{"role": "user", "content": text}, {"role": "assistant", "content": message}])
        self.messages = self.messages[-24:]
        self.total_turns += 1
        return {"message": message, "proposal": proposal, "source": source, "warning": warning,
                "generation": generation, "status": self.status()}

    def clear_chat(self):
        self.ticket = None
        self.session.runner.emit({"event": "chat_clear"})
        self.session.save()
        self.messages, self.total_turns = [], 0
        return {"message": "Conversation context cleared; audit records, calibration state and budgets are unchanged."}

    def pause(self):
        self.ticket = None
        self.session.pause()
        return {"message": "Saved and paused; model process released.", "status": self.status()}

    def report(self, target=None):
        if target is not None and (not isinstance(target, str) or not target):
            raise ValueError("Invalid report directory")
        destination = Path(target) if target else self.session.directory / f"export-{self.session.journal.sequence:06d}"
        destination = self.session.export(destination)
        result = {"directory": str(destination), "status": self.status()}
        png = destination / "iq_report.png"
        if (destination / "plot_error.json").exists():
            result["plot_note"] = "Numerical report saved, but plotting failed. Inspect plot_error.json; report dependencies may be missing."
        elif png.is_file() and png.stat().st_size <= 1024 * 1024:
            import base64
            import hashlib
            data = png.read_bytes()
            result["preview"] = {"name": "iq_report.png", "sha256": hashlib.sha256(data).hexdigest(),
                                 "base64": base64.b64encode(data).decode("ascii")}
        else:
            result["plot_note"] = "No recorded IQ measurement; no image generated" if not png.exists() else "Image saved on server; exceeds inline preview size limit"
        return result

    def sessions(self):
        return list_sessions(self.home)

    def _switch(self, new):
        try:
            self.session.pause("Switching session")
        except BaseException:
            new.close()
            raise
        self.session.close()
        self.session = new
        self.ticket = None
        self._restore_chat()
        return self.info()

    def new(self, seed=None):
        settings = load_settings(self.home)
        if seed is not None:
            settings = Settings.from_dict(settings.to_dict() | {"seed": seed})
        return self._switch(CalibrationSession.create(self.home, settings))

    def resume(self, session_id):
        return self._switch(CalibrationSession.resume(self.home, session_id))

    def close(self):
        self.ticket = None
        try:
            if not self.session.broken:
                self.session.pause("Client disconnected / terminal closed")
        finally:
            self.session.close()
