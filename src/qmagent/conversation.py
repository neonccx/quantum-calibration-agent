"""Chat routing: proposals are data, never executable model output."""

import json
import re


class ReplyStream:
    """Stream prose, withholding split action tags until the final reply is validated."""

    marker = "<qm_action>"

    def __init__(self, callback):
        self.callback = callback
        self.pending = self.emitted = ""
        self.hidden = False
        self.received = 0

    def feed(self, text):
        if not isinstance(text, str):
            raise ValueError("Invalid text delta")
        self.received += len(text)
        if self.received > 65536:
            raise ValueError("Model stream exceeds reply limit")
        if self.hidden:
            return
        self.pending += text
        if self.marker in self.pending:
            self.pending = self.pending.split(self.marker, 1)[0]
            self.hidden = True
        hold = 0
        if not self.hidden:
            for size in range(1, len(self.marker)):
                if self.pending.endswith(self.marker[:size]):
                    hold = size
        visible = self.pending[:-hold] if hold else self.pending
        visible = visible.rstrip()
        self.pending = self.pending[len(visible):]
        if not self.emitted:
            visible = visible.lstrip()
        self._emit(visible)

    def _emit(self, text):
        if text:
            self.emitted += text
            self.callback(text)

    def finish(self, message):
        # The final envelope is authoritative. A tokenizer may revise whitespace
        # in its full decode; let the UI reconcile instead of losing the reply.
        if not message.startswith(self.emitted):
            return False
        self._emit(message[len(self.emitted):])
        return True


def validate_message(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 8000:
        raise ValueError("Question must contain 1-8000 characters")
    if re.search(r"(?i)(密码|password|api[_ -]?key|access[_ -]?token|验证码)\s*(?:是|[:=：])\s*\S+", text):
        raise ValueError("Do not enter credentials here. Content was not recorded or sent to the model. Use system SSH for login.")


def validate_proposal(value):
    if not isinstance(value, dict) or set(value) - {"command", "limit"}:
        raise ValueError("Invalid proposed command")
    command = value.get("command")
    if command not in {"plan", "step", "run", "status", "report"}:
        raise ValueError("Unlisted proposed command")
    if "limit" in value and (command != "run" or type(value["limit"]) is not int or not 1 <= value["limit"] <= 1000):
        raise ValueError("Invalid proposed run limit")
    return dict(value)


def parse_reply(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 65536:
        raise ValueError("Model returned an empty/oversized reply")
    marker = "<qm_action>"
    if marker not in text:
        return text.strip(), None, None
    match = re.fullmatch(r"(.*?)\s*<qm_action>(.*?)</qm_action>\s*", text, re.S)
    if not match or text.count(marker) != 1:
        return text.strip(), None, "Malformed operation tag; no tool called"
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate proposal key")
            value[key] = item
        return value
    try:
        proposal = validate_proposal(json.loads(match[2], object_pairs_hook=unique))
        return match[1].strip() or "I propose the following operation, subject to your confirmation.", proposal, None
    except (ValueError, TypeError, RecursionError):
        return match[1].strip() or "The model proposed an invalid operation.", None, "Proposed operation is outside the allowlist; not executed"


def explicit_intent(text):
    """Fast paths for common requests; other sentences go to the actual LLM."""
    normalized = re.sub(r"[\s，。？！?!]", "", text)
    if normalized.lower() in {"run", "plan", "step", "status", "report"}:
        return {"command": normalized.lower()}
    if normalized in {"状态", "当前状态", "现在校准到哪了", "现在校准到哪一步了", "校准到哪一步了", "看看当前状态"}:
        return {"command": "status"}
    if re.fullmatch(r"(?:请|帮我|请帮我|我想|麻烦你)?(?:开始|继续)(?:进行)?校准(?:这个比特|这个量子比特|一下)?", normalized):
        return {"command": "run"}
    if normalized in {"下一步", "执行下一步", "帮我执行下一步", "运行一步", "执行一步"}:
        return {"command": "step"}
    if normalized in {"看看下一步计划", "给我下一步计划", "帮我制定下一步计划"}:
        return {"command": "plan"}
    return None
