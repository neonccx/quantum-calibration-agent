"""Small, optional terminal design layer; no model or experiment dependencies."""

import json
import os
import sys
import re
from contextlib import nullcontext

EDITION = "STUDIO 03"
PALETTES = {
    "amber": {"accent": "yellow", "success": "green", "warning": "yellow", "error": "red"},
    "cyan": {"accent": "cyan", "success": "green", "warning": "yellow", "error": "red"},
    "mono": {"accent": "bold", "success": "bold", "warning": "bold", "error": "bold"},
}


def create_design():
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb" or os.environ.get("QM_AGENT_PLAIN") == "1":
        return None
    try:
        from rich.console import Console
        return TerminalDesign(Console(highlight=False))
    except ImportError:
        return None


class TerminalDesign:
    def __init__(self, console):
        self.console = console
        self.ascii = os.environ.get("QM_AGENT_ASCII") == "1"
        self.set_theme(os.environ.get("QM_AGENT_THEME", "amber"))
        self.input_session = None
        self.context = "SIMULATION"

    def set_theme(self, name):
        if name not in PALETTES:
            raise ValueError("Available themes: amber / cyan / mono")
        self.theme = name
        self.styles = PALETTES["mono" if os.environ.get("NO_COLOR") else name]
        self.accent = self.styles["accent"]

    def text(self, value, style=""):
        from rich.text import Text
        from .terminal import safe_text
        value = safe_text(value)
        if self.ascii:
            for a, b in {"·": "/", "→": "->", "─": "-", "❯": ">", "π": "pi", "µ": "u"}.items():
                value = value.replace(a, b)
        return Text(value, style=style)

    def banner(self, info, remote=False):
        from rich.table import Table
        settings = info["settings"]
        model = "RULE · deterministic policy, no LLM" if settings["policy"] == "rule" else str(settings["model"]).rstrip("/").split("/")[-1]
        if settings.get("adapter"):
            model += " + LoRA"
        self.context = f"{'SSH' if remote else 'LOCAL'} · {model} · SIMULATION"
        self.console.print()
        self.console.print(self.text(f"  QM / CALIBRATION                         {EDITION}", "bold " + self.accent))
        self.console.print(self.text("  Single-qubit calibration laboratory"))
        self.console.print()
        if self.console.width >= 72 and not self.ascii:
            # An identity mark, not a pretend plot or a progress indicator.
            rows = [
                ("      ·  ───  ·", model),
                ("   ·     ╱     ·", "Ask a question or start an experiment."),
                ("  │    ●───○    │", "/plan   Preview the next action"),
                ("   ·     ╲     ·", "/run    Confirm and run simulation"),
                ("      ·  ───  ·", "/report View IQ discrimination"),
            ]
            table = Table.grid(padding=(0, 3))
            table.add_column(width=24)
            table.add_column(overflow="fold")
            for mark, content in rows:
                table.add_row(self.text(mark, self.accent), self.text(content))
            self.console.print(table)
        else:
            self.console.print(self.text("  " + model, "bold"))
            self.console.print(self.text("  /plan Preview   /run Simulate   /report Plot"))
        self.console.print()
        self.console.print(self.text(f"  {'SSH REMOTE' if remote else 'LOCAL'} · SIMULATION · v{info['version']} · No real instruments connected"))
        self.console.print(self.text(f"  Session {info['session_id']}"))
        self.console.print(self.text("  /help Commands   /model Model   /theme Theme   /quit Exit"))

    def section(self, title):
        self.console.print(self.text("\n  " + title, "bold " + self.accent))

    def values(self, data):
        """Readable structured values, with full values retained in scrollback."""
        from rich.table import Table
        table = Table.grid(padding=(0, 3))
        table.add_column(no_wrap=False)
        table.add_column(overflow="fold", justify="right")
        for key, value in data.items():
            if isinstance(value, dict):
                self.console.print(self.text("  " + str(key), "bold"))
                self.values(value)
                continue
            formatted = f"{value:.8g}" if isinstance(value, float) else json.dumps(value, ensure_ascii=False)
            if isinstance(value, str):
                formatted = value
            table.add_row(self.text("  " + str(key)), self.text(formatted))
        if table.row_count:
            self.console.print(table)

    def plan(self, plan):
        self.section("Pending confirmation · not executed")
        self.console.print(self.text("  " + plan["next_tool"], "bold"))
        self.console.print(self.text("  " + plan["reason"]))
        for group, label in (("updates", "Parameter updates"), ("scan", "Scan settings")):
            if plan["parameter_action"][group]:
                self.console.print(self.text("  " + label, "bold"))
                self.values(plan["parameter_action"][group])

    def experiment(self, data):
        reliable = data["quality"]["reliable"]
        self.section(f"{data['step']:02d} / {data['tool']}  ·  {'reliable fit' if reliable else 'unreliable fit'}")
        self.values(data["fit_result"])
        self.console.print(self.text(f"  Consecutive IQ passes {data['iq_passes']}/2"))

    def help(self, source):
        from rich.table import Table
        self.section("Command reference")
        lines = source.splitlines()
        groups = (("Conversation & interface", {"/help", "/theme", "/model", "/clear"}),
                  ("Experiments & evidence", {"/status", "/plan", "/step", "/run", "/why", "/history", "/report"}),
                  ("Sessions", {"/sessions", "/new", "/resume", "/pause", "/quit"}))
        for title, names in groups:
            self.console.print(self.text("  " + title, "bold"))
            table = Table.grid(padding=(0, 2))
            table.add_column()
            table.add_column()
            for line in lines:
                if not line or line.split()[0] not in names:
                    continue
                match = re.match(r"^(/\S+(?:\s+\[[^\]]+\]|\s+<[^>]+>)?)\s+(.+)$", line)
                if match:
                    table.add_row(self.text("  " + match[1], self.accent), self.text(match[2]))
            self.console.print(table)
        for line in lines:
            if line and not line.startswith("/"):
                self.console.print(self.text("  " + line))

    def waiting(self, label):
        if not self.console.is_terminal:
            return nullcontext()
        return self.console.status(self.text("  " + label + " · Ctrl-C cancel"),
                                   spinner="line" if self.ascii else "dots", spinner_style=self.accent)

    def status(self, result):
        self.section("Controller state · SIMULATION")
        self.console.print(self.text(
            f"{result['status']}  ·  Experiments {result['experiment_count']}/{result['max_steps']}"
            f"  ·  Consecutive IQ {result['consecutive_iq_passes']}/2"))
        self.console.print(self.text("Stages  " + (" → ".join(result["completed_stages"]) or "none completed")))
        self.values(result["final_state"])
        gate = result.get("final_iq_gate")
        if gate:
            self.section("Independent IQ acceptance")
            self.values(gate)
        if result.get("reason"):
            self.console.print(self.text("Reason  " + result["reason"]))

    def prompt(self):
        # Keep readline in charge of input, history, and Ctrl-C. No ANSI in prompt.
        from rich.rule import Rule
        self.console.print(Rule(style=self.accent, characters="-" if self.ascii else "─"))
        return "> " if self.ascii else "❯ "

    def read_input(self, source):
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.styles import Style
        except ImportError:
            return input(self.prompt())
        if self.input_session is None:
            commands = [line.split()[0] for line in source.splitlines() if line.startswith("/")]
            self.input_session = PromptSession(completer=WordCompleter(commands, sentence=True),
                                               complete_while_typing=True, reserve_space_for_menu=2)
        style = Style.from_dict({"prompt": "bold", "bottom-toolbar": "bg:default fg:default noreverse",
                                 "completion-menu": "bg:default fg:default noreverse",
                                 "completion-menu.completion": "bg:default fg:default noreverse",
                                 "completion-menu.completion.current": "bg:default fg:default bold noreverse",
                                 "completion-menu.scrollbar": "bg:default",
                                 "completion-menu.scrollbar.background": "bg:default",
                                 "completion-menu.scrollbar.button": "bg:default"})
        return self.input_session.prompt([("class:prompt", self.prompt())], style=style,
            bottom_toolbar=[("", " /help Commands  ·  Tab Complete  ·  Ctrl-C Cancel  ·  /quit Exit ")])
