"""Optional Rich UI. No model imports; protocol and redirected output stay plain."""

import os
import sys
import time


def create_markdown_view():
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb" or os.environ.get("QM_AGENT_PLAIN") == "1":
        return None
    try:
        from rich.console import Console
        return MarkdownReply(Console(highlight=False))
    except ImportError:
        return None


class RenderedLines:
    def __init__(self, lines):
        self.lines = lines

    def __rich_console__(self, console, options):
        from rich.segment import Segment
        for line in self.lines:
            yield from line
            yield Segment.line()


class MarkdownReply:
    """Re-render the current Markdown prefix, not each token as a separate document.

    A bounded trailing viewport avoids Rich Live's full-screen overflow problem.
    On completion it is erased and the complete document enters scrollback once.
    """

    def __init__(self, console, clock=time.monotonic):
        self.console, self.clock = console, clock
        self.text = ""
        self.live = None
        self.last_refresh = float("-inf")
        self.closed = False

    def render_lines(self):
        from rich.markdown import Markdown
        from rich.segment import Segment
        from rich.style import Style
        from rich.theme import Theme
        # Inline code inherits the terminal's readable foreground, without a
        # cyan-on-black badge. Never paint a background over the user's theme.
        with self.console.use_theme(Theme({"markdown.code": "bold", "markdown.code_block": "none",
                                           "markdown.link": "bold", "markdown.link_url": "none"})):
            lines = self.console.render_lines(Markdown(self.text, hyperlinks=False, code_theme="monokai"),
                                              self.console.options, pad=False)
        # Do not turn model-supplied URLs into terminal OSC hyperlinks.
        lines = [[Segment(segment.text, (segment.style.without_color.update_link(None) +
                    Style(underline=False, underline2=False, reverse=False, conceal=False))
                    if segment.style else None)
                 for segment in line if not segment.control] for line in lines]
        # Rich pads code/paragraphs to full width. Avoid wrap-at-last-column
        # artifacts in terminal emulators; keep leading spaces and code intact.
        for line in lines:
            while line:
                last = line[-1]
                text = last.text.rstrip(" \t")
                if text:
                    line[-1] = Segment(text, last.style)
                    break
                line.pop()
        while lines and not lines[-1]:
            lines.pop()
        return lines

    def viewport(self):
        from rich.segment import Segment
        lines = self.render_lines()
        height = max(1, min(8, self.console.height - 4))
        if len(lines) > height and height > 1:
            hint = Segment.adjust_line_length([Segment("... Generating: latest text (full response retained on completion)")],
                                              self.console.width, pad=False)
            lines = [hint] + lines[-(height - 1):]
        else:
            lines = lines[-height:]
        return RenderedLines(lines)

    def feed(self, text):
        if self.closed:
            raise RuntimeError("Markdown reply already closed")
        # Same terminal-control escaping as plain output, before Markdown parsing.
        from .terminal import safe_text
        self.text += safe_text(text)
        now = self.clock()
        if self.live is None:
            from rich.live import Live
            from rich.text import Text
            self.console.print(Text("Nanbeige · Explanation (not an execution record):", style="bold"))
            self.live = Live(self.viewport(), console=self.console, auto_refresh=False,
                             transient=True, vertical_overflow="crop")
            self.live.start(refresh=True)
            self.last_refresh = now
        elif now - self.last_refresh >= 0.08:
            self.live.update(self.viewport(), refresh=True)
            self.last_refresh = now

    def finish(self, final_text=None):
        if self.closed:
            return
        self.closed = True
        if final_text is not None:
            from .terminal import safe_text
            self.text = safe_text(final_text)
        try:
            if self.live is not None:
                self.live.stop()  # Restore cursor/stdout even after cancellation.
        finally:
            if self.text:
                self.console.print(RenderedLines(self.render_lines()), end="")
