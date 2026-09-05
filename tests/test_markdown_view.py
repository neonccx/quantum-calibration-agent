import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from qmagent.markdown_view import MarkdownReply, create_markdown_view
from qmagent.terminal import Terminal

HAS_RICH = importlib.util.find_spec("rich") is not None


class MarkdownIntegrationTests(unittest.TestCase):
    def test_non_tty_and_opt_out_keep_plain_stream(self):
        with patch("sys.stdout.isatty", return_value=False):
            self.assertIsNone(create_markdown_view())
        with patch("sys.stdout.isatty", return_value=True), patch.dict(os.environ, QM_AGENT_PLAIN="1"):
            self.assertIsNone(create_markdown_view())

    def test_terminal_routes_stream_and_closes_renderer(self):
        class View:
            def __init__(self):
                self.events = []
            def feed(self, text):
                self.events.append(text)
            def finish(self):
                self.events.append("CLOSED")
        class Service:
            def ask(self, text, on_text):
                on_text("# Heading\n")
                on_text("**bold**")
                return {"source": "model", "message": "# Heading\n**bold**", "warning": None, "proposal": None}
        view = View()
        output = []
        Terminal(Path("."), service=Service(), output_fn=output.append,
                 markdown_factory=lambda: view).ask("test")
        self.assertEqual(view.events, ["# Heading\n", "**bold**", "CLOSED"])
        self.assertFalse(any("Heading" in line for line in output))

    def test_cancel_closes_renderer_before_warning(self):
        events = []
        class View:
            def feed(self, text):
                events.append(text)
            def finish(self):
                events.append("CLOSED")
        class Service:
            def ask(self, text, on_text):
                on_text("partial")
                raise KeyboardInterrupt
        terminal = Terminal(Path("."), service=Service(), output_fn=events.append, markdown_factory=View)
        with self.assertRaises(KeyboardInterrupt):
            terminal.ask("test")
        self.assertLess(events.index("CLOSED"), next(i for i, text in enumerate(events) if "incomplete" in text))


@unittest.skipUnless(HAS_RICH, "Install optional terminal extra to test Markdown rendering")
class RichMarkdownTests(unittest.TestCase):
    def make_view(self, width=70, height=20, terminal=False):
        from rich.console import Console
        output = io.StringIO()
        console = Console(file=output, width=width, height=height, force_terminal=terminal,
                          color_system="standard" if terminal else None, _environ={"TERM": "xterm-256color"})
        return MarkdownReply(console, clock=lambda: 0), output

    def visible(self, view):
        return "\n".join("".join(segment.text for segment in line) for line in view.render_lines())

    def test_headings_bold_lists_table_and_code(self):
        view, _ = self.make_view()
        view.text = ('# Heading\n\n**Bold** and `inline_name`\n\n- first\n- second\n\n'
                     '| Name | Value |\n| --- | --- |\n| A | 42 |\n\n'
                     '```python\nx_name = "#literal"\n# comment\nvalue = 2 * 3\n```\n')
        visible = self.visible(view)
        self.assertIn("Heading", visible)
        self.assertNotIn("# Heading", visible)
        self.assertNotIn("**Bold**", visible)
        self.assertNotIn("```", visible)
        for text in ("inline_name", "first", "second", "Name", "42", "#literal", "# comment", "2 * 3"):
            self.assertIn(text, visible)
        self.assertTrue(any(s.style and s.style.bold for line in view.render_lines() for s in line if "Bold" in s.text))

    def test_long_live_viewport_shows_tail_within_terminal_height(self):
        from rich.segment import Segment
        view, _ = self.make_view(width=26, height=9)
        view.text = "\n\n".join(f"paragraph {i}" for i in range(60))
        lines = view.viewport().lines
        self.assertLessEqual(len(lines), 5)
        self.assertTrue(all(Segment.get_line_length(line) <= 26 for line in lines))
        self.assertIn("paragraph 59", "".join(s.text for line in lines for s in line))
        self.assertIn("paragraph 0", self.visible(view))  # Final output is not truncated.

    def test_no_background_badges_or_underlines(self):
        view, _ = self.make_view()
        view.text = '# Heading\n\n`inline_code` and [link](https://example.com)\n\n```python\nx = 2 * 3\n```'
        segments = [s for line in view.render_lines() for s in line]
        for segment in segments:
            if segment.style:
                self.assertIsNone(segment.style.color)
                self.assertIsNone(segment.style.bgcolor)
                self.assertFalse(segment.style.underline)
                self.assertFalse(segment.style.reverse)
        inline = next(s for s in segments if 'inline_code' in s.text)
        self.assertIsNone(inline.style.color)
        self.assertTrue(inline.style.bold)

    def test_final_text_reconciles_without_losing_body(self):
        view, _ = self.make_view(terminal=True)
        view.feed("preview  text")
        view.finish("final text\n\n```python\n    x = 1\n```")
        self.assertIn("final text", self.visible(view))
        self.assertNotIn("preview", self.visible(view))
        self.assertIn("    x = 1", self.visible(view))
        self.assertTrue(all(not s.text.endswith(" ") for line in view.render_lines() for s in line[-1:]))

    def test_split_markdown_live_refresh_and_final_cleanup(self):
        view, output = self.make_view(terminal=True)
        ticks = iter((0.0, 0.2, 0.4))
        view.clock = lambda: next(ticks)
        view.feed("# Heading\n\n**bo")
        first = output.getvalue()
        view.feed("ld**\n\n```python\nprint(1)")
        self.assertGreater(len(output.getvalue()), len(first))
        view.feed("\n```\n")
        view.finish()
        size = len(output.getvalue())
        view.finish()
        self.assertEqual(size, len(output.getvalue()))
        self.assertFalse(view.live.is_started)
        self.assertIn("\x1b[?25h", output.getvalue())  # Cursor restored.
        self.assertNotIn("**bold**", self.visible(view))
        self.assertNotIn("```", self.visible(view))

    def test_model_controls_escaped_and_links_not_osc(self):
        view, output = self.make_view(terminal=True)
        view.feed("[link](https://example.com)\n\n\x1b[2Jbad\x07")
        view.finish()
        self.assertIn("\\u001b", self.visible(view))
        self.assertNotIn("\x1b[2J", output.getvalue())
        self.assertNotIn("\x1b]8;", output.getvalue())


if __name__ == "__main__":
    unittest.main()
