import importlib.util
import io
import unittest
import os
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("rich"), "Optional terminal dependency")
class DesignTests(unittest.TestCase):
    def test_banner_literal_data_and_narrow_width(self):
        from rich.console import Console
        from qmagent.terminal_design import TerminalDesign
        from rich.cells import cell_len
        for width in (32, 60, 80, 120):
            output = io.StringIO()
            design = TerminalDesign(Console(file=output, width=width, color_system=None))
            design.banner({"version": "0.3", "session_id": "[red]session", "directory": "/tmp/example",
                           "settings": {"policy": "rule"}})
            rendered = output.getvalue()
            self.assertIn("SIMULATION", rendered)
            self.assertIn("[red]session", rendered)
            self.assertNotIn("\x1b", rendered)
            self.assertIn("STUDIO 03", rendered)
            self.assertTrue(all(cell_len(line) <= width for line in rendered.splitlines()))

    def test_theme_ascii_and_help_are_safe(self):
        from rich.console import Console
        from qmagent.terminal_design import TerminalDesign
        from qmagent.terminal import HELP
        output = io.StringIO()
        with patch.dict(os.environ, {"NO_COLOR": "1", "QM_AGENT_ASCII": "1"}):
            design = TerminalDesign(Console(file=output, width=60, color_system=None))
            design.set_theme("cyan")
            self.assertEqual(design.accent, "bold")
            self.assertEqual(design.prompt(), "> ")
            self.assertEqual(design.text("─ π →").plain, "- pi ->")
            design.help(HELP)
        self.assertIn("/theme", output.getvalue())
        with self.assertRaises(ValueError):
            design.set_theme("[red]bad")

    def test_plan_and_experiment_keep_evidence(self):
        from rich.console import Console
        from qmagent.terminal_design import TerminalDesign
        output = io.StringIO()
        design = TerminalDesign(Console(file=output, width=80, color_system=None))
        design.plan({"next_tool": "sq.s21", "reason": "[red]literal",
                     "parameter_action": {"updates": {}, "scan": {"frequency_span_hz": 10000000}}})
        design.experiment({"step": 1, "tool": "sq.s21", "quality": {"reliable": False},
                           "fit_result": {"r2": 0.1}, "iq_passes": 0})
        self.assertIn("not executed", output.getvalue())
        self.assertIn("unreliable fit", output.getvalue())
        self.assertIn("[red]literal", output.getvalue())

    def test_status_displays_controller_counts(self):
        from rich.console import Console
        from qmagent.terminal_design import TerminalDesign
        output = io.StringIO()
        design = TerminalDesign(Console(file=output, width=80, color_system=None))
        design.status({"status": "active", "experiment_count": 3, "max_steps": 30,
                       "consecutive_iq_passes": 0, "completed_stages": ["s21"],
                       "final_state": {"frequency": 5.0}, "final_iq_gate": None, "reason": None})
        self.assertIn("3/30", output.getvalue())
        self.assertIn("0/2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
