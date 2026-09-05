import importlib.util
from pathlib import Path
import unittest

from qmagent.contracts import decision
from qmagent.runtime import AgentRunner

spec = importlib.util.spec_from_file_location("score_policy_predictions", Path(__file__).resolve().parents[1]/"scripts/score_policy_predictions.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


class ControllerScoringTests(unittest.TestCase):
    def test_prerequisites_and_fail_closed(self):
        context = AgentRunner(None, None).context()
        self.assertTrue(scoring.executable(context, decision("sq.s21"))[0])
        self.assertFalse(scoring.executable(context, decision("sq.iqraw"))[0])
        self.assertFalse(scoring.executable(context, decision("FINISH"))[0])
        self.assertFalse(scoring.executable(context, None)[0])

    def test_budget_and_safe_escalation(self):
        context = AgentRunner(None, None).context()
        context["budget"]["remaining_experiments"] = 0
        self.assertFalse(scoring.executable(context, decision("sq.s21"))[0])
        self.assertTrue(scoring.executable(context, decision("ESCALATE_HARDWARE_REVIEW"))[0])
        context["budget"]["remaining_experiments"] = 10
        context["budget"]["tool_counts"]["sq.s21"] = context["budget"]["max_calls_per_tool"]
        self.assertFalse(scoring.executable(context, decision("sq.s21"))[0])
