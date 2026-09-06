import json
import unittest
from qmagent.contracts import DEFAULT_STATE
from qmagent.protocol import parse_call
from qmagent.policies import HuggingFacePolicy
from qmagent.runtime import AgentRunner


class FitProtocolTests(unittest.TestCase):
    def setUp(self):
        self.context = AgentRunner(None, None).context()
        self.context["state"]["drive_frequency_hz"] = 5006625256.681527
        self.context["observation"] = dict(tool="sq.ramsey_df", current_parameters=dict(self.context["state"]),
            quality={"reliable": True}, fit_result={"frequency_correction_hz": -6119.177277957632, "t2_star_us": 27.682111422735215})
        self.raw = '<tool_call>'+json.dumps({"name": "calibration.step_from_fit", "arguments": {
            "next_tool": "sq.t1", "scan": {}, "reason": "Apply reliable Ramsey fit"}})+'</tool_call>'

    def test_explicit_opt_in_and_exact_calculation(self):
        with self.assertRaises(ValueError):
            parse_call(self.raw, self.context)
        action = parse_call(self.raw, self.context, allow_fit_tool=True)
        self.assertAlmostEqual(action["parameter_action"]["updates"]["drive_frequency_hz"], 5006619137.504249, places=5)
        controller = AgentRunner(None, None)
        controller.state = self.context["state"]
        controller.observation = self.context["observation"]
        with self.assertRaises(ValueError):
            controller._candidate(action)  # Missing prerequisites must still reject.
        controller.completed = {0, 1, 2, 3}
        controller._candidate(action)

    def test_hf_path_exposes_tool_and_records_raw_choice(self):
        policy = HuggingFacePolicy.__new__(HuggingFacePolicy)
        policy.fit_update_tool, policy.max_new_tokens, policy.last_generation = True, 512, None
        def generate(messages, limit, **kwargs):
            self.assertEqual(kwargs["tools"][-1]["function"]["name"], "calibration.step_from_fit")
            return self.raw
        policy._generate = generate
        action = json.loads(policy.decide(self.context))
        self.assertEqual(action["next_tool"], "sq.t1")
        self.assertEqual(policy.last_generation["native_call"], self.raw)

    def test_unreliable_and_terminal_rejected(self):
        self.context["observation"]["quality"]["reliable"] = False
        with self.assertRaises(ValueError):
            parse_call(self.raw, self.context, True)
        self.context["observation"]["quality"]["reliable"] = True
        with self.assertRaises(ValueError):
            parse_call(self.raw.replace("sq.t1", "FINISH"), self.context, True)
