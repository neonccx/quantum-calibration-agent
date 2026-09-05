import copy
import json
import unittest
from qmagent.protocol import public_context, parse_call, assistant_call, policy_messages
from qmagent.contracts import decision, DEFAULT_STATE
from qmagent.physical_backend import PhysicalSimulator

class ProtocolTests(unittest.TestCase):
    def test_roundtrip(self):
        action = decision("sq.s21", reason="Acquire")
        call = assistant_call(action)["tool_calls"][0]["function"]
        self.assertEqual(parse_call("<tool_call>"+json.dumps(call)+"</tool_call>"), action)

    def test_reject_extra_or_duplicate_calls(self):
        call = '<tool_call>{"name":"calibration.step","arguments":{}}</tool_call>'
        for text in (call+call, "Okay "+call, call.replace('"name":', '"name":"evil","name":')):
            with self.assertRaises(ValueError):
                parse_call(text)

    def test_public_view_is_idempotent_and_preserves_evidence(self):
        raw = {"observation": PhysicalSimulator().measure("sq.s21", DEFAULT_STATE, {})}
        original = copy.deepcopy(raw)
        view = public_context(raw)
        self.assertEqual(public_context(view), view)
        self.assertEqual(policy_messages(raw), policy_messages(view))
        self.assertEqual(raw, original)
        self.assertNotIn("measurement", view["observation"])
        self.assertEqual(view["observation"]["fit_result"], raw["observation"]["fit_result"])
