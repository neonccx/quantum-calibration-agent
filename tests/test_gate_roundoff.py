import unittest
from qmagent.iq_report import gates_equivalent


class GateRoundoffTests(unittest.TestCase):
    def test_platform_roundoff_only(self):
        self.assertTrue(gates_equivalent({"passed": True, "threshold": -0.002713530624077161},
                                         {"passed": True, "threshold": -0.00271353062407716}))
        self.assertFalse(gates_equivalent({"passed": True}, {"passed": False}))
        self.assertFalse(gates_equivalent({"passed": True}, {"passed": 1}))
        self.assertFalse(gates_equivalent({"f0": .99}, {"f0": .98}))
        self.assertFalse(gates_equivalent(float("nan"), float("nan")))
