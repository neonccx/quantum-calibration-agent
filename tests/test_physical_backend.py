import copy
import json
import unittest

import numpy as np

from qmagent.analysis_tools import analyze
from qmagent.contracts import DEFAULT_STATE
from qmagent.physical_backend import PhysicalSimulator
from qmagent.physics import coherence_time, complex_notch, iq_pointer
from qmagent.policies import RulePolicy
from qmagent.runtime import AgentRunner


class PhysicalTests(unittest.TestCase):
    def test_nominal_closed_loop(self):
        for seed in (100, 101, 20260903):
            result = AgentRunner(PhysicalSimulator(seed), RulePolicy()).run()
            self.assertEqual(result["status"], "accepted")
            self.assertGreaterEqual(result["consecutive_iq_passes"], 2)

    def test_raw_then_independent_analysis(self):
        backend = PhysicalSimulator(100)
        raw = backend.acquire("sq.s21", DEFAULT_STATE, {})
        self.assertNotIn("fit_result", raw)
        self.assertNotIn("_truth", raw)
        original = copy.deepcopy(raw)
        result = analyze(raw)
        self.assertEqual(raw, original)
        self.assertTrue(result["quality"]["reliable"])
        expected = backend._readout_frequency(DEFAULT_STATE["z_bias"])
        self.assertLess(abs(result["fit_result"]["frequency_hz"]-expected), 2e4)
        self.assertLess(abs(result["fit_result"]["cable_delay_s"]-backend._truth["delay"]), 1e-9)
        json.dumps(result, allow_nan=False)

    def test_checkpoint_replay(self):
        backend = PhysicalSimulator(50, profile="drift")
        checkpoint = backend.checkpoint()
        first = backend.measure("sq.s21", DEFAULT_STATE, {})
        backend.restore(checkpoint)
        self.assertEqual(first, backend.measure("sq.s21", DEFAULT_STATE, {}))

    def test_correlated_coherence_and_invalid_times(self):
        for seed in range(20):
            t = PhysicalSimulator(seed)._truth
            self.assertLessEqual(coherence_time(t["t1"], t["tphi"]), 2*t["t1"])
        with self.assertRaises(ValueError):
            coherence_time(-1, 2)

    def test_unreliable_data_does_not_advance(self):
        result = AgentRunner(PhysicalSimulator(100, noise_scale=30), RulePolicy(), max_tool_calls=2).run()
        self.assertNotEqual(result["status"], "accepted")
        self.assertEqual(result["completed_stages"], [])

    def test_ambiguous_spectroscopy(self):
        raw = PhysicalSimulator(100, profile="ambiguous").acquire("sq.spectroscopy", DEFAULT_STATE, {})
        result = analyze(raw)
        self.assertTrue(result["quality"]["ambiguous_peaks"])
        self.assertFalse(result["quality"]["reliable"])

    def test_acquisition_rejects_invalid_scan(self):
        with self.assertRaises(ValueError):
            PhysicalSimulator().acquire("sq.s21", DEFAULT_STATE, {"frequency_center_hz": 6e9})

    def test_pointer_limits(self):
        args = (6.5e9, -1e6, 6.5e9, 2e6, .5)
        ground = iq_pointer(1e-6, np.array([0.]), *args)[0]
        excited = iq_pointer(1e-6, np.array([2e-6]), *args)[0]
        decayed = iq_pointer(1e-6, np.array([1e-18]), *args)[0]
        self.assertLess(abs(ground-decayed), 1e-10)
        self.assertGreater(abs(ground-excited), .2)
