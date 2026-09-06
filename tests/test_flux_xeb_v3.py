import copy
import json
import unittest

from qmagent.analysis_tools import analyze
from qmagent.contracts import ContractError, DEFAULT_STATE, TOOLS, decision, validate_observation
from qmagent.physical_backend import PhysicalSimulator
from qmagent.policies import RulePolicy
from qmagent.protocol import public_context
from qmagent.runtime import AgentRunner


class FluxSweepTests(unittest.TestCase):
    def test_zpa2d_shape_fit_and_truth_isolation(self):
        backend = PhysicalSimulator(2026090607)
        raw = backend.acquire("sq.s21_zpa2d", DEFAULT_STATE, {})
        self.assertEqual(len(raw["measurement"]["i"]), 41)
        self.assertEqual(len(raw["measurement"]["i"][0]), 241)
        fitted = raw | analyze(raw)
        validate_observation(fitted, "sq.s21_zpa2d", DEFAULT_STATE)
        self.assertTrue(fitted["quality"]["reliable"])
        self.assertLess(abs(fitted["fit_result"]["sweet_spot_zpa"]-backend._truth["sweet_zpa"]), .03)
        view = public_context({"observation": fitted})
        encoded = json.dumps(view)
        self.assertNotIn("measurement", view["observation"])
        self.assertNotIn("sweet_zpa", encoded)
        self.assertNotIn("junction_asymmetry", encoded)
        self.assertEqual(view["observation"]["raw_reference"]["array_shapes"]["i"], [41, 241])

    def test_spectroscopy_uses_committed_flux_bias(self):
        backend = PhysicalSimulator(2026090608, noise_scale=0)
        f0 = backend._qubit_frequency(DEFAULT_STATE["z_bias"])
        first = dict(DEFAULT_STATE, drive_frequency_hz=f0)
        a = backend.measure("sq.spectroscopy", first, {"frequency_center_hz": f0})
        shifted = dict(DEFAULT_STATE, z_bias=.05)
        f1 = backend._qubit_frequency(shifted["z_bias"])
        shifted["drive_frequency_hz"] = f1
        b = backend.measure("sq.spectroscopy", shifted, {"frequency_center_hz": f1})
        self.assertGreater(abs(a["fit_result"]["frequency_hz"]-b["fit_result"]["frequency_hz"]), 5e6)

    def test_rejects_ragged_or_mismatched_grid(self):
        backend = PhysicalSimulator(4)
        obs = backend.measure("sq.s21_zpa2d", DEFAULT_STATE, {})
        obs["measurement"]["q"].pop()
        with self.assertRaisesRegex(ContractError, "different shapes"):
            validate_observation(obs, "sq.s21_zpa2d", DEFAULT_STATE)


class XebProxyTests(unittest.TestCase):
    def test_xeb_decay_has_explicit_proxy_metadata_and_threshold(self):
        backend = PhysicalSimulator(2026090609, noise_scale=0)
        # Isolate analysis from earlier calibration stages while using exact hidden calibration.
        state = dict(DEFAULT_STATE, z_bias=backend._truth["sweet_zpa"],
                     drive_frequency_hz=backend._qubit_frequency(backend._truth["sweet_zpa"]),
                     pi_amplitude=backend._truth["pi"], pi_over_2_amplitude=backend._truth["pi"]/2)
        obs = backend.measure("sq.xeb", state, {})
        self.assertTrue(obs["acquisition"]["proxy"])
        self.assertTrue(obs["quality"]["reliable"])
        self.assertTrue(obs["fit_result"]["passed"])
        self.assertAlmostEqual(obs["fit_result"]["error_per_cycle"],
                               1-obs["fit_result"]["per_cycle_fidelity"])

    def test_xeb_below_threshold_cannot_complete(self):
        backend = PhysicalSimulator(2026090610, noise_scale=0)
        backend._truth["xeb_cycle_fidelity"] = .975
        state = dict(DEFAULT_STATE, z_bias=backend._truth["sweet_zpa"],
                     drive_frequency_hz=backend._qubit_frequency(backend._truth["sweet_zpa"]),
                     pi_amplitude=backend._truth["pi"], pi_over_2_amplitude=backend._truth["pi"]/2)
        obs = backend.measure("sq.xeb", state, {"max_depth": 128, "circuits_per_depth": 256})
        self.assertTrue(obs["quality"]["reliable"])
        self.assertFalse(obs["fit_result"]["passed"])

    def test_xeb_prerequisites_are_enforced(self):
        runner = AgentRunner(PhysicalSimulator(1), RulePolicy())
        runner.completed = set(range(6))
        with self.assertRaisesRegex(ContractError, "Prerequisite"):
            runner._candidate(decision("sq.xeb"))


class CheckpointV3Tests(unittest.TestCase):
    def test_v1_checkpoint_migrates_and_requires_new_stages(self):
        backend = PhysicalSimulator(33)
        runner = AgentRunner(backend, RulePolicy())
        snapshot = runner.snapshot()
        snapshot["version"] = 1
        snapshot["state"].pop("z_bias")
        snapshot["completed"] = list(range(6))
        restored = AgentRunner(PhysicalSimulator(33), RulePolicy())
        restored.restore(copy.deepcopy(snapshot))
        self.assertEqual(restored.state["z_bias"], 0.0)
        self.assertNotIn(1, restored.completed)
        self.assertNotIn(7, restored.completed)

    def test_rule_policy_full_nine_tool_flow(self):
        result = AgentRunner(PhysicalSimulator(2026090611), RulePolicy(), max_steps=40).run()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["completed_stages"], list(TOOLS[:-1]))
        self.assertEqual(result["tool_counts"]["sq.s21_zpa2d"], 1)
        self.assertGreaterEqual(result["tool_counts"]["sq.xeb"], 1)


if __name__ == "__main__":
    unittest.main()
