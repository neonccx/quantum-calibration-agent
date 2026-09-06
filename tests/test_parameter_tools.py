import unittest
from qmagent.parameter_tools import calculate_fit_updates


class ParameterToolTests(unittest.TestCase):
    def test_signed_ramsey_arithmetic_and_stale_measurement(self):
        state = {"drive_frequency_hz": 5006625256.681527}
        observation = dict(tool="sq.ramsey_df", quality={"reliable": True}, current_parameters=dict(state),
                           fit_result={"frequency_correction_hz": -6119.177277957632, "t2_star_us": 27.682111422735215})
        value = calculate_fit_updates(observation, state)["updates"]["drive_frequency_hz"]
        self.assertAlmostEqual(value, 5006619137.504249, places=5)
        with self.assertRaisesRegex(ValueError, "stale"):
            calculate_fit_updates(observation, {"drive_frequency_hz": state["drive_frequency_hz"]+100})

    def test_reject_unreliable_and_compute_reset(self):
        observation = dict(tool="sq.t1", quality={"reliable": False}, fit_result={"t1_us": 25.})
        with self.assertRaises(ValueError):
            calculate_fit_updates(observation, {})
        observation["quality"]["reliable"] = True
        self.assertEqual(calculate_fit_updates(observation, {})["updates"]["relaxation_delay_us"], 125.)

    def test_zpa2d_fit_commits_bias_and_readout(self):
        observation = {"tool": "sq.s21_zpa2d", "quality": {"reliable": True},
                       "current_parameters": {"z_bias": 0.0, "readout_frequency_hz": 6.5e9},
                       "fit_result": {"sweet_spot_zpa": -0.037,
                                      "readout_frequency_hz": 6.4991e9}}
        result = calculate_fit_updates(observation, observation["current_parameters"])
        self.assertEqual(result["updates"], {"z_bias": -0.037,
                                             "readout_frequency_hz": 6.4991e9})
        self.assertEqual(result["version"], "0.2")
