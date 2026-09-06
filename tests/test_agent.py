import copy
import json
import unittest

import numpy as np

from qmagent.contracts import (ContractError, DEFAULT_STATE, TOOLS, decision,
                               parse_decision, validate_scan)
from qmagent.policies import RulePolicy, model_context
from qmagent.runtime import AgentRunner, invalidate, stage_passed
from qmagent.simulator import AnalyticSimulator, iq_gate


class FixedPolicy:
    def __init__(self, value):
        self.value = value

    def decide(self, context):
        return self.value


class ContractTests(unittest.TestCase):
    def test_roundtrip(self):
        value = decision("sq.s21", scan={"frequency_span_hz": 40e6})
        self.assertEqual(parse_decision(json.dumps(value)), value)

    def test_unknown_tool(self):
        with self.assertRaises(ContractError):
            parse_decision(decision("os.system"))

    def test_reject_invalid_numbers(self):
        for value in (float("nan"), float("inf"), True, "0.5", -1, 999):
            with self.subTest(value=value), self.assertRaises(ContractError):
                parse_decision(decision("sq.iqraw", {"readout_amplitude": value}))

    def test_unknown_update_and_scan(self):
        with self.assertRaises(ContractError):
            parse_decision(decision("sq.s21", {"shell": 1}))
        with self.assertRaises(ContractError):
            parse_decision(decision("sq.t1", scan={"amplitude_max": 0.8}))

    def test_no_terminal_mutation(self):
        with self.assertRaises(ContractError):
            parse_decision(decision("FINISH", {"pi_amplitude": 0.2}))

    def test_duplicate_and_prose_json(self):
        for text in ('{"next_tool":"sq.s21","next_tool":"FINISH"}',
                     '```json\n{}\n```', 'null', '[]'):
            with self.subTest(text=text), self.assertRaises(ContractError):
                parse_decision(text)

    def test_old_policy_schema_not_silently_accepted(self):
        with self.assertRaises(ContractError):
            parse_decision({"next_tool": "FINISH", "parameter_action": {"action": "finish_calibration"},
                            "final_iq_acceptance": {"passed": True}})

    def test_integer_shots(self):
        with self.assertRaises(ContractError):
            parse_decision(decision("sq.iqraw", scan={"shots": 1024.5}))

    def test_sweep_edges_and_ramsey_bandwidth(self):
        with self.assertRaises(ContractError):
            validate_scan("sq.s21", DEFAULT_STATE, {"frequency_center_hz": 6e9, "frequency_span_hz": 40e6})
        with self.assertRaises(ContractError):
            validate_scan("sq.ramsey_df", DEFAULT_STATE, {"delay_max_us": 100})


class RuntimeTests(unittest.TestCase):
    def test_rule_full_loop(self):
        result = AgentRunner(AnalyticSimulator(20260903), RulePolicy()).run()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["completed_stages"], list(TOOLS[:-1]))
        self.assertGreaterEqual(result["consecutive_iq_passes"], 2)
        self.assertTrue(result["final_iq_gate"]["passed"])

    def test_reproducible(self):
        a = AgentRunner(AnalyticSimulator(123), RulePolicy()).run()
        b = AgentRunner(AnalyticSimulator(123), RulePolicy()).run()
        self.assertEqual(a, b)

    def test_unknown_and_skip_and_finish_fail_closed(self):
        for tool in ("shell", "sq.iqraw", "sq.piamp", "FINISH"):
            result = AgentRunner(AnalyticSimulator(), FixedPolicy(decision(tool))).run()
            with self.subTest(tool=tool):
                self.assertEqual(result["status"], "invalid_action")
                self.assertEqual(result["experiment_count"], 0)
                self.assertEqual(result["final_state"], DEFAULT_STATE)

    def test_early_finish_after_one_iq(self):
        class EarlyPolicy(RulePolicy):
            def decide(self, context):
                if context["consecutive_iq_passes"] == 1:
                    return decision("FINISH")
                return super().decide(context)
        result = AgentRunner(AnalyticSimulator(), EarlyPolicy()).run()
        self.assertEqual(result["status"], "invalid_action")
        self.assertEqual(result["consecutive_iq_passes"], 1)

    def test_iq_parameter_change_resets_streak(self):
        class ChangePolicy(RulePolicy):
            changed = False
            def decide(self, context):
                if context["consecutive_iq_passes"] == 1 and not self.changed:
                    self.changed = True
                    return decision("sq.iqraw", {"readout_amplitude": context["state"]["readout_amplitude"] + 0.00001})
                return super().decide(context)
        result = AgentRunner(AnalyticSimulator(), ChangePolicy()).run()
        iq = [event for event in result["events"] if event["event"] == "experiment" and event["tool"] == "sq.iqraw"]
        self.assertGreaterEqual(len(iq), 3)
        self.assertLessEqual(iq[1]["consecutive_iq_passes"], 1)

    def test_total_budget(self):
        result = AgentRunner(AnalyticSimulator(), RulePolicy(), max_steps=1).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["experiment_count"], 1)

    def test_tool_budget(self):
        result = AgentRunner(AnalyticSimulator(), FixedPolicy(decision("sq.s21")), max_tool_calls=2).run()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["experiment_count"], 2)

    def test_escalation(self):
        result = AgentRunner(AnalyticSimulator(), FixedPolicy(decision("ESCALATE_HARDWARE_REVIEW"))).run()
        self.assertEqual(result["status"], "escalated")

    def test_exception_rolls_back_state_and_rng(self):
        class BrokenBackend(AnalyticSimulator):
            def measure(self, tool, state, scan):
                self.rng.random(10)
                raise RuntimeError("simulated failure")
        backend = BrokenBackend()
        before = backend.checkpoint()
        result = AgentRunner(backend, FixedPolicy(decision("sq.s21", {"readout_frequency_hz": 6.51e9}))).run()
        self.assertEqual(result["status"], "tool_error")
        self.assertEqual(result["experiment_count"], 1)
        self.assertEqual(result["final_state"], DEFAULT_STATE)
        self.assertEqual(before, backend.checkpoint())
        self.assertIn("rollback", [event["event"] for event in result["events"]])

    def test_policy_cannot_mutate_controller_or_see_truth(self):
        class MutatingPolicy:
            def decide(self, context):
                self.context = copy.deepcopy(context)
                context["state"]["pi_amplitude"] = 100
                return decision("ESCALATE_HARDWARE_REVIEW")
        policy = MutatingPolicy()
        result = AgentRunner(AnalyticSimulator(), policy).run()
        self.assertEqual(result["final_state"], DEFAULT_STATE)
        self.assertNotIn("truth", json.dumps(policy.context))

    def test_nonfinite_observation_rejected(self):
        class NonfiniteBackend(AnalyticSimulator):
            def measure(self, tool, state, scan):
                obs = super().measure(tool, state, scan)
                obs["measurement"]["i"][0] = float("nan")
                return obs
        result = AgentRunner(NonfiniteBackend(), RulePolicy()).run()
        self.assertEqual(result["status"], "tool_error")

    def test_model_metrics_cannot_override_gate(self):
        class LyingBackend(AnalyticSimulator):
            def _iq(self, obs, state, scan):
                super()._iq(obs, state, scan)
                obs["measurement"]["i"] = [0.0] * len(obs["measurement"]["i"])
                obs["measurement"]["q"] = [0.0] * len(obs["measurement"]["q"])
                obs["acceptance"]["passed"] = True
        result = AgentRunner(LyingBackend(), RulePolicy()).run()
        self.assertNotEqual(result["status"], "accepted")
        self.assertFalse(result["final_iq_gate"]["passed"])

    def test_invalidation(self):
        changed = DEFAULT_STATE | {"readout_frequency_hz": 6.51e9}
        self.assertEqual(invalidate(set(range(len(TOOLS)-1)), DEFAULT_STATE, changed), {0})
        changed = DEFAULT_STATE | {"drive_frequency_hz": 5.01e9}
        self.assertEqual(invalidate(set(range(len(TOOLS)-1)), DEFAULT_STATE, changed), {0, 1})

    def test_rabi_requires_repeat_even_if_initial_guess_is_correct(self):
        obs = {"tool": "sq.piamp", "quality": {"reliable": True}, "round_in_experiment": 1,
               "current_parameters": dict(DEFAULT_STATE), "fit_result": {"pi_amplitude": 0.25}}
        self.assertFalse(stage_passed(obs, DEFAULT_STATE))
        obs["round_in_experiment"] = 2
        self.assertTrue(stage_passed(obs, DEFAULT_STATE))


class MeasurementTests(unittest.TestCase):
    def test_iq_holdout_not_training_accuracy(self):
        # Even shots are perfectly separable; odd shots are deliberately reversed.
        p0 = [-1.0, 1.0] * 128
        p1 = [1.0, -1.0] * 128
        gate = iq_gate({"i": p0 + p1, "q": [0.0] * 512, "prepared_state": [0] * 256 + [1] * 256})
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["metrics"]["assignment_fidelity"], 0)

    def test_observation_has_no_oracle_and_measure_does_not_update_state(self):
        state = dict(DEFAULT_STATE)
        backend = AnalyticSimulator()
        for tool in TOOLS:
            observation = backend.measure(tool, state, {})
            serialized = json.dumps(observation, allow_nan=False)
            for forbidden in ("ground_truth", "recommended_update", "optimal_readout", "_truth"):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(state, DEFAULT_STATE)

    def test_model_view_discloses_sampling_and_keeps_full_log(self):
        observation = AnalyticSimulator().measure("sq.iqraw", DEFAULT_STATE, {})
        before = copy.deepcopy(observation)
        view = model_context({"observation": observation})
        self.assertNotIn("measurement", view["observation"])
        data = view["observation"]["raw_reference"]
        self.assertEqual(data["array_lengths"]["i"], 2048)
        self.assertIn("sha256", data)
        self.assertEqual(observation, before)

    def test_bad_labels_fail(self):
        with self.assertRaises(ValueError):
            iq_gate({"i": [0] * 256, "q": [0] * 256, "prepared_state": [2] * 256})

    def test_negative_noise_rejected(self):
        with self.assertRaises(ValueError):
            AnalyticSimulator(noise_scale=-1)


if __name__ == "__main__":
    unittest.main()
