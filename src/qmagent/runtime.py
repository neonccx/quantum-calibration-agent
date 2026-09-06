"""Bounded state machine. Only this controller can commit parameter changes."""

from __future__ import annotations

import copy
import math
import json
from collections import Counter

from .contracts import (CALIBRATION_STAGES, ContractError, DEFAULT_STATE, IQ_THRESHOLDS, XEB_THRESHOLDS,
                        SCAN_KEYS, SCAN_LIMITS, STATE_LIMITS, TOOLS, parse_decision,
                        validate_observation, validate_scan)
from .simulator import iq_gate


def _close(a: float, b: float, tolerance: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def stage_passed(observation: dict, candidate: dict) -> bool:
    """Quality AND consistency with the state to be committed, not just a good fit."""
    if not observation.get("quality", {}).get("reliable", False):
        return False
    tool, fit = observation["tool"], observation["fit_result"]
    previous = observation["current_parameters"]
    if tool == "sq.s21":
        return _close(candidate["readout_frequency_hz"], fit["frequency_hz"], 1e5)
    if tool == "sq.s21_zpa2d":
        return (_close(candidate["z_bias"], fit["sweet_spot_zpa"], .01)
                and _close(candidate["readout_frequency_hz"], fit["readout_frequency_hz"], 2e5))
    if tool == "sq.spectroscopy":
        return _close(candidate["drive_frequency_hz"], fit["frequency_hz"], 1e5)
    if tool == "sq.piamp":
        pi = fit["pi_amplitude"]
        return (observation.get("round_in_experiment", 0) >= 2
                and _close(candidate["pi_amplitude"], pi, 0.03 * pi)
                and _close(candidate["pi_over_2_amplitude"], pi / 2, 0.03 * pi)
                and _close(previous["pi_amplitude"], pi, 0.03 * pi)
                and _close(previous["pi_over_2_amplitude"], pi / 2, 0.03 * pi))
    if tool == "sq.ramsey_df":
        correction = fit["frequency_correction_hz"]
        return (abs(correction) < 5e4
                and _close(candidate["drive_frequency_hz"], previous["drive_frequency_hz"] + correction, 5e4)
                and _close(candidate["t2_star_us"], fit["t2_star_us"], 0.1 * fit["t2_star_us"]))
    if tool == "sq.t1":
        return (_close(candidate["t1_us"], fit["t1_us"], 0.1 * fit["t1_us"])
                and candidate["relaxation_delay_us"] >= 5 * fit["t1_us"])
    if tool == "sq.t2_echo":
        return _close(candidate["t2_echo_us"], fit["t2_echo_us"], 0.1 * fit["t2_echo_us"])
    if tool == "sq.xeb":
        return bool(fit.get("passed") and fit["per_cycle_fidelity"] >= XEB_THRESHOLDS["per_cycle_fidelity"])
    return False  # IQ is always checked independently from raw held-out shots.


def invalidate(completed: set, old: dict, new: dict) -> set:
    affected = {"readout_frequency_hz": 1, "z_bias": 1, "drive_frequency_hz": 4, "pi_amplitude": 3,
                "pi_over_2_amplitude": 3, "t2_star_us": 4, "t1_us": 5,
                "t2_echo_us": 6, "readout_amplitude": 8, "relaxation_delay_us": 8}
    first = len(TOOLS)
    for key in new:
        if new[key] != old[key]:
            start = affected[key]
            if key == "drive_frequency_hz" and abs(new[key] - old[key]) >= 5e5:
                start = 2
            first = min(first, start)
    return {index for index in completed if index < first}


class AgentRunner:
    def __init__(self, backend, policy, max_steps: int = 30, max_tool_calls: int = 8, on_event=None):
        if not 1 <= max_steps <= 1000 or not 1 <= max_tool_calls <= 100:
            raise ValueError("Budgets outside allowed range")
        self.backend, self.policy = backend, policy
        self.max_steps, self.max_tool_calls = max_steps, max_tool_calls
        self.on_event = on_event
        self.state = dict(DEFAULT_STATE)
        self.completed: set[int] = set()
        self.counts: Counter = Counter()
        self.observation = None
        self.history, self.events = [], []
        self.passes = 0
        self.latest_gate = None
        self.pending = None
        self.status, self.reason = "active", ""
        self.decision_index = 0

    def context(self) -> dict:
        return copy.deepcopy({
                "schema_version": "runtime-0.2", "state": self.state, "observation": self.observation,
                "recent_history": self.history[-8:], "completed_stages": [TOOLS[i] for i in sorted(self.completed)],
                "consecutive_iq_passes": self.passes, "required_iq_passes": 2,
                "available_tools": list(TOOLS), "iq_thresholds": IQ_THRESHOLDS,
                "xeb_thresholds": XEB_THRESHOLDS,
                "parameter_bounds": STATE_LIMITS, "scan_bounds": SCAN_LIMITS,
                "tool_scan_keys": {tool: sorted(keys) for tool, keys in SCAN_KEYS.items()},
                "budget": {"remaining_experiments": self.max_steps - sum(self.counts.values()),
                           "max_calls_per_tool": self.max_tool_calls, "tool_counts": dict(self.counts)},
            })

    def emit(self, event):
        self.events.append(copy.deepcopy(event))
        if self.on_event:
            self.on_event(copy.deepcopy(event))

    def result(self) -> dict:
        return copy.deepcopy({"status": self.status, "reason": self.reason,
            "experiment_count": sum(self.counts.values()), "tool_counts": dict(self.counts),
            "final_state": self.state, "completed_stages": [TOOLS[i] for i in sorted(self.completed)],
            "consecutive_iq_passes": self.passes, "final_iq_gate": self.latest_gate, "events": self.events})

    def finish(self, status, reason):
        self.status, self.reason, self.pending = status, reason, None
        self.emit({"event": "terminal", "status": status, "reason": reason, "state": self.state})

    def _candidate(self, action):
        tool = action["next_tool"]
        candidate = self.state | action["parameter_action"]["updates"]
        completed = invalidate(self.completed, self.state, candidate)
        if tool == "FINISH":
            if not (self.passes >= 2 and self.completed == set(range(len(CALIBRATION_STAGES))) and self.observation
                    and self.observation["tool"] == "sq.iqraw"):
                raise ContractError("Premature FINISH: independent IQ gate/prerequisites not satisfied")
        elif tool != "ESCALATE_HARDWARE_REVIEW":
            validate_scan(tool, candidate, action["parameter_action"]["scan"])
            if self.observation and self.observation["tool"] != "sq.iqraw" and stage_passed(self.observation, candidate):
                completed.add(TOOLS.index(self.observation["tool"]))
            if not set(range(TOOLS.index(tool))) <= completed:
                raise ContractError(f"Prerequisite stages not calibrated for {tool}")
        return candidate, completed

    def plan(self) -> dict | None:
        """Validate and cache one proposal. Never measure or commit state here."""
        if self.status != "active":
            return None
        if self.pending is not None:
            return copy.deepcopy(self.pending)
        try:
            raw = self.policy.decide(self.context())
        except Exception as exc:
            self.emit({"event": "policy_error", "error": str(exc)})
            self.finish("policy_error", str(exc))
            return None
        generation = getattr(self.policy, "last_generation", None)
        if generation is not None:
            self.emit({"event": "generation", "index": self.decision_index, "metrics": generation})
        try:
            action = parse_decision(raw)
            self._candidate(action)
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            self.emit({"event": "rejected_decision", "index": self.decision_index,
                       "error": str(exc), "raw_decision": str(raw)[:16384]})
            self.finish("invalid_action", f"Policy/schema error: {exc}")
            return None
        tool = action["next_tool"]
        if tool in TOOLS and (sum(self.counts.values()) >= self.max_steps or self.counts[tool] >= self.max_tool_calls):
            self.finish("budget_exhausted", f"Experiment budget exhausted before {tool}")
            return None
        self.pending = copy.deepcopy(action)
        self.emit({"event": "decision", "index": self.decision_index, "decision": action})
        self.decision_index += 1
        return copy.deepcopy(action)

    def step(self) -> dict:
        action = self.plan()
        if action is None:
            return self.result()
        # Revalidate after preview/resume. The persisted proposal is not trusted.
        try:
            action = parse_decision(action)
            candidate, completed = self._candidate(action)
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            self.finish("invalid_action", str(exc))
            return self.result()
        self.pending = None
        tool = action["next_tool"]
        if tool == "ESCALATE_HARDWARE_REVIEW":
            self.finish("escalated", action["reason"] or "Policy requested review; no hardware action taken")
        elif tool == "FINISH":
            self.finish("accepted", "Two consecutive independent IQ batches passed all gates")
        elif sum(self.counts.values()) >= self.max_steps or self.counts[tool] >= self.max_tool_calls:
            self.finish("budget_exhausted", f"Experiment budget exhausted before {tool}")
        else:
            before, event_count = self.snapshot(), len(self.events)
            try:
                self._measure(tool, candidate, completed, action["parameter_action"]["scan"])
            except KeyboardInterrupt:
                attempted = self.counts[tool] > before["counts"].get(tool, 0)
                self.restore(before)
                if attempted:
                    self.counts[tool] += 1
                self.pending = None
                del self.events[event_count:]
                self.emit({"event": "rollback", "tool": tool, "error": "Interrupted; uncommitted simulation rolled back",
                           "restored_state": self.state})
                raise
        return self.result()

    def _measure(self, tool, candidate, completed, scan):
        snapshot = dict(self.state)
        backend_checkpoint = self.backend.checkpoint()
        self.counts[tool] += 1  # Failed/cancelled executions also consume budget.
        try:
            obs = self.backend.measure(tool, copy.deepcopy(candidate), copy.deepcopy(scan))
            validate_observation(obs, tool, candidate)
            obs["round_in_experiment"] = self.counts[tool]
            json.dumps(obs, allow_nan=False)
            gate = iq_gate(obs["measurement"]) if tool == "sq.iqraw" else None
            if gate is not None:
                obs["fit_result"] = copy.deepcopy(gate.get("metrics", {}))
                obs["quality"] = {"reliable": gate["passed"], "metric": "held_out_iq_assignment"}
        except (Exception, KeyboardInterrupt) as exc:
            self.backend.restore(backend_checkpoint)
            self.emit({"event": "rollback", "tool": tool, "error": str(exc) or "Interrupted",
                       "restored_state": snapshot})
            if isinstance(exc, KeyboardInterrupt):
                raise
            self.finish("tool_error", str(exc))
            return
        changed = candidate != self.state
        self.state, self.completed, self.observation = candidate, completed, obs
        if changed or tool != "sq.iqraw":
            self.passes, self.latest_gate = 0, None
        if gate is not None:
            self.latest_gate = gate
            obs["acceptance"] = copy.deepcopy(gate)
            self.passes = self.passes + 1 if gate["passed"] else 0
        elif not obs["quality"]["reliable"]:
            self.completed = {index for index in self.completed if index < TOOLS.index(tool)}
        self.history.append({"step": sum(self.counts.values()), "tool": tool, "state": dict(self.state),
                             "fit_result": obs["fit_result"], "quality": obs["quality"], "acceptance": gate})
        self.history = self.history[-8:]
        self.emit({"event": "experiment", "step": sum(self.counts.values()), "tool": tool,
                   "state_before": snapshot, "state_after": dict(self.state), "observation": obs,
                   "consecutive_iq_passes": self.passes})

    def snapshot(self) -> dict:
        """Private simulator recovery data; never send this to a policy."""
        return copy.deepcopy({"version": 2, "state": self.state, "completed": sorted(self.completed),
            "counts": dict(self.counts), "observation": self.observation, "history": self.history,
            "passes": self.passes, "latest_gate": self.latest_gate, "pending": self.pending,
            "status": self.status, "reason": self.reason, "decision_index": self.decision_index,
            "backend_checkpoint": self.backend.checkpoint()})

    def restore(self, snapshot: dict) -> None:
        """Restore only a versioned, finite snapshot from the local session journal."""
        from .contracts import bounded
        data = copy.deepcopy(snapshot)
        json.dumps(data, allow_nan=False)
        if data.get("version") == 1:
            data["state"].setdefault("z_bias", DEFAULT_STATE["z_bias"])
            old_to_new = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
            data["completed"] = [old_to_new[index] for index in data["completed"] if index in old_to_new]
            data["version"] = 2
        if data.get("version") != 2 or set(data["state"]) != set(DEFAULT_STATE):
            raise ValueError("Unsupported checkpoint schema")
        bounded(data["state"], STATE_LIMITS)
        if (not set(data["completed"]) <= set(range(len(CALIBRATION_STAGES)))
                or not set(data["counts"]) <= set(TOOLS)
                or any(type(n) is not int or not 0 <= n <= self.max_tool_calls for n in data["counts"].values())
                or sum(data["counts"].values()) > self.max_steps
                or type(data["passes"]) is not int or not 0 <= data["passes"] <= self.max_steps
                or data["status"] not in {"active", "accepted", "escalated", "budget_exhausted",
                                         "invalid_action", "policy_error", "tool_error"}):
            raise ValueError("Invalid checkpoint counters/status")
        if data["pending"] is not None:
            parse_decision(data["pending"])
        self.backend.restore(data["backend_checkpoint"])
        for name in ("state", "observation", "history", "passes", "latest_gate", "pending", "status", "reason", "decision_index"):
            setattr(self, name, data[name])
        self.completed, self.counts = set(data["completed"]), Counter(data["counts"])

    def run(self) -> dict:
        while self.status == "active":
            self.step()
        return self.result()
