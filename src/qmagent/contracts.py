"""Strict runtime contract and simulation-specific bounds, NOT hardware limits."""

from __future__ import annotations

import json
import math
from typing import Any

TOOLS = ("sq.s21", "sq.s21_zpa2d", "sq.spectroscopy", "sq.piamp", "sq.ramsey_df",
         "sq.t1", "sq.t2_echo", "sq.xeb", "sq.iqraw")
CALIBRATION_STAGES = TOOLS[:-1]
TERMINALS = ("FINISH", "ESCALATE_HARDWARE_REVIEW")
STATE_LIMITS = {
    "readout_frequency_hz": (6e9, 7e9),
    "z_bias": (-1.0, 1.0),
    "drive_frequency_hz": (4e9, 6e9),
    "pi_amplitude": (0.05, 0.6),
    "pi_over_2_amplitude": (0.025, 0.3),
    "readout_amplitude": (0.1, 0.9),
    "relaxation_delay_us": (1.0, 1000.0),
    "t1_us": (1.0, 200.0),
    "t2_star_us": (1.0, 200.0),
    "t2_echo_us": (1.0, 200.0),
}
DEFAULT_STATE = {
    "readout_frequency_hz": 6.5e9, "z_bias": 0.0, "drive_frequency_hz": 5e9,
    "pi_amplitude": 0.25, "pi_over_2_amplitude": 0.125,
    "readout_amplitude": 0.3, "relaxation_delay_us": 200.0,
    "t1_us": 30.0, "t2_star_us": 20.0, "t2_echo_us": 40.0,
}
SCAN_LIMITS = {
    "frequency_center_hz": (4e9, 7e9), "frequency_span_hz": (1e6, 300e6),
    "amplitude_max": (0.2, 1.0), "delay_max_us": (2.0, 200.0),
    "shots": (256, 4096),
    "zpa_center": (-1.0, 1.0), "zpa_span": (0.2, 2.0),
    "zpa_points": (21, 101), "frequency_points": (101, 501),
    "max_depth": (8, 256), "depth_points": (5, 32), "circuits_per_depth": (16, 512),
}
SCAN_KEYS = {
    "sq.s21": {"frequency_center_hz", "frequency_span_hz"},
    "sq.s21_zpa2d": {"frequency_center_hz", "frequency_span_hz", "zpa_center", "zpa_span",
                       "zpa_points", "frequency_points"},
    "sq.spectroscopy": {"frequency_center_hz", "frequency_span_hz"},
    "sq.piamp": {"amplitude_max"}, "sq.ramsey_df": {"delay_max_us"},
    "sq.t1": {"delay_max_us"}, "sq.t2_echo": {"delay_max_us"},
    "sq.xeb": {"max_depth", "depth_points", "circuits_per_depth"}, "sq.iqraw": {"shots"},
}
IQ_THRESHOLDS = {"snr": 2.5, "visibility": 0.8, "f0": 0.88, "f1": 0.88, "assignment_fidelity": 0.9}
XEB_THRESHOLDS = {"per_cycle_fidelity": 0.985, "fit_r2": 0.90, "max_standard_error": 0.005}


class ContractError(ValueError):
    pass


def bounded(values: dict, limits: dict) -> None:
    for key, value in values.items():
        if key not in limits:
            raise ContractError(f"Unknown parameter: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError(f"{key} must be a finite number")
        low, high = limits[key]
        if not low <= value <= high:
            raise ContractError(f"{key} outside simulation bounds [{low}, {high}]")


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_decision(value: Any) -> dict:
    if isinstance(value, str):
        if len(value) > 16384:
            raise ContractError("Decision exceeds 16 KiB")
        try:
            value = json.loads(value, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError) as exc:
            raise ContractError(f"Invalid strict JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"next_tool", "parameter_action", "reason"}:
        raise ContractError("Expected exactly next_tool, parameter_action, reason (runtime schema 0.1)")
    if not isinstance(value["next_tool"], str) or value["next_tool"] not in TOOLS + TERMINALS:
        raise ContractError("Unknown tool")
    action = value["parameter_action"]
    if not isinstance(action, dict) or set(action) != {"updates", "scan"}:
        raise ContractError("parameter_action requires exactly updates and scan")
    if not isinstance(action["updates"], dict) or not isinstance(action["scan"], dict):
        raise ContractError("updates and scan must be objects")
    if not isinstance(value["reason"], str) or len(value["reason"]) > 2000:
        raise ContractError("reason must be a string of at most 2000 characters")
    bounded(action["updates"], STATE_LIMITS)
    bounded(action["scan"], SCAN_LIMITS)
    tool = value["next_tool"]
    if tool in TERMINALS:
        if action["updates"] or action["scan"]:
            raise ContractError("Terminal actions cannot modify measured state")
    elif not set(action["scan"]) <= SCAN_KEYS[tool]:
        raise ContractError(f"Scan arguments not supported by {tool}")
    for key in ("shots", "zpa_points", "frequency_points", "max_depth", "depth_points", "circuits_per_depth"):
        if key in action["scan"] and type(action["scan"][key]) is not int:
            raise ContractError(f"{key} must be an integer")
    return value


def validate_scan(tool: str, state: dict, scan: dict) -> None:
    bounded(state, STATE_LIMITS)
    if tool not in TOOLS or not isinstance(scan, dict) or not set(scan) <= SCAN_KEYS[tool]:
        raise ContractError("Unsupported scan arguments")
    bounded(scan, SCAN_LIMITS)
    for key in ("shots", "zpa_points", "frequency_points", "max_depth", "depth_points", "circuits_per_depth"):
        if key in scan and type(scan[key]) is not int:
            raise ContractError(f"{key} must be an integer")
    if tool in ("sq.s21", "sq.s21_zpa2d", "sq.spectroscopy"):
        key = "readout_frequency_hz" if tool.startswith("sq.s21") else "drive_frequency_hz"
        center = scan.get("frequency_center_hz", state[key])
        span = scan.get("frequency_span_hz", 40e6 if tool.startswith("sq.s21") else 100e6)
        low, high = STATE_LIMITS[key]
        if center - span / 2 < low or center + span / 2 > high:
            raise ContractError("Full frequency sweep must stay inside simulation bounds")
    if tool == "sq.s21_zpa2d":
        center = scan.get("zpa_center", state["z_bias"])
        span = scan.get("zpa_span", 1.0)
        low, high = STATE_LIMITS["z_bias"]
        if center-span/2 < low or center+span/2 > high:
            raise ContractError("Full ZPA sweep must stay inside simulation bounds")
        if scan.get("zpa_points", 41) < 21 or scan.get("frequency_points", 241) < 101:
            raise ContractError("ZPA2D grid is undersampled")
    if tool == "sq.ramsey_df" and scan.get("delay_max_us", 12.0) > 20.0:
        raise ContractError("Ramsey delay limited to 20 us to retain sampling bandwidth")


def decision(tool: str, updates: dict | None = None, scan: dict | None = None, reason: str = "") -> dict:
    return {"next_tool": tool, "parameter_action": {"updates": updates or {}, "scan": scan or {}}, "reason": reason}


def validate_observation(observation: dict, tool: str, candidate: dict) -> None:
    """Reject malformed backend output before any controller state commit."""
    if not isinstance(observation, dict) or observation.get("tool") != tool or observation.get("current_parameters") != candidate:
        raise ContractError("Backend observation/state mismatch")
    quality, fit = observation.get("quality"), observation.get("fit_result")
    if (not isinstance(quality, dict) or type(quality.get("reliable")) is not bool
            or not isinstance(fit, dict)):
        raise ContractError("Backend must provide quality.reliable (boolean) and fit_result (object)")
    data = observation.get("measurement")
    if not isinstance(data, dict):
        raise ContractError("Missing measurement object")
    def numeric_shape(values):
        if not isinstance(values, list) or not values:
            raise ContractError("Measurements must be nonempty numeric I/Q arrays")
        if all(isinstance(value, list) for value in values):
            widths = {len(value) for value in values}
            if len(widths) != 1 or 0 in widths:
                raise ContractError("Measurement grids must be rectangular")
            flat = [item for row in values for item in row]
            shape = (len(values), len(values[0]))
        elif any(isinstance(value, list) for value in values):
            raise ContractError("Measurement arrays cannot mix scalars and rows")
        else:
            flat, shape = values, (len(values),)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in flat):
            raise ContractError("Nonfinite/nonnumeric I/Q data")
        return shape
    i_shape, q_shape = numeric_shape(data.get("i")), numeric_shape(data.get("q"))
    if i_shape != q_shape:
        raise ContractError("I/Q arrays have different shapes")
    if tool == "sq.s21_zpa2d":
        sweep = observation.get("sweep", {})
        frequencies, zpas = sweep.get("frequency_values_hz"), sweep.get("zpa_values")
        if (len(i_shape) != 2 or not isinstance(frequencies, list) or not isinstance(zpas, list)
                or i_shape != (len(zpas), len(frequencies)) or len(zpas) < 21 or len(frequencies) < 101):
            raise ContractError("ZPA2D axes do not match the I/Q grid")
        if any(b <= a for axis in (frequencies, zpas) for a, b in zip(axis, axis[1:])):
            raise ContractError("ZPA2D axes must be strictly increasing")
    elif len(i_shape) != 1:
        raise ContractError("Only sq.s21_zpa2d may return two-dimensional I/Q arrays")
    required = {"sq.s21": ("frequency_hz",),
                "sq.s21_zpa2d": ("sweet_spot_zpa", "readout_frequency_hz", "flux_period_zpa"),
                "sq.spectroscopy": ("frequency_hz",),
                "sq.piamp": ("pi_amplitude",), "sq.ramsey_df": ("frequency_correction_hz", "t2_star_us"),
                "sq.t1": ("t1_us",), "sq.t2_echo": ("t2_echo_us",),
                "sq.xeb": ("per_cycle_fidelity", "error_per_cycle")}
    if quality["reliable"]:
        for key in required.get(tool, ()):
            value = fit.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ContractError(f"Reliable fit is missing finite numeric {key}")
    json.dumps(observation, allow_nan=False)
