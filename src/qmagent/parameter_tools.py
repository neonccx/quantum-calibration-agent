"""Observation-only arithmetic for the next protocol revision.

Produces candidate values, never chooses the next experiment or commits state.
The frozen v2 model evaluation does not inject or apply these values automatically.
"""
import math
from .storage import digest


def calculate_fit_updates(observation, state):
    if not observation or observation.get("quality", {}).get("reliable") is not True:
        raise ValueError("A reliable analyzed observation is required")
    fit, tool = observation["fit_result"], observation["tool"]
    updates = {}
    if tool == "sq.s21":
        updates = {"readout_frequency_hz": fit["frequency_hz"]}
    elif tool == "sq.spectroscopy":
        updates = {"drive_frequency_hz": fit["frequency_hz"]}
    elif tool == "sq.piamp":
        updates = {"pi_amplitude": fit["pi_amplitude"], "pi_over_2_amplitude": fit["pi_amplitude"]/2}
    elif tool == "sq.ramsey_df":
        measured_at = observation["current_parameters"]["drive_frequency_hz"]
        if state["drive_frequency_hz"] != measured_at:
            raise ValueError("Ramsey observation is stale for the current drive frequency")
        updates = {"drive_frequency_hz": measured_at+fit["frequency_correction_hz"], "t2_star_us": fit["t2_star_us"]}
    elif tool == "sq.t1":
        updates = {"t1_us": fit["t1_us"], "relaxation_delay_us": 5*fit["t1_us"]}
    elif tool == "sq.t2_echo":
        updates = {"t2_echo_us": fit["t2_echo_us"]}
    else:
        raise ValueError("No fit-update arithmetic registered for this observation")
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in updates.values()):
        raise ValueError("Calculated parameter values must be finite numbers")
    return {"tool": "analysis.calculate_fit_updates", "version": "0.1", "updates": updates,
            "source_sha256": digest({"fit_result": fit, "current_parameters": observation.get("current_parameters"),
                                     "tool": tool, "quality": observation["quality"]}),
            "requires_controller_validation": True}
