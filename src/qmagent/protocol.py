"""Shared native function-call contract for training and online inference."""

import copy
import json
import re

from .contracts import TOOLS, parse_decision
from .storage import digest

PROTOCOL_VERSION = "calibration-step-0.3"
TOOLS_SCHEMA = [{"type": "function", "function": {
    "name": "calibration.step",
    "description": "Apply bounded parameter updates, acquire the named simulated experiment and run its registered deterministic analysis. Terminal actions acquire nothing. No hardware or shell access.",
    "parameters": {"type": "object", "additionalProperties": False,
        "required": ["next_tool", "parameter_action", "reason"],
        "properties": {
            "next_tool": {"type": "string", "enum": list(TOOLS)+["FINISH", "ESCALATE_HARDWARE_REVIEW"]},
            "parameter_action": {"type": "object", "additionalProperties": False,
                "required": ["updates", "scan"], "properties": {
                    "updates": {"type": "object", "additionalProperties": {"type": "number"}},
                    "scan": {"type": "object", "additionalProperties": {"type": "number"}}}},
            "reason": {"type": "string"}}}}}]


def public_context(context):
    """Keep exact fitted values; raw arrays and hidden simulator state never enter prompts."""
    view = copy.deepcopy(context)
    obs = view.get("observation")
    if obs:
        measurement = obs.pop("measurement", {})
        sweep = obs.pop("sweep", {})
        if "raw_reference" not in obs:
            obs["raw_reference"] = obs.pop("raw_artifact", {"sha256": digest({"measurement": measurement, "sweep": sweep})})
            def shape(value):
                if not isinstance(value, list):
                    return None
                return [len(value)]+(shape(value[0]) or []) if value else [0]
            obs["raw_reference"]["array_shapes"] = {key: shape(value) for key, value in measurement.items()
                                                     if isinstance(value, list)}
            obs["raw_reference"]["array_lengths"] = {key: len(value) for key, value in measurement.items()
                                                      if isinstance(value, list)}
        obs["raw_reference"]["availability"] = "controller audit artifact; not model input"
        if sweep:
            obs["sweep_summary"] = {key: value for key, value in sweep.items()
                                    if key not in {"values", "frequency_values_hz", "zpa_values"}}
            values = sweep.get("values", [])
            if values:
                obs["sweep_summary"].update(start=values[0], stop=values[-1], count=len(values))
            for key in ("frequency_values_hz", "zpa_values"):
                values = sweep.get(key, [])
                if values:
                    obs["sweep_summary"][key.removesuffix("_values_hz").removesuffix("_values")+"_axis"] = {
                        "start": values[0], "stop": values[-1], "count": len(values)}
    return view


def policy_messages(context, fit_update_tool=False, prompt_profile="skill"):
    from .policies import MINIMAL_POLICY_PROMPT, SYSTEM_PROMPT
    if prompt_profile == "minimal":
        instruction = MINIMAL_POLICY_PROMPT
    elif prompt_profile == "skill":
        instruction = SYSTEM_PROMPT.replace("Return one strict JSON object, no markdown, with exactly:",
            "Call calibration.step exactly once with the following argument schema (no prose outside the call):")
    else:
        raise ValueError("Invalid calibration prompt profile")
    if fit_update_tool:
        instruction += ("\nWhen applying a reliable non-IQ fit, call calibration.step_from_fit exactly once instead of calibration.step. "
            "Choose next_tool, scan and reason; the tool computes all parameter updates including signed Ramsey addition, pi/2 and 5*T1. "
            "For initial measurements, unreliable-fit rescans, IQ actions and terminal actions, use calibration.step. Never do fit arithmetic yourself.")
    return [{"role": "system", "content": instruction + "\nUse English for reason. Raw data are analyzed by registered tools, never estimate fitted numbers yourself."},
            {"role": "user", "content": json.dumps(public_context(context), ensure_ascii=True, separators=(",", ":"))}]


def assistant_call(action):
    checked = parse_decision(action)
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "calibration.step", "arguments": checked}}]}


def parse_call(text, context=None, allow_fit_tool=False):
    """Reject parallel calls, extra prose, unknown functions and malformed decisions."""
    match = re.fullmatch(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", text, re.S)
    if not match:
        raise ValueError("Expected exactly one native JSON tool call")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate tool-call key")
            result[key] = value
        return result
    call = json.loads(match.group(1), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite tool call")))
    if set(call) != {"name", "arguments"}:
        raise ValueError("Unregistered function call")
    if call["name"] == "calibration.step_from_fit" and allow_fit_tool:
        from .parameter_tools import calculate_fit_updates
        args = call["arguments"]
        if not isinstance(args, dict) or set(args) != {"next_tool", "scan", "reason"}:
            raise ValueError("Invalid fit-tool arguments")
        if args["next_tool"] not in TOOLS:
            raise ValueError("Fit tool requires an experiment, not a terminal action")
        updates = calculate_fit_updates(context["observation"], context["state"])["updates"]
        return parse_decision({"next_tool": args["next_tool"],
            "parameter_action": {"updates": updates, "scan": args["scan"]}, "reason": args["reason"]})
    if call["name"] != "calibration.step":
        raise ValueError("Unregistered function call")
    return parse_decision(call["arguments"])


FIT_TOOL_SCHEMA = {"type": "function", "function": {
    "name": "calibration.step_from_fit",
    "description": "Explicitly apply the latest reliable fit using deterministic arithmetic, then acquire your chosen experiment and analyze it. The controller still checks prerequisites and bounds. Choose the next experiment yourself; supply no calculated parameter numbers.",
    "parameters": {"type": "object", "additionalProperties": False,
        "required": ["next_tool", "scan", "reason"], "properties": {
            "next_tool": {"type": "string", "enum": list(TOOLS)},
            "scan": {"type": "object", "additionalProperties": {"type": "number"}},
            "reason": {"type": "string"}}}}}
