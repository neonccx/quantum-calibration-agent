"""Independent analytic surrogate. Scans use public settings, fits use measurements.

No dataset-generator auto-updates, reference actions, or truth-derived scan ranges.
Ramsey uses phase-cycled coherence quadratures; this is not a QICK hardware model.
"""

from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import curve_fit

from .contracts import IQ_THRESHOLDS, TOOLS


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - prediction) ** 2) / denominator) if denominator > 1e-15 else 0.0


def iq_gate(measurement: dict) -> dict:
    """Fit on even shots, evaluate on odd shots, separately within each label.

    Recomputed by the controller; model-supplied metrics cannot satisfy this gate.
    These are synthetic prepared-state assignment rates, not SPAM-corrected fidelity.
    """
    points = np.column_stack([measurement["i"], measurement["q"]])
    labels = np.asarray(measurement["prepared_state"])
    if len(labels) != len(points) or not np.isfinite(points).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Malformed IQ observations")
    groups = [points[labels == label] for label in (0, 1)]
    if any(len(group) < 128 for group in groups):
        raise ValueError("At least 128 shots per prepared state required")
    means = [group[::2].mean(axis=0) for group in groups]
    axis = means[1] - means[0]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return {"passed": False, "checks": {}, "reason": "Degenerate training centroids"}
    axis /= norm
    threshold = float((means[0] + means[1]) @ axis / 2)
    p0, p1 = [group[1::2] @ axis for group in groups]
    f0, f1 = float(np.mean(p0 < threshold)), float(np.mean(p1 >= threshold))
    metrics = {
        "f0": f0, "f1": f1, "assignment_fidelity": (f0 + f1) / 2,
        "visibility": f0 + f1 - 1,
        "snr": float(abs(p1.mean() - p0.mean()) / np.sqrt(max(p0.var(ddof=1) + p1.var(ddof=1), 1e-12))),
    }
    checks = {key: metrics[key] >= limit for key, limit in IQ_THRESHOLDS.items()}
    return {"passed": all(checks.values()), "metrics": metrics, "checks": checks,
            "thresholds": dict(IQ_THRESHOLDS), "evaluation": "held_out_odd_shots",
            "classifier": {"axis": axis.tolist(), "threshold": threshold},
            "test_shots_per_state": [len(p0), len(p1)]}


class AnalyticSimulator:
    backend_name = "analytic_surrogate_v0.1"

    def __init__(self, seed: int = 20260903, noise_scale: float = 1.0):
        if not np.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        self.rng = np.random.default_rng(seed)
        self.noise_scale = noise_scale
        # Only the measurement backend holds device truth. Policies receive dicts.
        self._truth = {
            "fr": 6.5e9 + self.rng.uniform(-8e6, 8e6),
            "fq": 5e9 + self.rng.uniform(-20e6, 20e6),
            "pi": self.rng.uniform(0.18, 0.32),
            "t1": self.rng.uniform(25, 45), "t2": self.rng.uniform(15, 25),
            "echo": self.rng.uniform(30, 50), "angle": self.rng.uniform(-np.pi, np.pi),
            "readout_optimum": self.rng.uniform(0.45, 0.65),
        }

    def checkpoint(self) -> dict:
        return copy.deepcopy(self.rng.bit_generator.state)

    def restore(self, checkpoint: dict) -> None:
        self.rng.bit_generator.state = copy.deepcopy(checkpoint)

    def _noise(self, size: int, sigma: float = 0.008) -> np.ndarray:
        return self.rng.normal(0, sigma * self.noise_scale, size)

    def measure(self, tool: str, state: dict, scan: dict) -> dict:
        if tool not in TOOLS:
            raise ValueError("Unregistered simulator tool")
        obs = {"tool": tool, "current_parameters": copy.deepcopy(state),
               "scan": copy.deepcopy(scan), "synthetic": True, "backend": self.backend_name}
        if tool in TOOLS[:2]:
            self._spectroscopy(obs, state, scan)
        elif tool == "sq.piamp":
            self._rabi(obs, state, scan)
        elif tool == "sq.ramsey_df":
            self._ramsey(obs, state, scan)
        elif tool in ("sq.t1", "sq.t2_echo"):
            self._decay(obs, state, scan)
        else:
            self._iq(obs, state, scan)
        return obs

    def _curve(self, obs: dict, x: np.ndarray, z: np.ndarray, unit: str) -> None:
        obs["sweep"] = {"values": x.tolist(), "unit": unit}
        obs["measurement"] = {"i": z.real.tolist(), "q": z.imag.tolist(), "unit": "a.u."}

    def _fit(self, obs: dict, x: np.ndarray, y: np.ndarray, fn, guesses: list, bounds, keys: dict) -> None:
        candidates = []
        for guess in guesses:
            try:
                pars, _ = curve_fit(fn, x, y, p0=guess, bounds=bounds, maxfev=12000)
                score = r2(y, fn(x, *pars))
                if np.isfinite(pars).all() and np.isfinite(score):
                    candidates.append((score, pars))
            except (ValueError, RuntimeError, FloatingPointError):
                continue
        if not candidates:
            obs.update(fit_result={}, quality={"fit_r2": None, "reliable": False, "reason": "fit_failed"})
            return
        score, pars = max(candidates, key=lambda candidate: candidate[0])
        obs["fit_result"] = {key: float(pars[index] * scale) for key, (index, scale) in keys.items()}
        obs["quality"] = {"fit_r2": score, "reliable": score >= 0.85}

    def _spectroscopy(self, obs: dict, state: dict, scan: dict) -> None:
        resonator = obs["tool"] == "sq.s21"
        center = scan.get("frequency_center_hz", state["readout_frequency_hz" if resonator else "drive_frequency_hz"])
        span = scan.get("frequency_span_hz", 40e6 if resonator else 100e6)
        x = np.linspace(center - span / 2, center + span / 2, 241)
        xm = (x - center) / 1e6
        truth_center = self._truth["fr" if resonator else "fq"]
        width = 2e6 if resonator else 3e6
        sign = -1 if resonator else 1
        signal = 0.5 + sign * 0.4 / (1 + (2 * (x - truth_center) / width) ** 2)
        # Fixed instrument analysis axis for these simplified scalar-response scans.
        z = signal + self._noise(x.size) + 1j * self._noise(x.size)
        self._curve(obs, x, z, "Hz")

        def lorentz(f, base, amp, f0, gamma):
            return base + amp / (1 + (2 * (f - f0) / gamma) ** 2)

        y = sign * z.real
        self._fit(obs, xm, y, lorentz,
                  [[float(np.median(y)), 0.4, float(xm[np.argmax(y)]), 3]],
                  ([-2, 0, xm.min(), 0.1], [2, 2, xm.max(), span / 1e6]),
                  {"frequency_offset_hz": (2, 1e6), "linewidth_hz": (3, 1e6)})
        if obs["fit_result"]:
            f = obs["fit_result"]
            f["frequency_hz"] = center + f.pop("frequency_offset_hz")
            f["edge_hit"] = abs(f["frequency_hz"] - center) > 0.42 * span
            obs["quality"]["reliable"] &= not f["edge_hit"]

    def _rabi(self, obs: dict, state: dict, scan: dict) -> None:
        x = np.linspace(0, scan.get("amplitude_max", 0.8), 161)
        contrast = 1 / (1 + ((self._truth["fq"] - state["drive_frequency_hz"]) / 1.5e6) ** 2)
        y = 0.5 - 0.45 * contrast * np.cos(np.pi * x / self._truth["pi"]) + self._noise(x.size)
        z = y + 1j * self._noise(x.size)
        self._curve(obs, x, z, "normalized_amplitude")

        def oscillation(a, base, amp, pi):
            return base - amp * np.cos(np.pi * a / pi)

        self._fit(obs, x, y, oscillation, [[0.5, 0.4, pi] for pi in (0.15, 0.22, 0.3, 0.4)],
                  ([-1, 0, 0.05], [2, 2, 0.6]), {"pi_amplitude": (2, 1), "contrast": (1, 2)})
        f = obs["fit_result"]
        if f:
            obs["quality"]["reliable"] = bool(obs["quality"]["reliable"] and f["contrast"] >= 0.3 and 2.1 * f["pi_amplitude"] <= x.max())

    def _ramsey(self, obs: dict, state: dict, scan: dict) -> None:
        x = np.linspace(0, scan.get("delay_max_us", 12.0), 241)
        offset_mhz = 0.3
        delta_mhz = (self._truth["fq"] - state["drive_frequency_hz"]) / 1e6
        pulse_contrast = np.sin(np.pi * state["pi_over_2_amplitude"] / self._truth["pi"]) ** 2
        z = 0.45 * pulse_contrast * np.exp(-x / self._truth["t2"]) * np.exp(2j * np.pi * (offset_mhz - delta_mhz) * x)
        z += self._noise(x.size) + 1j * self._noise(x.size)
        self._curve(obs, x, z, "us")
        obs["measurement"]["representation"] = "phase_cycled_coherence_quadratures"
        obs["programmed_detuning_mhz"] = offset_mhz
        phase = np.unwrap(np.angle(z))
        slope, intercept = np.polyfit(x, phase, 1, w=np.maximum(np.abs(z), 1e-6))
        decay, log_amp = np.polyfit(x, np.log(np.maximum(np.abs(z), 1e-8)), 1)
        tau = float(-1 / decay) if decay < 0 else 0.0
        pred = np.exp(log_amp + decay * x) * np.exp(1j * (slope * x + intercept))
        score = r2(np.r_[z.real, z.imag], np.r_[pred.real, pred.imag])
        obs["fit_result"] = {"frequency_correction_hz": float((offset_mhz - slope / (2 * np.pi)) * 1e6), "t2_star_us": tau}
        obs["quality"] = {"fit_r2": score, "reliable": bool(score >= 0.85 and 1 <= tau <= 200 and abs(z).mean() > 0.1)}

    def _decay(self, obs: dict, state: dict, scan: dict) -> None:
        t1 = obs["tool"] == "sq.t1"
        x = np.linspace(0, scan.get("delay_max_us", 120.0), 161)
        pulse_contrast = np.sin(np.pi * state["pi_amplitude"] / (2 * self._truth["pi"])) ** 2
        signal = 0.05 + 0.85 * pulse_contrast * np.exp(-x / self._truth["t1" if t1 else "echo"])
        z = signal + self._noise(x.size) + 1j * self._noise(x.size)
        self._curve(obs, x, z, "us")

        def exponential(t, base, amp, tau):
            return base + amp * np.exp(-t / tau)

        key = "t1_us" if t1 else "t2_echo_us"
        self._fit(obs, x, z.real, exponential, [[0.05, 0.85, 35]],
                  ([-1, 0, 1], [2, 2, 200]), {key: (2, 1), "contrast": (1, 1)})
        if obs["fit_result"]:
            obs["quality"]["reliable"] = bool(obs["quality"]["reliable"] and obs["fit_result"]["contrast"] >= 0.3 and x.max() >= 2 * obs["fit_result"][key])

    def _iq(self, obs: dict, state: dict, scan: dict) -> None:
        shots = scan.get("shots", 1024)
        detuning = (state["drive_frequency_hz"] - self._truth["fq"]) / 1e6
        excitation = np.sin(np.pi * state["pi_amplitude"] / (2 * self._truth["pi"])) ** 2 / (1 + (detuning / 1.5) ** 2)
        readout_gain = np.exp(-((state["readout_frequency_hz"] - self._truth["fr"]) / 2e6) ** 2)
        amplitude_gain = np.exp(-((state["readout_amplitude"] - self._truth["readout_optimum"]) / 0.25) ** 2)
        separation = 1.6 * readout_gain * amplitude_gain
        direction = np.array([np.cos(self._truth["angle"]), np.sin(self._truth["angle"])])
        thermal = 0.008 + 0.2 * np.exp(-state["relaxation_delay_us"] / self._truth["t1"])
        groups = []
        for probability in (thermal, 0.985 * excitation):
            actual = self.rng.random(shots) < probability
            means = (actual.astype(float) - 0.5)[:, None] * separation * direction
            groups.append(means + self.rng.normal(0, max(1e-6, 0.16 * self.noise_scale), (shots, 2)))
        points = np.vstack(groups)
        obs["measurement"] = {"i": points[:, 0].tolist(), "q": points[:, 1].tolist(),
                              "prepared_state": [0] * shots + [1] * shots, "unit": "a.u."}
        obs["acceptance"] = iq_gate(obs["measurement"])
        obs["fit_result"] = obs["acceptance"].get("metrics", {})
        obs["quality"] = {"reliable": obs["acceptance"]["passed"], "metric": "held_out_iq_assignment"}
