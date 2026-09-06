"""Deterministic observation-only analysis registry; no model or backend access."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy.signal import find_peaks, savgol_filter

from .physics import complex_notch, PHYSICS_VERSION, SOURCES
from .simulator import iq_gate, r2
from .storage import digest
from .contracts import XEB_THRESHOLDS

ANALYSIS_VERSION = "analysis-flux-xeb-0.3"
REGISTRY = {name: "analysis." + name.split(".")[1] for name in (
    "sq.s21", "sq.s21_zpa2d", "sq.spectroscopy", "sq.piamp", "sq.ramsey_df",
    "sq.t1", "sq.t2_echo", "sq.xeb", "sq.iqraw")}


def _fit_curve(x, y, fn, guesses, bounds, names):
    options = []
    for guess in guesses:
        try:
            pars, covariance = curve_fit(fn, x, y, p0=guess, bounds=bounds, maxfev=12000)
            score = r2(y, fn(x, *pars))
            if np.isfinite(pars).all() and np.isfinite(score) and np.isfinite(covariance).all():
                options.append((score, pars, covariance))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    if not options:
        return {}, {"reliable": False, "reason": "fit_failed", "fit_r2": None}
    score, pars, cov = max(options, key=lambda item: item[0])
    result = {name: float(pars[index]) for index, name in enumerate(names)}
    uncertainty = {name: float(np.sqrt(max(0, cov[index, index]))) for index, name in enumerate(names)}
    near_bound = bool(np.any(np.isclose(pars, bounds[0], rtol=0, atol=1e-5))
                      or np.any(np.isclose(pars, bounds[1], rtol=0, atol=1e-5)))
    return result, {"reliable": bool(score >= .85 and not near_bound), "fit_r2": float(score),
                    "reason": "fit_assessed", "standard_errors": uncertainty,
                    "uncertainty_method": "local_least_squares_covariance", "bound_hit": near_bound}


def fit_s21(obs):
    f = np.asarray(obs["sweep"]["values"])
    z = np.asarray(obs["measurement"]["i"]) + 1j * np.asarray(obs["measurement"]["q"])
    reference, half = float((f[0] + f[-1]) / 2), float((f[-1] - f[0]) / 2)
    # Fit gain/phase/delay from the measured trace, not latent simulator settings.
    edge_indices = np.r_[np.arange(max(8, len(f)//8)), np.arange(len(f)-max(8, len(f)//8), len(f))]
    slope, intercept = np.polyfit((f[edge_indices]-reference)/1e6, np.unwrap(np.angle(z))[edge_indices], 1)
    delay_ns = float(np.clip(-slope/(2*np.pi)*1000, -95, 95))
    initial_f = float(np.clip((f[np.argmin(np.abs(z))]-reference)/half, -.99, .99))
    initial_gain = float(np.clip(np.median(np.abs(z[edge_indices])), .15, 2.5))
    # offset, log Ql, depth, mismatch phase, log gain, reference phase, delay(ns)
    lower = [-1, np.log(100), .01, -.6, np.log(.1), intercept-2, -100]
    upper = [1, np.log(1e6), .99, .6, np.log(3), intercept+2, 100]

    def residual(pars):
        prediction = complex_notch(f, reference+half*pars[0], np.exp(pars[1]), pars[2],
            pars[3], np.exp(pars[4]), pars[5], pars[6]*1e-9, reference)
        error = prediction-z
        return np.r_[error.real, error.imag]

    candidates = []
    for q in (1500, 4000, 10000):
        result = least_squares(residual, [initial_f, np.log(q), .5, 0, np.log(initial_gain), intercept, delay_ns],
                               bounds=(lower, upper), max_nfev=1200)
        if result.success and np.isfinite(result.x).all():
            candidates.append(result)
    if not candidates:
        return {}, {"reliable": False, "reason": "fit_failed"}
    best = min(candidates, key=lambda item: np.sum(item.fun**2))
    pars = best.x
    frequency, ql = reference+half*pars[0], np.exp(pars[1])
    residual_variance = float(np.sum(best.fun**2)/(2*len(f)-7))
    covariance = np.linalg.pinv(best.jac.T @ best.jac) * residual_variance
    stderr = float(np.sqrt(max(covariance[0, 0], 0)) * half)
    pred = z + best.fun[:len(f)] + 1j*best.fun[len(f):]
    # Center each quadrature separately; a constant I offset must not inflate R2.
    denominator = float(np.sum(np.abs(z-z.mean())**2))
    score = 1-float(np.sum(np.abs(z-pred)**2))/max(denominator, 1e-15)
    edge = bool(abs(pars[0]) > .84)
    width = frequency/ql
    resonance_snr = float(np.exp(pars[4])*pars[2]/max(np.sqrt(residual_variance), 1e-12))
    checks = {"fit_r2": score >= .9, "resonance_snr": resonance_snr >= 10, "not_at_edge": not edge,
              "coverage": 2*half >= 4*width, "sampling": np.diff(f).max() <= width/4,
              "frequency_uncertainty": stderr < min(1e5, .1*width), "not_at_bound": not bool(np.any(best.active_mask))}
    checks = {key: bool(value) for key, value in checks.items()}
    return {"frequency_hz": float(frequency), "linewidth_hz": float(width), "loaded_q": float(ql),
            "edge_hit": edge, "depth": float(pars[2]), "mismatch_phase_rad": float(pars[3]),
            "receiver_gain": float(np.exp(pars[4])), "receiver_phase_rad": float(pars[5]),
            "cable_delay_s": float(pars[6]*1e-9)}, {
                "reliable": bool(all(checks.values())), "fit_r2": score, "checks": checks,
                "resonance_amplitude_snr": resonance_snr,
                "standard_errors": {"frequency_hz": stderr},
                "uncertainty_method": "local_least_squares_covariance", "reason": "complex_notch_fit"}


def fit_spectroscopy(obs):
    f = np.asarray(obs["sweep"]["values"])
    y = np.asarray(obs["measurement"]["i"])
    center = float((f[0]+f[-1])/2)
    x = (f-center)/1e6
    smooth = savgol_filter(y, 11, 2)
    prominence = max(.07, .22*float(np.ptp(smooth)))
    peaks, properties = find_peaks(smooth, prominence=prominence, distance=12)
    candidates = [{"frequency_hz": float(f[p]), "prominence": float(v)}
                  for p, v in zip(peaks, properties["prominences"])]
    candidates.sort(key=lambda item: item["prominence"], reverse=True)
    fn = lambda x, base, amp, f0, width: base+amp/(1+(2*(x-f0)/width)**2)
    fit, quality = _fit_curve(x, y, fn, [[.02, .45, float(x[np.argmax(y)]), 3]],
        ([-.2, 0, x.min(), .1], [.5, 1, x.max(), float(np.ptp(x))]), ["base", "amplitude", "center_mhz", "linewidth_mhz"])
    if fit:
        fit["frequency_hz"] = center+fit.pop("center_mhz")*1e6
        fit["linewidth_hz"] = fit.pop("linewidth_mhz")*1e6
        fit["edge_hit"] = bool(abs(fit["frequency_hz"]-center) > .42*(f[-1]-f[0]))
        fit["peak_candidates"] = candidates
        ambiguous = len(candidates) >= 2 and candidates[1]["prominence"] > .5*candidates[0]["prominence"]
        quality["ambiguous_peaks"] = ambiguous
        quality["reliable"] = bool(quality["reliable"] and not fit["edge_hit"] and not ambiguous and fit["amplitude"] > .1)
        quality["standard_errors"]["frequency_hz"] = quality["standard_errors"].pop("center_mhz")*1e6
    return fit, quality


def fit_s21_zpa2d(obs):
    """Extract the measured resonance ridge, then locate its periodic sweet spot.

    The fit sees only the complex S21 grid.  It does not receive SQUID truth or the
    qubit-frequency curve used by the simulator.
    """
    frequency = np.asarray(obs["sweep"]["frequency_values_hz"], dtype=float)
    zpa = np.asarray(obs["sweep"]["zpa_values"], dtype=float)
    z = np.asarray(obs["measurement"]["i"], dtype=float)+1j*np.asarray(obs["measurement"]["q"], dtype=float)
    ridge = []
    for trace in z:
        window = min(9, len(trace) if len(trace) % 2 else len(trace)-1)
        magnitude = savgol_filter(np.abs(trace), window, 2)
        index = int(np.argmin(magnitude))
        if 0 < index < len(frequency)-1:
            xs = frequency[index-1:index+2]
            coefficients = np.polyfit(xs-frequency[index], magnitude[index-1:index+2], 2)
            offset = -coefficients[1]/(2*coefficients[0]) if coefficients[0] > 0 else 0.0
            ridge.append(float(frequency[index]+np.clip(offset, xs[0]-frequency[index], xs[-1]-frequency[index])))
        else:
            ridge.append(float(frequency[index]))
    ridge = np.asarray(ridge)
    center = float((zpa[0]+zpa[-1])/2)
    candidates = []
    for period in np.linspace(max(.55, .75*np.ptp(zpa)), min(1.45, 1.6*np.ptp(zpa)), 181):
        phase = 2*np.pi*(zpa-center)/period
        design = np.column_stack([np.ones_like(zpa), np.cos(phase), np.sin(phase),
                                  np.cos(2*phase), np.sin(2*phase)])
        coefficients, *_ = np.linalg.lstsq(design, ridge, rcond=None)
        prediction = design@coefficients
        candidates.append((r2(ridge, prediction), period, coefficients))
    score, period, coefficients = max(candidates, key=lambda item: item[0])
    dense = np.linspace(zpa[0], zpa[-1], 10001)
    phase = 2*np.pi*(dense-center)/period
    prediction = np.column_stack([np.ones_like(dense), np.cos(phase), np.sin(phase),
                                  np.cos(2*phase), np.sin(2*phase)])@coefficients
    gradient = np.gradient(prediction, dense)
    turns = np.where(np.signbit(gradient[:-1]) != np.signbit(gradient[1:]))[0]+1
    margin = .06*np.ptp(zpa)
    turns = turns[(dense[turns] > zpa[0]+margin) & (dense[turns] < zpa[-1]-margin)]
    if not len(turns):
        return {}, {"reliable": False, "reason": "no_interior_ridge_extremum", "fit_r2": float(score)}
    # In this declared negative-anharmonicity dispersive model the maximum-f01
    # transmon sweet spot gives the most negative resonator shift.
    sweet_index = int(turns[np.argmin(prediction[turns])])
    sweet = float(dense[sweet_index])
    readout = float(prediction[sweet_index])
    step = float(np.median(np.diff(frequency)))
    rmse = float(np.sqrt(np.mean((ridge-(np.column_stack([
        np.ones_like(zpa), np.cos(2*np.pi*(zpa-center)/period), np.sin(2*np.pi*(zpa-center)/period),
        np.cos(4*np.pi*(zpa-center)/period), np.sin(4*np.pi*(zpa-center)/period)])@coefficients))**2)))
    checks = {"ridge_fit_r2": score >= .85, "interior_sweet_spot": True,
              "ridge_contrast": float(np.ptp(ridge)) >= 3*step, "ridge_rmse": rmse <= 2.5*step}
    return {"sweet_spot_zpa": sweet, "readout_frequency_hz": readout,
            "flux_period_zpa": float(period), "ridge_span_hz": float(np.ptp(ridge)),
            "ridge_rmse_hz": rmse}, {"reliable": bool(all(checks.values())), "fit_r2": float(score),
            "checks": {key: bool(value) for key, value in checks.items()},
            "reason": "periodic_resonance_ridge_fit", "uncertainty_method": "grid_resolution",
            "standard_errors": {"sweet_spot_zpa": float(np.diff(dense).mean()),
                                "readout_frequency_hz": max(step, rmse)}}


def fit_xeb(obs):
    depth = np.asarray(obs["sweep"]["values"], dtype=float)
    fidelity = np.asarray(obs["measurement"]["i"], dtype=float)
    fn = lambda m, offset, amplitude, cycle: offset+amplitude*cycle**m
    fitted, quality = _fit_curve(depth, fidelity, fn,
        [[.02, .98, value] for value in (.97, .985, .995)],
        ([0, .2, .9], [.2, 1.2, .99999]), ["spam_offset", "amplitude", "per_cycle_fidelity"])
    if not fitted:
        return fitted, quality
    fitted["error_per_cycle"] = 1-fitted["per_cycle_fidelity"]
    stderr = quality["standard_errors"]["per_cycle_fidelity"]
    checks = {"fit_r2": quality["fit_r2"] >= XEB_THRESHOLDS["fit_r2"],
              "cycle_uncertainty": stderr <= XEB_THRESHOLDS["max_standard_error"],
              "depth_coverage": depth.max() >= 32 and len(depth) >= 5}
    quality["checks"] = {key: bool(value) for key, value in checks.items()}
    quality["reliable"] = bool(all(checks.values()) and not quality.get("bound_hit", False))
    fitted["passed"] = bool(quality["reliable"] and
                            fitted["per_cycle_fidelity"] >= XEB_THRESHOLDS["per_cycle_fidelity"])
    fitted["threshold"] = XEB_THRESHOLDS["per_cycle_fidelity"]
    quality["reason"] = "single_qubit_xeb_style_decay_fit"
    return fitted, quality


def analyze(observation: dict) -> dict:
    """Return analysis separately; never mutate the acquisition artifact."""
    obs = copy.deepcopy(observation)
    tool = obs["tool"]
    if tool not in REGISTRY:
        raise ValueError("Unregistered analysis tool")
    y = np.asarray(obs["measurement"]["i"], dtype=float)
    q = np.asarray(obs["measurement"]["q"], dtype=float)
    if tool == "sq.s21_zpa2d":
        if y.ndim != 2 or min(y.shape) < 16 or q.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(q).all():
            raise ValueError("Invalid finite aligned ZPA2D I/Q grid")
        fitted, quality = fit_s21_zpa2d(obs)
        return {"fit_result": fitted, "quality": quality, "analysis_tool": REGISTRY[tool],
                "analysis_version": ANALYSIS_VERSION, "input_sha256": digest(observation),
                "physics_version": PHYSICS_VERSION, "sources": SOURCES}
    if y.ndim != 1 or y.size < 5 or q.shape != y.shape or not np.isfinite(np.r_[y, q]).all():
        raise ValueError("Invalid finite aligned I/Q observations")
    if tool == "sq.iqraw":
        gate = iq_gate(obs["measurement"])
        return {"fit_result": gate.get("metrics", {}), "quality": {"reliable": gate["passed"],
                "metric": "held_out_iq_assignment"}, "acceptance": gate, "analysis_tool": REGISTRY[tool],
                "analysis_version": ANALYSIS_VERSION, "input_sha256": digest(observation)}
    x = np.asarray(obs["sweep"]["values"], dtype=float)
    if x.shape != y.shape or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Sweep must be finite, strictly increasing and aligned")
    if tool == "sq.s21":
        fitted, quality = fit_s21(obs)
    elif tool == "sq.spectroscopy":
        fitted, quality = fit_spectroscopy(obs)
    elif tool == "sq.piamp":
        fn = lambda a, base, amp, pi: base-amp*np.cos(np.pi*a/pi)
        fitted, quality = _fit_curve(x, y, fn, [[.5, .45, pi] for pi in (.15, .22, .3, .4)],
            ([-.2, 0, .05], [1.2, 1, .6]), ["baseline", "amplitude", "pi_amplitude"])
        if fitted:
            fitted["contrast"] = fitted["amplitude"]*2
            quality["reliable"] = bool(quality["reliable"] and fitted["contrast"] >= .3
                and x.max() >= 2.1*fitted["pi_amplitude"])
    elif tool in ("sq.t1", "sq.t2_echo"):
        name = "t1_us" if tool == "sq.t1" else "t2_echo_us"
        fn = lambda t, base, amp, tau: base+amp*np.exp(-t/tau)
        fitted, quality = _fit_curve(x, y, fn, [[.05, .8, 35]],
            ([-.2, 0, 1], [1, 1, 200]), ["baseline", "contrast", name])
        if fitted:
            quality["reliable"] = bool(quality["reliable"] and fitted["contrast"] >= .3
                and x.max() >= 2*fitted[name])
    elif tool == "sq.xeb":
        fitted, quality = fit_xeb(obs)
    else:
        z = y+1j*q
        # FFT supplies a bounded frequency guess; nonlinear complex fitting avoids log-envelope bias.
        dt = float(np.diff(x).mean())
        frequency = float(np.fft.fftfreq(len(x), dt)[np.argmax(abs(np.fft.fft(z)))])
        nyquist = .5/dt
        def residual(pars):
            pred = pars[0]*np.exp(-x/pars[1])*np.exp(1j*(2*np.pi*pars[2]*x+pars[3]))
            return np.r_[(pred-z).real, (pred-z).imag]
        result = least_squares(residual, [.45, 20, frequency, float(np.angle(z[0]))],
            bounds=([0, 1, -nyquist, -2*np.pi], [1, 200, nyquist, 2*np.pi]), max_nfev=2000)
        covariance = np.linalg.pinv(result.jac.T@result.jac)*float(np.sum(result.fun**2))/(2*len(x)-4)
        stderr = float(np.sqrt(max(0, covariance[2, 2]))*1e6)
        denominator = max(float(np.sum(abs(z-z.mean())**2)), 1e-15)
        score = 1-float(np.sum(result.fun**2))/denominator
        fitted = {"frequency_correction_hz": float((obs["programmed_detuning_mhz"]-result.x[2])*1e6),
                  "t2_star_us": float(result.x[1])}
        quality = {"reliable": bool(result.success and score >= .9 and result.x[0] >= .15 and stderr < 5e4
                        and abs(result.x[2]) < .8*nyquist), "fit_r2": score,
                   "standard_errors": {"frequency_correction_hz": stderr,
                       "t2_star_us": float(np.sqrt(max(0, covariance[1, 1])))},
                   "uncertainty_method": "local_least_squares_covariance", "reason": "complex_ramsey_fit",
                   "nyquist_hz": nyquist*1e6}
    return {"fit_result": fitted, "quality": quality, "analysis_tool": REGISTRY[tool],
            "analysis_version": ANALYSIS_VERSION, "input_sha256": digest(observation),
            "physics_version": PHYSICS_VERSION, "sources": SOURCES}


def analyze_file(path, expected_sha256):
    """Explicit local artifact entry point; no latest-file lookup or code evaluation."""
    import hashlib
    raw = Path(path).read_bytes()
    if len(raw) > 4*1024*1024 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Artifact size/hash check failed")
    return analyze(json.loads(raw))
