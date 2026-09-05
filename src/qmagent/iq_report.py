"""Deterministic six-panel IQ report from measured shots, never LLM numbers."""

import json
import math
from pathlib import Path

import numpy as np

from .simulator import iq_gate
from .storage import atomic_json, digest


def gates_equivalent(actual, recorded):
    """Allow only numerical roundoff across platforms; gate decisions remain exact."""
    if isinstance(actual, dict) and isinstance(recorded, dict):
        return actual.keys() == recorded.keys() and all(gates_equivalent(actual[k], recorded[k]) for k in actual)
    if isinstance(actual, list) and isinstance(recorded, list):
        return len(actual) == len(recorded) and all(gates_equivalent(a, b) for a, b in zip(actual, recorded))
    if type(actual) is bool or type(recorded) is bool:
        return type(actual) is type(recorded) and actual == recorded
    if isinstance(actual, (int, float)) and isinstance(recorded, (int, float)):
        return math.isfinite(actual) and math.isfinite(recorded) and math.isclose(actual, recorded, rel_tol=1e-12, abs_tol=1e-14)
    return type(actual) is type(recorded) and actual == recorded


def analyze_iq(measurement):
    gate = iq_gate(measurement)
    if "classifier" not in gate:
        raise ValueError("Cannot draw a discriminant for degenerate IQ centroids")
    points = np.column_stack([measurement["i"], measurement["q"]])
    labels = np.asarray(measurement["prepared_state"])
    groups = [points[labels == label] for label in (0, 1)]
    axis = np.asarray(gate["classifier"]["axis"])
    midpoint = gate["classifier"]["threshold"]
    train = [group[::2] @ axis for group in groups]
    test = [group[1::2] @ axis for group in groups]
    values = np.unique(np.concatenate(train))
    cuts = np.r_[np.nextafter(values[0], -np.inf), values[:-1] / 2 + values[1:] / 2,
                 np.nextafter(values[-1], np.inf)]
    cdfs = [np.searchsorted(np.sort(sample), cuts, side="left") / len(sample) for sample in train]
    visibility = cdfs[0] - cdfs[1]
    candidates = np.flatnonzero(visibility == visibility.max())
    best = candidates[np.argmin(abs(cuts[candidates] - midpoint))]
    threshold = float(cuts[best])
    f0 = float(np.mean(test[0] < threshold))
    f1 = float(np.mean(test[1] >= threshold))
    def wilson(successes, total):
        z = 1.959963984540054
        p = successes/total
        scale = 1+z*z/total
        center = (p+z*z/(2*total))/scale
        radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/scale
        return [float(center-radius), float(center+radius)]
    correct = [int(np.sum(test[0] < midpoint)), int(np.sum(test[1] >= midpoint))]
    intervals = [wilson(correct[i], len(test[i])) for i in (0, 1)]
    return {"gate": gate, "measurement_sha256": digest(measurement),
            "snr_amplitude": gate["metrics"]["snr"], "snr_power": gate["metrics"]["snr"] ** 2,
            "max_visibility_diagnostic": {"threshold": threshold,
                "training_visibility": float(visibility[best]), "held_out_f0": f0, "held_out_f1": f1,
                "held_out_visibility": f0 + f1 - 1, "used_for_controller_acceptance": False},
            "split": "Within each prepared label: even shots train, odd shots evaluate",
            "state_preparation_error": None,
            "held_out_confusion_counts": [[correct[0], len(test[0])-correct[0]], [len(test[1])-correct[1], correct[1]]],
            "confidence_intervals": {"level": .95, "method": "Wilson binomial per prepared label; conditional on training classifier",
                "f0": intervals[0], "f1": intervals[1], "used_for_controller_acceptance": False},
            "caveat": "Assignment errors are not separately identified SPAM errors. Gaussian curves are descriptive only."}


def render_iq_report(measurement, directory, *, policy, status, step=None, consecutive_passes=0):
    # Lazy import keeps remote clients and non-plotting runs lightweight.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from scipy.stats import norm

    directory = Path(directory)
    for name in ("iq_report.png", "iq_report.svg", "iq_report.pdf", "iq_metrics.json"):
        if (directory / name).exists():
            raise ValueError("Refusing to overwrite an IQ report")
    analysis = analyze_iq(measurement)
    analysis.update(policy=policy, controller_status=status, step=step,
                    consecutive_iq_passes=consecutive_passes, synthetic=True)
    points = np.column_stack([measurement["i"], measurement["q"]])
    labels = np.asarray(measurement["prepared_state"])
    groups = [points[labels == label] for label in (0, 1)]
    gate = analysis["gate"]
    axis = np.asarray(gate["classifier"]["axis"])
    perpendicular = np.array([-axis[1], axis[0]])
    centers = [g[::2].mean(axis=0) for g in groups]
    origin = (centers[0] + centers[1]) / 2
    mid = gate["classifier"]["threshold"]
    optimal = analysis["max_visibility_diagnostic"]["threshold"]
    rotated = [np.column_stack(((g - origin) @ axis, (g - origin) @ perpendicular)) for g in groups]
    train = [g[::2] @ axis for g in groups]
    test = [g[1::2] @ axis for g in groups]
    colors = ["#e82736", "#224eff"]
    names = ["prepared |0>", "prepared |1>"]
    metrics = gate["metrics"]
    fig = Figure(figsize=(8.4, 11.2), facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 2)
    fig.suptitle(f"IQ discrimination | SIMULATED | {policy.upper()}\n"
                 f"Controller: {status}   |   step {step}   |   consecutive IQ passes {consecutive_passes}/2",
                 fontsize=13, color="#a71928", fontweight="bold")
    for ax in axes.flat:
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=8)
        ax.set_axisbelow(True)
    for index, (g, r) in enumerate(zip(groups, rotated)):
        axes[0, 0].scatter(*g.T, s=5, c=colors[index], alpha=0.48, label=names[index], rasterized=True)
        axes[0, 1].scatter(*r.T, s=5, c=colors[index], alpha=0.48, rasterized=True)
        axes[1, 0].scatter(*g[1::2].T, s=6, c=colors[index], alpha=0.5, rasterized=True)
    centers_array = np.asarray(centers)
    axes[0, 0].plot(*centers_array.T, "ko-", markersize=5, linewidth=1)
    axes[0, 0].set(title="A  Raw IQ shots & training centroids", xlabel="I (a.u.)", ylabel="Q (a.u.)")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(title=f"B  Rotated IQ | SNR amp={analysis['snr_amplitude']:.2f}\n"
                        f"Separation={np.linalg.norm(centers_array[1]-centers_array[0]):.3f}; "
                        f"SNR power={analysis['snr_power']:.2f}", xlabel="Discrimination axis (a.u.)", ylabel="Orthogonal axis (a.u.)")
    low, high = points.min(axis=0), points.max(axis=0)
    span = max(float(np.max(high - low)), 0.01)
    display_center = (low+high)/2
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xlim(display_center[0]-span*.56, display_center[0]+span*.56)
        ax.set_ylim(display_center[1]-span*.56, display_center[1]+span*.56)
        ax.set_aspect("equal", adjustable="box")
    rotated_span = float(np.max(np.abs(np.concatenate(rotated))))*1.08
    axes[0, 1].set_xlim(-rotated_span, rotated_span)
    axes[0, 1].set_ylim(-rotated_span, rotated_span)
    axes[0, 1].set_aspect("equal", adjustable="box")
    endpoints = axis * mid + np.array([-2 * span, 2 * span])[:, None] * perpendicular
    axes[1, 0].plot(*endpoints.T, "k--", linewidth=1)
    for index, g in enumerate(groups):
        heldout = g[1::2]
        wrong = ((heldout @ axis >= mid) != index)
        axes[1, 0].scatter(*heldout[wrong].T, marker="x", s=22, c="black", linewidths=.7)
    axes[1, 0].set(title=f"C  Held-out decisions | F0={metrics['f0']:.3f}, F1={metrics['f1']:.3f}",
                   xlabel="I (a.u.)", ylabel="Q (a.u.)")
    all_proj = np.concatenate(test)
    bins = np.linspace(all_proj.min(), all_proj.max(), 55)
    for index, sample in enumerate(test):
        axes[1, 1].hist(sample, bins=bins, color=colors[index], alpha=.4, label=names[index])
    axes[1, 1].axvline(mid, color="black", linestyle="--", linewidth=1, label="controller midpoint")
    axes[1, 1].axvline(optimal, color="#11875d", linestyle=":", linewidth=1, label="training max-V")
    axes[1, 1].set(title=f"D  Held-out projection\nAssignment fidelity={metrics['assignment_fidelity']:.3%}",
                   xlabel="Projection (a.u.)", ylabel="Shots / bin")
    axes[1, 1].legend(fontsize=7)
    grid = np.linspace(min(np.concatenate(train).min(), all_proj.min()),
                       max(np.concatenate(train).max(), all_proj.max()), 600)
    cdfs = [np.searchsorted(np.sort(s), grid, side="left") / len(s) for s in train]
    for index, cdf in enumerate(cdfs):
        axes[2, 0].plot(grid, cdf, color=colors[index], label=f"CDF {index} (train)")
    axes[2, 0].plot(grid, cdfs[0] - cdfs[1], color="black", label="V(t), training only")
    axes[2, 0].axvline(optimal, color="#11875d", linestyle=":", linewidth=1)
    axes[2, 0].set(title=f"E  Max training V={analysis['max_visibility_diagnostic']['training_visibility']:.3f}\n"
                        f"Controller held-out V={metrics['visibility']:.3f} (unchanged)",
                   xlabel="Threshold (a.u.)", ylabel="CDF / visibility", ylim=(-.05, 1.05))
    axes[2, 0].legend(fontsize=7)
    pdfs = []
    for index, sample in enumerate(train):
        std = max(float(sample.std(ddof=1)), 1e-9)
        pdf = norm.pdf(grid, float(sample.mean()), std)
        pdfs.append(pdf)
        axes[2, 1].plot(grid, pdf, color=colors[index], label=f"Gaussian {index} (train)")
    axes[2, 1].fill_between(grid, np.minimum(*pdfs), color="#777777", alpha=.3, label="model overlap")
    axes[2, 1].axvline(mid, color="black", linestyle="--", linewidth=1)
    axes[2, 1].set(title="F  Gaussian approximation\nNot a SPAM / state-error decomposition",
                   xlabel="Projection (a.u.)", ylabel="Probability density")
    axes[2, 1].legend(fontsize=7)
    fig.supxlabel("Even shots train; odd shots evaluate. Max-V is diagnostic only.\n"
                  + ("Rule-policy simulation. " if policy == "rule" else "Model-policy simulation. ")
                  + "All points come from the saved measurement.",
                  fontsize=8)
    for ax in axes.flat:
        ax.title.set_fontsize(9)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
    directory.mkdir(parents=True, exist_ok=True)
    from matplotlib import rc_context
    with rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        fig.savefig(directory / "iq_report.png", dpi=150)
        fig.savefig(directory / "iq_report.svg")
        fig.savefig(directory / "iq_report.pdf")
    atomic_json(directory / "iq_metrics.json", analysis)
    fig.clear()
    return analysis


def from_trajectory(trajectory, directory, *, policy, status, consecutive_passes=0):
    latest = None
    with Path(trajectory).open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "experiment" and event.get("tool") == "sq.iqraw":
                latest = event
    if latest is None:
        return None  # No IQ measurement means no invented IQ result image.
    return render_iq_report(latest["observation"]["measurement"], directory, policy=policy,
                            status=status, step=latest["step"], consecutive_passes=consecutive_passes)
