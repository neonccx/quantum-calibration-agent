#!/usr/bin/env python3
"""Cross-check reduced formulas against independently integrated QuTiP dynamics.

No instruments or models are loaded. This validates specified reduced regimes,
not transmon leakage, real receiver behavior, or complete simulator realism.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

from qmagent.physics import (PHYSICS_VERSION, coherence_time, rabi_probability,
                            steady_excitation, free_coherence, relaxation, iq_pointer)


def validate():
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    lowering = zero*one.dag()
    projector = one*one.dag()
    checks = []
    def record(name, actual, expected, tolerance=2e-6):
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        checks.append({"name": name, "max_abs_error": error, "tolerance": tolerance, "passed": error < tolerance})
    options = {"atol": 1e-10, "rtol": 1e-9, "nsteps": 100000}
    # SI seconds and rad/s throughout; explicit |1><1| avoids spin-label ambiguity.
    for detuning in (-2e6, 0, 2e6):
        omega = 2*np.pi*8e6
        h = np.pi*detuning*qt.sigmaz()+omega/2*qt.sigmax()
        times = np.linspace(0, 150e-9, 101)
        result = qt.sesolve(h, zero, times, e_ops=[projector], options=options)
        record(f"rabi_detuning_{detuning:g}_hz", result.expect[0], rabi_probability(omega, 2*np.pi*detuning, times))
    for t1, tphi in ((30e-6, 40e-6), (45e-6, 60e-6)):
        collapse = [np.sqrt(1/t1)*lowering, np.sqrt(1/(2*tphi))*qt.sigmaz()]
        times = np.linspace(0, 120e-6, 201)
        result = qt.mesolve(0*qt.sigmaz(), one, times, collapse, e_ops=[projector], options=options)
        record(f"t1_{t1:g}", result.expect[0], relaxation(times, 1, 0, t1))
        plus = (zero+one).unit()
        for detuning in (-.3e6, .3e6):
            result = qt.mesolve(np.pi*detuning*qt.sigmaz(), plus, times, collapse,
                               e_ops=[qt.sigmax(), qt.sigmay()], options=options)
            measured = np.asarray(result.expect[0])+1j*np.asarray(result.expect[1])
            record(f"ramsey_signed_{t1:g}_{detuning:g}", measured, free_coherence(times, detuning, t1, tphi))
        omega, delta = 2*np.pi*1e6, 2*np.pi*2e6
        rho = qt.steadystate(delta/2*qt.sigmaz()+omega/2*qt.sigmax(), collapse)
        record(f"spectroscopy_steady_{t1:g}", qt.expect(projector, rho), steady_excitation(delta, omega, t1, coherence_time(t1, tphi)))
        echo_delays = np.linspace(0, 80e-6, 21)
        echoes = []
        for delay in echo_delays:
            if delay == 0:
                echoes.append(1.)
                continue
            h = np.pi*.2e6*qt.sigmaz()
            first = qt.mesolve(h, plus, [0, delay/2], collapse, options=options).states[-1]
            flip = qt.sigmax()*first*qt.sigmax()
            end = qt.mesolve(h, flip, [0, delay/2], collapse, options=options).states[-1]
            echoes.append(float(qt.expect(qt.sigmax(), end)))
        record(f"echo_refocus_{t1:g}", echoes, np.exp(-echo_delays/coherence_time(t1, tphi)))
    # Independent time-domain ODE, including continuity at a T1 jump.
    duration, jump = 1e-6, .4e-6
    fr, chi, probe, width, amplitude = 6.5e9, -1e6, 6.5002e9, 2e6, .4
    kappa = 2*np.pi*width
    def integrate(start, end, state, pole):
        rate = kappa/2+2j*np.pi*(pole-probe)
        return solve_ivp(lambda time, value: [amplitude*kappa/2-rate*value[0], value[0]],
                         (start, end), state, rtol=1e-10, atol=1e-12).y[:, -1]
    first = integrate(0, jump, np.array([0j, 0j]), fr+chi)
    end = integrate(jump, duration, first, fr-chi)
    record("integrated_IQ_with_T1_jump", end[1]/duration,
           iq_pointer(duration, jump, fr, chi, probe, width, amplitude), 1e-8)
    return {"physics_version": PHYSICS_VERSION, "qutip_version": qt.__version__,
            "all_passed": all(item["passed"] for item in checks), "checks": checks,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Two-level Markov/RWA dynamics and linear cavity ODE; not hardware validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_passed"] else 1)
