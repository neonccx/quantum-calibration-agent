"""Declared reduced single-qubit physics, in SI units.

Krantz et al., APR 6, 021318 (2019), sections III.B, IV.C, V.A-C:
https://doi.org/10.1063/1.5089550
Probst et al., RSI 86, 024706 (2015), Eq. 1:
https://doi.org/10.1063/1.4907935

Two levels, rotating-wave/Markov limits, short rectangular control pulses,
linear dispersive cavity and Gaussian receiver noise. Not a hardware twin.
Parameter priors are project assumptions, not measured device populations.
"""

from __future__ import annotations

import numpy as np

PHYSICS_VERSION = "reduced-cqed-0.2"
SOURCES = ["https://doi.org/10.1063/1.5089550", "https://doi.org/10.1063/1.4907935"]


def complex_notch(f, fr, ql, depth, phase=0., gain=1., phase_offset=0., delay_s=0., reference_hz=0.):
    f = np.asarray(f, dtype=float)
    return gain * np.exp(1j * (phase_offset - 2 * np.pi * (f - reference_hz) * delay_s)) * (
        1 - depth * np.exp(1j * phase) / (1 + 2j * ql * (f / fr - 1)))


def coherence_time(t1_s, tphi_s):
    if min(t1_s, tphi_s) <= 0:
        raise ValueError("Relaxation and dephasing times must be positive")
    return 1 / (1 / (2 * t1_s) + 1 / tphi_s)


def rabi_probability(omega_rad_s, detuning_rad_s, duration_s):
    omega = np.asarray(omega_rad_s, dtype=float)
    total = np.sqrt(omega ** 2 + detuning_rad_s ** 2)
    fraction = np.divide(omega ** 2, total ** 2, out=np.zeros_like(total), where=total != 0)
    return fraction * np.sin(total * duration_s / 2) ** 2


def steady_excitation(detuning_rad_s, omega_rad_s, t1_s, t2_s):
    saturation = omega_rad_s ** 2 * t1_s * t2_s
    return saturation / (2 * (1 + np.asarray(detuning_rad_s) ** 2 * t2_s ** 2 + saturation))


def free_coherence(time_s, detuning_hz, t1_s, tphi_s, static_noise_hz=0.):
    t = np.asarray(time_s)
    return np.exp(-t / coherence_time(t1_s, tphi_s) - .5 * (2 * np.pi * static_noise_hz * t) ** 2
                  + 2j * np.pi * detuning_hz * t)


def relaxation(time_s, initial, equilibrium, t1_s):
    return equilibrium + (initial - equilibrium) * np.exp(-np.asarray(time_s) / t1_s)


def cavity_integral(duration_s, steady, rate, initial=0.):
    """Integral of alpha(t) for d alpha/dt = -rate*(alpha-steady)."""
    duration = np.asarray(duration_s)
    return steady * duration + (initial - steady) * (-np.expm1(-rate * duration)) / rate


def iq_pointer(duration_s, jump_s, fr_hz, chi_hz, probe_hz, linewidth_hz, amplitude):
    """Integrated linear response; an excited-state T1 jump switches the cavity pole.

    jump_s=0 is ground preparation; jump_s>=duration is no decay during readout.
    State transitions change the cavity pole, not its instantaneous field.
    """
    kappa = 2 * np.pi * linewidth_hz
    rate0 = kappa / 2 + 2j * np.pi * (fr_hz - chi_hz - probe_hz)
    rate1 = kappa / 2 + 2j * np.pi * (fr_hz + chi_hz - probe_hz)
    drive = amplitude * kappa / 2
    steady0, steady1 = drive / rate0, drive / rate1
    first = np.minimum(np.asarray(jump_s), duration_s)
    at_jump = steady1 * (-np.expm1(-rate1 * first))
    return (cavity_integral(first, steady1, rate1)
            + cavity_integral(duration_s - first, steady0, rate0, at_jump)) / duration_s
