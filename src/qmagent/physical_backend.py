"""Measurement generation and analysis are separate, replayable operations."""

import copy
import numpy as np

from .physics import (PHYSICS_VERSION, SOURCES, complex_notch, coherence_time,
                      rabi_probability, steady_excitation, free_coherence, relaxation, iq_pointer)
from .contracts import TOOLS, validate_scan
from .analysis_tools import analyze
from .storage import digest


class PhysicalSimulator:
    backend_name = PHYSICS_VERSION

    def __init__(self, seed=20260903, noise_scale=1., profile="nominal"):
        if not np.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        if profile not in ("nominal", "drift", "quasistatic", "ambiguous"):
            raise ValueError("Unknown physical simulator profile")
        self.rng = np.random.default_rng(seed)
        self.noise_scale, self.profile, self.count = noise_scale, profile, 0
        self._truth = {"fr": 6.5e9+self.rng.uniform(-8e6, 8e6),
            "fq": 5e9+self.rng.uniform(-20e6, 20e6), "pi": self.rng.uniform(.18, .32),
            "t1": self.rng.uniform(25e-6, 45e-6), "tphi": self.rng.uniform(35e-6, 60e-6),
            "ql": self.rng.uniform(2500, 4000), "depth": self.rng.uniform(.4, .7),
            "gain": self.rng.uniform(.8, 1.2), "phase": self.rng.uniform(-1, 1),
            "delay": self.rng.uniform(10e-9, 40e-9), "mismatch": self.rng.uniform(-.15, .15),
            "g_hz": self.rng.uniform(95e6, 115e6), "anh_hz": -250e6,
            "receiver_sigma": .045, "readout_s": 1e-6, "pulse_s": 40e-9}

    def checkpoint(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "count": self.count}

    def restore(self, checkpoint):
        self.rng.bit_generator.state = copy.deepcopy(checkpoint["rng"])
        self.count = checkpoint["count"]

    def acquire(self, tool, state, scan):
        if tool not in TOOLS:
            raise ValueError("Unknown experiment")
        validate_scan(tool, state, scan)
        t = self._truth
        fq = t["fq"]+(self.count*5e4 if self.profile == "drift" else 0)
        t2 = coherence_time(t["t1"], t["tphi"])
        obs = {"tool": tool, "current_parameters": copy.deepcopy(state), "scan": copy.deepcopy(scan),
               "synthetic": True, "backend": self.backend_name,
               "model_assumptions": "two_level_RWA_Markov_linear_dispersive_short_pulse", "sources": SOURCES}
        noise = lambda n: self.rng.normal(0, .006*self.noise_scale, n)
        if tool in TOOLS[:2]:
            is_s21 = tool == TOOLS[0]
            center = scan.get("frequency_center_hz", state["readout_frequency_hz" if is_s21 else "drive_frequency_hz"])
            span = scan.get("frequency_span_hz", 40e6 if is_s21 else 100e6)
            x = np.linspace(center-span/2, center+span/2, 401)
            if is_s21:
                z = complex_notch(x, t["fr"], t["ql"], t["depth"], t["mismatch"],
                                  t["gain"], t["phase"], t["delay"], center)
            else:
                population = steady_excitation(2*np.pi*(x-fq), 2*np.pi*1e6, t["t1"], t2)
                if self.profile == "ambiguous":
                    population += .9*steady_excitation(2*np.pi*(x-fq-18e6), 2*np.pi*1e6, t["t1"], t2)
                z = .02+.9*population+0j
                obs["measurement_representation"] = "calibrated_population_receiver"
            z = z+noise(len(x))+1j*noise(len(x))
            unit = "Hz"
        elif tool == "sq.piamp":
            x = np.linspace(0, scan.get("amplitude_max", .8), 161)
            population = rabi_probability(np.pi*x/(t["pi"]*t["pulse_s"]),
                                          2*np.pi*(fq-state["drive_frequency_hz"]), t["pulse_s"])
            z = .03+.9*population+noise(len(x))+1j*noise(len(x))
            unit = "normalized_amplitude"
        elif tool == "sq.ramsey_df":
            x = np.linspace(0, scan.get("delay_max_us", 12), 241)
            contrast = np.sin(np.pi*state["pi_over_2_amplitude"]/t["pi"])**2
            programmed = .3e6
            z = .45*contrast*free_coherence(x*1e-6, programmed-(fq-state["drive_frequency_hz"]),
                t["t1"], t["tphi"], 18000 if self.profile == "quasistatic" else 0)
            z += noise(len(x))+1j*noise(len(x))
            obs["programmed_detuning_mhz"] = programmed/1e6
            obs["measurement_representation"] = "phase_cycled_coherence_quadratures"
            unit = "us"
        elif tool in ("sq.t1", "sq.t2_echo"):
            x = np.linspace(0, scan.get("delay_max_us", 120), 161)
            excitation = float(rabi_probability(np.pi*state["pi_amplitude"]/(t["pi"]*t["pulse_s"]),
                                                2*np.pi*(fq-state["drive_frequency_hz"]), t["pulse_s"]))
            if tool == "sq.t1":
                population = relaxation(x*1e-6, excitation, .005, t["t1"])
            else:
                # Ideal echo refocuses static detuning, not Markov dephasing.
                population = .5+.5*excitation*np.exp(-x*1e-6/t2)
            z = .03+.9*population+noise(len(x))+1j*noise(len(x))
            unit = "us"
        else:
            shots = scan.get("shots", 1024)
            if type(shots) is not int or not 256 <= shots <= 4096:
                raise ValueError("Invalid IQ shot count")
            excitation = float(rabi_probability(np.pi*state["pi_amplitude"]/(t["pi"]*t["pulse_s"]),
                                                2*np.pi*(fq-state["drive_frequency_hz"]), t["pulse_s"]))
            thermal = float(relaxation(state["relaxation_delay_us"]*1e-6, .5, .005, t["t1"]))
            delta = fq-t["fr"]
            chi = t["g_hz"]**2*t["anh_hz"]/(delta*(delta+t["anh_hz"]))
            groups = []
            for population in (thermal, thermal+(1-2*thermal)*excitation):
                excited = self.rng.random(shots) < population
                jump = self.rng.exponential(t["t1"], shots)*excited
                z = iq_pointer(t["readout_s"], jump, t["fr"], chi, state["readout_frequency_hz"],
                               t["fr"]/t["ql"], state["readout_amplitude"])
                z *= np.exp(1j*t["phase"])
                sigma = t["receiver_sigma"]*self.noise_scale/np.sqrt(t["readout_s"]/1e-6)
                z += sigma*(self.rng.standard_normal(shots)+1j*self.rng.standard_normal(shots))
                groups.append(z)
            z = np.concatenate(groups)
            obs["measurement"] = {"i": z.real.tolist(), "q": z.imag.tolist(),
                "prepared_state": [0]*shots+[1]*shots, "unit": "normalized_receiver_voltage"}
            obs["acquisition"] = {"integration_s": t["readout_s"], "shots_per_state": shots}
            self.count += 1
            return obs
        obs["sweep"] = {"values": x.tolist(), "unit": unit}
        obs["measurement"] = {"i": z.real.tolist(), "q": z.imag.tolist(), "unit": "a.u."}
        self.count += 1
        return obs

    def measure(self, tool, state, scan):
        raw = self.acquire(tool, state, scan)
        result = analyze(raw)
        return raw | result | {"raw_artifact": {"id": "sha256:"+digest(raw), "sha256": digest(raw)},
            "tool_trace": [{"tool": tool, "operation": "acquire", "output_sha256": digest(raw)},
                           {"tool": result["analysis_tool"], "operation": "analyze", "input_sha256": digest(raw)}]}
