# Flux and XEB-style v3 assumptions

## Workflow boundary

Runtime schema `runtime-0.2`, protocol `calibration-step-0.3` and physics
`reduced-cqed-flux-xeb-0.3` remain simulation-only and single-qubit-only:

`S21 → S21_ZPA2D → spectroscopy → PiAmp → Ramsey → T1 → Echo → XEB-style → IQraw × 2`.

There is no coupler, no `zpa2d_coupler`, no two-qubit gate and no hardware transport. A controller,
not the model, validates shapes, bounds, stage prerequisites, fit quality, XEB acceptance and the two
independent IQ batches.

## Flux-tunable transmon

For normalized flux `phi = (zpa - sweet_zpa) / flux_period_zpa`, the hidden simulator uses

`EJ(phi) = EJ_sum * sqrt(cos(pi phi)^2 + d^2 sin(pi phi)^2)`

and the leading transmon approximation

`f01(phi) = sqrt(8 EC EJ(phi)) - EC`.

The hidden asymmetry `d`, period and sweet offset are sampled per virtual device. A reduced
dispersive shift `chi = g^2 alpha / (Delta (Delta + alpha))` moves the simulated resonator ridge.
The acquisition returns a rectangular complex S21 grid indexed by ascending ZPA and frequency axes.
The registered analysis extracts each measured notch, fits a periodic ridge without simulator access,
and returns `sweet_spot_zpa`, `readout_frequency_hz` and fit quality. The controller commits those as
`state.z_bias` and `state.readout_frequency_hz`; subsequent spectroscopy computes `f01` at that bias.

This is a reduced formula-level environment, not a multilevel circuit quantization, EM simulation,
flux-line transfer-function calibration or hardware digital twin. Primary basis: Koch et al.,
PRA 76, 042319 (2007), https://doi.org/10.1103/PhysRevA.76.042319, and Krantz et al., APR 6,
021318 (2019), https://doi.org/10.1063/1.5089550.

## Single-qubit XEB-style proxy

For each circuit depth, the simulator averages a noisy single-qubit random-circuit fidelity proxy and
the analysis fits

`F(m) = offset + amplitude * p_cycle^m`.

Reliability requires R² ≥ 0.90, standard error of `p_cycle` ≤ 0.005 and adequate depth coverage.
Acceptance additionally requires `p_cycle ≥ 0.985`; otherwise the rule baseline escalates because this
project does not expose a hidden gate-retuning knob.

The stage borrows the decay shape and terminology of XEB, based on Arute et al., Nature 574,
505–510 (2019), https://doi.org/10.1038/s41586-019-1666-5. It is **not** equivalent to multiqubit
random-circuit-sampling XEB. Standard Clifford randomized benchmarking is generally more common for
single-qubit gate characterization; see Knill et al., PRA 77, 012307 (2008),
https://doi.org/10.1103/PhysRevA.77.012307.
