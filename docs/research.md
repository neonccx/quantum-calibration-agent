# Research basis and adoption decisions

This project uses a reduced, explicitly simulated single-qubit calibration environment. A citation
supports a model form; it does not validate our synthetic priors, receiver noise or a hardware claim.

Primary scientific references:

- Krantz et al., *A Quantum Engineer's Guide to Superconducting Qubits*, Applied Physics Reviews 6,
  021318 (2019): https://doi.org/10.1063/1.5089550
- Probst et al., *Efficient and robust analysis of complex scattering data under noise in microwave
  resonators* (2015): https://doi.org/10.1063/1.4907935
- QuTiP Lindblad dynamics documentation: https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-master.html

The implemented `reduced-cqed-0.2` backend shares device parameters across S21, spectroscopy,
finite-pulse Rabi, signed Ramsey, T1, echo and IQ acquisition. QuTiP 5.3.1 independently checked 14
reduced-regime cases. The largest population/coherence discrepancy was 2.57e-7 against a 2e-6
tolerance; the cavity discrepancy was 2.57e-12 against 1e-8. This is not multilevel, EM or hardware
validation.

QMClaw was reviewed at pinned commit `18d7fa1594949a1203fca4866e651641bbde021f`.
All 144 authored files were statically read; 59 generated/dependency artifacts were only inventoried,
and none of the upstream services were executed. We adopted the experiment/analysis/quality-gate
separation and the multi-panel IQ presentation, not its executable command paths. The snapshot lacks
the external `lqms` IQ fitter and the frontend's imported `src/lib/api`. Static review also found unit,
status, concurrency, unsafe-code and mutable-latest-dataset hazards. It is therefore a design reference,
not a verified simulator or an industry-wide standard.

The final IQ gate freezes its discriminator on training shots, evaluates F0, F1, assignment fidelity,
visibility and SNR on held-out shots, and requires two independent passing acquisitions. Reports use
the recorded artifact and export PNG, SVG and PDF with an explicit `SIMULATED` label.

NVIDIA Ising/QCalEval remains a separate vision-plot comparison. No Ising checkpoint has been
downloaded or evaluated, and plot-question accuracy must not be presented as closed-loop calibration
success. A pinned snapshot and OpenAI-compatible VLM runner are implemented in
`scripts/prepare_qcaleval_snapshot.py` and `scripts/evaluate_qcaleval.py`. Without a declared LLM
judge they report only the reproducible Q2/Q4/Q5/Q6 subset; Q1/Q3 remain unscored rather than being
silently converted into an unofficial overall score.
