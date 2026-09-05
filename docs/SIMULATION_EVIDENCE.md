# Simulation evidence

## Scope

This evidence uses the reduced-cQED synthetic backend. It demonstrates that the typed Agent,
registered fit-update tool, deterministic controller and frozen Nanbeige LoRA policy can execute the
complete seven-stage workflow. It does not establish performance on a physical device.

## Fresh-seed result

Three seeds excluded from the training/test/OOD dataset were evaluated with a budget of eight calls
per experiment and 30 total steps. The registered `calibration.step_from_fit` policy accepted all
three episodes in nine experiments:

| Seed | Experiments | Result | Final IQ fidelity | Visibility | SNR amplitude |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2026090500 | 9 | accepted | 0.9873 | 0.9746 | 3.51 |
| 2026090501 | 9 | accepted | 0.9844 | 0.9688 | 3.28 |
| 2026090502 | 9 | accepted | 0.9756 | 0.9512 | 3.07 |

Every episode completed S21, spectroscopy, PiAmp, Ramsey, T1 and T2 Echo, then passed two
independent IQraw batches. The classifier was trained on even-indexed shots and evaluated on the
held-out odd-indexed shots: 512 test shots per prepared state.

The original SFT action policy, before introducing deterministic fit arithmetic, accepted two of
the same three seeds. Its remaining action was rejected safely for requesting T2 Echo before T1 had
been accepted by the controller. The base checkpoint accepted none of the three. Because this is a
three-seed diagnostic, report the individual outcomes rather than presenting 3/3 as a population
success estimate.

## Figure provenance

[`iq_report.png`](assets/evaluation_20260905/iq_report.png), its SVG counterpart and
[`iq_metrics.json`](assets/evaluation_20260905/iq_metrics.json) were regenerated from the saved
episode-0 trajectory by `scripts/build_iq_report.py`. The exporter refuses an existing output path
and rejects the report if the recomputed IQ gate differs from the original recorded result.

The six panels show raw shots and centroids, rotated IQ, held-out classification errors, held-out
projection histograms, the training-only maximum-visibility diagnostic and descriptive Gaussian
fits. The Gaussian overlap is not a SPAM decomposition and is not used to change controller
acceptance.
