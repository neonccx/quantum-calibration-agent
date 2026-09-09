# v3 implementation evidence

## Local verification (2026-09-06)

Commands:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src /Users/soraka/lab/3/2.微调数据集构建/.conda-env/bin/python \
  scripts/validate_dynamics.py --output runs/v3_20260906_local/dynamics.json
PYTHONPATH=src python3 -m qmagent.cli run --policy rule --backend physical \
  --seed 2026090700 --episodes 20 --max-steps 40 --max-tool-calls 10 \
  --output-dir runs/v3_20260906_local/rule_20
```

Results:

- 175 tests passed in 23.84 seconds in the final local regression run.
- 14/14 QuTiP reduced-dynamics checks passed. These pre-existing checks cover two-level
  Rabi/Ramsey/T1/Echo and linear-cavity equations; they do not independently validate the new flux
  ridge or XEB proxy.
- 20/20 fresh nominal rule-policy episodes were accepted, with no invalid action, tool error,
  policy error, escalation or budget exhaustion.
- Mean successful experiment count was 12.15. Every accepted episode completed S21, S21-ZPA2D,
  spectroscopy, repeated PiAmp verification, Ramsey, T1, Echo, XEB-style decay and two independent
  IQ batches.

This establishes executable simulator/controller behavior, not Nanbeige policy success and not
hardware validity. The rule policy is deterministic and shares the public workflow contract.

## Dataset v3

Generation and static audit commands:

```bash
PYTHONPATH=src python3 scripts/build_dataset_v3.py --devices 32 \
  --output /Users/soraka/lab/3/nanbeige-calibration-sft/dataset_v3
PYTHONPATH=src python3 scripts/audit_dataset_v3.py \
  /Users/soraka/lab/3/nanbeige-calibration-sft/dataset_v3
```

The audit passed 1,839 examples, 1,598 referenced content-addressed artifacts, disjoint device
splits, rectangular ZPA2D grids, presence of both new native-call targets and absence of named
simulator-truth fields in model-visible rows. Exact counts are recorded in the dataset manifest and
`dataset_v3/audit_report.json`.

## Completed model evaluation (2026-09-09)

The independent v3 run completed on one RTX 5090. It recorded the unmodified base-model test and OOD
baselines before training, then ran one epoch (103 optimizer steps) of BF16 LoRA with rank 32, alpha
64, dropout 0.05, effective batch 10 and assistant-only sparse projection. Training loss was
0.1177623.

| Split / policy | Valid | Next tool | Arguments | Controller-executable | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| test base (182) | 0.9176 | 0.5604 | 0.1044 | not scored | 3.94 s |
| test SFT (182) | 1.0000 | 0.9780 | 0.9341 | 0.9890 | 3.52 s |
| OOD base (450) | 0.9311 | 0.6000 | 0.0889 | not scored | 3.68 s |
| OOD SFT (450) | 1.0000 | 0.9089 | 0.8689 | 0.9222 | 3.48 s |

The same three fresh simulator seeds were then used for both policies. Base accepted 0/3 episodes
(two invalid actions and one policy error). SFT accepted 2/3 in 13 and 12 experiments, while the third
recorded a policy error after three experiments. Both accepted episodes finished the full S21,
ZPA2D, spectroscopy, PiAmp, Ramsey, T1, Echo, XEB-style and IQraw workflow and ended with two
independent held-out IQ passes.

The shared server was under contention during evaluation. A terminated SFT OOD run was resumed only
after validating immutable configuration, dataset hash and the exact 100-record prediction-ID prefix;
the remaining 350 records were appended. Training and previously completed results were not rerun or
overwritten. Nonzero closed-loop process returns are retained scientific failures, not missing data.

The v3 adapter is published as
[release v0.2.0](https://github.com/neonccx/nanbeige-calibration-sft/releases/tag/v0.2.0). Exact metrics,
artifact hashes and limitations are in the training repository's
[path-free evidence JSON](https://github.com/neonccx/nanbeige-calibration-sft/blob/main/evaluation/public_evidence_v3_20260909.json).

![Recorded v3 LoRA IQ discrimination](assets/evaluation_v3_20260909/iq_report.png)

The displayed episode has F0 0.9824, F1 0.9785, assignment fidelity 0.9805, visibility 0.9609 and
SNR amplitude 3.11. The image is recomputed from saved simulated IQ shots and is not hardware data.
