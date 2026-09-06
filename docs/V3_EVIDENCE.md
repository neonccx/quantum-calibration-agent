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

## GPU status

The v3 H100 pipeline is defined in the training repository. At the time of this local verification,
both previously supplied SSH endpoints (`10.130.144.39` and `10.130.144.45`) timed out before SSH
authentication. Therefore no v3 baseline, tokenizer audit, optimization or model metric is claimed
in this document. When connectivity returns, the ordered pipeline records the unmodified baseline
before creating a new adapter and writes every stage into a new run directory.
