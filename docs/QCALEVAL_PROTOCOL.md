# QCalEval / Ising comparison protocol

This is a separate visual-understanding evaluation. It is not the simulated closed-loop
calibration acceptance test and its numbers must never be merged with controller success,
fit quality, or final IQ assignment fidelity.

## Pinned basis

- Official evaluator: `NVIDIA/QCalEval` commit
  `e9e9b9eb8b95f93e0db1c578e699fc212cf0bb84` (Apache-2.0).
- Dataset: `nvidia/QCalEval` (CC-BY-4.0), pinned by an explicit immutable Hugging Face
  commit when a snapshot is prepared. The reviewed 243-entry revision is
  `b611794244a251c47fc67eb801b92f05722c369c`.
- The official zero-shot protocol sends each plot and each of six questions as a separate
  OpenAI-compatible chat-completions request.

Q1 describes plot structure, Q2 classifies experimental outcome, Q3 gives scientific
reasoning, Q4 assesses fit reliability, Q5 extracts parameters, and Q6 returns a
family-specific calibration diagnosis. Official Q1 is half programmatic and half LLM judged;
official Q3 is LLM judged. Q2, Q4, Q5 and Q6 have reproducible programmatic components.

## Prepare once

Install the preparation dependencies with `python -m pip install -e '.[qcaleval]'` in an
isolated environment, then freeze a
specific dataset revision. Never use a moving `main` revision in a reported experiment.

```bash
PYTHONPATH=src python scripts/prepare_qcaleval_snapshot.py \
  --revision b611794244a251c47fc67eb801b92f05722c369c \
  --output-dir /absolute/path/qcaleval_b611794
```

The snapshot stores the six prompts and answers, normalized PNGs, per-image hashes, Q5
tolerances and a whole-snapshot SHA-256. The evaluator rejects changed images or metadata.

## Validate before serving a large model

```bash
PYTHONPATH=src python scripts/evaluate_qcaleval.py \
  --snapshot /absolute/path/qcaleval_<commit>/snapshot.json \
  --api-base http://127.0.0.1:8000/v1/chat/completions \
  --model-id exact-served-model-id \
  --output runs/qcaleval/plan.json \
  --limit 1 --dry-run
```

For a real run, remove `--dry-run` and set `QCAL_VLM_API_KEY` in the environment. The key is
never accepted on the command line or written to results. Every prompt, image hash, raw API
response, request hash and latency is retained.

Result paths are append-only: the runner refuses to overwrite an existing file. Remote endpoints
must use HTTPS; plain HTTP is accepted only for a loopback vLLM/NIM service.

The local aggregate is explicitly labelled `Q2/Q4/Q5/Q6 programmatic subset`. Q1 and Q3
remain `null` until an independently declared judge protocol is run. Consequently this result
is useful for regression and engineering comparisons, but is not an official QCalEval score.
No Ising weights are downloaded as part of this protocol.
