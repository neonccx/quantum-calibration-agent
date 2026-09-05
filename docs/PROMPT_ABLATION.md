# Calibration prompt ablation

The preregistered B0/B1/F0/F1 comparison changes one factor: the calibration system
instruction. It does not change the checkpoint, adapter, native function schema, public context,
controller, simulator, seeds, test rows, decoding settings or generation budget.

| Arm | Weights | `--prompt-profile` |
| --- | --- | --- |
| B0 | Unmodified Nanbeige4.2-3B | `minimal` |
| B1 | Unmodified Nanbeige4.2-3B | `skill` |
| F0 | Frozen LoRA adapter | `minimal` |
| F1 | Frozen LoRA adapter | `skill` |

The minimal prompt asks for one safe native call based on the public context, forbids invented
measurements and requires an English reason. It contains no stage order, fitted-parameter update
formula, quality threshold or recovery rule. The `skill` prompt is the full deployed calibration
instruction used to construct v2 training messages.

Every frozen-context result records `prompt_profile` and a SHA-256 of the exact system prompt.
Every closed-loop run records the profile in `run_config.json`. Compare invalid calls, next action,
arguments and controller executability first; report closed-loop IQ acceptance separately. A model
trained on the skill prompt may legitimately degrade under the minimal prompt—this is the effect the
F0/F1 contrast is designed to measure, not a new training result.
