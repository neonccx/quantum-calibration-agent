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

## Completed result

The full 277-example test and 478-example OOD arms completed on 2026-09-05. Under the minimal prompt,
SFT next-tool agreement was 0.4007 on test and 0.4331 on OOD; controller-executable rates were 0.2996
and 0.2908. Under the deployed skill prompt, the corresponding next-tool rates were 1.0000 and
0.9289, while controller-executable rates were 0.9819 and 0.9226.

Both minimal-prompt fresh-loop arms accepted 0/3 episodes; the controller rejected all six unsafe or
out-of-order actions. In contrast, the skill-prompt SFT accepted 2/3 with the original action schema,
and the registered deterministic fit-update path accepted 3/3. The current release is therefore a
composite system: frozen LoRA weights plus the complete workflow instruction, deterministic tools and
controller. It is not a prompt-independent autonomous calibration model.
