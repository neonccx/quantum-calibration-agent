# Explicit fit-update tool

Enable in a batch model run with `--fit-update-tool`. The default remains the frozen v2
protocol so existing evaluations remain comparable. Local persisted configuration also accepts
`fit_update_tool: true` for model sessions.

The additional function is `calibration.step_from_fit` with `next_tool`, `scan`, and `reason`.
It computes parameter updates from the current reliable analyzed observation: resonator/drive
frequency, pi and pi/2 amplitude, signed Ramsey correction, T1/reset delay or echo time.
The model chooses this function explicitly; ordinary `calibration.step` predictions are not
silently corrected. Terminal actions continue through `calibration.step` with empty updates.

The calculation rejects unreliable fits, stale Ramsey drive frequencies and nonfinite results.
The existing controller validates the resulting action, prerequisites, ranges and budgets before
any acquisition or parameter commit. Its generation trace preserves the original native call and
records protocol `calibration-step-fit-0.1`, so tool assistance can be separated from v2 results.

Current status: integrated and unit tested; 158 local regression tests passed. A candidate
curriculum reserializes all 2,705 frozen v2 decisions and contains 1,263 explicit fit-tool targets.
The installed Nanbeige tokenizer audit passed every row, including exact training/runtime prompt
equality, assistant-loss boundaries and decoded controller-action equivalence (maximum 7,724
tokens). The candidate remains marked not training-ready until the model-selection probe finishes.

Real-model
regression/closed-loop verification is queued after the frozen full evaluation on H100, in the
isolated source copy `/home/caochuangxin/bishe/qcal-fit-tool-20260905/ordered_validation/`.
An initial concurrent probe was stopped because its model was offloaded to CPU under memory
contention; it provides no success or latency result. The ordered probe requires the model to
select the explicit new tool and pass controller validation for both known failing contexts,
then measures the same three new-device seeds as the unassisted comparison.

This is a tool-interface experiment using the existing adapter. It does not establish that the
adapter has learned the new function reliably; retraining the tool-call targets may still be needed.
