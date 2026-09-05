"""Real GPU equivalence/performance tests for the production graph implementation."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from qmagent.policies import HuggingFacePolicy, CHAT_PROMPT, SYSTEM_PROMPT, RulePolicy, model_context
    from qmagent.runtime import AgentRunner
    from qmagent.simulator import AnalyticSimulator
    policy = HuggingFacePolicy(args.model, trust_remote_code=True, decode_backend="auto")
    if policy.graph_generator is None:
        raise RuntimeError(policy.graph_reason)
    cases = [
        ("english", "Explain how a Python dictionary works in about 80 words.", 32),
        ("chinese", "请用两句话说明什么是量子比特。", 32),
        ("eos", "Reply with only OK.", 32),
        ("one_token", "What is Python?", 1),
        ("capacity2048", "Notes: Apples grow on trees. " * 100 + "\nName the fruit in one word.", 16),
        ("capacity4096", "Notes: Apples grow on trees. " * 350 + "\nName the fruit in one word.", 16),
        ("reuse_short", "In one sentence, explain recursion.", 48),
    ]
    context = AgentRunner(AnalyticSimulator(), RulePolicy()).context()
    cases.append(("calibration_plan", json.dumps(model_context(context), separators=(",", ":")), 160))
    results = []
    for name, prompt, limit in cases:
        messages = [{"role": "system", "content": SYSTEM_PROMPT if name == "calibration_plan" else CHAT_PROMPT},
                    {"role": "user", "content": prompt}]
        baseline = policy.graph_generator
        policy.graph_generator = None
        policy.decode_backend = "hf"
        try:
            slow = policy._generate(messages, limit, allow_graph=name != "calibration_plan")
            slow_stats = dict(policy.last_generation)
        finally:
            policy.graph_generator = baseline
            policy.decode_backend = "auto"
        fast = policy._generate(messages, limit, allow_graph=name != "calibration_plan")
        fast_stats = dict(policy.last_generation)
        case = {"name": name, "same_text": fast == slow,
                "same_token_ids": fast_stats["output_token_sha256"] == slow_stats["output_token_sha256"],
                "same_token_count": fast_stats["generated_tokens"] == slow_stats["generated_tokens"],
                "baseline": slow_stats, "accelerated": fast_stats, "baseline_text": slow, "fast_text": fast}
        results.append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)
        (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        expected_backend = "hf" if name == "calibration_plan" else "cuda_graph"
        if not case["same_text"] or not case["same_token_ids"] or not case["same_token_count"] or fast_stats["decode_backend"] != expected_backend:
            raise AssertionError("Equivalence/backend check failed: " + name)


if __name__ == "__main__":
    main()
