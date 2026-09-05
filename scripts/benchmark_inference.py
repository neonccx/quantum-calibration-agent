"""Bounded, offline inference diagnostic. Does not alter model files or run experiments."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    from qmagent.policies import HuggingFacePolicy
    policy = HuggingFacePolicy(args.model, trust_remote_code=True, decode_backend="hf")
    torch = policy.torch
    info = {"load_seconds": time.perf_counter() - started, "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "device_map": {k: str(v) for k, v in policy.model.hf_device_map.items()},
            "parameter_devices": dict(Counter(str(p.device) for p in policy.model.parameters())),
            "parameter_dtypes": dict(Counter(str(p.dtype) for p in policy.model.parameters())),
            "parameter_count": sum(p.numel() for p in policy.model.parameters()),
            "num_loops": policy.model.config.num_loops, "use_cache": policy.model.config.use_cache}
    print(json.dumps({"environment": info}), flush=True)
    messages = [{"role": "user", "content": "Explain how a Python dictionary works in about 80 words."}]
    context = {"status": "active", "experiment_count": 0, "backend": "analytic simulator only"}
    policy.chat(messages, context, max_new_tokens=4)
    cases = []
    for threads in (4, 1, 1, 4):
        torch.set_num_threads(threads)
        lengths = []
        def hook(module, arguments, kwargs):
            ids = kwargs.get("input_ids")
            if ids is not None:
                lengths.append(int(ids.shape[-1]))
        handle = policy.model.register_forward_pre_hook(hook, with_kwargs=True)
        wall = time.perf_counter()
        try:
            answer = policy.chat(messages, context, max_new_tokens=32)
        finally:
            handle.remove()
        stats = policy.last_generation
        case = {"cpu_threads": threads, "wall_seconds": time.perf_counter() - wall,
                "generation": stats, "tokens_per_second": stats["generated_tokens"] / stats["seconds"],
                "forward_input_lengths": lengths, "answer": answer}
        cases.append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)
    report = {"environment": info, "cases": cases,
              "outputs_equal": len({c["answer"] for c in cases}) == 1}
    if args.profile:
        torch.set_num_threads(4)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            policy.chat(messages, context, max_new_tokens=8)
            torch.cuda.synchronize()
        table = profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        print(table, flush=True)
        (args.output / "profile.txt").write_text(table)
        report["profile"] = [{"name": item.key, "count": item.count,
                              "self_cpu_us": item.self_cpu_time_total,
                              "self_device_us": getattr(item, "self_device_time_total", 0)}
                             for item in profile.key_averages()]
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
