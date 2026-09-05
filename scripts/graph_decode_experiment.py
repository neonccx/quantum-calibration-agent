"""Experimental fixed-capacity, loop-aware graph decode. Never edits model weights/files."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers.cache_utils import DynamicCache


class LoopGraphCache(DynamicCache):
    def __init__(self, prefill, capacity):
        super().__init__()
        self.capacity = capacity
        for key, value in zip(prefill.key_cache, prefill.value_cache):
            shape = list(key.shape)
            shape[2] = capacity
            self.key_cache.append(torch.zeros(shape, dtype=key.dtype, device=key.device))
            self.value_cache.append(torch.zeros(shape, dtype=value.dtype, device=value.device))
        self.length = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        position = cache_kwargs["cache_position"]
        self.key_cache[layer_idx].index_copy_(2, position, key_states)
        self.value_cache[layer_idx].index_copy_(2, position, value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.length

    def load(self, prefill):
        self.length = prefill.key_cache[0].shape[2]
        for dstk, dstv, key, value in zip(self.key_cache, self.value_cache,
                                         prefill.key_cache, prefill.value_cache):
            dstk.zero_()
            dstv.zero_()
            dstk[:, :, :self.length].copy_(key)
            dstv[:, :, :self.length].copy_(value)


class GraphDecoder:
    def __init__(self, model, prefill, capacity):
        self.model, self.capacity = model, capacity
        self.cache = LoopGraphCache(prefill, capacity)
        self.cache.load(prefill)
        device = prefill.key_cache[0].device
        dtype = prefill.key_cache[0].dtype
        self.token = torch.zeros((1, 1), device=device, dtype=torch.long)
        self.position = torch.tensor([self.cache.length], device=device, dtype=torch.long)
        self.slots = torch.arange(capacity, device=device)
        self.mask = torch.zeros((1, 1, 1, capacity), device=device, dtype=dtype)
        self.min_value = torch.finfo(dtype).min
        original = model.model._update_causal_mask
        # Only during private warmup/capture: bypass Python .max().item() validation
        # of the already-validated, single-query causal mask built below.
        model.model._update_causal_mask = lambda *args, **kwargs: self.mask
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.forward()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output_token = self.forward()
        finally:
            model.model._update_causal_mask = original
        torch.cuda.synchronize()

    def forward(self):
        self.mask.copy_(torch.where(self.slots <= self.position[0], 0.0,
                                   self.min_value).to(self.mask.dtype).reshape(1, 1, 1, -1))
        result = self.model(input_ids=self.token, position_ids=self.position.view(1, 1),
                            cache_position=self.position, past_key_values=self.cache,
                            use_cache=True, return_dict=True)
        return result.logits[:, -1].argmax(dim=-1).view(1, 1)

    def generate(self, prefill, first_token, limit, eos):
        self.cache.load(prefill)
        token = first_token
        result = [int(token.item())]
        for index in range(limit - 1):
            if result[-1] == eos:
                break
            self.token.copy_(token)
            self.position.fill_(prefill.key_cache[0].shape[2] + index)
            self.graph.replay()
            token = self.output_token
            result.append(int(token.item()))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from qmagent.policies import HuggingFacePolicy, CHAT_PROMPT
    start = time.perf_counter()
    policy = HuggingFacePolicy(args.model, trust_remote_code=True, decode_backend="hf")
    print(json.dumps({"load_seconds": time.perf_counter() - start}), flush=True)
    cases = []
    decoder = None
    prompts = ["Explain how a Python dictionary works in about 80 words.",
               "请用两句话说明什么是量子比特。",
               "In one sentence, explain recursion.",
               "Explain how a Python dictionary works in about 80 words."]
    with torch.inference_mode():
        for prompt in prompts:
            text = policy.tokenizer.apply_chat_template([{"role": "system", "content": CHAT_PROMPT},
                {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False, preserve_thinking=False)
            encoded = policy.tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            length = encoded["input_ids"].shape[1]
            start = time.perf_counter()
            baseline = policy.model.generate(**encoded, do_sample=False, max_new_tokens=32,
                pad_token_id=policy.tokenizer.pad_token_id, eos_token_id=policy.tokenizer.eos_token_id)
            baseline_ids = baseline[0, length:].tolist()
            baseline_seconds = time.perf_counter() - start
            start = time.perf_counter()
            prefill = policy.model(**encoded, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
            first_token = prefill.logits[:, -1].argmax(dim=-1).view(1, 1)
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - start
            capture_seconds = 0
            if decoder is None:
                start = time.perf_counter()
                decoder = GraphDecoder(policy.model, prefill.past_key_values, 1024)
                capture_seconds = time.perf_counter() - start
            start = time.perf_counter()
            fast_ids = decoder.generate(prefill.past_key_values, first_token, 32, policy.tokenizer.eos_token_id)
            fast_seconds = time.perf_counter() - start
            case = {"prompt": prompt, "prompt_tokens": length, "output_tokens": len(fast_ids),
                    "baseline_seconds": baseline_seconds, "prefill_seconds": prefill_seconds,
                    "capture_seconds": capture_seconds, "decode_seconds": fast_seconds,
                    "tokens_exactly_equal": fast_ids == baseline_ids,
                    "baseline_ids": baseline_ids, "fast_ids": fast_ids,
                    "baseline_text": policy.tokenizer.decode(baseline_ids, skip_special_tokens=True),
                    "fast_text": policy.tokenizer.decode(fast_ids, skip_special_tokens=True)}
            cases.append(case)
            print(json.dumps(case, ensure_ascii=False), flush=True)
            (args.output / "results.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
