"""Opt-in/auto CUDA graph decode for the audited Nanbeige 4.2 loop architecture.

No torch import at module import time: the Mac RPC client remains stdlib-only.
Unpadded batch=1 greedy inference only; not a training or general HF backend.
"""
import hashlib
import inspect
from pathlib import Path
import time

AUDITED_MODEL_SHA256 = "547737f989f5cb741c1a568acaf85f83521fa67a8f7268e19f9a37e60127c0d5"
CAPACITIES = (1024, 2048, 4096)


def graph_capacity(prompt_tokens, max_new_tokens):
    if type(prompt_tokens) is not int or type(max_new_tokens) is not int or min(prompt_tokens, max_new_tokens) < 1:
        raise ValueError("Invalid generation token budget")
    return next((size for size in CAPACITIES if prompt_tokens + max_new_tokens <= size), None)


def generation_supported(config):
    # do_sample=False and max_new_tokens are explicitly set by our policy. Do not
    # bypass custom logits processors, stopping rules or generation constraints.
    neutral = {"num_beams": (1,), "num_beam_groups": (1,), "num_return_sequences": (1,),
        "repetition_penalty": (1.0,), "encoder_repetition_penalty": (1.0,),
        "no_repeat_ngram_size": (0,), "encoder_no_repeat_ngram_size": (0,),
        "min_length": (0,), "min_new_tokens": (None, 0), "bad_words_ids": (None,),
        "force_words_ids": (None,), "forced_bos_token_id": (None,),
        "forced_eos_token_id": (None,), "suppress_tokens": (None,),
        "begin_suppress_tokens": (None,), "sequence_bias": (None,), "constraints": (None,),
        "penalty_alpha": (None,), "guidance_scale": (None, 1.0), "stop_strings": (None,),
        "watermarking_config": (None,), "renormalize_logits": (False,),
        "remove_invalid_values": (False,), "exponential_decay_length_penalty": (None,),
        "max_time": (None,), "use_cache": (True,)}
    return all(config.get(key, values[0]) in values for key, values in neutral.items())


def support_reason(model, adapter, torch_version, transformers_version):
    if adapter:
        return "LoRA uses the original HF path"
    if not torch_version.startswith("2.11.") or transformers_version != "4.45.1":
        return "Unaudited torch/transformers combination"
    config = model.config
    if (getattr(config, "model_type", None) != "nanbeige"
            or getattr(config, "num_loops", None) != 2
            or getattr(config, "num_hidden_layers", None) != 22
            or getattr(config, "num_key_value_heads", None) != 8
            or getattr(config, "num_attention_heads", None) != 48
            or getattr(config, "rope_scaling", None) is not None
            or getattr(config, "loop_loss_weights", None)
            or getattr(config, "_attn_implementation", None) != "sdpa"
            or not getattr(config, "use_cache", False)):
        return "Unsupported model architecture/cache configuration"
    for name in ("enable_depth_attention", "loop_share_kv", "enable_hyper_connection",
                 "enable_double_loop_split", "enable_mhc"):
        if getattr(config, name, False):
            return "Unsupported architecture feature: " + name
    if model.model.ngram_embeddings is not None:
        return "Ngram model is not supported"
    try:
        source = Path(inspect.getfile(type(model))).read_bytes()
    except (OSError, TypeError):
        return "Cannot verify model source"
    if hashlib.sha256(source).hexdigest() != AUDITED_MODEL_SHA256:
        return "Model source differs from audited revision"
    if model.training or any(str(p.device) != "cuda:0" or str(p.dtype) != "torch.bfloat16"
                             for p in model.parameters()):
        return "Requires evaluation-only BF16 model entirely on cuda:0"
    if not generation_supported(model.generation_config.to_dict()):
        return "Custom generation processors/stopping rules require HF"
    return None


def make_cache(prefill, capacity):
    import torch
    from transformers.cache_utils import DynamicCache
    if len(prefill.key_cache) != 44 or len(prefill.value_cache) != 44:
        raise ValueError("Expected 44 independent layer/loop cache entries")

    class LoopGraphCache(DynamicCache):
        def __init__(self, source):
            super().__init__()
            self.length = 0
            for key, value in zip(source.key_cache, source.value_cache):
                shape = list(key.shape)
                shape[2] = capacity
                self.key_cache.append(torch.zeros(shape, dtype=key.dtype, device=key.device))
                self.value_cache.append(torch.zeros(shape, dtype=value.dtype, device=value.device))

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            position = cache_kwargs["cache_position"]
            self.key_cache[layer_idx].index_copy_(2, position, key_states)
            self.value_cache[layer_idx].index_copy_(2, position, value_states)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx=0):
            return self.length

        def load(self, source):
            if len(source.key_cache) != len(self.key_cache):
                raise ValueError("Loop cache layer count changed")
            self.length = source.key_cache[0].shape[2]
            if not 0 < self.length < capacity:
                raise ValueError("Prefill exceeds graph capacity")
            for dstk, dstv, key, value in zip(self.key_cache, self.value_cache,
                                             source.key_cache, source.value_cache):
                if key.shape != value.shape or key.shape[2] != self.length:
                    raise ValueError("Inconsistent loop cache shapes")
                # Each request gets clean memory: never retain another turn's KV.
                dstk.zero_()
                dstv.zero_()
                dstk[:, :, :self.length].copy_(key)
                dstv[:, :, :self.length].copy_(value)

    return LoopGraphCache(prefill)


class GraphDecoder:
    def __init__(self, model, prefill, capacity):
        import torch
        self.model, self.capacity = model, capacity
        self.cache = make_cache(prefill, capacity)
        self.cache.load(prefill)
        device, dtype = prefill.key_cache[0].device, prefill.key_cache[0].dtype
        self.token = torch.zeros((1, 1), device=device, dtype=torch.long)
        self.position = torch.tensor([self.cache.length], device=device, dtype=torch.long)
        self.slots = torch.arange(capacity, device=device)
        self.mask = torch.zeros((1, 1, 1, capacity), device=device, dtype=dtype)
        self.min_value = torch.finfo(dtype).min
        self.graph = None
        original = model.model._update_causal_mask
        # This private worker serves one request at a time. During capture only,
        # bypass the upstream Python GPU->CPU mask check. Our single-query mask
        # is causal by construction and excludes every unwritten cache slot.
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
        import torch
        self.mask.copy_(torch.where(self.slots <= self.position[0], 0.0,
                                   self.min_value).to(self.mask.dtype).reshape(1, 1, 1, -1))
        result = self.model(input_ids=self.token, position_ids=self.position.view(1, 1),
                            cache_position=self.position, past_key_values=self.cache,
                            use_cache=True, return_dict=True)
        return result.logits[:, -1].argmax(dim=-1).view(1, 1)

    def generate(self, prefill, first_token, limit, eos, on_token=None):
        if type(limit) is not int or limit < 1 or prefill.key_cache[0].shape[2] + limit > self.capacity:
            raise ValueError("Output budget exceeds graph capacity")
        self.cache.load(prefill)
        token = first_token
        result = [int(token.item())]
        for index in range(limit - 1):
            if result[-1] == eos:
                break
            self.token.copy_(token)
            self.position.fill_(self.cache.length + index)
            self.graph.replay()
            token = self.output_token
            result.append(int(token.item()))
            if on_token:
                on_token(result[-1])
        return result


class GraphGenerator:
    def __init__(self, model):
        self.model, self.decoder = model, None

    def generate(self, encoded, max_new_tokens, eos, on_token=None):
        import torch
        from transformers.cache_utils import DynamicCache
        length = encoded["input_ids"].shape[1]
        capacity = graph_capacity(length, max_new_tokens)
        if capacity is None or encoded["input_ids"].shape[0] != 1 or type(eos) is not int:
            raise ValueError("Unsupported CUDA graph input")
        if "attention_mask" in encoded and not bool(encoded["attention_mask"].eq(1).all().item()):
            raise ValueError("CUDA graph path requires unpadded input")
        start = time.perf_counter()
        with torch.inference_mode():
            prefill = self.model(**encoded, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
            first_token = prefill.logits[:, -1].argmax(dim=-1).view(1, 1)
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - start
            if on_token:
                on_token(int(first_token.item()))
            if max_new_tokens == 1 or int(first_token.item()) == eos:
                return [int(first_token.item())], {"prefill_seconds": prefill_seconds,
                    "capture_seconds": 0.0, "decode_seconds": 0.0, "graph_capacity": capacity}
            capture_seconds = 0.0
            if self.decoder is None or self.decoder.capacity != capacity:
                # Bound retained graph memory to one bucket, not one per turn.
                self.decoder = None
                start = time.perf_counter()
                self.decoder = GraphDecoder(self.model, prefill.past_key_values, capacity)
                capture_seconds = time.perf_counter() - start
            start = time.perf_counter()
            tokens = self.decoder.generate(prefill.past_key_values, first_token, max_new_tokens, eos, on_token)
        return tokens, {"prefill_seconds": prefill_seconds, "capture_seconds": capture_seconds,
                        "decode_seconds": time.perf_counter() - start, "graph_capacity": capacity}
