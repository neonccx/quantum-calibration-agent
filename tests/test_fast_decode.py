import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qmagent.fast_decode import AUDITED_MODEL_SHA256, generation_supported, graph_capacity, support_reason
from qmagent.settings import Settings


class FastDecodeTests(unittest.TestCase):
    def model(self):
        config = SimpleNamespace(model_type="nanbeige", num_loops=2, num_hidden_layers=22,
            num_key_value_heads=8, num_attention_heads=48, rope_scaling=None,
            _attn_implementation="sdpa", use_cache=True)
        return SimpleNamespace(config=config, model=SimpleNamespace(ngram_embeddings=None),
            training=False, parameters=lambda: iter([SimpleNamespace(device="cuda:0", dtype="torch.bfloat16")]),
            generation_config=SimpleNamespace(to_dict=lambda: {}))

    def test_capacity_boundaries(self):
        for prompt, budget, expected in ((512, 32, 1024), (1000, 24, 1024), (1000, 25, 2048),
                                         (2000, 100, 4096), (4000, 96, 4096), (4000, 97, None)):
            self.assertEqual(graph_capacity(prompt, budget), expected)

    def test_invalid_budget(self):
        for args in ((0, 3), (1, 0), (True, 3), (3.5, 5), (1, -1)):
            with self.assertRaises(ValueError):
                graph_capacity(*args)

    def test_no_custom_generation_processors(self):
        self.assertTrue(generation_supported({"do_sample": True, "temperature": 0.6, "top_p": 0.95}))
        for config in ({"repetition_penalty": 1.2}, {"num_beams": 2}, {"stop_strings": ["stop"]},
                       {"min_new_tokens": 10}, {"forced_eos_token_id": 42}, {"use_cache": False},
                       {"bad_words_ids": [[3]]}, {"max_time": 3}):
            self.assertFalse(generation_supported(config), config)

    def test_adapter_uses_hf(self):
        self.assertIn("LoRA", support_reason(self.model(), "/adapter", "2.11.0", "4.45.1"))

    def test_unsupported_versions_use_hf(self):
        for torch_version, hf_version in (("2.10.0", "4.45.1"), ("2.11.0", "5.0.0")):
            self.assertIn("Unaudited", support_reason(self.model(), None, torch_version, hf_version))

    def test_architecture_features_do_not_silently_change(self):
        model = self.model()
        model.config.enable_depth_attention = True
        self.assertIn("Unsupported", support_reason(model, None, "2.11.0", "4.45.1"))
        model = self.model()
        model.config.num_loops = 1
        self.assertIn("Unsupported", support_reason(model, None, "2.11.0", "4.45.1"))

    def checked_reason(self, model, digest=AUDITED_MODEL_SHA256):
        with patch("qmagent.fast_decode.inspect.getfile", return_value="/fake/model.py"), \
             patch("qmagent.fast_decode.Path.read_bytes", return_value=b"test"), \
             patch("qmagent.fast_decode.hashlib.sha256", return_value=SimpleNamespace(hexdigest=lambda: digest)):
            return support_reason(model, None, "2.11.0+cu128", "4.45.1")

    def test_audited_configuration_supported(self):
        self.assertIsNone(self.checked_reason(self.model()))

    def test_changed_source_refused(self):
        self.assertIn("source differs", self.checked_reason(self.model(), digest="wrong"))

    def test_training_or_cpu_parameters_refused(self):
        model = self.model()
        model.training = True
        self.assertIn("evaluation-only", self.checked_reason(model))
        model = self.model()
        model.parameters = lambda: iter([SimpleNamespace(device="cpu", dtype="torch.bfloat16")])
        self.assertIn("evaluation-only", self.checked_reason(model))

    def test_backend_setting(self):
        for backend in ("auto", "hf", "cuda_graph"):
            self.assertEqual(Settings.from_dict({"decode_backend": backend}).decode_backend, backend)
        with self.assertRaises(ValueError):
            Settings.from_dict({"decode_backend": "unknown"})
