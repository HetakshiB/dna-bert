"""Verify the exact pretrained DNABERT-6 cache and requested device placement."""
from __future__ import annotations

import unittest
import sys
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.backbone import CACHE_DIR, HF_MODEL_ID, _cache_is_complete, get_dnabert, seq_to_kmers


class DNABERTLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.model = get_dnabert(freeze=True)

    def test_cache_has_real_weights_and_tokenizer(self):
        self.assertTrue(_cache_is_complete(CACHE_DIR), f"Incomplete cache for {HF_MODEL_ID}: {CACHE_DIR}")
        self.assertGreater(sum(p.numel() for p in self.model.parameters()), 80_000_000)
        self.assertEqual(self.model.config.hidden_size, 768)
        self.assertEqual(self.model.config.num_hidden_layers, 12)
        self.assertEqual(len(self.tokenizer), self.model.config.vocab_size)

    def test_kmer_tokenizer_and_backbone_forward(self):
        encoded = self.tokenizer(seq_to_kmers("ACGT" * 16), return_tensors="pt", truncation=True, max_length=64)
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = self.model(**encoded)
        self.assertEqual(tuple(result.last_hidden_state.shape), (1, encoded["input_ids"].shape[1], 768))

    def test_gpu_and_frozen_parameters(self):
        if torch.cuda.is_available():
            self.assertEqual(next(self.model.parameters()).device.type, "cuda")
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))

    def test_huggingface_cache_stays_in_project(self):
        cache_root = (ROOT / "models" / "hf-cache").resolve()
        for variable in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
            self.assertTrue(Path(os.environ[variable]).resolve().is_relative_to(cache_root), variable)


if __name__ == "__main__":
    unittest.main(verbosity=2)
