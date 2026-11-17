"""Focused CPU tests for preprocessing contracts and the multimodal model."""
from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.backbone import seq_to_kmers
from model.losses import MultimodalLoss
from model.multimodal import MultimodalCRISPR
from train import MultimodalCRISPRDataset, _atomic_torch_save
from utils.preprocess import FEATURE_STATUS, save_split_tensors
from scripts.inference import predict


class TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 768)

    def forward(self, input_ids, attention_mask):
        return type("Output", (), {"last_hidden_state": self.embedding(input_ids)})()


class PipelineSmokeTests(unittest.TestCase):
    def test_kmer_conversion_and_validation(self):
        self.assertEqual(seq_to_kmers("ACGTNAC"), "ACGTNA CGTNAC")
        with self.assertRaises(ValueError):
            seq_to_kmers("ACGT-X")

    def test_inference_reports_missing_features(self):
        class TinyTokenizer:
            def __call__(self, text, return_tensors, truncation, max_length, padding):
                return {"input_ids": torch.ones(1, max_length, dtype=torch.long),
                        "attention_mask": torch.ones(1, max_length, dtype=torch.long)}

        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            checkpoint = Path(tmp) / "model.pt"
            trained = MultimodalCRISPR(dnabert_model=TinyBackbone())
            torch.save({"model_state": trained.state_dict()}, checkpoint)
            with patch("scripts.inference.get_dnabert", return_value=(TinyTokenizer(), TinyBackbone())):
                output = predict("ACGT" * 10, checkpoint, None, None, None,
                                 allow_missing_features=True, max_length=16)
        self.assertIn("predicted_cleavage_frequency", output)
        self.assertIn("high_efficiency_probability", output)
        self.assertIn("placeholder", output["feature_provenance"]["epigenetic"])
        self.assertIn("placeholder", output["feature_provenance"]["3d_genome"])

    def test_model_loss_dataset_checkpoint_and_preprocessing(self):
        torch.manual_seed(7)
        model = MultimodalCRISPR(dnabert_model=TinyBackbone())
        ids = torch.randint(0, 32, (4, 16))
        mask = torch.ones_like(ids)
        reg, cls, fused = model(ids, mask, torch.randn(4, 6), torch.randn(4, 8), torch.randn(4, 1))
        self.assertEqual(tuple(reg.shape), (4, 1))
        self.assertEqual(tuple(cls.shape), (4, 1))
        self.assertEqual(tuple(fused.shape), (4, 256))
        loss, _ = MultimodalLoss(epoch_steps=2)(reg, cls, torch.rand(4), torch.tensor([0, 1, 0, 1]))
        loss.backward()

        test_output_root = Path(__file__).resolve().parent.parent / "outputs"
        test_output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_output_root) as tmp:
            root = Path(tmp)
            n = 50
            df = pd.DataFrame({
                "id": np.arange(n), "cell_line": ["K562"] * n,
                "study_name": ["test"] * n, "cleavage_freq": np.linspace(0, 1, n),
                "high_eff": np.array([0] * 38 + [1] * 12, dtype=np.int8),
            })
            seq = {"input_ids": torch.randint(0, 32, (n, 16)),
                   "attention_mask": torch.ones(n, 16, dtype=torch.long)}
            FEATURE_STATUS.clear()
            FEATURE_STATUS.update({"epigenetic": "synthetic test values", "3d_genome": "missing; placeholder"})
            save_split_tensors(seq, np.random.randn(n, 6).astype("float32"),
                               np.zeros((n, 8), dtype="float32"), np.zeros((n, 1), dtype="float32"),
                               df, root)
            self.assertTrue((root / "meta.csv").is_file())
            train_ds = MultimodalCRISPRDataset(root / "train.pt")
            batch = train_ds[0]
            self.assertEqual(tuple(batch["epigen_vec"].shape), (6,))
            self.assertEqual(tuple(batch["genome_vec"].shape), (8,))
            ckpt = root / "checkpoint.pt"
            _atomic_torch_save({"model_state": model.state_dict()}, ckpt)
            restored = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(restored["model_state"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
