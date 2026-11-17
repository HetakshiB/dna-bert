"""Unit tests for all multimodal data loaders and split tensors."""
from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import MultimodalCRISPRDataset
from utils.preprocess import FEATURE_STATUS, save_split_tensors

TENSOR_DIR = PROJECT_ROOT / "outputs" / "tensors"


class TestMultimodalDataLoaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs")
        out = Path(cls._tmp.name)
        n = 50
        df = pd.DataFrame({
            "id": np.arange(n), "cell_line": ["test"] * n, "study_name": ["test"] * n,
            "cleavage_freq": np.linspace(0, 1, n),
            "high_eff": np.array([0] * 38 + [1] * 12, dtype=np.int8),
        })
        seq = {"input_ids": torch.ones(n, 512, dtype=torch.long),
               "attention_mask": torch.ones(n, 512, dtype=torch.long)}
        FEATURE_STATUS.clear()
        FEATURE_STATUS.update({"epigenetic": "test fixture", "3d_genome": "test fixture",
                               "singlecell": "test fixture"})
        save_split_tensors(seq, np.zeros((n, 6), dtype=np.float32),
                           np.zeros((n, 8), dtype=np.float32), np.zeros((n, 1), dtype=np.float32),
                           df, out)
        cls.train_pt = out / "train.pt"
        cls.val_pt = out / "val.pt"
        cls.test_pt = out / "test.pt"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_01_tensor_files_exist(self):
        """Ensure all split .pt files exist."""
        self.assertTrue(self.train_pt.exists(), f"Missing {self.train_pt}")
        self.assertTrue(self.val_pt.exists(), f"Missing {self.val_pt}")
        self.assertTrue(self.test_pt.exists(), f"Missing {self.test_pt}")
        self.assertTrue((self.train_pt.parent / "meta.csv").exists(), "Missing split/provenance metadata")

    def test_02_split_sizes_and_sum(self):
        """Ensure split sizes are close to the expected 80/10/10 distribution."""
        train_ds = MultimodalCRISPRDataset(self.train_pt)
        val_ds   = MultimodalCRISPRDataset(self.val_pt)
        test_ds  = MultimodalCRISPRDataset(self.test_pt)

        total = len(train_ds) + len(val_ds) + len(test_ds)
        self.assertGreaterEqual(total, 10)
        self.assertAlmostEqual(len(train_ds) / total, 0.80, delta=0.02)
        self.assertAlmostEqual(len(val_ds) / total, 0.10, delta=0.02)
        self.assertAlmostEqual(len(test_ds) / total, 0.10, delta=0.02)

    def test_03_tensor_shapes_and_types(self):
        """Ensure all tensor keys, shapes, and dtypes are correct for each split."""
        splits = [("train", self.train_pt), ("val", self.val_pt), ("test", self.test_pt)]
        required_keys = {"seq_ids", "seq_mask", "epigen_vec", "genome_vec", "sc_expr_vec", "reg_label", "cls_label"}

        for split_name, pt_file in splits:
            ds = MultimodalCRISPRDataset(pt_file)
            sample = ds[0]

            # Check keys
            self.assertTrue(required_keys.issubset(set(sample.keys())), f"Missing keys in {split_name}")

            # Check shapes & dtypes
            self.assertEqual(sample["seq_ids"].shape, (512,), f"Bad seq_ids shape in {split_name}")
            self.assertEqual(sample["seq_ids"].dtype, torch.long)

            self.assertEqual(sample["seq_mask"].shape, (512,), f"Bad seq_mask shape in {split_name}")
            self.assertEqual(sample["seq_mask"].dtype, torch.long)

            self.assertEqual(sample["epigen_vec"].shape, (6,), f"Bad epigen_vec shape in {split_name}")
            self.assertEqual(sample["epigen_vec"].dtype, torch.float32)

            self.assertEqual(sample["genome_vec"].shape, (8,), f"Bad genome_vec shape in {split_name}")
            self.assertEqual(sample["genome_vec"].dtype, torch.float32)

            self.assertEqual(sample["sc_expr_vec"].shape, (1,), f"Bad sc_expr_vec shape in {split_name}")
            self.assertEqual(sample["sc_expr_vec"].dtype, torch.float32)

            self.assertEqual(sample["reg_label"].dtype, torch.float32)
            self.assertEqual(sample["cls_label"].dtype, torch.long)
            self.assertIn(int(sample["cls_label"]), (0, 1))

    def test_04_no_nans_or_infs(self):
        """Ensure no NaNs or Infs exist across sample tensors."""
        for split_name, pt_file in [("train", self.train_pt), ("val", self.val_pt), ("test", self.test_pt)]:
            ds = MultimodalCRISPRDataset(pt_file)
            # Inspect first 100 samples
            for i in range(min(100, len(ds))):
                item = ds[i]
                for key in ["epigen_vec", "genome_vec", "sc_expr_vec", "reg_label"]:
                    t = item[key]
                    self.assertFalse(torch.isnan(t).any(), f"NaN found in {key} of {split_name} sample {i}")
                    self.assertFalse(torch.isinf(t).any(), f"Inf found in {key} of {split_name} sample {i}")

    def test_05_dataloader_collation(self):
        """Ensure PyTorch DataLoader batches correctly with proper tensor dimensions."""
        train_ds = MultimodalCRISPRDataset(self.train_pt)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        batch = next(iter(loader))
        batch_size = min(16, len(train_ds))
        self.assertEqual(batch["seq_ids"].shape, (batch_size, 512))
        self.assertEqual(batch["seq_mask"].shape, (batch_size, 512))
        self.assertEqual(batch["epigen_vec"].shape, (batch_size, 6))
        self.assertEqual(batch["genome_vec"].shape, (batch_size, 8))
        self.assertEqual(batch["sc_expr_vec"].shape, (batch_size, 1))
        self.assertEqual(batch["reg_label"].shape, (batch_size,))
        self.assertEqual(batch["cls_label"].shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()
