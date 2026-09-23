"""Unit tests for all multimodal data loaders and split tensors."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import MultimodalCRISPRDataset

TENSOR_DIR = PROJECT_ROOT / "outputs" / "tensors"


class TestMultimodalDataLoaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_pt = TENSOR_DIR / "train.pt"
        cls.val_pt   = TENSOR_DIR / "val.pt"
        cls.test_pt  = TENSOR_DIR / "test.pt"

    def test_01_tensor_files_exist(self):
        """Ensure all split .pt files exist."""
        self.assertTrue(self.train_pt.exists(), f"Missing {self.train_pt}")
        self.assertTrue(self.val_pt.exists(), f"Missing {self.val_pt}")
        self.assertTrue(self.test_pt.exists(), f"Missing {self.test_pt}")

    def test_02_split_sizes_and_sum(self):
        """Ensure split sizes match expected 80/10/10 distribution (25,554 total)."""
        train_ds = MultimodalCRISPRDataset(self.train_pt)
        val_ds   = MultimodalCRISPRDataset(self.val_pt)
        test_ds  = MultimodalCRISPRDataset(self.test_pt)

        self.assertEqual(len(train_ds), 20443)
        self.assertEqual(len(val_ds), 2555)
        self.assertEqual(len(test_ds), 2556)
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), 25554)

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
        self.assertEqual(batch["seq_ids"].shape, (16, 512))
        self.assertEqual(batch["seq_mask"].shape, (16, 512))
        self.assertEqual(batch["epigen_vec"].shape, (16, 6))
        self.assertEqual(batch["genome_vec"].shape, (16, 8))
        self.assertEqual(batch["sc_expr_vec"].shape, (16, 1))
        self.assertEqual(batch["reg_label"].shape, (16,))
        self.assertEqual(batch["cls_label"].shape, (16,))


if __name__ == "__main__":
    unittest.main()
