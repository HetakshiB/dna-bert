"""Exercise all three short training stages without touching user checkpoints."""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train as training
from utils.preprocess import FEATURE_STATUS, save_split_tensors


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Training smoke test requires the requested CUDA GPU.")
    with tempfile.TemporaryDirectory(prefix="training_smoke_", dir=ROOT / "outputs") as tmp:
        root = Path(tmp)
        tensor_dir = root / "tensors"
        run_dir = root / "run"
        checkpoint_dir = run_dir / "checkpoints"
        log_dir = run_dir / "logs"
        checkpoint_dir.mkdir(parents=True)
        log_dir.mkdir(parents=True)
        n = 50
        labels = np.array([0] * 38 + [1] * 12, dtype=np.int8)
        df = pd.DataFrame({"id": np.arange(n), "cell_line": ["test"] * n,
                           "study_name": ["test"] * n,
                           "cleavage_freq": np.linspace(0, 1, n), "high_eff": labels})
        seq = {"input_ids": torch.randint(1, 4000, (n, 512)),
               "attention_mask": torch.ones(n, 512, dtype=torch.long)}
        FEATURE_STATUS.clear()
        FEATURE_STATUS.update({"epigenetic": "test fixture", "3d_genome": "test fixture",
                               "singlecell": "test fixture"})
        save_split_tensors(seq, np.zeros((n, 6), dtype=np.float32),
                           np.zeros((n, 8), dtype=np.float32), np.zeros((n, 1), dtype=np.float32),
                           df, tensor_dir)

        training.TENSOR_DIR = tensor_dir
        training.CHECKPOINT_DIR = checkpoint_dir
        training.LOG_DIR = log_dir
        training.BEST_MODEL_PATH = run_dir / "model_best.pt"
        training.CSV_METRICS_PATH = log_dir / "train_metrics.csv"
        training.TB_DIR = log_dir / "tensorboard"
        for handler in training.log.handlers:
            handler.close()
        training.log.handlers.clear()
        training.log.addHandler(logging.FileHandler(log_dir / "train.log", encoding="utf-8"))
        training.log.setLevel(logging.INFO)

        training.train(smoke_test=True, batch_size=2, accumulation_steps=4,
                       enable_swa=False, device_name="cuda")
        assert training.BEST_MODEL_PATH.is_file(), "Best checkpoint was not saved"
        assert (checkpoint_dir / "last_checkpoint.pt").is_file(), "Resume checkpoint was not saved"
        assert (log_dir / "train_metrics.csv").is_file(), "Metrics CSV was not saved"
        print("Three-stage CUDA smoke training passed; checkpoints/logs were isolated under the E: mount.")


if __name__ == "__main__":
    main()
