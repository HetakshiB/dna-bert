"""Small end-to-end preprocessing check using a temporary E: output directory."""
from __future__ import annotations

import shutil
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from utils.preprocess import preprocess


def main() -> None:
    temporary = Path(tempfile.mkdtemp(prefix="preprocess_smoke_", dir=ROOT / "outputs"))
    try:
        preprocess(sample=40, run_tests=True, max_length=64, output_dir=temporary)
        for name in ("train.pt", "val.pt", "test.pt", "meta.csv", "feature_scaler.json"):
            assert (temporary / name).is_file(), f"Missing preprocessing output: {name}"
        train = torch.load(temporary / "train.pt", map_location="cpu", weights_only=True)
        assert train and train[0]["seq_ids"].shape == (64,)
        meta = pd.read_csv(temporary / "meta.csv")
        assert set(meta["split"]) == {"train", "val", "test"}
        for column in ("epigenetic_provenance", "3d_genome_provenance", "singlecell_provenance"):
            assert column in meta.columns
        print("End-to-end preprocessing smoke test passed; temporary outputs are on the project E: mount.")
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
