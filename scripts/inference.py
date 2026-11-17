"""Run one multimodal CRISPR prediction from an explicit DNA sequence/features."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.backbone import get_dnabert, seq_to_kmers
from model.multimodal import MultimodalCRISPR


def _feature(values: list[float] | None, width: int, name: str, allow_missing: bool) -> tuple[torch.Tensor, str]:
    if values is None:
        if not allow_missing:
            raise ValueError(f"--{name} is required unless --allow-missing-features is set")
        return torch.zeros(1, width, dtype=torch.float32), "missing; zero placeholder (not measured)"
    if len(values) != width:
        raise ValueError(f"--{name} needs exactly {width} values, received {len(values)}")
    return torch.tensor([values], dtype=torch.float32), "explicit user-provided feature values"


def predict(sequence: str, checkpoint: Path, epigen: list[float] | None,
            genome: list[float] | None, sc_expression: float | None,
            allow_missing_features: bool, max_length: int = 512) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, backbone = get_dnabert(freeze=True, device=device)
    model = MultimodalCRISPR(dnabert_model=backbone).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload)
    model.eval()

    epigen_t, epigen_status = _feature(epigen, 6, "epigen", allow_missing_features)
    if epigen is not None:
        scaler_path = ROOT / "outputs" / "tensors" / "feature_scaler.json"
        if not scaler_path.is_file():
            raise FileNotFoundError(
                f"Training feature scaler is missing: {scaler_path}. Re-run preprocessing before inference."
            )
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        epigen_t = (epigen_t - torch.tensor(scaler["epigenetic_mean"])) / torch.tensor(scaler["epigenetic_std"])
        epigen_status += "; raw values transformed with train-split scaler"
    genome_t, genome_status = _feature(genome, 8, "genome", allow_missing_features)
    if genome is not None:
        if genome[0] < 0 or genome[1] < 0:
            raise ValueError("Raw loop count and TAD distance cannot be negative")
        genome_t[0, 0] = math.log1p(genome[0])
        genome_t[0, 1] = min(math.log10(genome[1] + 1.0), 8.0)
        genome_status += "; loop count and TAD distance transformed as in preprocessing"
    if sc_expression is None:
        if not allow_missing_features:
            raise ValueError("--sc-expression is required unless --allow-missing-features is set")
        sc_t = torch.zeros(1, 1, dtype=torch.float32)
        sc_status = "missing; zero placeholder (not measured)"
    else:
        sc_t = torch.tensor([[sc_expression]], dtype=torch.float32)
        sc_status = "explicit user-provided expression value"

    encoded = tokenizer(
        seq_to_kmers(sequence), return_tensors="pt", truncation=True,
        max_length=max_length, padding="max_length",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        reg, cls, _ = model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
            epigen_vec=epigen_t.to(device), genome_vec=genome_t.to(device),
            sc_expr_vec=sc_t.to(device),
        )
        probability = float(torch.sigmoid(cls).item())
    return {
        "device": str(device),
        "predicted_cleavage_frequency": float(reg.item()),
        "high_efficiency_probability": probability,
        "high_efficiency_prediction": bool(probability >= 0.5),
        "feature_provenance": {
            "sequence": "DNABERT-6 encoded from input sequence",
            "epigenetic": epigen_status,
            "3d_genome": genome_status,
            "singlecell_expression": sc_status,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="DNA sequence containing A/C/G/T/N")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "model_best.pt")
    parser.add_argument("--epigen", nargs=6, type=float, metavar=("H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3", "ATAC", "DNase"))
    parser.add_argument("--genome", nargs=8, type=float, metavar=("loop_count", "tad_distance_bp", "A1", "A2", "B1", "B2", "B3", "NA"), help="Raw loop count, TAD boundary distance in bp, then compartment one-hot.")
    parser.add_argument("--sc-expression", type=float)
    parser.add_argument("--allow-missing-features", action="store_true", help="Explicitly allow missing biological features as zero placeholders.")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not 8 <= args.max_length <= 512:
        parser.error("--max-length must be between 8 and 512")
    result = predict(args.sequence, args.checkpoint, args.epigen, args.genome,
                     args.sc_expression, args.allow_missing_features, args.max_length)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
