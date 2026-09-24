"""Evaluation and Interpretability pipeline for the Multimodal CRISPR Predictor.

Functionality:
  1. Loads best model checkpoint (outputs/model_best.pt or fallback).
  2. Runs inference on held-out test split (outputs/tensors/test.pt).
  3. Computes quantitative performance metrics:
     - Regression: RMSE, Pearson r.
     - Classification: AUROC, AUPRC (PR-AUC), F1@0.5.
     - Calibration: Brier score.
  4. Generates Reliability Diagram (outputs/reliability_diagram.png).
  5. Computes Integrated Gradients attribution on sequence representation
     (outputs/attribution_seq.png).
  6. Computes SHAP feature attributions on epigenetic (6 marks) and 3D genome
     features (8 dimensions), saved to outputs/attribution_epigen_3d.png.
  7. Export metrics summary to outputs/metrics.json.

Usage:
  python evaluate.py
  python evaluate.py --sample 500  # Evaluate on first 500 test samples for fast run
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    mean_squared_error,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr
import shap

PROJECT_ROOT = Path(__file__).resolve().parent
TENSOR_DIR = PROJECT_ROOT / "outputs" / "tensors"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
TB_DIR = LOG_DIR / "tensorboard"
BEST_MODEL_PATH = OUTPUT_DIR / "model_best.pt"
FALLBACK_CKPT_PATH = OUTPUT_DIR / "checkpoints" / "best_model.pt"
METRICS_JSON_PATH = OUTPUT_DIR / "metrics.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

from torch.utils.tensorboard import SummaryWriter

def log_image_to_tb(writer: SummaryWriter, tag: str, img_path: Path) -> None:
    """Read an image from disk and log as [C, H, W] tensor in TensorBoard."""
    if not img_path.exists():
        return
    try:
        arr = plt.imread(str(img_path))
        if arr.dtype == np.uint8:
            t = torch.from_numpy(arr).float() / 255.0
        else:
            t = torch.from_numpy(arr).float()
        if t.ndim == 3 and t.shape[-1] in (3, 4):
            t = t.permute(2, 0, 1)[:3]  # take RGB channels
        writer.add_image(tag, t)
    except Exception as e:
        log.warning("[TB] Could not log image %s: %s", img_path.name, e)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "evaluate.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(PROJECT_ROOT))
from model.multimodal import MultimodalCRISPR, build_model  # noqa: E402
from model.losses import MultimodalLoss                    # noqa: E402
from train import MultimodalCRISPRDataset                  # noqa: E402


# ============================================================================
# 1. Model Loading
# ============================================================================

def load_evaluation_model(device: torch.device) -> Tuple[MultimodalCRISPR, Path]:
    """Load model architecture and weights from model_best.pt or fallback."""
    ckpt_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FALLBACK_CKPT_PATH
    if not ckpt_path.exists():
        ckpt_path = OUTPUT_DIR / "checkpoints" / "final_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("No trained model checkpoint found in outputs/ or outputs/checkpoints/")

    log.info("[EVAL] Loading model checkpoint from %s ...", ckpt_path)
    model = build_model(load_dnabert=True, freeze_bert=True, device=str(device))
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    log.info("[EVAL] Model loaded successfully onto %s.", device)
    return model, ckpt_path


# ============================================================================
# 2. Inference Loop
# ============================================================================

@torch.no_grad()
def run_test_inference(
    model: MultimodalCRISPR,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run full test split inference and collect predictions, targets, and latent embeddings."""
    model.eval()

    reg_preds_list, reg_targets_list = [], []
    cls_logits_list, cls_probs_list, cls_targets_list = [], [], []
    fused_embeds_list = []
    epigen_vecs_list, genome_vecs_list = [], []

    for batch in test_loader:
        seq_ids     = batch["seq_ids"].to(device)
        seq_mask    = batch["seq_mask"].to(device)
        epigen_vec  = batch["epigen_vec"].to(device)
        genome_vec  = batch["genome_vec"].to(device)
        sc_expr_vec = batch["sc_expr_vec"].to(device)
        reg_label   = batch["reg_label"].to(device)
        cls_label   = batch["cls_label"].to(device)

        cls_emb = batch.get("cls_emb")
        if cls_emb is not None:
            cls_emb = cls_emb.to(device)

        reg_out, cls_out, fused = model(
            input_ids=seq_ids,
            attention_mask=seq_mask,
            epigen_vec=epigen_vec,
            genome_vec=genome_vec,
            sc_expr_vec=sc_expr_vec,
            cls_emb=cls_emb,
        )

        reg_preds = reg_out.squeeze(-1).cpu().numpy()
        cls_logits = cls_out.squeeze(-1).cpu().numpy()
        cls_probs = torch.sigmoid(cls_out.squeeze(-1)).cpu().numpy()

        reg_preds_list.extend(reg_preds)
        reg_targets_list.extend(reg_label.cpu().numpy())
        cls_logits_list.extend(cls_logits)
        cls_probs_list.extend(cls_probs)
        cls_targets_list.extend(cls_label.cpu().numpy())

        fused_embeds_list.append(fused.cpu().numpy())
        epigen_vecs_list.append(epigen_vec.cpu().numpy())
        genome_vecs_list.append(genome_vec.cpu().numpy())

    return {
        "reg_preds": np.array(reg_preds_list, dtype=np.float32),
        "reg_targets": np.array(reg_targets_list, dtype=np.float32),
        "cls_logits": np.array(cls_logits_list, dtype=np.float32),
        "cls_probs": np.array(cls_probs_list, dtype=np.float32),
        "cls_targets": np.array(cls_targets_list, dtype=np.int64),
        "fused_embeds": np.concatenate(fused_embeds_list, axis=0),
        "epigen_vecs": np.concatenate(epigen_vecs_list, axis=0),
        "genome_vecs": np.concatenate(genome_vecs_list, axis=0),
    }


# ============================================================================
# 3. Metrics Computation & Reliability Diagram
# ============================================================================

def compute_metrics_and_plots(
    results: Dict[str, np.ndarray],
    output_dir: Path,
) -> Dict[str, Any]:
    """Compute quantitative metrics and save reliability diagram plot."""
    reg_preds = results["reg_preds"]
    reg_targets = results["reg_targets"]
    cls_probs = results["cls_probs"]
    cls_targets = results["cls_targets"]

    # 1. Regression Metrics
    rmse = float(np.sqrt(mean_squared_error(reg_targets, reg_preds)))
    try:
        p_corr, _ = pearsonr(reg_targets, reg_preds)
        pearson_r = float(p_corr) if not np.isnan(p_corr) else 0.0
    except Exception:
        pearson_r = 0.0

    # 2. Classification Metrics
    try:
        roc_auc = float(roc_auc_score(cls_targets, cls_probs))
    except Exception:
        roc_auc = 0.5

    try:
        prec, rec, _ = precision_recall_curve(cls_targets, cls_probs)
        pr_auc = float(auc(rec, prec))
    except Exception:
        pr_auc = 0.0

    cls_preds_binary = (cls_probs >= 0.5).astype(int)
    f1 = float(f1_score(cls_targets, cls_preds_binary, zero_division=0))

    # 3. Calibration & Brier Score
    brier = float(brier_score_loss(cls_targets, cls_probs))

    # Plot Reliability Diagram
    prob_true, prob_pred = calibration_curve(cls_targets, cls_probs, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, "s-", color="#1f77b4", linewidth=2, label=f"Multimodal (BS={brier:.4f})")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Reliability Diagram (Classification Calibration)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    rel_diagram_path = output_dir / "reliability_diagram.png"
    fig.savefig(rel_diagram_path, dpi=300)
    plt.close(fig)
    log.info("[EVAL] Saved reliability diagram to %s", rel_diagram_path)

    metrics_dict = {
        "regression": {
            "rmse": round(rmse, 6),
            "pearson_r": round(pearson_r, 6),
        },
        "classification": {
            "auroc": round(roc_auc, 6),
            "auprc": round(pr_auc, 6),
            "f1_at_0.5": round(f1, 6),
            "brier_score": round(brier, 6),
        },
    }

    return metrics_dict


# ============================================================================
# 4. Integrated Gradients Attribution on Sequence Branch
# ============================================================================

def compute_integrated_gradients(
    model: MultimodalCRISPR,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 50,
    steps: int = 20,
) -> Dict[str, float]:
    """Compute Integrated Gradients attribution on sequence representation."""
    log.info("[ATTRIBUTION] Computing Integrated Gradients on sequence branch (%d samples)...", num_samples)

    try:
        from captum.attr import IntegratedGradients
        captum_available = True
    except ImportError:
        captum_available = False

    model.eval()

    # Collect sample tensors
    sample_seq_ids, sample_seq_mask, sample_epigen, sample_genome, sample_sc = [], [], [], [], []

    count = 0
    for batch in test_loader:
        sample_seq_ids.append(batch["seq_ids"])
        sample_seq_mask.append(batch["seq_mask"])
        sample_epigen.append(batch["epigen_vec"])
        sample_genome.append(batch["genome_vec"])
        sample_sc.append(batch["sc_expr_vec"])
        count += batch["seq_ids"].size(0)
        if count >= num_samples:
            break

    seq_ids_t  = torch.cat(sample_seq_ids, dim=0)[:num_samples].to(device)
    seq_mask_t = torch.cat(sample_seq_mask, dim=0)[:num_samples].to(device)
    epigen_t   = torch.cat(sample_epigen, dim=0)[:num_samples].to(device)
    genome_t   = torch.cat(sample_genome, dim=0)[:num_samples].to(device)
    sc_t       = torch.cat(sample_sc, dim=0)[:num_samples].to(device)

    # Compute CLS token embeddings as baseline / target
    with torch.no_grad():
        cls_embs = model._get_cls_emb(seq_ids_t, seq_mask_t, cls_emb=None)  # [N, 768]

    # Model wrapper over CLS embeddings for IG attribution
    class SeqModelWrapper(nn.Module):
        def __init__(self, full_model: MultimodalCRISPR):
            super().__init__()
            self.fm = full_model

        def forward(self, cls_emb, epigen_vec, genome_vec, sc_vec):
            reg_out, cls_out, _ = self.fm(
                input_ids=seq_ids_t,
                attention_mask=seq_mask_t,
                epigen_vec=epigen_vec,
                genome_vec=genome_vec,
                sc_expr_vec=sc_vec,
                cls_emb=cls_emb,
            )
            return cls_out  # high_eff classification logit

    wrapper = SeqModelWrapper(model)

    if captum_available:
        ig = IntegratedGradients(wrapper)
        baseline = torch.zeros_like(cls_embs)
        attributions, delta = ig.attribute(
            inputs=cls_embs,
            baselines=baseline,
            additional_forward_args=(epigen_t, genome_t, sc_t),
            return_convergence_delta=True,
        )
        attr_np = attributions.cpu().numpy()
    else:
        # Self-contained PyTorch Integrated Gradients fallback
        baseline = torch.zeros_like(cls_embs)
        scaled_inputs = [baseline + (float(i) / steps) * (cls_embs - baseline) for i in range(steps + 1)]
        grads = []
        for input_i in scaled_inputs:
            input_i.requires_grad = True
            output = wrapper(input_i, epigen_t, genome_t, sc_t)
            grad = torch.autograd.grad(output.sum(), input_i)[0]
            grads.append(grad.detach().cpu())
        avg_grads = torch.stack(grads, dim=0).mean(dim=0)
        attr_np = ((cls_embs.cpu() - baseline.cpu()) * avg_grads).numpy()

    # Aggregate importance per hidden feature dimension (768-d CLS embedding)
    mean_attr = np.abs(attr_np).mean(axis=0)  # [768]
    top10_idx = np.argsort(mean_attr)[-10:][::-1]
    top10_vals = mean_attr[top10_idx]

    # Plot sequence attribution bar plot
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(10), top10_vals, color="#2ca02c", alpha=0.85)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"Dim {i}" for i in top10_idx], rotation=45)
    ax.set_ylabel("Mean |Attribution|", fontsize=11)
    ax.set_title("Integrated Gradients: Top 10 Sequence CLS Dimensions", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    seq_plot_path = output_dir / "attribution_seq.png"
    fig.savefig(seq_plot_path, dpi=300)
    plt.close(fig)
    log.info("[ATTRIBUTION] Saved sequence attribution plot to %s", seq_plot_path)

    return {"top_seq_cls_dim": int(top10_idx[0]), "top_seq_cls_attr": float(top10_vals[0])}


# ============================================================================
# 5. SHAP Values on Epigenetic and 3D-Genome Branches
# ============================================================================

def compute_epigen_3d_shap(
    model: MultimodalCRISPR,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 100,
    bg_samples: int = 20,
) -> Dict[str, Any]:
    """Compute SHAP feature values on Epigenetic (6 marks) and 3D Genome (8 features)."""
    log.info("[SHAP] Computing SHAP values on Epigenetic & 3D Genome branches...")

    epigen_marks = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3", "ATAC-seq", "DNase-seq"]
    genome_features = ["loop_count", "tad_dist", "A1", "A2", "B1", "B2", "B3", "NA"]

    model.eval()

    # Collect data for SHAP
    all_epigen, all_genome, all_sc, all_cls_emb = [], [], [], []

    count = 0
    with torch.no_grad():
        for batch in test_loader:
            seq_ids  = batch["seq_ids"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            cls_emb  = batch.get("cls_emb")
            if cls_emb is None:
                cls_emb = model._get_cls_emb(seq_ids, seq_mask, cls_emb=None)
            else:
                cls_emb = cls_emb.to(device)

            all_cls_emb.append(cls_emb.cpu())
            all_epigen.append(batch["epigen_vec"])
            all_genome.append(batch["genome_vec"])
            all_sc.append(batch["sc_expr_vec"])
            count += seq_ids.size(0)
            if count >= num_samples:
                break

    epigen_np  = torch.cat(all_epigen, dim=0)[:num_samples].numpy()   # [N, 6]
    genome_np  = torch.cat(all_genome, dim=0)[:num_samples].numpy()   # [N, 8]
    sc_t       = torch.cat(all_sc, dim=0)[:num_samples].to(device)    # [N, 1]
    cls_emb_t  = torch.cat(all_cls_emb, dim=0)[:num_samples].to(device)# [N, 768]

    # Combine epigenetic (6) + 3D genome (8) into a single 14-dim input for SHAP
    combined_features = np.hstack([epigen_np, genome_np])  # [N, 14]
    feature_names = epigen_marks + genome_features

    # Use a single reference sample for the fixed modalities (sc_expr, cls_emb).
    # SHAP's KernelExplainer generates coalition batches of arbitrary size
    # (often >> num_samples), so we expand the reference to match each batch.
    sc_ref = sc_t[:1]          # [1, sc_dim]
    cls_emb_ref = cls_emb_t[:1]  # [1, 768]

    # Model prediction function taking [B, 14] combined vector
    def predict_fn(combined_arr: np.ndarray) -> np.ndarray:
        n = len(combined_arr)
        arr_t = torch.from_numpy(combined_arr).float().to(device)
        epigen_in = arr_t[:, :6]
        genome_in = arr_t[:, 6:]
        sc_in = sc_ref.expand(n, -1)
        cls_in = cls_emb_ref.expand(n, -1)
        with torch.no_grad():
            reg_out, cls_out, _ = model(
                input_ids=None,
                attention_mask=None,
                epigen_vec=epigen_in,
                genome_vec=genome_in,
                sc_expr_vec=sc_in,
                cls_emb=cls_in,
            )
            probs = torch.sigmoid(cls_out.squeeze(-1)).cpu().numpy()
        return probs

    # Compute SHAP values using KernelExplainer or Permutation explainer
    bg_data = combined_features[:bg_samples]
    explainer = shap.KernelExplainer(predict_fn, bg_data)

    eval_data = combined_features[:min(50, num_samples)]
    shap_values = explainer.shap_values(eval_data, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # [14]

    epigen_shap = mean_abs_shap[:6]
    genome_shap = mean_abs_shap[6:]

    top_epigen_idx = int(np.argmax(epigen_shap))
    top_genome_idx = int(np.argmax(genome_shap))

    top_epigen_mark = epigen_marks[top_epigen_idx]
    top_genome_feat = genome_features[top_genome_idx]

    # Plot SHAP importance bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.barh(epigen_marks, epigen_shap, color="#ff7f0e", alpha=0.85)
    ax1.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax1.set_title("Epigenetic Branch (6 Marks)", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="x", linestyle=":", alpha=0.6)

    ax2.barh(genome_features, genome_shap, color="#9467bd", alpha=0.85)
    ax2.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax2.set_title("3D Genome Branch (8 Features)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.suptitle("SHAP Feature Importance (Multimodal CRISPR Predictor)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    shap_plot_path = output_dir / "attribution_epigen_3d.png"
    fig.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("[SHAP] Saved epigenetic & 3D genome SHAP plot to %s", shap_plot_path)

    return {
        "top_epigenetic_mark": top_epigen_mark,
        "top_epigenetic_shap": float(epigen_shap[top_epigen_idx]),
        "top_3dgenome_feature": top_genome_feat,
        "top_3dgenome_shap": float(genome_shap[top_genome_idx]),
        "mean_epigenetic_shap": {m: float(s) for m, s in zip(epigen_marks, epigen_shap)},
        "mean_3dgenome_shap": {f: float(s) for f, s in zip(genome_features, genome_shap)},
    }


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Multimodal CRISPR Predictor.")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate on first N test samples.")
    args = parser.parse_args()

    log.info("=" * 75)
    log.info("Starting Multimodal CRISPR Predictor Full Evaluation & Attribution")
    log.info("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model, ckpt_path = load_evaluation_model(device)

    # 2. Load Test Split
    test_pt = TENSOR_DIR / "test.pt"
    if not test_pt.exists():
        log.error("Test dataset file %s not found. Run utils/preprocess.py first.", test_pt)
        sys.exit(1)

    test_ds = MultimodalCRISPRDataset(test_pt)
    if args.sample is not None:
        test_ds.data = test_ds.data[:args.sample]
        log.info("[EVAL] Subsampled test split to %d examples.", len(test_ds))

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 3. Run Inference & Compute Quantitative Metrics
    results = run_test_inference(model, test_loader, device)
    metrics_summary = compute_metrics_and_plots(results, OUTPUT_DIR)

    log.info("=" * 75)
    log.info("QUANTITATIVE TEST METRICS:")
    log.info("  Regression  : RMSE = %.4f | Pearson r = %.4f",
             metrics_summary["regression"]["rmse"], metrics_summary["regression"]["pearson_r"])
    log.info("  Classification: AUROC = %.4f | AUPRC = %.4f | F1@0.5 = %.4f | Brier = %.4f",
             metrics_summary["classification"]["auroc"], metrics_summary["classification"]["auprc"],
             metrics_summary["classification"]["f1_at_0.5"], metrics_summary["classification"]["brier_score"])
    log.info("=" * 75)

    # 4. Integrated Gradients Attribution
    seq_attr_summary = compute_integrated_gradients(model, test_loader, device, OUTPUT_DIR)

    # 5. SHAP Feature Attribution
    shap_attr_summary = compute_epigen_3d_shap(model, test_loader, device, OUTPUT_DIR)

    # 6. Combine and Export Metrics JSON
    metrics_summary["checkpoint_used"] = str(ckpt_path)
    metrics_summary["test_samples"] = len(test_ds)
    metrics_summary["attribution_summary"] = {
        "sequence": seq_attr_summary,
        "epigenetic_3d": shap_attr_summary,
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # 7. Comprehensive TensorBoard Logging
    tb_writer = SummaryWriter(log_dir=str(TB_DIR))

    # (a) Log Scalars
    tb_writer.add_scalar("Evaluation/RMSE", metrics_summary["regression"]["rmse"])
    tb_writer.add_scalar("Evaluation/Pearson_R", metrics_summary["regression"]["pearson_r"])
    tb_writer.add_scalar("Evaluation/AUROC", metrics_summary["classification"]["auroc"])
    tb_writer.add_scalar("Evaluation/AUPRC", metrics_summary["classification"]["auprc"])
    tb_writer.add_scalar("Evaluation/F1_at_0.5", metrics_summary["classification"]["f1_at_0.5"])
    tb_writer.add_scalar("Evaluation/Brier_Score", metrics_summary["classification"]["brier_score"])

    # (b) Log PR Curve
    try:
        tb_writer.add_pr_curve("Evaluation/PR_Curve", results["cls_targets"], results["cls_probs"])
    except Exception as e:
        log.warning("[TB] Could not log PR curve: %s", e)

    # (c) Log Images
    log_image_to_tb(tb_writer, "Evaluation/Plots/01_Reliability_Diagram", OUTPUT_DIR / "reliability_diagram.png")
    log_image_to_tb(tb_writer, "Evaluation/Plots/02_Sequence_Integrated_Gradients", OUTPUT_DIR / "attribution_seq.png")
    log_image_to_tb(tb_writer, "Evaluation/Plots/03_Epigenetic_and_3D_Genome_SHAP", OUTPUT_DIR / "attribution_epigen_3d.png")
    log_image_to_tb(tb_writer, "Evaluation/Plots/04_Benchmark_Comparison", OUTPUT_DIR / "benchmark_comparison.png")

    # (d) Log Markdown Summaries
    eval_markdown = f"""### Multimodal CRISPR Evaluation Summary
- **Test Samples**: {len(test_ds)}
- **Checkpoint Used**: `{ckpt_path.name}`

| Task | Metric | Value |
|---|---|---|
| **Regression** | RMSE | {metrics_summary['regression']['rmse']:.4f} |
| **Regression** | Pearson r | {metrics_summary['regression']['pearson_r']:.4f} |
| **Classification** | AUROC | {metrics_summary['classification']['auroc']:.4f} |
| **Classification** | AUPRC | {metrics_summary['classification']['auprc']:.4f} |
| **Classification** | F1 @ 0.5 | {metrics_summary['classification']['f1_at_0.5']:.4f} |
| **Calibration** | Brier Score | {metrics_summary['classification']['brier_score']:.4f} |

#### Key Attribution Insights
- **Top Sequence Hidden Dimension**: Dim {seq_attr_summary['top_seq_cls_dim']} (Mean |Attribution| = {seq_attr_summary['top_seq_cls_attr']:.6f})
- **Top Epigenetic Mark**: `{shap_attr_summary['top_epigenetic_mark']}` (Mean |SHAP| = {shap_attr_summary['top_epigenetic_shap']:.6f})
- **Top 3D Structural Feature**: `{shap_attr_summary['top_3dgenome_feature']}` (Mean |SHAP| = {shap_attr_summary['top_3dgenome_shap']:.6f})
"""
    tb_writer.add_text("Evaluation/Summary_Report", eval_markdown)
    tb_writer.flush()
    tb_writer.close()
    log.info("[TB] Successfully logged metrics, PR curve, images, and text report to TensorBoard (%s)", TB_DIR)

    log.info("=" * 75)
    log.info("Successfully exported evaluation summary to %s", METRICS_JSON_PATH)
    log.info("Generated Plots:")
    log.info("  - %s", OUTPUT_DIR / "reliability_diagram.png")
    log.info("  - %s", OUTPUT_DIR / "attribution_seq.png")
    log.info("  - %s", OUTPUT_DIR / "attribution_epigen_3d.png")
    log.info("=" * 75)


if __name__ == "__main__":
    main()
