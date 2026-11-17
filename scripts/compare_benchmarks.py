"""Descriptive comparison with literature-reported baseline references.

Compares current metrics from outputs/metrics.json against published baseline
metrics recorded below. Baseline code in data/benchmarks/ is not run by this script.
The baseline list includes:
  1. CRISPR-Net (Lin et al. 2020) - Recurrent CNN (Sequence)
  2. CRISPR-DIPOFF (Pranto et al. 2025) - Interpretable RNN (Sequence)
  3. DNABERT-6 (Zhou et al. 2021) - Vanilla DNABERT (Sequence)
  4. EpiCRISPR (Peng et al. 2023) - Flanking + Epigenetic
  5. DNABERT-Epi (Kimata et al. 2025) - DNABERT + Epigenetics (Strongest Baseline)
  6. Multimodal CRISPR (Ours) - Sequence + Epigenetic + 3D Genome + Single-Cell

Outputs:
  - outputs/benchmark_comparison.json
  - outputs/benchmark_comparison.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
BENCHMARK_JSON = OUTPUT_DIR / "benchmark_comparison.json"
BENCHMARK_PLOT = OUTPUT_DIR / "benchmark_comparison.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Safe console output across Windows cp1252 / UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "benchmark_comparison.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Baseline benchmarks data gathered from data/benchmarks/ repositories & publications
BASELINES: Dict[str, Dict[str, Any]] = {
    "CRISPR-DIPOFF": {
        "modalities": "Sequence",
        "architecture": "Bidirectional LSTM + Attention",
        "auroc": 0.7820,
        "auprc": 0.6240,
        "f1": 0.6850,
        "reference": "Pranto et al. 2025 (data/benchmarks/02_CRISPR-DIPOFF)",
    },
    "CRISPR-Net": {
        "modalities": "Sequence (mismatches + indels)",
        "architecture": "Recurrent CNN",
        "auroc": 0.8040,
        "auprc": 0.6480,
        "f1": 0.7120,
        "reference": "Lin & Wong 2020 (data/benchmarks/01_CRISPR-Net)",
    },
    "DNABERT-6 (Vanilla)": {
        "modalities": "Sequence (6-mer tokens)",
        "architecture": "12-layer Transformer (BERT-base)",
        "auroc": 0.8160,
        "auprc": 0.6650,
        "f1": 0.7280,
        "reference": "Zhou et al. 2021 (data/benchmarks/03_DNABERT)",
    },
    "EpiCRISPR": {
        "modalities": "Sequence + Epigenetics (ChIP-seq)",
        "architecture": "BiLSTM + Epigenetic Dense",
        "auroc": 0.8290,
        "auprc": 0.6830,
        "f1": 0.7410,
        "reference": "Peng et al. 2023 (data/benchmarks/06_EpiCRISPR)",
    },
    "DNABERT-Epi": {
        "modalities": "Sequence + Epigenetics (ATAC/H3K27ac/H3K4me3)",
        "architecture": "DNABERT + Epigenetic Fusion MLP",
        "auroc": 0.8450,
        "auprc": 0.7090,
        "f1": 0.7620,
        "reference": "Kimata et al. PLOS ONE 2025 (data/benchmarks/04_DNABERT-Epi)",
    },
}

# Production target for the 4-layer multimodal architecture (Seq + Epigen + 3D + scRNA)
OUR_MULTIMODAL_MODEL = {
    "name": "Multimodal CRISPR (Ours)",
    "modalities": "Sequence + Epigenetics + 3D Genome + Single-Cell",
    "architecture": "DNABERT-6 + 4-Branch MLP Fusion + Dual Heads",
    "auroc": None,
    "auprc": None,
    "f1": None,
    "reference": "Our 4-Modality Engine (576-d Fusion)",
}


def run_benchmark_comparison() -> Dict[str, Any]:
    log.info("=" * 78)
    log.info("CRISPR Predictor: Descriptive Literature Reference Comparison")
    log.info("=" * 78)

    if not METRICS_JSON.is_file():
        raise FileNotFoundError(f"Run evaluate.py first; model metrics not found: {METRICS_JSON}")
    with open(METRICS_JSON, encoding="utf-8") as metrics_file:
        measured = json.load(metrics_file)
    provenance = measured.get("provenance", {})
    if provenance.get("evaluation_scope") != "full_test" or provenance.get("evaluated_samples") != provenance.get("full_test_samples"):
        raise ValueError("Benchmark output requires canonical full-test metrics in outputs/metrics.json; sampled or legacy metrics are not accepted.")
    OUR_MULTIMODAL_MODEL["auroc"] = float(measured["classification"]["auroc"])
    OUR_MULTIMODAL_MODEL["auprc"] = float(measured["classification"]["auprc"])
    OUR_MULTIMODAL_MODEL["f1"] = float(measured["classification"]["f1_at_0.5"])
    our_auroc = OUR_MULTIMODAL_MODEL["auroc"]
    strongest_baseline_name = "DNABERT-Epi"
    strongest_baseline_auroc = BASELINES[strongest_baseline_name]["auroc"]

    relative_gain = ((our_auroc - strongest_baseline_auroc) / strongest_baseline_auroc) * 100.0
    absolute_gain = our_auroc - strongest_baseline_auroc

    log.info("Current full-test model AUROC: %.4f", our_auroc)
    log.info("DNABERT-Epi literature-reported AUROC reference: %.4f", strongest_baseline_auroc)
    log.info("Descriptive numeric difference across reported protocols: %+.4f (%+.2f%%)", absolute_gain, relative_gain)
    log.warning("Baseline implementations were not independently run; differences are not head-to-head evidence and do not establish superiority or a target result.")

    # Build comparison records
    comparison_records = []
    for name, data in BASELINES.items():
        rel_g = ((our_auroc - data["auroc"]) / data["auroc"]) * 100.0
        abs_g = our_auroc - data["auroc"]
        comparison_records.append({
            "model": name,
            "modalities": data["modalities"],
            "architecture": data["architecture"],
            "auroc": data["auroc"],
            "auprc": data["auprc"],
            "f1": data["f1"],
            "descriptive_auroc_difference_vs_literature_reference": round(abs_g, 4),
            "descriptive_relative_auroc_difference_percent": round(rel_g, 2),
            "reference": data["reference"],
            "result_type": "literature-reported reference; baseline not independently reproduced",
        })

    # Summary table
    log.info("-" * 78)
    log.info("%-22s | %-10s | %-10s | %-10s | %-24s", "Model/reference", "AUROC", "AUPRC", "F1", "Descriptive difference")
    log.info("-" * 78)
    for r in comparison_records:
        log.info("%-22s |   %.4f   |   %.4f   |   %.4f   | %+.4f descriptive",
                 r["model"], r["auroc"], r["auprc"], r["f1"], r["descriptive_auroc_difference_vs_literature_reference"])
    log.info("%-22s |   %.4f   |   %.4f   |   %.4f   |   (current full-test result)",
             OUR_MULTIMODAL_MODEL["name"], OUR_MULTIMODAL_MODEL["auroc"],
             OUR_MULTIMODAL_MODEL["auprc"], OUR_MULTIMODAL_MODEL["f1"])
    log.info("-" * 78)

    # Assert criterion
    log.info("Comparisons use different published protocols and are not reproduced head-to-head experiments.")

    # Export JSON
    output_data = {
        "our_model": OUR_MULTIMODAL_MODEL,
        "strongest_baseline": strongest_baseline_name,
        "comparison_type": "descriptive comparison; literature-reported baseline references only",
        "provenance": provenance,
        "warning": "Baseline implementations were not independently run. Different datasets/protocols mean these differences do not establish superiority or meeting a target.",
        "descriptive_relative_difference_percent": round(relative_gain, 2),
        "descriptive_absolute_difference": round(absolute_gain, 4),
        "baseline_comparisons": comparison_records,
    }
    with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    log.info("Saved benchmark comparison JSON to %s", BENCHMARK_JSON)

    # Generate Visualization Plot
    models = [r["model"] for r in comparison_records] + [OUR_MULTIMODAL_MODEL["name"]]
    aurocs = [r["auroc"] for r in comparison_records] + [OUR_MULTIMODAL_MODEL["auroc"]]
    colors = ["#4c72b0", "#4c72b0", "#55a868", "#c44e52", "#8172b3", "#d95f02"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, aurocs, color=colors, height=0.6, alpha=0.9, edgecolor="black", linewidth=0.8)

    # Reference threshold lines
    ax.axvline(x=strongest_baseline_auroc, color="grey", linestyle="--", alpha=0.7, label=f"DNABERT-Epi literature reference ({strongest_baseline_auroc:.3f})")
    ax.set_xlim(0.70, 0.95)
    ax.set_xlabel("AUROC (model and literature-reported references)", fontsize=12, fontweight="bold")
    ax.set_title("Descriptive AUROC Reference Comparison (Protocols Differ)", fontsize=13, fontweight="bold", pad=12)

    # Bar labels
    for bar, auroc in zip(bars, aurocs):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{auroc:.4f}",
                va="center", ha="left", fontsize=10, fontweight="bold")

    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(BENCHMARK_PLOT, dpi=300)
    plt.close(fig)
    log.info("Saved benchmark comparison figure to %s", BENCHMARK_PLOT)

    # -------------------------------------------------------------
    # TensorBoard Logging
    # -------------------------------------------------------------
    try:
        tb_log_dir = PROJECT_ROOT / "logs" / "tensorboard"
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

        # 1. Scalars
        tb_writer.add_scalar("Benchmarks/Descriptive_Relative_Difference_Pct", relative_gain, 0)
        tb_writer.add_scalar("Benchmarks/Descriptive_Absolute_Difference", absolute_gain, 0)
        tb_writer.add_scalar("Benchmarks/Current_Full_Test_AUROC", our_auroc, 0)
        tb_writer.add_scalar("Benchmarks/DNABERT_Epi_Literature_Reference_AUROC", strongest_baseline_auroc, 0)

        for r in comparison_records:
            model_tag = r["model"].replace(" ", "_").replace("(", "").replace(")", "")
            tb_writer.add_scalar(f"Benchmarks/AUROC/{model_tag}", r["auroc"], 0)
            tb_writer.add_scalar(f"Benchmarks/AUPRC/{model_tag}", r["auprc"], 0)
            tb_writer.add_scalar(f"Benchmarks/F1/{model_tag}", r["f1"], 0)

        # 2. Image
        img_arr = plt.imread(str(BENCHMARK_PLOT))
        img_tensor = torch.from_numpy(img_arr).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.permute(2, 0, 1)[:3]
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            tb_writer.add_image("Benchmarks/Baseline_Comparison_Plot", img_tensor, 0)

        # 3. Markdown Text Table
        md_table = "### Descriptive comparison with literature-reported reference results\n\n"
        md_table += "Baseline implementations were not independently run. Published datasets and protocols may differ; numeric differences are not head-to-head evidence and do not establish superiority or target attainment.\n\n"
        md_table += "| Model/reference | Modalities | Architecture | Reported AUROC | Reported AUPRC | Reported F1 | Descriptive AUROC difference |\n"
        md_table += "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n"
        for r in comparison_records:
            md_table += f"| {r['model']} (literature reference) | {r['modalities']} | {r['architecture']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['f1']:.4f} | {r['descriptive_auroc_difference_vs_literature_reference']:+.4f} |\n"
        md_table += f"| {OUR_MULTIMODAL_MODEL['name']} (current full-test metrics) | {OUR_MULTIMODAL_MODEL['modalities']} | {OUR_MULTIMODAL_MODEL['architecture']} | {OUR_MULTIMODAL_MODEL['auroc']:.4f} | {OUR_MULTIMODAL_MODEL['auprc']:.4f} | {OUR_MULTIMODAL_MODEL['f1']:.4f} | n/a |\n"

        tb_writer.add_text("Benchmarks/Comparison_Report", md_table, 0)
        tb_writer.flush()
        tb_writer.close()
        log.info("Successfully logged benchmark comparisons and plots to TensorBoard (%s)", tb_log_dir)
    except Exception as tb_err:
        log.warning("TensorBoard logging failed for benchmark comparison: %s", tb_err)

    return output_data


if __name__ == "__main__":
    run_benchmark_comparison()
