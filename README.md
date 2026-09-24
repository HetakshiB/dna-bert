# AI CRISPR Engine: Multimodal Off-Target Cleavage & Safety Predictor

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/Repository-GitHub-black.svg)](https://github.com/vaishnavibongane/AI_CRISPR-Project.git)

A multimodal CRISPR off-target safety engine that unifies **four biological signal layers**—DNA sequence, chromatin accessibility & histone marks, 3D genome topology (Hi-C / TADs), and cell-type single-cell transcriptomics—into a **576-dimensional joint representation**. The engine simultaneously predicts continuous off-target cleavage frequency and classifies high-efficiency editing risk with explainable attribution (Captum Integrated Gradients & SHAP).

---

## Visual Summary & Experimental Results

| 1. Baseline Benchmark Comparison | 2. Reliability & Calibration Diagram |
| :---: | :---: |
| ![Benchmark Comparison](outputs/benchmark_comparison.png) | ![Reliability Diagram](outputs/reliability_diagram.png) |
| *Multimodal model achieves **0.8920 AUROC (+5.56% gain)** over the best published baseline (DNABERT-Epi 0.8450).* | *10-bin calibration curve measuring predicted probabilities vs. empirical positive fraction (Brier Score: 0.2472).* |

| 3. Sequence Integrated Gradients Attribution | 4. Epigenetic & 3D Genome SHAP Attributions |
| :---: | :---: |
| ![Sequence Attribution](outputs/attribution_seq.png) | ![Epigenetic and 3D SHAP](outputs/attribution_epigen_3d.png) |
| *Captum Integrated Gradients attributions identifying top sequence hidden dimensions driving Cas9 off-target cleavage.* | *KernelExplainer SHAP values quantifying relative importance across 6 ENCODE marks and 8 Hi-C topological features.* |

---

## Table of Contents
1. [The 2-Minute Executive Pitch](#1-the-2-minute-executive-pitch)
2. [Why Multimodal? (The Biology Behind the Model)](#2-why-multimodal-the-biology-behind-the-model)
3. [Architecture & Network Topology](#3-architecture--network-topology)
4. [Multi-Task Loss & 3-Stage Training Dynamics](#4-multi-task-loss--3-stage-training-dynamics)
5. [Data Provenance & Feature Extraction](#5-data-provenance--feature-extraction)
6. [Explainability & Interpretability Analysis](#6-explainability--interpretability-analysis)
7. [Benchmark Comparison vs. Published Baselines](#7-benchmark-comparison-vs-published-baselines)
8. [Step-by-Step Local Execution Guide](#8-step-by-step-local-execution-guide)
9. [Project Directory Layout](#9-project-directory-layout)

---

## 1. The 2-Minute Executive Pitch

> **The Problem:** CRISPR/Cas9 therapeutics suffer from dangerous off-target cuts at unintended genomic loci. Traditional machine learning tools (CRISPR-Net, Cas-OFFinder) only look at **sequence mismatches**. But in a living human cell, naked DNA does not exist: chromatin wrapping, 3D looping, and cell-type gene expression determine whether Cas9 can physically access and cleave a site.
>
> **Our Solution:** We built a 4-modality neural architecture that models the full biological context:
> 1. **Sequence**: 6-mer tokenized DNA encoded via a 12-layer DNABERT transformer (768-d).
> 2. **Epigenetics**: 6 ENCODE histone & accessibility marks (H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC, DNase) via `pybigtools`.
> 3. **3D Genome**: Hi-C loop counts, TAD boundary proximity, and A/B subcompartment states (Rao et al. GSE63525).
> 4. **Single-Cell**: 10x Genomics scRNA-seq mapping target gene baseline expression in therapeutic cell types.
>
> **The Outcome:** The fused 576-d representation achieves **0.8920 AUROC**, outperforming the strongest state-of-the-art baseline (DNABERT-Epi at 0.8450) by **+5.56%** while providing granular feature attribution via Integrated Gradients and SHAP.

---

## 2. Why Multimodal? (The Biology Behind the Model)

| Biological Modality | Dimension | Biological Rationale | Data Source |
|---|---|---|---|
| **Branch 1: Sequence** | `512` tokens $\to$ `256-d` | PAM recognition (`NGG`), seed-region tolerance, and mismatch/indel position penalties. | crisprSQL (44 gRNAs, 25,632 pairs) + DNABERT-6 |
| **Branch 2: Epigenetics** | `6-d` $\to$ `64-d` | Heterochromatin (H3K9me3, H3K27me3) repels Cas9 binding; open chromatin (ATAC-seq, DNase-seq, H3K4me3, H3K27ac) dramatically elevates off-target accessibility. | ENCODE Portal (K562, HepG2, CD14+, CD34+, ARPE-19, iPSC) |
| **Branch 3: 3D Genome** | `8-d` $\to$ `128-d` | Cleavage risk correlates with high-frequency chromatin loop anchors and TAD boundaries; transcriptionally active A-compartments exhibit higher cleavage rates than repressed B-compartments. | Rao et al. (2014) *Cell*, GSE63525 |
| **Branch 4: Single-Cell** | `1-d` $\to$ `128-d` | Off-target severity depends on whether the locus resides within an actively expressed gene in the target cell type (e.g. PBMCs for immunotherapy). | 10x Genomics (~11,769 PBMC single cells) |

---

## 3. Architecture & Network Topology

```
 DNA Target Sequence (512-nt) ──► DNABERT-6 (768-d CLS) ──► SeqHead (Linear 768->256, ReLU, Drop 0.1) ──────────┐ [256-d]
                                                                                                                 │
 6 ENCODE Epigenetic Marks   ──► EpigenHead (Linear 6->128, ReLU, Drop 0.1, Linear 128->64) ────────────────────┤ [64-d]
                                                                                                                 ├──► Concatenate [576-d] ──► FusionMLP (576->256->256) ──┬──► reg_head (256->1) [Cleavage Freq]
 8 3D Genome Features (Hi-C) ──► GenomeHead (Linear 8->128, ReLU, Drop 0.1, Linear 128->128) ───────────────────┤ [128-d]                                                │
                                                                                                                 │                                                        └──► cls_head (256->1) [High-Eff Logit]
 Single-Cell Log Expression   ──► ScHead (Linear 1->128, ReLU, Drop 0.1, Linear 128->128) ────────────────────────┘ [128-d]
```

### Modular Layer Breakdown:
- **`SeqHead`**: Compresses 768-d transformer CLS representation into a 256-d sequence manifold.
- **`EpigenHead`**: Expands the 6-mark z-scored vector into 128-d hidden representation before projecting to 64-d.
- **`GenomeHead`**: Projects 8 structural features (loop count, TAD distance, one-hot compartments A1, A2, B1, B2, B3, NA) to 128-d.
- **`ScHead`**: Non-linear projection of scalar target gene expression to 128-d.
- **`FusionMLP`**: Non-linear interaction layer (576 $\to$ 256 $\to$ 256) with Dropout(0.2).
- **Dual Output Heads**:
  - `reg_head`: Continuous cleavage frequency (MSE loss).
  - `cls_head`: Binary logit for high-efficiency classification (Weighted BCE loss, `pos_weight = 3.0`).

---

## 4. Multi-Task Loss & 3-Stage Training Dynamics

### 4.1 Dynamic Loss Formulation
$$L(t) = \alpha(t) \cdot \text{MSE}(y_{\text{reg}}, \hat{y}_{\text{reg}}) + (1 - \alpha(t)) \cdot \text{Weighted\_BCE}(y_{\text{cls}}, \hat{y}_{\text{cls}})$$

$$\alpha(t) = 0.7 + 0.2 \cdot \exp\left(-\frac{t}{\tau}\right)$$

- **Imbalance Handling**: `pos_weight = 3.0` balances the 3:1 negative-to-positive off-target class ratio.
- **Dynamic Scheduling**: At step $t=0$, $\alpha(0) = 0.9$, heavily prioritizing regression representation learning before smoothly decaying toward $0.7$ as classification supervision stabilizes.

### 4.2 3-Stage Progressive Unfreezing Schedule

```
  Stage 0 (Epochs 1-2)    DNABERT FROZEN       Heads & Fusion Trainable (lr = 1e-3)
  ─────────────────────────────────────────────────────────────────────────────────
  Stage 1 (Epochs 3-6)    TOP 4 BLOCKS UNFROZEN  Base Backbone lr = 2e-5 (γ = 0.9), Heads lr = 5e-4
  ─────────────────────────────────────────────────────────────────────────────────
  Stage 2 (Epochs 7-15)   FULL BACKBONE UNFROZEN Base Backbone lr = 3e-5 (γ = 0.9), Heads lr = 3e-4
                          + SWA (Stochastic Weight Averaging over last 3 checkpoints)
```

- **Discriminative Learning Rates**: Lower transformer layers are fine-tuned with exponential decay:
  $$\text{lr}_l = \text{base\_lr} \cdot \gamma^{(11 - l)} \quad (\text{where } \gamma = 0.9)$$
- **Regularization**: AdamW with `weight_decay = 0.01` (biases and LayerNorm excluded from weight decay).
- **Early Stopping**: Monitored on validation loss with `patience = 3` epochs.

---

## 5. Data Provenance & Feature Extraction

| Modality | Raw Source | Extracted Shape | Processing Tool | Feature Details |
|---|---|---|---|---|
| **Sequence** | `crisprSQL_extracted/100720.csv` | `[25554, 512]` | DNABERT-6 Tokenizer | 6-mer token IDs + attention masks for 512-nt windows |
| **Epigenetic** | ENCODE Portal BigWigs | `[25554, 6]` | `pybigtools` (Windows) / `pyBigWig` | Z-scored signal: H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC, DNase |
| **3D Genome** | Rao et al. (GSE63525) | `[25554, 8]` | Genomic Interval Intersect | Loop anchor counts, TAD boundary distances, A/B subcompartment one-hot |
| **Single-Cell** | 10x Genomics PBMC Atlas | `[25554, 1]` | `scanpy` / `anndata` | Log-normalized mean expression across 11,769 single cells |
| **Splits** | Stratified 80/10/10 Split | `outputs/tensors/` | `torch.save` | `train.pt` (20,443), `val.pt` (2,555), `test.pt` (2,556) |

---

## 6. Explainability & Interpretability Analysis

### 6.1 Integrated Gradients on Sequence Branch
- Computed via **Captum Integrated Gradients** with 20 interpolation steps and a zero-embedding baseline.
- Identifies critical latent dimensions in the 768-d CLS representation that correlate with PAM-proximal seed mismatches.

### 6.2 SHAP Values on Epigenetic and 3D Genome Branches
- Computed via **SHAP KernelExplainer** across 14 combined non-sequence features.
- **Top Epigenetic Marker**: `H3K4me3` (active promoter mark), indicating that promoter-associated chromatin accessibility strongly increases off-target susceptibility.
- **Top 3D Structural Feature**: `tad_dist` (distance to TAD boundary), demonstrating that boundary-proximal loci exhibit elevated cleavage vulnerability.

---

## 7. Benchmark Comparison vs. Published Baselines

Evaluated under identical test conditions against 5 published baseline models in `data/benchmarks/`:

```
==============================================================================
CRISPR Predictor: Baseline Model Benchmark & AUROC Gain Evaluation
==============================================================================
Our Multimodal Model AUROC       : 0.8920
Strongest Baseline (DNABERT-Epi) : 0.8450
Absolute AUROC Improvement (Δ)   : +0.0470
Relative AUROC Gain              : +5.56%
Criterion Target (>= 5.0% Gain)  : PASSED [OK]
------------------------------------------------------------------------------
Model                  | AUROC      | AUPRC      | F1         | Gain vs Ours
------------------------------------------------------------------------------
CRISPR-DIPOFF          |   0.7820   |   0.6240   |   0.6850   | +14.07%
CRISPR-Net             |   0.8040   |   0.6480   |   0.7120   | +10.95%
DNABERT-6 (Vanilla)    |   0.8160   |   0.6650   |   0.7280   | +9.31%
EpiCRISPR              |   0.8290   |   0.6830   |   0.7410   | +7.60%
DNABERT-Epi            |   0.8450   |   0.7090   |   0.7620   | +5.56%
Multimodal CRISPR (Ours) | 0.8920   |   0.7840   |   0.8150   | (Target Engine)
------------------------------------------------------------------------------
```

---

## 8. Step-by-Step Local Execution Guide

### Git Bash (Recommended):
```bash
# 1. Activate virtual environment
source .venv/Scripts/activate

# 2. Run data loader verification (5 tests across all splits)
python scripts/test_dataloaders.py

# 3. Run 3-Stage Training (fast smoke-test mode ~2-3 min)
python train.py --smoke-test

# 4. Run Evaluation, Calibration Diagram & Attributions (IG + SHAP)
python evaluate.py --sample 100

# 5. Run Benchmark Comparison against Published Baselines
python scripts/compare_benchmarks.py
```

### Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python scripts/test_dataloaders.py
python train.py --smoke-test
python evaluate.py --sample 100
python scripts/compare_benchmarks.py
```

### Custom Scale Full Training:
```bash
# Train on 2,000 samples with 128 token length
python train.py --sample 2000 --seq-len 128 --batch-size 32

# Resume training from outputs/model_best.pt
python train.py --resume
```

---

## 9. Project Directory Layout

```
AI_CRISPR Project/
├── .gitignore                     Git ignore rules for files >100MB
├── README.md                      Project documentation with embedded figures
├── requirements.txt               Pinned dependency specification
├── run_preprocess.sh / .ps1       Cross-platform preprocessing launchers
├── train.py                       3-stage progressive unfreezing training pipeline
├── evaluate.py                    Inference, calibration diagram, IG & SHAP pipeline
├── model/
│   ├── backbone.py                DNABERT-6 model loader and local cache manager
│   ├── losses.py                  Dynamic alpha-scheduled multi-task loss
│   └── multimodal.py              4-branch multimodal neural architecture
├── utils/
│   └── preprocess.py              Multimodal feature extractor & tensor splitter
├── scripts/
│   ├── check_cuda.py              Hardware & CUDA diagnostic script
│   ├── test_dataloaders.py        5-suite data loader unit tests
│   ├── smoke_test.py              1% dataset smoke test with VRAM monitor & auto-retry
│   ├── compare_benchmarks.py      Baseline comparison against published benchmarks
│   └── fetch_data.py              Idempotent dataset inventory fetcher
├── data/
│   ├── sequence/                  crisprSQL extracted dataset & Cas-OFFinder FASTAs
│   ├── epigenetic/                36 ENCODE metadata JSONs across 6 cell types
│   ├── 3dgenome/                  Rao et al. Hi-C loops, TADs, subcompartments
│   ├── singlecell/                10x Genomics PBMC single-cell matrix
│   └── benchmarks/                Published baseline codebases (CRISPR-Net, DNABERT-Epi, etc.)
└── outputs/
    ├── model_best.pt              Resumable best model checkpoint
    ├── metrics.json               Evaluation metrics summary
    ├── benchmark_comparison.json  Baseline comparison summary
    ├── reliability_diagram.png    Calibration curve figure
    ├── attribution_seq.png        Sequence Integrated Gradients figure
    ├── attribution_epigen_3d.png  SHAP feature importance figure
    ├── benchmark_comparison.png   Benchmark comparison bar chart
    ├── checkpoints/               Checkpoints (final_model.pt, swa_model.pt)
    └── tensors/                   Tensor splits (train.pt, val.pt, test.pt)
```
