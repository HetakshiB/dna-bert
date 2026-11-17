"""Training pipeline for the Multimodal CRISPR Off-Target Predictor.

Implements the 3-stage progressive unfreezing and discriminative learning rate schedule:

  Stage 0 (Epochs 1-2):
    - Backbone (DNABERT-6) completely frozen.
    - Heads (seq_head, epigen_head, genome_head, sc_head, fusion, reg_head, cls_head)
      trainable at head_lr = 1e-3.

  Stage 1 (Epochs 3-6):
    - Unfreeze top 4 transformer blocks of DNABERT (blocks 8..11).
    - Base backbone LR = 2e-5 with discriminative decay γ ≈ 0.9 per layer deeper.
    - Heads trainable at head_lr = 5e-4.

  Stage 2 (Epochs 7-15):
    - Full backbone unfrozen (blocks 0..11 + embeddings).
    - Base backbone LR = 3e-5 with discriminative decay γ ≈ 0.9 across all layers.
    - Heads trainable at head_lr = 3e-4.

Key Optimization & Engineering Features:
  - Early Stopping (patience=3 epochs) based on validation combined loss.
  - Resumable Checkpoint (outputs/model_best.pt) containing model, optimizer, scheduler, and epoch state.
  - Per-epoch Logging to CSV (logs/train_metrics.csv) and TensorBoard (logs/tensorboard/).
  - Stochastic Weight Averaging (SWA) over the last 3 Stage 2 checkpoints with single-pass norm stat update.
  - AdamW optimizer with weight_decay = 1e-2 (biases & LayerNorm excluded).
  - Warmup + Cosine decay schedule (10% warmup per stage).
  - Gradient accumulation (micro-batch size 32, accumulation steps 2 → effective batch size 64).
  - Mixed precision (torch.amp / GradScaler) with graceful CPU/CUDA fallback.

Usage:
  python train.py                     # Full 15-epoch 3-stage training
  python train.py --smoke-test        # Fast smoke test
  python train.py --resume            # Resume from outputs/model_best.pt
"""

from __future__ import annotations

import os
import shutil
import sys
import csv
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = PROJECT_ROOT / "models" / "hf-cache"
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_CACHE_DIR / "modules")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "hub")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, mean_squared_error
from scipy.stats import pearsonr
from transformers import get_cosine_schedule_with_warmup

TENSOR_DIR = PROJECT_ROOT / "outputs" / "tensors"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
BEST_MODEL_PATH = PROJECT_ROOT / "outputs" / "model_best.pt"
CSV_METRICS_PATH = LOG_DIR / "train_metrics.csv"
TB_DIR = LOG_DIR / "tensorboard"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

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
        logging.FileHandler(LOG_DIR / "train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(PROJECT_ROOT))
from model.multimodal import MultimodalCRISPR, build_model  # noqa: E402
from model.losses import MultimodalLoss                    # noqa: E402


# ============================================================================
# Dataset Definition
# ============================================================================

class MultimodalCRISPRDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed multimodal tensor dictionaries."""

    def __init__(self, pt_path: Path) -> None:
        if not pt_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {pt_path}")
        self.data: List[Dict[str, torch.Tensor]] = torch.load(pt_path, map_location="cpu", weights_only=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# ============================================================================
# Parameter Group Builder
# ============================================================================

def build_parameter_groups(
    model: MultimodalCRISPR,
    stage: int,
    base_backbone_lr: float,
    head_lr: float,
    weight_decay: float = 1e-2,
    gamma: float = 0.9,
) -> List[Dict[str, Any]]:
    """Build AdamW parameter groups with discriminative LRs and weight decay rules."""
    if model.dnabert is not None:
        if stage == 0:
            for p in model.dnabert.parameters():
                p.requires_grad = False
        elif stage == 1:
            for p in model.dnabert.parameters():
                p.requires_grad = False
            for layer_idx in range(8, 12):
                for p in model.dnabert.encoder.layer[layer_idx].parameters():
                    p.requires_grad = True
        elif stage == 2:
            for p in model.dnabert.parameters():
                p.requires_grad = True

    for name, p in model.named_parameters():
        if "dnabert" not in name:
            p.requires_grad = True

    param_groups: List[Dict[str, Any]] = []

    def should_no_decay(name: str, param: torch.Tensor) -> bool:
        if param.ndim <= 1 or "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            return True
        return False

    if model.dnabert is not None:
        for name, param in model.dnabert.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if should_no_decay(name, param) else weight_decay
            if "encoder.layer." in name:
                parts = name.split("encoder.layer.")[1].split(".")
                layer_idx = int(parts[0])
                decay_power = 11 - layer_idx
                group_lr = base_backbone_lr * (gamma ** decay_power)
            elif "embeddings" in name:
                group_lr = base_backbone_lr * (gamma ** 12)
            else:
                group_lr = base_backbone_lr

            param_groups.append({
                "params": [param],
                "lr": group_lr,
                "weight_decay": wd,
                "name": f"backbone_{name}",
            })

    for name, param in model.named_parameters():
        if "dnabert" in name or not param.requires_grad:
            continue
        wd = 0.0 if should_no_decay(name, param) else weight_decay
        param_groups.append({
            "params": [param],
            "lr": head_lr,
            "weight_decay": wd,
            "name": f"head_{name}",
        })

    return param_groups


# ============================================================================
# Evaluation Function
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: MultimodalLoss,
    device: torch.device,
    use_amp: bool = False,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on validation/test set and return metrics."""
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_bce = 0.0
    count = 0

    all_reg_preds: List[float] = []
    all_reg_targets: List[float] = []
    all_cls_probs: List[float] = []
    all_cls_targets: List[int] = []
    all_cls_logits: List[float] = []

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for batch in dataloader:
        seq_ids     = batch["seq_ids"].to(device)
        seq_mask    = batch["seq_mask"].to(device)
        epigen_vec  = batch["epigen_vec"].to(device)
        genome_vec  = batch["genome_vec"].to(device)
        sc_expr_vec = batch["sc_expr_vec"].to(device)
        reg_label   = batch["reg_label"].to(device)
        cls_label   = batch["cls_label"].to(device)

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            reg_out, cls_out, _ = model(
                input_ids=seq_ids,
                attention_mask=seq_mask,
                epigen_vec=epigen_vec,
                genome_vec=genome_vec,
                sc_expr_vec=sc_expr_vec,
            )
            loss, metrics = loss_fn(reg_out, cls_out, reg_label, cls_label, step_increment=0)

        batch_size = seq_ids.size(0)
        total_loss += metrics["loss"] * batch_size
        total_mse  += metrics["mse_loss"] * batch_size
        total_bce  += metrics["bce_loss"] * batch_size
        count      += batch_size

        all_reg_preds.extend(reg_out.squeeze(-1).cpu().numpy().tolist())
        all_reg_targets.extend(reg_label.cpu().numpy().tolist())

        cls_probs = torch.sigmoid(cls_out.squeeze(-1)).cpu().numpy().tolist()
        all_cls_probs.extend(cls_probs)
        if return_predictions:
            all_cls_logits.extend(cls_out.squeeze(-1).cpu().numpy().tolist())
        all_cls_targets.extend(cls_label.cpu().numpy().tolist())

    val_loss = total_loss / max(1, count)
    val_mse  = total_mse / max(1, count)
    val_bce  = total_bce / max(1, count)

    reg_rmse = float(np.sqrt(mean_squared_error(all_reg_targets, all_reg_preds)))
    try:
        p_corr, _ = pearsonr(all_reg_targets, all_reg_preds)
        pearson_r = float(p_corr) if not np.isnan(p_corr) else 0.0
    except Exception:
        pearson_r = 0.0

    try:
        roc_auc = float(roc_auc_score(all_cls_targets, all_cls_probs))
    except Exception:
        roc_auc = 0.5

    try:
        prec, rec, _ = precision_recall_curve(all_cls_targets, all_cls_probs)
        pr_auc = float(auc(rec, prec))
    except Exception:
        pr_auc = 0.0

    preds_binary = [1 if p >= 0.5 else 0 for p in all_cls_probs]
    f1 = float(f1_score(all_cls_targets, preds_binary, zero_division=0))

    result: Dict[str, Any] = {
        "val_loss": val_loss,
        "val_mse": val_mse,
        "val_bce": val_bce,
        "rmse": reg_rmse,
        "pearson_r": pearson_r,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
    }
    if return_predictions:
        result["predictions"] = {
            "reg_preds": np.asarray(all_reg_preds, dtype=np.float32),
            "reg_targets": np.asarray(all_reg_targets, dtype=np.float32),
            "cls_logits": np.asarray(all_cls_logits, dtype=np.float32),
            "cls_probs": np.asarray(all_cls_probs, dtype=np.float32),
            "cls_targets": np.asarray(all_cls_targets, dtype=np.int64),
        }
    return result


# ============================================================================
# CSV Logger Helper
# ============================================================================

def log_metrics_to_csv(csv_path: Path, row_dict: Dict[str, Any]) -> None:
    """Append metric row to CSV log file."""
    file_exists = csv_path.exists()
    headers = [
        "epoch", "stage", "train_loss", "train_mse", "train_bce",
        "val_loss", "val_mse", "val_bce", "val_rmse", "val_pearson_r",
        "val_auroc", "val_pr_auc", "val_f1", "learning_rate"
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Write checkpoints via a sibling temporary file to avoid partial final files."""
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


# ============================================================================
# SWA Weight Averaging & Normalization Update
# ============================================================================

def run_swa_averaging(
    model: MultimodalCRISPR,
    stage2_ckpt_paths: List[Path],
    train_loader: DataLoader,
    device: torch.device,
) -> AveragedModel:
    """Average weights over Stage 2 checkpoints and re-run normalization statistics."""
    log.info("=" * 75)
    log.info("[SWA] Starting Stochastic Weight Averaging over %d Stage-2 checkpoints...", len(stage2_ckpt_paths))

    swa_model = AveragedModel(model).to(device)

    for i, ckpt_path in enumerate(stage2_ckpt_paths):
        log.info("[SWA] Loading checkpoint %d/%d: %s", i + 1, len(stage2_ckpt_paths), ckpt_path.name)
        ckpt = torch.load(ckpt_path)
        temp_model = build_model(load_dnabert=True, freeze_bert=False, device=str(device))
        temp_model.load_state_dict(ckpt["model_state"])
        swa_model.update_parameters(temp_model)

    log.info("[SWA] Updating LayerNorm/BatchNorm statistics with pass over training set...")
    swa_model.train()
    with torch.no_grad():
        for batch in train_loader:
            seq_ids     = batch["seq_ids"].to(device)
            seq_mask    = batch["seq_mask"].to(device)
            epigen_vec  = batch["epigen_vec"].to(device)
            genome_vec  = batch["genome_vec"].to(device)
            sc_expr_vec = batch["sc_expr_vec"].to(device)
            _ = swa_model(
                input_ids=seq_ids,
                attention_mask=seq_mask,
                epigen_vec=epigen_vec,
                genome_vec=genome_vec,
                sc_expr_vec=sc_expr_vec,
            )

    log.info("[SWA] Statistics update complete.")
    return swa_model


# ============================================================================
# Main Training Loop
# ============================================================================

def train(
    smoke_test: bool = False,
    epochs_override: Optional[int] = None,
    batch_size: int = 2,
    accumulation_steps: int = 16,
    patience: int = 3,
    resume: bool = False,
    enable_swa: bool = False,
    sample: Optional[int] = None,
    seq_len: Optional[int] = 256,
    num_workers: int = 0,
    overwrite_checkpoints: bool = False,
    device_name: str = "cuda",
) -> None:
    if batch_size < 1 or accumulation_steps < 1 or num_workers < 0:
        raise ValueError("batch_size and accumulation_steps must be positive; num_workers cannot be negative")
    if seq_len is not None and not 8 <= seq_len <= 512:
        raise ValueError("seq_len must be between 8 and 512")
    log.info("=" * 75)
    log.info("Starting Multimodal CRISPR Off-Target Predictor 3-Stage Training")
    log.info("=" * 75)

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Check Docker --gpus all and nvidia-smi; CPU fallback is not automatic.")
    device = torch.device(device_name)
    log.info("Target Training Device: %s", device)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    existing_checkpoints = [p for p in [BEST_MODEL_PATH, *CHECKPOINT_DIR.glob("*.pt")] if p.is_file()]
    if existing_checkpoints and not resume:
        if not overwrite_checkpoints:
            raise FileExistsError(
                f"Existing checkpoints found ({existing_checkpoints[0]}). Use --resume to continue or "
                "--overwrite-checkpoints to back it up and start a new run."
            )
        backup_dir = CHECKPOINT_DIR / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path in [BEST_MODEL_PATH, *CHECKPOINT_DIR.glob("*.pt")]:
            if path.is_file():
                shutil.copy2(path, backup_dir / path.name)
        log.warning("[CHECKPOINT] Existing checkpoints backed up to %s", backup_dir)

    tb_writer = SummaryWriter(log_dir=str(TB_DIR))

    # 1. Load Datasets
    train_pt = TENSOR_DIR / "train.pt"
    val_pt   = TENSOR_DIR / "val.pt"
    test_pt  = TENSOR_DIR / "test.pt"

    if not train_pt.exists():
        log.error("Preprocessed tensors not found at %s. Run utils/preprocess.py first.", TENSOR_DIR)
        sys.exit(1)

    train_ds = MultimodalCRISPRDataset(train_pt)
    val_ds   = MultimodalCRISPRDataset(val_pt)
    test_ds  = MultimodalCRISPRDataset(test_pt)

    if smoke_test:
        train_ds.data = train_ds.data[:64]
        val_ds.data   = val_ds.data[:32]
        test_ds.data  = test_ds.data[:32]
        for ds in [train_ds, val_ds, test_ds]:
            for item in ds.data:
                item["seq_ids"]  = item["seq_ids"][:64]
                item["seq_mask"] = item["seq_mask"][:64]
        log.info("[SMOKE TEST] Subsampled datasets to train=%d, val=%d, test=%d (seq_len=64)",
                 len(train_ds), len(val_ds), len(test_ds))
    elif sample is not None:
        train_ds.data = train_ds.data[:sample]
        val_ds.data   = val_ds.data[:max(32, sample // 10)]
        test_ds.data  = test_ds.data[:max(32, sample // 10)]
        log.info("[SAMPLE] Subsampled datasets to train=%d, val=%d, test=%d",
                 len(train_ds), len(val_ds), len(test_ds))

    if not smoke_test and seq_len is not None:
        for ds in [train_ds, val_ds, test_ds]:
            for item in ds.data:
                item["seq_ids"]  = item["seq_ids"][:seq_len]
                item["seq_mask"] = item["seq_mask"][:seq_len]
        log.info("[SEQ_LEN] Truncated sequence tokens to seq_len=%d", seq_len)

    loader_options = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_options)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_options)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_options)

    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    log.info("DataLoader ready: train_samples=%d, batches=%d, steps_per_epoch=%d",
             len(train_ds), len(train_loader), steps_per_epoch)

    # 2. Build Model & Loss
    model = build_model(load_dnabert=True, freeze_bert=True, device=str(device))
    loss_fn = MultimodalLoss(epoch_steps=steps_per_epoch, pos_weight=3.0).to(device)

    # 3. Define Stage Schedule
    stage_schedule = [
        (0, 2, 0.0,  1e-3),   # Stage 0: Epochs 1-2
        (1, 4, 2e-5, 5e-4),   # Stage 1: Epochs 3-6
        (2, 9, 3e-5, 3e-4),   # Stage 2: Epochs 7-15
    ]

    if smoke_test:
        stage_schedule = [
            (0, 1, 0.0,  1e-3),
            (1, 1, 2e-5, 5e-4),
            (2, 1, 3e-5, 3e-4),
        ]
    elif epochs_override is not None:
        e = epochs_override
        s0 = max(1, int(e * 0.15))
        s1 = max(1, int(e * 0.30))
        s2 = max(1, e - s0 - s1)
        stage_schedule = [
            (0, s0, 0.0,  1e-3),
            (1, s1, 2e-5, 5e-4),
            (2, s2, 3e-5, 3e-4),
        ]

    total_epochs = sum(s[1] for s in stage_schedule)
    log.info("Training Schedule: %d total epochs across %d stages.", total_epochs, len(stage_schedule))

    best_val_loss = float("inf")
    patience_counter = 0
    current_epoch = 1
    stage2_checkpoints: List[Path] = []
    resume_payload = None

    # Handle Resumability
    latest_path = CHECKPOINT_DIR / "last_checkpoint.pt"
    if resume:
        checkpoint_path = latest_path if latest_path.exists() else BEST_MODEL_PATH
        if not checkpoint_path.exists():
            raise FileNotFoundError("--resume requested but no last or best checkpoint exists.")
        log.info("[RESUME] Loading checkpoint from %s ...", checkpoint_path)
        resume_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model_state"])
        best_val_loss = resume_payload.get("best_val_loss", best_val_loss)
        current_epoch = resume_payload.get("epoch", 0) + 1
        patience_counter = resume_payload.get("patience_counter", 0)
        if "loss_state" in resume_payload:
            loss_fn.load_state_dict(resume_payload["loss_state"])
        if "scaler_state" in resume_payload:
            scaler.load_state_dict(resume_payload["scaler_state"])
        log.info("[RESUME] Next epoch %d (best val loss %.4f)", current_epoch, best_val_loss)

    # Loop over stages
    stage_start_epoch = 1
    for stage_id, stage_epochs, base_backbone_lr, head_lr in stage_schedule:
        stage_end_epoch = stage_start_epoch + stage_epochs - 1

        if current_epoch > stage_end_epoch:
            log.info("[STAGE %d] Skipped (already completed up to epoch %d).", stage_id, stage_end_epoch)
            stage_start_epoch = stage_end_epoch + 1
            continue

        log.info("-" * 75)
        log.info("STAGE %d START | Epochs: %d..%d (%d epochs) | Backbone LR: %e | Head LR: %e",
                 stage_id, current_epoch, stage_end_epoch, stage_epochs, base_backbone_lr, head_lr)
        log.info("-" * 75)

        param_groups = build_parameter_groups(
            model=model,
            stage=stage_id,
            base_backbone_lr=base_backbone_lr,
            head_lr=head_lr,
            weight_decay=1e-2,
            gamma=0.9,
        )

        optimizer = torch.optim.AdamW(param_groups)

        total_stage_steps = steps_per_epoch * stage_epochs
        warmup_steps = max(1, int(0.10 * total_stage_steps))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_stage_steps,
        )

        if resume_payload is not None and resume_payload.get("stage") == stage_id:
            if "optimizer_state" in resume_payload:
                optimizer.load_state_dict(resume_payload["optimizer_state"])
            if "scheduler_state" in resume_payload:
                scheduler.load_state_dict(resume_payload["scheduler_state"])

        log.info("Optimizer initialized: %d parameter groups. Stage steps: %d (warmup: %d)",
                 len(param_groups), total_stage_steps, warmup_steps)

        for epoch_in_stage in range(1, stage_epochs + 1):
            model.train()
            optimizer.zero_grad()

            running_loss = 0.0
            running_mse = 0.0
            running_bce = 0.0
            accum_samples = 0
            t0 = time.time()

            for step_idx, batch in enumerate(train_loader):
                seq_ids     = batch["seq_ids"].to(device)
                seq_mask    = batch["seq_mask"].to(device)
                epigen_vec  = batch["epigen_vec"].to(device)
                genome_vec  = batch["genome_vec"].to(device)
                sc_expr_vec = batch["sc_expr_vec"].to(device)
                reg_label   = batch["reg_label"].to(device)
                cls_label   = batch["cls_label"].to(device)

                autocast_device = "cuda" if device.type == "cuda" else "cpu"

                with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                    reg_out, cls_out, _ = model(
                        input_ids=seq_ids,
                        attention_mask=seq_mask,
                        epigen_vec=epigen_vec,
                        genome_vec=genome_vec,
                        sc_expr_vec=sc_expr_vec,
                    )
                    loss, metrics = loss_fn(reg_out, cls_out, reg_label, cls_label, step_increment=0)
                    group_start = (step_idx // accumulation_steps) * accumulation_steps
                    group_size = min(accumulation_steps, len(train_loader) - group_start)
                    loss = loss / group_size

                scaler.scale(loss).backward()

                batch_size_actual = seq_ids.size(0)
                running_loss += metrics["loss"] * batch_size_actual
                running_mse  += metrics["mse_loss"] * batch_size_actual
                running_bce  += metrics["bce_loss"] * batch_size_actual
                accum_samples += batch_size_actual

                if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    loss_fn.step.add_(1)

            elapsed = time.time() - t0
            train_loss = running_loss / max(1, accum_samples)
            train_mse  = running_mse / max(1, accum_samples)
            train_bce  = running_bce / max(1, accum_samples)

            val_metrics = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp)

            current_lr = float(optimizer.param_groups[0]["lr"])

            log.info(
                "Epoch %02d/%02d (Stage %d) [%.1fs] | Train Loss: %.4f (MSE: %.4f, BCE: %.4f) | "
                "Val Loss: %.4f | RMSE: %.4f | AUROC: %.4f | PR-AUC: %.4f | F1: %.4f",
                current_epoch, total_epochs, stage_id, elapsed,
                train_loss, train_mse, train_bce,
                val_metrics["val_loss"], val_metrics["rmse"],
                val_metrics["roc_auc"], val_metrics["pr_auc"], val_metrics["f1"],
            )

            # TensorBoard Logging
            tb_writer.add_scalar("Loss/train_total", train_loss, current_epoch)
            tb_writer.add_scalar("Loss/val_total", val_metrics["val_loss"], current_epoch)
            tb_writer.add_scalar("Loss_Components/train_mse", train_mse, current_epoch)
            tb_writer.add_scalar("Loss_Components/train_bce", train_bce, current_epoch)
            tb_writer.add_scalar("Loss_Components/val_mse", val_metrics["val_mse"], current_epoch)
            tb_writer.add_scalar("Loss_Components/val_bce", val_metrics["val_bce"], current_epoch)
            tb_writer.add_scalar("Metrics/val_rmse", val_metrics["rmse"], current_epoch)
            tb_writer.add_scalar("Metrics/val_auroc", val_metrics["roc_auc"], current_epoch)
            tb_writer.add_scalar("Metrics/val_pr_auc", val_metrics["pr_auc"], current_epoch)
            tb_writer.add_scalar("Metrics/val_f1", val_metrics["f1"], current_epoch)
            tb_writer.add_scalar("Hyperparameters/learning_rate", current_lr, current_epoch)
            tb_writer.add_scalar("Hyperparameters/stage", stage_id, current_epoch)

            # Parameter Histograms for trainable heads
            for p_name, param in model.named_parameters():
                if param.requires_grad and "dnabert" not in p_name:
                    clean_name = p_name.replace(".", "/")
                    tb_writer.add_histogram(f"Parameters/{clean_name}", param.data, current_epoch)

            # CSV Logging
            log_metrics_to_csv(CSV_METRICS_PATH, {
                "epoch": current_epoch,
                "stage": stage_id,
                "train_loss": round(train_loss, 6),
                "train_mse": round(train_mse, 6),
                "train_bce": round(train_bce, 6),
                "val_loss": round(val_metrics["val_loss"], 6),
                "val_mse": round(val_metrics["val_mse"], 6),
                "val_bce": round(val_metrics["val_bce"], 6),
                "val_rmse": round(val_metrics["rmse"], 6),
                "val_pearson_r": round(val_metrics["pearson_r"], 6),
                "val_auroc": round(val_metrics["roc_auc"], 6),
                "val_pr_auc": round(val_metrics["pr_auc"], 6),
                "val_f1": round(val_metrics["f1"], 6),
                "learning_rate": current_lr,
            })

            # Keep epoch snapshots only when optional SWA needs them.
            if enable_swa and stage_id == 2:
                epoch_ckpt_path = CHECKPOINT_DIR / f"epoch_{current_epoch:02d}.pt"
                _atomic_torch_save({
                    "epoch": current_epoch,
                    "stage": stage_id,
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                }, epoch_ckpt_path)
                stage2_checkpoints.append(epoch_ckpt_path)

            # Early Stopping Check
            val_loss = val_metrics["val_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save Resumable Best Model Checkpoint
                resumable_ckpt = {
                    "epoch": current_epoch,
                    "stage": stage_id,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "patience_counter": patience_counter,
                    "loss_state": loss_fn.state_dict(),
                    "scaler_state": scaler.state_dict(),
                }
                _atomic_torch_save(resumable_ckpt, BEST_MODEL_PATH)
                log.info("  --> Saved resumable best model (Val Loss: %.4f) to %s", best_val_loss, BEST_MODEL_PATH)
            else:
                patience_counter += 1
                log.info("  [EARLY STOPPING] Patience counter: %d/%d (Best Val Loss: %.4f)",
                         patience_counter, patience, best_val_loss)

            last_ckpt = {
                "epoch": current_epoch,
                "stage": stage_id,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "val_metrics": val_metrics,
                "loss_state": loss_fn.state_dict(),
                "scaler_state": scaler.state_dict(),
            }
            _atomic_torch_save(last_ckpt, latest_path)
            current_epoch += 1

            if patience_counter >= patience:
                log.warning("[EARLY STOPPING] Triggered after %d epochs without val loss improvement.", patience)
                break

        stage_start_epoch = stage_end_epoch + 1
        if patience_counter >= patience:
            break

    # Save final model checkpoint
    final_ckpt = CHECKPOINT_DIR / "final_model.pt"
    _atomic_torch_save({"epoch": current_epoch - 1, "model_state": model.state_dict()}, final_ckpt)
    log.info("Saved final model checkpoint to %s", final_ckpt)

    # Run SWA over last 3 Stage 2 Checkpoints if enabled
    if enable_swa and len(stage2_checkpoints) >= 1:
        log.info("Running Stage-2 Stochastic Weight Averaging (SWA)...")
        swa_model = run_swa_averaging(
            model=model,
            stage2_ckpt_paths=stage2_checkpoints[-3:],
            train_loader=train_loader,
            device=device,
        )
        swa_ckpt_path = CHECKPOINT_DIR / "swa_model.pt"
        _atomic_torch_save({"model_state": swa_model.module.state_dict()}, swa_ckpt_path)
        log.info("Saved SWA model to %s", swa_ckpt_path)

        # Evaluate SWA model on Test Set
        log.info("=" * 75)
        log.info("Evaluating SWA Model on Test Set...")
        swa_test_metrics = evaluate(swa_model, test_loader, loss_fn, device, use_amp=use_amp)
        log.info("SWA TEST RESULTS:")
        log.info("  Test Loss    : %.4f", swa_test_metrics["val_loss"])
        log.info("  Test RMSE    : %.4f", swa_test_metrics["rmse"])
        log.info("  Pearson R    : %.4f", swa_test_metrics["pearson_r"])
        log.info("  ROC-AUC      : %.4f", swa_test_metrics["roc_auc"])
        log.info("  PR-AUC       : %.4f", swa_test_metrics["pr_auc"])
        log.info("  F1 Score     : %.4f", swa_test_metrics["f1"])

    # Final Evaluation of Best Model on Test Set
    log.info("=" * 75)
    log.info("Evaluating Best Model (model_best.pt) on Test Set...")
    if BEST_MODEL_PATH.exists():
        ckpt_data = torch.load(BEST_MODEL_PATH)
        model.load_state_dict(ckpt_data["model_state"])

    test_metrics = evaluate(model, test_loader, loss_fn, device, use_amp=use_amp)
    log.info("BEST MODEL TEST RESULTS:")
    log.info("  Test Loss    : %.4f", test_metrics["val_loss"])
    log.info("  Test RMSE    : %.4f", test_metrics["rmse"])
    log.info("  Pearson R    : %.4f", test_metrics["pearson_r"])
    log.info("  ROC-AUC      : %.4f", test_metrics["roc_auc"])
    log.info("  PR-AUC       : %.4f", test_metrics["pr_auc"])
    log.info("  F1 Score     : %.4f", test_metrics["f1"])
    log.info("=" * 75)

    # TensorBoard Training Summary
    train_summary_md = f"""### Multimodal CRISPR Training Complete
- **Final Epoch Completed**: {current_epoch - 1}
- **Device**: `{device}`
- **Best Model Checkpoint**: `{BEST_MODEL_PATH.name}`

| Model Evaluated | Test Loss | Test RMSE | Pearson r | Test AUROC | Test AUPRC | Test F1 |
|---|---|---|---|---|---|---|
| **Best Model** | {test_metrics['val_loss']:.4f} | {test_metrics['rmse']:.4f} | {test_metrics['pearson_r']:.4f} | {test_metrics['roc_auc']:.4f} | {test_metrics['pr_auc']:.4f} | {test_metrics['f1']:.4f} |
"""
    if enable_swa and 'swa_test_metrics' in locals():
        train_summary_md += f"| **SWA Model** | {swa_test_metrics['val_loss']:.4f} | {swa_test_metrics['rmse']:.4f} | {swa_test_metrics['pearson_r']:.4f} | {swa_test_metrics['roc_auc']:.4f} | {swa_test_metrics['pr_auc']:.4f} | {swa_test_metrics['f1']:.4f} |\n"

    tb_writer.add_text("Summary/Training_Report", train_summary_md)
    tb_writer.flush()
    tb_writer.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal CRISPR Predictor.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 3-stage smoke test.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size (RTX 4050 friendly default).")
    parser.add_argument("--accum-steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience epochs.")
    parser.add_argument("--resume", action="store_true", help="Resume from outputs/model_best.pt.")
    parser.add_argument("--swa", action="store_true", help="Enable optional SWA averaging (uses additional VRAM).")
    parser.add_argument("--sample", type=int, default=None, help="Train on first N examples.")
    parser.add_argument("--seq-len", type=int, default=256, help="Truncate sequence tokens (max 512).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 is robust in Docker/Windows mounts.")
    parser.add_argument("--overwrite-checkpoints", action="store_true", help="Back up existing checkpoints before a fresh run.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Training device; default requires CUDA.")
    args = parser.parse_args()

    train(
        smoke_test=args.smoke_test,
        epochs_override=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accum_steps,
        patience=args.patience,
        resume=args.resume,
        enable_swa=args.swa,
        sample=args.sample,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        overwrite_checkpoints=args.overwrite_checkpoints,
        device_name=args.device,
    )
