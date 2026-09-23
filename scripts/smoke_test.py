"""Smoke test & VRAM budget verification script for Multimodal CRISPR Off-Target Predictor.

Functionality:
  1. Loads 1% of preprocessed training dataset (outputs/tensors/train.pt).
  2. Runs 200 steps of training using MultimodalCRISPR model and MultimodalLoss.
  3. Tracks loss reduction and asserts loss decreases (loss_end < loss_start).
  4. Wraps training execution with VRAM memory logging (via nvidia-smi / PyTorch CUDA memory).
  5. Asserts peak GPU memory stays under 7.0 GB budget.
  6. If peak memory exceeds 7.0 GB budget or CUDA OOM occurs, automatically reduces
     micro-batch size (32 → 16) and retries training once.
  7. On CPU-only environments (no NVIDIA GPU), logs VRAM usage as 0.00 GB and validates
     loss convergence and execution integrity cleanly.

Usage:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --steps 200 --batch-size 32
"""

from __future__ import annotations

import sys
import time
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENSOR_DIR = PROJECT_ROOT / "outputs" / "tensors"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
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
        logging.FileHandler(LOG_DIR / "smoke_test.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(PROJECT_ROOT))
from model.multimodal import build_model  # noqa: E402
from model.losses import MultimodalLoss    # noqa: E402
from train import MultimodalCRISPRDataset, build_parameter_groups  # noqa: E402


# ============================================================================
# GPU Memory Helper (nvidia-smi & PyTorch CUDA)
# ============================================================================

def get_peak_gpu_memory_mb() -> float:
    """Return peak GPU VRAM used in MB via nvidia-smi or PyTorch CUDA API.

    Returns 0.0 if running on CPU or nvidia-smi is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0

    # 1. Try PyTorch CUDA max memory allocated
    try:
        pytorch_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pytorch_mb = 0.0

    # 2. Try nvidia-smi command if available
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.run(
                [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            smi_mb = float(res.stdout.strip().split("\n")[0])
            return max(pytorch_mb, smi_mb)
        except Exception:
            pass

    return pytorch_mb


# ============================================================================
# 1% Data Training Step Engine
# ============================================================================

def run_smoke_test_run(
    batch_size: int = 32,
    num_steps: int = 200,
    seq_len: int = 512,
    vram_budget_mb: float = 7168.0,  # 7.0 GB in MB
    unfreeze_bert: bool = False,
) -> Tuple[bool, float, float, float]:
    """Execute smoke test training loop for num_steps on 1% dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_bert = not unfreeze_bert if device.type == "cpu" else not unfreeze_bert

    log.info("[SMOKE TEST] Device: %s | Micro-batch: %d | Steps: %d | VRAM Budget: %.2f GB | Freeze BERT: %s",
             device, batch_size, num_steps, vram_budget_mb / 1024.0, freeze_bert)

    train_pt = TENSOR_DIR / "train.pt"
    if not train_pt.exists():
        raise FileNotFoundError(f"Training tensors not found at {train_pt}. Run utils/preprocess.py first.")

    dataset = MultimodalCRISPRDataset(train_pt)
    total_len = len(dataset)
    subset_len = max(32, int(total_len * 0.01))  # 1% of dataset (~204 rows)

    dataset.data = dataset.data[:subset_len]

    if seq_len < 512:
        for item in dataset.data:
            item["seq_ids"]  = item["seq_ids"][:seq_len]
            item["seq_mask"] = item["seq_mask"][:seq_len]

    log.info("[SMOKE TEST] Loaded 1%% training subset: %d examples out of %d total.", len(dataset), total_len)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model = build_model(load_dnabert=True, freeze_bert=freeze_bert, device=str(device))
    model.train()

    # Pre-compute CLS embeddings if backbone is frozen for 600x CPU speedup
    if freeze_bert and model.dnabert is not None:
        log.info("[SMOKE TEST] Pre-computing CLS embeddings for 1%% subset (%d samples)...", len(dataset))
        model.eval()
        loader_pre = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_cls_list = []
        with torch.no_grad():
            for b in loader_pre:
                s_ids = b["seq_ids"].to(device)
                s_mask = b["seq_mask"].to(device)
                out = model.dnabert(input_ids=s_ids, attention_mask=s_mask)
                all_cls_list.append(out.last_hidden_state[:, 0, :].cpu())
        all_cls_tensor = torch.cat(all_cls_list, dim=0)  # [subset_len, 768]
        for idx in range(len(dataset.data)):
            dataset.data[idx]["cls_emb"] = all_cls_tensor[idx]
        model.train()
        log.info("[SMOKE TEST] Pre-computation complete. Starting fast training loop...")

    loss_fn = MultimodalLoss(epoch_steps=50, pos_weight=3.0).to(device)

    stage_id = 2 if unfreeze_bert else 0
    param_groups = build_parameter_groups(
        model=model,
        stage=stage_id,
        base_backbone_lr=3e-5 if unfreeze_bert else 0.0,
        head_lr=1e-3,
        weight_decay=1e-2,
        gamma=0.9,
    )

    optimizer = torch.optim.AdamW(param_groups)

    step_count = 0
    losses: List[float] = []
    peak_vram_mb = 0.0

    t0 = time.time()
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    try:
        while step_count < num_steps:
            for batch in train_loader:
                if step_count >= num_steps:
                    break

                optimizer.zero_grad()

                seq_ids     = batch["seq_ids"].to(device)
                seq_mask    = batch["seq_mask"].to(device)
                epigen_vec  = batch["epigen_vec"].to(device)
                genome_vec  = batch["genome_vec"].to(device)
                sc_expr_vec = batch["sc_expr_vec"].to(device)
                reg_label   = batch["reg_label"].to(device)
                cls_label   = batch["cls_label"].to(device)
                cls_emb     = batch.get("cls_emb")
                if cls_emb is not None:
                    cls_emb = cls_emb.to(device)

                autocast_device = "cuda" if device.type == "cuda" else "cpu"

                with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                    reg_out, cls_out, _ = model(
                        input_ids=seq_ids,
                        attention_mask=seq_mask,
                        epigen_vec=epigen_vec,
                        genome_vec=genome_vec,
                        sc_expr_vec=sc_expr_vec,
                        cls_emb=cls_emb,
                    )
                    loss, metrics = loss_fn(reg_out, cls_out, reg_label, cls_label, step_increment=1)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                step_count += 1
                losses.append(metrics["loss"])

                # Periodically monitor VRAM usage
                current_vram = get_peak_gpu_memory_mb()
                if current_vram > peak_vram_mb:
                    peak_vram_mb = current_vram

                if step_count % 20 == 0 or step_count == 1:
                    log.info("  Step %03d/%03d | Loss: %.4f (MSE: %.4f, BCE: %.4f) | VRAM Peak: %.2f GB",
                             step_count, num_steps, metrics["loss"], metrics["mse_loss"], metrics["bce_loss"],
                             peak_vram_mb / 1024.0)

    except torch.cuda.OutOfMemoryError:
        log.error("[OOM] CUDA Out Of Memory error detected during smoke test training!")
        return False, 0.0, 0.0, peak_vram_mb / 1024.0

    elapsed = time.time() - t0
    peak_vram_gb = peak_vram_mb / 1024.0
    loss_start = float(np.mean(losses[:10])) if len(losses) >= 10 else losses[0]
    loss_end = float(np.mean(losses[-10:])) if len(losses) >= 10 else losses[-1]

    log.info("[SMOKE TEST] Completed %d steps in %.1fs (%.2f steps/sec)", step_count, elapsed, step_count / elapsed)
    log.info("[SMOKE TEST] Initial Loss (First 10 steps avg): %.4f  |  Final Loss (Last 10 steps avg): %.4f",
             loss_start, loss_end)
    log.info("[SMOKE TEST] Peak VRAM: %.2f GB (Budget: %.2f GB)", peak_vram_gb, vram_budget_mb / 1024.0)

    # Check VRAM budget
    if peak_vram_mb > vram_budget_mb:
        log.warning("[MEMORY BUDGET EXCEEDED] Peak VRAM %.2f GB > 7.00 GB budget!", peak_vram_gb)
        return False, loss_start, loss_end, peak_vram_gb

    return True, loss_start, loss_end, peak_vram_gb


# ============================================================================
# Main Smoke Test Execution & Retry Engine
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1% dataset smoke test with VRAM monitoring.")
    parser.add_argument("--steps", type=int, default=200, help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Initial micro-batch size.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length.")
    parser.add_argument("--max-vram-gb", type=float, default=7.0, help="Maximum VRAM budget in GB.")
    parser.add_argument("--unfreeze-bert", action="store_true", help="Unfreeze BERT backbone during test.")
    args = parser.parse_args()

    vram_budget_mb = args.max_vram_gb * 1024.0

    log.info("=" * 75)
    log.info("Starting 1%% Data Smoke Test & Memory Budget Check")
    log.info("=" * 75)

    batch_size = args.batch_size
    success, loss_start, loss_end, peak_vram_gb = run_smoke_test_run(
        batch_size=batch_size,
        num_steps=args.steps,
        seq_len=args.seq_len,
        vram_budget_mb=vram_budget_mb,
        unfreeze_bert=args.unfreeze_bert,
    )

    # Automatic Retry with Halved Micro-Batch Size if VRAM exceeded or OOM occurred
    if not success and batch_size > 1:
        reduced_batch_size = max(1, batch_size // 2)
        log.warning("=" * 75)
        log.warning("[RETRY TRIGGERED] Reducing micro-batch size %d → %d and retrying smoke test once...",
                    batch_size, reduced_batch_size)
        log.warning("=" * 75)

        batch_size = reduced_batch_size
        success, loss_start, loss_end, peak_vram_gb = run_smoke_test_run(
            batch_size=batch_size,
            num_steps=args.steps,
            seq_len=args.seq_len,
            vram_budget_mb=vram_budget_mb,
            unfreeze_bert=args.unfreeze_bert,
        )

    # Final Assertions
    log.info("=" * 75)
    log.info("SMOKE TEST ASSERTIONS & VERIFICATION:")
    log.info("  1. Batch Size              : %d", batch_size)
    log.info("  2. Peak VRAM Usage         : %.2f GB (Budget: %.2f GB)", peak_vram_gb, args.max_vram_gb)
    log.info("  3. Loss Start (Step 1)     : %.4f", loss_start)
    log.info("  4. Loss End (Step %d avg) : %.4f", args.steps, loss_end)

    # Assert 1: Memory budget
    assert peak_vram_gb <= args.max_vram_gb, (
        f"ASSERTION FAILED: Peak VRAM ({peak_vram_gb:.2f} GB) exceeded budget ({args.max_vram_gb:.2f} GB) "
        f"even after retry with batch_size={batch_size}."
    )
    log.info("  [ASSERT 1 PASS] Peak VRAM is within %.2f GB budget.", args.max_vram_gb)

    # Assert 2: Loss reduction
    assert loss_end < loss_start, (
        f"ASSERTION FAILED: Training loss did not decrease! Loss start = {loss_start:.4f}, loss end = {loss_end:.4f}"
    )
    log.info("  [ASSERT 2 PASS] Training loss decreased successfully from %.4f to %.4f.", loss_start, loss_end)

    log.info("=" * 75)
    log.info("All smoke test assertions PASSED successfully!")
    log.info("=" * 75)


if __name__ == "__main__":
    main()
