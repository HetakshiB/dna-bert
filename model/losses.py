"""Multimodal CRISPR Off-Target Predictor — Loss Functions & Dynamic Schedulers.

Implements the multi-task combined loss:
    L(t) = α(t) · MSE(reg_pred, reg_target) + (1 - α(t)) · Weighted_BCE(cls_pred, cls_target)

where:
  - MSE is the mean squared error for continuous cleavage frequency regression.
  - Weighted_BCE is binary cross-entropy with logits using pos_weight ≈ N_neg / N_pos
    to address class imbalance in high_eff labels (~3:1 ratio).
  - α(t) is a step-dependent weighting factor:
        α(t) = 0.7 + 0.2 · exp(-t / τ)
    with τ set to the number of steps in one epoch (epoch_steps).
    At t=0, α(0) = 0.9 (90% regression weight during early warm-up).
    As t → ∞, α(t) decay smoothly to 0.7.

Exposes ``MultimodalLoss`` which tracks execution steps, computes loss,
and returns scalar loss alongside component dictionary for TensorBoard/logging.
"""

from __future__ import annotations

import math
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


class MultimodalLoss(nn.Module):
    """Combined dynamic Multi-Task Loss with pos_weight and decay schedule.

    Parameters
    ----------
    epoch_steps : int
        Number of steps per epoch (τ in α(t) schedule). Must be > 0.
    pos_weight : float or torch.Tensor, optional
        Positive class weight N_neg / N_pos. Default is 3.0 (matching ~25% high_eff ratio).
    alpha_min : float, default=0.7
        Asymptotic minimum value for regression weight α(t).
    alpha_range : float, default=0.2
        Initial weight range boost (so α(0) = alpha_min + alpha_range = 0.9).
    """

    def __init__(
        self,
        epoch_steps: int,
        pos_weight: float | torch.Tensor = 3.0,
        alpha_min: float = 0.7,
        alpha_range: float = 0.2,
    ) -> None:
        super().__init__()
        if epoch_steps <= 0:
            raise ValueError(f"epoch_steps must be > 0, got {epoch_steps}")

        self.epoch_steps = float(epoch_steps)
        self.alpha_min = alpha_min
        self.alpha_range = alpha_range

        # Convert pos_weight to FloatTensor buffer for device compatibility
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32)
        else:
            pos_weight_tensor = pos_weight.float()
            if pos_weight_tensor.ndim == 0:
                pos_weight_tensor = pos_weight_tensor.unsqueeze(0)

        self.register_buffer("pos_weight", pos_weight_tensor)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def get_alpha(self, step: Optional[int | torch.Tensor] = None) -> float:
        """Compute α(t) = alpha_min + alpha_range * exp(-t / τ)."""
        t = float(self.step.item() if step is None else (step.item() if isinstance(step, torch.Tensor) else step))
        return self.alpha_min + self.alpha_range * math.exp(-t / self.epoch_steps)

    def reset_step(self, step: int = 0) -> None:
        """Reset step counter."""
        self.step.fill_(step)

    def forward(
        self,
        reg_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        reg_target: torch.Tensor,
        cls_target: torch.Tensor,
        step_increment: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task combined loss.

        Parameters
        ----------
        reg_pred : torch.Tensor [B, 1] or [B]
            Continuous regression predictions.
        cls_pred : torch.Tensor [B, 1] or [B]
            Binary classification logits.
        reg_target : torch.Tensor [B, 1] or [B]
            Target cleavage frequency values.
        cls_target : torch.Tensor [B, 1] or [B]
            Target binary high_eff labels (0 or 1).
        step_increment : int, default=1
            Amount to increment internal step counter by after computing loss.

        Returns
        -------
        total_loss : torch.Tensor (scalar)
            Differentiable total combined loss.
        metrics : Dict[str, float]
            Dictionary containing scalar values for logging:
            'loss', 'mse_loss', 'bce_loss', 'alpha', 'step'
        """
        # Ensure 1D/2D shape alignment
        reg_pred = reg_pred.view(-1)
        reg_target = reg_target.view(-1).float()
        cls_pred = cls_pred.view(-1)
        cls_target = cls_target.view(-1).float()

        # 1. Regression Loss (MSE)
        mse_loss = F.mse_loss(reg_pred, reg_target)

        # 2. Classification Loss (Weighted BCE with logits)
        bce_loss = F.binary_cross_entropy_with_logits(
            cls_pred, cls_target, pos_weight=self.pos_weight
        )

        # 3. Dynamic Alpha Schedule
        current_step = int(self.step.item())
        alpha_t = self.get_alpha(current_step)

        # 4. Total Combined Loss
        total_loss = alpha_t * mse_loss + (1.0 - alpha_t) * bce_loss

        # Step update
        if self.training and step_increment > 0:
            self.step.add_(step_increment)

        metrics = {
            "loss": float(total_loss.item()),
            "mse_loss": float(mse_loss.item()),
            "bce_loss": float(bce_loss.item()),
            "alpha": float(alpha_t),
            "step": current_step,
        }

        return total_loss, metrics


# ============================================================================
# Smoke Test
# ============================================================================

def _smoke_test() -> None:
    print("=" * 60)
    print("  MultimodalLoss Smoke Test")
    print("=" * 60)

    epoch_steps = 100
    pos_weight = 3.0
    loss_fn = MultimodalLoss(epoch_steps=epoch_steps, pos_weight=pos_weight)

    # 1. Check Initial Alpha
    alpha_0 = loss_fn.get_alpha(0)
    print(f"  [SMOKE] alpha(0)      : {alpha_0:.4f} (expected 0.9000)")
    assert math.isclose(alpha_0, 0.9, abs_tol=1e-4)

    # 2. Check Alpha at t = τ (1 epoch)
    alpha_100 = loss_fn.get_alpha(100)
    expected_100 = 0.7 + 0.2 * math.exp(-1.0)
    print(f"  [SMOKE] alpha(100)    : {alpha_100:.4f} (expected {expected_100:.4f})")
    assert math.isclose(alpha_100, expected_100, abs_tol=1e-4)

    # 3. Check Alpha at t -> ∞
    alpha_inf = loss_fn.get_alpha(10000)
    print(f"  [SMOKE] alpha(10000)  : {alpha_inf:.4f} (expected 0.7000)")
    assert math.isclose(alpha_inf, 0.7, abs_tol=1e-4)

    # 4. Dummy forward pass
    B = 8
    reg_pred = torch.randn(B, 1, requires_grad=True)
    cls_pred = torch.randn(B, 1, requires_grad=True)
    reg_target = torch.rand(B, 1)
    cls_target = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32).unsqueeze(1)

    loss_fn.train()
    loss, metrics = loss_fn(reg_pred, cls_pred, reg_target, cls_target)

    print(f"  [SMOKE] Total Loss  : {metrics['loss']:.4f}")
    print(f"  [SMOKE] MSE Loss    : {metrics['mse_loss']:.4f}")
    print(f"  [SMOKE] BCE Loss    : {metrics['bce_loss']:.4f}")
    print(f"  [SMOKE] Alpha       : {metrics['alpha']:.4f}")
    print(f"  [SMOKE] Step        : {metrics['step']}")

    # Verification checks
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    loss.backward()
    assert reg_pred.grad is not None and cls_pred.grad is not None
    print("  [SMOKE] Gradients calculated successfully.")

    # Check step increment
    assert loss_fn.step.item() == 1
    print("  [SMOKE] Step counter auto-incremented to 1.")
    print("=" * 60)
    print("  All loss smoke tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
