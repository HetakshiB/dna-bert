"""Multimodal CRISPR off-target predictor.

Architecture
============
Four branch encoders process each modality independently, their outputs are
concatenated into a 576-d joint representation, passed through a shared fusion
MLP, then split into two parallel output heads (regression + classification).

                 ┌─────────────────┐
  DNA seq ───────► SeqHead         │ 256-d ─┐
                 └─────────────────┘        │
                 ┌─────────────────┐        │
  6 epigen ──────► EpigenHead      │  64-d ─┤
                 └─────────────────┘        │
                 ┌─────────────────┐        ├──► [576-d] ──► FusionMLP ──► 256-d
  8 genome ──────► GenomeHead      │ 128-d ─┤                              │    │
                 └─────────────────┘        │                         reg_head cls_head
                 ┌─────────────────┐        │
  sc_expr ───────► ScHead          │ 128-d ─┘
                 └─────────────────┘

Branch details
--------------
  SeqHead    : Linear(768→256), ReLU, Dropout(0.1)
  EpigenHead : Linear(6→128), ReLU, Dropout(0.1), Linear(128→64)
  GenomeHead : Linear(in→128), ReLU, Dropout(0.1), Linear(128→128)
               ``in`` defaults to 8 (actual data dim); spec reference value is 3.
  ScHead     : Linear(num_sc→128), ReLU, Dropout(0.1), Linear(128→128)

Fusion       : Linear(576→256), ReLU, Dropout(0.2), Linear(256→256)

Output heads
------------
  reg_head   : Linear(256→1)  — continuous cleavage_freq  (use MSE / Huber loss)
  cls_head   : Linear(256→1)  — binary high_eff logit      (use BCEWithLogitsLoss)

Both the scalar outputs AND the 256-d fused representation are returned so that
downstream interpretability work (SHAP, attention rollout, probing classifiers)
can operate on the latent space without re-running the backbone.

Usage
-----
    python model/multimodal.py          # smoke test
    python model/multimodal.py --batch 4 --seq-len 512

Forward signature
-----------------
    reg_out, cls_out, fused = model(
        input_ids, attention_mask, epigen_vec, genome_vec, sc_expr_vec
    )
    # reg_out : [B, 1]   float
    # cls_out : [B, 1]   float (logit; apply sigmoid for probability)
    # fused   : [B, 256] float (post-fusion representation)
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ============================================================================
# Individual branch encoders
# ============================================================================

class SeqHead(nn.Module):
    """Branch 1 — sequence encoder on top of the DNABERT CLS token.

    Input  : [B, 768]   CLS token from DNABERT
    Output : [B, 256]
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:  # [B,768] → [B,256]
        return self.net(cls_emb)


class EpigenHead(nn.Module):
    """Branch 2 — epigenetic signal encoder.

    Input  : [B, 6]    z-scored 6-mark ENCODE signal
    Output : [B, 64]
    """

    def __init__(self, in_features: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,6] → [B,64]
        return self.net(x)


class GenomeHead(nn.Module):
    """Branch 3 — 3D genome feature encoder.

    Spec reference input dim: 3.
    Actual data dim          : 8  (loop_count + tad_distance + 6-way compartment OH).

    The default is 8 to match the preprocessed tensors; pass ``in_features=3``
    to use only the three scalar features (loop_count, tad_distance, compartment_id).

    Input  : [B, in_features]
    Output : [B, 128]
    """

    def __init__(self, in_features: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,in] → [B,128]
        return self.net(x)


class ScHead(nn.Module):
    """Branch 4 — single-cell expression encoder.

    Input  : [B, num_sc_features]   log-normalised mean expression per locus
    Output : [B, 128]
    """

    def __init__(self, num_sc_features: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_sc_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,sc] → [B,128]
        return self.net(x)


# ============================================================================
# Fusion block
# ============================================================================

class FusionMLP(nn.Module):
    """Shared fusion block operating on the 576-d concatenated representation.

    576 = 256 (seq) + 64 (epigen) + 128 (genome) + 128 (sc)

    Input  : [B, 576]
    Output : [B, 256]
    """

    def __init__(self, in_features: int = 576, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,576] → [B,256]
        return self.net(x)


# ============================================================================
# Top-level multimodal model
# ============================================================================

class MultimodalCRISPR(nn.Module):
    """Full multimodal CRISPR off-target predictor.

    Parameters
    ----------
    dnabert_model:
        A frozen (or partially frozen) ``transformers.BertModel`` from
        :func:`model.backbone.get_dnabert`.  If ``None`` the forward pass
        expects pre-computed CLS embeddings of shape [B, 768] via the
        ``cls_emb`` argument.  Pass a model instance for end-to-end training.
    num_genome_features:
        Input dimension of the 3D-genome head (default 8; spec reference 3).
    num_sc_features:
        Input dimension of the single-cell head (default 1).
    seq_dropout / epigen_dropout / genome_dropout / sc_dropout:
        Per-branch dropout rates (default 0.1 each).
    fusion_dropout:
        Dropout inside the fusion MLP (default 0.2).
    """

    SEQ_OUT    = 256
    EPIGEN_OUT = 64
    GENOME_OUT = 128
    SC_OUT     = 128
    FUSION_IN  = SEQ_OUT + EPIGEN_OUT + GENOME_OUT + SC_OUT  # = 576

    def __init__(
        self,
        dnabert_model=None,
        num_genome_features: int = 8,
        num_sc_features: int = 1,
        seq_dropout: float = 0.1,
        epigen_dropout: float = 0.1,
        genome_dropout: float = 0.1,
        sc_dropout: float = 0.1,
        fusion_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Optional BERT backbone (None → expects pre-computed CLS embeddings)
        self.dnabert = dnabert_model   # may be None

        # Branch encoders
        self.seq_head    = SeqHead(dropout=seq_dropout)
        self.epigen_head = EpigenHead(dropout=epigen_dropout)
        self.genome_head = GenomeHead(in_features=num_genome_features, dropout=genome_dropout)
        self.sc_head     = ScHead(num_sc_features=num_sc_features, dropout=sc_dropout)

        # Fusion MLP
        self.fusion = FusionMLP(in_features=self.FUSION_IN, dropout=fusion_dropout)

        # Parallel output heads
        self.reg_head = nn.Linear(256, 1)   # regression: cleavage_freq
        self.cls_head = nn.Linear(256, 1)   # classification: high_eff logit

        self._log_param_counts()

    # ------------------------------------------------------------------ #
    def _log_param_counts(self) -> None:
        def _count(m: nn.Module) -> tuple[int, int]:
            total     = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        bert_total = 0
        if self.dnabert is not None:
            bt, btr = _count(self.dnabert)
            bert_total = bt
            log.info("[MODEL] DNABERT backbone : %dM total / %dM trainable", bt//1_000_000, btr//1_000_000)

        head_total, head_train = _count(nn.ModuleList([
            self.seq_head, self.epigen_head, self.genome_head,
            self.sc_head, self.fusion, self.reg_head, self.cls_head
        ]))
        log.info("[MODEL] Head parameters   : %d total / %d trainable", head_total, head_train)
        log.info("[MODEL] Grand total       : %dM", (bert_total + head_total) // 1_000_000)

    # ------------------------------------------------------------------ #
    def _get_cls_emb(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return [B, 768] CLS embedding from backbone or pre-computed tensor."""
        if cls_emb is not None:
            return cls_emb
        if self.dnabert is None:
            raise ValueError(
                "Neither `dnabert_model` nor `cls_emb` was provided. "
                "Pass the DNABERT model to __init__ or supply pre-computed CLS embeddings."
            )
        if not any(p.requires_grad for p in self.dnabert.parameters()):
            with torch.no_grad():
                out = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # [B, 768]

    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids:     torch.Tensor,          # [B, seq_len]   LongTensor
        attention_mask: torch.Tensor,          # [B, seq_len]   LongTensor
        epigen_vec:    torch.Tensor,           # [B, 6]         FloatTensor
        genome_vec:    torch.Tensor,           # [B, 8]         FloatTensor
        sc_expr_vec:   torch.Tensor,           # [B, num_sc]    FloatTensor
        cls_emb:       torch.Tensor | None = None,  # [B, 768] pre-computed optional
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all four branches, fusion, and output heads.

        Parameters
        ----------
        input_ids, attention_mask:
            DNABERT tokeniser outputs; ignored if ``cls_emb`` is provided.
        epigen_vec:
            z-scored 6-mark epigenetic signal vector.
        genome_vec:
            3D genome feature vector (loop_count, tad_distance, compartment OH).
        sc_expr_vec:
            Log-normalised mean expression of nearest gene.
        cls_emb:
            Optional pre-computed CLS token embedding [B, 768].  When provided,
            DNABERT is bypassed (faster training when backbone is fully frozen).

        Returns
        -------
        reg_out : [B, 1]   — cleavage_freq logit (no activation)
        cls_out : [B, 1]   — high_eff logit (apply sigmoid for probability)
        fused   : [B, 256] — post-fusion representation (for interpretability)
        """
        # Branch 1 — sequence
        cls = self._get_cls_emb(input_ids, attention_mask, cls_emb)  # [B,768]
        h_seq = self.seq_head(cls)                                     # [B,256]

        # Branch 2 — epigenetic
        h_epigen = self.epigen_head(epigen_vec)                        # [B,64]

        # Branch 3 — 3D genome
        h_genome = self.genome_head(genome_vec)                        # [B,128]

        # Branch 4 — single cell
        h_sc = self.sc_head(sc_expr_vec)                               # [B,128]

        # Concatenate → 576-d joint representation
        joint = torch.cat([h_seq, h_epigen, h_genome, h_sc], dim=-1)  # [B,576]

        # Fusion MLP
        fused = self.fusion(joint)                                     # [B,256]

        # Output heads
        reg_out = self.reg_head(fused)                                 # [B,1]
        cls_out = self.cls_head(fused)                                 # [B,1]

        return reg_out, cls_out, fused


# ============================================================================
# Convenience factory
# ============================================================================

def build_model(
    load_dnabert: bool = True,
    freeze_bert: bool = True,
    num_genome_features: int = 8,
    num_sc_features: int = 1,
    device: str | None = None,
) -> MultimodalCRISPR:
    """Instantiate and return a ready-to-use MultimodalCRISPR model.

    Parameters
    ----------
    load_dnabert:
        If True, load the DNABERT backbone from local cache.
    freeze_bert:
        If True, freeze all BERT parameters.
    device:
        Target device string ('cuda' / 'cpu'). Auto-detected if None.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bert_model = None
    if load_dnabert:
        from model.backbone import get_dnabert
        _, bert_model = get_dnabert(freeze=freeze_bert, device=device)

    model = MultimodalCRISPR(
        dnabert_model=bert_model,
        num_genome_features=num_genome_features,
        num_sc_features=num_sc_features,
    ).to(device)

    return model


# ============================================================================
# Smoke test
# ============================================================================

def _smoke_test(batch_size: int = 4, seq_len: int = 512) -> None:
    print()
    print("=" * 64)
    print("  MultimodalCRISPR — Smoke Test")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Mode A: with live DNABERT backbone ──────────────────────────
    print("\n  [Mode A] With DNABERT backbone ...")
    model_a = build_model(load_dnabert=True, freeze_bert=True, device=str(device))
    model_a.eval()

    input_ids     = torch.randint(0, 4096, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    epigen_vec    = torch.randn(batch_size, 6, device=device)
    genome_vec    = torch.randn(batch_size, 8, device=device)
    sc_expr_vec   = torch.randn(batch_size, 1, device=device)

    with torch.no_grad():
        reg_out, cls_out, fused = model_a(
            input_ids, attention_mask, epigen_vec, genome_vec, sc_expr_vec
        )

    print(f"  reg_out shape : {tuple(reg_out.shape)}  (expected [{batch_size}, 1])")
    print(f"  cls_out shape : {tuple(cls_out.shape)}  (expected [{batch_size}, 1])")
    print(f"  fused   shape : {tuple(fused.shape)}  (expected [{batch_size}, 256])")

    assert list(reg_out.shape) == [batch_size, 1],  f"reg_out shape mismatch: {reg_out.shape}"
    assert list(cls_out.shape) == [batch_size, 1],  f"cls_out shape mismatch: {cls_out.shape}"
    assert list(fused.shape)   == [batch_size, 256], f"fused shape mismatch: {fused.shape}"
    print("  [Mode A] PASS")

    # ── Mode B: with pre-computed CLS embeddings (backbone bypass) ──
    print("\n  [Mode B] Pre-computed CLS embeddings (bypass BERT) ...")
    model_b = MultimodalCRISPR(dnabert_model=None).to(device)
    model_b.eval()

    cls_emb_fake = torch.randn(batch_size, 768, device=device)
    dummy_ids    = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    dummy_mask   = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        reg_b, cls_b, fused_b = model_b(
            dummy_ids, dummy_mask, epigen_vec, genome_vec, sc_expr_vec,
            cls_emb=cls_emb_fake
        )

    assert list(reg_b.shape)   == [batch_size, 1]
    assert list(cls_b.shape)   == [batch_size, 1]
    assert list(fused_b.shape) == [batch_size, 256]
    print("  [Mode B] PASS")

    # ── Gradient flow check ─────────────────────────────────────────
    print("\n  [Grad check] Verifying gradient flow through heads ...")
    model_b.train()
    reg_b2, cls_b2, fused_b2 = model_b(
        dummy_ids, dummy_mask, epigen_vec, genome_vec, sc_expr_vec,
        cls_emb=cls_emb_fake
    )
    loss = reg_b2.mean() + cls_b2.mean()
    loss.backward()

    for name, param in model_b.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for: {name}"
    print("  [Grad check] PASS — all head parameters received gradients")

    # ── Architecture summary ────────────────────────────────────────
    print()
    print("  Branch output dims:")
    print(f"    SeqHead    : 256")
    print(f"    EpigenHead :  64")
    print(f"    GenomeHead : 128")
    print(f"    ScHead     : 128")
    print(f"    Concat     : {MultimodalCRISPR.FUSION_IN} (= 576)")
    print(f"    FusionMLP  : 256")
    print(f"    reg_head   :   1")
    print(f"    cls_head   :   1")
    print()
    print("=" * 64)
    print("  All smoke tests PASSED.")
    print("=" * 64)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultimodalCRISPR smoke test.")
    parser.add_argument("--batch",   type=int, default=4,   help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence token length.")
    args = parser.parse_args()
    _smoke_test(batch_size=args.batch, seq_len=args.seq_len)
