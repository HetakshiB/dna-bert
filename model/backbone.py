"""DNABERT backbone loader for the Multimodal CRISPR Off-Target Predictor.

This module wraps HuggingFace transformers to provide a single
``get_dnabert(freeze)`` entry-point that:

1. Downloads / loads **DNABERT-6** (zhihan1996/DNA_6mer-bert-110M)
   and its matching 6-mer tokenizer.
2. Caches everything under  ``<project_root>/models/dnabert-base/``
   so subsequent runs are fully offline.
3. Optionally freezes all BERT weights for feature-extraction mode.

Architecture note
-----------------
The model is a standard BERT-base encoder:
  - 12 transformer layers, hidden dim = 768, 12 attention heads
  - Input: k-mer tokenised DNA string (k=6)
  - Output: last_hidden_state  shape [B, L, 768]
           CLS token           shape [B, 768]   ← used by the sequence branch

Freeze strategy
---------------
When ``freeze=True`` every parameter in the BertModel is set to
``requires_grad=False``.  Only the projection/classification heads added on
top of the backbone will be trained.  This is the recommended default during
Phase-1 warm-up (blueprint §7.1).

Smoke test
----------
Run directly to verify the download and a single forward pass:

    python model/backbone.py

Expected output (approximate):
    [DNABERT] Loading from local cache  models/dnabert-base/ ...
    [DNABERT] Model loaded. Params: 110M (frozen=True)
    [SMOKE ] Sequence length : 300 bp
    [SMOKE ] 6-mer tokens    : 295 tokens  (300 - 6 + 1)
    [SMOKE ] Input IDs shape : torch.Size([1, 297])  (+ [CLS] + [SEP])
    [SMOKE ] CLS token shape : torch.Size([1, 768])  <- OK
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "models" / "dnabert-base"
HF_MODEL_ID = "zhihan1996/DNA_bert_6"

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 6-mer tokenisation helper
# ---------------------------------------------------------------------------
def seq_to_kmers(seq: str, k: int = 6) -> str:
    """Convert a raw DNA sequence string into a space-separated k-mer string.

    DNABERT was pre-trained with inputs formatted as overlapping k-mers joined
    by spaces, e.g. ``"ATCGTA"`` → ``"ATCGTA TCGTAG CGTAGA ..."`` for k=6.

    Parameters
    ----------
    seq:
        Upper-case DNA string (A/T/G/C/N).
    k:
        k-mer size (6 for DNABERT-6).

    Returns
    -------
    str
        Space-separated k-mer string ready for the DNABERT tokenizer.
    """
    seq = seq.upper().strip()
    if len(seq) < k:
        raise ValueError(f"Sequence length {len(seq)} is shorter than k={k}.")
    return " ".join(seq[i : i + k] for i in range(len(seq) - k + 1))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def get_dnabert(
    freeze: bool = True,
) -> Tuple[PreTrainedTokenizerFast, BertModel]:
    """Load DNABERT-6 tokenizer and model, caching locally.

    Parameters
    ----------
    freeze:
        If ``True`` (default), set ``requires_grad=False`` on every BERT
        parameter.  Useful for feature-extraction warm-up training.

    Returns
    -------
    tokenizer:
        HuggingFace 6-mer tokenizer.
    model:
        ``transformers.BertModel`` with ``output_hidden_states=True``.
        Moved to the best available device (cuda → cpu).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Determine whether the model is already cached
    config_path = CACHE_DIR / "config.json"
    if config_path.exists():
        log.info("[DNABERT] Loading from local cache  %s ...", CACHE_DIR)
        load_from = str(CACHE_DIR)
    else:
        log.info(
            "[DNABERT] Cache not found at %s — downloading %s from HuggingFace Hub ...",
            CACHE_DIR,
            HF_MODEL_ID,
        )
        load_from = HF_MODEL_ID

    # Load tokenizer -----------------------------------------------------------
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        load_from,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
    )

    # Load model ---------------------------------------------------------------
    model: BertModel = AutoModel.from_pretrained(
        load_from,
        cache_dir=str(CACHE_DIR),
        output_hidden_states=True,
        trust_remote_code=True,
    )

    # Persist to canonical local path so next load is purely offline ----------
    if load_from != str(CACHE_DIR):
        log.info("[DNABERT] Saving to local cache %s ...", CACHE_DIR)
        tokenizer.save_pretrained(str(CACHE_DIR))
        model.save_pretrained(str(CACHE_DIR))
        log.info("[DNABERT] Saved.")

    # Freeze -------------------------------------------------------------------
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        log.info("[DNABERT] All BERT weights frozen (requires_grad=False).")

    # Move to best device ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Report -------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(
        "[DNABERT] Model loaded. Total params: %.1fM  |  Trainable: %.1fM  |  "
        "frozen=%s  |  device=%s",
        total_params,
        trainable,
        freeze,
        device,
    )

    return tokenizer, model


# ---------------------------------------------------------------------------
# CLS embedding convenience
# ---------------------------------------------------------------------------
def encode_sequence(
    seq: str,
    tokenizer: PreTrainedTokenizerFast,
    model: BertModel,
    max_length: int = 512,
) -> torch.Tensor:
    """Return the CLS embedding for a single DNA sequence.

    Parameters
    ----------
    seq:
        Raw DNA string (any case, up to ``max_length`` tokens after k-merisation).
    tokenizer:
        DNABERT-6 tokenizer returned by :func:`get_dnabert`.
    model:
        DNABERT BertModel returned by :func:`get_dnabert`.
    max_length:
        Maximum token length (including [CLS] and [SEP]); default 512.

    Returns
    -------
    torch.Tensor
        CLS embedding of shape ``[1, 768]``.
    """
    device = next(model.parameters()).device
    kmer_str = seq_to_kmers(seq)
    enc = tokenizer(
        kmer_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**enc)

    # outputs.last_hidden_state: [1, seq_len, 768]
    cls_emb = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    return cls_emb


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Run a quick sanity-check forward pass and print tensor shapes."""
    print()
    print("=" * 60)
    print("  DNABERT Backbone Smoke Test")
    print("=" * 60)

    tokenizer, model = get_dnabert(freeze=True)
    device = next(model.parameters()).device

    # Sample 300 bp sequence (typical CRISPR off-target window + flanks)
    sample_seq = (
        "ATCGATCGATCGATCGATCGATCGATCGATCG"
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
        "TTTTAAAACCCCGGGGTTTTAAAACCCCGGGG"
        "ATCGATCGATCGATCGATCGATCGATCGATCG"
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
        "TTTTAAAACCCCGGGGTTTTAAAACCCCGGGG"
        "ATCGATCGATCGATCGATCGATCGATCGATCG"
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
        "TTTTAAAACCCCGGGGTTTTAAAACCCCGGGG"
        "ATCGATCGATCGATCGATCGATCGATCGATCG"
    )  # exactly 320 chars — trimmed below to 300
    sample_seq = sample_seq[:300]
    assert len(sample_seq) == 300, f"Expected 300 bp, got {len(sample_seq)}"

    kmer_str = seq_to_kmers(sample_seq, k=6)
    n_kmers = len(sample_seq) - 6 + 1  # 295

    print(f"  [SMOKE] Sequence length : {len(sample_seq)} bp")
    print(f"  [SMOKE] 6-mer tokens    : {n_kmers} tokens  (300 - 6 + 1)")

    enc = tokenizer(
        kmer_str,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    print(f"  [SMOKE] Input IDs shape : {enc['input_ids'].shape}  (includes [CLS] + [SEP])")

    model.eval()
    with torch.no_grad():
        outputs = model(**enc)

    cls_emb = outputs.last_hidden_state[:, 0, :]
    last_hidden = outputs.last_hidden_state

    print(f"  [SMOKE] last_hidden_state shape : {last_hidden.shape}")
    print(f"  [SMOKE] CLS token shape         : {cls_emb.shape}  <-- should be [1, 768]")

    assert list(cls_emb.shape) == [1, 768], (
        f"CLS shape mismatch: expected [1, 768], got {list(cls_emb.shape)}"
    )
    print()
    print("  [SMOKE] PASS — CLS shape is [1, 768] as expected.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _smoke_test()
