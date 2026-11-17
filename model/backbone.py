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
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "models" / "dnabert-base"
HF_CACHE_DIR = PROJECT_ROOT / "models" / "hf-cache"
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_CACHE_DIR / "modules")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "hub")

import torch
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
HF_MODEL_ID = "zhihan1996/DNA_bert_6"
HF_REVISION = "c56e67ea5827e0ddc67ef059addcf71569b1216e"
MIN_EXPECTED_PARAMETERS = 80_000_000


def _cache_is_complete(path: Path) -> bool:
    """Return true only when the cache contains weights and tokenizer assets."""
    weight_files = [p for name in ("model.safetensors", "pytorch_model.bin")
                    if (p := path / name).is_file() and p.stat().st_size > 100_000_000]
    tokenizer_files = ((path / "tokenizer.json").is_file() or (path / "vocab.txt").is_file())
    required = (path / "config.json").is_file() and (path / "tokenizer_config.json").is_file()
    if not (weight_files and tokenizer_files and required):
        return False
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not (config.get("model_type") == "bert" and config.get("hidden_size") == 768
            and config.get("num_hidden_layers") == 12 and config.get("vocab_size") == 4101):
        return False
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=True
        )
        probe = tokenizer("AAAAAA", add_special_tokens=True)["input_ids"]
        return (len(tokenizer) == config["vocab_size"] and tokenizer.cls_token_id is not None
                and tokenizer.sep_token_id is not None and len(probe) >= 3)
    except Exception:
        return False


def _download_complete_snapshot() -> None:
    """Fetch the pinned repo into a staging directory, then install atomically per file."""
    from huggingface_hub import snapshot_download

    staging = CACHE_DIR.parent / f".{CACHE_DIR.name}.staging"
    staging.mkdir(parents=True, exist_ok=True)
    log.info("[DNABERT] Fetching %s into resumable staging cache %s", HF_MODEL_ID, staging)
    snapshot_download(repo_id=HF_MODEL_ID, revision=HF_REVISION, local_dir=str(staging))
    if not _cache_is_complete(staging):
        raise RuntimeError(
            f"Downloaded DNABERT snapshot is incomplete at {staging}; expected a valid tokenizer "
            "and full model.safetensors or pytorch_model.bin. No random weights were initialized."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for source in staging.iterdir():
        if not source.is_file() or source.name == ".cache":
            continue
        target = CACHE_DIR / source.name
        temporary = target.with_name(target.name + ".partial")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    if not _cache_is_complete(CACHE_DIR):
        raise RuntimeError(f"DNABERT installation did not validate at {CACHE_DIR}")
    # The canonical local cache is now complete; remove the duplicate staging copy.
    shutil.rmtree(staging)

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
    invalid = sorted(set(seq) - set("ACGTN"))
    if invalid:
        raise ValueError(f"Sequence contains unsupported DNA symbols: {invalid}")
    return " ".join(seq[i : i + k] for i in range(len(seq) - k + 1))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def get_dnabert(
    freeze: bool = True,
    device: str | torch.device | None = None,
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
    if _cache_is_complete(CACHE_DIR):
        log.info("[DNABERT] Loading from local cache  %s ...", CACHE_DIR)
        load_from = str(CACHE_DIR)
    else:
        log.info(
            "[DNABERT] Cache missing weights or tokenizer at %s; downloading exact model %s ...",
            CACHE_DIR,
            HF_MODEL_ID,
        )
        _download_complete_snapshot()
        load_from = str(CACHE_DIR)

    # Load tokenizer -----------------------------------------------------------
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        load_from,
        local_files_only=True,
        trust_remote_code=True,
    )

    # Load model ---------------------------------------------------------------
    model, loading_info = AutoModel.from_pretrained(
        load_from,
        local_files_only=True,
        output_hidden_states=True,
        trust_remote_code=True,
        output_loading_info=True,
    )
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise RuntimeError(
            "DNABERT checkpoint did not fully initialize the requested backbone; "
            f"missing={loading_info.get('missing_keys')}, "
            f"mismatched={loading_info.get('mismatched_keys')}, errors={loading_info.get('error_msgs')}. "
            "Refusing to use random or partially initialized weights."
        )
    if loading_info.get("unexpected_keys"):
        log.info("[DNABERT] Ignoring task-head-only checkpoint keys: %s", loading_info["unexpected_keys"])
    if getattr(model.config, "hidden_size", None) != 768 or getattr(model.config, "num_hidden_layers", None) != 12:
        raise RuntimeError(f"Unexpected DNABERT architecture: {model.config}")
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count < MIN_EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"DNABERT loaded only {parameter_count:,} parameters; expected the 12-layer, "
            "768-hidden DNABERT-6 encoder (roughly 89M parameters with its 4,101-token vocabulary)."
        )
    if len(tokenizer) != model.config.vocab_size:
        raise RuntimeError(
            f"Tokenizer/model vocabulary mismatch: {len(tokenizer)} vs {model.config.vocab_size}"
        )
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise RuntimeError("DNABERT tokenizer is missing required CLS/SEP special tokens")

    # Freeze -------------------------------------------------------------------
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        log.info("[DNABERT] All BERT weights frozen (requires_grad=False).")

    # Move to best device ------------------------------------------------------
    target_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(target_device)

    # Report -------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(
        "[DNABERT] Model loaded. Total params: %.1fM  |  Trainable: %.1fM  |  "
        "frozen=%s  |  device=%s",
        total_params,
        trainable,
        freeze,
        target_device,
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
