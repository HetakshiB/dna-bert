"""Extended multimodal preprocessing pipeline for the CRISPR Off-Target Predictor.

Extends the sequence-only pipeline with three additional feature extractors:

  (a) extract_epigenetic()   — average BigWig/CSV signal across 6 ENCODE marks;
                               train-only z-score scaling is applied after splitting.
  (b) extract_3d_genome()    — intersect target loci with Hi-C loop anchors
                               (loop_count), TAD boundaries (tad_distance),
                               and A/B compartment one-hot from subcompartments.
  (c) extract_singlecell()   — map each locus to its nearest gene (by overlap
                               or genomic proximity) and return mean log-norm
                               expression across cells from the PBMC 10x atlas.

Output per-example dict saved as stratified 80/10/10 .pt split files:
  outputs/tensors/train.pt
  outputs/tensors/val.pt
  outputs/tensors/test.pt

Each split is a list of dicts with keys:
  seq_ids      LongTensor  [512]      — DNABERT input_ids
  seq_mask     LongTensor  [512]      — attention mask
  epigen_vec   FloatTensor [6]        — train-only z-scored signal (6 marks)
  genome_vec   FloatTensor [8]        — Hi-C/TAD/compartment features
  sc_expr_vec  FloatTensor [1]        — mean log-norm expression of nearest gene
  reg_label    FloatTensor []         — continuous cleavage_freq (regression)
  cls_label    LongTensor  []         — binary high_eff (classification)

Graceful degradation
--------------------
* Epigenetic: if bigWig files are not downloaded, vector is zeros + a warning.
* 3D Genome:  falls back to zeros if chromosome not in the Hi-C reference.
* Single-cell: if gene not found in atlas, vector is 0.0 (not expressed).

Usage
-----
    python utils/preprocess.py [--sample N] [--no-tests]
"""

from __future__ import annotations

import sys
import os
import gzip
import json
import logging
import argparse
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = Path(os.environ.get("CRISPR_DATA_DIR", PROJECT_ROOT / "data"))
CSV_PATH     = DATA_DIR / "sequence" / "crisprSQL_extracted" / "100720.csv"
TENSOR_DIR   = PROJECT_ROOT / "outputs" / "tensors"
LOG_FILE     = PROJECT_ROOT / "logs" / "preprocess.log"
EPIGEN_DIR   = DATA_DIR / "epigenetic"
GENOME3D_DIR = DATA_DIR / "3dgenome"
SC_DIR       = DATA_DIR / "singlecell"

TENSOR_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

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
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
)
log = logging.getLogger(__name__)
FEATURE_STATUS: dict[str, str] = {}

sys.path.insert(0, str(PROJECT_ROOT))
from model.backbone import get_dnabert, seq_to_kmers  # noqa: E402


# ===========================================================================
# STEP 1 — Load & Clean  (unchanged from original preprocess.py)
# ===========================================================================
def load_and_clean(csv_path: Path, sample: Optional[int] = None) -> pd.DataFrame:
    """Load crisprSQL CSV and apply cleaning rules."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required crisprSQL input CSV not found: {csv_path}")
    log.info("[LOAD] Reading %s ...", csv_path)
    df = pd.read_csv(csv_path)
    log.info("[LOAD] Raw shape: %s", df.shape)

    before = len(df)
    df = df.dropna(subset=["target_sequence"])
    df = df[df["target_sequence"].str.strip() != ""]
    df["target_sequence"] = df["target_sequence"].astype(str).str.upper().str.strip()
    missing_or_empty = before - len(df)
    invalid = ~df["target_sequence"].str.fullmatch(r"[ACGTN]+", na=False)
    if invalid.any():
        log.warning("[CLEAN] Dropping %d sequences containing non-DNA characters.", int(invalid.sum()))
        df = df.loc[~invalid].copy()
    log.info("[CLEAN] Dropped %d rows with missing/empty target_sequence.", missing_or_empty)

    df["cleavage_freq"] = pd.to_numeric(df["cleavage_freq"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["cleavage_freq"])
    log.info("[CLEAN] Dropped %d rows with NaN cleavage_freq.", before - len(df))

    n_neg = (df["cleavage_freq"] < 0).sum()
    if n_neg > 0:
        log.info("[CLEAN] Clipping %d negative cleavage_freq values to 0.", n_neg)
        df["cleavage_freq"] = df["cleavage_freq"].clip(lower=0.0)

    df = df.reset_index(drop=True)

    if sample is not None:
        df = df.head(sample)
        log.info("[LOAD] Subsampled to %d rows.", len(df))

    log.info("[LOAD] Clean shape: %s", df.shape)
    return df


# ===========================================================================
# STEP 2 — Label Engineering
# ===========================================================================
def add_binary_label(df: pd.DataFrame, percentile: float = 0.75,
                     threshold: Optional[float] = None) -> pd.DataFrame:
    if threshold is None:
        threshold = float(df["cleavage_freq"].quantile(percentile))
    df["high_eff"] = (df["cleavage_freq"] > threshold).astype("int8")
    pos, neg = int(df["high_eff"].sum()), int((df["high_eff"] == 0).sum())
    log.info("[LABEL] threshold=%.6f | +%d / -%d (%.1f%%)", threshold, pos, neg, 100*pos/len(df))
    FEATURE_STATUS["labels"] = (
        f"cleavage_freq from crisprSQL; high_eff is > training-split {percentile:.0%} percentile "
        f"(threshold={threshold:.8g})"
    )
    return df


# ===========================================================================
# STEP 3 — Sequence Tokenisation
# ===========================================================================
def tokenise_sequences(df: pd.DataFrame, tokenizer, max_length: int = 512,
                       batch_size: int = 256) -> dict[str, torch.Tensor]:
    all_ids, all_masks = [], []
    seqs = df["target_sequence"].tolist()
    log.info("[TOKENISE] %d sequences (max_length=%d) ...", len(seqs), max_length)
    for start in tqdm(range(0, len(seqs), batch_size), desc="Tokenising"):
        batch = [seq_to_kmers(s.upper().strip()) for s in seqs[start:start+batch_size]]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=max_length, padding="max_length")
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
    return {"input_ids": torch.cat(all_ids, 0), "attention_mask": torch.cat(all_masks, 0)}


# ===========================================================================
# STEP 4a — Epigenetic Features via pybigtools
# ===========================================================================
ENCODE_BASE = "https://www.encodeproject.org"
EPIGEN_MARKS = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3", "ATAC-seq", "DNase-seq"]
# Map crisprSQL cell_line → ENCODE metadata cell key
CELLLINE_MAP = {
    "K562":   "K562",
    "HepG2":  "HepG2",
    "HAP1":   "K562",   # nearest surrogate
    "HEK293": "K562",   # nearest surrogate
    "U2OS":   "K562",   # nearest surrogate
    "HeLa":   "HepG2",
    "embryo": "K562",
}

def _load_encode_urls() -> dict[tuple[str, str], str]:
    """Build mapping (cell_key, mark) -> full bigWig URL from metadata JSONs."""
    url_map: dict[tuple[str, str], str] = {}
    meta_dir = EPIGEN_DIR / "metadata"
    for jf in meta_dir.glob("*.json"):
        if jf.stat().st_size < 10:
            continue
        cell_key = jf.stem.rsplit("_", 1)[0]   # e.g. K562_ATAC-seq -> K562
        mark = jf.stem.rsplit("_", 1)[-1]      # e.g. K562_ATAC-seq -> ATAC-seq
        try:
            records = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(records, list) or not records:
            continue
        # Prefer preferred_default=True; fall back to first record
        preferred = [r for r in records if r.get("preferred_default")]
        rec = preferred[0] if preferred else records[0]
        href = rec.get("href", "")
        if href:
            url_map[(cell_key, mark)] = ENCODE_BASE + href
    return url_map


_BIGWIG_HANDLES: dict[Path, tuple[str, object]] = {}


def _get_bigwig_handle(bw_path: Path) -> tuple[str, object]:
    handle = _BIGWIG_HANDLES.get(bw_path)
    if handle is not None:
        return handle
    try:
        import pyBigWig
        result = ("pyBigWig", pyBigWig.open(str(bw_path)))
    except ImportError:
        import pybigtools
        result = ("pybigtools", pybigtools.open(str(bw_path)))
    _BIGWIG_HANDLES[bw_path] = result
    return result


def _bigwig_mean(bw_path: Path, chrom: str, start: int, end: int) -> float:
    """Return mean signal over [start, end) with either supported BigWig reader."""
    try:
        reader, bw = _get_bigwig_handle(bw_path)
        if reader == "pyBigWig":
            vals = bw.values(chrom, start, end, numpy=True)
        else:
            vals = bw.values(chrom, start, end, missing=0.0)
        arr = np.asarray(vals, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        return float(arr.mean()) if len(arr) > 0 else 0.0
    except Exception:
        return 0.0


def extract_epigenetic(df: pd.DataFrame) -> np.ndarray:
    """Extract 6-mark epigenetic signal per locus, z-scored across dataset.

    For each row:
      - Determine cell_line → ENCODE cell_key via CELLLINE_MAP
      - For each of 6 marks, look up local bigWig or use 0.0 fallback
      - Average signal over [target_start, target_end)

    Returns
    -------
    np.ndarray  shape [N, 6], float32, z-scored per mark column.
    """
    log.info("[EPIGEN] Building epigenetic feature matrix (%d rows x 6 marks) ...", len(df))
    url_map = _load_encode_urls()
    bigwig_cache_dir = EPIGEN_DIR / "bigwig"
    bigwig_cache_dir.mkdir(exist_ok=True)

    # Pre-build local cache paths for available bigWigs
    local_bw: dict[tuple[str, str], Path] = {}
    for (cell, mark), url in url_map.items():
        fname = url.split("/")[-1].replace("@@download", "").replace("//", "").strip("/")
        fpath = bigwig_cache_dir / fname
        if fpath.exists() and fpath.stat().st_size > 1000:
            local_bw[(cell, mark)] = fpath

    n_available = len(local_bw)
    if n_available == 0:
        log.warning("[EPIGEN] No local bigWig files found. Returning zero vectors.")
        log.warning("[EPIGEN] Download bigWig files into data/epigenetic/bigwig/ using:")
        log.warning("[EPIGEN] python scripts/fetch_data.py and add bigWig URLs to DATASET_REGISTRY.")
    else:
        log.info("[EPIGEN] Found %d local bigWig files.", n_available)

    matrix = np.zeros((len(df), 6), dtype=np.float32)

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Epigenetic")):
        chrom = str(row.target_chr)
        start = int(row.target_start) if not pd.isna(row.target_start) else 0
        end   = int(row.target_end)   if not pd.isna(row.target_end)   else start + 23
        if start >= end:
            end = start + 23
        cell_key = CELLLINE_MAP.get(str(row.cell_line), "K562")

        for j, mark in enumerate(EPIGEN_MARKS):
            bw_path = local_bw.get((cell_key, mark))
            if bw_path:
                matrix[i, j] = _bigwig_mean(bw_path, chrom, start, end)
            else:
                # Fallback: use crisprSQL epigenetic columns if available
                col_map = {
                    "H3K4me3":  "epigen_h3k4me3",
                    "DNase-seq": "epigen_dnase",
                    "ATAC-seq":  "epigen_dnase",  # best proxy in crisprSQL
                }
                fallback_col = col_map.get(mark)
                if fallback_col and hasattr(row, fallback_col):
                    val = getattr(row, fallback_col)
                    matrix[i, j] = float(val) if not pd.isna(val) else 0.0

    for _, handle in _BIGWIG_HANDLES.values():
        try:
            if hasattr(handle, "close"):
                handle.close()
            elif hasattr(handle, "__exit__"):
                handle.__exit__(None, None, None)
        except Exception:
            pass
    _BIGWIG_HANDLES.clear()

    populated = [int(np.count_nonzero(matrix[:, j])) for j in range(matrix.shape[1])]
    csv_fallbacks = {
        "H3K4me3": "epigen_h3k4me3", "DNase-seq": "epigen_dnase", "ATAC-seq": "epigen_dnase"
    }
    mark_sources = {}
    for mark in EPIGEN_MARKS:
        has_bigwig = any((CELLLINE_MAP.get(str(cell), "K562"), mark) in local_bw
                         for cell in df["cell_line"].dropna().unique())
        fallback_col = csv_fallbacks.get(mark)
        has_csv = bool(fallback_col and fallback_col in df.columns)
        mark_sources[mark] = "available" if has_bigwig or has_csv else "missing"
    FEATURE_STATUS["epigenetic_missing_marks"] = ",".join(
        mark for mark, availability in mark_sources.items() if availability == "missing"
    )
    FEATURE_STATUS["epigenetic"] = (
        f"mark availability={mark_sources}; nonzero rows per mark={dict(zip(EPIGEN_MARKS, populated))}; "
        f"local BigWigs={n_available}; CSV fallback columns={csv_fallbacks}"
    )
    if not any(populated):
        log.warning("[EPIGEN] No measured ENCODE/CSV values found; features are placeholders (zeros).")
    else:
        log.info("[EPIGEN] Measured nonzero values by mark: %s", dict(zip(EPIGEN_MARKS, populated)))
    log.info("[EPIGEN] Done. shape=%s, mean=%.4f, std=%.4f",
             matrix.shape, matrix.mean(), matrix.std())
    return matrix


# ===========================================================================
# STEP 4b — 3D Genome Features from Hi-C Loop & TAD files
# ===========================================================================
COMPARTMENT_CLASSES = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "B3": 4, "NA": 5}
N_COMPARTMENT = len(COMPARTMENT_CLASSES)   # 6
GENOME_VEC_DIM = 1 + 1 + N_COMPARTMENT    # loop_count + tad_distance + compartment_onehot = 8


def _load_loops(gz_path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Load Hi-C loop anchors → dict{chr: [(x1,x2,y1,y2), ...]}"""
    loops: dict[str, list] = {}
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            if line.startswith("chr1\tx1"):  # header
                continue
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            try:
                c1, x1, x2 = parts[0], int(parts[1]), int(parts[2])
                c2, y1, y2 = parts[3], int(parts[4]), int(parts[5])
                # Only use cis-chromosomal loops
                if c1 != c2:
                    continue
                chrom = c1 if c1.startswith("chr") else f"chr{c1}"
                loops.setdefault(chrom, []).append((x1, x2, y1, y2))
            except ValueError:
                continue
    return loops


def _load_tads(gz_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Load TAD domain boundaries → dict{chr: [(start, end), ...]}"""
    tads: dict[str, list] = {}
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            if line.startswith("chr1\tx1") or line.startswith("chr\t"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
                start, end = int(parts[1]), int(parts[2])
                tads.setdefault(chrom, []).append((start, end))
            except ValueError:
                continue
    return tads


def _load_compartments(gz_path: Path) -> dict[str, list[tuple[int, int, str]]]:
    """Load A/B subcompartment annotations → dict{chr: [(start, end, label), ...]}"""
    comps: dict[str, list] = {}
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            try:
                start, end = int(parts[1]), int(parts[2])
                label = parts[3] if parts[3] in COMPARTMENT_CLASSES else "NA"
            except ValueError:
                continue
            comps.setdefault(chrom, []).append((start, end, label))
    return comps


def _count_loops(loops_for_chr: list, pos: int, window: int = 500_000) -> int:
    """Count loops where one anchor overlaps pos ± window."""
    count = 0
    for (x1, x2, y1, y2) in loops_for_chr:
        if (x1 - window <= pos <= x2 + window) or (y1 - window <= pos <= y2 + window):
            count += 1
    return count


def _nearest_tad_distance(tads_for_chr: list, pos: int) -> float:
    """Distance (bp) to the nearest TAD boundary (start or end)."""
    if not tads_for_chr:
        return 0.0
    best = float("inf")
    for (s, e) in tads_for_chr:
        best = min(best, abs(pos - s), abs(pos - e))
    return float(best)


def _compartment_label(comps_for_chr: list, pos: int) -> str:
    """Return compartment label for pos; 'NA' if not found."""
    for (s, e, label) in comps_for_chr:
        if s <= pos < e:
            return label
    return "NA"


def extract_3d_genome(df: pd.DataFrame) -> np.ndarray:
    """Extract 3D genome features (8-dim) per locus.

    Features
    --------
    [0]    loop_count      : number of Hi-C loops whose anchor overlaps pos ± 500kb
    [1]    tad_distance    : distance in bp to nearest TAD boundary (log10-scaled)
    [2:8]  compartment_oh  : one-hot for {A1, A2, B1, B2, B3, NA}

    Returns np.ndarray [N, 8] float32.
    """
    log.info("[3DGENOME] Loading Hi-C reference files ...")

    # Use GM12878 as default reference (most complete in dataset)
    loops_path  = GENOME3D_DIR / "GSE63525_GM12878_HiCCUPS_looplist.txt.gz"
    tads_path   = GENOME3D_DIR / "GSE63525_GM12878_TAD_domains.txt.gz"
    comps_path  = GENOME3D_DIR / "GSE63525_GM12878_subcompartments.bed.gz"

    missing = [p.name for p in (loops_path, tads_path, comps_path) if not p.is_file()]
    if missing:
        FEATURE_STATUS["3d_genome"] = f"missing reference files: {','.join(missing)}; placeholder zeros used"
        log.warning("[3DGENOME] Missing reference files %s; 3D features will be placeholders.", missing)
        return np.zeros((len(df), GENOME_VEC_DIM), dtype=np.float32)

    loops = _load_loops(loops_path)
    tads  = _load_tads(tads_path)
    comps = _load_compartments(comps_path)
    FEATURE_STATUS["3d_genome"] = (
        "GM12878 reference measured; used as a proxy across cell lines; "
        f"loop chromosomes={len(loops)}, TAD chromosomes={len(tads)}, compartment chromosomes={len(comps)}"
    )

    log.info("[3DGENOME] Loaded %d chromosomes of loops, %d of TADs, %d of compartments.",
             len(loops), len(tads), len(comps))

    matrix = np.zeros((len(df), GENOME_VEC_DIM), dtype=np.float32)

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="3D Genome")):
        chrom = str(row.target_chr)
        # Normalise chromosome name (crisprSQL uses hg19 naming)
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        mid = int((row.target_start + row.target_end) / 2) if (
            not pd.isna(row.target_start) and not pd.isna(row.target_end)
        ) else 0

        # Feature 0: loop count (log1p-scaled)
        lc = _count_loops(loops.get(chrom, []), mid)
        matrix[i, 0] = np.log1p(lc)

        # Feature 1: TAD distance (log10-scaled, capped at 8)
        td = _nearest_tad_distance(tads.get(chrom, []), mid)
        matrix[i, 1] = min(np.log10(td + 1), 8.0)

        # Features 2-7: compartment one-hot
        label = _compartment_label(comps.get(chrom, []), mid)
        idx = COMPARTMENT_CLASSES.get(label, COMPARTMENT_CLASSES["NA"])
        matrix[i, 2 + idx] = 1.0

    log.info("[3DGENOME] Done. shape=%s", matrix.shape)
    return matrix


# ===========================================================================
# STEP 4c — Single-Cell Expression Features
# ===========================================================================
SC_H5_PATH = SC_DIR / "pbmc_10k_v3_filtered_feature_bc_matrix.h5"
SC_VEC_DIM = 1   # mean log-norm expression of nearest gene across all cells


def _build_gene_coord_index(var_df: pd.DataFrame) -> dict[str, str]:
    """Return {gene_name: gene_name} for quick lookup — coords come from GTF or name match."""
    return {g: g for g in var_df.index.tolist()}


def _find_nearest_gene_by_name(chrom: str, start: int, end: int,
                                 gene_index: dict[str, str]) -> Optional[str]:
    """Simple heuristic: return None (expression from crisprSQL geneid if available)."""
    return None   # placeholder; real GTF mapping below


def _build_gene_coord_map_from_crisprSQL(df: pd.DataFrame) -> dict[int, str]:
    """Use crisprSQL target_geneid column (e.g. 'HBB') to map row → gene name."""
    gene_col = "target_geneid"
    if gene_col not in df.columns:
        return {}
    return {i: str(gid).strip() for i, gid in enumerate(df[gene_col]) if pd.notna(gid) and str(gid).strip()}


def extract_singlecell(df: pd.DataFrame) -> np.ndarray:
    """Extract single-cell expression feature per locus (1-dim).

    Strategy
    --------
    1. Load PBMC 10k 10x h5 via scanpy, log-normalise (log1p(counts/10k)).
    2. Compute per-gene mean expression across all cells.
    3. For each row, map target_geneid → gene name → mean expression.
       Fall back to 0.0 if gene not in atlas.

    Returns np.ndarray [N, 1] float32.
    """
    log.info("[SINGLECELL] Loading PBMC 10k atlas from %s ...", SC_H5_PATH)

    if not SC_H5_PATH.exists():
        log.warning("[SINGLECELL] %s not found — returning zeros.", SC_H5_PATH)
        FEATURE_STATUS["singlecell"] = f"missing {SC_H5_PATH.name}; placeholder zeros used"
        return np.zeros((len(df), SC_VEC_DIM), dtype=np.float32)

    try:
        import scanpy as sc
        import scipy.sparse as sp
    except ImportError as exc:
        log.warning("[SINGLECELL] Optional scanpy dependency unavailable (%s); placeholder zeros used.", exc)
        FEATURE_STATUS["singlecell"] = f"scanpy unavailable; placeholder zeros used: {exc}"
        return np.zeros((len(df), SC_VEC_DIM), dtype=np.float32)

    sc.settings.verbosity = 0

    try:
        adata = sc.read_10x_h5(str(SC_H5_PATH))
    except Exception as exc:
        log.warning("[SINGLECELL] Could not read atlas (%s); placeholder zeros used.", exc)
        FEATURE_STATUS["singlecell"] = f"unreadable atlas; placeholder zeros used: {exc}"
        return np.zeros((len(df), SC_VEC_DIM), dtype=np.float32)
    adata.var_names_make_unique()

    # Log-normalise
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Compute per-gene mean across all cells
    X = adata.X
    if sp.issparse(X):
        gene_means = np.asarray(X.mean(axis=0)).flatten()   # [n_genes]
    else:
        gene_means = X.mean(axis=0).flatten()

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.tolist())}
    gene_ids_map = _build_gene_coord_map_from_crisprSQL(df)   # row_i -> gene_name

    log.info("[SINGLECELL] Atlas genes: %d | Rows with gene_id: %d",
             len(gene_to_idx), len(gene_ids_map))

    matrix = np.zeros((len(df), SC_VEC_DIM), dtype=np.float32)
    hits = 0
    for i in range(len(df)):
        gene_name = gene_ids_map.get(i)
        if gene_name and gene_name in gene_to_idx:
            matrix[i, 0] = float(gene_means[gene_to_idx[gene_name]])
            hits += 1

    log.info("[SINGLECELL] Done. Matched %d / %d rows to atlas genes.", hits, len(df))
    FEATURE_STATUS["singlecell"] = (
        f"PBMC 10x atlas loaded; expression matched for {hits}/{len(df)} rows; "
        "unmatched rows are placeholders (zero)"
    )
    return matrix


# ===========================================================================
# STEP 5 — Stratified 80/10/10 Split & Save
# ===========================================================================
def save_split_tensors(
    seq_tensors:  dict[str, torch.Tensor],
    epigen_mat:   np.ndarray,
    genome_mat:   np.ndarray,
    sc_mat:       np.ndarray,
    df:           pd.DataFrame,
    out_dir:      Path,
    overwrite:    bool = False,
    split_indices: Optional[dict[str, np.ndarray]] = None,
) -> None:
    """Combine all features into per-example dicts, split, and save .pt files.

    Split
    -----
    Uses a fixed 80/10/10 split. The classification threshold is learned from
    training targets only; the held-out validation/test rows are stratified
    when labels/counts permit.

    Each .pt file contains a list[dict] where every dict has keys:
      seq_ids, seq_mask, epigen_vec, genome_vec, sc_expr_vec, reg_label, cls_label
    """
    N = len(df)
    log.info("[SPLIT] Building %d per-example feature dicts ...", N)

    input_ids     = seq_tensors["input_ids"]      # [N, 512]
    attention_mask = seq_tensors["attention_mask"] # [N, 512]
    labels_np     = df["high_eff"].values.astype(np.int64)
    freq_np       = df["cleavage_freq"].values.astype(np.float32)

    epigen_t = torch.from_numpy(epigen_mat)  # [N, 6]
    genome_t = torch.from_numpy(genome_mat)  # [N, 8]
    sc_t     = torch.from_numpy(sc_mat)      # [N, 1]
    cls_t    = torch.from_numpy(labels_np)   # [N]
    reg_t    = torch.from_numpy(freq_np)     # [N]

    if N < 10:
        raise ValueError(f"At least 10 cleaned rows are required for an 80/10/10 split; got {N}.")

    if split_indices is None:
        # Utility/test callers without a precomputed training-only label threshold.
        indices = np.arange(N)
        strat = labels_np
        can_stratify = np.unique(strat, return_counts=True)[1].min() >= 2
        if not can_stratify:
            log.warning("[SPLIT] A class has fewer than 2 rows; using deterministic unstratified split.")
        idx_train, idx_temp = train_test_split(
            indices, test_size=0.20, random_state=42, stratify=strat if can_stratify else None
        )
        temp_can_stratify = can_stratify and np.unique(strat[idx_temp], return_counts=True)[1].min() >= 2
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=0.50, random_state=42,
            stratify=strat[idx_temp] if temp_can_stratify else None
        )
    else:
        idx_train, idx_val, idx_test = (np.asarray(split_indices[key], dtype=np.int64)
                                         for key in ("train", "val", "test"))
        all_indices = np.concatenate((idx_train, idx_val, idx_test))
        if len(all_indices) != N or set(all_indices.tolist()) != set(range(N)):
            raise ValueError("Precomputed train/val/test indices must partition every row exactly once.")
        log.info("[SPLIT] Using fixed train-derived label threshold split; validation/test are stratified where possible.")

    # Fit epigenetic normalization on training rows only to avoid split leakage.
    epigen_means = epigen_mat[idx_train].mean(axis=0, keepdims=True)
    epigen_stds = epigen_mat[idx_train].std(axis=0, keepdims=True)
    missing_marks = set(filter(None, FEATURE_STATUS.get("epigenetic_missing_marks", "").split(",")))
    for column, mark in enumerate(EPIGEN_MARKS):
        if mark in missing_marks:
            epigen_means[0, column] = 0.0
            epigen_stds[0, column] = 1.0
    epigen_stds[epigen_stds < 1e-8] = 1.0
    epigen_mat = ((epigen_mat - epigen_means) / epigen_stds).astype(np.float32)
    epigen_t = torch.from_numpy(epigen_mat)

    log.info("[SPLIT] train=%d  val=%d  test=%d", len(idx_train), len(idx_val), len(idx_test))

    def _build_split(idxs: np.ndarray) -> list[dict]:
        examples = []
        for i in idxs:
            examples.append({
                "seq_ids":     input_ids[i],         # [512]
                "seq_mask":    attention_mask[i],     # [512]
                "epigen_vec":  epigen_t[i],           # [6]
                "genome_vec":  genome_t[i],           # [8]
                "sc_expr_vec": sc_t[i],               # [1]
                "reg_label":   reg_t[i],              # scalar
                "cls_label":   cls_t[i],              # scalar
            })
        return examples

    splits = {
        "train": idx_train,
        "val":   idx_val,
        "test":  idx_test,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.iterdir() if p.is_file()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Preprocessing outputs already exist in {out_dir}. Use --overwrite to preserve a backup and regenerate."
        )
    if existing:
        backup_dir = out_dir / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=False)
        for path in existing:
            shutil.copy2(path, backup_dir / path.name)
        log.warning("[BACKUP] Existing preprocessing outputs copied to %s", backup_dir)
    row_split = np.empty(N, dtype=object)
    for split_name, idxs in splits.items():
        row_split[idxs] = split_name
    for split_name, idxs in splits.items():
        data = _build_split(idxs)
        out_path = out_dir / f"{split_name}.pt"
        torch.save(data, out_path)
        pos = int(sum(d["cls_label"].item() for d in data))
        log.info("[SAVE] %s.pt — %d examples  |  high_eff=1: %d  |  high_eff=0: %d",
                 split_name, len(data), pos, len(data) - pos)

    # Also save flat tensors for direct DataLoader use
    for tensor_name, tensor in [
        ("input_ids",      input_ids),
        ("attention_mask", attention_mask),
        ("epigen_vec",     epigen_t),
        ("genome_vec",     genome_t),
        ("sc_expr_vec",    sc_t),
        ("labels",         cls_t),
        ("cleavage_freq",  reg_t),
    ]:
        torch.save(tensor, out_dir / f"{tensor_name}.pt")

    # Save split index arrays for reproducibility
    torch.save({"train": idx_train, "val": idx_val, "test": idx_test},
               out_dir / "split_indices.pt")

    meta = df.copy()
    meta["split"] = row_split
    for modality, status in FEATURE_STATUS.items():
        meta[f"{modality}_provenance"] = status
    meta["epigenetic_normalization"] = "z-score fit on train split only"
    meta.to_csv(out_dir / "meta.csv", index=False)
    (out_dir / "feature_scaler.json").write_text(json.dumps({
        "epigenetic_mean": epigen_means.reshape(-1).tolist(),
        "epigenetic_std": epigen_stds.reshape(-1).tolist(),
        "fit_split": "train",
    }, indent=2), encoding="utf-8")
    log.info("[SAVE] Feature provenance and split assignment written to %s", out_dir / "meta.csv")

    log.info("[SAVE] All tensors and split files written to %s", out_dir)


# ===========================================================================
# STEP 6 — Unit Tests
# ===========================================================================
def run_unit_tests(
    seq_tensors: dict[str, torch.Tensor],
    epigen_mat:  np.ndarray,
    genome_mat:  np.ndarray,
    sc_mat:      np.ndarray,
    df:          pd.DataFrame,
    max_length:  int = 512,
) -> None:
    """9 + 5 unit tests for sequence + multimodal correctness."""
    N = len(df)
    errors: list[str] = []

    def chk(cond: bool, msg: str, test_id: str) -> None:
        if not cond:
            errors.append(f"{test_id} FAIL: {msg}")
            log.error("  %s FAIL: %s", test_id, msg)
        else:
            log.info("  %s PASS", test_id)

    # --- Sequence tests ---
    chk(df["cleavage_freq"].isna().sum() == 0, "NaN in cleavage_freq", "[T1 ]")
    chk(df["target_sequence"].isna().sum() == 0, "NaN in target_sequence", "[T2 ]")
    chk(set(df["high_eff"].unique()).issubset({0, 1}), "high_eff not binary", "[T3 ]")
    chk(tuple(seq_tensors["input_ids"].shape) == (N, max_length),
        f"input_ids shape {seq_tensors['input_ids'].shape} != ({N},{max_length})", "[T4 ]")
    chk(seq_tensors["attention_mask"].unique().tolist() in ([0, 1], [1], [0]),
        "attention_mask not binary", "[T5 ]")
    chk(len(df) == N, f"df length {len(df)} != N={N}", "[T6 ]")
    chk(bool((seq_tensors["attention_mask"][:, 0] == 1).all()), "CLS not attended", "[T7 ]")
    chk(not torch.isnan(torch.tensor(df["cleavage_freq"].values, dtype=torch.float32)).any(),
        "NaN in cleavage_freq tensor", "[T8 ]")
    chk(int(df["high_eff"].sum()) + int((df["high_eff"] == 0).sum()) == N,
        "Label counts mismatch", "[T9 ]")

    # --- Multimodal tests ---
    chk(epigen_mat.shape == (N, 6), f"epigen_mat shape {epigen_mat.shape} != ({N},6)", "[T10]")
    chk(not np.isnan(epigen_mat).any(), "NaN in epigen_mat", "[T11]")
    chk(genome_mat.shape == (N, GENOME_VEC_DIM),
        f"genome_mat shape {genome_mat.shape} != ({N},{GENOME_VEC_DIM})", "[T12]")
    chk(not np.isnan(genome_mat).any(), "NaN in genome_mat", "[T13]")
    chk(sc_mat.shape == (N, SC_VEC_DIM),
        f"sc_mat shape {sc_mat.shape} != ({N},{SC_VEC_DIM})", "[T14]")

    log.info("-" * 60)
    if errors:
        for e in errors:
            log.error("  %s", e)
        raise AssertionError(f"{len(errors)} unit test(s) FAILED.")
    log.info("  All 14 unit tests PASSED.")


# ===========================================================================
# Main Pipeline
# ===========================================================================
def preprocess(sample: Optional[int] = None, run_tests: bool = True,
               max_length: int = 512, overwrite: bool = False,
               output_dir: Path = TENSOR_DIR) -> None:
    FEATURE_STATUS.clear()
    log.info("=" * 72)
    log.info("CRISPR Off-Target Predictor — Multimodal Preprocessing Pipeline")
    log.info("=" * 72)

    if not 8 <= max_length <= 512:
        raise ValueError("--max-length must be between 8 and the DNABERT limit of 512 tokens.")
    log.info("[INIT] Loading DNABERT-6 tokenizer ...")
    # Tokenization itself is CPU work; leave the 6 GB GPU free for model training.
    tokenizer, _ = get_dnabert(freeze=True, device="cpu")

    df = load_and_clean(CSV_PATH, sample=sample)
    # Fix the held-out rows first; derive the risk threshold only from training targets.
    indices = np.arange(len(df))
    idx_train, idx_temp = train_test_split(indices, test_size=0.20, random_state=42)
    train_threshold = float(df.iloc[idx_train]["cleavage_freq"].quantile(0.75))
    df = add_binary_label(df, threshold=train_threshold)
    temp_labels = df["high_eff"].to_numpy()[idx_temp]
    temp_counts = np.unique(temp_labels, return_counts=True)[1]
    temp_stratify = temp_labels if len(temp_counts) > 1 and temp_counts.min() >= 2 else None
    if temp_stratify is None:
        log.warning("[SPLIT] Held-out pool is too small/imbalanced for stratified val/test; using random 50/50 split.")
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42,
                                         stratify=temp_stratify)
    split_indices = {"train": idx_train, "val": idx_val, "test": idx_test}

    # --- Sequence ---
    seq_tensors = tokenise_sequences(df, tokenizer, max_length=max_length)

    # --- Epigenetic ---
    epigen_mat = extract_epigenetic(df)

    # --- 3D Genome ---
    genome_mat = extract_3d_genome(df)

    # --- Single-Cell ---
    sc_mat = extract_singlecell(df)

    FEATURE_STATUS["sequence"] = (
        f"DNABERT-6 tokenizer; tokenized {len(df)} sequences; max_length={max_length}"
    )
    # --- Tests ---
    if run_tests:
        log.info("[TESTS] Running 14 unit tests ...")
        run_unit_tests(seq_tensors, epigen_mat, genome_mat, sc_mat, df, max_length)

    # --- Save + Split ---
    save_split_tensors(seq_tensors, epigen_mat, genome_mat, sc_mat, df, output_dir,
                       overwrite=overwrite, split_indices=split_indices)

    log.info("=" * 72)
    log.info("Multimodal preprocessing complete.")
    log.info("  %s/train.pt  val.pt  test.pt", output_dir)
    log.info("=" * 72)


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal CRISPR preprocessing.")
    parser.add_argument("--sample",     type=int, default=None, help="First N rows only.")
    parser.add_argument("--no-tests",  action="store_true",    help="Skip unit tests.")
    parser.add_argument("--max-length", type=int, default=512,  help="Token pad/trunc length.")
    parser.add_argument("--overwrite", action="store_true", help="Back up existing tensors/metadata before regenerating.")
    parser.add_argument("--output-dir", type=Path, default=TENSOR_DIR, help="Output directory (defaults to outputs/tensors).")
    args = parser.parse_args()
    preprocess(sample=args.sample, run_tests=not args.no_tests, max_length=args.max_length,
               overwrite=args.overwrite, output_dir=args.output_dir)
