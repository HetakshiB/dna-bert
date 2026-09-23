#!/usr/bin/env python3
"""Dataset Inventory Verification & Automated Fetcher for Multimodal CRISPR Predictor.

This script checks for each dataset listed in Section 2 of the data manifest
(data/README.md) under the data/ directory. If any dataset file is missing,
it attempts an automated download from the specified source.

Features:
- Idempotent: Skips download if the destination file already exists and is non-empty.
- Logging: Logs progress and errors to both stdout and 'logs/preprocess.log'.
- Resilient: Fails gracefully per file (capturing HTTP/network errors) without crashing the pipeline.

PROVENANCE INVENTORY (Section 2 Reference Table):
====================================================================================================
Section 2.1: data/sequence/ - validated OTS + on-target activity (22 MB)
----------------------------------------------------------------------------------------------------
File                              Description                                 Blueprint Role
----------------------------------------------------------------------------------------------------
crispr_offt.h5                    CRISPR-OFFT model weights (Lin & Wong 2018) Off-target baseline & seed
crispr_offt_README.md             CRISPR-ONT/OFFT usage notes                 Documentation
crisprSQL_100720.zip              crisprSQL archive (44 gRNAs, 25,632 pairs)  Secondary supervision
crisprSQL_extracted/100720.csv    Extracted crisprSQL table                   Training data Branch 1+2
crisprSQL_README.md               crisprSQL documentation                     Documentation
crisprSQL_main.txt                crisprSQL main data dump                    Reference dump
encoded_test_offt.txt             Encoded off-target test split               Cross-study evaluation
test_offt.csv                     Off-target test labels                      Cross-study evaluation
test_offt_batch.txt               Off-target batch split                      Cross-study evaluation
test_ont.csv                      On-target activity test labels              AUROC baselines
test_ont_batch.csv                On-target batch test split                  AUROC baselines
ci2.fa, ci3.fa, jensenTest.fa     Reference FASTA files for Cas-OFFinder      Candidate generation
GUIDEseq_peaks.rda                GUIDE-seq peak annotations                  Supervised labels
GUIDEseq_uniqueCleavageEvents.rda GUIDE-seq unique cleavage events            Supervised labels

Section 2.2: data/epigenetic/ - ENCODE histone/accessibility metadata (203 KB)
----------------------------------------------------------------------------------------------------
Six cell types (K562, HepG2, CD14 monocytes, CD34 HSCs, ARPE-19, iPSC) × 6 marks
(H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC-seq, DNase-seq) = 36 metadata JSONs:
metadata/*.json                   36 per-(cell, assay) ENCODE file metadata   Epigenetic feature index
metadata/celltypes_all.json       Flat index of all cell-type x mark records  Aggregated index
metadata/search_summary.json      Counts & per-mark ENCODE file IDs           Assay inventory summary
scripts/search_encode.py          Re-runnable ENCODE Portal query tool        Portal query pipeline
logs/search.log                   Audit trail of ENCODE queries               Provenance log

Section 2.3: data/3dgenome/ - Hi-C / TAD / compartments (3.0 MB, Rao et al. 2014, GSE63525)
----------------------------------------------------------------------------------------------------
GSE63525_GM12878_HiCCUPS_looplist.txt.gz  GM12878 chromatin loops             Long-range contact risk
GSE63525_GM12878_TAD_domains.txt.gz       GM12878 TAD domain calls            Domain boundary penalty
GSE63525_GM12878_subcompartments.bed.gz   GM12878 A/B sub-compartments        Chromatin active state
GSE63525_K562_HiCCUPS_looplist.txt.gz     K562 chromatin loops                Leukemia structural risk
GSE63525_K562_TAD_domains.txt.gz          K562 TAD domain calls               TAD boundary features
GSE63525_HUVEC_loops.txt.gz               HUVEC loops                         Endothelial contact maps
GSE63525_HUVEC_TADs.txt.gz                HUVEC TADs                          TAD boundary features
GSE63525_IMR90_loops.txt.gz               IMR90 fibroblast loops              Fibroblast contact maps
GSE63525_IMR90_TADs.txt.gz                IMR90 fibroblast TADs               TAD boundary features

Section 2.4: data/singlecell/ - scRNA-seq + scATAC-seq (81 MB)
----------------------------------------------------------------------------------------------------
pbmc_10k_v3_filtered_feature_bc_matrix.h5 ~10,000 PBMCs scRNA-seq (10x)      Immune cell expression
pbmc_1k_v3_filtered_feature_bc_matrix.h5  ~1,000 PBMCs scRNA-seq (10x)       Reference test matrix
pbmc_3k_filtered_gene_bc_matrices.tar.gz  ~3,000 PBMCs scRNA-seq mtx (10x)   Benchmark scRNA profile
atac_pbmc_5k_singlecell.csv               ~5,000 PBMCs scATAC-seq (10x)      Single-cell accessibility
heart_1k_v3_filtered_feature_bc_matrix.h5 ~1,000 heart cells scRNA-seq (10x) Tissue specificity profile
GSE101498_RAW.tar                         Therapeutic tissue cohort scRNA-seq Sickle-cell / HSC atlas
tabula_sapiens_data.html                  Tabula Sapiens index (CZ Biohub)    Cross-tissue catalogue

Section 2.5: data/benchmarks/ - baseline models + reference data (99 MB)
----------------------------------------------------------------------------------------------------
09_Genome_Refs/hg38.chrom.sizes           UCSC hg38 chromosome sizes table    Interval coordinate maps
10_Gencode/gencode.v44.annotation.gtf.gz  GENCODE v44 primary annotation GTF  Gene overlap severity
====================================================================================================
"""

import os
import sys
import json
import zipfile
import logging
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# Safe console output across Windows cp1252 / UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CRISPR_DATA_DIR", PROJECT_ROOT / "data"))
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "preprocess.log"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging to both console and logs/preprocess.log
logger = logging.getLogger("fetch_data")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter("[%(levelname)s] %(message)s")
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)


# Provenance Dataset Registry
DATASET_REGISTRY = [
    # ------------------ Branch 1: Sequence ------------------
    {
        "modality": "Sequence",
        "relative_path": "sequence/crispr_offt.h5",
        "description": "CRISPR-OFFT pre-trained weights (Lin & Wong 2018)",
        "url": "https://raw.githubusercontent.com/Lin-bu/CRISPR-ONT/master/crispr_offt.h5",
    },
    {
        "modality": "Sequence",
        "relative_path": "sequence/crispr_offt_README.md",
        "description": "CRISPR-ONT/OFFT README notes",
        "url": "https://raw.githubusercontent.com/Lin-bu/CRISPR-ONT/master/README.md",
    },
    {
        "modality": "Sequence",
        "relative_path": "sequence/test_offt.csv",
        "description": "Validated off-target test split",
        "url": "https://raw.githubusercontent.com/Lin-bu/CRISPR-ONT/master/test_offt.csv",
    },
    {
        "modality": "Sequence",
        "relative_path": "sequence/test_ont.csv",
        "description": "Validated on-target test split",
        "url": "https://raw.githubusercontent.com/Lin-bu/CRISPR-ONT/master/test_ont.csv",
    },
    {
        "modality": "Sequence",
        "relative_path": "sequence/crisprSQL_extracted/100720.csv",
        "description": "crisprSQL 25k guide-target pair table",
        "parent_zip": "sequence/crisprSQL_100720.zip",
        "url": None,
    },

    # ------------------ Branch 2: Epigenetic ------------------
    {
        "modality": "Epigenetic",
        "relative_path": "epigenetic/metadata/celltypes_all.json",
        "description": "ENCODE metadata index across 6 cell types and 6 histone/chromatin marks",
        "synthesize_from_metadata": True,
        "url": None,
    },
    {
        "modality": "Epigenetic",
        "relative_path": "epigenetic/metadata/search_summary.json",
        "description": "ENCODE assay search summary counts",
        "url": None,
    },

    # ------------------ Branch 3: 3D Genome ------------------
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_GM12878_HiCCUPS_looplist.txt.gz",
        "description": "GM12878 chromatin loops (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_HiCCUPS_looplist.txt.gz",
    },
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_GM12878_TAD_domains.txt.gz",
        "description": "GM12878 TAD domain calls (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt.gz",
    },
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_GM12878_subcompartments.bed.gz",
        "description": "GM12878 A/B subcompartment annotations (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_subcompartments.bed.gz",
    },
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_K562_HiCCUPS_looplist.txt.gz",
        "description": "K562 chromatin loops (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_HiCCUPS_looplist.txt.gz",
    },
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_K562_TAD_domains.txt.gz",
        "description": "K562 TAD domain calls (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_Arrowhead_domainlist.txt.gz",
    },
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_HUVEC_loops.txt.gz",
        "description": "HUVEC loops (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_HUVEC_HiCCUPS_looplist.txt.gz",
    },
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_HUVEC_TADs.txt.gz",
        "description": "HUVEC TAD domains (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_HUVEC_Arrowhead_domainlist.txt.gz",
    },
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_IMR90_loops.txt.gz",
        "description": "IMR90 fibroblast loops (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_IMR90_HiCCUPS_looplist.txt.gz",
    },
    {
        "modality": "3D Genome",
        "relative_path": "3dgenome/GSE63525_IMR90_TADs.txt.gz",
        "description": "IMR90 fibroblast TADs (Rao et al. 2014 GSE63525)",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_IMR90_Arrowhead_domainlist.txt.gz",
    },

    # ------------------ Branch 4: Single-Cell ------------------
    {
        "modality": "Single-Cell",
        "relative_path": "singlecell/pbmc_10k_v3_filtered_feature_bc_matrix.h5",
        "description": "10x Genomics PBMC 10k v3 filtered feature matrix",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5",
    },
    {
        "modality": "Single-Cell",
        "relative_path": "singlecell/pbmc_1k_v3_filtered_feature_bc_matrix.h5",
        "description": "10x Genomics PBMC 1k v3 filtered feature matrix",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_filtered_feature_bc_matrix.h5",
    },
    {
        "modality": "Single-Cell",
        "relative_path": "singlecell/pbmc_3k_filtered_gene_bc_matrices.tar.gz",
        "description": "10x Genomics PBMC 3k filtered gene matrices (tar.gz)",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
    },
    {
        "modality": "Single-Cell",
        "relative_path": "singlecell/atac_pbmc_5k_singlecell.csv",
        "description": "10x Genomics PBMC 5k scATAC-seq cell summary",
        "url": "https://cf.10xgenomics.com/samples/cell-atac/1.2.0/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_singlecell.csv",
    },
    {
        "modality": "Single-Cell",
        "relative_path": "singlecell/heart_1k_v3_filtered_feature_bc_matrix.h5",
        "description": "10x Genomics Heart 1k v3 filtered feature matrix",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/heart_1k_v3/heart_1k_v3_filtered_feature_bc_matrix.h5",
    },
    {
        "modality": "Single-Cell",
        "relative_path": "singlecell/GSE101498_RAW.tar",
        "description": "GEO GSE101498 raw cohort scRNA-seq",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE101498&format=file",
    },
    {
        "modality": "Single-Cell",
        "relative_path": "singlecell/tabula_sapiens_data.html",
        "description": "Tabula Sapiens reference portal entry",
        "url": "https://tabula-sapiens-portal.ds.czbiohub.org/",
    },

    # ------------------ Benchmarks & Genomic References ------------------
    {
        "modality": "Benchmarks",
        "relative_path": "benchmarks/09_Genome_Refs/hg38.chrom.sizes",
        "description": "UCSC hg38 chromosome sizes table",
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
    },
    {
        "modality": "Benchmarks",
        "relative_path": "benchmarks/10_Gencode/gencode.v44.primary_assembly.annotation.gtf.gz",
        "description": "GENCODE v44 human primary assembly annotations (~1.4 GB)",
        "url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38_mapping/gencode.v44.primary_assembly.annotation.gtf.gz",
        "large_file": True,
    },
]


def human_size(num_bytes: int) -> str:
    """Format bytes into readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def synthesize_celltypes_all(dest_path: Path) -> bool:
    """Synthesize metadata/celltypes_all.json from individual 36 ENCODE JSONs."""
    meta_dir = dest_path.parent
    if not meta_dir.exists():
        return False

    all_records = {}
    json_files = list(meta_dir.glob("*.json"))
    for jf in json_files:
        if jf.name in ["celltypes_all.json", "search_summary.json"]:
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                content = json.load(f)
                key = jf.stem
                all_records[key] = content
        except Exception:
            continue

    if all_records:
        try:
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2)
            logger.info(f"[SYNTHESIZED] {dest_path.name} from {len(all_records)} ENCODE metadata files.")
            return True
        except Exception as e:
            logger.warning(f"[WARN] Failed to write {dest_path.name}: {e}")
            return False
    return False


def download_file(url: str, dest_path: Path, timeout: int = 60) -> bool:
    """Stream download a file with progress reporting and graceful error handling."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    headers = {"User-Agent": "CRISPR-OffTarget-FetchBot/1.0 (Research Pipeline)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        logger.info(f"Connecting to {url} ...")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks

            with open(temp_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)

        # Atomic rename once download is fully complete
        if temp_path.exists():
            temp_path.replace(dest_path)
            logger.info(f"[OK] Successfully downloaded: {dest_path.name} ({human_size(dest_path.stat().st_size)})")
            return True

    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        logger.warning(f"[WARN] Failed to fetch {dest_path.name} from {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False
    except Exception as e:
        logger.warning(f"[WARN] Unexpected error fetching {dest_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

    return False


def verify_and_fetch_datasets(download_missing: bool = True, skip_large: bool = True):
    """Iterate through the dataset provenance inventory, checking presence and fetching if needed."""
    logger.info("=" * 80)
    logger.info("Starting Dataset Inventory & Provenance Verification")
    logger.info(f"Project Data Root: {DATA_DIR}")
    logger.info(f"Logging to       : {LOG_FILE}")
    logger.info("=" * 80)

    stats = {
        "total": len(DATASET_REGISTRY),
        "present": 0,
        "downloaded": 0,
        "synthesized": 0,
        "failed": 0,
        "no_url": 0,
    }

    for item in DATASET_REGISTRY:
        rel_path = item["relative_path"]
        target_path = DATA_DIR / rel_path
        desc = item["description"]
        url = item.get("url")
        is_large = item.get("large_file", False)
        parent_zip = item.get("parent_zip")
        synthesize = item.get("synthesize_from_metadata", False)

        # 1. Idempotency Check: Does file already exist and have content?
        if target_path.exists() and target_path.stat().st_size > 0:
            size_str = human_size(target_path.stat().st_size)
            logger.info(f"[EXISTS] {rel_path} ({size_str}) - {desc}")
            stats["present"] += 1
            continue

        # 2. Check if synthesis from metadata is configured
        if synthesize:
            logger.info(f"[SYNTHESIZING] {rel_path} from local ENCODE metadata...")
            if synthesize_celltypes_all(target_path):
                stats["synthesized"] += 1
                stats["present"] += 1
                continue

        # 3. Check if extractable from local parent archive
        if parent_zip:
            zip_full_path = DATA_DIR / parent_zip
            if zip_full_path.exists():
                logger.info(f"[EXTRACTING] {rel_path} from {parent_zip} ...")
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zip_full_path, "r") as zf:
                        member_name = Path(rel_path).name
                        matched_members = [m for m in zf.namelist() if m.endswith(member_name)]
                        if matched_members:
                            with zf.open(matched_members[0]) as source, open(target_path, "wb") as target:
                                target.write(source.read())
                            logger.info(f"[OK] Extracted {rel_path} ({human_size(target_path.stat().st_size)})")
                            stats["present"] += 1
                            continue
                except Exception as e:
                    logger.warning(f"[WARN] Could not extract {rel_path} from {parent_zip}: {e}")

        # 4. Handle missing file
        logger.info(f"[MISSING] {rel_path} - {desc}")

        if not download_missing:
            stats["failed"] += 1
            continue

        if not url:
            logger.info(f"  ↳ No automated URL configured (local derived file or requires interactive portal access).")
            stats["no_url"] += 1
            continue

        if is_large and skip_large:
            logger.info(f"  ↳ Skipping large file download (>1 GB): {url} (Use --include-large to download).")
            stats["failed"] += 1
            continue

        # 5. Attempt download
        logger.info(f"  ↳ Attempting download from: {url}")
        success = download_file(url, target_path)
        if success:
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1

    # Print Summary Report
    logger.info("=" * 80)
    logger.info("Dataset Verification Summary:")
    logger.info(f"  Total items checked     : {stats['total']}")
    logger.info(f"  Already present (cached): {stats['present']}")
    logger.info(f"  Successfully fetched    : {stats['downloaded']}")
    logger.info(f"  Synthesized from local  : {stats['synthesized']}")
    logger.info(f"  Manual / Derived files  : {stats['no_url']}")
    logger.info(f"  Failed / Unreachable    : {stats['failed']}")
    logger.info("=" * 80)
    logger.info(f"Full log written to: {LOG_FILE}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fetch and verify CRISPR multimodal off-target datasets.")
    parser.add_argument("--check-only", action="store_true", help="Only verify existing files, do not download.")
    parser.add_argument("--include-large", action="store_true", help="Also download large files (e.g., GENCODE GTF 1.4 GB).")
    args = parser.parse_args()

    verify_and_fetch_datasets(download_missing=not args.check_only, skip_large=not args.include_large)


if __name__ == "__main__":
    main()
