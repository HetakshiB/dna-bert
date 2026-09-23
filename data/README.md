# CRISPR Multimodal Safety Engine — Dataset Manifest

This directory contains all datasets gathered to build the **Multimodal CRISPR Off-Target Safety Engine** described in `CRISPR_Multimodal_Safety_Engine_Blueprint.docx`. The system fuses **four biological signal layers**; each layer has a dedicated subfolder.

---

## 1. The Four Modality Mapping (per blueprint §4.2)

| Blueprint branch | Folder | What feeds it | What the model learns |
|---|---|---|---|
| **Branch 1 — Sequence** | `data/sequence/` | Validated on/off-target cleavage sites, gRNA-DNA mismatch features | Local mismatch patterns, PAM-distal tolerance, bulge handling |
| **Branch 2 — Epigenetic** | `data/epigenetic/` | ENCODE histone marks (H3K4me3, H3K27ac, H3K27me3, H3K9me3), ATAC-seq, DNase-seq | Whether the OTS locus sits in open/closed chromatin |
| **Branch 3 — 3D Genome** | `data/3dgenome/` | Hi-C TAD boundaries, chromatin loops, A/B sub-compartments | Long-range regulatory contact risk, structural fragility |
| **Branch 4 — Single-Cell** | `data/singlecell/` | scRNA-seq + scATAC-seq atlases for therapeutic cell types | Whether the OTS is active in the relevant subpopulation |
| **Cross-cutting** | `data/benchmarks/` | Baseline models (CRISPR-Net, DIPOFF, DNABERT, EpiCRISPR, …) | Comparison targets for AUROC / AUPRC / Recall@1%FPR |

The four branch embeddings are concatenated to a **576-dim joint vector** (128+256+64+128), passed through cross-attention between sequence↔epigenetic, then through multi-head outputs (cleavage probability, editing frequency, functional severity, uncertainty).

---

## 2. Dataset Inventory

### 2.1 `data/sequence/` — validated OTS + on-target activity (22 MB)

| File | Description | Blueprint role |
|---|---|---|
| `crispr_offt.h5` | **CRISPR-OFFT** model weights (Lin & Wong 2018) | Off-target baseline + training seed |
| `crispr_offt_README.md` | CRISPR-ONT/OFFT usage notes | — |
| `crisprSQL_100720.zip` | **crisprSQL** archive — 44 gRNAs, 25 632 guide-target pairs, epigenetic marker references | §5.1 secondary supervision + epigenetic feature alignment |
| `crisprSQL_extracted/100720.csv` | Extracted crisprSQL table | Training data for Branch 1+2 |
| `crisprSQL_README.md` | crisprSQL documentation | — |
| `crisprSQL_main.txt` | crisprSQL main data dump | — |
| `encoded_test_offt.txt`, `test_offt.csv`, `test_offt_batch.txt` | Test splits for off-target prediction | Cross-study evaluation |
| `test_ont.csv`, `test_ont_batch.csv` | On-target activity test splits | AUROC baselines |
| `ci2.fa`, `ci3.fa`, `jensenTest.fa` | Reference FASTA files for Cas-OFFinder | Candidate generation layer |

> **Missing (large / API-gated):** the live **CRISPRoffT (2025)** database (PMC12476096) and full GUIDE-seq / CHANGE-seq / CIRCLE-seq MAF-style tables were not directly downloadable as a single file. crisprSQL + CRISPR-OFFT together cover the supervised training labels required for Phase 1 of the roadmap (blueprint §7.1). For CRISPRoffT, query the authors or fetch via the publication's supplementary API.

### 2.2 `data/epigenetic/` — ENCODE histone/accessibility metadata (203 KB)

**Six cell types × six marks = 36 metadata JSONs**, plus a search script and summary:

| Cell type | Marks covered |
|---|---|
| **K562** (myelogenous leukemia — DNABERT-Epi, EpiCRISPR reference) | H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC-seq, DNase-seq |
| **HepG2** (hepatocellular carcinoma — liver target) | H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC-seq, DNase-seq |
| **CD14+ monocytes** (PBMC surrogate) | H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC-seq, DNase-seq |
| **CD34+ HSCs** (hematopoietic stem cells — sickle-cell / β-thalassemia clinical target) | H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC-seq, DNase-seq |
| **ARPE-19** (retinal pigment epithelium — retinal dystrophy clinical target) | H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC-seq, DNase-seq |
| **iPSC** (stem cell — many regenerative programs) | H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC-seq, DNase-seq |

Supporting files:
- `scripts/search_encode.py` — re-runnable ENCODE Portal query tool
- `logs/search.log` — full audit trail of ENCODE queries
- `metadata/celltypes_all.json` — flat index of every cell type×mark record
- `metadata/search_summary.json` — counts and per-mark file IDs

> **What this gives you:** the metadata/file IDs for every bigWig you need. The JSONs contain ENCODE Portal file accessions, biosample summaries, assembly, and download URLs. Run `scripts/search_encode.py` (or `curl` the URLs in the JSONs) to pull the actual `*.bigWig` signal tracks. Storing bigWigs locally is optional — DuckDB+`pyBigWig` lets you query them on-the-fly by genomic interval (blueprint §4.2 Branch 2).

### 2.3 `data/3dgenome/` — Hi-C / TAD / compartments (3.0 MB, all from Rao et al. 2014, GSE63525)

| File | Cell type | Content |
|---|---|---|
| `GSE63525_GM12878_HiCCUPS_looplist.txt.gz` | GM12878 | Chromatin loops (CCCTC-bound contacts) |
| `GSE63525_GM12878_TAD_domains.txt.gz` | GM12878 | Topologically associating domain calls |
| `GSE63525_GM12878_subcompartments.bed.gz` | GM12878 | A/B sub-compartment annotations |
| `GSE63525_K562_HiCCUPS_looplist.txt.gz` | K562 | Loops |
| `GSE63525_K562_TAD_domains.txt.gz` | K562 | TADs |
| `GSE63525_HUVEC_loops.txt.gz` | HUVEC | Loops |
| `GSE63525_HUVEC_TADs.txt.gz` | HUVEC | TADs |
| `GSE63525_IMR90_loops.txt.gz` | IMR90 (fibroblast) | Loops |
| `GSE63525_IMR90_TADs.txt.gz` | IMR90 | TADs |

Maps to blueprint §4.2 Branch 3 (12-feature structural vector: TAD distance, A/B compartment, contact frequency to on-target locus, CTCF proximity, LAD annotation).

### 2.4 `data/singlecell/` — scRNA-seq + scATAC-seq (81 MB)

| File | Cells | Modality | Source |
|---|---|---|---|
| `pbmc_10k_v3_filtered_feature_bc_matrix.h5` | ~10 000 PBMCs | scRNA-seq | 10x Genomics |
| `pbmc_1k_v3_filtered_feature_bc_matrix.h5` | ~1 000 PBMCs | scRNA-seq | 10x Genomics |
| `pbmc_3k_filtered_gene_bc_matrices.tar.gz` | ~3 000 PBMCs | scRNA-seq (mtx) | 10x Genomics |
| `atac_pbmc_5k_singlecell.csv` | ~5 000 PBMCs | scATAC-seq fragment/cell summary | 10x Genomics |
| `heart_1k_v3_filtered_feature_bc_matrix.h5` | ~1 000 heart cells | scRNA-seq | 10x Genomics |
| `GSE101498_RAW.tar` | Therapeutic tissue cohort | scRNA-seq raw | GEO (for sickle-cell/HSC atlas) |
| `tabula_sapiens_data.html` | Multi-tissue reference index | scRNA-seq catalogue | Tabula Sapiens (CZ Biohub) |

> **Note:** the high-priority therapeutic cell types (CD34+ HSCs, ARPE-19, iPSC, liver) are not all available as 10x-hosted processed matrices. The Tabula Sapiens HTML index provides a navigable cross-tissue entry point. The HCA (https://data.humancellatlas.org/) and CELLxGENE Discover (https://cellxgene.cziscience.com/) portals carry the production-grade atlases for the missing cell types — query by `ontology_label` (e.g. "hematopoietic stem cell", "retinal pigment epithelial cell") to fetch them via their `curl`-friendly download endpoints.

### 2.5 `data/benchmarks/` — baseline models + reference data (99 MB)

| Subfolder | What | Why we need it |
|---|---|---|
| `01_CRISPR-Net/` | Code + sample data | Sequence-only baseline (blueprint §6.3) |
| `02_CRISPR-DIPOFF/` | DIPOFF code | Interpretable RNN baseline (blueprint §6.3) |
| `03_DNABERT/` | DNABERT code + LICENSE | Foundation model for Branch 1 (blueprint §4.2 Branch 1) |
| `04_DNABERT-Epi/` | Epigenetic-integrated DNABERT code | Direct competitor (blueprint §6.3) |
| `05_CROP-IT/` | (README only — repo was private) | Cas9-ChIP+chromatin baseline |
| `06_EpiCRISPR/` | EpiCRISPR code | Flanking+epigenetic baseline |
| `07_CRISPR-MFH/` | CRISPR-MFH code | Lightweight multi-feature baseline |
| `08_Cas-OFFinder/` | Cas-OFFinder binary | Candidate generation engine (blueprint §4.1) |
| `09_Genome_Refs/` | `hg38.chrom.sizes` | Required by bedtools / pyBigWig / Cas-OFFinder |
| `10_Gencode/` | (empty — GENCODE GTF is ~1.4 GB; document the FTP URL in code instead) | Gene annotation for functional severity head (blueprint §4.3 Head 3) |
| `11_scEDIT/` | (README only — preprint supplementary not yet on GitHub) | scDNA-seq heterogeneity tool |
| `12_CRISPR-Embedding/` | k-mer embedding code | DNA-embedding baseline |

---

## 3. Quick-start Wiring (per blueprint §4.1 Pipeline)

```
gRNA (20nt) ─┐
             │ Cas-OFFinder ──► candidate OTS list ──► Feature Tensor Construction
Locus ───────┘                                            │
Cell type ───────────────────────────────────────────────  │
                                                           ▼
                                            ┌──────────────┴──────────────┐
                                            │   4-branch multimodal       │
                                            │   neural scorer             │
                                            │                              │
   Branch 1 (sequence)        ←─── data/sequence (crisprSQL, CRISPR-OFFT)
   Branch 2 (epigenetic)       ←─── data/epigenetic (ENCODE JSONs → bigWig)
   Branch 3 (3D genome)        ←─── data/3dgenome (Rao 2014 Hi-C, TADs)
   Branch 4 (single-cell)      ←─── data/singlecell (10x, Tabula, HCA)
                                            │                              │
                                            └──────────────┬──────────────┘
                                                           ▼
                                            Ranked OTS table + PDF/JSON
                                            with feature attribution
```

---

## 4. Coverage vs. Blueprint §5.1 Data Strategy Table

| Blueprint source | Status | Location |
|---|---|---|
| CRISPRoffT (2025) | Partial (CRISPR-OFFT + crisprSQL as proxy) | `data/sequence/` |
| crisprSQL | ✅ Full | `data/sequence/crisprSQL_100720.zip` |
| ENCODE Project | ✅ Metadata for 6 cell types × 6 marks | `data/epigenetic/metadata/` |
| 4D Nucleome / TCGA Hi-C | ✅ 4 cell types (Rao 2014) | `data/3dgenome/` |
| Human Cell Atlas / GEO | Partial (PBMC + heart, others via portal) | `data/singlecell/` |
| CRISPOR / Cas-OFFinder | ✅ Code | `data/benchmarks/08_Cas-OFFinder/`, `data/sequence/*.fa` |

---

## 5. What is *not* in this folder (and how to fetch)

- **CRISPRoffT 2025** — fetch via the publication supplementary or contact authors (https://pmc.ncbi.nlm.nih.gov/articles/PMC12476096/).
- **ENCODE bigWig signal files** — query the URLs in `data/epigenetic/metadata/*.json` via `curl`; store under `data/epigenetic/bigwig/` for DuckDB/pyBigWig access. Total is ~tens of GB; on-demand download is recommended.
- **GENCODE v44 GTF** — ~1.4 GB, fetch on demand: `https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38_mapping/gencode.v44.primary_assembly.annotation.gtf`.
- **HCA project atlases for CD34+ / ARPE-19 / iPSC / liver** — use the HCA DCP API: `https://service.azul.data.humancellatlas.org/` (filter by `ontology_label`), or CELLxGENE Discover.
- **Tabula Sapiens raw matrices** — fetch from `https://tabula-sapiens-portal.ds.czbiohub.org/`.

---

## 6. Storage summary

```
data/
├── 3dgenome/      3.0 MB   (9 files)
├── benchmarks/   99.0 MB   (~30 files across 12 subfolders)
├── epigenetic/    0.2 MB   (38 files — JSON metadata; bigWigs on demand)
├── sequence/     22.0 MB   (12 files + 1 extracted CSV)
├── singlecell/   81.0 MB   (6 processed matrices + 1 GEO tar + 1 index)
└── (raw/         reserved for on-demand bigWig/GTF/FASTQ downloads)
```

**Total on disk: ~205 MB** (vs. tens of GB if all bigWigs/GTF/FASTQs were staged locally — design choice favours metadata + on-demand fetching, per blueprint §4.2 Branch 2 implementation note).
