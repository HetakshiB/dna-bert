# Scientific Validation and Reporting Audit

Audit date: 2026-11-17  
Scope: read-only review of the frozen checkpoint, current tensors and splits, preprocessing, feature provenance, model dimensions, metrics, and benchmark reporting. No training, checkpoint writes, tensor writes, split regeneration, or preprocessing changes were performed. No additional model experiments were run.

## Executive findings

- The split is a deterministic random row split (seed 42), not grouped by sequence, genomic locus, guide, study, cell line, or experimental sample. Indices partition all 25,535 rows once and agree with metadata, but sequences and genomic intervals recur across partitions. The test results are not independent guide-, locus-, or study-held-out validation.
- Exact target-sequence groups shared across train/validation, train/test, and validation/test: 860, 835, and 189. These correspond to 5,970, 5,222, and 669 cross-split row-pair combinations. Reverse-complement target sequence pairs were also found (1, 2, and 0).
- Exact target genomic intervals recur across splits (assembly-aware comparison): 200, 231, and 54 shared interval groups; 330, 344, and 58 row-pair combinations. Overlapping target intervals: 799, 684, and 110 row pairs. This is confirmed repeated sequence/locus exposure; records may still represent different experimental observations.
- No experiment/sample identifier is available to decide whether repeats are technical or biological replicates. The row id is unique but is not a sample identifier. The same 17 studies and 7 non-missing cell-line categories occur in all splits.
- Tensor dimensions match the architecture, but three epigenetic dimensions are all-zero, ATAC duplicates the DNase input, and GM12878 3D features are applied without checking assembly/species compatibility.
- The full-test metrics are tied to checkpoint and tensor hashes, but independent generalization is not established because of row-level sequence/locus/source overlap. Literature baselines are references, not reproduced head-to-head comparisons.

## Frozen evaluation identity

Metrics report: outputs/metrics.json  
Checkpoint: outputs/model_best.pt  
Test tensor: outputs/tensors/test.pt  
Samples: 2,554; sequence length: 256; batch size: 2; scope: full_test.

Checkpoint SHA256: a12654b360e959a3ee6b9eab47f7a463152dcf813a90ec1e84b9f7f6260b4558  
Test tensor SHA256: 4a6ff663e0556cff4d4427531a09b9abf7f01d8b5d64510a0ae369bbcf6f6fb8

Reported metrics: RMSE 0.108988; Pearson r 0.752398; ROC-AUC 0.916372; PR-AUC 0.814999; F1 at 0.5 0.695355; Brier score 0.118236. No metrics were recomputed for this audit.

## Phase 1 — Split and leakage audit

### Split construction and integrity

The preprocessing code first calls train_test_split with test_size=0.20 and random_state=42 without stratification to select train versus held-out rows. It then splits the held-out 20% in half (seed 42), stratifying on the binary label when class counts allow. This is a random row split, not a grouped or chromosome-based split.

Observed sizes: train 20,428; validation 2,553; test 2,554. The split index arrays contain 25,535 unique indices, partition every row exactly once, and agree with the metadata split column.

### Sequence, guide, identifier, and context overlap

Method: uppercase exact string sets and multiplicity-weighted row-pair intersections for each split pair. Reverse complements use A↔T and C↔G (N preserved). Pair totals multiply duplicate counts; they are not independent experiment counts.

| Field | Train–val | Train–test | Val–test |
|---|---:|---:|---:|
| Exact target_sequence distinct strings shared | 860 | 835 | 189 |
| Target-sequence cross-split row pairs | 5,970 | 5,222 | 669 |
| Reverse-complement target sequence pairs | 1 | 2 | 0 |
| Exact grna_target_sequence distinct strings shared | 57 | 69 | 49 |
| Reverse-complement guide sequence pairs | 1 | 2 | 1 |
| Exact target_context distinct strings shared | 225 | 265 | 59 |
| Target-context cross-split row pairs | 408 | 430 | 67 |

Multiplicity-weighted guide-sequence row-pair totals are 9,028,459; 9,114,486; and 1,115,480 because guide sequences are reused across many target records. These indicate guide-level sharing, not millions of unique leaked observations.

The id field has 25,535 unique values and no cross-split overlap; it is row-level. No guide_id, experiment_id, replicate_id, or biological sample_id exists. target_geneid is gene-level, missing in 18,495 rows, and overlaps by 348, 375, and 104 distinct genes across split pairs. Shared target-sequence groups can have differing class labels (169 train/val, 146 train/test, 38 val/test), so repeated sequence groups are not all duplicate labels or identical observations.

### Genomic interval overlap

Method: intervals were compared only within the same genome assembly label and chromosome, using half-open interval overlap. Coordinates are available for 11,355 rows; 14,180 rows lack coordinate/genome fields. Thirteen target-locus and 27 guide-target intervals have nonpositive width; exact-interval counts below exclude those invalid-width intervals, and half-open overlap tests naturally do not count them.

| Coordinate field | Pair | Exact shared intervals | Exact row pairs | All overlapping interval row pairs |
|---|---|---:|---:|---:|
| Target locus | train–val | 200 | 330 | 799 |
| Target locus | train–test | 231 | 344 | 684 |
| Target locus | val–test | 54 | 58 | 110 |
| Guide target locus | train–val | 71 | 928,109 | 1,161,290 |
| Guide target locus | train–test | 78 | 864,115 | 1,065,980 |
| Guide target locus | val–test | 60 | 113,495 | 139,305 |

Guide-target coordinate totals are multiplicity-weighted because the same guide locus recurs across records. They establish locus reuse, not independent events. Non-overlap cannot be established for rows without coordinate metadata.

### Study and biological-source sharing

All 17 non-missing study_name values occur in each split. The same 7 non-missing cell_line categories occur across split pairs, and all 5 non-missing genome labels occur in each split. These prove source-category sharing, not reuse of the same physical sample. There are no sample/experiment identifiers, so sample-level independence is unknown; missing IDs are not evidence of no leakage.

### Preprocessing, labels, and target-derived features

- high_eff is cleavage_freq greater than the training split's 75th percentile. Recalculation gives 0.001862 and reproduces all stored binary labels. The threshold is not selected from validation/test labels.
- feature_scaler.json declares fit_split=train; the code computes epigenetic mean/std from training rows only. Missing marks get mean 0/std 1. No test-fitted feature normalization was found.
- The model input construction inspected does not use cleavage_freq or high_eff as features. Those are intended supervised labels (reg_label, cls_label). energy_*, study, gene, and coordinate metadata are not in saved model examples.
- Signal, coordinate-reference, and expression lookup are per row or external-reference based. No direct label-derived input feature was identified. This cannot rule out source-level confounding or biological replicate overlap.

## Phase 2 — Feature and modality audit

Tensor feature order is H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC-seq, DNase-seq.

| Feature | Evidence and interpretation |
|---|---|
| H3K4me3 | CSV epigen_h3k4me3 fallback; 1,861/25,535 raw rows nonzero. No local ENCODE BigWig was present, so assay provenance was not independently validated. |
| H3K27ac | No fallback or local BigWig; tensor dimension is zero for every row. Unavailable/zero-filled. |
| H3K27me3 | No fallback or local BigWig; tensor dimension is zero for every row. Unavailable/zero-filled. |
| H3K9me3 | No fallback or local BigWig; tensor dimension is zero for every row. Unavailable/zero-filled. |
| DNase-seq | CSV epigen_dnase fallback; 493/25,535 raw rows nonzero. Not independently validated against a local track. |
| ATAC-seq | Uses the same epigen_dnase CSV field as DNase. It is a proxy, not an independent ATAC measurement; standardized ATAC and DNase tensors are exactly equal. |

No local BigWigs were present. The three missing histone dimensions have no missingness masks; the model receives numeric zeros, indistinguishable from measured zero. This applies to other unavailable/zero-valued features as well.

The epigenetic scaler is declared train-only in the code and metadata; missing marks are assigned mean 0/std 1. The model has no missingness mask. Other transforms are fixed source transformations rather than dataset-fitted scaling.

### 3D genome

genome_vec has 8 values: log1p loop count, capped log10 distance to nearest TAD boundary, and a six-way A1/A2/B1/B2/B3/NA one-hot. The tensor and model dimensions agree. The compartment one-hot sums to one per row.

The source is GM12878 Hi-C/TAD/compartment data used as a proxy across cell lines (23 loop chromosomes, 23 TAD chromosomes, 22 compartment chromosomes). The extractor uses target chromosome and coordinates without checking the metadata genome assembly. Data include hg19, hg38, mm10, rn5, and mm9 (and 14,180 rows with no genome label). Applying a single human reference across these builds/species without lift-over or validity masking is a confirmed compatibility risk.

Both continuous 3D values are zero for 14,194 rows. This can mean no loop, zero TAD distance, or absent/mismatched coverage; there is no mask to distinguish these. The NA compartment is explicit, but the continuous channels have no availability mask.

### Single-cell expression

The code loads a PBMC 10x 10k atlas, normalizes/log-transforms it, averages expression across cells, and matches target_geneid to atlas gene names. Metadata records 5,502/25,535 matched rows; the saved tensor is nonzero for 4,892 rows. Unmatched rows are zero; some matched gene means can also be zero. The model has no missingness mask.

This is one PBMC-wide mean, not target-cell-type- or subpopulation-specific expression. Gene profiles recur across split rows. The dataset includes non-human genome labels, while the code maps gene symbols against the human PBMC atlas without species validation; cross-species symbol collisions cannot be excluded.

### Dimension contract

Observed tensor shapes: seq_ids/seq_mask [N,512], epigen_vec [N,6], genome_vec [N,8], sc_expr_vec [N,1], scalar reg_label and scalar cls_label. Model contracts: DNABERT CLS 768→256; epigenetic 6→64; genome 8→128; expression 1→128; concatenation 576→fusion 256; scalar regression and classification heads. Shapes are compatible; biological provenance/availability concerns remain.

## Benchmark and research-reporting audit

metrics.json includes full-test provenance and checkpoint/tensor hashes. compare_benchmarks.py reads canonical full-test metrics, labels the comparison descriptive, warns that baselines were not independently run, and no longer emits target_met. Hardcoded baseline values are literature-reported references. Their dataset, split, prevalence, threshold, and metric protocols are not harmonized in the comparison output, so the numeric differences are not head-to-head evidence.

The dashboard reports current evaluation scope and labels baselines as references. Current metrics provenance records 2,554 quantitative samples, 50 IG samples, and 50 SHAP explained samples. Attribution summaries are exploratory; three epigenetic features are all zero and ATAC duplicates DNase.

The current AUROC must be described as performance on this row-random test partition, not as unseen-guide/locus/study/sample performance. No confidence intervals, grouped cross-validation, held-out-study test, independent external test, or modality ablation is in the inspected results. Multimodal contribution and between-study generalization are unestablished. Baseline F1 values may use different thresholds/protocols and are not comparative evidence.

## Audit limits and conclusion

This audit establishes split construction, observed cross-split overlap, tensor dimensions, stored availability, and the inspected code paths. It cannot determine whether repeated records are biological or technical replicates, recover omitted sample relationships, validate raw CSV assays, or establish generalization to a new study/build. No expensive experiments were run and no remediation was made because the data and preprocessing are frozen by instruction.

Report these results as performance on the current row-random test partition, disclose sequence/locus/source overlap and incomplete/proxy modalities, and avoid claims of independent biological validation or superiority over literature baselines from these results alone.
