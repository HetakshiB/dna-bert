# Demo validation report

Status: BLOCKED — no eligible sample.

Selection rule: held-out test records only, valid A/C/G/T/N sequence, all six epigenetic channels measured, record-supported 3D features, and matched single-cell expression. The project provenance reports H3K27ac, H3K27me3, and H3K9me3 unavailable; all three are placeholders. It also reports DNase-seq reused for ATAC-seq. Thus no test row can meet the complete-measured-epigenetics requirement.

No direct inference or FastAPI request was run. There are no API/direct output comparisons. Checkpoint outputs/model_best.pt, tensor outputs/tensors/test.pt, and existing reports were not modified. Labels are cleavage_freq from crisprSQL; high_eff is a dataset-derived annotation using a training-split 75th-percentile threshold, not an independently verified experimental binary outcome.

Next step: acquire the missing biological measurements and use a reviewed, versioned dataset/API representation that records per-feature availability, or demonstrate only via the existing explicit missing-feature mode with its warning.
