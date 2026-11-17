# Genomic overlap and sequence similarity audit

## Scope

- Compared 20,428 training records with 1,612 retained test records.
- Screening only; no training, inference, model changes, or existing artifact writes. The test rows remain from the original random row split and this is not an independent evaluation.

## Why exact guide-target intervals remain after sequence filtering

The prior filter removes rows based on target_sequence (the modeled target/off-target sequence). The grna_target_* columns identify the guide RNA target locus. These are distinct entities: target and guide-target loci can be on different chromosomes, and many distinct target observations can share the same guide target. Repeated guide-target coordinates therefore do not imply exact target_sequence duplication. They do expose guide/locus grouping across splits and potential row dependence.

## Coordinate overlap method

Coordinates were compared only when assembly/genome label and chromosome matched after trim/case normalization. Intervals were treated as half-open [start,end): overlap iff max(starts) < min(ends), overlap bp = min(ends)-max(starts); exact interval means identical boundaries. No liftover or chromosome alias normalization was done. Preprocessing BigWig queries use half-open intervals, but the source CSV convention is undocumented and not verified.

### target interval

- Valid training intervals: 9,026; valid retained test intervals: 709.
- Retained test rows overlapping training: 15; pairwise training-record overlaps: 50.
- Test rows with an exact interval match: 0; exact interval pairs: 0.
- Test rows lacking required coordinate/assembly fields: 903.


### guide target interval

- Valid training intervals: 9,018; valid retained test intervals: 705.
- Retained test rows overlapping training: 705; pairwise training-record overlaps: 623,988.
- Test rows with an exact interval match: 705; exact interval pairs: 535,430.
- Test rows lacking required coordinate/assembly fields: 903.

Pair counts include multiple training records at repeated coordinates and do not represent distinct genomic sites. The sample CSV has one row per overlapping retained test sample and interval type, with overlap counts and example training IDs.

## Sequence similarity

Sequences were compared to all training target sequences in forward and reverse-complement orientation. Identity is ungapped, left-aligned positional matches divided by the longer sequence length; extra trailing bases count as mismatches. This is a screening score, not an edit-distance or local alignment. Exact/RC match counts remaining after filtering were 0 / 0.

| Minimum identity | Test rows | Fraction |
|---:|---:|---:|
| 70% | 1,605 | 99.6% |
| 80% | 1,448 | 89.8% |
| 90% | 1,087 | 67.4% |
| 95% | 962 | 59.7% |
| 100% | 0 | 0.0% |

Close sequence families remain common. A full alignment-aware similarity check should precede sequence-independence claims.

## Limitations

- Recorded genome labels are heterogeneous: hg19 (9,324), hg38 (1,630), mm10 (324), rn5 (43), mm9 (34); they include hg19/hg38, mouse labels mm10/mm9, rat rn5, and missing values. References were not independently verified.
- Coordinate-system conventions, chromosome alias handling, and assembly provenance are incompletely documented; missing data prevent checking all records.
- Sample IDs, replicate IDs, and stable experiment IDs that establish biological independence are unavailable. Study and cell-line labels exist but cannot establish replicate/source independence. This is NOT VERIFIABLE.
- Exact sequence, reverse-complement, approximate sequence identity, target-locus overlap, and guide-target-locus overlap are distinct checks. Neither the sequence filter nor this coordinate audit establishes independence.

## Recommendation

Grouped retraining is scientifically necessary before claiming generalization to unseen guides, genomic loci, or experiments. The random row split contains high sequence similarity and repeated guide-target loci across splits, while biological sample independence cannot be verified. A future evaluation should group by guide/target identity, locus, and study/experiment/sample where metadata supports it, then retrain on that new split. No such experiment was run.
