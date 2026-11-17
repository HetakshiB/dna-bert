# Live demo sample selection

No eligible complete multimodal examples were produced. The dataset provenance marks H3K27ac, H3K27me3, and H3K9me3 unavailable; these channels are zero-filled in preprocessing. ATAC-seq reuses the DNase-seq signal as a proxy. The API requires six epigenetic values and treats a supplied six-value vector as available, so sending those placeholders would misrepresent missing measurements. No predictions or experimental outcomes are fabricated.

Use the explicitly consented missing_feature_demo mode only for a clearly labeled incomplete-feature demonstration; it does not support partial missingness inside an epigenetic vector. To produce complete samples, obtain the missing mark measurements and update the data with a separately versioned, scientifically reviewed workflow.
