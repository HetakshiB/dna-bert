# Sequence-only missing-feature live demo

## How to present this demonstration

These five sequences are valid DNA target sequences drawn from the saved held-out test split. No training or validation records were used. The request contains only the DNA sequence and explicit consent for `missing_feature_demo`. The application zero-fills the omitted epigenetic, 3D genome, and single-cell inputs. These are **not full multimodal predictions**, are **not experimentally validated results**, and must not be presented as such.

The displayed cleavage frequency and classification probability are model predictions. The comparison labels are dataset-provided annotations: project provenance says `cleavage_freq` comes from crisprSQL and `high_eff` is derived using a training-split 75th-percentile threshold. The available project documentation does not establish that each value is an independently verified experimental measurement. Comparing predictions to those labels under missing-feature mode is exploratory and is not the original full-model evaluation.

## Samples and results

### Sample 1 — test position 491 (source row 23340, ID 23437)
- DNA sequence: `GTGATAAGTGGAATAGCACATGTGG` (25 nt).
- Dataset annotations: cleavage frequency `5.7966937e-21`, high-efficiency label `0`.
- Model output: cleavage frequency `-0.0075573381`; high-efficiency probability `0.0500%`; predicted class `0`.
- Exploratory comparison: absolute error `0.0075573381`; classification matches the dataset label.
- Inputs: sequence only. Epigenetic, 3D genome, and single-cell modalities were omitted and zero-filled by the application; these are not measured inputs.

### Sample 2 — test position 1452 (source row 15107, ID 15204)
- DNA sequence: `GACGCCATAAAGCATGAGACGCTGG` (25 nt).
- Dataset annotations: cleavage frequency `0.00011278859`, high-efficiency label `0`.
- Model output: cleavage frequency `-0.0090262899`; high-efficiency probability `0.3118%`; predicted class `0`.
- Exploratory comparison: absolute error `0.0091390785`; classification matches the dataset label.
- Inputs: sequence only. Epigenetic, 3D genome, and single-cell modalities were omitted and zero-filled by the application; these are not measured inputs.

### Sample 3 — test position 1409 (source row 7898, ID 7992)
- DNA sequence: `GCAGGAGGGGAATTTGCTCTGGG` (23 nt).
- Dataset annotations: cleavage frequency `0.001121`, high-efficiency label `0`.
- Model output: cleavage frequency `0.039020859`; high-efficiency probability `88.2425%`; predicted class `1`.
- Exploratory comparison: absolute error `0.037899859`; classification does not match the dataset label.
- Inputs: sequence only. Epigenetic, 3D genome, and single-cell modalities were omitted and zero-filled by the application; these are not measured inputs.

### Sample 4 — test position 873 (source row 8293, ID 8387)
- DNA sequence: `AATTCCAAGCAGAAGAAAAAGGA` (23 nt).
- Dataset annotations: cleavage frequency `0.003303`, high-efficiency label `1`.
- Model output: cleavage frequency `0.068956532`; high-efficiency probability `96.3914%`; predicted class `1`.
- Exploratory comparison: absolute error `0.065653532`; classification matches the dataset label.
- Inputs: sequence only. Epigenetic, 3D genome, and single-cell modalities were omitted and zero-filled by the application; these are not measured inputs.

### Sample 5 — test position 1766 (source row 993, ID 1006)
- DNA sequence: `GACCCCCCCCACCCCGCCCCCGG` (23 nt).
- Dataset annotations: cleavage frequency `0.041`, high-efficiency label `1`.
- Model output: cleavage frequency `0.070736527`; high-efficiency probability `94.4360%`; predicted class `1`.
- Exploratory comparison: absolute error `0.029736527`; classification matches the dataset label.
- Inputs: sequence only. Epigenetic, 3D genome, and single-cell modalities were omitted and zero-filled by the application; these are not measured inputs.

## Validation

All five requests were sent to the running FastAPI `/predict` endpoint on CUDA and compared with direct calls to the same application `Predictor` implementation and checkpoint. Maximum absolute difference across cleavage-frequency predictions and probabilities: `0`. Classifications agreed for all five. Checkpoint and test-tensor hashes were unchanged before and after inference.

## Reproduce tomorrow

1. Start the backend using the Docker PowerShell command in `backend/README.md` (port 8000).
2. Run from the project root in PowerShell:

```powershell
docker exec -i <running-backend-container-name> python /workspace/AI_CRISPR-Project/outputs/live_demo_samples/run_missing_demo.py --api-url http://127.0.0.1:8000
```

The script verifies the frozen checkpoint and test tensor hashes, selects from saved held-out test indices, sends the sequence-only requests, checks direct/API agreement, then refreshes only the three `missing_demo_*` files in this directory.
