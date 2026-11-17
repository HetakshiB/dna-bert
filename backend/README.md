# AI CRISPR FastAPI backend

Serves the existing multimodal checkpoint. DNABERT-6 and outputs/model_best.pt load once at startup; inference runs in eval/inference mode and uses 256 tokens to match evaluation. No training or substitute-weight download occurs.

## Windows PowerShell

From project root, in a Python environment with project PyTorch/Transformers installed:

~~~powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
~~~

Open http://127.0.0.1:8000/docs.

## Docker PowerShell

~~~powershell
docker run --rm -it --gpus all -p 8000:8000 -v "E:\AI_CRISPR-Project:/workspace/AI_CRISPR-Project" -w /workspace/AI_CRISPR-Project ghcr.io/hetakshib/ai-crispr:cuda bash -lc "python -m pip install -r backend/requirements.txt && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
~~~

Project/model files remain on the E: mount. CUDA is selected when available, otherwise CPU. Startup requires a complete local DNABERT cache and refuses incomplete pretrained weights.

## Endpoints

- GET /health - readiness, device, or startup error.
- GET /model-info - architecture, backbone id, checkpoint relative path/SHA256, parameter count, device, token length, scaler availability.
- POST /predict - validated DNA and multimodal inputs. Invalid requests return 422, model-not-ready returns 503, unexpected inference errors return 500 and are logged.

## Full multimodal request

Six raw epigenetic inputs are ordered H3K4me3, H3K27ac, H3K27me3, H3K9me3, ATAC, DNase. Eight genome inputs are loop count, TAD distance in bp, then six compartment indicators. Expression must use the preprocessing log-normalized scale.

~~~json
{"sequence":"ACGTACGTACGTACGTACGTACGT","mode":"full_multimodal","epigenetic_features":[0.1,0,0,0,0.4,0.2],"genome_features":[2,1500,1,0,0,0,0,0],"single_cell_expression":1.25}
~~~

PowerShell request:

~~~powershell
$body = @{ sequence = "ACGTACGTACGTACGTACGTACGT"; mode = "full_multimodal"; epigenetic_features = @(0.1,0,0,0,0.4,0.2); genome_features = @(2,1500,1,0,0,0,0,0); single_cell_expression = 1.25 } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
~~~

## Explicit missing-feature demonstration

Omit unavailable modalities and set mode to missing_feature_demo and allow_missing_features to true. Omitted values become zero placeholders, are labelled missing in response provenance, and trigger a warning; they are not biological measurements.

~~~json
{"sequence":"ACGTACGTACGTACGTACGTACGT","mode":"missing_feature_demo","allow_missing_features":true}
~~~

The response contains predicted cleavage frequency, high-efficiency probability, binary prediction at 0.5, sequence/token lengths, model/checkpoint info, modality availability/provenance, and an incomplete-feature warning. The API cannot infer measurements or genomic coordinates from sequence alone.

## CPU-only tests

From project root:

~~~powershell
python -m unittest backend.tests.test_api backend.tests.test_service -v
~~~

Tests mock model loading/inference and do not load the checkpoint, require CUDA, or retrain.
