# CRISPR Atlas frontend

React + TypeScript + Vite interface for the existing FastAPI multimodal inference backend. Tailwind CSS provides responsive styling. It submits only user-entered DNA and features; it does not create or retrieve biological measurements.

## Run frontend and backend together

Keep both processes running in separate terminals from the repository root.

### Terminal 1: FastAPI backend

For the project Docker image, in PowerShell:

~~~powershell
docker run --rm -it --gpus all -p 8000:8000 -v "E:\AI_CRISPR-Project:/workspace/AI_CRISPR-Project" -w /workspace/AI_CRISPR-Project ghcr.io/hetakshib/ai-crispr:cuda bash -lc "python -m pip install -r backend/requirements.txt && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
~~~

Or use the local Python environment:

~~~powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
~~~

### Terminal 2: Vite frontend

~~~powershell
cd frontend
npm install
npm run dev
~~~

Open http://127.0.0.1:5173. The Vite development server proxies /api requests to http://127.0.0.1:8000, so no backend CORS changes are needed. Override the target with VITE_BACKEND_URL if FastAPI uses another host/port.

The backend loads the existing DNABERT cache and outputs/model_best.pt once at startup. The frontend displays readiness, checkpoint metadata, device, and parameter count from /health and /model-info.

## Input behavior

- The frontend exposes only **Missing Feature Demo** mode. The existing backend continues to support full multimodal requests; this presentation change does not alter backend behavior.
- The form accepts DNA containing A/C/G/T/N, ignores whitespace, requires at least six nucleotides, and sends a fixed 256-token input window.
- Every request omits epigenetic, 3D genome, and single-cell inputs and sends `mode: "missing_feature_demo"` with `allow_missing_features: true`. The user must explicitly consent before submission.
- The API zero-fills omitted modalities and reports them as unavailable. These placeholders are not biological measurements. Results are exploratory software demonstrations, not full multimodal predictions or experimentally validated biological results.
- Outputs are predicted cleavage frequency, high-efficiency probability, and binary classification at the backend's 0.5 threshold.
- Supplied sequence provenance is user-provided; the application does not infer coordinates or biological features from sequence.
## Build

~~~powershell
cd frontend
npm run build
~~~

Vite's development proxy is configured in vite.config.ts. A separately hosted production frontend will need an API reverse proxy or a backend CORS policy; this project leaves backend code unchanged.
