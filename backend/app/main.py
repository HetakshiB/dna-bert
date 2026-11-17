from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI,HTTPException,Request
from backend.app.schemas import PredictionRequest,PredictionResponse
from backend.app.service import load_predictor
logging.basicConfig(level=logging.INFO); logger=logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app:FastAPI):
 app.state.predictor=None; app.state.load_error=None
 try: app.state.predictor=load_predictor()
 except Exception as e: app.state.load_error=f"{type(e).__name__}: {e}"; logger.exception("Model startup failed")
 yield
 app.state.predictor=None
app=FastAPI(title="AI CRISPR Prediction API",version="1.0.0",description="Inference with the existing multimodal checkpoint.",lifespan=lifespan)
def ready(req:Request)->Any:
 p=getattr(req.app.state,"predictor",None)
 if p is None: raise HTTPException(503,detail=f"Model is not ready: {getattr(req.app.state,'load_error',None) or 'loading'}")
 return p
@app.get("/health")
def health(req:Request)->dict:
 p=getattr(req.app.state,"predictor",None)
 return {"status":"ready" if p else "not_ready","device":p.model_info["device"] if p else None,"error":getattr(req.app.state,"load_error",None)}
@app.get("/model-info")
def model_info(req:Request)->dict:
 p=ready(req); return {**p.model_info,"status":"loaded","epigenetic_scaler_available":p.epigenetic_scaler is not None}
@app.post("/predict",response_model=PredictionResponse)
def predict(body:PredictionRequest,req:Request)->dict:
 p=ready(req)
 try: return p.predict(body)
 except ValueError as e: raise HTTPException(422,detail=str(e)) from e
 except Exception as e:
  logger.exception("Inference failed"); raise HTTPException(500,detail="Model inference failed; inspect API logs") from e
