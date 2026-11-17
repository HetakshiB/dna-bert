from __future__ import annotations
import hashlib,json,logging,math,threading
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/"outputs"/"model_best.pt"
SCALER_PATH=ROOT/"outputs"/"tensors"/"feature_scaler.json"
SEQ_LEN=256
log=logging.getLogger(__name__)
class Predictor:
 def __init__(self,tokenizer:Any,model:Any,device:torch.device,info:dict,scaler:dict|None):
  self.tokenizer=tokenizer; self.model=model.eval(); self.device=device; self.model_info=info; self.epigenetic_scaler=scaler; self.lock=threading.Lock()
 def predict(self,req:Any)->dict:
  from model.backbone import seq_to_kmers
  missing=[]; status={"sequence":{"available":True,"provenance":"user sequence encoded with cached DNABERT-6 tokenizer"}}
  if req.epigenetic_features is None:
   epi=torch.zeros(1,6); missing.append("epigenetic_features"); status["epigenetic"]={"available":False,"provenance":"missing; zero placeholder, not measured"}
  else:
   if self.epigenetic_scaler is None: raise RuntimeError("training feature scaler unavailable for epigenetic inputs")
   mean=torch.tensor(self.epigenetic_scaler["epigenetic_mean"],dtype=torch.float32); std=torch.tensor(self.epigenetic_scaler["epigenetic_std"],dtype=torch.float32)
   epi=(torch.tensor([req.epigenetic_features],dtype=torch.float32)-mean)/std
   status["epigenetic"]={"available":True,"provenance":"user-supplied raw values transformed with training-split scaler"}
  if req.genome_features is None:
   genome=torch.zeros(1,8); missing.append("genome_features"); status["3d_genome"]={"available":False,"provenance":"missing; zero placeholder, not measured"}
  else:
   v=req.genome_features
   if v[0]<0 or v[1]<0: raise ValueError("raw loop count and TAD distance cannot be negative")
   genome=torch.tensor([v],dtype=torch.float32); genome[0,0]=math.log1p(v[0]); genome[0,1]=min(math.log10(v[1]+1),8.0)
   status["3d_genome"]={"available":True,"provenance":"user-supplied; loop count and TAD distance transformed as in preprocessing"}
  if req.single_cell_expression is None:
   expr=torch.zeros(1,1); missing.append("single_cell_expression"); status["singlecell_expression"]={"available":False,"provenance":"missing; zero placeholder, not measured"}
  else:
   expr=torch.tensor([[req.single_cell_expression]],dtype=torch.float32); status["singlecell_expression"]={"available":True,"provenance":"user-supplied; expected in preprocessing log-normalized scale"}
  toks=self.tokenizer(seq_to_kmers(req.sequence),return_tensors="pt",truncation=True,max_length=SEQ_LEN,padding="max_length")
  ids=toks["input_ids"].to(self.device); mask=toks["attention_mask"].to(self.device)
  with self.lock,torch.inference_mode():
   reg,cls,_=self.model(input_ids=ids,attention_mask=mask,epigen_vec=epi.to(self.device),genome_vec=genome.to(self.device),sc_expr_vec=expr.to(self.device))
   freq=float(reg.reshape(-1)[0].item()); prob=float(torch.sigmoid(cls).reshape(-1)[0].item())
  warning=None if not missing else "Incomplete-feature demonstration: missing modalities were zero-filled with explicit consent; these are not biological measurements."
  return {"input_mode":req.mode,"predicted_cleavage_frequency":freq,"high_efficiency_probability":prob,"high_efficiency_prediction":prob>=.5,"sequence_nucleotides":len(req.sequence),"sequence_tokens_used":int(mask.sum().item()),"sequence_length_tokens":SEQ_LEN,"model":self.model_info,"feature_availability":status,"warning":warning}
def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
 return h.hexdigest()
def load_predictor(checkpoint:Path=CHECKPOINT)->Predictor:
 from model.backbone import CACHE_DIR,HF_MODEL_ID,_cache_is_complete,get_dnabert
 from model.multimodal import MultimodalCRISPR
 if not checkpoint.is_file(): raise FileNotFoundError(f"checkpoint not found: {checkpoint.relative_to(ROOT)}")
 if not _cache_is_complete(CACHE_DIR): raise RuntimeError(f"DNABERT cache incomplete at {CACHE_DIR}; restore pretrained weights and tokenizer")
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 try: tokenizer,backbone=get_dnabert(freeze=True,device=device)
 except Exception as e: raise RuntimeError(f"DNABERT loading failed: {e}") from e
 model=MultimodalCRISPR(dnabert_model=backbone).to(device)
 try:
  payload=torch.load(checkpoint,map_location="cpu",weights_only=False); state=payload["model_state"] if isinstance(payload,dict) and "model_state" in payload else payload
  model.load_state_dict(state,strict=True)
 except Exception as e: raise RuntimeError(f"checkpoint loading failed: {e}") from e
 finally:
  if "payload" in locals(): del payload
 model.eval(); scaler=None
 if SCALER_PATH.is_file():
  try:
   scaler=json.loads(SCALER_PATH.read_text(encoding="utf-8"))
   if len(scaler["epigenetic_mean"])!=6 or len(scaler["epigenetic_std"])!=6: raise ValueError("expected six scaler values")
  except Exception as e: raise RuntimeError(f"epigenetic scaler load failed: {e}") from e
 info={"architecture":"MultimodalCRISPR (DNABERT-6 + epigenetic + 3D genome + single-cell)","pretrained_backbone":HF_MODEL_ID,"checkpoint_path":checkpoint.resolve().relative_to(ROOT).as_posix(),"checkpoint_sha256":sha256(checkpoint),"parameter_count":sum(p.numel() for p in model.parameters()),"device":str(device),"sequence_length_tokens":SEQ_LEN}
 log.info("Loaded %s on %s",info["checkpoint_path"],device)
 return Predictor(tokenizer,model,device,info,scaler)
