from __future__ import annotations
import math
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,field_validator,model_validator
class PredictionRequest(BaseModel):
 model_config=ConfigDict(extra="forbid")
 sequence:str=Field(min_length=1,max_length=10000)
 mode:Literal["full_multimodal","missing_feature_demo"]="full_multimodal"
 epigenetic_features:list[float]|None=Field(default=None,min_length=6,max_length=6)
 genome_features:list[float]|None=Field(default=None,min_length=8,max_length=8)
 single_cell_expression:float|None=None
 allow_missing_features:bool=False
 @field_validator("sequence")
 @classmethod
 def valid_sequence(cls,v):
  v=v.strip().upper()
  if not v: raise ValueError("sequence must not be empty")
  bad=sorted(set(v)-set("ACGTN"))
  if bad: raise ValueError(f"unsupported DNA characters: {', '.join(bad)}")
  if len(v)<6: raise ValueError("sequence must contain at least 6 nucleotides for DNABERT-6")
  return v
 @field_validator("epigenetic_features","genome_features")
 @classmethod
 def finite_vectors(cls,v):
  if v is not None and not all(math.isfinite(x) for x in v): raise ValueError("feature values must be finite")
  return v
 @field_validator("single_cell_expression")
 @classmethod
 def finite_expression(cls,v):
  if v is not None and not math.isfinite(v): raise ValueError("expression must be finite")
  return v
 @model_validator(mode="after")
 def check_mode(self):
  supplied=(self.epigenetic_features is not None,self.genome_features is not None,self.single_cell_expression is not None)
  if self.mode=="full_multimodal":
   if not all(supplied): raise ValueError("full_multimodal requires all three feature inputs")
   if self.allow_missing_features: raise ValueError("consent flag only applies in missing_feature_demo mode")
  else:
   if not self.allow_missing_features: raise ValueError("missing_feature_demo requires allow_missing_features=true")
   if all(supplied): raise ValueError("all modalities supplied; use full_multimodal mode")
  return self
class FeatureStatus(BaseModel):
 available:bool
 provenance:str
class ModelInfo(BaseModel):
 architecture:str
 pretrained_backbone:str
 checkpoint_path:str
 checkpoint_sha256:str
 parameter_count:int
 device:str
 sequence_length_tokens:int
class PredictionResponse(BaseModel):
 input_mode:Literal["full_multimodal","missing_feature_demo"]
 predicted_cleavage_frequency:float
 high_efficiency_probability:float
 high_efficiency_prediction:bool
 sequence_nucleotides:int
 sequence_tokens_used:int
 sequence_length_tokens:int
 model:ModelInfo
 feature_availability:dict[str,FeatureStatus]
 warning:str|None=None
