import unittest
from types import SimpleNamespace
import torch
from backend.app.schemas import PredictionRequest
from backend.app.service import Predictor
class FakeTokenizer:
 def __call__(self,text,**kwargs):
  self.text=text; self.kwargs=kwargs
  return {"input_ids":torch.ones(1,256,dtype=torch.long),"attention_mask":torch.ones(1,256,dtype=torch.long)}
class FakeModel:
 def __init__(self): self.was_eval=False; self.inputs=None
 def eval(self): self.was_eval=True; return self
 def __call__(self,**kwargs):
  self.inputs=kwargs
  return torch.tensor([[0.3]]),torch.tensor([[0.0]]),torch.zeros(1,256)
class PredictorTests(unittest.TestCase):
 def setUp(self):
  self.tok=FakeTokenizer(); self.model=FakeModel(); self.predictor=Predictor(self.tok,self.model,torch.device("cpu"),{"checkpoint_path":"outputs/model_best.pt","sequence_length_tokens":256},{"epigenetic_mean":[0]*6,"epigenetic_std":[1]*6})
 def test_full_inputs_tokenization_and_transforms(self):
  req=PredictionRequest(sequence="acgtacgt",mode="full_multimodal",epigenetic_features=[1]*6,genome_features=[1,999,1,0,0,0,0,0],single_cell_expression=.75)
  result=self.predictor.predict(req)
  self.assertTrue(self.model.was_eval); self.assertEqual(self.tok.kwargs["max_length"],256); self.assertEqual(result["sequence_length_tokens"],256)
  self.assertEqual(tuple(self.model.inputs["epigen_vec"].shape),(1,6)); self.assertEqual(tuple(self.model.inputs["genome_vec"].shape),(1,8)); self.assertEqual(tuple(self.model.inputs["sc_expr_vec"].shape),(1,1))
  self.assertAlmostEqual(float(self.model.inputs["genome_vec"][0,0]),torch.log(torch.tensor(2.)).item(),places=6)
  self.assertTrue(result["feature_availability"]["epigenetic"]["available"]); self.assertIsNone(result["warning"])
 def test_missing_features_are_explicit_zero_placeholders(self):
  req=PredictionRequest(sequence="ACGTACGT",mode="missing_feature_demo",allow_missing_features=True)
  result=self.predictor.predict(req)
  self.assertTrue(torch.equal(self.model.inputs["epigen_vec"],torch.zeros(1,6))); self.assertTrue(torch.equal(self.model.inputs["genome_vec"],torch.zeros(1,8)))
  self.assertFalse(result["feature_availability"]["epigenetic"]["available"]); self.assertIn("not biological measurements",result["warning"])
if __name__=="__main__": unittest.main(verbosity=2)
