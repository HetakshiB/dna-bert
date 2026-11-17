import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app import main
class FakePredictor:
 def __init__(self):
  self.calls=0; self.model_info={"architecture":"mock","pretrained_backbone":"mock/DNABERT-6","checkpoint_path":"outputs/model_best.pt","checkpoint_sha256":"0"*64,"parameter_count":123,"device":"cpu","sequence_length_tokens":256}; self.epigenetic_scaler={}
 def predict(self,r):
  self.calls+=1
  return {"input_mode":r.mode,"predicted_cleavage_frequency":.25,"high_efficiency_probability":.75,"high_efficiency_prediction":True,"sequence_nucleotides":len(r.sequence),"sequence_tokens_used":10,"sequence_length_tokens":256,"model":self.model_info,"feature_availability":{"sequence":{"available":True,"provenance":"mock"},"epigenetic":{"available":r.epigenetic_features is not None,"provenance":"mock"},"3d_genome":{"available":r.genome_features is not None,"provenance":"mock"},"singlecell_expression":{"available":r.single_cell_expression is not None,"provenance":"mock"}},"warning":None if r.mode=="full_multimodal" else "Incomplete-feature demonstration"}
class BackendTests(unittest.TestCase):
 def setUp(self):
  self.predictor=FakePredictor(); self.loader=patch.object(main,"load_predictor",return_value=self.predictor); self.load_mock=self.loader.start(); self.addCleanup(self.loader.stop)
  self.context=TestClient(main.app); self.client=self.context.__enter__(); self.addCleanup(self.context.__exit__,None,None,None)
 def test_health_model_info_and_single_startup_load(self):
  self.assertEqual(self.client.get("/health").json()["status"],"ready"); self.client.get("/model-info"); self.client.get("/health"); self.load_mock.assert_called_once_with()
  self.assertEqual(self.client.get("/model-info").json()["sequence_length_tokens"],256)
 def test_full_prediction(self):
  body={"sequence":"acgtacgtacgt","mode":"full_multimodal","epigenetic_features":[0]*6,"genome_features":[0]*8,"single_cell_expression":.5}; res=self.client.post("/predict",json=body)
  self.assertEqual(res.status_code,200); self.assertEqual(res.json()["high_efficiency_probability"],.75); self.assertEqual(res.json()["sequence_length_tokens"],256); self.assertEqual(self.predictor.calls,1)
 def test_demo_requires_consent(self):
  body={"sequence":"ACGTACGT","mode":"missing_feature_demo"}; self.assertEqual(self.client.post("/predict",json=body).status_code,422)
  body["allow_missing_features"]=True; res=self.client.post("/predict",json=body); self.assertEqual(res.status_code,200); self.assertIn("Incomplete-feature",res.json()["warning"]); self.assertFalse(res.json()["feature_availability"]["epigenetic"]["available"])
 def test_invalid_sequence_and_width(self):
  body={"sequence":"ACGTACGT","mode":"full_multimodal","epigenetic_features":[0]*6,"genome_features":[0]*8,"single_cell_expression":.5}
  self.assertEqual(self.client.post("/predict",json={**body,"sequence":"ACGT-X"}).status_code,422); self.assertEqual(self.client.post("/predict",json={**body,"genome_features":[0]*7}).status_code,422)
if __name__=="__main__": unittest.main(verbosity=2)
