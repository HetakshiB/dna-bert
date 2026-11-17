from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import urllib.request
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from backend.app.schemas import PredictionRequest
from backend.app.service import load_predictor, sha256

ROOT = Path.cwd()
OUT = ROOT / "outputs" / "live_demo_samples"
TEST_PATH = ROOT / "outputs" / "tensors" / "test.pt"
META_PATH = ROOT / "outputs" / "tensors" / "meta.csv"
SPLITS_PATH = ROOT / "outputs" / "tensors" / "split_indices.pt"
CHECKPOINT = ROOT / "outputs" / "model_best.pt"
EXPECTED_CHECKPOINT_SHA = "a12654b360e959a3ee6b9eab47f7a463152dcf813a90ec1e84b9f7f6260b4558"
EXPECTED_TEST_SHA = "4a6ff663e0556cff4d4427531a09b9abf7f01d8b5d64510a0ae369bbcf6f6fb8"


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def choose_records(meta: pd.DataFrame, split_indices: dict, test_data: list[dict]) -> list[dict]:
    test_indices = [int(i) for i in split_indices["test"]]
    if len(test_indices) != len(test_data):
        raise ValueError(f"saved test index count {len(test_indices)} != test tensor count {len(test_data)}")
    candidates = []
    for test_pos, source_idx in enumerate(test_indices):
        row = meta.iloc[source_idx]
        seq = str(row.get("target_sequence", "")).strip().upper()
        if row.get("split") != "test":
            raise ValueError(f"split metadata mismatch at source row {source_idx}: {row.get('split')}")
        if not seq or any(ch not in "ACGTN" for ch in seq) or len(seq) < 6:
            continue
        freq = float(row["cleavage_freq"])
        cls = int(row["high_eff"])
        tensor = test_data[test_pos]
        if not math.isclose(float(tensor["reg_label"].item()), freq, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError(f"regression label mismatch at test position {test_pos}")
        if int(tensor["cls_label"].item()) != cls:
            raise ValueError(f"classification label mismatch at test position {test_pos}")
        candidates.append({"test_position": test_pos, "source_row_index": source_idx,
                           "sample_id": int(row["id"]) if pd.notna(row.get("id")) else None,
                           "sequence": seq, "cleavage_freq": freq, "high_eff": cls})

    chosen = []
    used_sequences, used_frequencies = set(), set()
    # Aim for 3 low-efficiency and 2 high-efficiency records. Choose varied
    # cleavage labels at reproducible quantiles, falling back to nearest valid row.
    for cls, quantiles in ((0, (0.15, 0.55, 0.90)), (1, (0.25, 0.75))):
        group = sorted((x for x in candidates if x["high_eff"] == cls), key=lambda x: (x["cleavage_freq"], x["test_position"]))
        if not group:
            continue
        for q in quantiles:
            target = q * (len(group) - 1)
            order = sorted(range(len(group)), key=lambda j: (abs(j - target), j))
            record = next((group[j] for j in order
                           if group[j]["sequence"] not in used_sequences
                           and group[j]["cleavage_freq"] not in used_frequencies), None)
            if record is not None:
                chosen.append(record)
                used_sequences.add(record["sequence"])
                used_frequencies.add(record["cleavage_freq"])
    if len(chosen) != 5 or {x["high_eff"] for x in chosen} != {0, 1}:
        raise ValueError(f"Could not select five unique valid test sequences across both labels; found {len(chosen)}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out sequence-only missing-feature demo through FastAPI.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="base URL of running FastAPI application")
    args = parser.parse_args()
    for filename in ("missing_demo_samples.json", "missing_demo_results.csv", "missing_demo_guide.md"):
        if not (OUT / filename).is_file():
            continue
        # These are the designated demo outputs and may be refreshed by an explicit rerun.
        print(f"Refreshing demo output: {OUT / filename}")

    checkpoint_hash_before = sha256(CHECKPOINT)
    test_hash_before = sha256(TEST_PATH)
    if checkpoint_hash_before != EXPECTED_CHECKPOINT_SHA or test_hash_before != EXPECTED_TEST_SHA:
        raise RuntimeError("Checkpoint or held-out test tensor hash differs from the recorded frozen artifact")
    meta = pd.read_csv(META_PATH)
    split_indices = torch.load(SPLITS_PATH, map_location="cpu", weights_only=False)
    test_data = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
    if len(test_data) != 2554:
        raise ValueError(f"Expected 2554 held-out test records, found {len(test_data)}")
    selected = choose_records(meta, split_indices, test_data)

    health = json.loads(urllib.request.urlopen(args.api_url.rstrip("/") + "/health", timeout=15).read().decode("utf-8"))
    info = json.loads(urllib.request.urlopen(args.api_url.rstrip("/") + "/model-info", timeout=15).read().decode("utf-8"))
    if health.get("status") != "ready" or health.get("device") != "cuda":
        raise RuntimeError(f"API must be ready on CUDA for this run; got {health}")
    if info.get("checkpoint_sha256") != checkpoint_hash_before or info.get("checkpoint_path") != "outputs/model_best.pt":
        raise RuntimeError("Live API is not serving the expected frozen checkpoint")

    # Load the same application predictor independently for direct-vs-HTTP comparison.
    predictor = load_predictor(CHECKPOINT)
    if predictor.model_info["device"] != "cuda":
        raise RuntimeError(f"Direct predictor is not on CUDA: {predictor.model_info['device']}")

    sample_entries, result_rows, guide_entries = [], [], []
    diffs = []
    for record in selected:
        request_body = {"sequence": record["sequence"], "mode": "missing_feature_demo", "allow_missing_features": True}
        api_result = post_json(args.api_url.rstrip("/") + "/predict", request_body)
        direct_result = predictor.predict(PredictionRequest(**request_body))
        for key in ("predicted_cleavage_frequency", "high_efficiency_probability"):
            diffs.append(abs(float(api_result[key]) - float(direct_result[key])))
        if api_result["high_efficiency_prediction"] != direct_result["high_efficiency_prediction"]:
            raise AssertionError("API and direct classification disagree")
        for modality in ("epigenetic", "3d_genome", "singlecell_expression"):
            status = api_result["feature_availability"][modality]
            if status["available"] is not False:
                raise AssertionError(f"API did not mark {modality} as missing")
        if not api_result.get("warning"):
            raise AssertionError("Missing-feature warning absent from API response")

        freq_pred = float(api_result["predicted_cleavage_frequency"])
        probability = float(api_result["high_efficiency_probability"])
        pred_class = bool(api_result["high_efficiency_prediction"])
        abs_error = abs(freq_pred - record["cleavage_freq"])
        class_match = int(pred_class) == record["high_eff"]
        sample_entries.append({"test_position": record["test_position"], "source_row_index": record["source_row_index"],
                               "sample_id": record["sample_id"], "recorded_cleavage_freq": record["cleavage_freq"],
                               "recorded_high_eff": record["high_eff"], "request": request_body,
                               "api_response": api_result, "direct_application_response": direct_result})
        result_rows.append({"test_position": record["test_position"], "source_row_index": record["source_row_index"],
                            "sample_id": record["sample_id"], "sequence": record["sequence"],
                            "sequence_length_nucleotides": len(record["sequence"]),
                            "recorded_cleavage_freq": record["cleavage_freq"], "recorded_high_eff": record["high_eff"],
                            "predicted_cleavage_frequency": freq_pred,
                            "predicted_high_efficiency_probability_percent": probability * 100.0,
                            "predicted_high_efficiency_class": int(pred_class), "absolute_error": abs_error,
                            "classification_matches_label": class_match,
                            "api_direct_frequency_difference": abs(float(api_result["predicted_cleavage_frequency"]) - float(direct_result["predicted_cleavage_frequency"])),
                            "api_direct_probability_difference": abs(probability - float(direct_result["high_efficiency_probability"])),
                            "api_device": api_result["model"]["device"], "sequence_tokens_used": api_result["sequence_tokens_used"],
                            "input_mode": api_result["input_mode"],
                            "missing_modalities": "epigenetic;3d_genome;singlecell_expression"})
        guide_entries.append(
            f"### Sample {len(guide_entries)+1} — test position {record['test_position']} (source row {record['source_row_index']}, ID {record['sample_id']})\n"
            f"- DNA sequence: `{record['sequence']}` ({len(record['sequence'])} nt).\n"
            f"- Dataset annotations: cleavage frequency `{record['cleavage_freq']:.8g}`, high-efficiency label `{record['high_eff']}`.\n"
            f"- Model output: cleavage frequency `{freq_pred:.8g}`; high-efficiency probability `{probability*100:.4f}%`; predicted class `{int(pred_class)}`.\n"
            f"- Exploratory comparison: absolute error `{abs_error:.8g}`; classification {'matches' if class_match else 'does not match'} the dataset label.\n"
            "- Inputs: sequence only. Epigenetic, 3D genome, and single-cell modalities were omitted and zero-filled by the application; these are not measured inputs.\n"
        )

    max_diff = max(diffs, default=0.0)
    if max_diff > 1e-5:
        raise AssertionError(f"API/direct numerical mismatch exceeds tolerance: max difference {max_diff}")
    checkpoint_hash_after = sha256(CHECKPOINT)
    test_hash_after = sha256(TEST_PATH)
    if checkpoint_hash_after != checkpoint_hash_before or test_hash_after != test_hash_before:
        raise AssertionError("Frozen checkpoint or test tensor changed during demo")

    payload = {
        "provenance": {"checkpoint_path": "outputs/model_best.pt", "checkpoint_sha256": checkpoint_hash_before,
                       "test_tensor_path": "outputs/tensors/test.pt", "test_tensor_sha256": test_hash_before,
                       "test_split_size": len(test_data), "device": "cuda", "api_base_url": args.api_url,
                       "input_mode": "missing_feature_demo", "sequence_length_tokens": 256,
                       "selection": "five valid, distinct DNA sequences drawn only from saved held-out test indices; 3 label-0 and 2 label-1 examples when available; varied cleavage-frequency quantiles",
                       "feature_policy": "no epigenetic, genome, or expression values supplied; API zero-filled omitted modalities with explicit consent",
                       "interpretation": "exploratory missing-feature outputs; not full multimodal predictions or experimentally validated results"},
        "samples": sample_entries
    }
    (OUT / "missing_demo_samples.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "missing_demo_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    guide = """# Sequence-only missing-feature live demo\n\n## How to present this demonstration\n\nThese five sequences are valid DNA target sequences drawn from the saved held-out test split. No training or validation records were used. The request contains only the DNA sequence and explicit consent for `missing_feature_demo`. The application zero-fills the omitted epigenetic, 3D genome, and single-cell inputs. These are **not full multimodal predictions**, are **not experimentally validated results**, and must not be presented as such.\n\nThe displayed cleavage frequency and classification probability are model predictions. The comparison labels are dataset-provided annotations: project provenance says `cleavage_freq` comes from crisprSQL and `high_eff` is derived using a training-split 75th-percentile threshold. The available project documentation does not establish that each value is an independently verified experimental measurement. Comparing predictions to those labels under missing-feature mode is exploratory and is not the original full-model evaluation.\n\n## Samples and results\n\n""" + "\n".join(guide_entries) + f"\n## Validation\n\nAll five requests were sent to the running FastAPI `/predict` endpoint on CUDA and compared with direct calls to the same application `Predictor` implementation and checkpoint. Maximum absolute difference across cleavage-frequency predictions and probabilities: `{max_diff:.9g}`. Classifications agreed for all five. Checkpoint and test-tensor hashes were unchanged before and after inference.\n\n## Reproduce tomorrow\n\n1. Start the backend using the Docker PowerShell command in `backend/README.md` (port 8000).\n2. Run from the project root in PowerShell:\n\n```powershell\ndocker exec -i <running-backend-container-name> python /workspace/AI_CRISPR-Project/outputs/live_demo_samples/run_missing_demo.py --api-url http://127.0.0.1:8000\n```\n\nThe script verifies the frozen checkpoint and test tensor hashes, selects from saved held-out test indices, sends the sequence-only requests, checks direct/API agreement, then refreshes only the three `missing_demo_*` files in this directory.\n"
    (OUT / "missing_demo_guide.md").write_text(guide, encoding="utf-8")
    print(json.dumps({"samples": result_rows, "max_api_direct_abs_difference": max_diff,
                      "checkpoint_sha256_before_after": [checkpoint_hash_before, checkpoint_hash_after],
                      "test_tensor_sha256_before_after": [test_hash_before, test_hash_after],
                      "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
