import os
import json
import torch
import torch.nn as nn

class SpatiotemporalTemporalDynamicsHeadV2(nn.Module):
    def __init__(self, in_dim=2304, hidden_dim1=256, hidden_dim2=64, out_dim=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def verify_reproducibility():
    checkpoint_path = "backend/weights/videomae_general_head_v2.pt"
    cache_path = "dataset_v2/features_cache_v2.pt"
    manifest_path = "dataset_v2/manifest.json"
    results_path = "dataset_v2/v2_training_and_comparison_results.json"

    assert os.path.exists(checkpoint_path), "Missing v2 checkpoint"
    assert os.path.exists(cache_path), "Missing feature cache"
    assert os.path.exists(manifest_path), "Missing manifest"
    assert os.path.exists(results_path), "Missing results file"

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    config = ckpt.get("config", {})
    temp_T = config.get("temperature_T", 1.1408)
    tau_real = config.get("tau_real", 0.40)
    tau_fake = config.get("tau_fake", 0.60)

    head = SpatiotemporalTemporalDynamicsHeadV2(2304, 256, 64, 2, 0.3)
    head.load_state_dict(state_dict)
    head.eval()

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    with open(results_path, "r") as f:
        saved_results = json.load(f)

    test_cache_items = [item for item in cache["items"] if item.get("split") == "test"]

    print("==================================================================")
    print("VERIFYING HELD-OUT TEST SET REPRODUCIBILITY FROM SAVED V2 CHECKPOINT")
    print("==================================================================")
    print(f"Loaded Checkpoint: {checkpoint_path}")
    print(f"Calibration Parameters: Temp={temp_T}, tau_real={tau_real}, tau_fake={tau_fake}")
    print(f"Found {len(test_cache_items)} test items in cache.\n")

    mismatches = 0
    reproduced_entries = []
    tp = fp = tn = fn = inc = 0

    saved_detailed_by_filename = {d["filename"]: d for d in saved_results["test_detailed"]}

    for item in test_cache_items:
        fn_key = item["filename"]
        cid = str(item["id"])
        feat = item["dynamics_feat_2304"].unsqueeze(0)
        label = item["label"]

        with torch.no_grad():
            raw_logits = head(feat)
            cal_logits = raw_logits / temp_T
            cal_probs = torch.softmax(cal_logits, dim=-1)
            raw_probs = torch.softmax(raw_logits, dim=-1)

        p_cal = float(cal_probs[0, 1].item())
        p_raw = float(raw_probs[0, 1].item())

        if p_cal >= tau_fake:
            verdict = "LIKELY AI-GENERATED / MANIPULATED"
            if label == "fake":
                tp += 1
            else:
                fp += 1
        elif p_cal <= tau_real:
            verdict = "LIKELY REAL"
            if label == "real":
                tn += 1
            else:
                fn += 1
        else:
            verdict = "INCONCLUSIVE"
            inc += 1

        saved = saved_detailed_by_filename.get(fn_key)
        saved_cal = saved["v2_model"]["calibrated_score"] if saved else None
        saved_verdict = saved["v2_model"]["verdict"] if saved else None

        diff = abs(p_cal - saved_cal) if saved_cal is not None else 0.0
        match = diff < 0.005 and verdict == saved_verdict
        if not match:
            mismatches += 1

        print(f"Clip {cid:<4} ({fn_key:<45}) | True: {label.upper():<4} | Computed: {p_cal:.2%} ({verdict:<30}) | Saved: {saved_cal:.2%} ({saved_verdict}) | Match: {'YES' if match else 'MISMATCH'}")

    print("\n------------------------------------------------------------------")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}, Inconclusive={inc}")
    print(f"Mismatches with Saved Results: {mismatches}")
    if mismatches == 0:
        print(">>> SUCCESS: 100% of reported held-out test predictions reproduced exactly.")
    else:
        print(">>> WARNING: Discrepancies detected.")

if __name__ == "__main__":
    verify_reproducibility()
