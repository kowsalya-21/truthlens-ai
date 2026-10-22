import os
import sys
import json
import time
import math
import random
import cv2
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEModel

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MODEL_NAME = "MCG-NJU/videomae-base"
DATASET_DIR = "E:/truthlens-ai/dataset_v3"
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
FEATURES_CACHE_PATH = os.path.join(DATASET_DIR, "features_cache_v3.pt")
WEIGHTS_DIR = "E:/truthlens-ai/backend/weights"
V2_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "videomae_general_head_v2.pt")
V3_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "videomae_general_head_v3.pt")
REPORT_PATH = os.path.join(DATASET_DIR, "v3_evaluation_report.json")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

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

class SpatiotemporalTemporalDynamicsHeadV3(nn.Module):
    """
    V3 Multi-Scale Spatiotemporal Dynamics Classifier.
    Inputs: 2304-dimensional multi-scale token dynamics representation:
      - [0:768]:    Mean spatiotemporal representation across tokens
      - [768:1536]: Temporal slice standard deviation across time tubelets
      - [1536:2304]: Inter-frame slice step difference delta
    """
    def __init__(self, in_dim=2304, hidden_dim1=256, hidden_dim2=64, out_dim=2, dropout=0.25):
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

def sample_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 48
    
    indices = [int(i * (total_frames - 1) / max(1, num_frames - 1)) for i in range(num_frames)]
    indices = sorted(list(set(indices)))
    while len(indices) < num_frames:
        indices.append(indices[-1])
        
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            frames.append(Image.new("RGB", (224, 224), (0, 0, 0)))
    cap.release()
    return frames

def extract_and_cache_features(manifest):
    if os.path.exists(FEATURES_CACHE_PATH):
        print(f"Loading cached features from {FEATURES_CACHE_PATH}...")
        cache = torch.load(FEATURES_CACHE_PATH, map_location="cpu", weights_only=False)
        if len(cache.get("items", [])) == len(manifest):
            print("Features cache is complete and up to date!")
            return cache

    print("Loading VideoMAE Backbone for feature extraction...")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    backbone = VideoMAEModel.from_pretrained(MODEL_NAME)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    items = []
    t0 = time.perf_counter()
    for idx, item in enumerate(manifest, 1):
        v_path = os.path.join(DATASET_DIR, item["filename"])
        frames = sample_video_frames(v_path, 16)
        if frames is None or len(frames) == 0:
            print(f"Error reading frames for {v_path}")
            continue

        inputs = processor(frames, return_tensors="pt")
        with torch.no_grad():
            outputs = backbone(**inputs)
            tokens = outputs.last_hidden_state # [1, 1568, 768]
            
            # 1. Standard global pooled mean [768]
            mean_pooled = tokens.mean(dim=1).squeeze(0)
            
            # 2. Temporal slices: 8 time tubelets x 196 spatial patches
            slices = tokens.view(1, 8, 196, 768).mean(dim=2).squeeze(0) # [8, 768]
            slice_mean = slices.mean(dim=0) # [768]
            slice_std = slices.std(dim=0)   # [768]
            slice_diff = (slices[1:] - slices[:-1]).abs().mean(dim=0) # [768]
            
            dynamics_feat = torch.cat([slice_mean, slice_std, slice_diff], dim=0)

        label_int = 0 if item["label"].upper() == "REAL" else 1

        items.append({
            "id": item["id"],
            "filename": item["filename"],
            "label": item["label"],
            "label_int": label_int,
            "category": item["category"],
            "ai_generator": item.get("ai_generator", "N/A"),
            "source_dataset": item.get("source_dataset", "N/A"),
            "split": item["split"],
            "scene": item.get("original_scene_source_identifier", ""),
            "mean_pooled_768": mean_pooled.cpu(),
            "dynamics_feat_2304": dynamics_feat.cpu()
        })
        print(f"  [{idx:02d}/{len(manifest)}] Cached features: {item['id']:<8} | {item['label']:<15} | {item['category']}")

    total_time = round(time.perf_counter() - t0, 2)
    print(f"Feature extraction complete in {total_time}s ({total_time/len(manifest):.2f}s/clip).")
    
    cache = {
        "items": items,
        "feature_dim_v1": 768,
        "feature_dim_v3": 2304,
        "extracted_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    torch.save(cache, FEATURES_CACHE_PATH)
    print(f"Features saved to: {FEATURES_CACHE_PATH} ({os.path.getsize(FEATURES_CACHE_PATH)/1024:.1f} KB)")
    return cache

def train_and_validate():
    print("==================================================================")
    print("TRUTHLENS AI - GENERAL VIDEOMAE CLASSIFIER V3 TRAINING & AUDIT")
    print("==================================================================")
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cache = extract_and_cache_features(manifest)
    items = cache["items"]

    # Partition by strict split
    train_items = [it for it in items if it["split"] == "train"]
    val_items = [it for it in items if it["split"] == "validation"]
    test_items = [it for it in items if it["split"] == "held-out test"]

    print(f"\nDataset Splits:")
    print(f"  Train Set      : {len(train_items)} clips ({sum(1 for x in train_items if x['label_int']==0)} Real, {sum(1 for x in train_items if x['label_int']==1)} Fake)")
    print(f"  Validation Set : {len(val_items)} clips ({sum(1 for x in val_items if x['label_int']==0)} Real, {sum(1 for x in val_items if x['label_int']==1)} Fake)")
    print(f"  Held-Out Test  : {len(test_items)} clips ({sum(1 for x in test_items if x['label_int']==0)} Real, {sum(1 for x in test_items if x['label_int']==1)} Fake)")

    # Prepare Tensors
    X_train = torch.stack([it["dynamics_feat_2304"] for it in train_items])
    y_train = torch.tensor([it["label_int"] for it in train_items], dtype=torch.long)

    X_val = torch.stack([it["dynamics_feat_2304"] for it in val_items])
    y_val = torch.tensor([it["label_int"] for it in val_items], dtype=torch.long)

    # Initialize model
    model = SpatiotemporalTemporalDynamicsHeadV3(in_dim=2304, hidden_dim1=256, hidden_dim2=64, out_dim=2, dropout=0.25)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state_dict = None
    best_epoch = 0

    print(f"\nBeginning V3 Training ({epochs} epochs, AdamW lr=5e-4, weight_decay=1e-3)...")

    for ep in range(1, epochs + 1):
        model.train()
        indices = list(range(len(train_items)))
        random.shuffle(indices)
        
        batch_size = 12
        ep_loss = 0.0
        batches = math.ceil(len(indices) / batch_size)
        
        for b in range(batches):
            b_idx = indices[b * batch_size : (b + 1) * batch_size]
            bx = X_train[b_idx]
            by = y_train[b_idx]
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            
        scheduler.step()
        ep_loss /= batches

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val)
            v_loss = criterion(v_logits, y_val).item()
            v_preds = torch.argmax(v_logits, dim=-1)
            v_acc = (v_preds == y_val).float().mean().item()

        if v_acc > best_val_acc or (v_acc == best_val_acc and v_loss < best_val_loss):
            best_val_acc = v_acc
            best_val_loss = v_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep

        if ep % 10 == 0 or ep == epochs:
            print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {ep_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.1f}% (Best: {best_val_acc*100:.1f}% @ Ep {best_epoch})")

    print(f"\nBest Model Selected at Epoch {best_epoch} with Val Acc: {best_val_acc*100:.1f}%, Val Loss: {best_val_loss:.4f}")

    # Load best weights into model
    model.load_state_dict(best_state_dict)
    model.eval()

    # Temperature Scaling & Threshold Calibration ON VALIDATION SET ONLY
    print("\n--- VALIDATION CALIBRATION & THRESHOLD TUNING (ON VALIDATION SET ONLY) ---")
    with torch.no_grad():
        val_logits = model(X_val) # [12, 2]

    # Find optimal temperature T on validation logits
    def nll_for_temp(temp):
        scaled_logits = val_logits / temp
        return criterion(scaled_logits, y_val).item()

    best_t = 1.0
    best_t_loss = float("inf")
    for t_cand in np.linspace(0.5, 3.0, 51):
        cur_loss = nll_for_temp(t_cand)
        if cur_loss < best_t_loss:
            best_t_loss = cur_loss
            best_t = round(float(t_cand), 2)

    print(f"Optimal Temperature on Validation Data: T = {best_t:.2f} (Loss reduced from {best_val_loss:.4f} to {best_t_loss:.4f})")

    # Evaluate operating thresholds on validation set
    calibrated_val_probs = torch.softmax(val_logits / best_t, dim=-1)[:, 1].numpy()
    
    # Grid search operating thresholds to minimize false-AI and false-REAL on validation
    tau_real_opts = [0.25, 0.30, 0.35, 0.40]
    tau_fake_opts = [0.60, 0.65, 0.70, 0.75]
    
    best_thresh = (0.35, 0.65)
    best_val_metric = -1.0
    
    for tr in tau_real_opts:
        for tf in tau_fake_opts:
            if tf <= tr:
                continue
            # Evaluate on val
            val_decisions = []
            fp_cnt = 0
            fn_cnt = 0
            inc_cnt = 0
            correct_cnt = 0
            for p, y in zip(calibrated_val_probs, y_val.numpy()):
                if p >= tf:
                    dec = 1
                elif p <= tr:
                    dec = 0
                else:
                    dec = -1 # inconclusive
                
                if dec == -1:
                    inc_cnt += 1
                elif dec == y:
                    correct_cnt += 1
                elif dec == 1 and y == 0:
                    fp_cnt += 1
                elif dec == 0 and y == 1:
                    fn_cnt += 1
            decided = 12 - inc_cnt
            dec_acc = correct_cnt / max(1, decided)
            score = dec_acc - 0.5 * fp_cnt - 0.2 * (inc_cnt / 12)
            if score > best_val_metric:
                best_val_metric = score
                best_thresh = (tr, tf)

    tau_real, tau_fake = best_thresh
    print(f"Selected Operating Thresholds on Validation Data: Real <= {tau_real:.2f}, Fake >= {tau_fake:.2f}")

    # Save V3 checkpoint with full metadata payload
    v3_checkpoint_payload = {
        "state_dict": best_state_dict,
        "config": {
            "in_dim": 2304,
            "hidden_dim1": 256,
            "hidden_dim2": 64,
            "out_dim": 2,
            "dropout": 0.25,
            "temperature": best_t,
            "thresholds": {"real": tau_real, "fake": tau_fake},
            "backbone": MODEL_NAME,
            "architecture": "VideoMAE (Frozen) + Multi-Scale Temporal Dynamics Head V3",
            "feature_version": "v3_dynamics_2304",
            "license": "Apache 2.0"
        },
        "labels": {0: "real", 1: "fake"},
        "trained_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_metrics": {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc": round(best_val_acc, 4),
            "temperature_T": best_t,
            "threshold_real": tau_real,
            "threshold_fake": tau_fake
        }
    }
    torch.save(v3_checkpoint_payload, V3_CHECKPOINT_PATH)
    print(f"Saved candidate checkpoint to: {V3_CHECKPOINT_PATH} ({os.path.getsize(V3_CHECKPOINT_PATH)/1024:.1f} KB)")

    # -------------------------------------------------------------------------
    # HELD-OUT TEST EVALUATION (V2 vs V3 SIDE-BY-SIDE)
    # -------------------------------------------------------------------------
    print("\n==================================================================")
    print("HELD-OUT TEST EVALUATION (12 COMPLETELY UNSEEN CLIPS)")
    print("==================================================================")

    # Load V2 model
    v2_model = SpatiotemporalTemporalDynamicsHeadV2()
    v2_payload = torch.load(V2_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    v2_state = v2_payload["state_dict"] if "state_dict" in v2_payload else v2_payload
    v2_model.load_state_dict(v2_state)
    v2_model.eval()

    # V2 calibration parameters from V2 payload config
    v2_config = v2_payload.get("config", {}) if isinstance(v2_payload, dict) else {}
    v2_t = v2_config.get("temperature", 1.05)
    v2_tau_real = v2_config.get("thresholds", {}).get("real", 0.35)
    v2_tau_fake = v2_config.get("thresholds", {}).get("fake", 0.65)

    test_predictions = []
    
    # Metrics accumulators
    v2_tp = v2_tn = v2_fp = v2_fn = v2_inc = 0
    v3_tp = v3_tn = v3_fp = v3_fn = v3_inc = 0

    for idx, it in enumerate(test_items, 1):
        feat = it["dynamics_feat_2304"].unsqueeze(0)
        gt_int = it["label_int"]
        gt_lbl = it["label"]
        iid = it["id"]
        fname = it["filename"]
        cat = it["category"]
        gen = it["ai_generator"]
        scene = it["scene"]

        with torch.no_grad():
            # V2 prediction
            v2_out = v2_model(feat)
            v2_logit = (v2_out[0, 1] - v2_out[0, 0]).item()
            v2_prob = torch.softmax(v2_out / v2_t, dim=-1)[0, 1].item()
            
            if v2_prob >= v2_tau_fake:
                v2_dec = "AI-GENERATED"
                v2_code = 1
            elif v2_prob <= v2_tau_real:
                v2_dec = "REAL"
                v2_code = 0
            else:
                v2_dec = "INCONCLUSIVE"
                v2_code = -1

            # V3 prediction
            v3_out = model(feat)
            v3_logit = (v3_out[0, 1] - v3_out[0, 0]).item()
            v3_prob = torch.softmax(v3_out / best_t, dim=-1)[0, 1].item()
            
            if v3_prob >= tau_fake:
                v3_dec = "AI-GENERATED"
                v3_code = 1
            elif v3_prob <= tau_real:
                v3_dec = "REAL"
                v3_code = 0
            else:
                v3_dec = "INCONCLUSIVE"
                v3_code = -1

        # V2 stats
        if v2_code == -1:
            v2_inc += 1
            v2_correct = False
        elif v2_code == gt_int:
            v2_correct = True
            if gt_int == 1:
                v2_tp += 1
            else:
                v2_tn += 1
        else:
            v2_correct = False
            if gt_int == 0:
                v2_fp += 1
            else:
                v2_fn += 1

        # V3 stats
        if v3_code == -1:
            v3_inc += 1
            v3_correct = False
        elif v3_code == gt_int:
            v3_correct = True
            if gt_int == 1:
                v3_tp += 1
            else:
                v3_tn += 1
        else:
            v3_correct = False
            if gt_int == 0:
                v3_fp += 1
            else:
                v3_fn += 1

        test_predictions.append({
            "id": iid,
            "filename": fname,
            "category": cat,
            "scene": scene,
            "generator": gen,
            "ground_truth": gt_lbl,
            "ground_truth_int": gt_int,
            "v2": {
                "logit_diff": round(v2_logit, 4),
                "fake_prob": round(v2_prob * 100, 2),
                "decision": v2_dec,
                "correct": v2_correct
            },
            "v3": {
                "logit_diff": round(v3_logit, 4),
                "fake_prob": round(v3_prob * 100, 2),
                "decision": v3_dec,
                "correct": v3_correct
            }
        })

    # Compute Summary Metrics
    total_test = len(test_items)
    
    # V2 metrics
    v2_decided = total_test - v2_inc
    v2_dec_acc = ((v2_tp + v2_tn) / v2_decided * 100) if v2_decided > 0 else 0.0
    v2_sens = (v2_tp / 6.0 * 100)
    v2_spec = (v2_tn / 6.0 * 100)
    v2_bal_acc = (v2_sens + v2_spec) / 2.0
    v2_false_ai = (v2_fp / 6.0 * 100)
    v2_false_real = (v2_fn / 6.0 * 100)
    v2_inc_rate = (v2_inc / 12.0 * 100)

    # V3 metrics
    v3_decided = total_test - v3_inc
    v3_dec_acc = ((v3_tp + v3_tn) / v3_decided * 100) if v3_decided > 0 else 0.0
    v3_sens = (v3_tp / 6.0 * 100)
    v3_spec = (v3_tn / 6.0 * 100)
    v3_bal_acc = (v3_sens + v3_spec) / 2.0
    v3_false_ai = (v3_fp / 6.0 * 100)
    v3_false_real = (v3_fn / 6.0 * 100)
    v3_inc_rate = (v3_inc / 12.0 * 100)

    print("\n--- INDIVIDUAL TEST PREDICTIONS ---")
    for p in test_predictions:
        v2_status = "[OK]" if p['v2']['correct'] else "[FAIL]"
        v3_status = "[OK]" if p['v3']['correct'] else "[FAIL]"
        print(f"[{p['id']}] GT: {p['ground_truth']:<10} | {p['category']:<22} | V2: {p['v2']['fake_prob']:>5.1f}% ({p['v2']['decision']:<12}) {v2_status:<6} | V3: {p['v3']['fake_prob']:>5.1f}% ({p['v3']['decision']:<12}) {v3_status}")

    print("\n--- SIDE-BY-SIDE METRICS COMPARISON ---")
    print(f"{'Metric':<28} | {'V2 Checkpoint':<18} | {'V3 Candidate':<18}")
    print("-" * 70)
    print(f"{'True Positives (AI detected)':<28} | {v2_tp:<18} | {v3_tp:<18}")
    print(f"{'True Negatives (Real detected)':<28} | {v2_tn:<18} | {v3_tn:<18}")
    print(f"{'False Positives (False AI)':<28} | {v2_fp:<18} | {v3_fp:<18}")
    print(f"{'False Negatives (False Real)':<28} | {v2_fn:<18} | {v3_fn:<18}")
    print(f"{'Inconclusive Clips':<28} | {v2_inc:<18} | {v3_inc:<18}")
    print(f"{'Decided Accuracy':<28} | {v2_dec_acc:>5.2f}%{'':<11} | {v3_dec_acc:>5.2f}%{'':<11}")
    print(f"{'Balanced Accuracy':<28} | {v2_bal_acc:>5.2f}%{'':<11} | {v3_bal_acc:>5.2f}%{'':<11}")
    print(f"{'False-AI Rate (FP / Real)':<28} | {v2_false_ai:>5.2f}%{'':<11} | {v3_false_ai:>5.2f}%{'':<11}")
    print(f"{'False-REAL Rate (FN / Fake)':<28} | {v2_false_real:>5.2f}%{'':<11} | {v3_false_real:>5.2f}%{'':<11}")
    print(f"{'Inconclusive Rate':<28} | {v2_inc_rate:>5.2f}%{'':<11} | {v3_inc_rate:>5.2f}%{'':<11}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "v2_checkpoint": V2_CHECKPOINT_PATH,
        "v3_checkpoint": V3_CHECKPOINT_PATH,
        "validation_calibration": {
            "best_temperature": best_t,
            "operating_thresholds": {"tau_real": tau_real, "tau_fake": tau_fake},
            "best_val_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss
        },
        "held_out_test_comparison": {
            "total_test_clips": total_test,
            "v2_metrics": {
                "tp": v2_tp, "tn": v2_tn, "fp": v2_fp, "fn": v2_fn, "inc": v2_inc,
                "decided_accuracy": round(v2_dec_acc, 2),
                "balanced_accuracy": round(v2_bal_acc, 2),
                "false_ai_rate": round(v2_false_ai, 2),
                "false_real_rate": round(v2_false_real, 2),
                "inconclusive_rate": round(v2_inc_rate, 2)
            },
            "v3_metrics": {
                "tp": v3_tp, "tn": v3_tn, "fp": v3_fp, "fn": v3_fn, "inc": v3_inc,
                "decided_accuracy": round(v3_dec_acc, 2),
                "balanced_accuracy": round(v3_bal_acc, 2),
                "false_ai_rate": round(v3_false_ai, 2),
                "false_real_rate": round(v3_false_real, 2),
                "inconclusive_rate": round(v3_inc_rate, 2)
            },
            "individual_predictions": test_predictions
        },
        "dataset_limitations": [
            "Sample size of 12 held-out clips means each individual prediction carries an 8.33% margin on overall accuracy and 16.67% on class-specific rates.",
            "Compression artifacts from legacy codecs (MPEG-4, DivX, XviD) still represent high high-frequency jitter compared to uncompressed raw sensor data.",
            "Novel AI generator architectures (e.g. Wan2.1, Luma Dream Machine) released after training may exhibit novel spatiotemporal distributions not represented in the training set."
        ]
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nComprehensive evaluation report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    train_and_validate()
