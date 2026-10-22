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
DATASET_DIR = "e:/truthlens-ai/dataset_v4"
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
FEATURES_CACHE_PATH = os.path.join(DATASET_DIR, "features_cache_v4.pt")
WEIGHTS_DIR = "e:/truthlens-ai/backend/weights"
V4_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "videomae_general_head_v4.pt")
REPORT_PATH = os.path.join(DATASET_DIR, "v4_evaluation_report.json")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

class SpatiotemporalTemporalDynamicsHeadV4(nn.Module):
    """
    V4 Enhanced Multi-Scale Spatiotemporal Dynamics Classifier (3072-d input).
    Processes:
      - [0:768]:     Slice Mean (global pooled representation across time and space)
      - [768:1536]:  Slice Temporal Std (variance across 8 time tubelets)
      - [1536:2304]: Slice Temporal Delta (step difference between consecutive time slices)
      - [2304:3072]: Spatial Patch Dispersion (spatial standard deviation across 196 spatial patches)
    """
    def __init__(self, in_dim=3072, hidden_dim1=384, hidden_dim2=96, out_dim=2, dropout=0.25):
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
            
            # 8 time tubelets x 196 spatial patches
            tubelets = tokens.view(1, 8, 196, 768)
            slices = tubelets.mean(dim=2).squeeze(0) # [8, 768]
            
            # 1. Global pooled mean [768]
            slice_mean = slices.mean(dim=0)
            
            # 2. Temporal slice standard deviation [768]
            slice_std = slices.std(dim=0)
            
            # 3. Inter-frame slice step difference delta [768]
            slice_diff = (slices[1:] - slices[:-1]).abs().mean(dim=0)
            
            # 4. Intra-frame spatial patch dispersion [768]
            spatial_dispersion = tubelets.std(dim=2).mean(dim=1).squeeze(0)
            
            dynamics_feat_3072 = torch.cat([slice_mean, slice_std, slice_diff, spatial_dispersion], dim=0)

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
            "dynamics_feat_3072": dynamics_feat_3072.cpu()
        })
        if idx % 10 == 0 or idx == len(manifest):
            print(f"  [{idx:02d}/{len(manifest)}] Cached features: {item['id']:<20} | {item['label']:<15} | {item['category']}")

    total_time = round(time.perf_counter() - t0, 2)
    print(f"Feature extraction complete in {total_time}s ({total_time/len(manifest):.2f}s/clip).")
    
    cache = {
        "items": items,
        "feature_dim": 3072,
        "extracted_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    torch.save(cache, FEATURES_CACHE_PATH)
    print(f"Features saved to: {FEATURES_CACHE_PATH} ({os.path.getsize(FEATURES_CACHE_PATH)/1024:.1f} KB)")
    return cache

def train_and_evaluate():
    print("==================================================================")
    print("TRUTHLENS AI - GENERAL VIDEOMAE V4 TRAINING & RIGOROUS AUDIT")
    print("==================================================================")
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cache = extract_and_cache_features(manifest)
    items = cache["items"]

    # Partition by strict split
    train_items = [it for it in items if it["split"] == "train"]
    val_items = [it for it in items if it["split"] == "validation"]
    test_items = [it for it in items if it["split"] == "held-out test"]

    print(f"\nDataset v4 Splits:")
    print(f"  Train Set      : {len(train_items)} clips ({sum(1 for x in train_items if x['label_int']==0)} Real, {sum(1 for x in train_items if x['label_int']==1)} Fake)")
    print(f"  Validation Set : {len(val_items)} clips ({sum(1 for x in val_items if x['label_int']==0)} Real, {sum(1 for x in val_items if x['label_int']==1)} Fake)")
    print(f"  Held-Out Test  : {len(test_items)} clips ({sum(1 for x in test_items if x['label_int']==0)} Real, {sum(1 for x in test_items if x['label_int']==1)} Fake)")

    # Prepare Tensors
    X_train = torch.stack([it["dynamics_feat_3072"] for it in train_items])
    y_train = torch.tensor([it["label_int"] for it in train_items], dtype=torch.long)

    X_val = torch.stack([it["dynamics_feat_3072"] for it in val_items])
    y_val = torch.tensor([it["label_int"] for it in val_items], dtype=torch.long)

    X_test = torch.stack([it["dynamics_feat_3072"] for it in test_items])
    y_test = torch.tensor([it["label_int"] for it in test_items], dtype=torch.long)

    # Initialize model
    model = SpatiotemporalTemporalDynamicsHeadV4(in_dim=3072, hidden_dim1=384, hidden_dim2=96, out_dim=2, dropout=0.25)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=2e-3)
    epochs = 70
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state_dict = None
    best_epoch = 0

    print(f"\nBeginning V4 Training ({epochs} epochs, AdamW lr=4e-4, weight_decay=2e-3)...")

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

    print(f"\nBest V4 Model Selected at Epoch {best_epoch} with Val Acc: {best_val_acc*100:.1f}%, Val Loss: {best_val_loss:.4f}")

    # Load best weights
    model.load_state_dict(best_state_dict)
    model.eval()

    # Temperature Scaling & Threshold Calibration ON VALIDATION SET ONLY
    print("\n--- VALIDATION CALIBRATION & THRESHOLD TUNING (ON VALIDATION SET ONLY) ---")
    with torch.no_grad():
        val_logits = model(X_val)

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
            correct_cnt = fp_cnt = fn_cnt = inc_cnt = 0
            for p, y in zip(calibrated_val_probs, y_val.numpy()):
                if p >= tf: dec = 1
                elif p <= tr: dec = 0
                else: dec = -1
                
                if dec == -1: inc_cnt += 1
                elif dec == y: correct_cnt += 1
                elif dec == 1 and y == 0: fp_cnt += 1
                elif dec == 0 and y == 1: fn_cnt += 1
            decided = len(val_items) - inc_cnt
            dec_acc = correct_cnt / max(1, decided)
            score = dec_acc - 0.6 * fp_cnt - 0.3 * fn_cnt - 0.2 * (inc_cnt / len(val_items))
            if score > best_val_metric:
                best_val_metric = score
                best_thresh = (tr, tf)

    tau_real, tau_fake = best_thresh
    print(f"Selected Operating Thresholds on Validation Data: Real <= {tau_real:.2f}, Fake >= {tau_fake:.2f}")

    # Save V4 checkpoint
    v4_checkpoint_payload = {
        "state_dict": best_state_dict,
        "config": {
            "in_dim": 3072,
            "hidden_dim1": 384,
            "hidden_dim2": 96,
            "out_dim": 2,
            "dropout": 0.25,
            "temperature": best_t,
            "thresholds": {"real": tau_real, "fake": tau_fake},
            "backbone": MODEL_NAME,
            "architecture": "VideoMAE (Frozen) + Multi-Scale Temporal Dynamics Head V4",
            "feature_version": "v4_dynamics_3072",
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
    torch.save(v4_checkpoint_payload, V4_CHECKPOINT_PATH)
    print(f"Saved candidate checkpoint to: {V4_CHECKPOINT_PATH} ({os.path.getsize(V4_CHECKPOINT_PATH)/1024:.1f} KB)")

    # -------------------------------------------------------------------------
    # HELD-OUT TEST EVALUATION ON UNTOUCHED 12 CLIPS (V2 vs V3 vs V4)
    # -------------------------------------------------------------------------
    print("\n==================================================================")
    print("HELD-OUT TEST EVALUATION (12 COMPLETELY UNSEEN CLIPS)")
    print("==================================================================")

    # Load V2
    from backend.experimental_detector import SpatiotemporalTemporalDynamicsHeadV2
    v2_model = SpatiotemporalTemporalDynamicsHeadV2()
    v2_payload = torch.load("backend/weights/videomae_general_head_v2.pt", map_location="cpu", weights_only=False)
    v2_model.load_state_dict(v2_payload["state_dict"] if "state_dict" in v2_payload else v2_payload)
    v2_model.eval()
    v2_t = v2_payload.get("config", {}).get("temperature", 1.05)
    v2_tr = v2_payload.get("config", {}).get("thresholds", {}).get("real", 0.35)
    v2_tf = v2_payload.get("config", {}).get("thresholds", {}).get("fake", 0.65)

    # Load V3
    v3_model = SpatiotemporalTemporalDynamicsHeadV2()
    v3_payload = torch.load("backend/weights/videomae_general_head_v3.pt", map_location="cpu", weights_only=False)
    v3_model.load_state_dict(v3_payload["state_dict"] if "state_dict" in v3_payload else v3_payload)
    v3_model.eval()
    v3_t = v3_payload.get("config", {}).get("temperature", 1.40)
    v3_tr = v3_payload.get("config", {}).get("thresholds", {}).get("real", 0.25)
    v3_tf = v3_payload.get("config", {}).get("thresholds", {}).get("fake", 0.60)

    # Load V3 features cache for V2 & V3 2304-d inputs
    v3_cache = torch.load("dataset_v3/features_cache_v3.pt", map_location="cpu", weights_only=False)
    v3_feat_map = {it["id"]: it["dynamics_feat_2304"] for it in v3_cache["items"]}

    test_predictions = []

    for idx, it in enumerate(test_items, 1):
        feat_3072 = it["dynamics_feat_3072"].unsqueeze(0)
        feat_2304 = v3_feat_map[it["id"]].unsqueeze(0)
        gt_int = it["label_int"]
        gt_lbl = it["label"]
        iid = it["id"]
        fname = it["filename"]
        cat = it["category"]
        gen = it["ai_generator"]
        scene = it["scene"]

        with torch.no_grad():
            # V2 prediction
            v2_out = v2_model(feat_2304)
            v2_prob = torch.softmax(v2_out / v2_t, dim=-1)[0, 1].item()
            if v2_prob >= v2_tf: v2_dec = "AI-GENERATED"
            elif v2_prob <= v2_tr: v2_dec = "REAL"
            else: v2_dec = "INCONCLUSIVE"

            # V3 prediction
            v3_out = v3_model(feat_2304)
            v3_prob = torch.softmax(v3_out / v3_t, dim=-1)[0, 1].item()
            if v3_prob >= v3_tf: v3_dec = "AI-GENERATED"
            elif v3_prob <= v3_tr: v3_dec = "REAL"
            else: v3_dec = "INCONCLUSIVE"

            # V4 prediction
            v4_out = model(feat_3072)
            v4_prob = torch.softmax(v4_out / best_t, dim=-1)[0, 1].item()
            if v4_prob >= tau_fake: v4_dec = "AI-GENERATED"
            elif v4_prob <= tau_real: v4_dec = "REAL"
            else: v4_dec = "INCONCLUSIVE"

        correct_v4 = (gt_lbl == "REAL" and v4_dec == "REAL") or (gt_lbl != "REAL" and v4_dec == "AI-GENERATED")
        status_flag = "[CORRECT]" if correct_v4 else (f"[{v4_dec}]")

        print(f"[{idx:02d}/12] {iid:<8} | True: {gt_lbl:<20} | V2: {v2_dec:<13} ({v2_prob:.1%}) | V3: {v3_dec:<13} ({v3_prob:.1%}) | V4: {v4_dec:<13} ({v4_prob:.1%}) | {status_flag}")

        test_predictions.append({
            "id": iid,
            "filename": fname,
            "category": cat,
            "generator": gen,
            "scene": scene,
            "ground_truth": gt_lbl,
            "ground_truth_int": gt_int,
            "v2": {"score": round(v2_prob, 4), "decision": v2_dec},
            "v3": {"score": round(v3_prob, 4), "decision": v3_dec},
            "v4": {"score": round(v4_prob, 4), "decision": v4_dec}
        })

    def calc_metrics(v_key):
        tn = fp = fn = tp = inc_real = inc_fake = 0
        for p in test_predictions:
            gt = p["ground_truth_int"]
            dec = p[v_key]["decision"]
            if dec == "REAL":
                if gt == 0: tn += 1
                else: fn += 1
            elif dec == "AI-GENERATED":
                if gt == 0: fp += 1
                else: tp += 1
            else:
                if gt == 0: inc_real += 1
                else: inc_fake += 1
        decided = 12 - (inc_real + inc_fake)
        dec_acc = (tp + tn) / max(1, decided) * 100.0
        overall_acc = (tp + tn) / 12.0 * 100.0
        rec_real = tn / 6.0 * 100.0
        rec_fake = tp / 6.0 * 100.0
        bal_acc = (rec_real + rec_fake) / 2.0
        prec_real = (tn / (tn + fn) * 100.0) if (tn + fn) > 0 else 0.0
        prec_fake = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        f1_real = (2 * prec_real * rec_real / (prec_real + rec_real)) if (prec_real + rec_real) > 0 else 0.0
        f1_fake = (2 * prec_fake * rec_fake / (prec_fake + rec_fake)) if (prec_fake + rec_fake) > 0 else 0.0
        return {
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Inc_Real": inc_real, "Inc_Fake": inc_fake, "Total_Inc": inc_real + inc_fake,
            "Decided_Total": decided,
            "Decided_Accuracy": round(dec_acc, 2),
            "Overall_Accuracy": round(overall_acc, 2),
            "Balanced_Accuracy": round(bal_acc, 2),
            "False_AI_Rate": round(fp / 6.0 * 100.0, 2),
            "False_REAL_Rate": round(fn / 6.0 * 100.0, 2),
            "Inconclusive_Rate": round((inc_real + inc_fake) / 12.0 * 100.0, 2),
            "Class_REAL_Precision": round(prec_real, 2),
            "Class_REAL_Recall": round(rec_real, 2),
            "Class_REAL_F1": round(f1_real, 2),
            "Class_AI_Precision": round(prec_fake, 2),
            "Class_AI_Recall": round(rec_fake, 2),
            "Class_AI_F1": round(f1_fake, 2)
        }

    m_v2 = calc_metrics("v2")
    m_v3 = calc_metrics("v3")
    m_v4 = calc_metrics("v4")

    print("\n--- SIDE-BY-SIDE HELD-OUT TEST METRICS ---")
    print(f"{'Metric':<30} | {'V2 Baseline':<12} | {'V3 Candidate':<12} | {'V4 Improved':<12}")
    print("-" * 72)
    for k in ["Overall_Accuracy", "Decided_Accuracy", "Balanced_Accuracy", "False_AI_Rate", "False_REAL_Rate", "Inconclusive_Rate", "Class_REAL_F1", "Class_AI_F1"]:
        print(f"{k:<30} | {m_v2[k]:>10.2f}% | {m_v3[k]:>10.2f}% | {m_v4[k]:>10.2f}%")

    report_payload = {
        "dataset": "dataset_v4",
        "held_out_test_comparison": {
            "v2_metrics": m_v2,
            "v3_metrics": m_v3,
            "v4_metrics": m_v4,
            "individual_predictions": test_predictions
        },
        "v4_training_parameters": {
            "epochs": epochs,
            "best_epoch": best_epoch,
            "temperature_T": best_t,
            "operating_thresholds": {"real": tau_real, "fake": tau_fake},
            "feature_dim": 3072,
            "feature_components": ["slice_mean", "slice_std", "slice_diff", "spatial_dispersion"]
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nEvaluation report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
