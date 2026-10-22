import os
import json
import torch
from train_videomae_v4 import SpatiotemporalTemporalDynamicsHeadV4

MANIFEST_PATH = "e:/truthlens-ai/dataset_v4/manifest.json"
CACHE_PATH = "e:/truthlens-ai/dataset_v4/features_cache_v4.pt"
V4_WEIGHTS = "e:/truthlens-ai/backend/weights/videomae_general_head_v4.pt"

# Load manifests and features
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
items = cache["items"]

# Load V4 model
v4_payload = torch.load(V4_WEIGHTS, map_location="cpu", weights_only=False)
v4_cfg = v4_payload.get("config", {})
v4_t = float(v4_cfg.get("temperature", 0.85))
v4_tr = float(v4_cfg.get("thresholds", {}).get("real", 0.25))
v4_tf = float(v4_cfg.get("thresholds", {}).get("fake", 0.70))

v4_model = SpatiotemporalTemporalDynamicsHeadV4(in_dim=3072, hidden_dim1=384, hidden_dim2=96, out_dim=2)
v4_model.load_state_dict(v4_payload["state_dict"])
v4_model.eval()

# Explicit domain mapping
def get_super_category(cat):
    c = cat.lower()
    if any(k in c for k in ["animal", "insect", "tiger", "bird", "horse", "cat", "dog"]):
        return "Animals & Wildlife"
    elif any(k in c for k in ["animation", "cartoon", "cel", "fantasy"]):
        return "Animation & Fantasy"
    elif any(k in c for k in ["people", "human", "portrait", "walk", "crowd", "driver", "classroom", "faceswap"]):
        return "Humans & Actions"
    elif any(k in c for k in ["nature", "tree", "wind", "water", "ocean", "landscape"]):
        return "Nature & Landscapes"
    else:
        return "Objects & Machinery"

group_results = {}

for it in items:
    feat = it["dynamics_feat_3072"].unsqueeze(0)
    lbl = it["label"]
    cat = it["category"]
    super_cat = get_super_category(cat)
    
    with torch.no_grad():
        out = v4_model(feat)
        prob = torch.softmax(out / v4_t, dim=-1)[0, 1].item()
        
    if prob >= v4_tf:
        dec = "AI-GENERATED"
    elif prob <= v4_tr:
        dec = "REAL"
    else:
        dec = "INCONCLUSIVE"
        
    is_correct = (lbl == "REAL" and dec == "REAL") or (lbl != "REAL" and dec == "AI-GENERATED")
    
    if super_cat not in group_results:
        group_results[super_cat] = {"total": 0, "correct": 0, "inc": 0, "real": 0, "fake": 0}
        
    g = group_results[super_cat]
    g["total"] += 1
    if lbl == "REAL": g["real"] += 1
    else: g["fake"] += 1
    if dec == "INCONCLUSIVE":
        g["inc"] += 1
    elif is_correct:
        g["correct"] += 1

print("==================================================================")
print("V4 ACCURACY BREAKDOWN BY VISUAL DOMAIN ACROSS ALL 70 CLIPS")
print("==================================================================")
print(f"{'Category Domain':<25} | {'Clips':<6} | {'Real':<5} | {'Fake':<5} | {'Correct':<8} | {'Inconcl.':<8} | {'Decided Acc':<12}")
print("-" * 80)

for grp, stats in group_results.items():
    decided = stats["total"] - stats["inc"]
    acc = stats["correct"] / max(1, decided) * 100.0
    print(f"{grp:<25} | {stats['total']:<6} | {stats['real']:<5} | {stats['fake']:<5} | {stats['correct']:<8} | {stats['inc']:<8} | {acc:>10.1f}%")

with open("e:/truthlens-ai/dataset_v4/v4_category_breakdown.json", "w") as f:
    json.dump(group_results, f, indent=2)
