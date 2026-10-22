import os
import json
import time
from backend.detector import analyze_video_file

EVAL_DIR = "e:/truthlens-ai/eval_dataset"
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.json")
OUTPUT_PATH = os.path.join(EVAL_DIR, "baseline_results.json")

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

print(f"Running baseline evaluation across {len(manifest)} smoke-test clips...")
print("Baseline Model: Existing Facial Specialist ViT ('dima806/deepfake_vs_real_image_detection')\n")

results = []
summary = {
    "total_clips": len(manifest),
    "genuine_clips": 0,
    "synthetic_clips": 0,
    "predictions": {
        "LIKELY REAL": 0,
        "LIKELY AI-GENERATED / MANIPULATED": 0,
        "INCONCLUSIVE": 0
    },
    "by_category": {},
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
}

start_all = time.perf_counter()

for idx, item in enumerate(manifest, 1):
    video_rel = item["filename"]
    video_path = os.path.join(EVAL_DIR, video_rel)
    true_label = item["label"]
    cat = item["category"]
    
    if true_label == "real":
        summary["genuine_clips"] += 1
    else:
        summary["synthetic_clips"] += 1
        
    t0 = time.perf_counter()
    # Run existing detector with 8 sampled frames for fast baseline
    res = analyze_video_file(video_path, max_samples=8)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    
    pred = res.get("prediction", "INCONCLUSIVE")
    score = res.get("model_score")
    status = res.get("analysis_status")
    v_det = res.get("video_details", {})
    faces_det = v_det.get("faces_detected", 0)
    frames_an = v_det.get("frames_analyzed", 0)
    
    summary["predictions"][pred] = summary["predictions"].get(pred, 0) + 1
    if cat not in summary["by_category"]:
        summary["by_category"][cat] = {"real": 0, "fake": 0, "inconclusive": 0, "total": 0}
    summary["by_category"][cat]["total"] += 1
    if pred == "LIKELY REAL":
        summary["by_category"][cat]["real"] += 1
    elif pred == "LIKELY AI-GENERATED / MANIPULATED":
        summary["by_category"][cat]["fake"] += 1
    else:
        summary["by_category"][cat]["inconclusive"] += 1
        
    record = {
        "id": item["id"],
        "filename": video_rel,
        "true_label": true_label,
        "category": cat,
        "source": item.get("source"),
        "prediction": pred,
        "model_score": score,
        "analysis_status": status,
        "faces_detected": faces_det,
        "frames_analyzed": frames_an,
        "latency_ms": elapsed_ms
    }
    results.append(record)
    
    flag = "[CORRECT]" if (true_label == "real" and pred == "LIKELY REAL") or (true_label == "fake" and pred == "LIKELY AI-GENERATED / MANIPULATED") else f"[{pred}]"
    print(f"[{idx:02d}/24] {item['id']:<6} | True: {true_label.upper():<4} | Pred: {pred:<33} | Score: {str(score):<6} | Faces: {faces_det}/{frames_an} | {flag} | {elapsed_ms}ms")

total_time = round(time.perf_counter() - start_all, 2)
summary["total_runtime_sec"] = total_time
summary["avg_latency_ms"] = round((total_time / len(manifest)) * 1000, 1)

output_payload = {
    "metadata": {
        "baseline_model": "dima806/deepfake_vs_real_image_detection (Facial Specialist ViT)",
        "scope": "Facial Manipulation and Face-Swaps (Baseline Smoke Test)",
        "hardware": "CPU (8 cores / 16 threads, AMD Radeon integrated)",
        "note": "Initial baseline smoke test on 24 clips. Not statistical proof of production accuracy."
    },
    "summary": summary,
    "detailed_results": results
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output_payload, f, indent=2)

print(f"\n==================================================================")
print(f"BASELINE RUN COMPLETED in {total_time}s (Avg {summary['avg_latency_ms']} ms/clip)")
print(f"Predictions: Real={summary['predictions']['LIKELY REAL']}, Fake={summary['predictions']['LIKELY AI-GENERATED / MANIPULATED']}, Inconclusive={summary['predictions']['INCONCLUSIVE']}")
print(f"Results successfully saved to: {OUTPUT_PATH}")
print(f"==================================================================")
