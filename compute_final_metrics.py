import json

REPORT_PATH = "E:/truthlens-ai/dataset_v3/v3_evaluation_report.json"

with open(REPORT_PATH, "r") as f:
    report = json.load(f)

preds = report["held_out_test_comparison"]["individual_predictions"]

def compute_metrics(version_key):
    # Classes: 0: REAL, 1: AI-GENERATED
    # Decisions: "REAL", "AI-GENERATED", "INCONCLUSIVE"
    # Confusion matrix on DECIDED samples:
    # Rows: Ground Truth (Real, Fake)
    # Cols: Predicted (Real, Fake)
    cm = {"Real_Real": 0, "Real_Fake": 0, "Fake_Real": 0, "Fake_Fake": 0}
    inc_counts = {"Real_Inc": 0, "Fake_Inc": 0}
    
    for p in preds:
        gt = p["ground_truth_int"] # 0 for Real, 1 for Fake
        dec = p[version_key]["decision"]
        if dec == "REAL":
            pred = 0
            if gt == 0:
                cm["Real_Real"] += 1
            else:
                cm["Fake_Real"] += 1
        elif dec == "AI-GENERATED":
            pred = 1
            if gt == 0:
                cm["Real_Fake"] += 1
            else:
                cm["Fake_Fake"] += 1
        else: # INCONCLUSIVE
            if gt == 0:
                inc_counts["Real_Inc"] += 1
            else:
                inc_counts["Fake_Inc"] += 1

    total_real = 6
    total_fake = 6
    total_samples = 12

    # Standard metrics
    tn = cm["Real_Real"]
    fp = cm["Real_Fake"]
    fn = cm["Fake_Real"]
    tp = cm["Fake_Fake"]
    inc_real = inc_counts["Real_Inc"]
    inc_fake = inc_counts["Fake_Inc"]
    total_inc = inc_real + inc_fake
    decided_total = total_samples - total_inc

    # Rates across all samples
    false_ai_rate = fp / total_real * 100.0       # Genuine Real clips falsely called AI
    false_real_rate = fn / total_fake * 100.0     # Synthetic AI clips falsely called Real
    inconclusive_rate = total_inc / total_samples * 100.0
    inc_real_rate = inc_real / total_real * 100.0
    inc_fake_rate = inc_fake / total_fake * 100.0

    # Decided metrics (only on samples where model made a definitive decision)
    decided_acc = (tp + tn) / max(1, decided_total) * 100.0
    
    # Recall per class
    recall_real = tn / total_real * 100.0
    recall_fake = tp / total_fake * 100.0
    balanced_acc = (recall_real + recall_fake) / 2.0

    # Precision per class (on predicted class)
    prec_real = (tn / (tn + fn) * 100.0) if (tn + fn) > 0 else 0.0
    prec_fake = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0

    # F1 score per class
    f1_real = (2 * prec_real * recall_real / (prec_real + recall_real)) if (prec_real + recall_real) > 0 else 0.0
    f1_fake = (2 * prec_fake * recall_fake / (prec_fake + recall_fake)) if (prec_fake + recall_fake) > 0 else 0.0

    # Decided precision/recall (conditioned on decided)
    decided_real = total_real - inc_real
    decided_fake = total_fake - inc_fake
    rec_real_decided = (tn / decided_real * 100.0) if decided_real > 0 else 0.0
    rec_fake_decided = (tp / decided_fake * 100.0) if decided_fake > 0 else 0.0
    bal_acc_decided = (rec_real_decided + rec_fake_decided) / 2.0

    return {
        "TN (Real->Real)": tn,
        "FP (Real->Fake)": fp,
        "FN (Fake->Real)": fn,
        "TP (Fake->Fake)": tp,
        "Inc Real": inc_real,
        "Inc Fake": inc_fake,
        "Total Inc": total_inc,
        "Decided Total": decided_total,
        "Decided Accuracy (%)": round(decided_acc, 2),
        "Overall Accuracy across 12 (%)": round((tp + tn) / 12.0 * 100.0, 2),
        "Balanced Accuracy (%)": round(balanced_acc, 2),
        "Balanced Accuracy on Decided (%)": round(bal_acc_decided, 2),
        "False-AI Rate (%)": round(false_ai_rate, 2),
        "False-REAL Rate (%)": round(false_real_rate, 2),
        "Inconclusive Rate (%)": round(inconclusive_rate, 2),
        "Class REAL Precision (%)": round(prec_real, 2),
        "Class REAL Recall (%)": round(recall_real, 2),
        "Class REAL F1 (%)": round(f1_real, 2),
        "Class AI Precision (%)": round(prec_fake, 2),
        "Class AI Recall (%)": round(recall_fake, 2),
        "Class AI F1 (%)": round(f1_fake, 2),
    }

v2 = compute_metrics("v2")
v3 = compute_metrics("v3")

print("=== V2 METRICS ===")
for k, v in v2.items():
    print(f"  {k:<32}: {v}")

print("\n=== V3 METRICS ===")
for k, v in v3.items():
    print(f"  {k:<32}: {v}")
