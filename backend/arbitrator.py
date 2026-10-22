"""
TruthLens AI - Multi-Detector Forensic Arbitration Layer
Handles disagreement resolution between Localized Facial Manipulation Specialist (ViT)
and Spatiotemporal General Video Authenticity Detector (VideoMAE v4).

Core Rules:
1. Facial manipulation and general synthetic-video evidence are evaluated separately.
2. Missing faces are NOT evidence that a video is REAL.
3. Scores are evaluated against each detector's calibrated operating thresholds, never averaged blindly.
4. If both detectors have adequate evidence and agree, return their supported combined verdict.
5. If reliable detectors strongly disagree, return INCONCLUSIVE with a clear reason and show both individual results.
6. If a detector is unavailable (e.g. 0 faces), return a transparent partial-evidence result or INCONCLUSIVE according to the API contract. Never silently substitute another model's verdict.
7. The number of detected faces alone never triggers INCONCLUSIVE.
"""

from typing import Dict, Any, Optional

def extract_facial_evidence(facial_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts structured facial detector signals and validity."""
    if not facial_result:
        return {
            "available": False,
            "verdict": "UNAVAILABLE",
            "score": None,
            "faces_detected": 0,
            "faces_analyzed": 0,
            "persons_tracked": 0,
            "mean_fake_score": None,
            "max_fake_score": None,
            "has_faces": False,
            "explanation": "Facial detector was not executed or returned no output."
        }
    
    video_details = facial_result.get("video_details", {})
    metrics = facial_result.get("metrics", {})
    faces_detected = video_details.get("faces_detected", 0) or 0
    faces_analyzed = video_details.get("faces_analyzed", 0) or 0
    persons_tracked = metrics.get("persons_tracked", 0) or 0
    raw_pred = facial_result.get("prediction", "INCONCLUSIVE")
    score = facial_result.get("model_score")
    explanation = facial_result.get("explanation", "")
    
    has_faces = faces_analyzed >= 2 or (faces_analyzed >= 1 and raw_pred != "INCONCLUSIVE")
    
    if not has_faces:
        verdict = "INSUFFICIENT_EVIDENCE"
    elif raw_pred == "LIKELY AI-GENERATED / MANIPULATED":
        verdict = "AI-GENERATED / MANIPULATED"
    elif raw_pred == "LIKELY REAL":
        verdict = "REAL"
    else:
        verdict = "INCONCLUSIVE"
        
    return {
        "available": True,
        "verdict": verdict,
        "score": score,
        "faces_detected": faces_detected,
        "faces_analyzed": faces_analyzed,
        "persons_tracked": persons_tracked,
        "mean_fake_score": metrics.get("mean_fake_score"),
        "max_fake_score": metrics.get("max_fake_score"),
        "has_faces": has_faces,
        "explanation": explanation,
        "raw_prediction": raw_pred
    }

def extract_videomae_evidence(videomae_raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts structured spatiotemporal VideoMAE signals and operating margins."""
    if not videomae_raw:
        return {
            "available": False,
            "verdict": "UNAVAILABLE",
            "score": None,
            "version": "unknown",
            "thresholds": {"real": 0.25, "fake": 0.70},
            "explanation": "VideoMAE detector was not executed."
        }
    
    p_fake = videomae_raw.get("p_fake")
    p_real = videomae_raw.get("p_real")
    tau_real = videomae_raw.get("tau_real", 0.25)
    tau_fake = videomae_raw.get("tau_fake", 0.70)
    version = videomae_raw.get("version", "v4")
    
    if p_fake is None:
        return {
            "available": False,
            "verdict": "UNAVAILABLE",
            "score": None,
            "version": version,
            "thresholds": {"real": tau_real, "fake": tau_fake},
            "explanation": "Spatiotemporal forward pass failed."
        }
        
    if p_fake >= tau_fake:
        verdict = "AI-GENERATED / MANIPULATED"
    elif p_fake <= tau_real:
        verdict = "REAL"
    else:
        verdict = "AMBIGUOUS"
        
    return {
        "available": True,
        "verdict": verdict,
        "score": round(p_fake, 4),
        "p_real": round(p_real, 4) if p_real is not None else None,
        "version": version,
        "thresholds": {"real": tau_real, "fake": tau_fake},
        "explanation": f"VideoMAE {version} calibrated score: {p_fake:.1%}"
    }

def arbitrate_models(
    facial_result: Optional[Dict[str, Any]],
    videomae_raw: Optional[Dict[str, Any]],
    requested_mode: str = "experimental"
) -> Dict[str, Any]:
    """
    Arbitrates evidence between Facial Specialist ViT and Spatiotemporal VideoMAE.
    Returns:
    - final_prediction: 'LIKELY REAL' | 'LIKELY AI-GENERATED / MANIPULATED' | 'INCONCLUSIVE'
    - analysis_status: detailed status string
    - explanation: multi-model forensic rationale
    - evidence_items: forensic evidence tags to append
    - arbitration_meta: full transparency payload for API response
    """
    f_ev = extract_facial_evidence(facial_result)
    v_ev = extract_videomae_evidence(videomae_raw)
    
    evidence_items = []
    
    # -------------------------------------------------------------------------
    # Case 1: VideoMAE is unavailable (e.g. Standard Mode direct facial call)
    # -------------------------------------------------------------------------
    if not v_ev["available"]:
        f_pred = f_ev["raw_prediction"]
        return {
            "prediction": f_pred,
            "analysis_status": facial_result.get("analysis_status", "Completed") if facial_result else "Unavailable",
            "explanation": f_ev["explanation"],
            "evidence_items": evidence_items,
            "arbitration_meta": {
                "arbitrated": False,
                "agreement": None,
                "mode": requested_mode,
                "resolution": "single_detector_facial",
                "facial_evidence": f_ev,
                "videomae_evidence": v_ev
            }
        }
        
    # -------------------------------------------------------------------------
    # Case 2: Facial detector has NO FACES (Animals, Nature, Animation, Machinery)
    # Rule: Missing faces are NOT evidence of REAL.
    # -------------------------------------------------------------------------
    if not f_ev["has_faces"]:
        v_verdict = v_ev["verdict"]
        v_score = v_ev["score"]
        v_ver = v_ev["version"]
        
        if v_verdict == "AI-GENERATED / MANIPULATED":
            pred = "LIKELY AI-GENERATED / MANIPULATED"
            status = "Completed (Spatiotemporal Full-Scene Analysis)"
            expl = (
                f"No human faces detected in video. Experimental Spatiotemporal VideoMAE ({v_ver}) "
                f"identified synthetic generation dynamics across 16 space-time frames (model score: {v_score:.1%}). "
                "Full-scene spatiotemporal continuity indicates generative video synthesis."
            )
            evidence_items.append({
                "category": "Cross-Detector Arbitration",
                "tag": "General Scene Synthesis (No Faces)",
                "anomaly": True,
                "status": "anomaly",
                "description": f"Facial detector found 0 human faces. Spatiotemporal VideoMAE affirmatively classified full scene as AI-generated ({v_score:.1%})."
            })
        elif v_verdict == "REAL":
            pred = "LIKELY REAL"
            status = "Completed (Spatiotemporal Full-Scene Analysis)"
            expl = (
                f"No human faces detected in video. Experimental Spatiotemporal VideoMAE ({v_ver}) "
                f"detected natural camera motion and coherent temporal physics across full-scene frames "
                f"(model score: {v_score:.1%})."
            )
            evidence_items.append({
                "category": "Cross-Detector Arbitration",
                "tag": "Natural Scene Dynamics (No Faces)",
                "anomaly": False,
                "status": "normal",
                "description": f"Facial detector found 0 human faces. Spatiotemporal VideoMAE confirmed natural temporal continuity ({v_score:.1%})."
            })
        else: # AMBIGUOUS
            pred = "INCONCLUSIVE"
            status = "Inconclusive (No Faces & Ambiguous Spatiotemporal Dynamics)"
            expl = (
                f"No human faces detected in video, and experimental VideoMAE ({v_ver}) spatiotemporal score "
                f"({v_score:.1%}) falls within the uncertain decision margin ({v_ev['thresholds']['real']:.0%}-{v_ev['thresholds']['fake']:.0%}). "
                "Evidence is insufficient to reach a conclusive verdict."
            )
            evidence_items.append({
                "category": "Cross-Detector Arbitration",
                "tag": "Insufficient Multi-Model Evidence",
                "anomaly": False,
                "status": "uncertain",
                "description": "0 faces detected and full-scene VideoMAE score is in the ambiguous decision margin."
            })
            
        return {
            "prediction": pred,
            "analysis_status": status,
            "explanation": expl,
            "evidence_items": evidence_items,
            "arbitration_meta": {
                "arbitrated": True,
                "agreement": None,
                "mode": requested_mode,
                "resolution": "non_facial_general_scene",
                "facial_evidence": f_ev,
                "videomae_evidence": v_ev
            }
        }

    # -------------------------------------------------------------------------
    # Case 3: Both Detectors Have Reliable Evidence (Faces Exist & VideoMAE Active)
    # -------------------------------------------------------------------------
    f_verdict = f_ev["verdict"]
    v_verdict = v_ev["verdict"]
    v_score = v_ev["score"]
    v_ver = v_ev["version"]
    f_mean = f_ev["mean_fake_score"]
    f_peak = f_ev["max_fake_score"]
    persons = f_ev["persons_tracked"]
    
    # Sub-case 3A: UNANIMOUS AI-GENERATED / MANIPULATED
    if f_verdict == "AI-GENERATED / MANIPULATED" and v_verdict == "AI-GENERATED / MANIPULATED":
        pred = "LIKELY AI-GENERATED / MANIPULATED"
        status = "Completed (Unanimous Synthesis Detection)"
        expl = (
            f"Both localized facial inspection (peak facial fake score: {f_peak:.1%}) and "
            f"full-scene spatiotemporal VideoMAE (score: {v_score:.1%}) affirmatively detected "
            "synthetic generation signatures across space and time."
        )
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Unanimous AI Detection",
            "anomaly": True,
            "status": "anomaly",
            "description": f"Facial Specialist (peak {f_peak:.1%}) and VideoMAE ({v_score:.1%}) agree on synthetic generation."
        })
        resolution = "unanimous_ai"
        agreement = True

    # Sub-case 3B: UNANIMOUS REAL
    elif f_verdict == "REAL" and v_verdict == "REAL":
        pred = "LIKELY REAL"
        status = "Completed (Unanimous Authentic Verification)"
        expl = (
            f"Both localized facial inspection (no manipulation detected across {persons} tracked individuals, "
            f"mean score: {f_mean:.1%}) and full-scene spatiotemporal VideoMAE (natural score: {v_score:.1%}) "
            "confirm authentic physical recording."
        )
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Unanimous Authentic Verification",
            "anomaly": False,
            "status": "normal",
            "description": f"Both facial analysis ({persons} tracked individuals) and spatiotemporal dynamics ({v_score:.1%}) show genuine video physics."
        })
        resolution = "unanimous_real"
        agreement = True

    # Sub-case 3C: STRONG CONFLICT - Facial says AI, VideoMAE says REAL (Face-Swap on Genuine Background)
    elif f_verdict == "AI-GENERATED / MANIPULATED" and v_verdict == "REAL":
        pred = "INCONCLUSIVE"
        status = "Inconclusive (Conflicting Forensic Signals)"
        expl = (
            f"Model disagreement: Localized facial analysis detected synthetic manipulation (peak facial fake score: {f_peak:.1%}), "
            f"but full-scene spatiotemporal VideoMAE detected natural background dynamics (model score: {v_score:.1%}). "
            "Verdict set to Inconclusive due to conflicting signals between localized facial and full-scene models (characteristic of face-swaps on authentic background footage)."
        )
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Conflicting Signals (Facial AI vs Scene Real)",
            "anomaly": True,
            "status": "anomaly",
            "description": f"Facial detector flagged manipulation (peak {f_peak:.1%}), but VideoMAE scored background motion as real ({v_score:.1%}). Human forensic audit required."
        })
        resolution = "conflict_facial_ai_scene_real"
        agreement = False

    # Sub-case 3D: STRONG CONFLICT - Facial says REAL, VideoMAE says AI (Whole-Scene Diffusion with Coherent Face)
    elif f_verdict == "REAL" and v_verdict == "AI-GENERATED / MANIPULATED":
        pred = "INCONCLUSIVE"
        status = "Inconclusive (Conflicting Forensic Signals)"
        expl = (
            f"Model disagreement: Localized facial analysis found no obvious face-swap manipulation (mean facial score: {f_mean:.1%}), "
            f"but full-scene spatiotemporal VideoMAE detected synthetic generation dynamics across 16 space-time frames (model score: {v_score:.1%}). "
            "Verdict set to Inconclusive due to conflicting signals (characteristic of whole-scene generative diffusion with photorealistic facial rendering)."
        )
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Conflicting Signals (Facial Real vs Scene AI)",
            "anomaly": True,
            "status": "anomaly",
            "description": f"Facial detector scored faces as real ({f_mean:.1%}), but VideoMAE detected generative scene diffusion ({v_score:.1%}). Human forensic audit required."
        })
        resolution = "conflict_facial_real_scene_ai"
        agreement = False

    # Sub-case 3E: Facial says AI, VideoMAE is AMBIGUOUS
    elif f_verdict == "AI-GENERATED / MANIPULATED" and v_verdict == "AMBIGUOUS":
        pred = "LIKELY AI-GENERATED / MANIPULATED"
        status = "Completed (Targeted Facial Manipulation Confirmed)"
        expl = (
            f"Localized facial analysis detected targeted synthetic manipulation (peak fake score: {f_peak:.1%}), "
            f"while full-scene spatiotemporal dynamics remained within the ambiguous margin (VideoMAE score: {v_score:.1%}). "
            "Classification determined by confirmed localized facial tampering."
        )
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Facial Manipulation Confirmed",
            "anomaly": True,
            "status": "anomaly",
            "description": f"Localized facial manipulation affirmatively verified ({f_peak:.1%}). Scene dynamics were borderline ({v_score:.1%})."
        })
        resolution = "facial_ai_scene_ambiguous"
        agreement = True

    # Sub-case 3F: Facial says REAL, VideoMAE is AMBIGUOUS
    elif f_verdict == "REAL" and v_verdict == "AMBIGUOUS":
        # Check if VideoMAE is mild (<= 0.45) or elevated (> 0.45)
        if v_score <= 0.45:
            pred = "LIKELY REAL"
            status = "Completed (Facial Authenticity Confirmed)"
            expl = (
                f"Localized facial analysis confirmed natural facial characteristics across {persons} tracked individuals "
                f"(mean fake score: {f_mean:.1%}). Full-scene VideoMAE score ({v_score:.1%}) remained in the low-uncertain range with no affirmative synthesis signals."
            )
            resolution = "facial_real_scene_mild"
            agreement = True
        else:
            pred = "INCONCLUSIVE"
            status = "Inconclusive (Ambiguous Combined Margin)"
            expl = (
                f"Localized facial analysis appeared natural (mean fake score: {f_mean:.1%}), but full-scene spatiotemporal "
                f"dynamics were elevated (VideoMAE score: {v_score:.1%}), resulting in an ambiguous combined confidence margin."
            )
            resolution = "facial_real_scene_elevated_ambiguous"
            agreement = False
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Facial Real / Scene Ambiguous",
            "anomaly": False,
            "status": "uncertain" if pred == "INCONCLUSIVE" else "normal",
            "description": f"Faces scored real ({f_mean:.1%}), VideoMAE score was {v_score:.1%}."
        })

    # Sub-case 3G: Facial is INCONCLUSIVE, VideoMAE is AI
    elif f_verdict == "INCONCLUSIVE" and v_verdict == "AI-GENERATED / MANIPULATED":
        pred = "LIKELY AI-GENERATED / MANIPULATED"
        status = "Completed (Spatiotemporal Synthesis Confirmed)"
        expl = (
            f"Localized facial analysis was inconclusive ({f_ev['explanation']}), but full-scene spatiotemporal VideoMAE ({v_ver}) "
            f"affirmatively identified synthetic generation dynamics (model score: {v_score:.1%})."
        )
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Spatiotemporal Synthesis Confirmed",
            "anomaly": True,
            "status": "anomaly",
            "description": f"Facial analysis inconclusive, but VideoMAE strongly detected synthetic generation ({v_score:.1%})."
        })
        resolution = "facial_inc_scene_ai"
        agreement = True

    # Sub-case 3H: Facial is INCONCLUSIVE, VideoMAE is REAL
    elif f_verdict == "INCONCLUSIVE" and v_verdict == "REAL":
        pred = "LIKELY REAL"
        status = "Completed (Spatiotemporal Authenticity Confirmed)"
        expl = (
            f"Localized facial analysis was inconclusive ({f_ev['explanation']}), but full-scene spatiotemporal VideoMAE ({v_ver}) "
            f"confirmed natural camera motion and coherent temporal physics (model score: {v_score:.1%})."
        )
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Spatiotemporal Authenticity Confirmed",
            "anomaly": False,
            "status": "normal",
            "description": f"Facial analysis inconclusive, but VideoMAE confirmed natural temporal continuity ({v_score:.1%})."
        })
        resolution = "facial_inc_scene_real"
        agreement = True

    # Sub-case 3I: Both INCONCLUSIVE / AMBIGUOUS
    else:
        pred = "INCONCLUSIVE"
        status = "Inconclusive (Ambiguous Margin on Both Detectors)"
        expl = (
            f"Both detectors returned ambiguous or uncertain scores: facial detector ({f_ev['explanation']}) and "
            f"VideoMAE ({v_score:.1%}). Evidence is insufficient to verify or refute synthesis."
        )
        evidence_items.append({
            "category": "Cross-Detector Arbitration",
            "tag": "Ambiguous Both Detectors",
            "anomaly": False,
            "status": "uncertain",
            "description": f"Neither detector produced high-confidence classification (VideoMAE score: {v_score:.1%})."
        })
        resolution = "both_inconclusive"
        agreement = False

    return {
        "prediction": pred,
        "analysis_status": status,
        "explanation": expl,
        "evidence_items": evidence_items,
        "arbitration_meta": {
            "arbitrated": True,
            "agreement": agreement,
            "mode": requested_mode,
            "resolution": resolution,
            "facial_evidence": f_ev,
            "videomae_evidence": v_ev
        }
    }
