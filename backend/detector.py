"""
TruthLens AI - Real Deepfake Detection Pipeline
Model: HuggingFace ViT ('dima806/deepfake_vs_real_image_detection')
Trained on 140k real and fake facial images (Deepfake and Real Images Dataset).

Verified Model Configuration (model.config.id2label):
- Class 0: 'Real'
- Class 1: 'Fake'
Preprocessing: ViTImageProcessor (224x224, normalized with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
Scope: Facial manipulation and GAN face-swap detection on facial crops.
Multi-Face Support: Analyzes all confidently detected faces per frame, preventing duplicate detections across rotations.
"""

import os
import time
import math
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

# Lazy-loaded singletons
_processor = None
_model = None
_frontal_cascade = None
_profile_cascade = None

def get_face_cascades():
    """Load OpenCV Haar Cascades for frontal and profile face detection."""
    global _frontal_cascade, _profile_cascade
    if _frontal_cascade is None:
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _frontal_cascade = cv2.CascadeClassifier(frontal_path)
    if _profile_cascade is None:
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        _profile_cascade = cv2.CascadeClassifier(profile_path)
    return _frontal_cascade, _profile_cascade

def get_deepfake_model():
    """Lazy load Vision Transformer deepfake model and verify classes."""
    global _processor, _model
    if _processor is None or _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        
        # Verify classes from model config
        id2label = _model.config.id2label
        assert id2label.get(0) == "Real" or id2label.get("0") == "Real", f"Unexpected class 0: {id2label}"
        assert id2label.get(1) == "Fake" or id2label.get("1") == "Fake", f"Unexpected class 1: {id2label}"
    return _processor, _model

def detect_skin_ratio(img_bgr):
    """Computes skin-tone pixel percentage in YCrCb color space."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    return float(np.sum(mask > 0) / (img_bgr.shape[0] * img_bgr.shape[1]))

def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) of two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0

def extract_all_face_crops(frame_bgr, frontal_cascade, profile_cascade):
    """
    Extracts all distinct, confidently detected facial regions from a video frame.
    1. Multi-orientation scan (0, 90 CW, 270 CW) using frontal and profile cascades.
    2. Maps bounding boxes back to original image space and filters duplicates via IoU (> 0.30).
    3. Corrects orientation so every face crop is upright for the ViT model.
    4. Handles pre-cropped benchmark square face videos (aspect ratio ~1:1 and size <= 512).
    5. Returns list of detected face dictionaries: [{ 'crop_rgb', 'orig_box', 'bbox' }, ...].
    """
    h, w = frame_bgr.shape[:2]
    aspect_ratio = w / float(h)
    is_square_crop = 0.9 <= aspect_ratio <= 1.1 and max(h, w) <= 512

    orientations = [(0, None), (90, cv2.ROTATE_90_CLOCKWISE), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    detected_faces = []

    for angle, rot_code in orientations:
        rot_frame = frame_bgr if rot_code is None else cv2.rotate(frame_bgr, rot_code)
        rh, rw = rot_frame.shape[:2]
        gray = cv2.cvtColor(rot_frame, cv2.COLOR_BGR2GRAY)

        scale = 640.0 / max(rh, rw) if max(rh, rw) > 640 else 1.0
        small_gray = cv2.resize(gray, (int(rw * scale), int(rh * scale))) if scale != 1.0 else gray

        faces = frontal_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36))
        method = "haar_frontal"

        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36))
            method = "haar_profile"

        for f in faces:
            rx = int(f[0] / scale)
            ry = int(f[1] / scale)
            rw_box = int(f[2] / scale)
            rh_box = int(f[3] / scale)

            # Map coordinates back to unrotated original image to prevent duplicates across angles
            if angle == 0:
                ox, oy, ow, oh = rx, ry, rw_box, rh_box
            elif angle == 90:
                ox = ry
                oy = h - (rx + rw_box)
                ow = rh_box
                oh = rw_box
            elif angle == 270:
                ox = w - (ry + rh_box)
                oy = rx
                ow = rh_box
                oh = rw_box

            orig_box = (ox, oy, ow, oh)

            # Deduplication: check overlap with already-accepted faces in this frame
            if any(compute_iou(orig_box, d['orig_box']) > 0.30 for d in detected_faces):
                continue

            # Expand bounding box by 25% with margin for facial perimeter context
            side = int(max(rw_box, rh_box) * 1.25)
            cx, cy = rx + rw_box // 2, ry + rh_box // 2

            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(rw, cx + side // 2)
            y2 = min(rh, cy + side // 2)

            crop = rot_frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            detected_faces.append({
                "crop_rgb": crop_rgb,
                "orig_box": orig_box,
                "bbox": {
                    "x": ox,
                    "y": oy,
                    "width": ow,
                    "height": oh,
                    "method": f"{method}_{angle}deg",
                    "angle": angle
                }
            })

    # Fallback strictly for pre-cropped square benchmark clips (e.g. 256x256 FFHQ/DFDC crops)
    if len(detected_faces) == 0 and is_square_crop:
        skin_ratio = detect_skin_ratio(frame_bgr)
        if skin_ratio >= 0.08:
            crop_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detected_faces.append({
                "crop_rgb": crop_rgb,
                "orig_box": (0, 0, w, h),
                "bbox": {
                    "x": 0,
                    "y": 0,
                    "width": w,
                    "height": h,
                    "method": "pre_cropped_square_face",
                    "angle": 0
                }
            })

    return detected_faces

def extract_face_crop(frame_bgr, frontal_cascade, profile_cascade):
    """Backward-compatible helper returning the primary (largest) face or None."""
    all_faces = extract_all_face_crops(frame_bgr, frontal_cascade, profile_cascade)
    if len(all_faces) == 0:
        return None, False, {"method": "none", "angle": 0}
    # Pick largest face
    primary = max(all_faces, key=lambda f: f["bbox"]["width"] * f["bbox"]["height"])
    return primary["crop_rgb"], True, primary["bbox"]

def track_faces_across_frames(frame_results, frame_w, frame_h):
    """
    Associates face detections across sampled frames into persistent Person Tracks.
    Uses spatial IoU and normalized centroid distance to link the same person across time.
    Retains person-level statistics to distinguish legitimate multi-person differences
    from true temporal instability.
    """
    tracks = []
    diag = math.sqrt(frame_w ** 2 + frame_h ** 2) if (frame_w > 0 and frame_h > 0) else 1000.0

    for fr in frame_results:
        if not fr.get("face_detected") or not fr.get("faces"):
            continue
        
        f_idx = fr["frame_index"]
        ts = fr["timestamp_sec"]

        for face in fr["faces"]:
            bbox = face["bbox"]
            box = (bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 50), bbox.get("height", 50))
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0

            best_track = None
            best_match_score = -1.0

            for tr in tracks:
                last_obs = tr["observations"][-1]
                last_box = last_obs["box"]
                iou = compute_iou(box, last_box)
                last_cx = last_box[0] + last_box[2] / 2.0
                last_cy = last_box[1] + last_box[3] / 2.0
                dist = math.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2) / diag

                if iou > 0.15:
                    match_metric = iou + 1.0
                elif dist < 0.22:
                    match_metric = 1.0 - dist
                else:
                    match_metric = -1.0

                if match_metric > best_match_score:
                    best_match_score = match_metric
                    best_track = tr

            obs_record = {
                "frame_index": f_idx,
                "timestamp_sec": ts,
                "fake_score": face["fake_score"],
                "real_score": face["real_score"],
                "bbox": bbox,
                "box": box
            }

            if best_track is not None and best_match_score > 0:
                best_track["observations"].append(obs_record)
                best_track["scores"].append(face["fake_score"])
            else:
                new_track = {
                    "person_id": len(tracks) + 1,
                    "observations": [obs_record],
                    "scores": [face["fake_score"]],
                    "primary_bbox": bbox
                }
                tracks.append(new_track)

    person_summaries = []
    for tr in tracks:
        sc = tr["scores"]
        m_fake = sum(sc) / len(sc)
        mx_fake = max(sc)
        mn_fake = min(sc)
        var = sum((s - m_fake) ** 2 for s in sc) / len(sc) if len(sc) > 1 else 0.0
        sd = math.sqrt(var)
        best_bb = max(tr["observations"], key=lambda o: o["box"][2] * o["box"][3])["bbox"]

        person_summaries.append({
            "person_id": tr["person_id"],
            "count": len(sc),
            "scores": sc,
            "mean_fake_score": round(m_fake, 4),
            "max_fake_score": round(mx_fake, 4),
            "min_fake_score": round(mn_fake, 4),
            "std_dev": round(sd, 4),
            "primary_bbox": best_bb
        })

    return person_summaries

def analyze_video_file(video_path: str, max_samples: int = 12):
    """
    Performs genuine deepfake detection on an uploaded video file.
    Analyzes all confidently detected faces per frame, preventing duplicate face detections.
    
    Returns:
    - prediction: "LIKELY REAL" | "LIKELY AI-GENERATED / MANIPULATED" | "INCONCLUSIVE"
    - model_score: raw float representing mean fake probability across all analyzed faces [0.0 to 1.0]
    - analysis_status: "Completed" | "Failed" | "Inconclusive (...)"
    - explanation: textual summary of why this decision was reached
    - video_details: metadata including frames sampled, faces detected, and faces analyzed
    - evidence: genuine computed measurements
    - latency_ms: actual measured execution time
    - model_info: exact model name, architecture, scope, and class mapping
    """
    start_time = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "status": "error",
            "prediction": "INCONCLUSIVE",
            "model_score": None,
            "analysis_status": "Failed (Unreadable Video)",
            "explanation": "Could not open or decode the video file. Please ensure it is a valid MP4/WebM/MOV video.",
            "video_details": {
                "filename": os.path.basename(video_path),
                "frames_sampled": 0,
                "frames_analyzed": 0,
                "frames_with_faces": 0,
                "faces_detected": 0,
                "faces_analyzed": 0
            },
            "evidence": [],
            "metrics": {},
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_info": {
                "name": MODEL_NAME,
                "architecture": "ViTForImageClassification (google/vit-base-patch16-224-in21k)",
                "target_scope": "Facial Manipulation / Face-Swap Detection",
                "classes": {"0": "Real", "1": "Fake"},
                "mode": "standard"
            }
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if total_frames > 0 else 0.0

    if total_frames <= 0:
        total_frames = 60

    num_samples = min(max_samples, max(1, total_frames))
    indices = [int(i * (total_frames - 1) / max(1, num_samples - 1)) for i in range(num_samples)]
    indices = sorted(list(set(indices)))

    processor, model = get_deepfake_model()
    frontal_cascade, profile_cascade = get_face_cascades()

    frame_results = []
    all_analyzed_faces = []
    frames_with_faces_count = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        timestamp_sec = round(idx / fps, 2)
        face_crops = extract_all_face_crops(frame_bgr, frontal_cascade, profile_cascade)

        if len(face_crops) > 0:
            frames_with_faces_count += 1
            frame_face_records = []

            for face_idx, face_obj in enumerate(face_crops):
                pil_img = Image.fromarray(face_obj["crop_rgb"])
                inputs = processor(images=pil_img, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                p_real = float(probs[0, 0].item())
                p_fake = float(probs[0, 1].item())

                record = {
                    "face_index": face_idx,
                    "fake_score": round(p_fake, 4),
                    "real_score": round(p_real, 4),
                    "bbox": face_obj["bbox"]
                }
                frame_face_records.append(record)
                all_analyzed_faces.append(record)

            # Frame-level summary: pick peak fake face for security/manipulation detection
            peak_face = max(frame_face_records, key=lambda r: r["fake_score"])
            primary_bbox = max(frame_face_records, key=lambda r: r["bbox"]["width"] * r["bbox"]["height"])["bbox"]

            frame_results.append({
                "frame_index": idx,
                "timestamp_sec": timestamp_sec,
                "face_detected": True,
                "face_count": len(frame_face_records),
                "fake_score": peak_face["fake_score"],
                "real_score": peak_face["real_score"],
                "bbox": primary_bbox,
                "faces": frame_face_records
            })
        else:
            frame_results.append({
                "frame_index": idx,
                "timestamp_sec": timestamp_sec,
                "face_detected": False,
                "face_count": 0,
                "fake_score": None,
                "real_score": None,
                "bbox": {"method": "none", "angle": 0},
                "faces": []
            })

    cap.release()
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    total_faces_detected = len(all_analyzed_faces)
    total_faces_analyzed = len(all_analyzed_faces)

    model_info_payload = {
        "name": MODEL_NAME,
        "architecture": "ViTForImageClassification (google/vit-base-patch16-224-in21k)",
        "target_scope": "Facial Manipulation / Face-Swap Deepfake Detection",
        "classes": {"0": "Real", "1": "Fake"},
        "mode": "standard",
        "scope_notice": (
            "Model trained on facial manipulation datasets (StyleGAN & face-swaps). "
            "Not trained to detect full-scene text-to-video diffusion generation (e.g. Sora, Kling, Runway)."
        )
    }

    # Case A: No faces detected across any sampled frame
    if total_faces_detected == 0:
        return {
            "status": "success",
            "prediction": "INCONCLUSIVE",
            "model_score": None,
            "analysis_status": "Inconclusive (No Faces Detected)",
            "explanation": (
                "No human faces were detected across the sampled video frames. "
                "TruthLens AI evaluates facial manipulation and requires visible facial regions."
            ),
            "video_details": {
                "filename": os.path.basename(video_path),
                "duration": round(duration, 2),
                "resolution": f"{width}x{height}",
                "fps": round(fps, 1),
                "total_frames": total_frames,
                "frames_sampled": len(indices),
                "frames_analyzed": len(frame_results),
                "frames_with_faces": 0,
                "faces_detected": 0,
                "faces_analyzed": 0
            },
            "evidence": [
                {
                    "category": "Facial Detection Verification",
                    "tag": "0 Faces Detected",
                    "anomaly": False,
                    "status": "uncertain",
                    "description": "Multi-orientation face detector scanned sampled frames but found no valid facial regions."
                }
            ],
            "metrics": {
                "frames_sampled": len(indices),
                "frames_with_faces": 0,
                "faces_detected": 0,
                "faces_analyzed": 0,
                "face_detection_rate": 0.0,
                "score_type": "vit_class1_softmax_probability",
                "is_calibrated": False,
                "mode": "standard"
            },
            "frame_analysis": frame_results,
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_info": model_info_payload
        }

    # Case B: Insufficient face coverage (< 25% of sampled frames)
    coverage_ratio = frames_with_faces_count / max(1, len(frame_results))
    if frames_with_faces_count < 2 or coverage_ratio < 0.25:
        return {
            "status": "success",
            "prediction": "INCONCLUSIVE",
            "model_score": None,
            "analysis_status": "Inconclusive (Insufficient Face Coverage)",
            "explanation": (
                f"Faces were detected in only {frames_with_faces_count} of {len(frame_results)} frames ({coverage_ratio:.0%}). "
                "Insufficient facial presence to reach a reliable forensic classification."
            ),
            "video_details": {
                "filename": os.path.basename(video_path),
                "duration": round(duration, 2),
                "resolution": f"{width}x{height}",
                "fps": round(fps, 1),
                "total_frames": total_frames,
                "frames_sampled": len(indices),
                "frames_analyzed": len(frame_results),
                "frames_with_faces": frames_with_faces_count,
                "faces_detected": total_faces_detected,
                "faces_analyzed": total_faces_analyzed
            },
            "evidence": [
                {
                    "category": "Facial Detection Coverage",
                    "tag": f"Low: {frames_with_faces_count}/{len(frame_results)} Frames",
                    "anomaly": False,
                    "status": "uncertain",
                    "description": f"Facial presence detected in only {frames_with_faces_count} frames, below the minimum threshold for forensic confidence."
                }
            ],
            "metrics": {
                "frames_sampled": len(indices),
                "frames_with_faces": frames_with_faces_count,
                "faces_detected": total_faces_detected,
                "faces_analyzed": total_faces_analyzed,
                "face_detection_rate": round(coverage_ratio, 2),
                "score_type": "vit_class1_softmax_probability",
                "is_calibrated": False,
                "mode": "standard"
            },
            "frame_analysis": frame_results,
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_info": model_info_payload
        }

    # Track faces across frames into persistent Person Tracks
    person_tracks = track_faces_across_frames(frame_results, width, height)
    total_persons = len(person_tracks)

    # Gather scores across all analyzed faces
    all_face_fake_scores = [f["fake_score"] for f in all_analyzed_faces]
    mean_fake_score = sum(all_face_fake_scores) / len(all_face_fake_scores)
    max_fake_score = max(all_face_fake_scores)
    min_fake_score = min(all_face_fake_scores)
    variance = sum((s - mean_fake_score) ** 2 for s in all_face_fake_scores) / len(all_face_fake_scores)
    std_dev = math.sqrt(variance)

    # Multi-Face and Forensic Decision Logic:
    # 1. Targeted Face-Swap / Specific Manipulated Person Track:
    #    A person with >= 2 observations and mean >= 0.50 (or max >= 0.75 and mean >= 0.40)
    manip_tracks = [
        p for p in person_tracks 
        if (p["count"] >= 2 and p["mean_fake_score"] >= 0.50) or 
           (p["max_fake_score"] >= 0.75 and p["mean_fake_score"] >= 0.40 and p.get("std_dev", 0.0) < 0.28)
    ]
    
    # 2. Track instability check: extreme within-track variance indicates pose/compression noise
    unstable_tracks = [p for p in person_tracks if p["count"] >= 3 and p["std_dev"] >= 0.28]
    
    # 3. Overall multi-frame synthetic generation signal (requires persistent high frames)
    high_manip_frames = sum(1 for s in all_face_fake_scores if s >= 0.60)
    very_high_manip_frames = sum(1 for s in all_face_fake_scores if s >= 0.80)

    if len(manip_tracks) > 0 or (mean_fake_score >= 0.52 and len(unstable_tracks) == 0) or (very_high_manip_frames >= 2 and high_manip_frames >= 3 and len(unstable_tracks) == 0):
        prediction = "LIKELY AI-GENERATED / MANIPULATED"
        analysis_status = "Completed"
        if len(manip_tracks) > 0:
            target_p = max(manip_tracks, key=lambda p: p["mean_fake_score"])
            explanation = (
                f"Pretrained Vision Transformer (ViT) detected targeted facial manipulation on Person #{target_p['person_id']} "
                f"(person mean fake score: {target_p['mean_fake_score']:.1%}, peak: {target_p['max_fake_score']:.1%}). "
                f"Evaluated {total_persons} tracked individuals across {frames_with_faces_count} frames."
            )
        else:
            manip_count = sum(1 for s in all_face_fake_scores if s >= 0.5)
            explanation = (
                f"Pretrained Vision Transformer (ViT) detected synthetic manipulation signatures across "
                f"{manip_count} of {total_faces_analyzed} analyzed faces "
                f"(mean fake score: {mean_fake_score:.1%}, peak: {max_fake_score:.1%})."
            )
    elif len(unstable_tracks) > 0 and len(manip_tracks) == 0 and mean_fake_score < 0.52:
        prediction = "INCONCLUSIVE"
        analysis_status = "Inconclusive (High Inter-Frame Inconsistency)"
        unstable_p = max(unstable_tracks, key=lambda p: p["std_dev"])
        explanation = (
            f"Model returned highly inconsistent scores within Person #{unstable_p['person_id']} track "
            f"(person std dev: {unstable_p['std_dev']:.1%}, overall peak: {max_fake_score:.1%}). "
            "Such within-track divergence indicates extreme lighting/pose variations, compression artifacts, "
            "or out-of-distribution media that this facial-swap model cannot reliably classify."
        )
    elif all(p["mean_fake_score"] <= 0.40 for p in person_tracks) and (max_fake_score < 0.70 or (max_fake_score < 0.75 and mean_fake_score <= 0.30)):
        prediction = "LIKELY REAL"
        analysis_status = "Completed"
        explanation = (
            f"Pretrained Vision Transformer detected no strong facial manipulation signals across "
            f"{total_persons} tracked individuals ({total_faces_analyzed} faces analyzed across {frames_with_faces_count} frames). "
            f"Mean fake score: {mean_fake_score:.1%}, peak: {max_fake_score:.1%}."
        )
    else:
        prediction = "INCONCLUSIVE"
        analysis_status = "Inconclusive (Ambiguous Margin)"
        explanation = (
            f"Model returned ambiguous scores across {total_faces_analyzed} analyzed faces in {total_persons} individuals "
            f"(mean fake score: {mean_fake_score:.1%}, variance: {std_dev:.1%}), "
            "falling within the uncertain decision margin."
        )

    evidence = [
        {
            "category": "Vision Transformer Mean Fake Score",
            "tag": f"Score: {mean_fake_score:.1%}",
            "anomaly": mean_fake_score >= 0.50,
            "status": "anomaly" if mean_fake_score >= 0.50 else "normal",
            "description": (
                f"Aggregated mean fake probability of {mean_fake_score:.1%} across {total_faces_analyzed} "
                f"evaluated faces in {frames_with_faces_count} video frames (ViT Class 1 'Fake' logit probability)."
            )
        },
        {
            "category": "Peak Frame Manipulation Score",
            "tag": f"Peak: {max_fake_score:.1%}",
            "anomaly": max_fake_score >= 0.70,
            "status": "anomaly" if max_fake_score >= 0.70 else ("normal" if max_fake_score < 0.40 else "uncertain"),
            "description": (
                f"Highest individual face fake score was {max_fake_score:.1%} "
                f"(Lowest was {min_fake_score:.1%}, score variance: {std_dev:.2%})."
            )
        },
        {
            "category": "Person Track Multi-Face Analysis",
            "tag": f"{total_persons} Individuals ({total_faces_analyzed} Faces)",
            "anomaly": len(manip_tracks) > 0,
            "status": "anomaly" if len(manip_tracks) > 0 else ("normal" if prediction == "LIKELY REAL" else "uncertain"),
            "description": (
                f"Tracked {total_persons} persistent individual(s) across {frames_with_faces_count} frames. "
                + (f"Identified manipulation on Person #{manip_tracks[0]['person_id']}." if len(manip_tracks) > 0 else "All tracked persons exhibit consistent natural characteristics without manipulation.")
            )
        },
        {
            "category": "Inter-Face Consistency",
            "tag": f"StdDev: {std_dev:.1%}",
            "anomaly": std_dev >= 0.28,
            "status": "anomaly" if std_dev >= 0.28 else "normal",
            "description": (
                f"Overall standard deviation across all faces is {std_dev:.1%}. "
                f"{'High score divergence triggered inconclusive handling.' if prediction.startswith('INCONCLUSIVE') and 'Inconsistency' in analysis_status else 'Score distribution aligned with individual tracks.'}"
            )
        },
        {
            "category": "Inference Latency & Sampling",
            "tag": f"{latency_ms} ms",
            "anomaly": False,
            "status": "normal",
            "description": (
                f"Completed {total_faces_analyzed} forward passes on CPU in {latency_ms}ms "
                f"({round(latency_ms / max(1, total_faces_analyzed), 1)} ms/face average)."
            )
        }
    ]

    return {
        "status": "success",
        "prediction": prediction,
        "model_score": round(mean_fake_score, 4),
        "analysis_status": analysis_status,
        "explanation": explanation,
        "video_details": {
            "filename": os.path.basename(video_path),
            "duration": round(duration, 2),
            "resolution": f"{width}x{height}",
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "frames_sampled": len(indices),
            "frames_analyzed": len(frame_results),
            "frames_with_faces": frames_with_faces_count,
            "faces_detected": total_faces_detected,
            "faces_analyzed": total_faces_analyzed,
            "persons_tracked": total_persons
        },
        "evidence": evidence,
        "metrics": {
            "frames_sampled": len(indices),
            "frames_with_faces": frames_with_faces_count,
            "faces_detected": total_faces_detected,
            "faces_analyzed": total_faces_analyzed,
            "persons_tracked": total_persons,
            "person_tracks": [
                {
                    "person_id": p["person_id"],
                    "count": p["count"],
                    "mean_fake_score": p["mean_fake_score"],
                    "max_fake_score": p["max_fake_score"],
                    "std_dev": p["std_dev"]
                }
                for p in person_tracks
            ],
            "mean_fake_score": round(mean_fake_score, 4),
            "max_fake_score": round(max_fake_score, 4),
            "min_fake_score": round(min_fake_score, 4),
            "score_std_dev": round(std_dev, 4),
            "face_detection_rate": round(frames_with_faces_count / max(1, len(frame_results)), 2),
            "score_type": "vit_class1_softmax_probability",
            "is_calibrated": False,
            "mode": "standard"
        },
        "detectors": {
            "facial_specialist": {
                "available": True,
                "verdict": prediction,
                "score": round(mean_fake_score, 4),
                "faces_detected": total_faces_detected,
                "faces_analyzed": total_faces_analyzed,
                "persons_tracked": total_persons
            },
            "spatiotemporal_videomae": {
                "available": False,
                "verdict": "NOT_RUN",
                "notice": "Standard mode executes facial manipulation specialist. For full-scene spatiotemporal analysis, select Experimental Mode (VideoMAE v4)."
            }
        },
        "arbitration": {
            "arbitrated": False,
            "mode": "standard",
            "resolution": "facial_specialist_only"
        },
        "frame_analysis": frame_results,
        "latency_ms": latency_ms,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_info": model_info_payload
    }
