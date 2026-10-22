"""
TruthLens AI - Experimental General Video Authenticity Detector
Model: MCG-NJU/videomae-base (Spatiotemporal Transformer, 86.8M params, Frozen)
Head: SpatiotemporalAuthenticityHead (Trained on 16-frame spatiotemporal representations)
Scope: General AI Video Detection across non-facial scenes (landscapes, objects, animals, and full-scene diffusion).
Status: Experimental / Research Mode (Isolated from production facial detector).
"""

import os
import time
import math
import cv2
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Any, List
from transformers import VideoMAEImageProcessor, VideoMAEModel

MODEL_NAME = "MCG-NJU/videomae-base"
WEIGHTS_PATH_V1 = os.path.join(os.path.dirname(__file__), "weights", "videomae_general_head.pt")
WEIGHTS_PATH_V2 = os.path.join(os.path.dirname(__file__), "weights", "videomae_general_head_v2.pt")
WEIGHTS_PATH_V3 = os.path.join(os.path.dirname(__file__), "weights", "videomae_general_head_v3.pt")
WEIGHTS_PATH_V4 = os.path.join(os.path.dirname(__file__), "weights", "videomae_general_head_v4.pt")
WEIGHTS_PATH = WEIGHTS_PATH_V4

_processor = None
_backbone = None
_head_v1 = None
_head_v2 = None
_head_v3 = None
_head_v4 = None
_v1_config = None
_v2_config = None
_v3_config = None
_v4_config = None

class SpatiotemporalAuthenticityHead(nn.Module):
    """V1 100k-parameter authenticity classifier head (768-d input)."""
    def __init__(self, in_dim=768, hidden_dim=128, out_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class SpatiotemporalTemporalDynamicsHeadV2(nn.Module):
    """
    V2 & V3 Multi-Scale Spatiotemporal Dynamics Head (2304-d input).
    Processes slice mean (768), slice standard deviation (768), and slice step-difference (768).
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

class SpatiotemporalTemporalDynamicsHeadV4(nn.Module):
    """
    V4 Enhanced Multi-Scale Spatiotemporal Dynamics Classifier (3072-d input).
    Processes slice mean (768), slice standard deviation (768), slice step-difference (768),
    and intra-frame spatial patch dispersion (768).
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

def get_experimental_videomae(version: str = "v4"):
    """Lazy loads VideoMAE backbone and chosen authenticity head (v1, v2, v3, or v4)."""
    global _processor, _backbone, _head_v1, _head_v2, _head_v3, _head_v4, _v1_config, _v2_config, _v3_config, _v4_config
    if _processor is None or _backbone is None:
        _processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
        _backbone = VideoMAEModel.from_pretrained(MODEL_NAME)
        _backbone.eval()
        for p in _backbone.parameters():
            p.requires_grad = False

    if version == "v4":
        if _head_v4 is None:
            if not os.path.exists(WEIGHTS_PATH_V4):
                raise FileNotFoundError(f"V4 checkpoint missing at '{WEIGHTS_PATH_V4}'.")
            checkpoint = torch.load(WEIGHTS_PATH_V4, map_location="cpu", weights_only=False)
            _v4_config = checkpoint.get("config", {})
            _head_v4 = SpatiotemporalTemporalDynamicsHeadV4(
                in_dim=3072,
                hidden_dim1=384,
                hidden_dim2=96,
                out_dim=2,
                dropout=0.25
            )
            _head_v4.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
            _head_v4.eval()
        return _processor, _backbone, _head_v4, _v4_config or {}
    elif version == "v3":
        if _head_v3 is None:
            if not os.path.exists(WEIGHTS_PATH_V3):
                raise FileNotFoundError(f"V3 checkpoint missing at '{WEIGHTS_PATH_V3}'.")
            checkpoint = torch.load(WEIGHTS_PATH_V3, map_location="cpu", weights_only=False)
            _v3_config = checkpoint.get("config", {})
            _head_v3 = SpatiotemporalTemporalDynamicsHeadV2(
                in_dim=2304,
                hidden_dim1=256,
                hidden_dim2=64,
                out_dim=2,
                dropout=0.25
            )
            _head_v3.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
            _head_v3.eval()
        return _processor, _backbone, _head_v3, _v3_config or {}
    elif version == "v2":
        if _head_v2 is None:
            if not os.path.exists(WEIGHTS_PATH_V2):
                raise FileNotFoundError(f"V2 checkpoint missing at '{WEIGHTS_PATH_V2}'.")
            checkpoint = torch.load(WEIGHTS_PATH_V2, map_location="cpu", weights_only=False)
            _v2_config = checkpoint.get("config", {})
            _head_v2 = SpatiotemporalTemporalDynamicsHeadV2(
                in_dim=2304,
                hidden_dim1=256,
                hidden_dim2=64,
                out_dim=2,
                dropout=0.3
            )
            _head_v2.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
            _head_v2.eval()
        return _processor, _backbone, _head_v2, _v2_config or {}
    else:
        if _head_v1 is None:
            if not os.path.exists(WEIGHTS_PATH_V1):
                raise FileNotFoundError(f"V1 checkpoint missing at '{WEIGHTS_PATH_V1}'.")
            checkpoint = torch.load(WEIGHTS_PATH_V1, map_location="cpu", weights_only=False)
            _v1_config = checkpoint.get("config", {})
            _head_v1 = SpatiotemporalAuthenticityHead(in_dim=768, hidden_dim=128, out_dim=2, dropout=0.2)
            _head_v1.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
            _head_v1.eval()
        return _processor, _backbone, _head_v1, _v1_config or {}

def sample_uniform_frames(video_path: str, num_frames: int = 16):
    """Uniformly samples num_frames from video clip and returns PIL images + metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], 0, 0.0, 0, 0, 0.0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if total_frames > 0 else 0.0
    
    if total_frames <= 0:
        cap.release()
        return None, [], 0, 0.0, 0, 0, 0.0
    indices = [int(i * (total_frames - 1) / max(1, num_frames - 1)) for i in range(num_frames)]
    indices = sorted(list(set(indices)))
    while len(indices) < num_frames:
        indices.append(indices[-1])
        
    frames_pil = []
    frame_records = []
    for f_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        ts = round(f_idx / fps, 2)
        if ret and frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(rgb))
            frame_records.append({
                "frame_index": f_idx,
                "timestamp_sec": ts,
                "status": "extracted"
            })
        else:
            frames_pil.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            frame_records.append({
                "frame_index": f_idx,
                "timestamp_sec": ts,
                "status": "fallback"
            })
            
    cap.release()
    return frames_pil, frame_records, total_frames, fps, width, height, duration

def analyze_video_experimental(video_path: str, num_frames: int = 16, version: str = "v4") -> Dict[str, Any]:
    """
    Experimental spatiotemporal video authenticity evaluation.
    Evaluates full-scene temporal continuity across 16 frames using VideoMAE.
    Does not depend on facial presence.
    Supports version="v4" (recommended candidate), version="v3", version="v2", and version="v1" (baseline).
    """
    start_time = time.perf_counter()
    
    frames_pil, frame_records, total_frames, fps, width, height, duration = sample_uniform_frames(video_path, num_frames)
    if frames_pil is None or len(frames_pil) == 0:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "status": "error",
            "prediction": "INCONCLUSIVE",
            "model_score": None,
            "analysis_status": "Failed (Unreadable Video)",
            "explanation": "Could not read video frames for spatiotemporal analysis.",
            "video_details": {"filename": os.path.basename(video_path)},
            "evidence": [],
            "latency_ms": latency_ms,
            "model_info": {
                "name": f"MCG-NJU/videomae-base (Experimental Spatiotemporal Head {version})",
                "mode": "experimental",
                "version": version,
                "error": "Unreadable video"
            }
        }
        
    processor, backbone, head, config = get_experimental_videomae(version=version)
    inputs = processor(frames_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = backbone(**inputs)
        if version == "v4":
            tokens = outputs.last_hidden_state # [1, 1568, 768]
            tubelets = tokens.view(1, 8, 196, 768)
            slices = tubelets.mean(dim=2).squeeze(0) # [8, 768]
            slice_mean = slices.mean(dim=0)
            slice_std = slices.std(dim=0)
            slice_diff = (slices[1:] - slices[:-1]).abs().mean(dim=0)
            spatial_dispersion = tubelets.std(dim=2).mean(dim=1).squeeze(0)
            feat = torch.cat([slice_mean, slice_std, slice_diff, spatial_dispersion], dim=0).unsqueeze(0) # [1, 3072]
            logits = head(feat)
            default_t = 0.85
            default_tr = 0.25
            default_tf = 0.70
            temp = float(config.get("temperature", default_t))
            probs = torch.softmax(logits / temp, dim=-1)
            tau_real = float(config.get("thresholds", {}).get("real", default_tr))
            tau_fake = float(config.get("thresholds", {}).get("fake", default_tf))
            is_calibrated = True
            score_type = "temperature_calibrated_softmax"
            model_name = f"MCG-NJU/videomae-base (Multi-Scale Temporal Dynamics Head {version})"
            arch = f"3D VideoMAE ViT-B (86.8M Frozen) + 1.2M Temporal Dynamics Head ({version})"
        elif version in ["v2", "v3"]:
            tokens = outputs.last_hidden_state # [1, 1568, 768]
            slices = tokens.view(1, 8, 196, 768).mean(dim=2).squeeze(0) # [8, 768]
            slice_mean = slices.mean(dim=0)
            slice_std = slices.std(dim=0)
            slice_diff = (slices[1:] - slices[:-1]).abs().mean(dim=0)
            feat = torch.cat([slice_mean, slice_std, slice_diff], dim=0).unsqueeze(0) # [1, 2304]
            logits = head(feat)
            default_t = 1.40 if version == "v3" else 1.1408
            default_tr = 0.25 if version == "v3" else 0.40
            default_tf = 0.60
            temp = float(config.get("temperature", default_t))
            probs = torch.softmax(logits / temp, dim=-1)
            tau_real = float(config.get("thresholds", {}).get("real", default_tr))
            tau_fake = float(config.get("thresholds", {}).get("fake", default_tf))
            is_calibrated = True
            score_type = "temperature_calibrated_softmax"
            model_name = f"MCG-NJU/videomae-base (Multi-Scale Temporal Dynamics Head {version})"
            arch = f"3D VideoMAE ViT-B (86.8M Frozen) + 606k Temporal Dynamics Head ({version})"
        else:
            emb = outputs.last_hidden_state.mean(dim=1)
            logits = head(emb)
            probs = torch.softmax(logits, dim=-1)
            tau_real = 0.45
            tau_fake = 0.55
            is_calibrated = False
            score_type = "uncalibrated_model_score"
            model_name = "MCG-NJU/videomae-base (Experimental Spatiotemporal Head v1)"
            arch = "3D VideoMAE ViT-B (86.8M Frozen) + 100k Authenticity Head (v1)"
        
    p_real = float(probs[0, 0].item())
    p_fake = float(probs[0, 1].item())
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    
    # Run specialized facial consistency check if faces exist (sample 12 frames for full coverage)
    from backend.detector import analyze_video_file
    from backend.arbitrator import arbitrate_models
    face_check = analyze_video_file(video_path, max_samples=12)
    
    videomae_raw = {
        "p_fake": p_fake,
        "p_real": p_real,
        "tau_real": tau_real,
        "tau_fake": tau_fake,
        "version": version
    }
    
    arb = arbitrate_models(face_check, videomae_raw, requested_mode="experimental")
    prediction = arb["prediction"]
    analysis_status = arb["analysis_status"]
    explanation = arb["explanation"]
    evidence = list(arb["evidence_items"])

    evidence.extend([
        {
            "category": "Spatiotemporal Temporal Dynamics",
            "tag": f"Model Score: {p_fake:.1%}",
            "anomaly": p_fake >= 0.50,
            "status": "anomaly" if p_fake >= 0.50 else "normal",
            "description": (
                f"Evaluated 16 spatiotemporal tubelet frames using VideoMAE-base ({arch}). "
                f"{'Temperature-calibrated score: ' + f'{p_fake:.2%}' if is_calibrated else 'Uncalibrated model score: ' + f'{p_fake:.2%}'}."
            )
        },
        {
            "category": "Model Architecture & Mode",
            "tag": f"Experimental VideoMAE ({version.upper()})",
            "anomaly": False,
            "status": "normal",
            "description": (
                f"Running in Experimental Research Mode ({arch}). Model operates across arbitrary visual scenes "
                "(landscapes, animals, objects, people) without requiring facial presence."
            )
        },
        {
            "category": "Inference Latency",
            "tag": f"{latency_ms} ms",
            "anomaly": False,
            "status": "normal",
            "description": f"Processed 16 spatiotemporal frames on CPU in {latency_ms} ms."
        }
    ])
    
    # Build per-frame breakdown for UI compatibility
    frame_analysis = []
    for fr in frame_records:
        frame_analysis.append({
            "frame_index": fr["frame_index"],
            "timestamp_sec": fr["timestamp_sec"],
            "face_detected": False, # Scene-wide analysis
            "face_count": 0,
            "fake_score": round(p_fake, 4),
            "real_score": round(p_real, 4),
            "bbox": {"method": "videomae_spatiotemporal_fullframe", "angle": 0}
        })
        
    return {
        "status": "success",
        "prediction": prediction,
        "model_score": round(p_fake, 4),
        "analysis_status": analysis_status,
        "explanation": explanation,
        "video_details": {
            "filename": os.path.basename(video_path),
            "duration": round(duration, 2),
            "resolution": f"{width}x{height}",
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "frames_sampled": len(frames_pil),
            "frames_analyzed": len(frames_pil),
            "frames_with_faces": face_check.get("video_details", {}).get("frames_with_faces", 0) if face_check else 0,
            "faces_detected": face_check.get("video_details", {}).get("faces_detected", 0) if face_check else 0,
            "faces_analyzed": face_check.get("video_details", {}).get("faces_analyzed", 0) if face_check else 0
        },
        "detectors": {
            "facial_specialist": arb["arbitration_meta"]["facial_evidence"],
            "spatiotemporal_videomae": arb["arbitration_meta"]["videomae_evidence"]
        },
        "arbitration": arb["arbitration_meta"],
        "evidence": evidence,
        "metrics": {
            "mean_fake_score": round(p_fake, 4),
            "model_score": round(p_fake, 4),
            "real_probability": round(p_real, 4),
            "fake_probability": round(p_fake, 4),
            "score_type": score_type,
            "is_calibrated": is_calibrated,
            "model_type": arch,
            "mode": "Experimental",
            "version": version,
            "backbone": MODEL_NAME
        },
        "frame_analysis": frame_analysis,
        "latency_ms": latency_ms,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_info": {
            "name": model_name,
            "architecture": arch,
            "target_scope": "Full-Scene AI Video Diffusion & Spatiotemporal Inconsistency",
            "mode": "experimental",
            "version": version,
            "classes": {"0": "Real", "1": "Fake"}
        }
    }
