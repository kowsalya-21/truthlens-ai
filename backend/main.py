"""
TruthLens AI - Python FastAPI Forensic Backend Service
Real Pretrained Deepfake Detection Model Integration
Model: dima806/deepfake_vs_real_image_detection (Vision Transformer)
"""

import os
import time
import tempfile
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.detector import analyze_video_file, MODEL_NAME
from backend.experimental_detector import analyze_video_experimental

app = FastAPI(
    title="TruthLens AI Forensic API",
    description="Real-time facial deepfake video detection and experimental spatiotemporal authenticity analysis",
    version="2.1.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Health ping confirming backend status and model availability."""
    return {
        "status": "online",
        "service": "TruthLens AI Forensic Engine",
        "version": "4.0.0",
        "production_model": MODEL_NAME,
        "experimental_model": "MCG-NJU/videomae-base + SpatiotemporalTemporalDynamicsHead (v4)",
        "inference_engine": "PyTorch + Transformers (ViT & VideoMAE)",
        "timestamp": time.time()
    }

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...), model: Optional[str] = "standard"):
    """
    Receives video upload, executes real neural model inference,
    and returns genuine forensic metrics.
    
    Parameters:
    - model: 'standard' (Facial Specialist ViT, Default) | 'experimental' (General VideoMAE Spatiotemporal)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = [".mp4", ".webm", ".mov", ".avi", ".mkv"]
    if ext and ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    # Save to temporary file for OpenCV frame decoding
    temp_dir = tempfile.mkdtemp(prefix="truthlens_")
    temp_video_path = os.path.join(temp_dir, f"upload{ext or '.mp4'}")

    try:
        with open(temp_video_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        file_size = os.path.getsize(temp_video_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 Bytes).")

        # Route to chosen detector mode
        m_lower = (model or "").lower()
        if m_lower in ["experimental_v4", "videomae_v4", "v4", "experimental", "videomae"]:
            analysis_result = analyze_video_experimental(temp_video_path, num_frames=16, version="v4")
        elif m_lower in ["experimental_v3", "videomae_v3", "v3"]:
            analysis_result = analyze_video_experimental(temp_video_path, num_frames=16, version="v3")
        elif m_lower in ["experimental_v2", "videomae_v2", "v2"]:
            analysis_result = analyze_video_experimental(temp_video_path, num_frames=16, version="v2")
        elif m_lower in ["experimental_v1", "videomae_v1", "v1"]:
            analysis_result = analyze_video_experimental(temp_video_path, num_frames=16, version="v1")
        else:
            analysis_result = analyze_video_file(temp_video_path, max_samples=12)

        # Attach original filename and upload payload size
        analysis_result["video_details"]["filename"] = file.filename
        analysis_result["video_details"]["file_size"] = file_size
        analysis_result["video_details"]["content_type"] = file.content_type

        return analysis_result

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Neural inference error: {str(exc)}"
        )
    finally:
        # Clean up temporary storage
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

@app.post("/api/analyze/experimental")
async def analyze_video_experimental_route(file: UploadFile = File(...), version: Optional[str] = "v4"):
    """Dedicated endpoint for experimental VideoMAE spatiotemporal analysis."""
    v = (version or "v4").lower()
    return await analyze_video(file=file, model=f"experimental_{v}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
