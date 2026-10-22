# 🔍 TruthLens AI — AI-Powered Facial Deepfake Video Detection

**TruthLens AI** is a digital-forensics platform engineered for hackathons and security teams to detect AI-generated facial deepfakes, face swaps, and neural video manipulation through genuine Vision Transformer (ViT) inference.

---

## 🤖 Pretrained AI Model Integration

TruthLens AI runs real neural inference using a verified pretrained Vision Transformer:
- **Model Name**: [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection)
- **Base Architecture**: `ViTForImageClassification` (`google/vit-base-patch16-224-in21k` backbone)
- **Dataset**: Fine-tuned on 140,000 real and manipulated facial images.
- **Verified Class Indices**:
  - `Class 0`: **Real**
  - `Class 1`: **Fake**
- **Face Isolation**: OpenCV Haar Cascade & YuNet facial localization with 1:1 aspect ratio normalization and uniform keyframe sampling.
- **Decision Engine**:
  1. 🛡️ **LIKELY REAL** — No strong manipulation signals detected across analyzed face frames (mean fake score ≤ 35%).
  2. 🚨 **LIKELY AI-GENERATED / MANIPULATED** — Model detects signatures associated with its trained manipulation classes (mean fake score ≥ 55% or peak ≥ 75%).
  3. ⚠️ **INCONCLUSIVE** — Ambiguous intermediate scores (35%–55%), unreadable stream, or zero detected faces.

> **Zero Simulated Measurements**: Unsupported claims regarding corneal glints, FFT discrete frequency grids, or eyelid blink dynamics have been removed. The platform reports only genuine measurements: ViT Mean Fake Probability, Peak Frame Manipulation Score, Face Detection Coverage, per-frame inference breakdown, and measured backend latency.

---

## 🛠️ Dual-Engine Architecture

TruthLens AI operates two complementary, decoupled forensic inspection engines:

1. **🛡️ Production Mode (Facial Manipulation Specialist)**:
   - **Backbone**: Vision Transformer (`dima806/deepfake_vs_real_image_detection` on `google/vit-base-patch16-224-in21k`)
   - **Domain**: Human facial deepfakes, face swaps, GAN-generated faces (StyleGAN).
   - **Tracking**: Spatial IoU and centroid distance tracking across frames into persistent Person Tracks with inter-frame variance validation.
   - **Scope**: Requires human faces. Honest fallback to `INCONCLUSIVE (No Faces Detected)` on non-human media.

2. **🧪 Experimental Mode (Spatiotemporal VideoMAE v4)**:
   - **Backbone**: 3D Spatiotemporal Transformer (`MCG-NJU/videomae-base`, frozen 86.8M parameters)
   - **Head**: `SpatiotemporalTemporalDynamicsHeadV4` (3072-d multi-scale feature pooling: mean, standard deviation, temporal step-differences, and spatial patch dispersion)
   - **Scope**: Full-scene general video authenticity (people, animals, animation, nature, machinery, generative diffusion like Sora, Kling, Runway).

3. **⚖️ Forensic Arbitration & Consensus Layer**:
   - Implemented in `backend/arbitrator.py`.
   - **Safe Disagreement**: When localized facial analysis and full-scene background dynamics conflict, the engine outputs **`INCONCLUSIVE (Conflicting Forensic Signals)`** rather than silently defaulting to REAL.
   - **Transparent Telemetry**: Side-by-side exposure of both detector verdicts, individual confidence scores, and arbitration reasoning in the UI and API response.

---

## 📦 Project Structure

```text
truthlens-ai/
├── backend/
│   ├── arbitrator.py            # Multi-model forensic consensus & arbitration layer
│   ├── detector.py              # Production Facial ViT pipeline & face tracker
│   ├── experimental_detector.py # Spatiotemporal VideoMAE (v4) dynamics engine
│   ├── main.py                  # FastAPI application with /health and /api/analyze
│   ├── requirements.txt         # Python dependencies
│   └── weights/                 # Lightweight classification heads (< 5MB)
│       └── videomae_general_head_v4.pt # Active V4 multi-scale dynamics head
├── src/
│   ├── components/
│   │   ├── Header.jsx           # Navigation, live backend status, mode selector
│   │   ├── Hero.jsx             # Architecture & detection scope overview
│   │   ├── VideoUploader.jsx    # Drag-and-drop zone with type & size validation
│   │   ├── VideoPreview.jsx     # HTML5 video preview & metadata extraction
│   │   ├── AnalysisProgress.jsx # Multi-stage neural scan progress indicator
│   │   ├── ResultsView.jsx      # Consensus dashboard, meter, per-frame breakdown
│   │   ├── BackendErrorBanner.jsx # Connection diagnostics
│   │   ├── FastApiGuideModal.jsx # Developer reference modal
│   │   ├── FeaturesSection.jsx  # Neural detection architecture explanation
│   │   └── Footer.jsx           # Attribution & scope notices
│   ├── services/
│   │   └── api.js               # API client streaming payloads to FastAPI
│   ├── utils/
│   │   └── fileValidation.js    # Video format, size validation & metadata extractor
│   ├── App.jsx                  # Application coordinator & state machine
│   ├── App.css                  # Forensic styling & component layouts
│   └── index.css                # Design tokens, color palette, animations, resets
├── video_genuine.mp4            # Sample authentic test video (yields LIKELY REAL)
├── video_manipulated.mp4        # Sample face-swap test video (yields LIKELY MANIPULATED)
├── test_api.py                  # API endpoint validation
├── test_arbitrator_unit.py      # Arbitration unit tests
├── test_regression_false_real.py# Safe disagreement regression test
├── test_end_to_end_verification.py # Full 10-point E2E verification suite
├── package.json                 # NPM dependencies & scripts
├── vite.config.js               # Vite configuration
└── .gitignore                   # Production ignore policy
```

---

## ⚡ Quickstart Guide

### 1. Start the FastAPI Backend
```bash
# In the project root (e:\truthlens-ai)
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```
- Health Check: `http://localhost:8000/health`
- Swagger Docs: `http://localhost:8000/docs`
- Analysis Endpoint: `POST http://localhost:8000/api/analyze`

### 2. Start the React Frontend
```bash
# In another terminal window
npm install
npm run dev
```
Open **`http://localhost:5173/`** in your browser.

### 3. Run Automated Video Detection Tests
```bash
python test_api.py
```
Expected output:
```text
=== 1. TESTING REAL VIDEO API CALL ===
Prediction: LIKELY REAL
Model Fake Score: ~0.23 (23.1%)
Status: Completed

=== 2. TESTING MANIPULATED VIDEO API CALL ===
Prediction: LIKELY AI-GENERATED / MANIPULATED
Model Fake Score: ~0.64 (64.4%)
Status: Completed
```

---

## 📡 API Response Schema

```json
{
  "status": "success",
  "prediction": "LIKELY AI-GENERATED / MANIPULATED",
  "model_score": 0.6442,
  "analysis_status": "Completed",
  "explanation": "Pretrained Vision Transformer (ViT) detected synthetic manipulation signatures across 12 of 12 analyzed face frames (mean fake score: 64.4%, peak: 64.9%).",
  "video_details": {
    "filename": "sample_video.mp4",
    "duration": 2.0,
    "resolution": "256x256",
    "fps": 15.0,
    "total_frames": 30,
    "frames_analyzed": 12,
    "faces_detected": 12
  },
  "evidence": [
    {
      "category": "Vision Transformer Mean Fake Score",
      "tag": "Score: 64.4%",
      "anomaly": true,
      "status": "anomaly",
      "description": "Aggregated mean fake probability of 64.4% across 12 evaluated face frames (Class 1 'Fake' logit probability)."
    }
  ],
  "metrics": {
    "mean_fake_score": 0.6442,
    "max_fake_score": 0.6491,
    "min_fake_score": 0.6401,
    "score_std_dev": 0.0035,
    "face_detection_rate": 1.0
  },
  "frame_analysis": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "face_detected": true,
      "fake_score": 0.6421,
      "real_score": 0.3579
    }
  ],
  "latency_ms": 1184,
  "timestamp": "2026-10-20T21:38:00Z",
  "model_info": {
    "name": "dima806/deepfake_vs_real_image_detection",
    "architecture": "ViTForImageClassification (google/vit-base-patch16-224-in21k)",
    "classes": {"0": "Real", "1": "Fake"}
  }
}
```

---

## ⚖️ AI Disclaimer & Verification Notice

AI deepfake detection models are probabilistic and can make mistakes (including false positives and false negatives). A classification of **LIKELY REAL** indicates that this model detected no strong manipulation signals, not that the video is conclusively authentic. These results should serve as supporting forensic signals alongside qualified human forensic examination.
