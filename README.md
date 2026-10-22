# TruthLens AI — AI-Powered Video Authenticity Analysis

TruthLens AI is an AI-powered video analysis platform designed to identify potential facial deepfakes, AI-generated videos, and manipulated media. It combines pretrained vision models with a spatiotemporal video-analysis model to provide evidence-based authenticity assessments.

## Features

- **Production Mode:** Uses a pretrained Vision Transformer (ViT) specialized in facial manipulation detection.
- **Experimental Mode:** Uses VideoMAE with a custom spatiotemporal classification head to analyze general-video signals.
- **Consensus Layer:** Compares detector results and can return `INCONCLUSIVE` when signals conflict.
- **Video Analysis:** Upload a video and inspect the model's prediction and available analysis details.
- **Transparent Results:** Displays model scores and analysis information to help users interpret predictions.

## Technology Stack

- **Frontend:** React, Vite, JavaScript, CSS
- **Backend:** Python, FastAPI
- **AI/ML:** PyTorch, Hugging Face Transformers, Vision Transformer (ViT), VideoMAE
- **Video Processing:** OpenCV

## Project Structure

```text
truthlens-ai/
├── backend/
│   ├── arbitrator.py
│   ├── detector.py
│   ├── experimental_detector.py
│   ├── main.py
│   ├── requirements.txt
│   └── weights/
├── src/
│   ├── components/
│   ├── services/
│   ├── utils/
│   ├── App.jsx
│   ├── App.css
│   └── index.css
├── package.json
├── vite.config.js
└── .gitignore
```

## Run Locally

### 1. Start the backend

From the project root, install the backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Start the FastAPI server:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Backend health check: `http://127.0.0.1:8000/health`

API documentation: `http://127.0.0.1:8000/docs`

### 2. Start the frontend

Open a second terminal in the project root:

```bash
npm install
npm run dev
```

Open the local URL printed by Vite, typically `http://localhost:5173`.

## Important Limitations

TruthLens AI provides probabilistic model assessments, not definitive proof of authenticity. Models can produce false positives and false negatives, and performance may vary with video quality, content type, and manipulation technique.

`LIKELY REAL` does not guarantee that a video is authentic. `LIKELY AI-GENERATED` indicates model-detected signals, not conclusive proof. `INCONCLUSIVE` means the available signals are insufficient or conflict.

This project is intended as a decision-support and demonstration tool, not a substitute for qualified forensic examination.

## Project Repository

Source code: https://github.com/kowsalya-21/truthlens-ai
