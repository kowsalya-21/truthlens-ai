import React, { useState } from 'react';
import { X, Copy, Check, Server, Terminal, FileCode, CheckCircle2 } from 'lucide-react';

const FASTAPI_SAMPLE_CODE = `# backend/main.py
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.detector import analyze_video_file, MODEL_NAME

app = FastAPI(title="TruthLens AI Forensic API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "TruthLens AI Forensic Engine",
        "model_loaded": MODEL_NAME
    }

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Receives video upload, executes real pretrained Vision Transformer inference
    across sampled face frames, and returns genuine forensic metrics.
    Categories: 'LIKELY REAL', 'LIKELY AI-GENERATED / MANIPULATED', 'INCONCLUSIVE'
    """
    import tempfile, shutil, os
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Runs real ViT inference across video frames
        result = analyze_video_file(temp_path, max_samples=12)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
`;

export default function FastApiGuideModal({ isOpen, onClose }) {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(FASTAPI_SAMPLE_CODE);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 100,
        background: 'rgba(3, 6, 12, 0.85)',
        backdropFilter: 'blur(10px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
      }}
      className="fade-in"
      onClick={onClose}
    >
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '780px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          background: '#0B1120',
          border: '1px solid var(--border-medium)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <Server size={22} color="var(--cyan-400)" />
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Python FastAPI Backend Contract</h3>
          </div>
          <button
            type="button"
            className="btn-secondary"
            onClick={onClose}
            style={{ padding: '0.4rem', borderRadius: '50%' }}
            aria-label="Close Modal"
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ overflowY: 'auto', paddingRight: '0.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            TruthLens AI frontend communicates directly with your FastAPI server. Copy this starter script to spin up
            a compatible backend immediately with <code>uvicorn</code>.
          </p>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--cyan-300)' }}>
              <FileCode size={16} />
              <span>backend/main.py</span>
            </div>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleCopy}
              style={{ padding: '0.35rem 0.85rem', fontSize: '0.8rem' }}
            >
              {copied ? <Check size={14} color="#10B981" /> : <Copy size={14} />}
              <span>{copied ? 'Copied Code' : 'Copy Python Code'}</span>
            </button>
          </div>

          <pre
            className="raw-json-box"
            style={{ maxHeight: '340px', fontSize: '0.8rem', color: '#BAE6FD' }}
          >
            {FASTAPI_SAMPLE_CODE}
          </pre>

          <div
            style={{
              background: 'rgba(15, 23, 42, 0.7)',
              padding: '1rem',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.85rem',
            }}
          >
            <strong style={{ color: 'var(--cyan-400)', display: 'block', marginBottom: '0.35rem' }}>
              Terminal Command to Run Backend:
            </strong>
            <code style={{ background: '#03060C', padding: '0.35rem 0.65rem', borderRadius: 4, display: 'block' }}>
              pip install fastapi uvicorn python-multipart
              <br />
              python -m uvicorn backend.main:app --reload --port 8000
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}
