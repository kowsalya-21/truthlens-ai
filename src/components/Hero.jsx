import React from 'react';
import { Sparkles, ArrowDown, ScanFace, Activity, ShieldCheck, Zap } from 'lucide-react';

export default function Hero({ onAnalyzeClick }) {
  return (
    <section className="hero-section">
      <div className="container">
        {/* Top Innovation Pill */}
        <div className="hero-pill">
          <Sparkles size={14} />
          <span>Next-Gen Facial Forensics & Neural Artifact Detection</span>
        </div>

        {/* Main Title */}
        <h1 className="hero-title">
          Unmask Synthetic Reality with <br />
          <span className="hero-title-highlight">TruthLens AI Forensics</span>
        </h1>

        {/* Short Crisp Explanation */}
        <p className="hero-subtitle">
          An advanced digital-forensics platform engineered to detect facial deepfakes, 
          face-swaps, and synthetic facial manipulation in digital video using frame-by-frame 
          Vision Transformer analysis.
        </p>

        {/* Prominent Call to Action */}
        <div style={{ marginBottom: '2.5rem' }}>
          <button
            type="button"
            className="btn-primary"
            onClick={onAnalyzeClick}
            id="hero-analyze-cta"
            aria-label="Analyze Video"
          >
            <ScanFace size={18} />
            <span>Analyze Video Now</span>
            <ArrowDown size={16} />
          </button>
        </div>

        {/* Forensic Telemetry Highlights */}
        <div className="hero-stats-row">
          <div className="hero-stat-badge">
            <ScanFace size={15} />
            <span>OpenCV Facial Extraction</span>
          </div>
          <div className="hero-stat-badge">
            <Activity size={15} />
            <span>Vision Transformer (ViT) Inference</span>
          </div>
          <div className="hero-stat-badge">
            <Zap size={15} />
            <span>Multi-Frame Sampling</span>
          </div>
          <div className="hero-stat-badge">
            <ShieldCheck size={15} />
            <span>FastAPI Neural Pipeline</span>
          </div>
        </div>
      </div>
    </section>
  );
}
