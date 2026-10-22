import React from 'react';
import { Shield, Sparkles, Terminal } from 'lucide-react';

export default function Footer({ onOpenApiGuide }) {
  return (
    <footer className="footer">
      <div className="container footer-inner">
        <div className="footer-brand">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            <Shield size={16} color="var(--cyan-400)" />
            <span>TruthLens AI</span>
          </div>
          <span>Hackathon Digital Forensics Platform • AI Facial Manipulation Detection</span>
        </div>

        <div className="footer-links">
          <button
            type="button"
            onClick={onOpenApiGuide}
            style={{ color: 'var(--text-muted)', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
          >
            <Terminal size={14} />
            <span>FastAPI Setup</span>
          </button>
          <span>•</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Frontend React + Vite
          </span>
        </div>
      </div>
    </footer>
  );
}
