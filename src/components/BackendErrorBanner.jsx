import React, { useState } from 'react';
import { AlertOctagon, RefreshCw, Terminal, Check, Server, ExternalLink, ArrowRight } from 'lucide-react';
import { DEFAULT_API_URL } from '../services/api';

export default function BackendErrorBanner({ error, onRetry, onOpenApiGuide }) {
  const [customUrl, setCustomUrl] = useState(() => localStorage.getItem('truthlens_api_url') || DEFAULT_API_URL);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [showTerminalGuide, setShowTerminalGuide] = useState(false);

  const handleSaveUrl = (e) => {
    e.preventDefault();
    localStorage.setItem('truthlens_api_url', customUrl.trim());
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2000);
  };

  const isNetworkFailure = error?.isConnectionError || !error?.status;

  return (
    <div className="backend-error-panel fade-in" role="alert">
      <div className="backend-error-header">
        <div className="backend-error-icon">
          <AlertOctagon size={28} />
        </div>
        <div className="backend-error-text" style={{ flex: 1 }}>
          <h3>
            {isNetworkFailure
              ? 'FastAPI Backend Connection Unavailable'
              : `FastAPI Backend Error (HTTP ${error.status})`}
          </h3>
          <p>
            {error?.message ||
              'Could not connect to the Python FastAPI backend service. TruthLens AI strictly communicates with your live model backend and does not display simulated results.'}
          </p>
          <div className="backend-endpoint-tag">
            Attempted Endpoint: <code>{error?.apiUrl || customUrl}</code>
          </div>
        </div>
      </div>

      {/* Troubleshooting Checklist */}
      <div className="troubleshooting-box">
        <div className="troubleshooting-title">
          <Terminal size={15} color="var(--cyan-400)" />
          <span>Troubleshooting & Setup Guide for Developers</span>
        </div>
        <ul className="troubleshooting-steps">
          <li>
            <span>1.</span>
            <div>
              Ensure your Python FastAPI backend is active and listening:
              <br />
              <code>uvicorn backend.main:app --reload --port 8000</code>
            </div>
          </li>
          <li>
            <span>2.</span>
            <div>
              Verify CORS middleware is enabled in FastAPI with <code>allow_origins=["*"]</code>.
            </div>
          </li>
          <li>
            <span>3.</span>
            <div>
              Ensure your endpoint matches <code>POST /api/analyze</code> (or update the URL below).
            </div>
          </li>
        </ul>
      </div>

      {/* Quick Endpoint Configuration Form */}
      <form onSubmit={handleSaveUrl} className="endpoint-editor">
        <input
          type="url"
          className="endpoint-input"
          value={customUrl}
          onChange={(e) => setCustomUrl(e.target.value)}
          placeholder="http://localhost:8000/api/analyze"
          title="FastAPI API Endpoint URL"
        />
        <button type="submit" className="btn-secondary" style={{ padding: '0.65rem 1rem' }}>
          {saveSuccess ? (
            <>
              <Check size={14} color="#10B981" />
              <span>Saved</span>
            </>
          ) : (
            <span>Update URL</span>
          )}
        </button>
        <button
          type="button"
          className="btn-primary"
          onClick={onRetry}
          style={{ padding: '0.65rem 1.4rem' }}
          id="retry-analysis-btn"
        >
          <RefreshCw size={15} />
          <span>Retry Analysis</span>
        </button>
        <button
          type="button"
          className="btn-secondary"
          onClick={onOpenApiGuide}
          style={{ padding: '0.65rem 1rem' }}
        >
          <Server size={14} />
          <span>View FastAPI Starter Code</span>
        </button>
      </form>
    </div>
  );
}
