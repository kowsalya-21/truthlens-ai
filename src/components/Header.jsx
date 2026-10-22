import React, { useState, useEffect } from 'react';
import { Shield, Radio, Settings, Server, ExternalLink, Check, AlertCircle } from 'lucide-react';
import { checkBackendHealth, DEFAULT_API_URL } from '../services/api';

export default function Header({ onOpenApiGuide }) {
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [showSettings, setShowSettings] = useState(false);
  const [apiUrl, setApiUrl] = useState(() => localStorage.getItem('truthlens_api_url') || DEFAULT_API_URL);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const verifyHealth = async () => {
      setBackendStatus('checking');
      const health = await checkBackendHealth();
      if (isMounted) {
        setBackendStatus(health.isOnline ? 'online' : 'offline');
      }
    };

    verifyHealth();
    // Periodically re-check every 30s
    const interval = setInterval(verifyHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [apiUrl]);

  const handleSaveApiUrl = (e) => {
    e.preventDefault();
    localStorage.setItem('truthlens_api_url', apiUrl.trim());
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2000);
    // Recheck immediately
    setBackendStatus('checking');
    checkBackendHealth().then((h) => setBackendStatus(h.isOnline ? 'online' : 'offline'));
  };

  return (
    <header className="header">
      <div className="container header-inner">
        {/* Brand Logo */}
        <a href="/" className="brand-logo" aria-label="TruthLens AI Home">
          <div className="brand-icon-wrapper">
            <Shield size={22} />
          </div>
          <div className="brand-text">
            <span className="brand-title">TruthLens AI</span>
            <span className="brand-badge">Forensics Lab v1.0</span>
          </div>
        </a>

        {/* Header Right Actions */}
        <div className="header-actions">
          {/* Backend Status Indicator */}
          <button
            type="button"
            className="backend-status-pill"
            onClick={() => setShowSettings(!showSettings)}
            title="Click to view or edit FastAPI backend endpoint"
            aria-label="Backend Connection Status"
          >
            <span
              className={`status-dot ${
                backendStatus === 'online'
                  ? 'online'
                  : backendStatus === 'offline'
                  ? 'offline'
                  : 'checking'
              }`}
            />
            <span>
              {backendStatus === 'online'
                ? 'Backend: Online'
                : backendStatus === 'offline'
                ? 'Backend: Offline'
                : 'Pinging API...'}
            </span>
            <Settings size={14} style={{ opacity: 0.7 }} />
          </button>

          {/* FastAPI Guide & Schema Button */}
          <button
            type="button"
            className="btn-secondary"
            style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}
            onClick={onOpenApiGuide}
          >
            <Server size={14} />
            <span>FastAPI Specs</span>
          </button>
        </div>
      </div>

      {/* Backend Endpoint Settings Dropdown */}
      {showSettings && (
        <div
          style={{
            background: '#0B1120',
            borderBottom: '1px solid var(--border-medium)',
            padding: '1rem 0',
            boxShadow: 'var(--shadow-card)',
          }}
          className="fade-in"
        >
          <div className="container">
            <form onSubmit={handleSaveApiUrl} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--cyan-300)' }}>
                  FastAPI Backend Endpoint Configuration
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Default: <code>http://localhost:8000/api/analyze</code>
                </span>
              </div>
              <div className="endpoint-editor">
                <input
                  type="url"
                  className="endpoint-input"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  placeholder="http://localhost:8000/api/analyze"
                  required
                />
                <button type="submit" className="btn-secondary" style={{ padding: '0.65rem 1rem' }}>
                  {saveSuccess ? (
                    <>
                      <Check size={14} color="#10B981" />
                      <span>Saved</span>
                    </>
                  ) : (
                    <span>Save URL</span>
                  )}
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => setShowSettings(false)}
                  style={{ padding: '0.65rem 1rem' }}
                >
                  Close
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </header>
  );
}
