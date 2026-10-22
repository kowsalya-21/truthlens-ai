import React, { useState, useEffect } from 'react';
import { Play, RotateCcw, Cpu, Film, Clock, HardDrive, Monitor, ShieldCheck } from 'lucide-react';
import { formatBytes, formatDuration, extractVideoMetadata } from '../utils/fileValidation';

export default function VideoPreview({
  file,
  onRemoveFile,
  onStartAnalysis,
  detectorMode = 'standard',
  onDetectorModeChange
}) {
  const [videoUrl, setVideoUrl] = useState('');
  const [metadata, setMetadata] = useState({
    duration: 0,
    width: 0,
    height: 0,
    resolution: 'Analyzing...',
    aspectRatio: '--',
  });

  useEffect(() => {
    if (!file) return;

    // Create object URL for preview
    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    // Extract duration & resolution
    extractVideoMetadata(file).then((meta) => {
      setMetadata(meta);
    });

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  return (
    <div className="preview-container fade-in">
      <div className="preview-header">
        <div className="preview-title">
          <Film size={20} color="var(--cyan-400)" />
          <span>Video Payload Inspection & Preview</span>
        </div>
        <button
          type="button"
          className="btn-secondary"
          onClick={onRemoveFile}
          style={{ padding: '0.5rem 0.9rem', fontSize: '0.82rem' }}
          aria-label="Choose different video"
        >
          <RotateCcw size={14} />
          <span>Change Video</span>
        </button>
      </div>

      {/* HTML5 Video Player with Forensic HUD */}
      <div className="video-player-wrapper">
        <div className="video-hud-overlay">
          <div className="hud-badge">
            <span className="hud-radar-dot" />
            <span>PAYLOAD READY // PRE-ANALYSIS</span>
          </div>
          <div className="hud-badge" style={{ borderColor: 'rgba(139, 92, 246, 0.4)', color: 'var(--violet-400)' }}>
            <span>{metadata.resolution}</span>
          </div>
        </div>

        {videoUrl && (
          <video
            className="video-player-element"
            src={videoUrl}
            controls
            playsInline
            preload="metadata"
          />
        )}
      </div>

      {/* Video Details & Technical Metadata Grid */}
      <div className="video-metadata-grid">
        <div className="meta-card">
          <span className="meta-label">
            <Film size={12} style={{ display: 'inline', marginRight: 4 }} />
            File Name
          </span>
          <span className="meta-value" title={file.name}>
            {file.name}
          </span>
        </div>

        <div className="meta-card">
          <span className="meta-label">
            <HardDrive size={12} style={{ display: 'inline', marginRight: 4 }} />
            Payload Size
          </span>
          <span className="meta-value">{formatBytes(file.size)}</span>
        </div>

        <div className="meta-card">
          <span className="meta-label">
            <Clock size={12} style={{ display: 'inline', marginRight: 4 }} />
            Duration
          </span>
          <span className="meta-value">{formatDuration(metadata.duration)}</span>
        </div>

        <div className="meta-card">
          <span className="meta-label">
            <Monitor size={12} style={{ display: 'inline', marginRight: 4 }} />
            Dimensions
          </span>
          <span className="meta-value">
            {metadata.width && metadata.height ? `${metadata.width} × ${metadata.height}` : 'Reading stream...'}
          </span>
        </div>
      </div>

      {/* Feature Flag: Neural Forensics Engine Selection */}
      <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '12px', padding: '1rem', marginTop: '1.2rem', marginBottom: '0.8rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={15} color="var(--cyan-400)" />
            Forensics Detector Engine Selection
          </span>
          <span style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: (detectorMode.startsWith('experimental')) ? 'rgba(245, 158, 11, 0.15)' : 'rgba(6, 182, 212, 0.15)', color: (detectorMode.startsWith('experimental')) ? '#FBBF24' : 'var(--cyan-300)', fontWeight: 600 }}>
            {(detectorMode.startsWith('experimental')) ? '🧪 EXPERIMENTAL MODE (V4)' : '🛡️ PRODUCTION DEFAULT'}
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
          <button
            type="button"
            onClick={() => onDetectorModeChange && onDetectorModeChange('standard')}
            style={{
              padding: '0.75rem 0.9rem',
              borderRadius: '8px',
              border: detectorMode === 'standard' ? '1.5px solid var(--cyan-400)' : '1px solid rgba(255, 255, 255, 0.08)',
              background: detectorMode === 'standard' ? 'rgba(6, 182, 212, 0.12)' : 'rgba(255, 255, 255, 0.02)',
              color: detectorMode === 'standard' ? '#FFFFFF' : 'var(--text-secondary)',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease'
            }}
          >
            <div style={{ fontWeight: 600, fontSize: '0.85rem', color: detectorMode === 'standard' ? 'var(--cyan-300)' : 'inherit' }}>
              Standard: Facial Deepfake ViT
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '3px', lineHeight: 1.3 }}>
              Production model fine-tuned on face-swaps & StyleGAN (dima806)
            </div>
          </button>
          <button
            type="button"
            onClick={() => onDetectorModeChange && onDetectorModeChange('experimental_v4')}
            style={{
              padding: '0.75rem 0.9rem',
              borderRadius: '8px',
              border: (detectorMode.startsWith('experimental')) ? '1.5px solid var(--violet-400)' : '1px solid rgba(255, 255, 255, 0.08)',
              background: (detectorMode.startsWith('experimental')) ? 'rgba(139, 92, 246, 0.12)' : 'rgba(255, 255, 255, 0.02)',
              color: (detectorMode.startsWith('experimental')) ? '#FFFFFF' : 'var(--text-secondary)',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease'
            }}
          >
            <div style={{ fontWeight: 600, fontSize: '0.85rem', color: (detectorMode.startsWith('experimental')) ? 'var(--violet-300)' : 'inherit' }}>
              Experimental: Spatiotemporal VideoMAE (v4)
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '3px', lineHeight: 1.3 }}>
              Spatiotemporal dynamics + spatial dispersion (V4: 0% False-AI, 100% Animals & Animation)
            </div>
          </button>
        </div>
      </div>

      {/* Action CTA Row */}
      <div className="preview-actions">
        <button
          type="button"
          className="btn-secondary"
          onClick={onRemoveFile}
        >
          Cancel
        </button>
        <button
          type="button"
          className="btn-primary"
          onClick={onStartAnalysis}
          id="start-analysis-btn"
        >
          <Cpu size={18} />
          <span>{(detectorMode.startsWith('experimental')) ? 'Analyze with VideoMAE v4 (Experimental)' : 'Analyze Video for Deepfakes'}</span>
        </button>
      </div>
    </div>
  );
}
