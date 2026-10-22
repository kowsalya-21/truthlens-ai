import React, { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  ShieldAlert,
  ShieldCheck,
  FileText,
  RotateCcw,
  Download,
  Copy,
  Check,
  Activity,
  Layers,
  Code2,
  Clock,
  HardDrive,
  Film,
  Info,
  Sparkles,
  Cpu,
  BarChart3,
  AlertCircle
} from 'lucide-react';
import { formatBytes, formatDuration } from '../utils/fileValidation';

export default function ResultsView({ result, onReset }) {
  const [activeTab, setActiveTab] = useState('evidence'); // 'evidence' | 'frames' | 'metrics' | 'json'
  const [copiedJson, setCopiedJson] = useState(false);

  const {
    prediction = 'INCONCLUSIVE',
    modelScore = null,
    analysisStatus = 'Completed',
    explanation = '',
    videoDetails = {},
    evidence = [],
    metrics = {},
    frameAnalysis = [],
    rawResponse = {},
    elapsedMs = 0,
    timestamp = new Date().toISOString(),
    modelInfo = {}
  } = result;

  const arbitration = result.arbitration || rawResponse?.arbitration || null;
  const detectors = result.detectors || rawResponse?.detectors || null;

  // Canonical classification categories
  const isManipulated = prediction === 'LIKELY AI-GENERATED / MANIPULATED';
  const isReal = prediction === 'LIKELY REAL';
  const isInconclusive = prediction === 'INCONCLUSIVE' || (!isManipulated && !isReal);

  // Actual detector mode returned by the API
  const apiMode = (modelInfo?.mode || metrics?.mode || rawResponse?.model_info?.mode || '').toLowerCase();
  const isExperimental = apiMode === 'experimental';
  const modelVersion = (modelInfo?.version || metrics?.version || 'v1').toLowerCase();
  const isCalibrated = metrics?.is_calibrated ?? false;

  const bannerClass = isManipulated
    ? 'likely-manipulated'
    : isReal
    ? 'authentic'
    : 'inconclusive';

  const categoryTag = isExperimental
    ? (isManipulated
        ? 'ALERT // SPATIOTEMPORAL SYNTHESIS DETECTED'
        : isReal
        ? 'OBSERVED // COHERENT TEMPORAL DYNAMICS'
        : 'NOTICE // INCONCLUSIVE INFERENCE')
    : (isManipulated
        ? 'ALERT // FACIAL MANIPULATION DETECTED'
        : isReal
        ? 'OBSERVED // NO FACIAL MANIPULATION DETECTED'
        : 'NOTICE // INCONCLUSIVE INFERENCE');

  const defaultExplanation = isManipulated
    ? (isExperimental
        ? 'The experimental VideoMAE model detected spatiotemporal inconsistencies across video frames associated with synthetic generation.'
        : 'The neural Vision Transformer detected manipulation patterns across sampled face frames associated with synthetic deepfake generation.')
    : isReal
    ? (isExperimental
        ? 'The experimental VideoMAE model detected natural camera motion and coherent temporal physics.'
        : 'The neural Vision Transformer detected no strong facial manipulation signals across sampled face frames. Note: This model tests facial regions for face-swaps and does not evaluate full-frame AI video diffusion.')
    : 'The analysis yielded inconclusive results. The model could not produce a reliable classification due to face detection constraints, high inter-frame variance, or ambiguous scores.';

  const displayExplanation = explanation || defaultExplanation;

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(rawResponse || result, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  const handleExportReport = () => {
    const reportData = {
      title: 'TruthLens AI - Deepfake Video Inspection Report',
      modelUsed: modelInfo.name || (isExperimental ? 'MCG-NJU/videomae-base' : 'dima806/deepfake_vs_real_image_detection'),
      modelArchitecture: modelInfo.architecture || (isExperimental ? '3D VideoMAE ViT-B' : 'ViTForImageClassification'),
      detectionScope: isExperimental ? 'Spatiotemporal Authenticity (Experimental Full-Scene)' : 'Facial Deepfake & Face-Swap Manipulation (GAN-trained)',
      timestamp: new Date().toISOString(),
      prediction,
      rawModelScore: modelScore !== null ? `${(modelScore * 100).toFixed(2)}%` : 'N/A',
      scoreDescription: isExperimental ? 'VideoMAE Spatiotemporal Model Score (Uncalibrated)' : 'Vision Transformer Class 1 (Fake) Softmax Probability',
      analysisStatus,
      backendLatencyMs: elapsedMs,
      explanation: displayExplanation,
      videoDetails,
      computedEvidence: evidence,
      perFrameAnalysis: frameAnalysis,
      scopeLimitations: isExperimental
        ? 'Experimental full-scene model evaluated on small smoke-test dataset. Requires specialist cross-corroboration.'
        : 'Trained on 140k facial images (StyleGAN & early deepfakes). Does not detect full-scene text-to-video diffusion generation (e.g. OpenAI Sora, Kling AI, Runway Gen-2/3, Luma Dream Machine).',
      disclaimer: 'AI detection models are probabilistic classifiers. Results must be corroborated with full context and human review.'
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `truthlens-report-${videoDetails.filename || 'video'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="results-container fade-in">
      {/* 1. Main Prediction Banner */}
      <div className={`prediction-banner ${bannerClass}`}>
        <div className="banner-left">
          <div className="verdict-icon-container">
            {isManipulated && <ShieldAlert size={32} />}
            {isReal && <ShieldCheck size={32} />}
            {isInconclusive && <HelpCircle size={32} />}
          </div>
          <div className="verdict-info">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', flexWrap: 'wrap' }}>
              <span className="verdict-tag">{categoryTag}</span>
              <span
                style={{
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  letterSpacing: '0.04em',
                  textTransform: 'uppercase',
                  padding: '0.2rem 0.55rem',
                  borderRadius: '4px',
                  background: isExperimental ? 'rgba(245, 158, 11, 0.15)' : 'rgba(6, 182, 212, 0.15)',
                  color: isExperimental ? '#FBBF24' : 'var(--cyan-300)',
                  border: `1px solid ${isExperimental ? 'rgba(245, 158, 11, 0.35)' : 'rgba(6, 182, 212, 0.35)'}`
                }}
              >
                {isExperimental ? `🧪 EXPERIMENTAL MODE (VideoMAE ${modelVersion.toUpperCase()})` : '🛡️ STANDARD MODE (Facial ViT)'}
              </span>
            </div>
            <h2 className="verdict-heading">{prediction}</h2>
            <p className="verdict-description">{displayExplanation}</p>
          </div>
        </div>

        {/* Real Model Score Meter (Honestly labeled: Facial Manipulation ViT Class 1 or Spatiotemporal Model Score) */}
        <div className="score-meter-card">
          <span className="score-meter-label">
            {isExperimental ? 'Spatiotemporal Model Score' : 'Facial Manipulation Score'}
          </span>
          <div className="score-meter-value">
            {modelScore !== null ? `${(modelScore * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <div className="score-bar-subtrack">
            <div
              className="score-bar-subfill"
              style={{
                width: modelScore !== null ? `${Math.min(100, Math.max(0, modelScore * 100))}%` : '0%',
                background: isManipulated
                  ? 'var(--status-manipulated)'
                  : isReal
                  ? 'var(--status-authentic)'
                  : 'var(--status-inconclusive)'
              }}
            />
          </div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem', textAlign: 'center' }}>
            {modelScore !== null
              ? (isExperimental
                  ? (isCalibrated ? 'VideoMAE Model Score (Calibrated Softmax)' : 'VideoMAE Model Score (Uncalibrated Softmax)')
                  : 'ViT Class 1 (Fake) Probability')
              : 'Unscored / Inconclusive'}
          </span>
        </div>
      </div>

      {/* Model Scope & Capability Callout */}
      <div className={`model-scope-banner ${isInconclusive ? 'warning' : ''}`} role="region" aria-label="Model Scope Notice">
        <AlertCircle size={20} color={isInconclusive ? 'var(--status-inconclusive)' : 'var(--cyan-400)'} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <div className="scope-title">
            <span>
              {isExperimental 
                ? 'Model Scope: Spatiotemporal Video Authenticity (Experimental)' 
                : 'Model Scope: Facial Deepfake & Face-Swap Detection'}
            </span>
            <span className="scope-tag">{modelInfo.name || (isExperimental ? 'MCG-NJU/videomae-base' : 'dima806/deepfake_vs_real_image_detection')}</span>
          </div>
          <p className="scope-text">
            {isExperimental ? (
              <>
                TruthLens evaluates full-scene spatiotemporal dynamics across video frames using a VideoMAE 3D Vision Transformer. 
                <strong> This experimental model operates across arbitrary scenes (people, landscapes, objects) without requiring faces</strong>, 
                yielding an uncalibrated spatiotemporal model score. Results must be corroborated with specialist detectors and human review.
                {isReal && ' A "Likely Real" classification reflects coherent full-scene motion dynamics and an absence of facial manipulation signatures.'}
                {isInconclusive && ' When whole-scene dynamics conflict with localized facial analysis, or when video length exceeds the experimental validation window, the verdict is set to Inconclusive.'}
              </>
            ) : (
              <>
                TruthLens evaluates isolated human facial regions using a Vision Transformer fine-tuned on GAN face-swap datasets. 
                <strong> It is not designed to detect fully AI-generated video scenes</strong> (e.g., text-to-video diffusion models like OpenAI Sora, Kling AI, Runway Gen-2/3, or Luma Dream Machine). 
                {isReal && ' A "Likely Real" classification reflects an absence of facial manipulation signatures within this model’s trained scope, not conclusive proof that the video was not synthetically generated.'}
                {isInconclusive && ' When videos exhibit extreme inter-frame score divergence or lack frontal facial crops, the model honestly flags the result as Inconclusive to avoid misleading verdicts.'}
              </>
            )}
          </p>
        </div>
      </div>

      {/* Multi-Model Forensic Consensus & Arbitration */}
      {detectors && (detectors.facial_specialist || detectors.spatiotemporal_videomae) && (
        <div className="arbitration-summary-panel" style={{
          background: 'rgba(15, 23, 42, 0.65)',
          border: arbitration?.agreement === false ? '1px solid rgba(245, 158, 11, 0.4)' : '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg)',
          padding: '1rem 1.25rem',
          marginBottom: '1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={18} color="var(--cyan-400)" />
              <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                Multi-Model Forensic Consensus
              </span>
            </div>
            {arbitration?.agreement !== undefined && (
              <span style={{
                fontSize: '0.72rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
                padding: '0.2rem 0.6rem',
                borderRadius: '4px',
                background: arbitration.agreement === true 
                  ? 'rgba(16, 185, 129, 0.15)' 
                  : arbitration.agreement === false 
                  ? 'rgba(245, 158, 11, 0.15)' 
                  : 'rgba(148, 163, 184, 0.15)',
                color: arbitration.agreement === true 
                  ? '#34D399' 
                  : arbitration.agreement === false 
                  ? '#FBBF24' 
                  : 'var(--text-muted)',
                border: `1px solid ${arbitration.agreement === true 
                  ? 'rgba(16, 185, 129, 0.35)' 
                  : arbitration.agreement === false 
                  ? 'rgba(245, 158, 11, 0.35)' 
                  : 'rgba(148, 163, 184, 0.25)'}`
              }}>
                {arbitration.agreement === true 
                  ? '✓ Unanimous Agreement' 
                  : arbitration.agreement === false 
                  ? '⚠ Disagreement (Arbitrated)' 
                  : 'Single-Model Assessment'}
              </span>
            )}
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '0.75rem'
          }}>
            {/* Facial Specialist Card */}
            {detectors.facial_specialist && (
              <div style={{
                background: 'rgba(0, 0, 0, 0.3)',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                border: '1px solid rgba(255,255,255,0.06)'
              }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                  FACIAL SPECIALIST (ViT dima806)
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{
                    fontWeight: 600,
                    fontSize: '0.85rem',
                    color: detectors.facial_specialist.verdict === 'AI-GENERATED / MANIPULATED'
                      ? '#F87171'
                      : detectors.facial_specialist.verdict === 'REAL'
                      ? '#34D399'
                      : 'var(--text-secondary)'
                  }}>
                    {detectors.facial_specialist.verdict}
                  </span>
                  <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--cyan-300)' }}>
                    {detectors.facial_specialist.score !== null ? `${(detectors.facial_specialist.score * 100).toFixed(1)}%` : '--'}
                  </span>
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                  {detectors.facial_specialist.faces_detected !== undefined
                    ? `${detectors.facial_specialist.faces_detected} faces (${detectors.facial_specialist.individuals_tracked ?? 1} tracked)`
                    : 'Facial frames analyzed'}
                </div>
              </div>
            )}

            {/* VideoMAE Spatiotemporal Card */}
            {detectors.spatiotemporal_videomae && (
              <div style={{
                background: 'rgba(0, 0, 0, 0.3)',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                border: '1px solid rgba(255,255,255,0.06)'
              }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                  SPATIOTEMPORAL (VideoMAE {detectors.spatiotemporal_videomae.version?.toUpperCase() || 'V4'})
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{
                    fontWeight: 600,
                    fontSize: '0.85rem',
                    color: detectors.spatiotemporal_videomae.verdict === 'AI-GENERATED / MANIPULATED'
                      ? '#F87171'
                      : detectors.spatiotemporal_videomae.verdict === 'REAL'
                      ? '#34D399'
                      : 'var(--text-secondary)'
                  }}>
                    {detectors.spatiotemporal_videomae.verdict}
                  </span>
                  <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--cyan-300)' }}>
                    {detectors.spatiotemporal_videomae.score !== null ? `${(detectors.spatiotemporal_videomae.score * 100).toFixed(1)}%` : '--'}
                  </span>
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                  {detectors.spatiotemporal_videomae.verdict === 'NOT_RUN'
                    ? 'Evaluated in Experimental Mode'
                    : '16-frame spatiotemporal dynamics'}
                </div>
              </div>
            )}
          </div>

          {arbitration?.reason && (
            <div style={{
              fontSize: '0.76rem',
              color: arbitration.agreement === false ? '#FBBF24' : 'var(--text-secondary)',
              background: 'rgba(0,0,0,0.2)',
              padding: '0.5rem 0.75rem',
              borderRadius: '4px',
              borderLeft: arbitration.agreement === false ? '3px solid #F59E0B' : '3px solid var(--cyan-400)'
            }}>
              <strong>Arbitration reasoning:</strong> {arbitration.reason}
            </div>
          )}
        </div>
      )}

      {/* 2. Video Details & Telemetry Bar */}
      <div className="forensics-meta-bar">
        <div className="meta-pill-group">
          <div className="meta-pill-item">
            <Film size={14} color="var(--cyan-400)" />
            <span>Target: <strong>{videoDetails.filename || 'Analyzed Video'}</strong></span>
          </div>
          {videoDetails.fileSize && (
            <div className="meta-pill-item">
              <HardDrive size={14} color="var(--violet-400)" />
              <span>Size: <strong>{formatBytes(videoDetails.fileSize)}</strong></span>
            </div>
          )}
          {videoDetails.duration && (
            <div className="meta-pill-item">
              <Clock size={14} color="var(--cyan-400)" />
              <span>Duration: <strong>{formatDuration(videoDetails.duration)}</strong></span>
            </div>
          )}
          {videoDetails.resolution && (
            <div className="meta-pill-item">
              <span>Resolution: <strong>{videoDetails.resolution}</strong></span>
            </div>
          )}
        </div>

        <div className="meta-pill-group">
          <div className="meta-pill-item">
            <span>Status: <strong>{analysisStatus}</strong></span>
          </div>
          {elapsedMs > 0 && (
            <div className="meta-pill-item">
              <span>Backend Time: <strong>{elapsedMs}ms</strong></span>
            </div>
          )}
        </div>
      </div>

      {/* 3. Forensic Tabs Navigation */}
      <div className="results-tabs-nav">
        <button
          type="button"
          className={`tab-btn ${activeTab === 'evidence' ? 'active' : ''}`}
          onClick={() => setActiveTab('evidence')}
        >
          <Layers size={15} />
          <span>Genuinely Computed Evidence ({evidence.length})</span>
        </button>

        {frameAnalysis && frameAnalysis.length > 0 && (
          <button
            type="button"
            className={`tab-btn ${activeTab === 'frames' ? 'active' : ''}`}
            onClick={() => setActiveTab('frames')}
          >
            <BarChart3 size={15} />
            <span>Per-Frame Inference ({frameAnalysis.length})</span>
          </button>
        )}

        <button
          type="button"
          className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          <Activity size={15} />
          <span>Model Scope & Telemetry</span>
        </button>

        <button
          type="button"
          className={`tab-btn ${activeTab === 'json' ? 'active' : ''}`}
          onClick={() => setActiveTab('json')}
        >
          <Code2 size={15} />
          <span>FastAPI Raw JSON</span>
        </button>
      </div>

      {/* Tab 1: Genuine Evidence Section */}
      {activeTab === 'evidence' && (
        <div className="evidence-section fade-in">
          {evidence.length === 0 ? (
            <div className="evidence-card">
              <p className="evidence-description" style={{ color: 'var(--text-muted)' }}>
                No facial evidence could be extracted from this payload.
              </p>
            </div>
          ) : (
            <div className="evidence-grid">
              {evidence.map((ev, index) => (
                <div key={index} className="evidence-card">
                  <div className="evidence-card-header">
                    <div className="evidence-card-title">
                      <Sparkles size={14} color="var(--cyan-400)" />
                      <span>{ev.category}</span>
                    </div>
                    <span
                      className={`evidence-flag ${
                        ev.status === 'anomaly'
                          ? 'anomaly'
                          : ev.status === 'normal'
                          ? 'normal'
                          : 'uncertain'
                      }`}
                    >
                      {ev.tag}
                    </span>
                  </div>
                  <p className="evidence-description">{ev.description}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Per-Frame Breakdown */}
      {activeTab === 'frames' && frameAnalysis.length > 0 && (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            Uniform frame sampling and per-frame Vision Transformer class probabilities (multi-orientation scan):
          </div>
          <div style={{ overflowX: 'auto', background: '#03060C', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  <th style={{ padding: '0.65rem 1rem' }}>Sample #</th>
                  <th style={{ padding: '0.65rem 1rem' }}>Time Offset</th>
                  <th style={{ padding: '0.65rem 1rem' }}>{isExperimental ? 'Analysis Scope' : 'Face Region'}</th>
                  <th style={{ padding: '0.65rem 1rem' }}>{isExperimental ? 'Fake Model Score' : 'Fake Probability'}</th>
                  <th style={{ padding: '0.65rem 1rem' }}>{isExperimental ? 'Real Model Score' : 'Real Probability'}</th>
                </tr>
              </thead>
              <tbody>
                {frameAnalysis.map((frame, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', fontFamily: 'var(--font-mono)' }}>
                    <td style={{ padding: '0.6rem 1rem', color: 'var(--text-primary)' }}>Frame {frame.frame_index}</td>
                    <td style={{ padding: '0.6rem 1rem', color: 'var(--cyan-300)' }}>{frame.timestamp_sec}s</td>
                    <td style={{ padding: '0.6rem 1rem' }}>
                      {frame.face_detected ? (
                        <span style={{ color: '#34D399' }}>
                          {frame.face_count && frame.face_count > 1
                            ? `Detected (${frame.face_count} faces - ${frame.bbox?.method || 'multi'})`
                            : `Detected (${frame.bbox?.method || 'detected'})`}
                        </span>
                      ) : (
                        <span style={{ color: isExperimental ? 'var(--cyan-400)' : '#F87171' }}>
                          {isExperimental ? 'Full-Scene Tubelet (VideoMAE)' : 'No Face (Skipped)'}
                        </span>
                      )}
                    </td>
                    <td style={{ padding: '0.6rem 1rem', color: frame.fake_score >= 0.5 ? '#F87171' : '#34D399', fontWeight: 600 }}>
                      {frame.fake_score !== null ? `${(frame.fake_score * 100).toFixed(1)}%` : '--'}
                    </td>
                    <td style={{ padding: '0.6rem 1rem', color: 'var(--text-secondary)' }}>
                      {frame.real_score !== null ? `${(frame.real_score * 100).toFixed(1)}%` : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 3: Model Telemetry & Scope */}
      {activeTab === 'metrics' && (
        <div className="evidence-section fade-in">
          <div className="video-metadata-grid">
            <div className="meta-card">
              <span className="meta-label">Model Checkpoint</span>
              <span className="meta-value" title={modelInfo.name}>{modelInfo.name || (isExperimental ? 'MCG-NJU/videomae-base' : 'dima806/deepfake_vs_real_image_detection')}</span>
            </div>
            <div className="meta-card">
              <span className="meta-label">Architecture</span>
              <span className="meta-value">{modelInfo.architecture || (isExperimental ? '3D VideoMAE ViT-B (Frozen) + Authenticity Head' : 'ViTForImageClassification')}</span>
            </div>
            <div className="meta-card">
              <span className="meta-label">Engine Mode</span>
              <span className="meta-value" style={{ color: isExperimental ? '#FBBF24' : 'var(--cyan-300)' }}>
                {isExperimental ? 'Experimental (Spatiotemporal Research)' : 'Standard (Production Facial ViT)'}
              </span>
            </div>
            <div className="meta-card">
              <span className="meta-label">Score Output Type</span>
              <span className="meta-value">
                {isExperimental ? 'Uncalibrated Model Score (Softmax)' : 'ViT Class 1 Softmax Probability'}
              </span>
            </div>
            <div className="meta-card">
              <span className="meta-label">Trained Domain</span>
              <span className="meta-value">
                {isExperimental ? 'General Video Spatiotemporal Dynamics' : 'Facial Manipulation (140k dataset)'}
              </span>
            </div>
            <div className="meta-card">
              <span className="meta-label">Full-Scene Video AI</span>
              <span className="meta-value" style={{ color: isExperimental ? '#34D399' : '#FBBF24' }}>
                {isExperimental ? 'Evaluated (16-frame spatiotemporal tubelets)' : 'Not Supported (Face-only ViT)'}
              </span>
            </div>
            <div className="meta-card">
              <span className="meta-label">Class Labels</span>
              <span className="meta-value">Class 0: Real, Class 1: Fake</span>
            </div>
            {!isExperimental && (
              <>
                <div className="meta-card">
                  <span className="meta-label">Frames with Faces</span>
                  <span className="meta-value">
                    {(videoDetails.framesWithFaces ?? videoDetails.frames_with_faces ?? videoDetails.facesDetected ?? '0')} of {(videoDetails.framesAnalyzed ?? videoDetails.frames_analyzed ?? videoDetails.framesSampled ?? '0')}
                  </span>
                </div>
                <div className="meta-card">
                  <span className="meta-label">Total Faces Analyzed</span>
                  <span className="meta-value">{videoDetails.facesAnalyzed ?? videoDetails.faces_analyzed ?? videoDetails.facesDetected ?? '0'} faces</span>
                </div>
              </>
            )}
            {metrics.mean_fake_score !== undefined && (
              <div className="meta-card">
                <span className="meta-label">
                  {isExperimental ? 'Mean Model Fake Score' : 'Mean Fake Probability'}
                </span>
                <span className="meta-value">{(metrics.mean_fake_score * 100).toFixed(1)}%</span>
              </div>
            )}
            {metrics.max_fake_score !== undefined && (
              <div className="meta-card">
                <span className="meta-label">
                  {isExperimental ? 'Peak Model Fake Score' : 'Peak Fake Probability'}
                </span>
                <span className="meta-value">{(metrics.max_fake_score * 100).toFixed(1)}%</span>
              </div>
            )}
            {metrics.score_std_dev !== undefined && (
              <div className="meta-card">
                <span className="meta-label">Score Std Deviation</span>
                <span className="meta-value">{(metrics.score_std_dev * 100).toFixed(2)}%</span>
              </div>
            )}
            <div className="meta-card">
              <span className="meta-label">Execution Latency</span>
              <span className="meta-value">{elapsedMs} ms</span>
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Raw FastAPI JSON */}
      {activeTab === 'json' && (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Exact unaltered response returned by the Python FastAPI backend
            </span>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleCopyJson}
              style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
            >
              {copiedJson ? <Check size={12} color="#10B981" /> : <Copy size={12} />}
              <span>{copiedJson ? 'Copied' : 'Copy JSON'}</span>
            </button>
          </div>
          <pre className="raw-json-box">
            {JSON.stringify(rawResponse && Object.keys(rawResponse).length > 0 ? rawResponse : result, null, 2)}
          </pre>
        </div>
      )}

      {/* 4. AI Disclaimer */}
      <div className="ai-disclaimer-card" role="note">
        <Info size={22} />
        <div>
          <h4 className="disclaimer-title">Forensic Limitations & Scope Notice</h4>
          <p className="disclaimer-text">
            TruthLens AI uses a facial deepfake classifier. A classification of <strong>LIKELY REAL</strong> indicates that 
            no strong facial manipulation signatures were found on isolated faces; it does NOT certify that an entire video 
            was not created by modern text-to-video generative models (such as Sora, Kling, or Runway). Results are probabilistic 
            and should be evaluated alongside context and human verification.
          </p>
        </div>
      </div>

      {/* 5. Results Action Bar */}
      <div className="results-actions-bar">
        <button
          type="button"
          className="btn-secondary"
          onClick={handleExportReport}
          id="export-report-btn"
        >
          <Download size={16} />
          <span>Export Inspection Report (.json)</span>
        </button>

        <button
          type="button"
          className="btn-primary"
          onClick={onReset}
          id="analyze-another-btn"
        >
          <RotateCcw size={16} />
          <span>Analyze Another Video</span>
        </button>
      </div>
    </div>
  );
}
