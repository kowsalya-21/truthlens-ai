import React, { useState, useEffect } from 'react';
import { Radar, Scan, Cpu, CheckCircle2, Loader2, XCircle } from 'lucide-react';

const ANALYSIS_STAGES = [
  { id: 1, label: 'Streaming video payload to Python FastAPI backend...' },
  { id: 2, label: 'Extracting keyframes & isolating facial bounding boxes...' },
  { id: 3, label: 'Computing discrete frequency transform & artifact residuals...' },
  { id: 4, label: 'Evaluating temporal coherence & neural synthesis probabilities...' }
];

export default function AnalysisProgress({ onCancel }) {
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [simulatedProgress, setSimulatedProgress] = useState(15);

  useEffect(() => {
    // Advance progress indicator naturally while waiting for real backend response
    const stageTimer = setInterval(() => {
      setCurrentStageIndex((prev) => {
        if (prev < ANALYSIS_STAGES.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 2800);

    const progressTimer = setInterval(() => {
      setSimulatedProgress((prev) => {
        if (prev < 92) {
          return prev + Math.floor(Math.random() * 8) + 3;
        }
        return prev;
      });
    }, 700);

    return () => {
      clearInterval(stageTimer);
      clearInterval(progressTimer);
    };
  }, []);

  return (
    <div className="analysis-progress-panel fade-in">
      {/* Radar Scanner Visualizer */}
      <div className="radar-scanner-wrapper">
        <div className="radar-outer-circle" />
        <div className="radar-inner-circle" />
        <div className="radar-sweep" />
        <div className="radar-center-icon">
          <Scan size={36} />
        </div>
      </div>

      <h3 className="progress-headline">Executing Deepfake Forensic Pipeline</h3>
      <p className="progress-subtext">
        Communicating with FastAPI inference engine. Running biometric artifact analysis on facial frames...
      </p>

      {/* Progress Bar */}
      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.min(simulatedProgress, 95)}%` }}
        />
      </div>

      {/* Pipeline Stages */}
      <div className="progress-stages-list">
        {ANALYSIS_STAGES.map((stage, idx) => {
          const isDone = idx < currentStageIndex;
          const isActive = idx === currentStageIndex;

          return (
            <div
              key={stage.id}
              className={`stage-item ${isDone ? 'completed' : ''} ${isActive ? 'active' : ''}`}
            >
              {isDone ? (
                <CheckCircle2 size={16} color="var(--cyan-400)" />
              ) : isActive ? (
                <Loader2 size={16} className="spin-animation" style={{ animation: 'rotateRadar 1s linear infinite' }} />
              ) : (
                <div style={{ width: 16, height: 16, borderRadius: '50%', border: '1px solid var(--border-subtle)' }} />
              )}
              <span>{stage.label}</span>
            </div>
          );
        })}
      </div>

      {/* Cancel Request Button */}
      {onCancel && (
        <button
          type="button"
          className="btn-cancel"
          onClick={onCancel}
          aria-label="Cancel Video Analysis"
        >
          Cancel Analysis
        </button>
      )}
    </div>
  );
}
