import React, { useState, useRef } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import VideoUploader from './components/VideoUploader';
import VideoPreview from './components/VideoPreview';
import AnalysisProgress from './components/AnalysisProgress';
import ResultsView from './components/ResultsView';
import BackendErrorBanner from './components/BackendErrorBanner';
import FastApiGuideModal from './components/FastApiGuideModal';
import FeaturesSection from './components/FeaturesSection';
import Footer from './components/Footer';
import { analyzeVideo } from './services/api';
import './App.css';

export default function App() {
  const [videoFile, setVideoFile] = useState(null);
  // 'idle' | 'preview' | 'analyzing' | 'results' | 'error'
  const [analysisState, setAnalysisState] = useState('idle');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [isApiGuideOpen, setIsApiGuideOpen] = useState(false);
  const [detectorMode, setDetectorMode] = useState('standard'); // 'standard' | 'experimental'

  const abortControllerRef = useRef(null);

  // Smooth scroll to analyzer area from Hero CTA
  const handleScrollToAnalyzer = () => {
    const el = document.getElementById('analyzer-section');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // When a video is selected & passed validation
  const handleFileSelected = (file) => {
    setVideoFile(file);
    setAnalysisState('preview');
    setApiError(null);
    setAnalysisResult(null);
  };

  // Clear current video and return to uploader
  const handleRemoveFile = () => {
    setVideoFile(null);
    setAnalysisState('idle');
    setAnalysisResult(null);
    setApiError(null);
  };

  // Trigger analysis call to FastAPI backend
  const handleStartAnalysis = async () => {
    if (!videoFile) return;

    setAnalysisState('analyzing');
    setApiError(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      // Direct call to Python FastAPI backend (routes to standard or experimental)
      const result = await analyzeVideo(videoFile, {
        signal: controller.signal,
        detectorMode
      });

      setAnalysisResult(result);
      setAnalysisState('results');
    } catch (err) {
      if (err.isAborted) {
        setAnalysisState('preview');
        return;
      }
      // Display genuine connection or backend error (NO simulated results)
      setApiError(err);
      setAnalysisState('error');
    } finally {
      abortControllerRef.current = null;
    }
  };

  // Abort analysis request
  const handleCancelAnalysis = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setAnalysisState('preview');
  };

  // Reset to analyze another video
  const handleResetToIdle = () => {
    setVideoFile(null);
    setAnalysisResult(null);
    setApiError(null);
    setAnalysisState('idle');
    handleScrollToAnalyzer();
  };

  return (
    <div className="app-container">
      {/* Top Header with live backend health & configuration */}
      <Header onOpenApiGuide={() => setIsApiGuideOpen(true)} />

      <main className="main-content">
        {/* Hero Section */}
        <Hero onAnalyzeClick={handleScrollToAnalyzer} />

        {/* Central Forensic Workstation */}
        <section className="container forensics-workstation" id="analyzer-section">
          <div className="glass-panel">
            {/* 1. Uploader View */}
            {analysisState === 'idle' && (
              <VideoUploader onFileSelected={handleFileSelected} />
            )}

            {/* 2. Video Preview View */}
            {analysisState === 'preview' && (
              <VideoPreview
                file={videoFile}
                onRemoveFile={handleRemoveFile}
                onStartAnalysis={handleStartAnalysis}
                detectorMode={detectorMode}
                onDetectorModeChange={setDetectorMode}
              />
            )}

            {/* 3. Analysis Progress View */}
            {analysisState === 'analyzing' && (
              <AnalysisProgress onCancel={handleCancelAnalysis} />
            )}

            {/* 4. Results View */}
            {analysisState === 'results' && analysisResult && (
              <ResultsView
                result={analysisResult}
                onReset={handleResetToIdle}
              />
            )}

            {/* 5. Backend Error View */}
            {analysisState === 'error' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <BackendErrorBanner
                  error={apiError}
                  onRetry={handleStartAnalysis}
                  onOpenApiGuide={() => setIsApiGuideOpen(true)}
                />
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={handleRemoveFile}
                  >
                    Select Different Video
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Forensic Capabilities Grid */}
        <FeaturesSection />
      </main>

      {/* Footer */}
      <Footer onOpenApiGuide={() => setIsApiGuideOpen(true)} />

      {/* FastAPI Backend Guide Modal */}
      <FastApiGuideModal
        isOpen={isApiGuideOpen}
        onClose={() => setIsApiGuideOpen(false)}
      />
    </div>
  );
}
