/**
 * TruthLens AI - API Service
 * 
 * Handles communication with the Python FastAPI backend.
 * Streams video payload, manages network exceptions,
 * and passes through genuine model inference output without simulation.
 */

export const DEFAULT_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/analyze';
export const DEFAULT_HEALTH_URL = import.meta.env.VITE_HEALTH_URL || 'http://localhost:8000/health';

/**
 * Standardizes prediction label to one of the 3 required categories:
 * 1. "LIKELY REAL"
 * 2. "LIKELY AI-GENERATED / MANIPULATED"
 * 3. "INCONCLUSIVE"
 */
export function normalizeCategory(rawPrediction) {
  if (!rawPrediction) return 'INCONCLUSIVE';

  const s = String(rawPrediction).trim().toUpperCase();

  if (s.includes('MANIPULATED') || s.includes('AI-GENERATED') || s.includes('DEEPFAKE') || s.includes('SYNTHETIC') || s.includes('FAKE')) {
    return 'LIKELY AI-GENERATED / MANIPULATED';
  }

  if (s.includes('LIKELY REAL') || s.includes('REAL') || s.includes('AUTHENTIC') || s.includes('NO STRONG')) {
    return 'LIKELY REAL';
  }

  return 'INCONCLUSIVE';
}

/**
 * Sends a video file to the Python FastAPI backend for real model inference.
 * 
 * @param {File} file - The video file object
 * @param {object} options - Options including apiUrl, signal
 * @returns {Promise<object>} Genuine forensic response from FastAPI backend
 */
export async function analyzeVideo(file, options = {}) {
  const {
    apiUrl = (localStorage.getItem('truthlens_api_url') || DEFAULT_API_URL),
    signal,
    detectorMode = 'standard',
    useExperimental = false,
  } = options;

  if (!file) {
    throw new Error('No video file provided for analysis.');
  }

  // Route to experimental endpoint if feature flag is active
  let targetUrl = apiUrl;
  if (useExperimental || detectorMode.startsWith('experimental')) {
    let version = 'v4';
    if (detectorMode === 'experimental_v3') version = 'v3';
    else if (detectorMode === 'experimental_v2') version = 'v2';
    else if (detectorMode === 'experimental_v1') version = 'v1';
    else if (detectorMode === 'experimental_v4' || detectorMode === 'experimental' || useExperimental) version = 'v4';

    const baseUrl = targetUrl.includes('/api/analyze/experimental')
      ? targetUrl.split('?')[0]
      : targetUrl.replace('/api/analyze', '/api/analyze/experimental').split('?')[0];
    targetUrl = `${baseUrl}?version=${version}`;
  }

  const formData = new FormData();
  formData.append('file', file);

  const clientStartTime = performance.now();

  try {
    const response = await fetch(targetUrl, {
      method: 'POST',
      body: formData,
      signal,
    });

    const clientElapsedMs = Math.round(performance.now() - clientStartTime);

    if (!response.ok) {
      let errorDetail = `Backend returned HTTP ${response.status} (${response.statusText})`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string'
            ? errorJson.detail
            : JSON.stringify(errorJson.detail);
        } else if (errorJson.message) {
          errorDetail = errorJson.message;
        }
      } catch {
        const errorText = await response.text().catch(() => '');
        if (errorText) errorDetail = errorText.slice(0, 300);
      }

      const error = new Error(errorDetail);
      error.status = response.status;
      error.isHttpError = true;
      error.apiUrl = apiUrl;
      throw error;
    }

    const json = await response.json();

    const normalizedPred = normalizeCategory(json.prediction);

    return {
      status: json.status || 'success',
      prediction: normalizedPred,
      rawPrediction: json.prediction,
      modelScore: json.model_score ?? json.confidence_score ?? null,
      analysisStatus: json.analysis_status || 'Completed',
      explanation: json.explanation || '',
      videoDetails: {
        filename: json.video_details?.filename || file.name,
        fileSize: json.video_details?.file_size || file.size,
        duration: json.video_details?.duration || null,
        resolution: json.video_details?.resolution || null,
        fps: json.video_details?.fps || null,
        totalFrames: json.video_details?.total_frames || null,
        framesSampled: json.video_details?.frames_sampled ?? null,
        framesWithFaces: json.video_details?.frames_with_faces ?? null,
        facesDetected: json.video_details?.faces_detected ?? null,
        facesAnalyzed: json.video_details?.faces_analyzed ?? null,
        framesAnalyzed: json.video_details?.frames_analyzed ?? null,
        ...json.video_details,
      },
      evidence: Array.isArray(json.evidence) ? json.evidence : [],
      metrics: json.metrics || {},
      frameAnalysis: json.frame_analysis || [],
      rawResponse: json,
      arbitration: json.arbitration || null,
      detectors: json.detectors || null,
      elapsedMs: json.latency_ms || clientElapsedMs,
      timestamp: json.timestamp || new Date().toISOString(),
      modelInfo: json.model_info || {
        name: 'dima806/deepfake_vs_real_image_detection',
        architecture: 'ViTForImageClassification'
      }
    };
  } catch (err) {
    if (err.name === 'AbortError') {
      const abortError = new Error('Video analysis was cancelled.');
      abortError.isAborted = true;
      throw abortError;
    }

    if (err.isHttpError) {
      throw err;
    }

    const connError = new Error(
      `Cannot connect to FastAPI backend at "${apiUrl}". Please ensure your FastAPI server is running with 'python -m uvicorn backend.main:app --reload --port 8000'.`
    );
    connError.isConnectionError = true;
    connError.apiUrl = apiUrl;
    connError.originalError = err.message;
    throw connError;
  }
}

/**
 * Checks backend health / connectivity status.
 */
export async function checkBackendHealth(healthUrl) {
  const url = healthUrl || (localStorage.getItem('truthlens_health_url') || DEFAULT_HEALTH_URL);
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3500);

    const res = await fetch(url, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json().catch(() => ({ status: 'ok' }));
      return { isOnline: true, details: data, url };
    }
    return { isOnline: false, error: `HTTP ${res.status}`, url };
  } catch (err) {
    return { isOnline: false, error: err.message, url };
  }
}
