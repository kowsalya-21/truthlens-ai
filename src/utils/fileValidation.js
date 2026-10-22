/**
 * TruthLens AI - File Validation & Metadata Utilities
 */

export const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB
export const SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mov', '.avi', '.mkv'];
export const SUPPORTED_VIDEO_MIME_TYPES = [
  'video/mp4',
  'video/webm',
  'video/quicktime',
  'video/x-msvideo',
  'video/avi',
  'video/x-matroska',
  'video/mkv'
];

/**
 * Format bytes to human readable string (KB, MB, GB)
 * @param {number} bytes
 * @param {number} [decimals=2]
 * @returns {string}
 */
export function formatBytes(bytes, decimals = 2) {
  if (!bytes || bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format seconds to MM:SS or HH:MM:SS
 * @param {number} seconds
 * @returns {string}
 */
export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || isNaN(seconds)) return '--:--';
  const s = Math.floor(seconds);
  const hrs = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = s % 60;

  if (hrs > 0) {
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Validates a candidate video file for type and size constraints.
 * 
 * @param {File} file
 * @param {object} [options]
 * @returns {{ isValid: boolean, error: string | null }}
 */
export function validateVideoFile(file, options = {}) {
  const maxSize = options.maxSizeBytes || MAX_FILE_SIZE_BYTES;

  if (!file) {
    return { isValid: false, error: 'No file selected.' };
  }

  // 1. Check file size
  if (file.size > maxSize) {
    return {
      isValid: false,
      error: `File size (${formatBytes(file.size)}) exceeds the maximum allowed limit of ${formatBytes(maxSize)}.`
    };
  }

  if (file.size === 0) {
    return {
      isValid: false,
      error: 'The selected file is empty (0 Bytes).'
    };
  }

  // 2. Check file type / extension
  const fileName = file.name ? file.name.toLowerCase() : '';
  const fileExt = fileName.slice(fileName.lastIndexOf('.'));
  const hasValidExt = SUPPORTED_VIDEO_EXTENSIONS.includes(fileExt);
  const hasValidMime = file.type ? SUPPORTED_VIDEO_MIME_TYPES.includes(file.type.toLowerCase()) : false;

  // Browsers sometimes leave file.type blank for .mkv or .avi, so accept if either MIME or extension matches
  if (!hasValidExt && !hasValidMime && !file.type.startsWith('video/')) {
    return {
      isValid: false,
      error: `Unsupported file format "${fileExt || file.type || 'unknown'}". Please upload a video in MP4, WebM, MOV, AVI, or MKV format.`
    };
  }

  return { isValid: true, error: null };
}

/**
 * Extracts resolution, duration, and aspect ratio from a video File using HTML5 Video.
 * 
 * @param {File} file
 * @returns {Promise<{ duration: number, width: number, height: number, resolution: string, aspectRatio: string }>}
 */
export function extractVideoMetadata(file) {
  return new Promise((resolve) => {
    try {
      const url = URL.createObjectURL(file);
      const video = document.createElement('video');
      video.preload = 'metadata';

      video.onloadedmetadata = () => {
        URL.revokeObjectURL(url);
        const width = video.videoWidth || 0;
        const height = video.videoHeight || 0;
        const duration = video.duration || 0;

        let resolution = 'Unknown';
        if (width && height) {
          if (width >= 3840 || height >= 2160) resolution = `4K (${width}x${height})`;
          else if (width >= 1920 || height >= 1080) resolution = `1080p FHD (${width}x${height})`;
          else if (width >= 1280 || height >= 720) resolution = `720p HD (${width}x${height})`;
          else resolution = `${width}x${height} SD`;
        }

        resolve({
          duration,
          width,
          height,
          resolution,
          aspectRatio: width && height ? `${(width / height).toFixed(2)}:1` : 'Unknown',
        });
      };

      video.onerror = () => {
        URL.revokeObjectURL(url);
        resolve({
          duration: 0,
          width: 0,
          height: 0,
          resolution: 'Unknown',
          aspectRatio: 'Unknown'
        });
      };

      video.src = url;
    } catch {
      resolve({
        duration: 0,
        width: 0,
        height: 0,
        resolution: 'Unknown',
        aspectRatio: 'Unknown'
      });
    }
  });
}
