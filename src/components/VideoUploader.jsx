import React, { useState, useRef } from 'react';
import { UploadCloud, Film, AlertTriangle, FileCheck, ShieldAlert } from 'lucide-react';
import { validateVideoFile, formatBytes, MAX_FILE_SIZE_BYTES } from '../utils/fileValidation';

export default function VideoUploader({ onFileSelected }) {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const fileInputRef = useRef(null);

  const handleProcessFile = (file) => {
    if (!file) return;

    // Validate file type and size
    const validation = validateVideoFile(file);
    if (!validation.isValid) {
      setValidationError(validation.error);
      return;
    }

    setValidationError(null);
    onFileSelected(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      handleProcessFile(droppedFile);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      handleProcessFile(selectedFile);
    }
  };

  const handleTriggerBrowse = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="uploader-wrapper fade-in" id="analyzer-section">
      <div
        className={`dropzone-container ${isDragging ? 'is-dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleTriggerBrowse}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && handleTriggerBrowse()}
        aria-label="Upload video for deepfake analysis"
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="video/mp4,video/webm,video/quicktime,video/x-msvideo,video/x-matroska,.mp4,.webm,.mov,.avi,.mkv"
          style={{ display: 'none' }}
          id="video-file-input"
        />

        <div className="dropzone-icon-ring">
          <UploadCloud size={36} />
        </div>

        <h3 className="dropzone-heading">
          {isDragging ? 'Drop Video to Inspect' : 'Drag & Drop Video or Browse'}
        </h3>
        <p className="dropzone-subtext">
          Upload any video containing facial sequences to perform neural synthesis analysis
        </p>

        <div>
          <button
            type="button"
            className="dropzone-button"
            onClick={(e) => {
              e.stopPropagation();
              handleTriggerBrowse();
            }}
          >
            <Film size={18} />
            <span>Select Video File</span>
          </button>
        </div>

        <div className="upload-specs">
          <span>
            <FileCheck size={14} color="var(--cyan-400)" />
            Supported: MP4, WebM, MOV, AVI, MKV
          </span>
          <span>
            <ShieldAlert size={14} color="var(--violet-400)" />
            Max file size: {formatBytes(MAX_FILE_SIZE_BYTES)}
          </span>
        </div>
      </div>

      {/* Validation Error Alert */}
      {validationError && (
        <div className="upload-error-alert fade-in" role="alert">
          <AlertTriangle size={20} flexShrink={0} />
          <div>
            <strong>File Validation Failed:</strong> {validationError}
          </div>
        </div>
      )}
    </div>
  );
}
