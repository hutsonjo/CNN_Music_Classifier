import { useRef, useState } from 'react';

const ACCEPTED_EXTENSIONS = ['.wav', '.mp3', '.au', '.ogg', '.flac', '.m4a'];
const MAX_BYTES = 30 * 1024 * 1024; // 30 MB — plenty for a single song

/**
 * Validate the file before uploading. Returns an error string or null.
 * We surface validation errors locally so the user gets immediate
 * feedback without a server round-trip.
 */
function validate(file) {
  if (!file) return 'No file selected.';
  if (file.size > MAX_BYTES) {
    return `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max is 30 MB.`;
  }
  const lower = file.name.toLowerCase();
  if (!ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    return `Unsupported file type. Use ${ACCEPTED_EXTENSIONS.join(', ')}.`;
  }
  return null;
}

/**
 * File upload UI: drag-and-drop or click-to-browse, with local validation
 * and a clear "submit" affordance. The actual network call lives in App
 * so this component stays focused on file selection.
 */
export default function FileUploader({ onSubmit, isSubmitting }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleSelect = (selectedFile) => {
    const validationError = validate(selectedFile);
    if (validationError) {
      setFile(null);
      setError(validationError);
      return;
    }
    setError(null);
    setFile(selectedFile);
  };

  const handleSubmit = () => {
    if (!file || isSubmitting) return;
    onSubmit(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragOver(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) handleSelect(dropped);
  };

  return (
    <div className="uploader">
      <div
        className={`dropzone ${isDragOver ? 'dropzone--active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload an audio file"
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={(e) => handleSelect(e.target.files?.[0])}
          hidden
          data-testid="file-input"
        />
        <div className="dropzone__icon" aria-hidden="true">🎵</div>
        <p className="dropzone__primary">
          {file ? file.name : 'Drop an audio file here or click to browse'}
        </p>
        <p className="dropzone__secondary">
          {file
            ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
            : `Supported: ${ACCEPTED_EXTENSIONS.join(', ')}`}
        </p>
      </div>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      <button
        type="button"
        className="submit-button"
        onClick={handleSubmit}
        disabled={!file || isSubmitting}
      >
        {isSubmitting ? 'Classifying…' : 'Classify Genre'}
      </button>
    </div>
  );
}