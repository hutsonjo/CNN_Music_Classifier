import { useState } from 'react';
import FileUploader from './components/FileUploader.jsx';
import PredictionResults from './components/PredictionResults.jsx';
import { predictGenre, ApiError } from './api.js';

/**
 * Top-level app: holds upload/loading/result state and orchestrates the
 * call to the Flask backend. Three states drive the UI:
 *
 *   - status='idle'      → uploader visible, no results
 *   - status='loading'   → uploader disabled, progress bar visible
 *   - status='success'   → results visible alongside the uploader
 *   - status='error'     → error banner visible, uploader re-enabled
 *
 * Errors from the API client are caught here and surfaced as a banner;
 * local file-validation errors are handled inside FileUploader.
 */

export default function App() {
  const [status, setStatus] = useState('idle');
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (file) => {
    setStatus('loading');
    setError(null);

    try {
      const results = await predictGenre(file);
      setPredictions(results);
      setStatus('success');
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : 'Unexpected error. See console for details.';
      if (!(err instanceof ApiError)) {
        console.error(err);
      }
      setError(message);
      setStatus('error');
    }
  };

  return (
    <main className="app">
      <header className="app__header">
        <h1>CNN Music Classifier</h1>
        <p className="app__tagline">
          Upload a song and the model will rank its top genre matches.
        </p>
      </header>

      <FileUploader onSubmit={handleSubmit} isSubmitting={status === 'loading'} />

      {status === 'loading' && (
        <div className="progress" role="status" aria-live="polite">
          <div className="progress__bar" />
          <span className="progress__text">Analyzing audio…</span>
        </div>
      )}

      {status === 'error' && error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      {status === 'success' && <PredictionResults predictions={predictions} />}

    
    </main>
  );
}