
const API_BASE = '/api';

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/**
 * Send an audio file to the backend and return ranked genre predictions.
 *
 * @param {File} file - The audio file selected by the user.
 * @param {number} [topN] - Optional cap on number of predictions returned.
 * @returns {Promise<Array<{ genre: string, confidence: number }>>}
 * @throws {ApiError} on non-2xx responses or network failure.
 */
export async function predictGenre(file, topN) {
  const form = new FormData();
  form.append('file', file);
  if (typeof topN === 'number') {
    form.append('top_n', String(topN));
  }

  let response;
  try {
    response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: form,
    });
  } catch (networkError) {
    throw new ApiError(
      'Could not reach the classifier. Check that the Flask server is running.',
      0
    );
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.error) {
        detail = payload.error;
      }
    } catch {
      // Response wasn't JSON — keep the generic message.
    }
    throw new ApiError(detail, response.status);
  }

  const payload = await response.json();
  if (!payload || !Array.isArray(payload.predictions)) {
    throw new ApiError('Malformed response from server.', 502);
  }
  return payload.predictions;
}