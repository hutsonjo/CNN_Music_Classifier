import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { predictGenre, ApiError } from '../api.js';

// predictGenre — happy path

describe('predictGenre', () => {
  const sampleFile = new File(['fake audio bytes'], 'song.wav', { type: 'audio/wav' });

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('posts the file as multipart form data to /api/predict', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        predictions: [{ genre: 'rock', probability: 0.7 }],
      }),
    });

    await predictGenre(sampleFile);

    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe('/api/predict');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect(init.body.get('audio')).toBe(sampleFile);
  });

  it('returns the predictions array on a 200 response', async () => {
    const expected = [
      { genre: 'rock', probability: 0.7 },
      { genre: 'metal', probability: 0.2 },
    ];
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ predictions: expected }),
    });

    const result = await predictGenre(sampleFile);

    expect(result).toEqual(expected);
  });

  // predictGenre — error handling

  it('throws ApiError with the server-provided error message on non-2xx', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ error: 'Unsupported audio format' }),
    });

    await expect(predictGenre(sampleFile)).rejects.toMatchObject({
      name: 'ApiError',
      message: 'Unsupported audio format',
      status: 400,
    });
  });

  it('throws ApiError with a generic message when the error body is not JSON', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error('not json');
      },
    });

    await expect(predictGenre(sampleFile)).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
      message: expect.stringContaining('500'),
    });
  });

  it('throws ApiError with a helpful message when fetch itself fails', async () => {
    fetch.mockRejectedValueOnce(new TypeError('Failed to fetch'));

    await expect(predictGenre(sampleFile)).rejects.toMatchObject({
      name: 'ApiError',
      status: 0,
      message: expect.stringContaining('Flask server'),
    });
  });

  it('throws ApiError when the response shape is malformed', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ wrong_key: 'oops' }),
    });

    await expect(predictGenre(sampleFile)).rejects.toBeInstanceOf(ApiError);
  });
});