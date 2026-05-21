import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App.jsx';

// These tests cover the full flow at the component level: file selection
// -> submit -> loading -> results/error rendering. Fetch is stubbed so
// the suite runs without a Flask backend.

function makeFile(name = 'song.wav') {
  return new File([new Uint8Array(1024)], name, { type: 'audio/wav' });
}

describe('App', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // Happy path

  it('shows predictions after a successful upload', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        predictions: [
          { genre: 'rock', confidence: 0.7 },
          { genre: 'metal', confidence: 0.2 },
        ],
      }),
    });

    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByTestId('file-input'), makeFile());
    await user.click(screen.getByRole('button', { name: /classify genre/i }));

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: /rock/i })).toBeInTheDocument()
    );
    expect(screen.getByText(/70\.0% confidence/)).toBeInTheDocument();
  });

  // Loading state

  it('shows a loading indicator while the request is in flight', async () => {
    let resolveRequest;
    fetch.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveRequest = resolve;
      })
    );

    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByTestId('file-input'), makeFile());
    await user.click(screen.getByRole('button', { name: /classify genre/i }));

    expect(screen.getByText(/analyzing audio/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /classifying/i })).toBeDisabled();

    // Cleanly resolve so the test doesn't leave a dangling promise.
    resolveRequest({
      ok: true,
      json: async () => ({ predictions: [{ genre: 'rock', confidence: 1 }] }),
    });
    await waitFor(() =>
      expect(screen.queryByText(/analyzing audio/i)).not.toBeInTheDocument()
    );
  });

  // Error handling

  it('shows the server error message when the backend returns 4xx', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 415,
      json: async () => ({ error: 'Unsupported audio codec' }),
    });

    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByTestId('file-input'), makeFile());
    await user.click(screen.getByRole('button', { name: /classify genre/i }));

    await waitFor(() => {
      const banner = screen.getByRole('alert');
      expect(banner).toHaveTextContent(/unsupported audio codec/i);
    });
  });

  it('shows a friendly message when the backend is unreachable', async () => {
    fetch.mockRejectedValueOnce(new TypeError('Failed to fetch'));

    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByTestId('file-input'), makeFile());
    await user.click(screen.getByRole('button', { name: /classify genre/i }));

    await waitFor(() => {
      const banner = screen.getByRole('alert');
      expect(banner).toHaveTextContent(/Flask server/i);
    });
  });

  it('allows re-submission after an error', async () => {
    fetch
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          predictions: [{ genre: 'jazz', confidence: 0.9 }],
        }),
      });

    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByTestId('file-input'), makeFile());
    await user.click(screen.getByRole('button', { name: /classify genre/i }));
    await waitFor(() => screen.getByRole('alert'));

    await user.click(screen.getByRole('button', { name: /classify genre/i }));

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: /jazz/i })).toBeInTheDocument()
    );
  });
});