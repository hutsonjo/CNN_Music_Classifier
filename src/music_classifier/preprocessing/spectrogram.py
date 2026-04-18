"""Mel-spectrogram generation from pre-segmented waveform arrays.

A *mel-spectrogram* transforms a 1-D audio waveform into a 2-D matrix that a
CNN can treat like an image.  The steps are:

1. **Short-Time Fourier Transform (STFT)** — slide a short window along the
   waveform and compute the frequency content of each frame using an FFT.
   This produces a *spectrogram*: a matrix of energy values indexed by
   (frequency bin, time frame).

2. **Mel filter bank** — group the raw FFT frequency bins into ``n_mels``
   bands spaced on the *mel scale*, which compresses high frequencies the
   same way the human auditory system does.  This discards perceptually
   irrelevant frequency detail and keeps the output compact.

3. **dB conversion** — convert power values to decibels (log scale) so that
   quiet and loud passages occupy a similar numeric range.  Without this,
   the dynamic range of the raw power spectrum would make training unstable.

The result is an array of shape ``(n_mels, n_frames)`` per segment, where
each element is the log-power at a particular mel frequency band and time
frame.  Stacking all segments from one audio file gives a 3-D array of shape
``(n_segments, n_mels, n_frames)``.
"""

from __future__ import annotations

import librosa
import numpy as np


def segments_to_mel_spectrograms(
    segments: np.ndarray,
    sr: int,
    *,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmax: float | None,
) -> np.ndarray:
    """Convert a batch of waveform segments into mel-spectrograms.

    Each row of *segments* is treated as an independent audio clip and
    converted to its own 2-D mel-spectrogram.  The results are stacked into
    a single 3-D array so they can be fed directly into a CNN as a batch.

    Parameters
    ----------
    segments:
        2-D float array of shape ``(n_segments, n_samples)``.  Each row is
        one fixed-length waveform clip, as produced by ``segment_waveform``.
        The array must be 2-D; pass a single segment as
        ``segments[np.newaxis, :]`` if needed.
    sr:
        Sample rate of the waveforms in Hz.  Must match the rate used when
        the audio was loaded — mismatching this will silently produce
        spectrograms with wrong frequency labels.
    n_mels:
        Number of mel frequency bands (rows in each output spectrogram).
        See ``SpectrogramConfig.n_mels`` for guidance on this value.
    n_fft:
        FFT window size in samples.  See ``SpectrogramConfig.n_fft``.
    hop_length:
        Samples between consecutive FFT frames.  Controls time resolution
        (number of columns in the output).  See ``SpectrogramConfig.hop_length``.
    fmax:
        Highest frequency in Hz to include in the mel filter bank.  ``None``
        uses the Nyquist limit (``sr / 2``).

    Returns
    -------
    np.ndarray
        3-D float32 array of shape ``(n_segments, n_mels, n_frames)`` where
        values are in decibels (dB).  ``n_frames`` is determined by the
        segment length and ``hop_length``:
        ``n_frames = floor(n_samples / hop_length) + 1``.

    Raises
    ------
    ValueError
        If *segments* is not 2-D or contains zero rows.
    """
    if segments.ndim != 2:
        raise ValueError(
            f"segments must be 2-D (n_segments, n_samples), got shape {segments.shape}. "
            "If you have a single segment, reshape it with segments[np.newaxis, :]."
        )
    if segments.shape[0] == 0:
        raise ValueError("segments array has zero rows — nothing to convert.")

    out: list[np.ndarray] = []
    for segment in segments:
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmax=fmax,
        )
        # Convert power spectrogram to dB scale.  ref=np.max anchors the
        # dB values relative to the loudest frequency in each clip, keeping
        # the dynamic range consistent across clips of different overall volume.
        mel_db = librosa.power_to_db(mel, ref=np.max)
        out.append(mel_db.astype(np.float32))

    return np.stack(out, axis=0)  # (n_segments, n_mels, n_frames)
