"""Fixed-length waveform segmentation with configurable overlap and short-clip policy.

*Segmentation* is the process of chopping a long audio waveform into a sequence
of shorter, equal-length clips.  Neural networks require all inputs to have the
same shape, so we cannot feed a 30-second waveform and a 28-second waveform to
the same model batch without either truncating or padding.  Segmenting into
fixed-length windows solves this cleanly and has the bonus of multiplying the
number of training examples: one 30-second GTZAN track becomes 10 non-overlapping
3-second segments, each treated as an independent training sample.
"""

from __future__ import annotations

import numpy as np


def segment_waveform(
    y: np.ndarray,
    sr: int,
    *,
    segment_seconds: float,
    hop_seconds: float | None = None,
    pad_short: bool = False,
) -> np.ndarray:
    """Split a 1-D waveform into a 2-D array of fixed-length segments.

    Conceptually this slides a window of ``segment_seconds`` along the
    waveform, collecting each frame, then returns all frames stacked into a
    matrix.  The *stride* between consecutive window positions is
    ``hop_seconds``.

    Example (non-overlapping, 30 s track, 3 s segments)::

        |---seg 0---|---seg 1---|---seg 2---| ... |---seg 9---|
        0s          3s          6s                27s         30s

    Example (overlapping, 30 s track, 3 s segment, 1.5 s hop)::

        |---seg 0---|
             |---seg 1---|
                  |---seg 2---|
                       ...

    Parameters
    ----------
    y:
        1-D float waveform array of shape ``(n_samples,)``.  Each element is
        an amplitude value, as returned by ``load_audio``.  The array must be
        1-D (mono); call ``load_audio(..., mono=True)`` or average channels
        before passing multi-channel audio here.
    sr:
        Sample rate of *y* in Hz — how many elements of *y* represent one
        second of audio.  Needed to convert ``segment_seconds`` and
        ``hop_seconds`` from time units into array index offsets.
    segment_seconds:
        Length of each output window in seconds.  See ``PreprocessConfig``
        for guidance on choosing this value.
    hop_seconds:
        Distance between the start of one window and the start of the next,
        in seconds.  ``None`` (default) means the windows do not overlap —
        the next window starts exactly where the previous one ended.  A value
        smaller than ``segment_seconds`` creates overlapping windows, which
        increases the number of segments and introduces redundancy between
        adjacent ones.
    pad_short:
        Policy for the final chunk when the track length is not an exact
        multiple of ``segment_seconds``.  See ``PreprocessConfig.pad_short``
        for a full explanation of when to use each option.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_segments, segment_samples)`` where
        ``segment_samples = round(segment_seconds * sr)``.  Row *i* is the
        waveform for segment *i*.  If *y* is shorter than one full segment
        and ``pad_short=False``, the returned array has shape
        ``(0, segment_samples)`` — an empty batch with the correct column
        count so downstream shape checks still pass.

    Raises
    ------
    ValueError
        If *y* is not 1-D, ``segment_seconds`` is non-positive, or *sr* is
        non-positive.
    """
    if y.ndim != 1:
        raise ValueError(
            f"Expected a 1-D waveform, got shape {y.shape}. "
            "Downmix to mono before segmenting."
        )
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}.")
    if segment_seconds <= 0:
        raise ValueError(f"segment_seconds must be positive, got {segment_seconds}.")

    if hop_seconds is None:
        hop_seconds = segment_seconds
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be positive, got {hop_seconds}.")

    # Convert time durations to array index counts.  Rounding (rather than
    # truncating) keeps the segment length as close to the requested duration
    # as possible when sr doesn't divide evenly into the requested seconds.
    segment_samples = int(round(segment_seconds * sr))
    hop_samples = int(round(hop_seconds * sr))

    n_samples = len(y)
    segments: list[np.ndarray] = []

    start = 0
    while start < n_samples:
        end = start + segment_samples
        chunk = y[start:end]

        if len(chunk) < segment_samples:
            if pad_short:
                padded = np.zeros(segment_samples, dtype=y.dtype)
                padded[: len(chunk)] = chunk
                segments.append(padded)
            # else: drop the short tail
            break

        segments.append(chunk.copy())
        start += hop_samples

    if not segments:
        return np.empty((0, segment_samples), dtype=y.dtype)

    return np.stack(segments, axis=0)
