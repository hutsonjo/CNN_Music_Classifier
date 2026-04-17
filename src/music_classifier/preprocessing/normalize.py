"""Per-spectrogram normalization for mel-spectrogram arrays.

After dB conversion, spectrogram values typically fall in the range [-80, 0]
dB, but the exact range varies between clips depending on their content and
volume.  Normalization rescales each spectrogram to a standard numeric range
so the model sees consistent input magnitudes regardless of the source clip.

Normalization is applied *per spectrogram* (one clip at a time) rather than
globally across the whole dataset.  This makes each clip self-contained,
which is critical for inference: when a user submits a single audio clip, the
model can normalize it without needing access to training-set statistics.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def normalize_spectrograms(
    spectrograms: np.ndarray,
    *,
    strategy: Literal["minmax", "standardize"] = "minmax",
) -> np.ndarray:
    """Normalize a batch of mel-spectrograms independently.

    Each spectrogram in the batch is normalized on its own — the statistics
    (min/max or mean/std) are computed per-spectrogram, not across the batch.

    Parameters
    ----------
    spectrograms:
        3-D float array of shape ``(n_segments, n_mels, n_frames)``, as
        returned by ``segments_to_mel_spectrograms``.
    strategy:
        Normalization method to apply to each spectrogram:

        - ``"minmax"`` (default): scales values to [0, 1] by subtracting the
          per-spectrogram minimum and dividing by the per-spectrogram range.
          If a spectrogram is perfectly flat (all values identical, range = 0),
          it is returned as all-zeros rather than producing a NaN.

        - ``"standardize"``: subtracts the per-spectrogram mean and divides by
          the per-spectrogram standard deviation, producing approximately
          zero-mean unit-variance values.  If the standard deviation is zero
          (flat spectrogram), returns all-zeros.

    Returns
    -------
    np.ndarray
        Float32 array of the same shape as *spectrograms* with normalized
        values.  The input array is not modified in-place.

    Raises
    ------
    ValueError
        If *spectrograms* is not 3-D or *strategy* is not recognised.
    """
    if spectrograms.ndim != 3:
        raise ValueError(
            f"spectrograms must be 3-D (n_segments, n_mels, n_frames), "
            f"got shape {spectrograms.shape}."
        )
    if strategy not in ("minmax", "standardize"):
        raise ValueError(
            f"strategy must be 'minmax' or 'standardize', got {strategy!r}."
        )

    result = np.empty_like(spectrograms, dtype=np.float32)

    for i, spec in enumerate(spectrograms):
        if strategy == "minmax":
            lo = spec.min()
            hi = spec.max()
            span = hi - lo
            if span == 0.0:
                result[i] = np.zeros_like(spec, dtype=np.float32)
            else:
                result[i] = ((spec - lo) / span).astype(np.float32)
        else:  # standardize
            mean = spec.mean()
            std = spec.std()
            if std == 0.0:
                result[i] = np.zeros_like(spec, dtype=np.float32)
            else:
                result[i] = ((spec - mean) / std).astype(np.float32)

    return result
