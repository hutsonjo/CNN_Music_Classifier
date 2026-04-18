"""Configuration dataclasses for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable configuration for audio loading and segmentation.

    Attributes
    ----------
    target_sr:
        Target sample rate in Hz — the number of audio samples captured per
        second.  All files are resampled to this rate on load, regardless of
        what rate they were recorded at.  22 050 Hz is the standard default for
        music analysis: it captures all frequencies up to ~11 kHz (sufficient
        for most music content) while keeping arrays half the size of CD-quality
        audio (44 100 Hz).

    segment_seconds:
        Each audio track is chopped into non-overlapping (or overlapping, see
        ``hop_seconds``) clips of this length before being fed to the model.
        Using fixed-length clips is necessary because neural networks expect
        inputs of a consistent shape.  3 seconds is a common choice for genre
        classification — long enough to capture rhythmic and harmonic patterns,
        short enough to produce many training examples per track.

    hop_seconds:
        How far to advance the window start between consecutive segments,
        measured in seconds.  Think of it like sliding a 3-second window along
        the track:

        - ``None`` (default): the window jumps forward by exactly
          ``segment_seconds`` each step, so segments do not overlap.
        - A value smaller than ``segment_seconds`` (e.g. 1.5 for a 3-second
          segment): the window advances by that amount, so adjacent segments
          share audio.  This is called *overlapping* and multiplies the number
          of training examples at the cost of some redundancy.

    pad_short:
        Controls what happens when a track's final chunk is shorter than
        ``segment_seconds`` (e.g. the last 1.2 s of a 31-second track with 3 s
        segments):

        - ``False`` (default): the short tail is silently discarded.  Use this
          when you want every segment to contain a full, unmodified clip.
        - ``True``: the tail is kept and extended to ``segment_seconds`` by
          appending silence (zeros) at the end.  Use this when you cannot afford
          to lose any audio, such as during inference on a user-uploaded clip.

    mono:
        Whether to collapse a stereo (or multi-channel) recording into a single
        channel before processing.

        - ``True`` (default): channels are averaged into one waveform.  This is
          almost always the right choice for genre classification because genre
          does not depend on stereo positioning, and keeping a single channel
          halves memory usage and keeps the model architecture simple.
        - ``False``: each channel is kept separate.  Only useful if downstream
          processing explicitly requires multi-channel input.
    """

    target_sr: int = 22050
    segment_seconds: float = 3.0
    hop_seconds: float | None = None
    pad_short: bool = False
    mono: bool = True

    def __post_init__(self) -> None:
        if self.target_sr <= 0:
            raise ValueError(f"target_sr must be positive, got {self.target_sr}.")
        if self.segment_seconds <= 0:
            raise ValueError(
                f"segment_seconds must be positive, got {self.segment_seconds}."
            )
        if self.hop_seconds is not None and self.hop_seconds <= 0:
            raise ValueError(
                f"hop_seconds must be positive when set, got {self.hop_seconds}."
            )


@dataclass(frozen=True)
class SpectrogramConfig:
    """Immutable configuration for mel-spectrogram generation and normalization.

    A mel-spectrogram is a 2-D image-like representation of audio.  The
    horizontal axis is time (divided into short overlapping frames), the
    vertical axis is frequency (grouped into *mel* bands that mimic how the
    human ear perceives pitch), and the pixel value is the energy (in dB) at
    that time-frequency point.  CNNs treat this 2-D array the same way they
    treat image pixels, which is why spectrograms are the standard input
    format for audio classification models.

    Attributes
    ----------
    n_mels:
        Number of mel frequency bands (rows in the output spectrogram).
        128 is the standard for music tasks — fine-grained enough to
        distinguish instruments and harmonics, compact enough to keep the
        model input small.

    n_fft:
        Size of the Fast Fourier Transform (FFT) window in samples.  Each
        short frame of audio is transformed from the time domain into the
        frequency domain using an FFT of this size.  2048 samples at
        22 050 Hz corresponds to ~93 ms per frame — short enough to capture
        temporal changes, long enough to resolve low frequencies cleanly.

    hop_length:
        Number of samples between the start of consecutive FFT frames.  This
        controls the time resolution of the spectrogram (the number of
        columns in the output).  512 samples ≈ 23 ms per step.  For a 3-
        second segment at 22 050 Hz: ⌊(3 × 22050) / 512⌋ + 1 ≈ 130 frames,
        giving an output shape of ``(n_mels, 130)`` per segment.

    fmax:
        Highest frequency (in Hz) included in the mel filter bank.  ``None``
        uses the Nyquist limit (``target_sr / 2``), which for 22 050 Hz is
        11 025 Hz — the maximum representable frequency.  Setting a lower
        value (e.g. 8 000 Hz) discards very high frequencies that carry
        little genre-relevant information and can improve model convergence.

    normalize:
        Normalization strategy applied to each spectrogram independently
        after dB conversion:

        - ``"minmax"`` (default): scales values to [0, 1] by subtracting the
          minimum and dividing by the range.  Keeps relative energy
          differences intact and is the most common choice for CNN inputs.
        - ``"standardize"``: subtracts the mean and divides by the standard
          deviation, producing zero-mean unit-variance values.  Can help
          when the model uses batch normalization layers.

        Normalization is applied *per spectrogram* (not globally across the
        whole dataset) so that each individual clip is self-contained.  This
        is essential for inference, where you process one clip at a time
        without access to dataset-wide statistics.
    """

    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmax: float | None = None
    normalize: Literal["minmax", "standardize"] = "minmax"

    def __post_init__(self) -> None:
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}.")
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}.")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}.")
        if self.fmax is not None and self.fmax <= 0:
            raise ValueError(f"fmax must be positive when set, got {self.fmax}.")
        if self.normalize not in ("minmax", "standardize"):
            raise ValueError(
                f"normalize must be 'minmax' or 'standardize', got {self.normalize!r}."
            )
