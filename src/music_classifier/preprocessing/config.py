"""Configuration dataclass for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


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
