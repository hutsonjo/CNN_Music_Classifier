"""Serialisation and deserialisation of processed spectrogram datasets.

After the full preprocessing pipeline (load → segment → mel-spectrogram →
normalize), saving the results to disk avoids reprocessing 1 000 audio files
every time the model is trained.  A single preprocessing run takes a few
seconds; reloading the saved `.npz` file takes milliseconds.

The `.npz` format (NumPy compressed archive) is used because:
- It is self-contained (one file holds all arrays and metadata).
- It loads directly into NumPy arrays with no external dependencies.
- Arrays can be passed straight to TensorFlow/Keras ``model.fit()``.
- The format is human-inspectable with ``np.load``.

Storage layout inside the `.npz` file
--------------------------------------
``X``
    Float32 array of shape ``(total_segments, n_mels, n_frames)``.  Each
    slice ``X[i]`` is one normalised mel-spectrogram ready for the model.
``y``
    Int64 array of shape ``(total_segments,)``.  Each element is the integer
    class index of the genre for that segment (0–9 for GTZAN).
``label_names``
    1-D array of strings of length ``n_classes``.  ``label_names[i]`` is the
    genre name for class index ``i``.  Stored alongside ``X`` and ``y`` so
    the file is self-documenting — you never need to remember what index
    corresponds to which genre.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .pipeline import SpectrogramRecord


def save_dataset(records: list[SpectrogramRecord], output_path: Path) -> None:
    """Serialise a list of ``SpectrogramRecord`` objects to a ``.npz`` file.

    All spectrograms from all records are stacked into a single ``X`` array.
    Genre labels are integer-encoded in sorted order (alphabetical by genre
    name) so the mapping is deterministic and does not depend on the order
    records are passed in.

    Parameters
    ----------
    records:
        List of ``SpectrogramRecord`` dicts as yielded by
        ``build_spectrogram_dataset``.  Each record contributes
        ``record["spectrograms"].shape[0]`` rows to ``X``.
    output_path:
        Destination file path.  The ``.npz`` extension is conventional but
        not enforced — NumPy will save correctly regardless of extension.
        Parent directories must already exist.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("records list is empty — nothing to save.")

    output_path = Path(output_path)

    # Build a sorted, deterministic label → integer mapping.
    label_names: list[str] = sorted({r["label"] for r in records})
    label_to_idx: dict[str, int] = {name: i for i, name in enumerate(label_names)}

    all_spectrograms: list[np.ndarray] = []
    all_labels: list[int] = []

    for record in records:
        specs = record["spectrograms"]  # (n_segments, n_mels, n_frames)
        n_segs = specs.shape[0]
        all_spectrograms.append(specs)
        all_labels.extend([label_to_idx[record["label"]]] * n_segs)

    X = np.concatenate(all_spectrograms, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        label_names=np.array(label_names),
    )


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a dataset saved by ``save_dataset`` and return model-ready arrays.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file written by ``save_dataset``.

    Returns
    -------
    (X, y, label_names)
        ``X``
            Float32 array of shape ``(total_segments, n_mels, n_frames)``.
            Slice ``X[i]`` is the mel-spectrogram for segment ``i``.

        ``y``
            Int64 array of shape ``(total_segments,)``.  Element ``y[i]`` is
            the integer genre index for segment ``i``.

        ``label_names``
            List of genre name strings.  ``label_names[k]`` is the genre
            corresponding to integer class ``k`` in ``y``.  Pass this to your
            evaluation code so confusion-matrix axes are human-readable.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If the archive is missing an expected array (e.g. it was not written
        by ``save_dataset``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    archive = np.load(path, allow_pickle=False)

    X: np.ndarray = archive["X"]
    y: np.ndarray = archive["y"]
    label_names: list[str] = archive["label_names"].tolist()

    return X, y, label_names
