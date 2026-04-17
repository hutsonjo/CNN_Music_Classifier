"""Stratified train/validation/test splitting at the file level.

Splitting at the *file level* (rather than the segment level) is critical for
preventing data leakage.  If segments from the same 30-second track appeared
in both the training set and the test set, the model could learn the specific
audio fingerprint of that track rather than generalising to unseen music.  By
ensuring each source file belongs to exactly one split, we guarantee that the
test set truly measures performance on audio the model has never encountered.

*Stratified* splitting ensures that each genre is proportionally represented
in every split.  Without stratification, random chance could put most of the
jazz files in the training set and few in the test set, making the test
evaluation unreliable for that genre.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .pipeline import SpectrogramRecord

T = TypeVar("T")


def stratified_split(
    records: list[SpectrogramRecord],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[SpectrogramRecord], list[SpectrogramRecord], list[SpectrogramRecord]]:
    """Split a list of ``SpectrogramRecord`` objects into train/val/test subsets.

    The split is performed at the *file* level — every record (one per source
    audio file) is assigned to exactly one subset.  Records with the same
    genre label are shuffled and split independently so each subset receives a
    proportional share of every genre (*stratification*).

    Parameters
    ----------
    records:
        List of ``SpectrogramRecord`` dicts, one per source audio file, as
        yielded by ``build_spectrogram_dataset``.
    train_ratio:
        Fraction of files to place in the training set.  Defaults to 0.8
        (80 %).
    val_ratio:
        Fraction of files to place in the validation set.  Defaults to 0.1
        (10 %).  The validation set is used during training to monitor
        overfitting and tune hyperparameters — the model never trains on it.
    test_ratio:
        Fraction of files to place in the test set.  Defaults to 0.1 (10 %).
        The test set is held out entirely until final evaluation; it must not
        influence any training or tuning decisions.
    seed:
        Random seed for the shuffle.  Using the same seed always produces the
        same split, which is essential for reproducibility: every team member
        and every CI run must evaluate the model on exactly the same files.

    Returns
    -------
    (train, val, test)
        Three lists of ``SpectrogramRecord``.  Every record from *records*
        appears in exactly one of the three lists.

    Raises
    ------
    ValueError
        If the ratios do not sum to approximately 1.0, any ratio is negative,
        or *records* is empty.
    """
    if not records:
        raise ValueError("records list is empty — nothing to split.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total:.6f}."
        )
    for name, ratio in [("train_ratio", train_ratio), ("val_ratio", val_ratio),
                        ("test_ratio", test_ratio)]:
        if ratio < 0:
            raise ValueError(f"{name} must be non-negative, got {ratio}.")

    # Group records by genre label for stratified shuffling.
    by_label: dict[str, list[SpectrogramRecord]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    rng = random.Random(seed)

    train: list[SpectrogramRecord] = []
    val: list[SpectrogramRecord] = []
    test: list[SpectrogramRecord] = []

    for label_records in by_label.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)

        n = len(shuffled)
        # Integer boundaries — use floor for train/val so test gets any remainder.
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    return train, val, test
