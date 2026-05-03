#!/usr/bin/env python3
"""Quick smoke runner for the preprocessing pipeline.

Usage
-----
python scripts/run_preprocess_smoke.py --dataset-root training_data/gtzan_dataset

# With all options
python scripts/run_preprocess_smoke.py \\
    --dataset-root training_data/gtzan_dataset \\
    --target-sr 22050 \\
    --segment-seconds 3.0 \\
    --hop-seconds 1.5 \\
    --limit-files 20
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline and print a summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("training_data/gtzan_dataset"),
        help="Root directory of the GTZAN-style dataset.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=22050,
        help="Target sample rate in Hz.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=3.0,
        help="Duration of each output segment in seconds.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=None,
        help="Hop between segment starts in seconds (None = non-overlapping).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Stop after processing this many files (useful for quick checks).",
    )
    parser.add_argument(
        "--pad-short",
        action="store_true",
        default=False,
        help="Zero-pad the final short segment rather than dropping it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Lazy import so script is importable without the package installed.
    try:
        from music_classifier.preprocessing.config import PreprocessConfig
        from music_classifier.preprocessing.pipeline import preprocess_dataset
    except ImportError as exc:
        print(
            f"[ERROR] Cannot import music_classifier: {exc}\n"
            "Make sure the package is installed:  pip install -e .[dev]",
            file=sys.stderr,
        )
        return 1

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        print(
            f"[ERROR] Dataset root not found: {dataset_root}",
            file=sys.stderr,
        )
        return 1

    config = PreprocessConfig(
        target_sr=args.target_sr,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        pad_short=args.pad_short,
    )

    hop_display = f"{config.hop_seconds}s" if config.hop_seconds is not None else "None (non-overlapping)"
    print(f"Dataset root  : {dataset_root}")
    print(f"Config        : sr={config.target_sr}  seg={config.segment_seconds}s  "
          f"hop={hop_display}  pad_short={config.pad_short}")
    if args.limit_files:
        print(f"Limit         : {args.limit_files} files")
    print()

    total_files = 0
    total_segments = 0
    segments_per_label: Counter[str] = Counter()
    errors: list[tuple[Path, str]] = []

    t0 = time.perf_counter()

    for record in preprocess_dataset(dataset_root, config):
        try:
            n_segs = record["segments"].shape[0]
            total_files += 1
            total_segments += n_segs
            segments_per_label[record["label"]] += n_segs

            if total_files % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  ... {total_files} files processed ({elapsed:.1f}s elapsed)")

            if args.limit_files and total_files >= args.limit_files:
                break

        except Exception as exc:  # noqa: BLE001
            errors.append((record["path"], str(exc)))

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 52)
    print(f"  Files processed : {total_files}")
    print(f"  Total segments  : {total_segments}")
    print(f"  Elapsed         : {elapsed:.2f}s")
    print()
    print("  Segments per label:")
    for label, count in sorted(segments_per_label.items()):
        print(f"    {label:<12} {count:>6}")
    print("=" * 52)

    if errors:
        print(f"\n  {len(errors)} file(s) failed:")
        for path, msg in errors:
            print(f"    {path.name}: {msg}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
