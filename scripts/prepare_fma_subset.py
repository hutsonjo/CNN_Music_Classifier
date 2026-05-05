#!/usr/bin/env python3
"""Prepare a balanced FMA subset for ingestion by the CNN music classifier.

Reads FMA metadata, maps FMA top-level genres to GTZAN genre names (see
docs/fma_genre_mapping.md), samples a fixed number of tracks per genre, and
copies the MP3 files into a GTZAN-style directory layout:

    <output-root>/
        hiphop/000002.mp3
        pop/000134.mp3
        rock/000140.mp3
        ...

The output directory can be passed directly to the existing preprocessing
pipeline in place of training_data/gtzan_dataset/.

Usage
-----
# Dry run — show what would be copied without touching the filesystem:
python scripts/prepare_fma_subset.py \\
    --fma-root     ../fma_small \\
    --metadata-dir ../fma_metadata \\
    --dry-run

# Full run with defaults (200 tracks/genre, 1 000 total):
python scripts/prepare_fma_subset.py \\
    --fma-root     ../fma_small \\
    --metadata-dir ../fma_metadata

# Custom output location or track count:
python scripts/prepare_fma_subset.py \\
    --fma-root          ../fma_small \\
    --metadata-dir      ../fma_metadata \\
    --output-root       training_data/fma_subset \\
    --tracks-per-genre  200 \\
    --seed              42
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Genre mapping — see docs/fma_genre_mapping.md for rationale
# ---------------------------------------------------------------------------
# Keys are FMA `genre_top` values; values are GTZAN directory names.
# FMA genres absent from this dict (Experimental, International, Instrumental)
# are intentionally excluded — no reasonable GTZAN equivalent exists.
FMA_TO_GTZAN: dict[str, str] = {
    "Hip-Hop":    "hiphop",   # strong match
    "Pop":        "pop",      # strong match
    "Rock":       "rock",     # strong match
    "Folk":       "country",  # moderate match — acoustic/roots overlap
    "Electronic": "disco",    # weak match   — closest rhythmic/dance GTZAN class
}


def fma_audio_path(fma_root: Path, track_id: int) -> Path:
    """Return the expected MP3 path for a given FMA track ID."""
    subdir = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"
    return fma_root / subdir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a balanced FMA subset into a GTZAN-style directory layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fma-root",
        type=Path,
        required=True,
        help="Root directory of the fma_small audio files (contains 000/, 001/, …).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="Directory containing FMA metadata files (tracks.csv, not_found.pickle).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_data/fma_subset"),
        help="Destination root; genre subdirectories are created here.",
    )
    parser.add_argument(
        "--tracks-per-genre",
        type=int,
        default=200,
        help="Number of tracks to sample per mapped genre (5 genres → 1 000 total).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing any files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    fma_root: Path = args.fma_root.resolve()
    metadata_dir: Path = args.metadata_dir.resolve()
    output_root: Path = args.output_root.resolve()

    tracks_csv = metadata_dir / "tracks.csv"
    not_found_pkl = metadata_dir / "not_found.pickle"

    for path in (fma_root, tracks_csv, not_found_pkl):
        if not path.exists():
            print(f"[ERROR] Path not found: {path}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    try:
        import pandas as pd
    except ImportError:
        print(
            "[ERROR] pandas is required: pip install pandas",
            file=sys.stderr,
        )
        return 1

    print("Loading FMA metadata …")
    # tracks.csv uses a two-level header
    tracks = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)

    with open(not_found_pkl, "rb") as f:
        not_found = pickle.load(f)
    missing_audio: set[int] = {int(tid) for tid in not_found.get("audio", [])}
    print(f"  Tracks with missing audio (will be skipped): {len(missing_audio)}")

    # ------------------------------------------------------------------
    # Filter to fma_small subset and apply genre mapping
    # ------------------------------------------------------------------
    small = tracks[tracks["set", "subset"] == "small"].copy()
    small_genre = small["track", "genre_top"]

    # Collect candidate track IDs per mapped GTZAN genre
    candidates: dict[str, list[int]] = defaultdict(list)
    for track_id, fma_genre in small_genre.items():
        if fma_genre not in FMA_TO_GTZAN:
            continue
        if int(track_id) in missing_audio:
            continue
        audio_path = fma_audio_path(fma_root, int(track_id))
        if not audio_path.exists():
            continue
        gtzan_genre = FMA_TO_GTZAN[fma_genre]
        candidates[gtzan_genre].append(int(track_id))

    print()
    print("Candidate tracks per genre (after excluding missing/not-found):")
    for genre, ids in sorted(candidates.items()):
        print(f"  {genre:<10} {len(ids):>5} candidates")

    # ------------------------------------------------------------------
    # Stratified sample
    # ------------------------------------------------------------------
    import random
    rng = random.Random(args.seed)

    selected: dict[str, list[int]] = {}
    for genre, ids in candidates.items():
        if len(ids) < args.tracks_per_genre:
            print(
                f"[WARN] {genre}: only {len(ids)} candidates available "
                f"(requested {args.tracks_per_genre}); using all.",
                file=sys.stderr,
            )
            selected[genre] = ids
        else:
            selected[genre] = rng.sample(ids, args.tracks_per_genre)

    total = sum(len(ids) for ids in selected.values())
    print()
    print(f"Selected {total} tracks across {len(selected)} genres:")
    for genre, ids in sorted(selected.items()):
        print(f"  {genre:<10} {len(ids):>5}")

    # ------------------------------------------------------------------
    # Copy files
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] No files written. Pass without --dry-run to copy.")
        return 0

    print(f"\nWriting to {output_root} …")
    copied = 0
    skipped = 0

    for gtzan_genre, track_ids in sorted(selected.items()):
        genre_dir = output_root / gtzan_genre
        genre_dir.mkdir(parents=True, exist_ok=True)

        for track_id in track_ids:
            src = fma_audio_path(fma_root, track_id)
            dst = genre_dir / src.name
            if dst.exists():
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1

    print(f"  Copied : {copied}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"\nDone. Run the preprocessing pipeline with:")
    print(f"  python scripts/run_preprocess_smoke.py --dataset-root {output_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
