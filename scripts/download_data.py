"""Download dataset npz files from Google Drive.

How to use:
    python scripts/download_data.py

Files land in:
    src/music_classifier/model/dataset/        # train.npz, val.npz, test.npz

Already-downloaded files are skipped. Re-run anytime to fetch missing pieces.

Setup (if file IDs are empty or new files are needed):
    1. In Drive, right-click each file -> Share -> "Anyone with the link" -> Viewer
    2. Copy the link.
    3. The file ID is the chunk between "/d/" and "/view".
    4. Paste each ID into the FILES dict below.
"""
from pathlib import Path
import sys

try:
    import gdown
except ImportError:
    sys.exit("Missing dependency: run `pip install gdown` (or `pip install -r requirements.txt`).")


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "src" / "music_classifier" / "model" / "dataset"

# (filename, destination_dir, file_id)
FILES = [
    ("train.npz", DATASET_DIR, "14ZA3lGqK5iC0XtSpOV8cAbFMkN0-gNgw"),
    ("val.npz",   DATASET_DIR, "1XKUEMq8dG3SkaqpGz-cT_PBRBHHvy1l7"),
    ("test.npz",  DATASET_DIR, "1Df6ZTXHo0ur-s6-hugrpcFTaLjFuycvR"),
]


def download_one(name: str, dest_dir: Path, file_id: str) -> bool:
    """Download a single file. Returns True on success/already-present, False on failure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / name

    if out.exists():
        print(f"  [skip] {name} already present at {out.relative_to(REPO_ROOT)}")
        return True

    if file_id.startswith("PASTE_"):
        print(f"  [skip] {name} — file ID not configured in scripts/download_data.py")
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  [get ] {name} -> {out.relative_to(REPO_ROOT)}")
    try:
        gdown.download(url, str(out), quiet=False)
    except Exception as e:
        print(f"  [fail] {name}: {e}")
        if out.exists():
            out.unlink()  # don't leave a partial file
        return False
    return True


def main() -> int:
    print(f"Repo root: {REPO_ROOT}\n")
    failures = []
    for name, dest, fid in FILES:
        if not download_one(name, dest, fid):
            failures.append(name)

    print()
    if failures:
        print(f"⚠  {len(failures)} file(s) not downloaded: {', '.join(failures)}")
        print("   Set the file IDs in scripts/download_data.py and re-run.")
        return 1
    print("✓ All files in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
