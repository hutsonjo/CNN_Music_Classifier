"""Download raw training audio from S3.

S3 bucket:  cnn-music-classifier-data
Prefix:     training_data/

How to use:
    python scripts/download_training_data.py

Files land in:
    training_data/     (repo root)

Already-present files are skipped. Re-run anytime to fetch missing pieces.

Setup (one-time, per machine):
    1. Install the AWS CLI: brew install awscli
    2. Create an IAM user with the cnn-music-classifier-s3-policy attached.
       Ask the project lead for credentials.
    3. Run: aws configure
       Enter the access key, secret key, region (us-east-1), and output (json).

Corrupted files excluded from sync:
    - 098565.mp3
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUCKET = "cnn-music-classifier-data"
S3_PREFIX = "training_data/"
LOCAL_DIR = REPO_ROOT / "training_data"

EXCLUDED = [
    "098565.mp3",
]


def build_exclude_flags() -> list[str]:
    flags = []
    for filename in EXCLUDED:
        flags += ["--exclude", f"*{filename}"]
    return flags


def main() -> int:
    LOCAL_DIR.mkdir(exist_ok=True)

    cmd = [
        "aws", "s3", "sync",
        f"s3://{BUCKET}/{S3_PREFIX}",
        str(LOCAL_DIR),
        "--exclude", ".*",
    ] + build_exclude_flags()

    print(f"Syncing s3://{BUCKET}/{S3_PREFIX} -> {LOCAL_DIR.relative_to(REPO_ROOT)}/")
    print("Skipping:", ", ".join(EXCLUDED))
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[fail] Sync failed. Make sure the AWS CLI is configured: run `aws configure`")
        return 1

    print("\n✓ Training data in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
