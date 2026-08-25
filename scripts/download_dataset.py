#!/usr/bin/env python3
"""
Downloads the g3e-vision-dataset ZIP from Hugging Face, verifies it,
extracts it to working storage, and (optionally) deletes the ZIP
afterward — per spec section 1's required lifecycle:

    1. Download dataset
    2. Verify integrity
    3. Extract to working storage
    4. Read dataset       <- prepare_g3e2.py does this part
    5. Train              <- g3e2/train.py does this part
    6. Delete temporary data when appropriate

Usage:
    python scripts/download_dataset.py \\
        --repo-id Godsave22/g3e-vision-dataset-v2-zip \\
        --filename g3e-vision-dataset-v2.0.zip \\
        --extract-to ./data/raw

Requires HF_TOKEN if the dataset repo is private — same credential
convention as g3e-data-engine (env var, or a .env file in the repo root).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path


def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="e.g. Godsave22/g3e-vision-dataset-v2-zip")
    parser.add_argument("--filename", required=True, help="e.g. g3e-vision-dataset-v2.0.zip")
    parser.add_argument("--extract-to", required=True)
    parser.add_argument("--expected-sha256", default=None, help="Optional — verify integrity against a known hash")
    parser.add_argument("--keep-zip", action="store_true", help="Don't delete the downloaded ZIP after extracting")
    args = parser.parse_args()

    _load_dotenv_once()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Missing dependency: pip install huggingface_hub", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN")

    print(f"1. Downloading {args.filename} from {args.repo_id} ...")
    zip_path = Path(
        hf_hub_download(repo_id=args.repo_id, filename=args.filename, repo_type="dataset", token=token)
    )
    print(f"   -> {zip_path}")

    print("2. Verifying integrity ...")
    if not zipfile.is_zipfile(zip_path):
        print("   FAILED — downloaded file is not a valid zip archive.", file=sys.stderr)
        return 1
    if args.expected_sha256:
        actual = sha256_of(zip_path)
        if actual != args.expected_sha256:
            print(f"   FAILED — sha256 mismatch (expected {args.expected_sha256}, got {actual})", file=sys.stderr)
            return 1
        print("   sha256 OK")
    else:
        print("   zip structure OK (no --expected-sha256 given, skipped hash check)")

    extract_to = Path(args.extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"3. Extracting to {extract_to} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    print("   done")

    if not args.keep_zip:
        print("4. Removing downloaded zip (working storage keeps only the extracted files) ...")
        try:
            zip_path.unlink()
        except OSError as exc:
            print(f"   (non-fatal) could not remove {zip_path}: {exc}")

    print(f"\nDataset ready at: {extract_to}")
    print("Next: python scripts/validate_dataset.py --dataset-dir " + str(extract_to))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
