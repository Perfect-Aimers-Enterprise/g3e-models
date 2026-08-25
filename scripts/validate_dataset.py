#!/usr/bin/env python3
"""
Standalone dataset validation — runs the same semantic checks
prepare_g3e2.py runs internally, without producing any training output.
Use this to sanity-check a freshly extracted g3e-vision-dataset before
committing to a full prepare_g3e2.py run.

    python scripts/validate_dataset.py --dataset-dir /path/to/g3e-vision-dataset

Exit code 0 = every split's semantic annotations are clean. Exit code 1 =
at least one issue found (see the printed report for exactly which files).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.validation import validate_semantic_directory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    all_ok = True

    for split in ("train", "val", "test"):
        semantic_dir = dataset_dir / "semantic" / split
        if not semantic_dir.exists():
            continue
        report = validate_semantic_directory(semantic_dir)
        print(f"[{split}]")
        print(report.render())
        print()
        all_ok = all_ok and report.ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
