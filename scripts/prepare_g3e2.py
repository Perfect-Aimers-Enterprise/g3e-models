#!/usr/bin/env python3
"""
Prepares G3E-2 (semantic reasoning) training data from an extracted
g3e-vision-dataset (the output of g3e-data-engine — images/, labels/,
semantic/, metadata/).

Pipeline:
  1. Validate every split's semantic annotations (shared/validation.py) —
     write a report, skip anything malformed rather than silently
     training on it.
  2. Convert each valid sample's YOLO label file into pixel-space
     [x1,y1,x2,y2] detections (G3E-1's output CONTRACT shape — see
     shared/schemas.py) using confidence=1.0, since these are ground-truth
     boxes standing in for G3E-1 until G3E-1 is actually trained. Swap
     this stage for real G3E-1 inference output once available — see the
     spec's "One architectural decision" section.
  3. Balance the TRAIN split only (shared/balancing.py) — val/test are
     left at their real, unmodified distribution.
  4. Write train.jsonl / val.jsonl / test.jsonl in the flat intermediate
     format g3e2/dataset.py consumes (NOT the final Qwen conversation
     format — that conversion happens at load time in dataset.py, so this
     script stays testable without needing torch/transformers/Qwen at all).

Usage:
    python scripts/prepare_g3e2.py \\
        --dataset-dir /path/to/extracted/g3e-vision-dataset \\
        --output-dir ./data/g3e2 \\
        --balance-strategy capped --max-multiplier 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `from shared...` when run directly

from shared.validation import validate_semantic_directory, validate_semantic_file
from shared.balancing import compute_balancing_plan, expand_samples
from shared.augmentation import flip_image, flip_bbox_xyxy
from shared.recommended_action import derive_recommended_action

SPLITS = ("train", "val", "test")


def yolo_line_to_pixel_xyxy(line: str, image_width: int, image_height: int) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return None

    x1 = (cx - w / 2.0) * image_width
    y1 = (cy - h / 2.0) * image_height
    x2 = (cx + w / 2.0) * image_width
    y2 = (cy + h / 2.0) * image_height
    return class_id, x1, y1, x2, y2


def build_sample(image_path: Path, label_path: Path, semantic_data: dict, class_names: dict[int, str]) -> dict:
    from PIL import Image

    with Image.open(image_path) as img:
        width, height = img.size

    detections = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parsed = yolo_line_to_pixel_xyxy(line, width, height)
            if parsed is None:
                continue
            class_id, x1, y1, x2, y2 = parsed
            detections.append({
                "class": class_names.get(class_id, f"unknown_{class_id}"),
                "class_id": class_id,
                "confidence": 1.0,  # ground truth stand-in for G3E-1 — see module docstring
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            })

    semantic = semantic_data["semantic"]
    target = {
        "state": semantic["state"],
        "severity": semantic["severity"],
        "description": semantic["description"],
        "reason": semantic.get("reason", ""),
        "recommended_action": derive_recommended_action(semantic["state"], semantic["severity"]),
    }

    return {
        "id": image_path.stem,
        "image": str(image_path),
        "image_width": width,
        "image_height": height,
        "detections": detections,
        "target": target,
        "augmented": False,
    }


def load_split_samples(dataset_dir: Path, split: str, class_names: dict[int, str]) -> tuple[list[dict], int]:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    semantic_dir = dataset_dir / "semantic" / split

    samples = []
    skipped = 0
    for image_path in sorted(images_dir.glob("*.jpg")):
        semantic_path = semantic_dir / f"{image_path.stem}.json"
        semantic_data, problems = validate_semantic_file(semantic_path)
        if semantic_data is None or problems:
            skipped += 1
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        samples.append(build_sample(image_path, label_path, semantic_data, class_names))

    return samples, skipped


def apply_augmented_duplicates(expanded: list[tuple[dict, bool]], augmented_images_dir: Path) -> list[dict]:
    """
    Materializes the (sample, is_duplicate) pairs from expand_samples()
    into final sample records — duplicates get a flipped copy of the image
    + flipped detection boxes (see shared/augmentation.py), written under
    `augmented_images_dir` rather than overwriting anything in the
    original dataset.
    """
    out = []
    dup_counter: dict[str, int] = {}

    for sample, is_duplicate in expanded:
        if not is_duplicate:
            out.append(sample)
            continue

        dup_counter[sample["id"]] = dup_counter.get(sample["id"], 0) + 1
        n = dup_counter[sample["id"]]
        aug_id = f"{sample['id']}_flip{n}"
        aug_path = augmented_images_dir / f"{aug_id}.jpg"
        flip_image(sample["image"], aug_path)

        flipped_detections = [
            {**d, "bbox": flip_bbox_xyxy(d["bbox"], sample["image_width"])} for d in sample["detections"]
        ]
        out.append({
            **sample,
            "id": aug_id,
            "image": str(aug_path),
            "detections": flipped_detections,
            "augmented": True,
        })

    return out


def write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Path to the EXTRACTED g3e-vision-dataset folder")
    parser.add_argument("--output-dir", required=True, help="Where to write train/val/test.jsonl")
    parser.add_argument("--balance-strategy", default="capped", choices=["none", "capped", "match_majority"])
    parser.add_argument("--max-multiplier", type=float, default=20.0)
    parser.add_argument(
        "--explicit-multiplier", action="append", default=[],
        help="class=multiplier, repeatable, e.g. --explicit-multiplier potential_threat=50",
    )
    args = parser.parse_args()

    explicit_multipliers = {}
    for pair in args.explicit_multiplier:
        cls, _, val = pair.partition("=")
        explicit_multipliers[cls] = float(val)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    classes_path = dataset_dir / "metadata" / "classes.json"
    classes_raw = json.loads(classes_path.read_text())
    if isinstance(classes_raw, dict) and "classes" in classes_raw:
        class_names = {c["id"]: c["name"] for c in classes_raw["classes"]}
    else:
        class_names = {int(k): v for k, v in classes_raw.items()}

    print("G3E-2 DATA PREPARATION\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        semantic_dir = dataset_dir / "semantic" / split
        if not semantic_dir.exists():
            continue
        report = validate_semantic_directory(semantic_dir)
        print(f"[{split}] " + report.render().replace("\n", "\n  "))
        (output_dir / f"{split}_validation_report.json").write_text(json.dumps(report.to_dict(), indent=2))
        print()

    all_split_samples: dict[str, list[dict]] = {}
    for split in SPLITS:
        images_dir = dataset_dir / "images" / split
        if not images_dir.exists():
            continue
        samples, skipped = load_split_samples(dataset_dir, split, class_names)
        print(f"[{split}] loaded {len(samples)} valid sample(s), skipped {skipped} invalid")
        all_split_samples[split] = samples

    train_samples = all_split_samples.get("train", [])
    class_counts: dict[str, int] = {}
    for s in train_samples:
        class_counts[s["target"]["state"]] = class_counts.get(s["target"]["state"], 0) + 1

    plan = compute_balancing_plan(
        class_counts, strategy=args.balance_strategy, max_multiplier=args.max_multiplier,
        explicit_multipliers=explicit_multipliers,
    )
    print("\n" + plan.render())
    (output_dir / "balancing_plan.json").write_text(json.dumps(plan.to_dict(), indent=2))

    expanded = expand_samples(train_samples, lambda s: s["target"]["state"], plan)
    augmented_dir = output_dir / "augmented_images"
    final_train_samples = apply_augmented_duplicates(expanded, augmented_dir)
    print(f"\n[train] final count after balancing: {len(final_train_samples)} "
          f"({sum(1 for s in final_train_samples if s['augmented'])} augmented)")
    write_jsonl(final_train_samples, output_dir / "train.jsonl")

    for split in ("val", "test"):
        if split in all_split_samples:
            write_jsonl(all_split_samples[split], output_dir / f"{split}.jsonl")
            print(f"[{split}] wrote {len(all_split_samples[split])} sample(s) — UNBALANCED, real distribution")

    print(f"\nDone. Output written to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
