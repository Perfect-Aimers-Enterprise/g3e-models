"""
Horizontal-flip augmentation for oversampled training examples.

Applied ONLY to duplicate copies produced by shared/balancing.py's
expand_samples() — the first (real) occurrence of every image is always
left untouched. This exists purely so that duplicated copies of a rare
class aren't byte-identical to each other, reducing (not eliminating —
nothing but more real data fully eliminates this) the risk of a
fine-tuned model memorizing a handful of source photos.

Flipping is a defensible augmentation for this dataset specifically:
G3E images come from fixed/static cameras and general photography, and
mirroring left-right doesn't change whether "a person is holding a knife"
or "there is fire in frame" — the semantic label stays valid. This would
NOT be a safe augmentation for a dataset where left/right orientation is
meaningful (e.g. reading direction, traffic-lane side).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from shared.schemas import DetectedObject


def flip_image(src_path: str | Path, dest_path: str | Path) -> None:
    img = Image.open(src_path)
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    flipped.save(dest_path, format="JPEG", quality=95)


def flip_bbox_xyxy(bbox: list[float], image_width: float) -> list[float]:
    """Mirrors a [x1, y1, x2, y2] pixel or normalized box horizontally."""
    x1, y1, x2, y2 = bbox
    return [image_width - x2, y1, image_width - x1, y2]


def flip_detected_objects(objects: list[DetectedObject], image_width: float) -> list[DetectedObject]:
    return [
        DetectedObject(
            **{
                "class": obj.class_name,
                "class_id": obj.class_id,
                "confidence": obj.confidence,
                "bbox": flip_bbox_xyxy(obj.bbox, image_width),
            }
        )
        for obj in objects
    ]
