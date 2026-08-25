import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_g3e2 import yolo_line_to_pixel_xyxy, build_sample, load_split_samples


def test_yolo_line_to_pixel_xyxy_center_box():
    result = yolo_line_to_pixel_xyxy("0 0.5 0.5 0.2 0.4", image_width=640, image_height=480)
    class_id, x1, y1, x2, y2 = result
    assert class_id == 0
    assert abs(x1 - (0.5 - 0.1) * 640) < 1e-6
    assert abs(x2 - (0.5 + 0.1) * 640) < 1e-6
    assert abs(y1 - (0.5 - 0.2) * 480) < 1e-6
    assert abs(y2 - (0.5 + 0.2) * 480) < 1e-6


def test_yolo_line_to_pixel_xyxy_rejects_malformed_line():
    assert yolo_line_to_pixel_xyxy("not a valid line", 640, 480) is None
    assert yolo_line_to_pixel_xyxy("0 0.5 0.5", 640, 480) is None


def test_build_sample_converts_labels_and_semantic(tmp_path):
    img_path = tmp_path / "img.jpg"
    Image.fromarray(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)).save(img_path)

    label_path = tmp_path / "img.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.4\n4 0.3 0.3 0.05 0.08\n")

    semantic_data = {
        "image": "img.jpg", "objects": [],
        "semantic": {"state": "potential_threat", "severity": "high", "description": "d", "reason": "r"},
    }

    class_names = {0: "person", 4: "knife"}
    sample = build_sample(img_path, label_path, semantic_data, class_names)

    assert sample["id"] == "img"
    assert len(sample["detections"]) == 2
    assert sample["detections"][0]["class"] == "person"
    assert sample["detections"][0]["confidence"] == 1.0
    assert sample["target"]["state"] == "potential_threat"
    assert sample["target"]["recommended_action"] == "alert_user"
    assert sample["image_width"] == 640
    assert sample["image_height"] == 480


def test_build_sample_handles_missing_label_file(tmp_path):
    img_path = tmp_path / "img.jpg"
    Image.fromarray(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)).save(img_path)
    label_path = tmp_path / "does_not_exist.txt"
    semantic_data = {"image": "img.jpg", "objects": [], "semantic": {
        "state": "normal", "severity": "none", "description": "d", "reason": "",
    }}
    sample = build_sample(img_path, label_path, semantic_data, {})
    assert sample["detections"] == []


def test_load_split_samples_skips_invalid_semantic(tmp_path):
    dataset_dir = tmp_path
    for sub in ("images/train", "labels/train", "semantic/train"):
        (dataset_dir / sub).mkdir(parents=True)

    Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(
        dataset_dir / "images/train/good.jpg"
    )
    (dataset_dir / "labels/train/good.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (dataset_dir / "semantic/train/good.json").write_text(json.dumps({
        "image": "good.jpg", "objects": [],
        "semantic": {"state": "normal", "severity": "none", "description": "d", "reason": ""},
    }))

    Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(
        dataset_dir / "images/train/bad.jpg"
    )
    (dataset_dir / "semantic/train/bad.json").write_text(json.dumps({
        "image": "bad.jpg", "objects": [],
        "semantic": {"state": "not_a_state", "severity": "none", "description": "d", "reason": ""},
    }))

    samples, skipped = load_split_samples(dataset_dir, "train", {0: "person"})
    assert len(samples) == 1
    assert skipped == 1
    assert samples[0]["id"] == "good"
