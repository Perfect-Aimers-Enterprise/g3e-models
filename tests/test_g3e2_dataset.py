import json

import numpy as np
from PIL import Image

from g3e2.dataset import G3E2Dataset, record_to_messages, build_user_prompt, build_target_json, SYSTEM_PROMPT


def _make_record(tmp_path, state="normal"):
    img_path = tmp_path / "sample.jpg"
    Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(img_path)
    return {
        "id": "sample",
        "image": str(img_path),
        "image_width": 64,
        "image_height": 64,
        "detections": [{"class": "person", "class_id": 0, "confidence": 1.0, "bbox": [1, 2, 3, 4]}],
        "target": {
            "state": state, "severity": "none", "description": "x", "reason": "",
            "recommended_action": "none",
        },
        "augmented": False,
    }


def test_build_user_prompt_embeds_detections_json():
    prompt = build_user_prompt([{"class": "gun", "class_id": 2, "confidence": 0.9, "bbox": [1, 2, 3, 4]}])
    assert "gun" in prompt
    assert "G3E-1" in prompt


def test_build_target_json_is_valid_json_with_expected_keys():
    target = {"state": "hazard", "severity": "high", "description": "d", "reason": "r", "recommended_action": "alert_user"}
    parsed = json.loads(build_target_json(target))
    assert set(parsed.keys()) == {"state", "severity", "description", "reason", "recommended_action"}


def test_record_to_messages_structure(tmp_path):
    record = _make_record(tmp_path)
    messages = record_to_messages(record)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT

    assert messages[1]["role"] == "user"
    content_types = [b["type"] for b in messages[1]["content"]]
    assert content_types == ["image", "text"]
    assert isinstance(messages[1]["content"][0]["image"], Image.Image)

    assert messages[2]["role"] == "assistant"
    json.loads(messages[2]["content"])


def test_g3e2_dataset_loads_jsonl(tmp_path):
    record = _make_record(tmp_path)
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text(json.dumps(record) + "\n")

    ds = G3E2Dataset(jsonl_path)
    assert len(ds) == 1
    item = ds[0]
    assert item["record"]["id"] == "sample"
    assert len(item["messages"]) == 3


def test_g3e2_dataset_handles_multiple_lines(tmp_path):
    lines = [json.dumps(_make_record(tmp_path, state=s)) for s in ("normal", "hazard", "caution")]
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text("\n".join(lines) + "\n")

    ds = G3E2Dataset(jsonl_path)
    assert len(ds) == 3


def test_g3e2_dataset_skips_blank_lines(tmp_path):
    record = _make_record(tmp_path)
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text(json.dumps(record) + "\n\n\n")
    ds = G3E2Dataset(jsonl_path)
    assert len(ds) == 1
