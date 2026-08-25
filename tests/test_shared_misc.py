import json

import numpy as np
from PIL import Image

from shared.augmentation import flip_bbox_xyxy, flip_image
from shared.schemas import DetectedObject, G3E1Output, G3E2Output, G3EEvent, EventImage
from shared.recommended_action import derive_recommended_action


def test_flip_bbox_mirrors_correctly():
    assert flip_bbox_xyxy([10, 20, 60, 80], image_width=640) == [580, 20, 630, 80]


def test_flip_bbox_preserves_width_and_height():
    box = [100, 50, 200, 300]
    flipped = flip_bbox_xyxy(box, image_width=640)
    assert flipped[2] - flipped[0] == box[2] - box[0]
    assert flipped[3] - flipped[1] == box[3] - box[1]


def test_flip_bbox_is_its_own_inverse():
    box = [10, 20, 60, 80]
    once = flip_bbox_xyxy(box, image_width=640)
    twice = flip_bbox_xyxy(once, image_width=640)
    assert twice == box


def test_flip_image_writes_correct_size(tmp_path):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    Image.fromarray(np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)).save(src)
    flip_image(src, dst)
    out = Image.open(dst)
    assert out.size == (200, 100)


def test_flip_image_actually_flips_pixels(tmp_path):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:, 0, :] = 255
    Image.fromarray(arr).save(src)
    flip_image(src, dst)
    out = np.array(Image.open(dst))
    assert out[:, -1, :].mean() > out[:, 0, :].mean()


def test_detected_object_accepts_class_alias():
    obj = DetectedObject(**{"class": "person", "class_id": 0, "confidence": 0.9, "bbox": [1, 2, 3, 4]})
    assert obj.class_name == "person"


def test_g3e1_output_defaults_to_empty_objects():
    out = G3E1Output()
    assert out.objects == []


def test_g3e2_output_requires_core_fields():
    out = G3E2Output(state="normal", severity="none", description="x", reason="")
    assert out.recommended_action == ""


def test_event_schema_round_trips():
    """
    NOTE: DetectedObject.class_name aliases to "class" on the wire (see
    shared/schemas.py) — any real serialization of these schemas MUST pass
    by_alias=True or the output will say "class_name" instead of "class",
    breaking the spec's documented event JSON shape. inference/event_builder.py
    (once built) must do this.
    """
    event = G3EEvent(
        event_id="evt_1",
        timestamp="2026-01-01T00:00:00Z",
        image=EventImage(original="a.jpg"),
        g3e1=G3E1Output(objects=[DetectedObject(**{"class": "fire", "class_id": 1, "confidence": 0.8, "bbox": [0, 0, 1, 1]})]),
        g3e2=G3E2Output(state="hazard", severity="high", description="fire", reason="r"),
    )
    d = json.loads(event.model_dump_json(by_alias=True))
    assert d["g3e1"]["objects"][0]["class"] == "fire"
    assert d["g3e2"]["state"] == "hazard"


def test_derive_recommended_action_known_combo():
    assert derive_recommended_action("potential_threat", "critical") == "alert_immediately"


def test_derive_recommended_action_normal_case():
    assert derive_recommended_action("normal", "none") == "none"


def test_derive_recommended_action_falls_back_for_unknown_combo():
    assert derive_recommended_action("normal", "critical") == "review"
