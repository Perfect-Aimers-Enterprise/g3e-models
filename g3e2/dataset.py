"""
G3E-2 dataset — turns prepare_g3e2.py's flat JSONL records into the
multimodal conversation format Qwen2.5-VL's `AutoProcessor` expects.

Per spec section 14: the image is supplied through the processor's proper
multimodal content block (`{"type": "image", "image": <PIL.Image>}`), never
as a filename string typed into the prompt text. This module is the one
place that distinction is enforced — prepare_g3e2.py deliberately stays
plain-JSON/no-torch so it's testable without a GPU; this file is where
`torch`/`transformers` actually get imported.

Target format (spec section 15): the assistant turn is REQUIRED to be
strict JSON matching `shared.schemas.G3E2Output` — no "Sure! Based on the
image..." preamble. `SYSTEM_PROMPT` below states this explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

SYSTEM_PROMPT = (
    "You are G3E-2, a visual security-reasoning system. You are given an "
    "image and a list of objects already detected in it by G3E-1 (class, "
    "confidence, bounding box in pixel [x1,y1,x2,y2] coordinates). "
    "Analyze the scene and respond with ONLY a single JSON object with "
    'exactly these keys: "state", "severity", "description", "reason", '
    '"recommended_action". Do not include any text before or after the '
    "JSON object."
)


def build_user_prompt(detections: list[dict]) -> str:
    detections_json = json.dumps(detections, indent=2)
    return (
        "Analyze the security status of this image using the detected "
        f"objects provided by G3E-1.\n\nDetected objects:\n{detections_json}"
    )


def build_target_json(target: dict) -> str:
    """The assistant's ground-truth response — must round-trip through
    json.loads() cleanly, since that's exactly what inference-time parsing
    will require of the model's real output too."""
    ordered = {
        "state": target["state"],
        "severity": target["severity"],
        "description": target["description"],
        "reason": target["reason"],
        "recommended_action": target["recommended_action"],
    }
    return json.dumps(ordered)


def record_to_messages(record: dict) -> list[dict]:
    """
    Builds the Qwen2.5-VL message list for one training record. The image
    is a real PIL.Image object in the content block — NOT a path string —
    so the processor actually encodes pixel data, matching spec section 14.
    """
    image = Image.open(record["image"]).convert("RGB")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_user_prompt(record["detections"])},
            ],
        },
        {"role": "assistant", "content": build_target_json(record["target"])},
    ]


class G3E2Dataset:
    """
    Thin, framework-agnostic wrapper around a prepare_g3e2.py JSONL file.
    Implements the plain Python sequence protocol (`__len__`/`__getitem__`)
    so it works as a `torch.utils.data.Dataset` without actually
    subclassing it here — avoids a hard torch import for anything that
    only needs to iterate records (tests, `scripts/validate_dataset.py`).

    Real training code wraps this with a torch Dataset subclass (or just
    subclasses this one directly, adding `torch` as a dependency at that
    point) that also calls the Qwen `AutoProcessor` in `__getitem__` to
    produce actual model inputs — that final tokenization step is left out
    of this file on purpose, since it depends on the exact processor
    version/API and is the one part of this pipeline that genuinely needs
    a real Qwen2.5-VL install to exercise.
    """

    def __init__(self, jsonl_path: str | Path):
        self.records: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        return {"record": record, "messages": record_to_messages(record)}
