#!/usr/bin/env python3
"""
Run G3E-2 inference: image + G3E-1 detections -> semantic JSON.

REQUIRES a trained LoRA adapter (g3e2/train.py's full_training stage,
once implemented and run) and the same base model used to train it.

Usage — with detections you already have (e.g. from a G3E-1 run, or
ground-truth boxes while G3E-1 doesn't exist yet):

    python g3e2/predict.py \\
        --adapter-dir ./checkpoints/g3e2/final \\
        --image ./samples/frame_001.jpg \\
        --detections ./samples/frame_001_detections.json

Usage — no detections available (G3E-2 reasons from the image alone; this
is a degraded mode — see README.md "Running predictions" for why
detections matter):

    python g3e2/predict.py \\
        --adapter-dir ./checkpoints/g3e2/final \\
        --image ./samples/frame_001.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g3e2.dataset import SYSTEM_PROMPT, build_user_prompt


def load_detections(path: str | None) -> list[dict]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_inference_messages(image_path: str, detections: list[dict]) -> list[dict]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_user_prompt(detections)},
            ],
        },
    ]


def parse_model_output(raw_text: str) -> dict:
    """
    The model is trained to output ONLY a JSON object (see
    g3e2/dataset.py SYSTEM_PROMPT). Real models occasionally still wrap it
    in stray whitespace/newlines or (rarely, if undertrained) a code
    fence — this strips the common cases before giving up, rather than
    silently returning garbage to the caller.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model output was not valid JSON after cleanup: {exc}\nRaw output: {raw_text!r}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-dir", required=True, help="Path to the trained LoRA adapter directory")
    parser.add_argument("--image", required=True)
    parser.add_argument("--detections", default=None, help="Path to a G3E-1-style detections JSON file")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.base_model)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    detections = load_detections(args.detections)
    if not detections:
        print(
            "[warning] no --detections provided — G3E-2 is reasoning from the image alone. "
            "It was trained expecting G3E-1's detections alongside the image; results without "
            "them are out of distribution. See README.md 'Running predictions'.",
            file=sys.stderr,
        )

    messages = build_inference_messages(args.image, detections)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = [block["image"] for block in messages[1]["content"] if block["type"] == "image"][0]
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(generated, skip_special_tokens=True)[0]

    result = parse_model_output(raw_text)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
