#!/usr/bin/env python3
"""
G3E-2 LoRA fine-tuning of Qwen2.5-VL.

LoRA ONLY — this script never imports BitsAndBytesConfig and never sets
load_in_4bit/load_in_8bit. If you need QLoRA later because of GPU memory,
that's a deliberate addition to make here, not a default to fall back to.

Per spec section 18, this NEVER jumps straight to a multi-hour training
run. `main()` walks through every stage in `config.yaml`'s `training.stages`
list, in order, and refuses to proceed to the next one if the current one
fails:

    dataset_validation -> sample_load_test -> batch_forward_test
    -> batch_backward_test -> lora_param_check -> tiny_overfit_test
    -> short_training_run -> full_training

Run with `--stage <name>` to run only up to (and including) one stage —
useful for iterating on the smoke tests without waiting through earlier
ones each time.

REQUIRES: torch, transformers, peft, and enough GPU memory to load
Qwen2.5-VL-3B-Instruct in bf16 (~6-7GB for the base model alone, more for
activations/optimizer state). This script cannot be smoke-tested without
those installed and a real GPU — see README.md "What you need to actually
run this" for exact requirements and a memory-budget breakdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from g3e2.dataset import G3E2Dataset

STAGE_ORDER = [
    "dataset_validation",
    "sample_load_test",
    "batch_forward_test",
    "batch_backward_test",
    "lora_param_check",
    "tiny_overfit_test",
    "short_training_run",
    "full_training",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_dataset_validation(cfg: dict) -> None:
    print("=== STAGE: dataset_validation ===")
    for split in ("train", "val", "test"):
        jsonl_path = Path(cfg["data"][f"{split}_jsonl"])
        if not jsonl_path.exists():
            raise RuntimeError(
                f"{jsonl_path} does not exist — run scripts/prepare_g3e2.py first."
            )
        ds = G3E2Dataset(jsonl_path)
        if len(ds) == 0:
            raise RuntimeError(f"{jsonl_path} has zero samples.")
        print(f"  {split}: {len(ds)} sample(s) — OK")
    print("  PASSED\n")


def stage_sample_load_test(cfg: dict) -> None:
    print("=== STAGE: sample_load_test ===")
    ds = G3E2Dataset(cfg["data"]["train_jsonl"])
    item = ds[0]
    messages = item["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert any(block["type"] == "image" for block in messages[1]["content"])
    assert messages[2]["role"] == "assistant"
    json.loads(messages[2]["content"])  # must be valid JSON
    print(f"  loaded sample 0 ({item['record']['id']}) — messages well-formed, target parses as JSON")
    print("  PASSED\n")


def _load_model_and_processor(cfg: dict):
    """
    Isolated so the earlier, cheap stages (dataset_validation,
    sample_load_test) can run and fail fast WITHOUT needing torch/
    transformers/a GPU/network access to Hugging Face at all — only the
    stages from batch_forward_test onward actually need the real model.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    processor = AutoProcessor.from_pretrained(model_cfg["base_model"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["base_model"], torch_dtype=dtype, device_map="auto"
    )

    # Print actual module names before attaching LoRA — the config's
    # target_modules must be verified against THIS list, not assumed.
    # See config.yaml's comment on not blindly trusting "merger" etc.
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [m for m in lora_cfg["target_modules"] if m not in module_names]
    if missing:
        raise RuntimeError(
            f"config target_modules {missing} not found among this model's actual module "
            f"names. Inspect `model.named_modules()` and fix g3e2/config.yaml before proceeding."
        )

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, processor


def stage_batch_forward_test(cfg: dict):
    print("=== STAGE: batch_forward_test ===")
    model, processor = _load_model_and_processor(cfg)
    ds = G3E2Dataset(cfg["data"]["train_jsonl"])
    batch = [ds[i]["messages"] for i in range(min(2, len(ds)))]

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in batch]
    images = [[block["image"] for block in m[1]["content"] if block["type"] == "image"][0] for m in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)

    outputs = model(**inputs)
    print(f"  forward pass OK — logits shape: {tuple(outputs.logits.shape)}")
    print("  PASSED\n")
    return model, processor, inputs


def stage_batch_backward_test(cfg: dict):
    print("=== STAGE: batch_backward_test ===")
    model, processor, inputs = stage_batch_forward_test(cfg)
    inputs["labels"] = inputs["input_ids"].clone()
    outputs = model(**inputs)
    outputs.loss.backward()
    print(f"  backward pass OK — loss: {outputs.loss.item():.4f}")
    print("  PASSED\n")
    return model, processor


def stage_lora_param_check(cfg: dict):
    print("=== STAGE: lora_param_check ===")
    model, processor = stage_batch_backward_test(cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora_trainable = [n for n in trainable if "lora_" not in n]
    if non_lora_trainable:
        raise RuntimeError(
            f"Found {len(non_lora_trainable)} trainable parameter(s) outside LoRA adapters "
            f"(e.g. {non_lora_trainable[:3]}) — the base model should be fully frozen."
        )
    print(f"  {len(trainable)} trainable parameter(s), all LoRA — base model correctly frozen")
    print("  PASSED\n")
    return model, processor


def stage_tiny_overfit_test(cfg: dict):
    print("=== STAGE: tiny_overfit_test ===")
    import torch

    model, processor = stage_lora_param_check(cfg)
    overfit_cfg = cfg["training"]["tiny_overfit_test"]
    ds = G3E2Dataset(cfg["data"]["train_jsonl"])
    n = min(overfit_cfg["num_samples"], len(ds))
    samples = [ds[i]["messages"] for i in range(n)]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    for step in range(overfit_cfg["num_steps"]):
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in samples]
        images = [[b["image"] for b in m[1]["content"] if b["type"] == "image"][0] for m in samples]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
        inputs["labels"] = inputs["input_ids"].clone()

        outputs = model(**inputs)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(outputs.loss.item())
        if step % 10 == 0:
            print(f"    step {step}: loss={outputs.loss.item():.4f}")

    threshold = overfit_cfg["loss_should_drop_below"]
    if losses[-1] >= threshold:
        raise RuntimeError(
            f"Tiny overfit test FAILED — final loss {losses[-1]:.4f} did not drop below "
            f"{threshold}. Do not proceed to full training; something is wrong with the "
            "data pipeline, model setup, or LoRA config. See spec section 18."
        )
    print(f"  final loss {losses[-1]:.4f} < {threshold} — model can learn from this data")
    print("  PASSED\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--stage", default="full_training", choices=STAGE_ORDER,
                         help="Run stages up to and including this one, then stop.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stages_to_run = STAGE_ORDER[: STAGE_ORDER.index(args.stage) + 1]

    stage_fns = {
        "dataset_validation": stage_dataset_validation,
        "sample_load_test": stage_sample_load_test,
        "batch_forward_test": stage_batch_forward_test,
        "batch_backward_test": stage_batch_backward_test,
        "lora_param_check": stage_lora_param_check,
        "tiny_overfit_test": stage_tiny_overfit_test,
    }

    for stage in stages_to_run:
        if stage in stage_fns:
            stage_fns[stage](cfg)
        elif stage == "short_training_run":
            print("=== STAGE: short_training_run ===")
            print("  Not yet implemented in this script — run a short Trainer-based loop "
                  "manually with num_train_epochs reduced, using the model/processor setup "
                  "above as a template, before running full_training.\n")
        elif stage == "full_training":
            print("=== STAGE: full_training ===")
            print("  Not yet implemented in this script — wire up transformers.Trainer (or "
                  "a manual loop) here using g3e2/config.yaml's `training` block once "
                  "short_training_run has been validated by hand. See README.md.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
