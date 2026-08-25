# g3e-models

Model training for **G3E-1** (visual detection, YOLO) and **G3E-2**
(visual semantic reasoning, Qwen2.5-VL + **LoRA** — not QLoRA), per the
G3E Model Development Specification. This README covers what's actually
implemented in this repo right now: **G3E-2's full data pipeline and training scaffold**, end to end.

> **Status check:** G3E-1 (YOLO) training scripts are not yet built here —
> use standard `ultralytics` CLI against `images/`+`labels/` in the
> interim (see "G3E-1 status" below). Everything in this README about
> G3E-2 is real, working code, tested where it can be without a GPU.

---

## 0. What you need to actually run this

| Stage                                                | Needs a GPU?      | Needs the real Qwen model downloaded? |
| ---------------------------------------------------- | ----------------- | ------------------------------------- |
| `scripts/download_dataset.py`                        | No                | No                                    |
| `scripts/validate_dataset.py`                        | No                | No                                    |
| `scripts/prepare_g3e2.py`                            | No                | No                                    |
| `g3e2/train.py --stage dataset_validation`           | No                | No                                    |
| `g3e2/train.py --stage sample_load_test`             | No                | No                                    |
| `g3e2/train.py --stage batch_forward_test` and later | Yes               | Yes                                   |
| `g3e2/predict.py`                                    | Yes (or slow CPU) | Yes, + a trained adapter              |

Qwen2.5-VL-3B-Instruct in bf16 needs roughly 7-9GB of GPU memory just to
load, before any training overhead (gradients, optimizer state,
activations for LoRA add a few more GB on top — a 16GB+ GPU is a
reasonable minimum for `per_device_train_batch_size: 1`). If you don't
have that, this is exactly the situation where QLoRA would normally come
up — this repo deliberately doesn't include it (see `g3e2/config.yaml`'s
comment on that); adding `BitsAndBytesConfig` + `load_in_4bit=True` back
into `g3e2/train.py`'s `_load_model_and_processor` is the change to make
if you decide you need it later.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch: install the CUDA build matching your machine from
# https://pytorch.org/get-started/locally/ if requirements.txt's default
# doesn't match your setup.

cp .env.example .env   # fill in HF_TOKEN if your dataset repo is private
```

Run the offline test suite (no GPU/model needed — covers balancing,
validation, schemas, augmentation, and the dataset loader):

```bash
PYTHONPATH=.:scripts pytest tests/ -q
```

## 2. Get the dataset

The dataset is g3e-data-engine's output, distributed as a ZIP on Hugging
Face (per spec section 1):

```bash
python scripts/download_dataset.py \
    --repo-id Godsave22/g3e-vision-dataset-v2-zip \
    --filename g3e-vision-dataset-v2.0.zip \
    --extract-to ./data/raw
```

This downloads, verifies it's a real zip (pass `--expected-sha256` if you
have a known hash to check against), extracts to `./data/raw`, and deletes
the downloaded zip afterward (pass `--keep-zip` to keep it). You should end
up with:

```
data/raw/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── semantic/{train,val,test}/
└── metadata/{classes.json, metadata.json, stats.json, versions.json}
```

Sanity-check it before doing anything else:

```bash
python scripts/validate_dataset.py --dataset-dir ./data/raw
```

This checks every `semantic/*.json` file for missing/empty required
fields, invalid `state`/`severity` values, and malformed JSON — read the
printed report before proceeding if it finds anything. It will also warn
you about repeated `description` text across many images (a known
side-effect of the rule-engine templates in g3e-data-engine) — that's not
fatal, but worth knowing before training on it.

## 3. Prepare G3E-2 training data (with class balancing)

```bash
python scripts/prepare_g3e2.py \
    --dataset-dir ./data/raw \
    --output-dir ./data/g3e2 \
    --balance-strategy capped \
    --max-multiplier 20
```

What this does:

1. Re-runs the same semantic validation as step 2, per split, writing
   `{split}_validation_report.json` — any file with a problem is skipped
   entirely (not trained on), not silently included.
2. Converts every valid sample's YOLO label file into pixel-space
   `[x1,y1,x2,y2]` detections — the shape G3E-1's real output will
   eventually have. These are ground-truth boxes standing in for G3E-1
   until it's trained; see "Swapping in real G3E-1 predictions" below.
3. Balances the TRAIN split only — val/test are left untouched at their
   real distribution, because balancing them would make your evaluation
   numbers lie about how good the model actually is at rare classes.
4. Writes `train.jsonl`, `val.jsonl`, `test.jsonl`, `balancing_plan.json`,
   and an `augmented_images/` folder holding the horizontal-flip copies
   created for oversampled classes.

### Balancing options — this is the part you asked about

Your actual data had `{'normal': 12693, 'hazard': 2073, 'caution': 1543,
'potential_threat': 68}` — a ~187:1 imbalance between the most common and
rarest class. Three ways to handle it:

```bash
# Default: oversample toward the majority, capped at 20x per class.
# potential_threat: 68 -> 1,360 (not 12,693 — see why below)
--balance-strategy capped --max-multiplier 20

# Exact 1:1 match — every class ends up at 12,693. Only do this if you
# have a lot of epochs' worth of patience for repeated data and you're
# not worried about memorization; NOT the default for a reason.
--balance-strategy match_majority

# Say the exact multiplier yourself for a specific class, everything
# else follows --balance-strategy as normal:
--explicit-multiplier potential_threat=50 --explicit-multiplier caution=10

# No balancing at all — train on the real, skewed distribution.
--balance-strategy none
```

Why not just exactly match the majority count? Duplicating 68 real photos
~187x means the model can see those exact same 68 images hundreds of
times across a few epochs — a strong risk that it memorizes those
specific photos rather than learning the general concept of
"potential_threat." The default `capped` strategy still gives
`potential_threat` far more representation than it had (68 to 1,360, ~20x)
without pushing it to the point of near-total repetition. Every duplicate
beyond the first real copy also gets horizontal-flipped (image + boxes)
rather than being byte-identical — see `shared/augmentation.py` — so it's
not training on literally the same tensor repeated dozens of times,
though this is a mitigation, not a fix; more real `potential_threat`
examples is the actual long-term answer.

Check `balancing_plan.json` after running to see exactly what happened
per class, and re-run with different flags if the numbers don't look
right — this step is idempotent and fast (no GPU involved).

### Swapping in real G3E-1 predictions

The spec's "One architectural decision" section recommends eventually
training G3E-2 on G3E-1's actual (imperfect) predictions rather than
ground truth, since that's what it will see in production. Once G3E-1 is
trained: run it over the same images, save its output in the same
`[{"class", "class_id", "confidence", "bbox"}, ...]` shape (see
`shared/schemas.py:G3E1Output`) per image, and modify
`prepare_g3e2.py:build_sample()` to read from those files instead of
converting the YOLO ground-truth labels. The rest of the pipeline
(balancing, JSONL format, `dataset.py`) doesn't need to change.

## 4. Train G3E-2 — staged, never a direct jump to full training

Per spec section 18, `g3e2/train.py` refuses to skip ahead. Run one stage
at a time and read the output before moving to the next:

```bash
# Cheap, no GPU/model needed — checks train/val/test.jsonl exist and load:
python g3e2/train.py --stage dataset_validation

# Still cheap — loads sample 0, checks the message structure is well-formed
# and the target parses as JSON:
python g3e2/train.py --stage sample_load_test

# From here on, needs the real model + a GPU:
python g3e2/train.py --stage batch_forward_test   # loads Qwen2.5-VL + attaches LoRA, one forward pass
python g3e2/train.py --stage batch_backward_test  # + one backward pass, checks loss is finite
python g3e2/train.py --stage lora_param_check     # confirms ONLY LoRA params are trainable
python g3e2/train.py --stage tiny_overfit_test    # trains on 8 samples for 50 steps — loss MUST drop
```

If `tiny_overfit_test` doesn't pass (loss doesn't drop below the
threshold in `g3e2/config.yaml`), stop — per spec section 18, do not
proceed to full training. Something is wrong with the data pipeline, the
LoRA config, or the model setup, and a longer run will not fix it.

`batch_forward_test` also prints every target module name it found on the
actual loaded model and errors out if any of `config.yaml`'s
`lora.target_modules` don't exist on it — this is deliberate (see the
config file's comment about not blindly trusting module names like
"merger").

`short_training_run` and `full_training` are not implemented yet in
`train.py` — the staged smoke tests above (through `tiny_overfit_test`)
are the part that's been built and is safe to run as-is. Once those pass,
wire up an actual training loop using the model/processor/LoRA setup in
`_load_model_and_processor()` as your starting point — either a manual
loop, `transformers.Trainer`, or `trl`'s `SFTTrainer` (which has built-in
vision-language + PEFT support and may save you writing your own data
collator). Whichever you pick, mask the system+user tokens out of the loss
(only the assistant's JSON should count) — an off-the-shelf collator that
doesn't do this by default will train the model to also predict your
prompt text, which is not what you want.

## 5. Running predictions — how to set up prediction data

Once you have a trained adapter (`./checkpoints/g3e2/<something>/` — a
PEFT adapter directory, not a full model checkpoint):

```bash
python g3e2/predict.py \
    --adapter-dir ./checkpoints/g3e2/final \
    --image ./samples/frame_001.jpg \
    --detections ./samples/frame_001_detections.json
```

### What prediction input actually looks like

**The image**: any JPG/PNG. It gets loaded, converted to RGB, and handed
to Qwen's processor — no manual resizing needed (letterboxing to 640 was
g3e-data-engine's training-data prep step; the processor handles arbitrary
input sizes at inference time), though matching roughly the size/aspect
ratio the model was trained on will generally give more reliable results.

**The detections file** (`--detections`, optional but strongly
recommended) — a JSON file matching G3E-1's output contract:

```json
[
  { "class": "person", "class_id": 0, "confidence": 0.98, "bbox": [120, 80, 420, 620] },
  { "class": "knife", "class_id": 4, "confidence": 0.91, "bbox": [300, 270, 355, 350] }
]
```

- `bbox` is pixel coordinates, `[x1, y1, x2, y2]` — matching the exact
  format `prepare_g3e2.py` trained on. Don't pass normalized 0-1
  coordinates or `[x,y,w,h]` — the model was never shown that format.
- `class` must be one of the frozen v1 class names (see
  `shared/schemas.py:G3E_CLASSES`) — an unfamiliar class name is
  out-of-distribution for the model.
- Where do these come from in practice? Either:
  1. You already have G3E-1 trained — run it on your image, save its
     output in this exact shape. This is the normal production path (see
     the final pipeline diagram in the spec, section 21).
  2. G3E-1 isn't trained yet — you can hand-write this file (e.g. from a
     quick manual annotation) to test G3E-2 in isolation, or point at
     ground-truth boxes the same way `prepare_g3e2.py` does during
     training data prep.

If you omit `--detections` entirely, G3E-2 reasons from the image alone.
`predict.py` prints a warning when you do this — the model was trained
expecting detections alongside the image (see
`g3e2/dataset.py:SYSTEM_PROMPT`), so skipping them puts it out of the
distribution it was trained on. Results in this mode should be treated as
unreliable until/unless you specifically train a detections-free variant.

### What you get back

`predict.py` prints (and you should capture) a JSON object matching
`shared/schemas.py:G3E2Output`:

```json
{
  "state": "potential_threat",
  "severity": "high",
  "description": "A person appears to be holding a knife.",
  "reason": "A weapon was detected in close proximity to a person.",
  "recommended_action": "alert_user"
}
```

If the model's raw output isn't valid JSON (rare once well-trained, more
likely early in training or with too few steps), `predict.py` raises a
`ValueError` showing the raw text rather than silently returning garbage —
that's a signal to keep training, not a bug to work around by loosening
the parser.

### Building a full G3E event (combining G3E-1 + G3E-2)

`shared/schemas.py:G3EEvent` is the final combined shape (spec section 22) — `inference/event_builder.py` (not yet built in this repo) is where
you'd assemble one from a G3E-1 output + a G3E-2 output + image paths +
a timestamp. Until that script exists, construct it directly:

```python
from shared.schemas import G3EEvent, EventImage, G3E1Output, G3E2Output
import datetime

event = G3EEvent(
    event_id="evt_001",
    timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    image=EventImage(original="frame.jpg", annotated="frame_annotated.jpg"),
    g3e1=G3E1Output(objects=detections),   # list[DetectedObject] or list[dict] with a "class" key
    g3e2=G3E2Output(**g3e2_prediction),
)
print(event.model_dump_json(by_alias=True, indent=2))  # by_alias=True is required — see below
```

Important: always pass `by_alias=True` when serializing anything
containing `DetectedObject` — its `class_name` field aliases to `"class"`
on the wire to match the spec's JSON shape, and `by_alias=True` is what
actually produces that key instead of `"class_name"`.

## 6. G3E-1 status

Not yet built in this repo. In the interim, train directly with
Ultralytics against the same `images/`+`labels/` folders:

```bash
pip install ultralytics
yolo detect train data=g3e1.yaml model=yolov8n.pt epochs=100 imgsz=640
```

where `g3e1.yaml` points at `./data/raw/images/{train,val}` and lists the
8 classes from `metadata/classes.json` in id order — do not let
Ultralytics infer class order from folder contents; it must match the
frozen v1 ontology exactly (see spec section 2).

## 7. Repository layout

```
g3e-models/
├── shared/
│   ├── schemas.py           # G3E1Output, G3E2Output, G3EEvent (pydantic)
│   ├── validation.py        # semantic annotation quality checks
│   ├── balancing.py         # class balancing plan + expansion (the oversampling logic)
│   ├── augmentation.py      # horizontal flip for image + boxes
│   └── recommended_action.py
├── g3e2/
│   ├── config.yaml          # LoRA config (no quantization), model id, data paths
│   ├── dataset.py           # JSONL -> Qwen2.5-VL conversation messages
│   ├── train.py             # staged smoke tests + LoRA attach (full_training loop: TODO)
│   └── predict.py           # inference: image + detections -> semantic JSON
├── scripts/
│   ├── download_dataset.py  # HF download -> verify -> extract -> cleanup
│   ├── validate_dataset.py  # standalone semantic validation CLI
│   └── prepare_g3e2.py      # the main data prep + balancing pipeline
└── tests/                    # offline, no GPU/model needed — 45 tests
```

## 8. Running tests

```bash
PYTHONPATH=.:scripts pytest tests/ -q
```

Covers: balancing math (including your exact real-world numbers as a
fixture), semantic validation, augmentation/flip correctness, schemas, the
dataset loader's message-building, and `prepare_g3e2.py`'s YOLO-to-pixel
conversion. Nothing here needs torch, a GPU, or network access — that's
deliberate, so the whole data pipeline can be verified correct before ever
touching the actual model.
