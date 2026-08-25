"""
Class balancing for G3E-2 training data.

The real problem this solves: a rule-engine-labeled dataset skews heavily
toward "normal" (nothing notable happened in most frames) — e.g. a real
G3E run producing {'normal': 12693, 'hazard': 2073, 'caution': 1543,
'potential_threat': 68}. Trained as-is, a model can get excellent loss by
just always predicting "normal" and barely ever learning what a real
potential_threat looks like.

Design decisions (read before changing the defaults):

1. **Only the TRAIN split is ever balanced.** val/test must stay at the
   real, unmodified distribution — that's what makes evaluation numbers
   mean anything. Balancing test data would make the model look better at
   detecting rare threats than it actually is.

2. **Oversampling, not undersampling.** Downsampling "normal" down to 68
   examples to match "potential_threat" would throw away ~99% of the
   dataset — oversampling the rare classes instead keeps everything.

3. **Capped, not exact-matched, by default.** Duplicating 68 images
   ~187x to exactly match 12,693 "normal" examples means the model can see
   the same 68 photos hundreds of times over a few epochs — a strong
   memorization risk for a vision-language model, which can learn "this
   exact image => potential_threat" rather than the general concept. The
   default strategy (`capped`) oversamples toward the majority count but
   never past `max_multiplier` copies of any single source image (default
   20x) — tune this per your actual epoch count and how much you trust the
   rule-engine labels for that class.

4. **Duplicates get a cheap augmentation, not a byte-identical copy.**
   Every oversampled copy beyond the first is horizontal-flipped (image +
   bboxes) — trivial to justify for static camera scenes (mirroring
   doesn't change "is there a person with a knife nearby"), and it means
   the model isn't training on literally the same tensor repeated dozens
   of times.

5. **You can also just say the number directly.** `explicit_multipliers`
   overrides the automatic calculation per class — this is the literal
   "multiply potential_threat by N" knob, for when you've looked at the
   data and know better than a generic heuristic.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class BalancingPlan:
    """What prepare_g3e2.py should do with each class, and why."""
    original_counts: dict[str, int]
    target_counts: dict[str, int]  # class -> desired count AFTER balancing
    multipliers: dict[str, float] = field(default_factory=dict)  # class -> effective multiplier applied

    def to_dict(self) -> dict:
        return {
            "original_counts": self.original_counts,
            "target_counts": self.target_counts,
            "multipliers": {k: round(v, 2) for k, v in self.multipliers.items()},
        }

    def render(self) -> str:
        lines = ["G3E-2 CLASS BALANCING PLAN (train split only)", ""]
        for cls in sorted(self.original_counts, key=lambda c: -self.original_counts[c]):
            orig = self.original_counts[cls]
            target = self.target_counts.get(cls, orig)
            mult = self.multipliers.get(cls, 1.0)
            lines.append(f"  {cls:<20} {orig:>6}  ->  {target:>6}   (x{mult:.1f})")
        return "\n".join(lines)


def compute_balancing_plan(
    class_counts: dict[str, int],
    strategy: str = "capped",
    max_multiplier: float = 20.0,
    explicit_multipliers: dict[str, float] | None = None,
) -> BalancingPlan:
    """
    strategy:
      - "none": no oversampling — target_counts == original_counts.
      - "match_majority": every class oversampled to exactly match the
        largest class. Can produce extreme multipliers for very rare
        classes (use explicit_multipliers or a low max_multiplier to
        avoid this if you go this route).
      - "capped" (default): oversample toward the majority count, but cap
        any single class's multiplier at `max_multiplier`.

    `explicit_multipliers` (e.g. {"potential_threat": 50}) always wins for
    whichever classes it names — this is the direct "multiply this class by
    N" control; strategy/max_multiplier only apply to classes it doesn't mention.
    """
    explicit_multipliers = explicit_multipliers or {}
    if not class_counts:
        return BalancingPlan(original_counts={}, target_counts={}, multipliers={})

    majority_count = max(class_counts.values())
    target_counts: dict[str, int] = {}
    multipliers: dict[str, float] = {}

    for cls, count in class_counts.items():
        if cls in explicit_multipliers:
            mult = explicit_multipliers[cls]
        elif strategy == "none":
            mult = 1.0
        elif strategy == "match_majority":
            mult = majority_count / count if count > 0 else 1.0
        elif strategy == "capped":
            ideal = majority_count / count if count > 0 else 1.0
            mult = min(ideal, max_multiplier)
        else:
            raise ValueError(f"Unknown balancing strategy: {strategy!r}")

        mult = max(1.0, mult)  # never downsample here — see module docstring point 2
        multipliers[cls] = mult
        target_counts[cls] = round(count * mult)

    return BalancingPlan(original_counts=dict(class_counts), target_counts=target_counts, multipliers=multipliers)


def expand_samples(
    samples: list[dict],
    state_key_fn,
    plan: BalancingPlan,
) -> list[tuple[dict, bool]]:
    """
    Applies `plan` to a list of samples (each an arbitrary dict —
    prepare_g3e2.py passes its own per-sample record shape; this module
    doesn't need to know it). `state_key_fn(sample) -> str` extracts the
    semantic state used to look up that sample's multiplier.

    Returns a list of (sample, is_duplicate) pairs — `is_duplicate=True`
    for every copy beyond the first occurrence of a given original sample,
    which is exactly the signal prepare_g3e2.py uses to decide which copies
    get the horizontal-flip augmentation applied (see module docstring
    point 4) so a real vs. augmented copy is never ambiguous downstream.

    Oversampling is deterministic (round-robin repeats of the same list,
    not random sampling with replacement) so a re-run with the same plan
    always produces the same expanded set — important for reproducible
    training runs.
    """
    by_state: dict[str, list[dict]] = {}
    for sample in samples:
        by_state.setdefault(state_key_fn(sample), []).append(sample)

    expanded: list[tuple[dict, bool]] = []
    for state, group in by_state.items():
        target = plan.target_counts.get(state, len(group))
        if not group:
            continue
        for i in range(target):
            original = group[i % len(group)]
            is_duplicate = i >= len(group)
            expanded.append((original, is_duplicate))

    return expanded


def summarize_expansion(expanded: list[tuple[dict, bool]], state_key_fn) -> dict[str, int]:
    return dict(Counter(state_key_fn(sample) for sample, _ in expanded))
