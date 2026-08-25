"""
Semantic annotation validation — spec section 19: "Do not blindly trust
semantic annotations." Runs BEFORE any G3E-2 training data is built, and
produces a report a human should actually read, not just a pass/fail.

This does not judge whether a semantic judgment is *correct* (that needs a
human or a much larger review process) — it catches the mechanical failure
modes that would otherwise poison training silently: missing/empty fields,
invalid state or severity values, malformed JSON, and duplicate
descriptions (a common symptom of a rule engine — see g3e-data-engine's
semantic/rules.py — stamping the same canned text on many different images,
which teaches a language model to repeat a phrase rather than reason).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from shared.schemas import SEMANTIC_STATES, SEMANTIC_SEVERITIES

REQUIRED_FIELDS = ("state", "severity", "description", "reason")


@dataclass
class SemanticIssue:
    path: str
    problem: str


@dataclass
class SemanticValidationReport:
    total_checked: int
    issues: list[SemanticIssue] = field(default_factory=list)
    state_counts: dict[str, int] = field(default_factory=dict)
    duplicate_description_count: int = 0

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> dict:
        return {
            "total_checked": self.total_checked,
            "valid": self.total_checked - len({i.path for i in self.issues}),
            "issue_count": len(self.issues),
            "issues": [{"path": i.path, "problem": i.problem} for i in self.issues],
            "state_counts": self.state_counts,
            "duplicate_description_count": self.duplicate_description_count,
        }

    def render(self) -> str:
        lines = [
            "G3E-2 SEMANTIC ANNOTATION VALIDATION",
            "",
            f"Checked: {self.total_checked}",
            f"Valid:   {self.total_checked - len({i.path for i in self.issues})}",
            f"Issues:  {len(self.issues)}",
            "",
            f"State distribution: {self.state_counts}",
        ]
        if self.duplicate_description_count:
            lines.append(
                f"\n[warning] {self.duplicate_description_count} description(s) are repeated "
                "across multiple images — a language model trained on this may learn to repeat "
                "a canned phrase rather than describe what it actually sees. Consider adding "
                "per-image detail (object count, position) to the rule engine's templates."
            )
        if self.issues:
            lines.append("\nISSUES (first 20 shown):")
            for issue in self.issues[:20]:
                lines.append(f"  ✗ {issue.path}: {issue.problem}")
            if len(self.issues) > 20:
                lines.append(f"  ... and {len(self.issues) - 20} more — see the full report.")
        return "\n".join(lines)


def validate_semantic_file(path: Path) -> tuple[dict | None, list[str]]:
    """Returns (parsed_dict_or_None, problems). A None dict means unusable — skip this sample entirely."""
    problems: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return None, [f"malformed JSON or unreadable file: {exc}"]

    semantic = data.get("semantic")
    if not isinstance(semantic, dict):
        return None, ["missing or non-object 'semantic' field"]

    for field_name in REQUIRED_FIELDS:
        value = semantic.get(field_name)
        if value is None:
            problems.append(f"missing field '{field_name}'")
        elif isinstance(value, str) and not value.strip() and field_name != "reason":
            # `reason` is allowed to be empty for the `default`/normal case
            # (see configs/semantic_rules.yaml in g3e-data-engine) — every
            # other field must be non-empty.
            problems.append(f"empty field '{field_name}'")

    state = semantic.get("state")
    if state is not None and state not in SEMANTIC_STATES:
        problems.append(f"invalid state {state!r} — expected one of {sorted(SEMANTIC_STATES)}")

    severity = semantic.get("severity")
    if severity is not None and severity not in SEMANTIC_SEVERITIES:
        problems.append(f"invalid severity {severity!r} — expected one of {sorted(SEMANTIC_SEVERITIES)}")

    if not isinstance(data.get("objects"), list):
        problems.append("missing or non-list 'objects' field")

    return data, problems


def validate_semantic_directory(semantic_dir: Path) -> SemanticValidationReport:
    files = sorted(semantic_dir.rglob("*.json"))
    issues: list[SemanticIssue] = []
    states: list[str] = []
    descriptions: list[str] = []

    for path in files:
        data, problems = validate_semantic_file(path)
        for problem in problems:
            issues.append(SemanticIssue(path=str(path), problem=problem))
        if data is not None:
            semantic = data.get("semantic", {})
            state = semantic.get("state")
            if state:
                states.append(state)
            description = semantic.get("description")
            if description:
                descriptions.append(description)

    description_counts = Counter(descriptions)
    duplicate_description_count = sum(c for c in description_counts.values() if c > 1)

    return SemanticValidationReport(
        total_checked=len(files),
        issues=issues,
        state_counts=dict(Counter(states)),
        duplicate_description_count=duplicate_description_count,
    )
