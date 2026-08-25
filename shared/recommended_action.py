"""
`recommended_action` isn't produced by g3e-data-engine's rule engine (it
only outputs state/severity/description/reason) — spec section 11 wants it
in G3E-2's target output regardless. This derives a default action from
(state, severity) so training targets have the field, clearly marked as
SYNTHESIZED, not ground truth from a human or the original rule engine.

Override `ACTION_MAP` (or pass a custom map into `derive_recommended_action`)
if G3E's actual response policy differs from these defaults — this is a
placeholder policy, not a decision this codebase should make silently.
"""
from __future__ import annotations

ACTION_MAP: dict[tuple[str, str], str] = {
    ("potential_threat", "critical"): "alert_immediately",
    ("potential_threat", "high"): "alert_user",
    ("potential_threat", "medium"): "alert_user",
    ("hazard", "high"): "alert_user",
    ("hazard", "medium"): "log_and_monitor",
    ("hazard", "low"): "log_and_monitor",
    ("caution", "medium"): "log_and_monitor",
    ("caution", "low"): "log_and_monitor",
    ("normal", "none"): "none",
}

DEFAULT_ACTION = "review"


def derive_recommended_action(state: str, severity: str) -> str:
    return ACTION_MAP.get((state, severity), DEFAULT_ACTION)
