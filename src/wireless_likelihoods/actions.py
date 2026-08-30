"""The lightweight, abstract actions used by the MVP."""

from .types import ActionSpec

ACTIONS = (
    ActionSpec(0, "fast_low_redundancy", 0.10, 1, 0.7, 0.5),
    ActionSpec(1, "interleaved", 0.20, 8, 1.0, 1.0),
    ActionSpec(2, "robust", 0.35, 16, 1.5, 2.0),
)


def get_action(action_id: int) -> ActionSpec:
    """Return an action by ID, raising a clear error for invalid IDs."""
    for action in ACTIONS:
        if action.action_id == action_id:
            return action
    raise ValueError(f"Unknown action_id {action_id}; expected 0, 1, or 2.")
