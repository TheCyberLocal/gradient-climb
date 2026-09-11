import math
from dataclasses import dataclass
from enum import Enum


class UIState(str, Enum):
    MAIN_MENU = "main_menu"
    SELECTION = "selection"
    STARTING = "starting"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    RESULT = "result"
    ADVERTISEMENT = "advertisement"
    RETURN = "return"
    UNEXPECTED = "unexpected"


@dataclass(frozen=True)
class StateEvidence:
    state: UIState
    confidence: float
    legitimate_close_visible: bool = False
    restart_visible: bool = False

    def __post_init__(self):
        if not isinstance(self.state, UIState):
            raise TypeError("A known UIState is required")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be finite and within [0,1]")
        if (
            type(self.legitimate_close_visible) is not bool
            or type(self.restart_visible) is not bool
        ):
            raise ValueError("Visible-control evidence must be boolean")


def permitted_action(evidence: StateEvidence, threshold: float = 0.97) -> str:
    """A recognizer supplies evidence; never infer a click from elapsed ad time."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0,1]")
    if evidence.confidence < threshold:
        return "release_and_halt"
    if evidence.state == UIState.PLAYING:
        return "policy"
    if evidence.state == UIState.ADVERTISEMENT:
        return "close_verified_ad" if evidence.legitimate_close_visible else "release_and_wait"
    if evidence.state in {UIState.GAME_OVER, UIState.RESULT, UIState.RETURN}:
        return "verified_restart" if evidence.restart_visible else "release_and_halt"
    if evidence.state == UIState.STARTING:
        return "release_and_wait"
    return "release_and_halt"
