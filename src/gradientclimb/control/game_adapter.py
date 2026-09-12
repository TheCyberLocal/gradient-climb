"""Bounded, template-guarded native reset actions for the inspected game layout.

Images remain local. Loading a profile sends no input. Every click requires a
fresh, matching full-frame state and independently matching named control patch.
Unknown states grant no action; only specifically labeled available ad closes are
allowed. No policy is implemented here.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gradientclimb.capture.screen import CapturedFrame, WindowCapture
from gradientclimb.capture.windows import ClientRect, WindowGuard, WindowTarget
from gradientclimb.perception.states import UIState

from .session import VerifiedControl
from .windows import INPUT, MOUSEINPUT, _NativeInput

GameState = Literal[
    "playing", "paused", "revive_offer", "bonus_offer", "result", "tune", "advertisement", "unknown"
]
CONTROL_STATES = {
    "pause_episode": "playing",
    "restart_paused": "paused",
    "resume_paused": "paused",
    "decline_revive_offer": "revive_offer",
    "decline_bonus_offer": "bonus_offer",
    "continue_result": "result",
    "start_episode": "tune",
    "legitimate_ad_close": "advertisement",
}
UI_STATES = {
    "playing": UIState.PLAYING,
    "paused": UIState.PAUSED,
    "revive_offer": UIState.SELECTION,
    "bonus_offer": UIState.SELECTION,
    "result": UIState.RESULT,
    "tune": UIState.SELECTION,
    "advertisement": UIState.ADVERTISEMENT,
    "unknown": UIState.UNEXPECTED,
}
NEXT_STATES = {
    "pause_episode": {"paused", "result", "revive_offer"},
    "restart_paused": {"playing", "advertisement"},
    "resume_paused": {"playing"},
    "decline_revive_offer": {"result"},
    "continue_result": {"tune", "bonus_offer", "advertisement"},
    "decline_bonus_offer": {"tune", "result", "advertisement"},
    "start_episode": {"playing", "advertisement"},
    "legitimate_ad_close": {"tune", "result", "bonus_offer", "playing", "advertisement"},
}
# Guard message emitted when another window took the foreground; a loss within a
# few seconds of an accepted click is effect-based evidence of an unintended action.
FOREGROUND_LOSS_MARKER = "Target is not the foreground window"
# Reset failures that mean the game is stuck on a screen the adapter may not touch.
# They may be escaped by restarting the application; guard faults never are.
STUCK_RESET_MARKERS = (
    "Unrecognized advertisement; no click",
    "Advertisement without legitimate control",
    "Unrecognized/unauthorized reset state",
    "did not recover within transition limit",
    "Unexpected or timed-out reset transition",
    "Reset time limit",
    "Reset click limit",
    "Reset capture limit",
)


class ReferenceVariant(BaseModel):
    """One hash-pinned full-frame reference with anchor patches and named controls.

    ``matching="white_glyph"`` (profile version 2) compares binarized white glyph
    shapes instead of raw pixels. It exists for advertisement-network chrome such
    as skip/close/mute icons drawn on translucent disks over arbitrary creatives,
    where raw RGB similarity is dominated by the creative behind the glyph. Such
    variants may carry a single anchor because the glyph shape itself is the
    creative-independent evidence; they are restricted to the advertisement state,
    and a click still requires the control glyph to match at its own threshold.
    """

    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1)
    state: GameState
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anchors: list[tuple[int, int, int, int]] = Field(min_length=1)
    controls: dict[str, tuple[int, int, int, int]] = Field(default_factory=dict)
    matching: Literal["rgb", "outlined_white", "white_glyph"] = "rgb"
    notes: str = ""

    @model_validator(mode="after")
    def allowlisted(self):
        if self.state == "unknown":
            raise ValueError("Unknown cannot authorize a reference/control")
        if self.matching == "outlined_white" and self.state != "result":
            raise ValueError("Outlined text matching is scoped to inspected result labels")
        if self.matching == "white_glyph" and self.state != "advertisement":
            raise ValueError("White glyph chrome matching is scoped to advertisement chrome")
        if self.matching != "white_glyph" and len(self.anchors) < 2:
            raise ValueError("Pixel-matched references require at least two anchors")
        for name in self.controls:
            if CONTROL_STATES.get(name) != self.state:
                raise ValueError("Control is not permitted for this exact game state")
        return self


class GameUIProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1, 2] = 1
    profile_id: str
    expected_size: tuple[int, int] = (1034, 581)
    threshold: float = Field(default=0.97, ge=0.97, le=1)
    margin_pixels: int = Field(default=4, ge=0, le=8)
    ambiguity_margin: float = Field(default=0.01, ge=0, le=0.1)
    glyph_threshold: float = Field(default=0.90, ge=0.85, le=1)
    glyph_control_threshold: float = Field(default=0.95, ge=0.90, le=1)
    evidence_scope: str = "local construction references; live reset unvalidated"
    variants: list[ReferenceVariant] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_regions(self):
        width, height = self.expected_size
        if width < 2 or height < 2 or len({v.label for v in self.variants}) != len(self.variants):
            raise ValueError("Profile size and unique reference labels are required")
        if self.glyph_control_threshold < self.glyph_threshold:
            raise ValueError("Control glyph threshold cannot be below the state glyph threshold")
        for variant in self.variants:
            if variant.matching == "white_glyph" and self.version < 2:
                raise ValueError("White glyph matching requires profile version 2")
            for left, top, right, bottom in [*variant.anchors, *variant.controls.values()]:
                if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
                    raise ValueError("Reference boxes must stay inside the normalized frame")
        return self

    def state_threshold(self, variant: ReferenceVariant) -> float:
        return self.glyph_threshold if variant.matching == "white_glyph" else self.threshold

    def control_threshold(self, variant: ReferenceVariant) -> float:
        return self.glyph_control_threshold if variant.matching == "white_glyph" else self.threshold


@dataclass(frozen=True)
class GameObservation:
    frame: CapturedFrame
    state: GameState
    confidence: float
    variant: str | None
    controls: tuple[VerifiedControl, ...]
    pixels_sha256: str


class StaleObservation(RuntimeError):
    """A valid observation has aged out; a fresh capture may recover safely."""


class GameUIRecognizer:
    def __init__(self, profile: GameUIProfile, reference_root: Path):
        self.profile = profile
        self.reference_root = Path(reference_root).resolve()
        self.references = []
        self.reference_masks = {}
        for variant in profile.variants:
            path = (self.reference_root / variant.file).resolve()
            if not path.is_relative_to(self.reference_root) or not path.is_file():
                raise ValueError("Reference must be a local file inside reference_root")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != variant.sha256:
                raise ValueError(f"Reference hash mismatch: {variant.label}")
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            if (rgb.shape[1], rgb.shape[0]) != profile.expected_size:
                raise ValueError("Reference geometry differs from capture profile")
            patches = []
            for index, box in enumerate([*variant.anchors, *variant.controls.values()]):
                left, top, right, bottom = box
                patch = rgb[top:bottom, left:right]
                if patch.size < 48 or patch.std(axis=(0, 1)).max() < 8:
                    raise ValueError("Each state/control patch must contain distinctive pixels")
                patches.append(patch.astype(np.float32))
                if variant.matching == "outlined_white":
                    self.reference_masks[(variant.label, index)] = self._text_mask(patch)
                elif variant.matching == "white_glyph":
                    self.reference_masks[(variant.label, index)] = self._glyph_mask(patch)
            self.references.append((variant, patches))

    @staticmethod
    def _text_mask(patch):
        import cv2

        # Match solid glyph cores and nearby dark outlines. Antialiased boundary
        # pixels blend with the moving world and are intentionally not evidence.
        white = (patch.min(axis=2) >= 240) & (np.ptp(patch, axis=2) <= 15)
        nearby = cv2.dilate(white.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        outline = (patch.max(axis=2) <= 20) & nearby
        if white.sum() < 24 or outline.sum() < 12:
            raise ValueError("Result text mask requires both white glyphs and dark outlines")
        return (white | outline).astype(np.uint8)

    @staticmethod
    def _white(rgb):
        return (rgb.min(axis=2) >= 225) & (np.ptp(rgb, axis=2) <= 25)

    @classmethod
    def _glyph_mask(cls, patch):
        # A chrome glyph is a compact solid white shape; its translucent disk and
        # the creative behind it are deliberately not evidence.
        white = cls._white(patch)
        if white.sum() < 40 or white.mean() > 0.5:
            raise ValueError("A chrome glyph reference needs a compact solid white shape")
        return white.astype(np.uint8)

    @classmethod
    def from_file(cls, path, reference_root):
        return cls(GameUIProfile.model_validate_json(Path(path).read_text()), reference_root)

    def _match(self, rgb, patch, box, mask=None, *, glyph=False):
        import cv2

        left, top, right, bottom = box
        margin = self.profile.margin_pixels
        x0, y0 = max(0, left - margin), max(0, top - margin)
        region = rgb[
            y0 : min(rgb.shape[0], bottom + margin), x0 : min(rgb.shape[1], right + margin)
        ]
        if glyph:
            # Dice overlap between binarized white shapes at the best translation.
            binary = self._white(region).astype(np.float32)
            reference = mask.astype(np.float32)
            _, _, point, _ = cv2.minMaxLoc(cv2.matchTemplate(binary, reference, cv2.TM_SQDIFF))
            x, y = point
            candidate = binary[y : y + reference.shape[0], x : x + reference.shape[1]] > 0
            glyph_pixels = mask > 0
            union = int(candidate.sum()) + int(glyph_pixels.sum())
            score = 2 * int((candidate & glyph_pixels).sum()) / union if union else 0.0
            return score, (x0 + x, y0 + y, x0 + x + reference.shape[1], y0 + y + reference.shape[0])
        minimum, _, point, _ = cv2.minMaxLoc(
            cv2.matchTemplate(region.astype(np.float32), patch, cv2.TM_SQDIFF, mask=mask)
        )
        count = patch.size if mask is None else int(mask.sum()) * 3
        score = 1 - math.sqrt(max(0, minimum) / count) / 255
        x, y = x0 + point[0], y0 + point[1]
        return score, (x, y, x + patch.shape[1], y + patch.shape[0])

    def observe(self, frame: CapturedFrame) -> GameObservation:
        rgb = frame.rgb
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.shape != (self.profile.expected_size[1], self.profile.expected_size[0], 3)
        ):
            raise ValueError("Unexpected capture geometry/type")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        ranked = []
        for variant, patches in self.references:
            glyph = variant.matching == "white_glyph"
            matches = [
                self._match(
                    rgb, patch, box, self.reference_masks.get((variant.label, index)), glyph=glyph
                )
                for index, (patch, box) in enumerate(
                    zip(patches[: len(variant.anchors)], variant.anchors, strict=True)
                )
            ]
            score = min(score for score, _ in matches)
            # Scores are compared as margins above each variant's own threshold so
            # pixel similarity and glyph overlap never compete on different scales.
            ranked.append((score - self.profile.state_threshold(variant), score, variant, patches))
        ranked.sort(key=lambda item: item[0], reverse=True)
        # A known modal has priority over unobscured HUD anchors behind it.
        # Near-threshold modal evidence vetoes PLAYING without granting a click.
        modal = [
            item
            for item in ranked
            if item[2].state != "playing" and item[0] >= -self.profile.ambiguity_margin
        ]
        if modal:
            ranked = [item for item in ranked if item[2].state != "playing"]
        excess, score, variant, patches = ranked[0]
        competitors = [e for e, _, v, _ in ranked if v.state != variant.state]
        if excess < 0 or (
            competitors and excess - max(competitors) < self.profile.ambiguity_margin
        ):
            return GameObservation(frame, "unknown", 0.0, None, (), digest)
        controls = []
        glyph = variant.matching == "white_glyph"
        for index, ((name, box), patch) in enumerate(
            zip(variant.controls.items(), patches[len(variant.anchors) :], strict=True),
            start=len(variant.anchors),
        ):
            control_score, bounds = self._match(
                rgb, patch, box, self.reference_masks.get((variant.label, index)), glyph=glyph
            )
            if control_score >= self.profile.control_threshold(variant):
                controls.append(
                    VerifiedControl(
                        name, UI_STATES[variant.state], frame.timestamp_ns, bounds, control_score
                    )
                )
        return GameObservation(frame, variant.state, score, variant.label, tuple(controls), digest)


def mouse_click_events(point: tuple[int, int], desktop: ClientRect) -> list[INPUT]:
    x, y = point
    if (
        any(type(v) is not int for v in point)
        or desktop.width < 2
        or desktop.height < 2
        or not desktop.left <= x < desktop.left + desktop.width
        or not desktop.top <= y < desktop.top + desktop.height
    ):
        raise ValueError("Click point is outside the physical virtual desktop")
    nx = round((x - desktop.left) * 65535 / (desktop.width - 1))
    ny = round((y - desktop.top) * 65535 / (desktop.height - 1))
    result = []
    for dx, dy, flags in ((nx, ny, 0xC001), (0, 0, 0x0002), (0, 0, 0x0004)):
        event = INPUT()
        event.type = 0
        event.mi = MOUSEINPUT(dx, dy, 0, flags, 0, 0)
        result.append(event)
    return result


class _NativeMenuInput:
    def __init__(self, guard):
        self.guard, self.api = guard, _NativeInput()
        self.api.user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        self.api.user32.GetSystemMetrics.restype = ctypes.c_int
        self.trace = []

    def is_down(self, key):
        return self.api.is_down(key)

    def park_pointer(self, point):
        if self.is_down(0x01):
            raise RuntimeError("Physical left mouse button is held; pointer movement refused")
        row = {
            "kind": "pointer_park",
            "point": point,
            "started_ns": time.perf_counter_ns(),
            "inserted": None,
        }
        try:
            with self.guard.api.physical_pixels():
                desktop = ClientRect(
                    *(self.api.user32.GetSystemMetrics(i) for i in (76, 77, 78, 79))
                )
                row["inserted"] = self.api.send([mouse_click_events(point, desktop)[0]])
            if row["inserted"] != 1:
                raise RuntimeError("SendInput did not insert the pointer parking movement")
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["completed_ns"] = time.perf_counter_ns()
            self.trace.append(row)

    def click(self, point):
        if self.is_down(0x01):
            raise RuntimeError("Physical left mouse button is already held")
        row = {
            "kind": "menu_click",
            "point": point,
            "started_ns": time.perf_counter_ns(),
            "inserted": None,
        }
        try:
            with self.guard.api.physical_pixels():
                desktop = ClientRect(
                    *(self.api.user32.GetSystemMetrics(i) for i in (76, 77, 78, 79))
                )
                count = self.api.send(mouse_click_events(point, desktop))
            row["inserted"] = count
            if count != 3:
                raise RuntimeError("SendInput did not insert the complete menu click")
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            release = INPUT()
            release.type = 0
            release.mi = MOUSEINPUT(0, 0, 0, 0x0004, 0, 0)
            try:
                row["cleanup_inserted"] = self.api.send([release])
            except BaseException as cleanup:  # noqa: BLE001 - preserve original interruption
                row["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
            raise
        finally:
            row["completed_ns"] = time.perf_counter_ns()
            self.trace.append(row)


class NativeGameAdapter:
    """Capture/reset adapter; compose a policy and PedalController explicitly.

    All methods are bounded except the supplied capture/callback's own execution;
    the independent pedal watchdog must remain active throughout. Only `.reset`
    and `.click_verified` can send the explicitly allowlisted menu clicks.
    """

    def __init__(
        self,
        target: WindowTarget,
        profile_path,
        reference_root="artifacts",
        *,
        guard=None,
        capture=None,
        capture_backend: Literal["dxcam", "mss", "pillow"] = "dxcam",
        sender=None,
        release_pedals=None,
        max_observation_age_seconds=0.45,
        clock=time.perf_counter,
        sleep=time.sleep,
    ):
        if capture_backend not in {"dxcam", "mss", "pillow"}:
            raise ValueError("Explicit capture backend must be dxcam, mss, or pillow")
        if (
            not math.isfinite(max_observation_age_seconds)
            or not 0 < max_observation_age_seconds <= 0.5
        ):
            raise ValueError("Observation age must be within (0,0.5] seconds")
        self.target = target
        self.guard = guard if guard is not None else WindowGuard(target)
        self.recognizer = GameUIRecognizer.from_file(profile_path, reference_root)
        self.capture_backend_name = capture_backend
        self.capture_backend = (
            capture
            if capture is not None
            else WindowCapture(
                self.guard,
                backend=capture_backend,
                output_size=self.recognizer.profile.expected_size,
            )
        )
        self.sender = sender if sender is not None else _NativeMenuInput(self.guard)
        if not callable(release_pedals):
            raise TypeError("A pedal-release callback is required before menu navigation")
        self.release_pedals = release_pedals
        self.max_age, self.clock, self.sleep = max_observation_age_seconds, clock, sleep
        self.latest = None
        self.trace = []
        self.guard_trace = []
        self.capture_trace = []
        self.restart_trace = []
        self._latched = False
        self._stopped = threading.Event()
        self._clicked = set()
        self._last_frame_ns = -1
        # Only the inspected wrapper's blank title-bar region is authorized for
        # post-click movement. This never adds a mouse-down or generic click.
        self.pointer_park_xy = (
            (520, 18) if self.recognizer.profile.expected_size == (1034, 581) else None
        )

    def __enter__(self):
        self.capture_backend.__enter__()
        return self

    def __exit__(self, *args):
        try:
            self.stop()
        finally:
            self.capture_backend.__exit__(*args)

    def stop(self):
        self._stopped.set()
        self.release_pedals()

    def _guard(self):
        try:
            if self._stopped.is_set() or self.sender.is_down(0x1B):
                raise RuntimeError("Adapter stopped by operator or previous fault")
            if self.guard.validate(require_foreground=True).client_rect != self.target.client_rect:
                raise RuntimeError("Game geometry changed; explicit profile reselection required")
        except Exception as exc:
            self._stopped.set()
            self._latched = True
            self._guard_event(f"{type(exc).__name__}: {exc}", latched=True)
            raise

    @property
    def last_click_ns(self):
        """Clock timestamp of the most recent accepted click, or None."""
        for row in reversed(self.trace):
            if row.get("accepted"):
                return row.get("completed_ns")
        return None

    def _guard_event(self, reason, *, latched=False):
        if self.guard_trace and (
            self.guard_trace[-1]["reason"],
            self.guard_trace[-1]["latched"],
        ) == (reason, latched):
            return
        self.guard_trace.append(
            {"timestamp_ns": int(self.clock() * 1e9), "reason": reason, "latched": latched}
        )

    def _fresh(self, frame):
        now = self.clock() * 1e9
        if (
            frame.source_rect != self.target.client_rect
            or not 0 <= frame.started_ns <= frame.completed_ns <= now
        ):
            raise RuntimeError("Invalid or mismatched capture")
        if (now - frame.started_ns) / 1e9 > self.max_age:
            raise StaleObservation("Capture exceeded maximum observation age")

    def observe(self):
        try:
            for attempt in range(2):
                self._guard()
                frame = self.capture_backend.grab()
                self._guard()
                try:
                    self._fresh(frame)
                except StaleObservation as exc:
                    self.capture_trace.append(
                        {**frame.metadata(), "discarded": True, "reason": str(exc)}
                    )
                    self._guard_event("Discarded stale capture; acquiring fresh frame")
                    self.release_pedals()
                    if attempt == 1:
                        raise
                    continue
                if frame.completed_ns <= self._last_frame_ns:
                    raise RuntimeError("Repeated/nonmonotonic capture timestamp")
                self._last_frame_ns = frame.completed_ns
                self.latest = self.recognizer.observe(frame)
                return self.latest
        except StaleObservation:
            # Both attempts are recorded and no old frame is returned. A caller
            # may explicitly retry; the watchdog remains false until fresh input.
            self.release_pedals()
            raise
        except Exception:
            self._stopped.set()
            self.release_pedals()
            raise

    def is_playing(self):
        try:
            self._guard()
            if self.latest is None or self.latest.state != "playing":
                self._guard_event("No recognized PLAYING observation")
                return False
            self._fresh(self.latest.frame)
            self._guard_event("Fresh PLAYING observation")
            return True
        except StaleObservation as exc:
            # The PedalController watchdog releases on False. Do not call its
            # release method from inside its state callback (it holds a lock).
            self._guard_event(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - watchdog receives a fail-closed state
            self._stopped.set()
            self._guard_event(f"{type(exc).__name__}: {exc}", latched=True)
            return False

    def click_verified(self, name, observation=None):
        try:
            self.release_pedals()
            self._guard()
            observation = observation if observation is not None else self.latest
            if observation is None or CONTROL_STATES.get(name) != observation.state:
                raise RuntimeError("Named control is not permitted in the current state")
            self._fresh(observation.frame)
            # Recompute from the exact source pixels; forged/stale control objects
            # cannot substitute arbitrary coordinates or a different state.
            checked = self.recognizer.observe(observation.frame)
            if (
                checked.state != observation.state
                or checked.pixels_sha256 != observation.pixels_sha256
            ):
                raise RuntimeError("Observation changed before click")
            candidates = [c for c in checked.controls if c.name == name]
            if len(candidates) != 1:
                raise RuntimeError("Unique verified named control unavailable")
            control = candidates[0]
            identity = (control.frame_timestamp_ns, name)
            if identity in self._clicked:
                raise RuntimeError("Control already clicked on this frame")
            left, top, right, bottom = control.bounds_xyxy
            width, height = self.recognizer.profile.expected_size
            rect = self.target.client_rect
            point = (
                rect.left + round((left + right) * rect.width / (2 * width)),
                rect.top + round((top + bottom) * rect.height / (2 * height)),
            )
            self._guard()
            self._fresh(observation.frame)
            self._clicked.add(identity)
            row = {
                "name": name,
                "state": checked.state,
                "variant": checked.variant,
                "frame_timestamp_ns": control.frame_timestamp_ns,
                "source_pixels_sha256": checked.pixels_sha256,
                "bounds_xyxy": control.bounds_xyxy,
                "physical_point": point,
                "started_ns": int(self.clock() * 1e9),
                "accepted": False,
            }
            try:
                self.sender.click(point)
                row["accepted"] = True
                if self.pointer_park_xy is not None and hasattr(self.sender, "park_pointer"):
                    self._guard()
                    x, y = self.pointer_park_xy
                    park_point = (
                        rect.left + round(x * rect.width / width),
                        rect.top + round(y * rect.height / height),
                    )
                    self.sender.park_pointer(park_point)
                    row["pointer_parked"] = True
            except BaseException as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                row["completed_ns"] = int(self.clock() * 1e9)
                self.trace.append(row)
        except BaseException:
            self._stopped.set()
            self.release_pedals()
            raise

    def reset(
        self,
        *,
        truncate=False,
        start_next=True,
        allow_initial_start=False,
        max_seconds=60.0,
        max_clicks=8,
        transition_seconds=10.0,
        ad_transition_seconds=45.0,
        ad_close_stability_seconds=0.15,
        on_terminal=None,
        on_observation=None,
    ):
        """Drive the inspected menu flow to the next boundary with only verified clicks.

        Advertisement phases are bounded by ``ad_transition_seconds`` measured from
        the most recent recognized advertisement chrome; unknown frames inside that
        window wait without input. A legitimate close/skip control must be observed
        on two separate fresh frames at least ``ad_close_stability_seconds`` apart,
        at the same location, before it is clicked, so a control that is fading or
        moving cannot be hit.
        """
        if (
            type(truncate) is not bool
            or type(start_next) is not bool
            or type(allow_initial_start) is not bool
            or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 120
            or not math.isfinite(transition_seconds)
            or not 0 < transition_seconds <= 15
            or not math.isfinite(ad_transition_seconds)
            or not 0 < ad_transition_seconds <= 90
            or not math.isfinite(ad_close_stability_seconds)
            or not 0 <= ad_close_stability_seconds <= 2
            or type(max_clicks) is not int
            or not 1 <= max_clicks <= 10
        ):
            raise ValueError("Reset requires finite bounded time/click limits")
        deadline = self.clock() + max_seconds
        pending = None
        clicks = 0
        ready = False
        result_saved = False
        observation = None
        known_ad_until = None
        known_result_until = None
        ad_close_seen = None
        ad_no_control_since = None
        try:
            self.release_pedals()
            for _ in range(2000):
                self._guard()
                if self.clock() >= deadline:
                    raise TimeoutError("Reset time limit")
                observation = self.observe()
                if on_observation is not None:
                    on_observation(observation)
                if self.clock() >= deadline:
                    raise TimeoutError("Reset time limit after observation callback")
                if pending:
                    previous, previous_variant, expected, until = pending
                    if observation.state in expected and (
                        observation.state != previous or observation.variant != previous_variant
                    ):
                        pending = None
                    elif observation.state in {previous, "unknown"} and self.clock() < until:
                        self._guard_event("Awaiting expected transition; no input while unknown")
                        self.sleep(0.03)
                        continue
                    else:
                        raise RuntimeError("Unexpected or timed-out reset transition")
                state = observation.state
                if state != "advertisement":
                    ad_close_seen = None
                    ad_no_control_since = None
                if state == "result":
                    known_result_until = min(deadline, self.clock() + transition_seconds)
                elif state == "unknown" and known_result_until is not None:
                    if self.clock() < known_result_until:
                        self._guard_event("Previously recognized result; no-input transient wait")
                        self.sleep(0.03)
                        continue
                    raise RuntimeError("Result recognition did not recover within transition limit")
                else:
                    known_result_until = None
                if state == "advertisement":
                    known_ad_until = min(deadline, self.clock() + ad_transition_seconds)
                elif state == "unknown" and known_ad_until is not None:
                    if self.clock() < known_ad_until:
                        self._guard_event("Unknown ad phase; bounded wait without input")
                        self.sleep(0.03)
                        continue
                    raise RuntimeError("Unrecognized advertisement; no click")
                else:
                    known_ad_until = None
                if state == "playing":
                    if ready or not truncate:
                        return observation
                    name = "pause_episode"
                elif state == "paused":
                    if not start_next:
                        return observation
                    name = "restart_paused"
                    ready = True
                elif state == "revive_offer":
                    name = "decline_revive_offer"
                elif state == "bonus_offer":
                    name = "decline_bonus_offer"
                elif state == "advertisement":
                    closes = [c for c in observation.controls if c.name == "legitimate_ad_close"]
                    if not closes:
                        ad_close_seen = None
                        if ad_no_control_since is None:
                            ad_no_control_since = self.clock()
                        elif self.clock() - ad_no_control_since >= ad_transition_seconds:
                            raise RuntimeError(
                                "Advertisement without legitimate control persisted; no click"
                            )
                        self._guard_event("Known advertisement; close unavailable; waiting")
                        self.sleep(0.03)
                        continue
                    ad_no_control_since = None
                    stamp, bounds = closes[0].frame_timestamp_ns, closes[0].bounds_xyxy
                    if ad_close_seen is None or any(
                        abs(a - b) > 2 for a, b in zip(ad_close_seen[1], bounds, strict=True)
                    ):
                        ad_close_seen = (stamp, bounds)
                        self._guard_event("Advertisement close visible; confirming stability")
                        self.sleep(0.03)
                        continue
                    if (stamp - ad_close_seen[0]) / 1e9 < ad_close_stability_seconds:
                        self.sleep(0.03)
                        continue
                    ad_close_seen = None
                    name = "legitimate_ad_close"
                elif state == "result":
                    if on_terminal is None:
                        raise RuntimeError(
                            "Terminal-frame callback required before dismissing results"
                        )
                    if not result_saved:
                        if on_terminal(observation) is False:
                            self._guard_event("Result callback requests another fresh observation")
                            self.sleep(0.03)
                            continue
                        result_saved = True
                    if not any(c.name == "continue_result" for c in observation.controls):
                        self._guard_event("Recognized result; Continue text unavailable; waiting")
                        self.sleep(0.03)
                        continue
                    name = "continue_result"
                elif state == "tune" and (result_saved or allow_initial_start):
                    if not start_next:
                        return observation
                    name = "start_episode"
                    ready = True
                else:
                    raise RuntimeError("Unrecognized/unauthorized reset state; no click")
                if clicks >= max_clicks:
                    raise RuntimeError("Reset click limit")
                if self.clock() >= deadline:
                    raise TimeoutError("Reset time limit before click")
                self.click_verified(name, observation)
                clicks += 1
                pending = (
                    state,
                    observation.variant,
                    NEXT_STATES[name],
                    min(
                        deadline,
                        self.clock()
                        + (
                            ad_transition_seconds
                            if state == "advertisement" or "advertisement" in NEXT_STATES[name]
                            else transition_seconds
                        ),
                    ),
                )
                self.sleep(0.03)
            raise RuntimeError("Reset capture limit")
        except Exception:
            self._stopped.set()
            self.release_pedals()
            raise

    def restart_app(
        self,
        launch,
        *,
        max_seconds=180.0,
        hide_seconds=20.0,
        poll_seconds=0.5,
        settle_states=("tune",),
        reason="stuck reset",
        on_observation=None,
    ):
        """Escape a stuck screen by restarting the game application; never by clicking.

        The only actions are a WM_CLOSE message to the pinned game window (the
        emulator's own exit path) and the caller's ``launch`` callable (the game's
        shortcut). Nothing inside the game is clicked, so an advertisement's
        creative, a store page or a purchase dialog cannot be activated. The window
        must hide within ``hide_seconds``, reappear with an unchanged identity and
        client geometry, take the foreground, and show a recognized state within
        ``max_seconds``; unknown boot frames are waited out without input.

        Refused after a latched guard fault (foreground loss, geometry change,
        capture or operator faults): those can indicate an external effect that
        needs an operator, and hiding the evidence behind a restart is not allowed.
        A stuck reset (``STUCK_RESET_MARKERS``) is not latched and may be recovered.
        """
        if not callable(launch):
            raise TypeError("A launch callable is required to restart the game")
        if (
            not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 300
            or not math.isfinite(hide_seconds)
            or not 0 < hide_seconds <= 60
            or not math.isfinite(poll_seconds)
            or not 0 < poll_seconds <= 5
        ):
            raise ValueError("Restart requires finite bounded time limits")
        if self._latched:
            self._stopped.set()
            self.release_pedals()
            raise RuntimeError("Restart refused after a latched guard fault")
        row = {
            "reason": reason,
            "started_ns": int(self.clock() * 1e9),
            "closed": False,
            "hidden_seconds": None,
            "launched": False,
            "visible_seconds": None,
            "settled_state": None,
            "settled_seconds": None,
            "error": None,
        }
        self.restart_trace.append(row)
        api = self.guard.api
        hwnd = self.target.hwnd
        deadline = self.clock() + max_seconds
        try:
            self.release_pedals()
            self.guard.validate(require_foreground=False)
            if not api.request_close(hwnd):
                raise RuntimeError("WM_CLOSE could not be posted to the game window")
            row["closed"] = True
            started = self.clock()
            while api.visible(hwnd):
                if self.clock() - started >= hide_seconds:
                    raise TimeoutError("Game window did not hide after WM_CLOSE")
                self.sleep(poll_seconds)
            row["hidden_seconds"] = self.clock() - started
            self.release_pedals()
            launch()
            row["launched"] = True
            started = self.clock()
            while not api.visible(hwnd):
                if self.clock() >= deadline:
                    raise TimeoutError("Game window did not reappear after relaunch")
                self.sleep(poll_seconds)
            row["visible_seconds"] = self.clock() - started
            current = self.guard.validate(require_foreground=False)
            if current.client_rect != self.target.client_rect:
                raise RuntimeError(
                    "Game geometry changed across restart; explicit profile reselection required"
                )
            if not api.bring_to_foreground(hwnd):
                raise RuntimeError("Game window could not take the foreground after relaunch")
            # The stuck fault is the only latch a restart may clear.
            self._stopped.clear()
            started = self.clock()
            while True:
                if self.clock() >= deadline:
                    raise TimeoutError("Game did not reach a recognized state after relaunch")
                try:
                    observation = self.observe()
                except StaleObservation:
                    self.sleep(poll_seconds)
                    continue
                if on_observation is not None:
                    on_observation(observation)
                if observation.state != "unknown":
                    row["settled_state"] = observation.state
                    row["settled_seconds"] = self.clock() - started
                    if observation.state not in settle_states:
                        self._guard_event(
                            f"Relaunch settled on {observation.state}; the caller's reset continues"
                        )
                    return observation
                self.sleep(poll_seconds)
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self._stopped.set()
            self.release_pedals()
            raise
        finally:
            row["completed_ns"] = int(self.clock() * 1e9)
