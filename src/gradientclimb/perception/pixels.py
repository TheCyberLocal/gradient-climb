"""Calibrated-profile pixel estimators; no pretrained game recognizer is bundled.

Template scores and geometric confidence are heuristics, not probabilities or
measured real-game accuracy. Profiles require locally collected labels and held-out
validation. Missing profiles, geometry changes, and ambiguous matches fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .states import StateEvidence, UIState


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install gradientclimb[capture] for pixel estimators") from exc
    return cv2


def _rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image with shape (height,width,3)")
    if min(frame.shape[:2]) < 2:
        raise ValueError("Image is too small")
    return frame


def _region(frame: np.ndarray, roi) -> tuple[np.ndarray, int, int]:
    if len(roi) != 4 or not all(math.isfinite(v) for v in roi):
        raise ValueError("ROI must contain four finite normalized coordinates")
    l, t, r, b = roi
    if not 0 <= l < r <= 1 or not 0 <= t < b <= 1:
        raise ValueError("ROI must have positive area within [0,1]")
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = round(l * width), round(t * height), round(r * width), round(b * height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI rounds to an empty image")
    return frame[y0:y1, x0:x1], x0, y0


@dataclass(frozen=True)
class UITemplate:
    label: str
    state: UIState
    rgb: np.ndarray
    roi: tuple[float, float, float, float]
    threshold: float = 0.98
    role: str = "state"

    def __post_init__(self):
        pixels = _rgb(self.rgb)
        if pixels.shape[0] * pixels.shape[1] < 16 or float(pixels.std(axis=(0, 1)).max()) < 8:
            raise ValueError("A UI template must contain distinctive visual structure")
        if not self.label or not isinstance(self.state, UIState):
            raise ValueError("A UI template requires a label and UIState")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Template threshold must be in [0,1]")
        if self.role not in {"state", "legitimate_ad_close", "restart"}:
            raise ValueError("Unknown UI template role")


@dataclass(frozen=True)
class TemplateMatch:
    label: str
    state: UIState
    role: str
    score: float
    accepted: bool
    bbox: tuple[int, int, int, int] | None


class TemplateRecognizer:
    def __init__(
        self,
        templates: Iterable[UITemplate],
        *,
        expected_size: tuple[int, int],
        threshold: float = 0.97,
        ambiguity_margin: float = 0.02,
    ):
        if len(expected_size) != 2 or any(type(v) is not int or v < 2 for v in expected_size):
            raise ValueError("Expected size must be a positive integer width/height pair")
        if not 0 <= threshold <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError("Recognizer thresholds must be in [0,1]")
        self.templates = tuple(templates)
        if len({t.label for t in self.templates}) != len(self.templates):
            raise ValueError("Template labels must be unique")
        self.expected_size = expected_size
        self.threshold, self.ambiguity_margin = threshold, ambiguity_margin
        probe = np.empty((expected_size[1], expected_size[0], 3), dtype=np.uint8)
        for template in self.templates:
            region, _, _ = _region(probe, template.roi)
            if any(a < b for a, b in zip(region.shape[:2], template.rgb.shape[:2])):
                raise ValueError(f"Template {template.label} does not fit its ROI")

    @classmethod
    def from_manifest(cls, path: str | Path):
        """Load local labeled crops; optional sha256 binds a crop to its label.

        Manifest fields: version=1, expected_size=[width,height], templates=[
        {label,state,file,roi:[left,top,right,bottom],threshold,role,sha256?}].
        Restart/advertisement controls must be independently labeled crops, not
        permissions inferred from a state label or a timer.
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("Unsupported UI template manifest version")
        templates = []
        for spec in payload["templates"]:
            image_path = (path.parent / spec["file"]).resolve()
            if not image_path.is_relative_to(path.parent.resolve()):
                raise ValueError("Template images must stay within the manifest directory")
            data = image_path.read_bytes()
            if spec.get("sha256") and hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError("Template image hash does not match its label manifest")
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            templates.append(
                UITemplate(
                    spec["label"],
                    UIState(spec["state"]),
                    pixels,
                    tuple(spec["roi"]),
                    spec.get("threshold", 0.98),
                    spec.get("role", "state"),
                )
            )
        return cls(
            templates,
            expected_size=tuple(payload["expected_size"]),
            threshold=payload.get("threshold", 0.97),
            ambiguity_margin=payload.get("ambiguity_margin", 0.02),
        )

    def matches(self, frame: np.ndarray) -> list[TemplateMatch]:
        frame = _rgb(frame)
        if (frame.shape[1], frame.shape[0]) != self.expected_size:
            return []
        cv2 = _cv2()
        result = []
        for template in self.templates:
            region, x, y = _region(frame, template.roi)
            distances = cv2.matchTemplate(region, template.rgb, cv2.TM_SQDIFF)
            minimum, _, location, _ = cv2.minMaxLoc(distances)
            score = float(np.clip(1 - np.sqrt(max(0, minimum) / template.rgb.size) / 255, 0, 1))
            left, top = x + location[0], y + location[1]
            box = (left, top, left + template.rgb.shape[1], top + template.rgb.shape[0])
            result.append(
                TemplateMatch(
                    template.label,
                    template.state,
                    template.role,
                    score,
                    score >= max(template.threshold, self.threshold),
                    box,
                )
            )
        return result

    def classify(self, frame: np.ndarray) -> StateEvidence:
        matches = self.matches(frame)
        by_state: dict[UIState, float] = {}
        for match in matches:
            if match.role == "state" and match.accepted:
                by_state[match.state] = max(by_state.get(match.state, 0), match.score)
        ranked = sorted(by_state.items(), key=lambda value: value[1], reverse=True)
        if not ranked:
            return StateEvidence(UIState.UNEXPECTED, 0.0)
        state, score = ranked[0]
        # Include near-threshold competing classes in ambiguity checks, too.
        competitors = [m.score for m in matches if m.role == "state" and m.state != state]
        if competitors and score - max(competitors) < self.ambiguity_margin:
            return StateEvidence(UIState.UNEXPECTED, 0.0)
        close = state == UIState.ADVERTISEMENT and any(
            m.accepted and m.state == state and m.role == "legitimate_ad_close" for m in matches
        )
        restart = state in {UIState.GAME_OVER, UIState.RESULT, UIState.RETURN} and any(
            m.accepted and m.state == state and m.role == "restart" for m in matches
        )
        return StateEvidence(state, score, close, restart)


@dataclass(frozen=True)
class ColorProfile:
    """HSV bounds use OpenCV hue 0..179 and saturation/value 0..255.

    Multiple ranges allow colors spanning hue zero. Define vehicle ROI to exclude
    the HUD; ground regions must connect to the bottom of their specified ROI.
    """

    expected_size: tuple[int, int]
    vehicle_roi: tuple[float, float, float, float]
    vehicle_hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    terrain_roi: tuple[float, float, float, float]
    terrain_hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    min_vehicle_pixels: int = 30
    max_vehicle_fraction: float = 0.20
    min_axis_ratio: float = 2.0


@dataclass(frozen=True)
class VehicleEstimate:
    valid: bool
    center_xy: tuple[float, float] | None
    pitch_mod_pi: float | None
    bbox: tuple[int, int, int, int] | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class TerrainEstimate:
    x_pixels: np.ndarray
    y_pixels: np.ndarray
    valid: np.ndarray
    coverage: float


def _color_mask(image, ranges):
    cv2 = _cv2()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not ranges:
        raise ValueError("Color segmentation requires explicitly calibrated HSV ranges")
    for lower, upper in ranges:
        low, high = np.asarray(lower), np.asarray(upper)
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or np.any(low < 0)
            or np.any(high > [179, 255, 255])
            or np.any(low > high)
        ):
            raise ValueError("Invalid OpenCV HSV bounds")
        mask |= cv2.inRange(hsv, low.astype(np.uint8), high.astype(np.uint8))
    return mask


class ColorGeometryEstimator:
    def __init__(self, profile: ColorProfile):
        self.profile = profile
        if (
            profile.min_vehicle_pixels < 3
            or not 0 < profile.max_vehicle_fraction <= 1
            or profile.min_axis_ratio <= 1
        ):
            raise ValueError("Invalid vehicle geometry limits")

    def vehicle(self, frame: np.ndarray) -> VehicleEstimate:
        frame = _rgb(frame)
        if (frame.shape[1], frame.shape[0]) != self.profile.expected_size:
            return VehicleEstimate(False, None, None, None, 0, "Resolution does not match profile")
        image, x0, y0 = _region(frame, self.profile.vehicle_roi)
        mask = _color_mask(image, self.profile.vehicle_hsv_ranges)
        cv2 = _cv2()
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return VehicleEstimate(False, None, None, None, 0, "No vehicle-colored component")
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[label, cv2.CC_STAT_AREA]
        if (
            not self.profile.min_vehicle_pixels
            <= area
            <= image.shape[0] * image.shape[1] * self.profile.max_vehicle_fraction
        ):
            return VehicleEstimate(
                False, None, None, None, 0, "Component area outside profile limits"
            )
        yy, xx = np.nonzero(labels == label)
        eigenvalues, vectors = np.linalg.eigh(np.cov(np.column_stack((xx, yy)), rowvar=False))
        ratio = eigenvalues[-1] / max(eigenvalues[0], 1e-9)
        if ratio < self.profile.min_axis_ratio:
            return VehicleEstimate(False, None, None, None, 0, "Vehicle axis is ambiguous")
        axis = vectors[:, -1]
        if axis[0] < 0:
            axis = -axis
        pitch = float(math.atan2(-axis[1], axis[0]))
        x, y, width, height, _ = stats[label]
        return VehicleEstimate(
            True,
            (float(centers[label, 0] + x0), float(centers[label, 1] + y0)),
            pitch,
            (int(x + x0), int(y + y0), int(x + x0 + width), int(y + y0 + height)),
            float(1 - 1 / ratio),
            "Principal color axis only; front/back and rollovers are ambiguous",
        )

    def terrain(self, frame: np.ndarray, *, samples: int = 32) -> TerrainEstimate:
        frame = _rgb(frame)
        if type(samples) is not int or samples < 2:
            raise ValueError("Terrain requires at least two samples")
        if (frame.shape[1], frame.shape[0]) != self.profile.expected_size:
            return TerrainEstimate(
                np.full(samples, np.nan),
                np.full(samples, np.nan),
                np.zeros(samples, dtype=bool),
                0.0,
            )
        image, x0, y0 = _region(frame, self.profile.terrain_roi)
        mask = _color_mask(image, self.profile.terrain_hsv_ranges) > 0
        xs = np.rint(np.linspace(0, image.shape[1] - 1, samples)).astype(int)
        ys, valid = np.full(samples, np.nan), np.zeros(samples, dtype=bool)
        for index, x in enumerate(xs):
            column = mask[:, max(0, x - 1) : min(mask.shape[1], x + 2)].mean(axis=1) >= 0.5
            run = int(np.cumprod(column[::-1]).sum())
            # All-ground columns have no observed sky/ground boundary.
            if 3 <= run < len(column):
                ys[index] = y0 + len(column) - run
                valid[index] = True
        return TerrainEstimate(xs.astype(float) + x0, ys, valid, float(valid.mean()))


def geometry_validation_metrics(predictions: list[VehicleEstimate], labels: list[dict]) -> dict:
    """Report coverage and errors for labeled images; invalid frames stay in denominator.

    Label fields: center_xy=[x,y], pitch_mod_pi=radians. Orientation errors are
    modulo pi because a principal axis cannot identify vehicle front versus back.
    """
    if len(predictions) != len(labels) or not labels:
        raise ValueError("Validation requires equally sized nonempty predictions and labels")
    center, angle = [], []
    for prediction, label in zip(predictions, labels):
        if prediction.valid:
            center.append(
                float(np.linalg.norm(np.asarray(prediction.center_xy) - label["center_xy"]))
            )
            delta = prediction.pitch_mod_pi - label["pitch_mod_pi"]
            angle.append(abs((delta + math.pi / 2) % math.pi - math.pi / 2))
    return {
        "labeled_frames": len(labels),
        "valid_frames": len(center),
        "coverage": len(center) / len(labels),
        "center_error_pixels_mean": float(np.mean(center)) if center else None,
        "pitch_error_radians_mean": float(np.mean(angle)) if angle else None,
    }


def state_validation_metrics(predictions: list[StateEvidence], labels: list[UIState]) -> dict:
    if len(predictions) != len(labels) or not labels:
        raise ValueError("Validation requires equally sized nonempty predictions and labels")
    confusion = {}
    for prediction, label in zip(predictions, labels):
        row = confusion.setdefault(label.value, {})
        row[prediction.state.value] = row.get(prediction.state.value, 0) + 1
    return {
        "labeled_frames": len(labels),
        "accuracy": sum(p.state == label for p, label in zip(predictions, labels)) / len(labels),
        "unknown_fraction": sum(p.state == UIState.UNEXPECTED for p in predictions) / len(labels),
        "confusion": confusion,
    }
