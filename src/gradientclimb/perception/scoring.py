"""Read the right-side result DISTANCE field only, with explicit screen evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .hud import HUDDigitReader
from .pixels import _cv2, _region, _rgb


def _white_text(image):
    hsv = _cv2().cvtColor(image, _cv2().COLOR_RGB2HSV)
    return (hsv[..., 1] <= 75) & (hsv[..., 2] >= 180)


class ResultDistanceReader:
    """A result-only numeric field; it does not classify UI state or click anything."""

    def __init__(self, reader, anchor, anchor_roi, *, anchor_threshold=0.90):
        if not isinstance(reader, HUDDigitReader) or not 0 < anchor_threshold <= 1:
            raise ValueError("A HUD reader and valid label threshold are required")
        anchor = np.asarray(anchor)
        if anchor.ndim != 2 or anchor.dtype != bool or anchor.sum() < 20:
            raise ValueError("A labeled binary DISTANCE text anchor is required")
        self.reader, self.anchor = reader, anchor
        self.anchor_roi, self.anchor_threshold = tuple(anchor_roi), anchor_threshold

    def read(self, frame, *, result_state_confirmed=False):
        invalid = {
            "valid": False,
            "distance_meters": None,
            "quality": 0.0,
            "field": "right_side_result_distance",
        }
        if result_state_confirmed is not True:
            return {**invalid, "reason": "Independent result screen confirmation required"}
        frame = _rgb(frame)
        if (frame.shape[1], frame.shape[0]) != self.reader.expected_size:
            return {**invalid, "reason": "Result profile geometry mismatch"}
        crop, _, _ = _region(frame, self.anchor_roi)
        mask = _white_text(crop)
        if mask.shape != self.anchor.shape:
            return {**invalid, "reason": "Result label anchor geometry mismatch"}
        score = float(2 * (mask & self.anchor).sum() / max(1, mask.sum() + self.anchor.sum()))
        if score < self.anchor_threshold:
            return {
                **invalid,
                "reason": "Right-side DISTANCE label not supported",
                "anchor_similarity": score,
            }
        numeric = self.reader.read(frame)
        return {
            "valid": numeric["valid"],
            "distance_meters": numeric.get("hud_displayed_progress_meters"),
            "quality": min(score, numeric["quality"]),
            "anchor_similarity": score,
            "field": "right_side_result_distance",
            "numeric_evidence": numeric,
            "reason": numeric["reason"],
        }

    @classmethod
    def from_manifest(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            raise ValueError("Unsupported result distance manifest")
        paths = {}
        for key in ("numeric_manifest", "anchor"):
            target = (path.parent / data[key]["file"]).resolve()
            if (
                not target.is_relative_to(path.parent)
                or hashlib.sha256(target.read_bytes()).hexdigest() != data[key]["sha256"]
            ):
                raise ValueError("Result reader artifact path/hash mismatch")
            paths[key] = target
        with Image.open(paths["anchor"]) as image:
            anchor = np.asarray(image.convert("L")) > 0
        return cls(
            HUDDigitReader.from_manifest(paths["numeric_manifest"]),
            anchor,
            data["anchor_roi"],
            anchor_threshold=data["anchor_threshold"],
        )


class PausedDistanceReader:
    """Score at an explicitly verified paused boundary, including pause-request delay.

    The narrower numeric field excludes the bright pause-modal border, which can
    dominate the adaptive threshold applied to the dimmed background digits.
    It is profile-specific; unknown or occluded digits remain unknown.
    """

    def __init__(self, reader):
        if not isinstance(reader, HUDDigitReader) or reader.expected_size != (1034, 581):
            raise ValueError("Paused readout requires the declared 1034x581 native HUD profile")
        self.reader = HUDDigitReader(
            expected_size=reader.expected_size,
            roi=(300 / 1034, 68 / 581, 417 / 1034, 110 / 581),
            threshold=reader.threshold,
            ambiguity_margin=reader.ambiguity_margin,
            alignment_pixels=reader.alignment_pixels,
        )
        self.reader.glyphs = list(reader.glyphs)

    @classmethod
    def from_gameplay_manifest(cls, path):
        return cls(HUDDigitReader.from_manifest(path))

    def read(self, frame, *, paused_state_confirmed=False):
        invalid = {
            "valid": False,
            "distance_meters": None,
            "quality": 0.0,
            "field": "paused_gameplay_displayed_progress",
            "score_boundary": "verified_paused_frame",
        }
        if paused_state_confirmed is not True:
            return {**invalid, "reason": "Independent paused state confirmation required"}
        numeric = self.reader.read(frame)
        return {
            **invalid,
            "valid": numeric["valid"],
            "distance_meters": numeric.get("hud_displayed_progress_meters"),
            "quality": numeric["quality"],
            "numeric_evidence": numeric,
            "reason": "Displayed progress at verified pause; includes release-to-pause delay"
            if numeric["valid"]
            else numeric["reason"],
        }
