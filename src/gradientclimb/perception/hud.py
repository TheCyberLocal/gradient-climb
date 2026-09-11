"""Conservative local glyph OCR for a declared HUD region; no pretrained OCR model.

Templates must come from explicitly labeled local number regions. A complete 0..9
alphabet is required for accepted readings so an unseen nine is not silently read
as a known six. Displayed progress is not assumed to be signed vehicle position.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .pixels import _cv2, _region, _rgb


def segment_digits(frame: np.ndarray, roi) -> list[np.ndarray]:
    """Segment bright neutral glyph interiors; reject clutter instead of guessing.

    The taller digit components exclude a smaller trailing 'm'. Appearance changes
    and overlays can defeat this heuristic and require independent negative labels.
    """
    cv2 = _cv2()
    image, _, _ = _region(_rgb(frame), roi)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    neutral = hsv[..., 1] <= 75
    if not neutral.any():
        return []
    peak = int(hsv[..., 2][neutral].max())
    if peak < 35:
        return []
    mask = np.uint8(neutral & (hsv[..., 2] >= max(30, peak * 0.65))) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [
        i
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= 8
        and stats[i, cv2.CC_STAT_HEIGHT] >= 6
        and stats[i, cv2.CC_STAT_WIDTH] <= stats[i, cv2.CC_STAT_HEIGHT]
        and stats[i, cv2.CC_STAT_AREA]
        / (stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT])
        >= 0.30
    ]
    if not components:
        return []
    height = max(stats[i, cv2.CC_STAT_HEIGHT] for i in components)
    components = [i for i in components if stats[i, cv2.CC_STAT_HEIGHT] >= height * 0.72]
    components.sort(key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    if not 1 <= len(components) <= 7:
        return []
    result = []
    for i in components:
        x, y, width, h, _ = stats[i]
        if width > height or width < 2 or h >= image.shape[0] - 1:
            return []
        component = np.uint8(labels[y : y + h, x : x + width] == i) * 255
        # Preserve aspect ratio; stretch would make narrow digits look like wide ones.
        ratio = min(18 / width, 30 / h)
        resized = cv2.resize(
            component,
            (max(1, round(width * ratio)), max(1, round(h * ratio))),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = np.zeros((32, 20), np.uint8)
        oy, ox = (32 - resized.shape[0]) // 2, (20 - resized.shape[1]) // 2
        canvas[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
        result.append(canvas)
    return result


class HUDDigitReader:
    def __init__(
        self,
        *,
        expected_size: tuple[int, int],
        roi: tuple[float, float, float, float],
        threshold=0.84,
        ambiguity_margin=0.04,
    ):
        if len(expected_size) != 2 or any(type(v) is not int or v < 2 for v in expected_size):
            raise ValueError("Exact analyzed frame dimensions are required")
        if not 0 < threshold <= 1 or not 0 < ambiguity_margin < 1:
            raise ValueError("Invalid glyph thresholds")
        self.expected_size, self.roi = tuple(expected_size), tuple(roi)
        _region(np.empty((expected_size[1], expected_size[0], 3), np.uint8), roi)
        self.threshold, self.ambiguity_margin = threshold, ambiguity_margin
        self.glyphs: list[dict] = []

    def add_labeled_region(
        self, frame: np.ndarray, text: str, roi, *, source_sha256: str, annotation_id: str
    ):
        if not text or any(char not in "0123456789" for char in text):
            raise ValueError(
                "A region label must contain only its visible digits, without unit suffix"
            )
        if (
            len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)
            or not annotation_id
        ):
            raise ValueError("Template training requires a source hash and annotation ID")
        segments = segment_digits(frame, roi)
        if len(segments) != len(text):
            raise ValueError(
                f"Segmented {len(segments)} glyphs but manual label has {len(text)}; inspect the region"
            )
        self.glyphs.extend(
            {
                "digit": digit,
                "pixels": pixels,
                "source_sha256": source_sha256,
                "annotation_id": annotation_id,
            }
            for digit, pixels in zip(text, segments)
        )

    def read(self, frame: np.ndarray) -> dict:
        frame = _rgb(frame)
        invalid = {"valid": False, "hud_displayed_progress_meters": None, "quality": 0.0}
        if (frame.shape[1], frame.shape[0]) != self.expected_size:
            return {**invalid, "reason": "HUD profile geometry mismatch"}
        missing = sorted(set("0123456789") - {g["digit"] for g in self.glyphs})
        if missing:
            return {
                **invalid,
                "reason": "Incomplete independently labeled digit alphabet",
                "missing_digits": missing,
            }
        segments = segment_digits(frame, self.roi)
        if not segments:
            return {**invalid, "reason": "HUD digit segmentation unavailable or cluttered"}
        digits, scores = [], []
        for segment in segments:
            by_digit = {}
            a = segment > 0
            for glyph in self.glyphs:
                b = glyph["pixels"] > 0
                score = float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))
                by_digit[glyph["digit"]] = max(by_digit.get(glyph["digit"], 0), score)
            ranked = sorted(by_digit.items(), key=lambda item: item[1], reverse=True)
            digit, score = ranked[0]
            if score < self.threshold or score - ranked[1][1] < self.ambiguity_margin:
                return {
                    **invalid,
                    "reason": "Unrecognized or ambiguous HUD glyph",
                    "best_similarity": score,
                }
            digits.append(digit)
            scores.append(score)
        text = "".join(digits)
        return {
            "valid": True,
            "text": text,
            "hud_displayed_progress_meters": int(text),
            "quality": min(scores),
            "glyph_similarities": scores,
            "reason": "HUD display reading; not signed current position, world velocity, or UI authorization",
        }

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "version": 1,
            "expected_size": self.expected_size,
            "roi": self.roi,
            "threshold": self.threshold,
            "ambiguity_margin": self.ambiguity_margin,
            "glyphs": [],
        }
        for i, glyph in enumerate(self.glyphs):
            path = directory / f"glyph-{i:03d}-{glyph['digit']}.png"
            Image.fromarray(glyph["pixels"]).save(path)
            manifest["glyphs"].append(
                {
                    "digit": glyph["digit"],
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_sha256": glyph["source_sha256"],
                    "annotation_id": glyph["annotation_id"],
                }
            )
        path = directory / "hud-glyphs.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_manifest(cls, path: str | Path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != 1:
            raise ValueError("Unsupported HUD glyph manifest")
        reader = cls(
            expected_size=tuple(data["expected_size"]),
            roi=tuple(data["roi"]),
            threshold=data["threshold"],
            ambiguity_margin=data["ambiguity_margin"],
        )
        for row in data["glyphs"]:
            file = (path.parent / row["file"]).resolve()
            if (
                not file.is_relative_to(path.parent)
                or hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise ValueError("Glyph file is outside manifest directory or its hash changed")
            with Image.open(file) as image:
                pixels = np.asarray(image.convert("L")).copy()
            if (
                pixels.shape != (32, 20)
                or row["digit"] not in "0123456789"
                or len(row["digit"]) != 1
            ):
                raise ValueError("Invalid normalized digit glyph")
            reader.glyphs.append({**row, "pixels": pixels})
        return reader
