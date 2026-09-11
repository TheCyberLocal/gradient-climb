"""Declared HUD field readers for the coin counter and the fuel gauge.

Each reader names its field, its normalized region and its semantics explicitly.
They classify no UI state and authorize no input; the caller supplies independent
state confirmation. Unknown values stay ``None``. A coin counter is lifetime
currency: only the boundary difference of one episode is an episode score gain.
"""

from __future__ import annotations

import math

import numpy as np

from .hud import HUDDigitReader
from .pixels import _cv2, _region, _rgb

COIN_FIELD_VERSION = "hud-coin-counter-2.0"
FUEL_GAUGE_VERSION = "hud-fuel-gauge-2.0"
# Normalized 1034x581 wrapper frame coordinates measured on stored gameplay frames.
COIN_FIELD_ROI = (118 / 1034, 82 / 581, 250 / 1034, 128 / 581)
FUEL_GAUGE_BOX = (128, 46, 270, 70)
FUEL_ICON_BOX = (74, 42, 118, 76)


class CoinCounterReader:
    """Read the top-left coin counter with the frozen gameplay digit bank.

    The counter shows lifetime currency with a thousands space that the segmenter
    ignores. It is readable only on an unobstructed playing frame; paused, result
    and menu frames dim or move it and must not be read as gameplay currency.
    """

    field = "hud_coin_counter"
    version = COIN_FIELD_VERSION

    def __init__(self, reader: HUDDigitReader, *, roi=COIN_FIELD_ROI):
        if not isinstance(reader, HUDDigitReader) or reader.expected_size != (1034, 581):
            raise ValueError("Coin reading requires the declared 1034x581 native HUD profile")
        self.reader = HUDDigitReader(
            expected_size=reader.expected_size,
            roi=roi,
            threshold=reader.threshold,
            ambiguity_margin=reader.ambiguity_margin,
            alignment_pixels=reader.alignment_pixels,
        )
        self.reader.glyphs = list(reader.glyphs)

    @classmethod
    def from_gameplay_manifest(cls, path):
        return cls(HUDDigitReader.from_manifest(path))

    def read(self, frame, *, playing_state_confirmed=False) -> dict:
        invalid = {
            "valid": False,
            "coins": None,
            "quality": 0.0,
            "field": self.field,
            "reader_version": self.version,
            "semantics": "lifetime currency display; episode score is a boundary difference",
        }
        if playing_state_confirmed is not True:
            return {**invalid, "reason": "Independent playing state confirmation required"}
        numeric = self.reader.read(frame)
        if not numeric["valid"]:
            return {**invalid, "numeric_evidence": numeric, "reason": numeric["reason"]}
        return {
            **invalid,
            "valid": True,
            "coins": numeric["hud_displayed_progress_meters"],
            "quality": numeric["quality"],
            "numeric_evidence": numeric,
            "reason": "Coin counter reading; lifetime balance, not an episode score",
        }


class FuelGaugeReader:
    """Displayed fuel-gauge fill fraction from the saturated bar inside its frame.

    The value is a screen fraction of the bar's inner width, not fuel units. The
    bar changes hue from green through yellow to red as it empties, so any
    saturated bright hue counts as fill; the gauge frame itself is dark. Fewer
    than a minimum number of filled rows, or no dark frame, yields unknown.
    """

    field = "hud_fuel_gauge"
    version = FUEL_GAUGE_VERSION

    def __init__(
        self,
        *,
        expected_size=(1034, 581),
        box=FUEL_GAUGE_BOX,
        icon_box=FUEL_ICON_BOX,
        minimum_rows=8,
        minimum_icon_pixels=200,
        saturation=120,
        value=150,
    ):
        if len(expected_size) != 2 or any(type(v) is not int or v < 2 for v in expected_size):
            raise ValueError("Exact frame dimensions are required")
        width, height = expected_size
        for name, (left, top, right, bottom) in (("Gauge", box), ("Icon", icon_box)):
            if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
                raise ValueError(f"{name} box must stay inside the frame")
        for name, count in (
            ("minimum_rows", minimum_rows),
            ("minimum_icon_pixels", minimum_icon_pixels),
        ):
            if type(count) is not int or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, threshold in (("saturation", saturation), ("value", value)):
            if not math.isfinite(threshold) or not 0 <= threshold <= 255:
                raise ValueError(f"{name} threshold must lie in 0..255")
        self.expected_size, self.box, self.icon_box = (
            tuple(expected_size),
            tuple(box),
            tuple(icon_box),
        )
        self.minimum_rows, self.minimum_icon_pixels = minimum_rows, minimum_icon_pixels
        self.saturation, self.value = saturation, value

    def read(self, frame, *, playing_state_confirmed=False) -> dict:
        invalid = {
            "valid": False,
            "fill_fraction": None,
            "field": self.field,
            "reader_version": self.version,
            "semantics": "displayed gauge fill fraction of the inner bar width; not fuel units",
        }
        if playing_state_confirmed is not True:
            return {**invalid, "reason": "Independent playing state confirmation required"}
        frame = _rgb(frame)
        if (frame.shape[1], frame.shape[0]) != self.expected_size:
            return {**invalid, "reason": "Gauge profile geometry mismatch"}
        left, top, right, bottom = self.box
        cv2 = _cv2()
        # The red fuel-can icon left of the bar anchors the gauge; menus and dimmed
        # modals lack it, so an all-dark bar there never reads as an empty tank.
        il, it, ir, ib = self.icon_box
        icon = cv2.cvtColor(frame[it:ib, il:ir], cv2.COLOR_RGB2HSV)
        red = ((icon[..., 0] <= 8) | (icon[..., 0] >= 170)) & (icon[..., 1] > 120)
        red &= icon[..., 2] > 120
        if int(red.sum()) < self.minimum_icon_pixels:
            return {**invalid, "reason": "Fuel icon anchor not visible; gauge absent or dimmed"}
        # The gauge frame is a dark outline just outside the inner bar on both sides.
        edges = frame[top:bottom, max(0, left - 4) : left], frame[top:bottom, right - 1 : right + 3]
        if any(
            edge.size == 0 or (edge.max(axis=2) < 80).any(axis=1).mean() < 0.6 for edge in edges
        ):
            return {**invalid, "reason": "Gauge frame edges not visible; bar occluded or absent"}
        crop = frame[top:bottom, left:right]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        fill = (hsv[..., 1] >= self.saturation) & (hsv[..., 2] >= self.value)
        columns = fill.sum(axis=0) >= self.minimum_rows
        filled = int(columns.sum())
        total = right - left
        if filled:
            start = int(np.argmax(columns))
            run = int(np.argmin(columns[start:])) if not columns[start:].all() else total - start
            if start > 3:
                return {**invalid, "reason": "Fill does not start at the gauge origin"}
            if run != filled:
                return {**invalid, "reason": "Fill is not a single contiguous run from the origin"}
        return {
            **invalid,
            "valid": True,
            "fill_fraction": filled / total,
            "filled_columns": filled,
            "total_columns": total,
            "hue_median": int(np.median(hsv[..., 0][fill])) if fill.any() else None,
            "reason": "Displayed fuel gauge fraction; empty bar reads zero only with a visible frame",
        }


def crop_field(frame, roi):
    """Expose the exact analyzed crop for annotation tooling."""
    return _region(_rgb(frame), roi)[0]
