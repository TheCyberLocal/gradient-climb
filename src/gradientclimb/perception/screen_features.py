"""Masked screen-relative policy observations; no inferred world state or input authority.

This schema is deliberately different from the surrogate's privileged features.
An actor must be trained for this exact schema and mask/history order.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .measurements import PixelMeasurement

SCHEMA_VERSION = "hcr-screen-relative-1"
TERRAIN_OFFSETS = (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
FEATURE_NAMES = (
    "body_x_fraction",
    "body_y_fraction",
    "body_width_fraction",
    "body_height_fraction",
    "body_sin_2angle",
    "body_cos_2angle",
    "body_visual_support",
    "axle_x_fraction",
    "axle_y_fraction",
    "axle_width_fraction",
    "left_radius_axles",
    "right_radius_axles",
    "sin_2angle",
    "cos_2angle",
    "wheel_visual_support",
    "image_angular_rate_radians_per_second",
    "image_axle_vx_widths_per_second",
    "image_axle_vy_heights_per_second",
    "image_log_scale_rate_per_second",
    *(f"terrain_y_relative_axles_at_{offset:g}" for offset in TERRAIN_OFFSETS),
    "body_to_terrain_axles",
    "left_wheel_to_terrain_axles",
    "right_wheel_to_terrain_axles",
    "terrain_slope_image_right",
    "hud_displayed_progress_div_1000",
    "hud_progress_increment_div_10",
    "previous_os_gas",
    "previous_os_brake",
    "previous_os_state_age_seconds",
    "frame_interval_seconds",
    "episode_elapsed_div_60",
    "body_image_angular_rate_radians_per_second",
    "body_to_terrain_spans",
    "body_terrain_slope_image_right",
    *(f"body_terrain_y_relative_spans_at_{offset:g}" for offset in TERRAIN_OFFSETS),
)


@dataclass
class ScreenObservation:
    schema_id: str
    timestamp_ns: int
    values: np.ndarray
    valid: np.ndarray
    vector: np.ndarray
    measurement: PixelMeasurement
    hud: dict
    temporal_contiguous: bool
    geometry_valid: bool
    hud_increment: dict
    failures: list[str]

    def supports(self, names):
        """A caller declares actor-required features; this never grants UI authority."""
        names = tuple(names)
        if not names:
            raise ValueError("At least one required feature must be declared")
        return all(self.valid[FEATURE_NAMES.index(name)] for name in names)

    def as_dict(self):
        return {
            "schema_id": self.schema_id,
            "timestamp_ns": self.timestamp_ns,
            "feature_names": list(FEATURE_NAMES),
            "values": self.values.tolist(),
            "valid": self.valid.tolist(),
            "vector": self.vector.tolist(),
            "measurement": self.measurement.as_dict(),
            "hud": self.hud,
            "temporal_contiguous": self.temporal_contiguous,
            "geometry_valid": self.geometry_valid,
            "body_geometry_valid": bool(self.measurement.body.get("valid")),
            "hud_increment": self.hud_increment,
            "failures": self.failures,
            "input_authorized": False,
        }


def _finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _terrain_at(terrain, x):
    """Interpolate only between adjacent valid recorded samples; no gap filling."""
    points = terrain.get("points", [])
    if not terrain.get("valid"):
        return None
    for point in points:
        if point.get("valid") and abs(point["x"] - x) < 1e-6:
            return point["y"]
    for left, right in itertools.pairwise(points):
        if left["x"] <= x <= right["x"] and left.get("valid") and right.get("valid"):
            return float(np.interp(x, [left["x"], right["x"]], [left["y"], right["y"]]))
    return None


class ScreenFeatureBridge:
    """Observe RGB or update from measured evidence, preserving missingness and time.

    ``vector`` concatenates [values, validity bits] per frame, oldest first.
    Missing history is zero padded with false masks. ``geometry_valid`` requires
    body and independently supported wheels; callers release inputs on failure.
    UI state, foreground, freshness and action lease guards remain separate.
    """

    def __init__(self, measurer, hud_reader, *, history=4, max_interval_seconds=0.5):
        if type(history) is not int or not 1 <= history <= 16:
            raise ValueError("History must be an integer in 1..16")
        if not _finite(max_interval_seconds) or not 0.01 <= max_interval_seconds <= 2:
            raise ValueError("Maximum interval must be finite in .01..2 seconds")
        self.measurer, self.hud_reader = measurer, hud_reader
        self.history, self.max_interval_seconds = history, max_interval_seconds
        self.schema = {
            "version": SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "history": history,
            "order": "oldest first; values then validity per frame",
            "max_interval_seconds": max_interval_seconds,
            "world_scale": None,
            "orientation": "image axle angle modulo pi",
            "hud_semantics": "displayed maximum progress hypothesis; never signed current x",
        }
        self.schema_id = hashlib.sha256(
            json.dumps(self.schema, sort_keys=True).encode()
        ).hexdigest()
        self.reset()

    @property
    def observation_dim(self):
        return 2 * len(FEATURE_NAMES) * self.history

    def reset(self):
        """Call at an independently observed episode boundary, never infer reset from OCR."""
        self._previous = None
        self._maximum_progress = None
        self._history = deque(maxlen=self.history)

    def observe(self, rgb, timestamp_ns, **context):
        return self.update(
            self.measurer.measure(rgb), self.hud_reader.read(rgb), timestamp_ns, **context
        )

    def update(
        self,
        measurement: PixelMeasurement,
        hud: dict,
        timestamp_ns: int,
        *,
        previous_action_code=None,
        action_age_seconds=None,
        episode_elapsed_seconds=None,
    ) -> ScreenObservation:
        if type(timestamp_ns) is not int or timestamp_ns <= 0:
            raise ValueError("A positive integer capture timestamp is required")
        if previous_action_code is not None and (
            type(previous_action_code) is not int or previous_action_code not in range(4)
        ):
            raise ValueError("Previous actual OS state must be None or a code in 0..3")
        for value in (action_age_seconds, episode_elapsed_seconds):
            if value is not None and (not _finite(value) or value < 0):
                raise ValueError("Context durations must be finite and nonnegative")
        previous = self._previous
        dt = (timestamp_ns - previous[0]) / 1e9 if previous else None
        if dt is not None and dt <= 0:
            self.reset()
            raise ValueError("Nonmonotonic capture timestamp; history cleared")
        contiguous = dt is not None and 0.01 <= dt <= self.max_interval_seconds
        if previous and not contiguous:
            self._history.clear()
        values = np.zeros(len(FEATURE_NAMES), np.float32)
        valid = np.zeros(len(FEATURE_NAMES), bool)
        indices = {name: i for i, name in enumerate(FEATURE_NAMES)}

        def put(name, value):
            if _finite(value) and abs(value) < 1e6:
                index = indices[name]
                values[index], valid[index] = value, True

        width, height = measurement.image_size
        body, wheels, terrain = measurement.body, measurement.wheels, measurement.terrain
        failures = [
            part.get("reason", "Invalid visual component")
            for part in (body, wheels, terrain)
            if not part.get("valid")
        ]
        if body.get("valid"):
            x, y = body["center_xy"]
            left, top, right, bottom = body["bbox"]
            for name, value in zip(
                FEATURE_NAMES[:7],
                (
                    x / width,
                    y / height,
                    (right - left) / width,
                    (bottom - top) / height,
                    math.sin(2 * body["axis_mod_pi_radians"]),
                    math.cos(2 * body["axis_mod_pi_radians"]),
                    body["quality"],
                ),
                strict=True,
            ):
                put(name, value)
            body_span = max(right - left, bottom - top)
            for offset in TERRAIN_OFFSETS:
                surface = _terrain_at(terrain, x + offset * body_span)
                if surface is not None:
                    put(f"body_terrain_y_relative_spans_at_{offset:g}", (surface - y) / body_span)
            near, ahead = _terrain_at(terrain, x), _terrain_at(terrain, x + body_span)
            if near is not None:
                put("body_to_terrain_spans", (near - y) / body_span)
            if near is not None and ahead is not None:
                put("body_terrain_slope_image_right", (ahead - near) / body_span)
            old_body = previous[1].body if contiguous else {}
            if old_body.get("valid"):
                difference = body["axis_mod_pi_radians"] - old_body["axis_mod_pi_radians"]
                change = 0.5 * math.atan2(math.sin(2 * difference), math.cos(2 * difference))
                if abs(change) < math.pi / 3:
                    put("body_image_angular_rate_radians_per_second", change / dt)
        if wheels.get("valid"):
            x, y = wheels["center_xy"]
            axle = wheels["axle_length_pixels"]
            angle = wheels["pitch_mod_pi_radians"]
            for name, value in zip(
                FEATURE_NAMES[7:15],
                (
                    x / width,
                    y / height,
                    axle / width,
                    wheels["left"]["radius_pixels"] / axle,
                    wheels["right"]["radius_pixels"] / axle,
                    math.sin(2 * angle),
                    math.cos(2 * angle),
                    wheels["quality"],
                ),
                strict=True,
            ):
                put(name, value)
            old = previous[1].wheels if contiguous else {}
            if old.get("valid"):
                delta = 0.5 * math.atan2(
                    math.sin(2 * (angle - old["pitch_mod_pi_radians"])),
                    math.cos(2 * (angle - old["pitch_mod_pi_radians"])),
                )
                # A large modulo-pi jump is aliased; no invented rotation direction.
                if abs(delta) < math.pi / 3:
                    put("image_angular_rate_radians_per_second", delta / dt)
                else:
                    failures.append("Image angle change ambiguous across interval")
                put("image_axle_vx_widths_per_second", (x - old["center_xy"][0]) / width / dt)
                put("image_axle_vy_heights_per_second", (y - old["center_xy"][1]) / height / dt)
                put(
                    "image_log_scale_rate_per_second",
                    math.log(axle / old["axle_length_pixels"]) / dt,
                )
            for offset in TERRAIN_OFFSETS:
                surface = _terrain_at(terrain, x + offset * axle)
                if surface is not None:
                    put(f"terrain_y_relative_axles_at_{offset:g}", (surface - y) / axle)
            if body.get("valid"):
                surface = _terrain_at(terrain, body["center_xy"][0])
                if surface is not None:
                    put("body_to_terrain_axles", (surface - body["center_xy"][1]) / axle)
            for side in ("left", "right"):
                wheel = wheels[side]
                surface = _terrain_at(terrain, wheel["center_xy"][0])
                if surface is not None:
                    put(
                        f"{side}_wheel_to_terrain_axles",
                        (surface - wheel["center_xy"][1] - wheel["radius_pixels"]) / axle,
                    )
            near, ahead = _terrain_at(terrain, x), _terrain_at(terrain, x + axle)
            if near is not None and ahead is not None:
                put("terrain_slope_image_right", (ahead - near) / axle)
        progress = hud.get("hud_displayed_progress_meters") if hud.get("valid") else None
        if (
            _finite(progress)
            and self._maximum_progress is not None
            and progress < self._maximum_progress
        ):
            failures.append("HUD decreased: OCR error or unannounced reset; increment invalid")
            progress = None
        if _finite(progress) and progress >= 0:
            put("hud_displayed_progress_div_1000", progress / 1000)
            self._maximum_progress = progress
        else:
            progress = None
            failures.append(hud.get("reason", "Unknown HUD progress"))
        increment = {
            "valid": False,
            "displayed_meters": None,
            "interval_seconds": dt,
            "reason": "Requires consecutive accepted monotonic HUD readings",
        }
        old_progress = previous[2] if contiguous else None
        if progress is not None and old_progress is not None:
            delta = progress - old_progress
            if delta >= 0:
                put("hud_progress_increment_div_10", delta / 10)
                increment.update(
                    valid=True,
                    displayed_meters=delta,
                    reason="Displayed progress increment; not velocity",
                )
            else:
                failures.append("HUD decreased: OCR error or unannounced reset; increment invalid")
        if previous_action_code is not None:
            put("previous_os_gas", previous_action_code & 1)
            put("previous_os_brake", (previous_action_code >> 1) & 1)
            put("previous_os_state_age_seconds", action_age_seconds)
        if contiguous:
            put("frame_interval_seconds", dt)
        if episode_elapsed_seconds is not None:
            put("episode_elapsed_div_60", episode_elapsed_seconds / 60)
        self._history.append(np.concatenate((values, valid.astype(np.float32))))
        frame_dim = 2 * len(FEATURE_NAMES)
        padding = [np.zeros(frame_dim, np.float32)] * (self.history - len(self._history))
        vector = np.concatenate([*padding, *self._history])
        geometry_valid = bool(body.get("valid") and wheels.get("valid"))
        self._previous = (timestamp_ns, measurement, progress)
        return ScreenObservation(
            self.schema_id,
            timestamp_ns,
            values,
            valid,
            vector,
            measurement,
            hud,
            contiguous,
            geometry_valid,
            increment,
            failures,
        )
