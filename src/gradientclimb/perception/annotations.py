"""Independent pixel labels and evaluation; no images or synthetic ground truth bundled."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PointLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    uncertainty_pixels: float = Field(default=2, ge=0)


class FrameAnnotation(BaseModel):
    """Null targets mean unannotated, never a zero-valued target or a failure label."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["pixel-annotation-1"] = "pixel-annotation-1"
    annotation_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_id: str = Field(min_length=1)
    split: Literal["train", "heldout"]
    evidence_domain: Literal["real_game", "synthetic"]
    width: int = Field(gt=1)
    height: int = Field(gt=1)
    timestamp_ns: int | None = Field(default=None, ge=0)
    annotator: str = Field(min_length=1)
    annotation_method: Literal["manual_independent", "synthetic_known_transform"]
    viewed_model_prediction: bool = False
    wheel_pair_visible: bool | None = None
    left_wheel: PointLabel | None = None
    right_wheel: PointLabel | None = None
    terrain_surface: Literal["visible_turf_soil_boundary"] = "visible_turf_soil_boundary"
    terrain_points: list[PointLabel] = Field(default_factory=list)
    previous_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    terrain_translation_xy: tuple[float, float] | None = None
    translation_uncertainty_pixels: float | None = Field(default=None, ge=0)
    notes: str = ""

    @model_validator(mode="after")
    def coherent(self):
        if self.split == "heldout" and self.viewed_model_prediction:
            raise ValueError("Held-out labels must be made without viewing predictions")
        if self.evidence_domain == "real_game" and self.annotation_method != "manual_independent":
            raise ValueError("Known synthetic transforms are not real-game labels")
        if (self.left_wheel is None) != (self.right_wheel is None):
            raise ValueError("Label both wheel centers or leave both unannotated")
        if self.left_wheel is not None and self.wheel_pair_visible is not True:
            raise ValueError("Wheel center labels require an explicitly visible pair")
        if self.terrain_translation_xy is not None and not self.previous_source_sha256:
            raise ValueError("Camera labels must identify the previous source frame")
        for point in [self.left_wheel, self.right_wheel, *self.terrain_points]:
            if point and not (point.x < self.width and point.y < self.height):
                raise ValueError("Annotation point is outside the source frame")
        return self


def validate_partition(labels: list[FrameAnnotation]) -> None:
    """Require independent sessions and original source hashes across splits."""
    if not labels:
        raise ValueError("No labels supplied")
    ids = [item.annotation_id for item in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation IDs")
    for key in ("session_id", "source_sha256"):
        train = {getattr(item, key) for item in labels if item.split == "train"}
        heldout = {getattr(item, key) for item in labels if item.split == "heldout"}
        if train & heldout:
            raise ValueError(f"Training/held-out {key} overlap")
    train_sources = {
        value
        for item in labels
        if item.split == "train"
        for value in (item.source_sha256, item.previous_source_sha256)
        if value
    }
    heldout_sources = {
        value
        for item in labels
        if item.split == "heldout"
        for value in (item.source_sha256, item.previous_source_sha256)
        if value
    }
    if train_sources & heldout_sources:
        raise ValueError("Training/held-out camera source frame overlap")


def evaluate_measurements(
    predictions: dict[str, dict], labels: list[FrameAnnotation], *, split="heldout"
) -> dict:
    """Predictions keyed by source SHA; report missing outputs in coverage denominator.

    Predictive confidence is not calibrated by these metrics. Human label uncertainty
    is reported separately and is never subtracted from the observed error.
    """
    validate_partition(labels)
    selected = [item for item in labels if item.split == split]
    if not selected:
        raise ValueError(f"No {split} annotations; accuracy is not measured")
    if len({item.evidence_domain for item in selected}) != 1:
        raise ValueError("Evaluate synthetic and real-game labels separately")
    if len({item.source_sha256 for item in selected}) != len(selected):
        raise ValueError("Duplicate source frames would overweight evaluation")
    wheel_eligible = wheel_valid = visible_labeled = false_wheel_pair = 0
    terrain_eligible = camera_eligible = 0
    wheel_errors, pitch_errors, terrain_errors, camera_errors, uncertainty = [], [], [], [], []
    for label in selected:
        prediction = predictions.get(label.source_sha256, {})
        measurement = prediction.get("measurement", prediction)
        if measurement and tuple(measurement.get("image_size", ())) != (label.width, label.height):
            raise ValueError("Prediction and annotation coordinate dimensions differ")
        wheels = measurement.get("wheels", {})
        if label.wheel_pair_visible is not None:
            visible_labeled += 1
            false_wheel_pair += int(wheels.get("valid", False) and not label.wheel_pair_visible)
        if label.left_wheel is not None:
            wheel_eligible += 1
            uncertainty.extend(
                [label.left_wheel.uncertainty_pixels, label.right_wheel.uncertainty_pixels]
            )
            if wheels.get("valid"):
                wheel_valid += 1
                truth = np.asarray(
                    [
                        [label.left_wheel.x, label.left_wheel.y],
                        [label.right_wheel.x, label.right_wheel.y],
                    ]
                )
                estimated = np.asarray([wheels["left"]["center_xy"], wheels["right"]["center_xy"]])
                # Left/right describe image ordering; never silently assume front/rear identity.
                truth, estimated = (
                    truth[np.argsort(truth[:, 0])],
                    estimated[np.argsort(estimated[:, 0])],
                )
                wheel_errors.extend(np.linalg.norm(estimated - truth, axis=1).tolist())
                delta = truth[1] - truth[0]
                angle = math.atan2(-delta[1], delta[0])
                pitch_errors.append(
                    abs(
                        (wheels["pitch_mod_pi_radians"] - angle + math.pi / 2) % math.pi
                        - math.pi / 2
                    )
                )
        points = measurement.get("terrain", {}).get("points", [])
        for point in label.terrain_points:
            terrain_eligible += 1
            # Require adjacent valid samples: never interpolate across an invalid gap.
            for left, right in pairwise(points):
                if left.get("valid") and right.get("valid") and left["x"] <= point.x <= right["x"]:
                    predicted_y = np.interp(
                        point.x, [left["x"], right["x"]], [left["y"], right["y"]]
                    )
                    terrain_errors.append(abs(float(predicted_y) - point.y))
                    break
        if label.terrain_translation_xy is not None:
            camera_eligible += 1
            camera = prediction.get("camera", {})
            if (
                camera.get("valid")
                and prediction.get("previous_source_sha256") == label.previous_source_sha256
            ):
                camera_errors.append(
                    float(
                        np.linalg.norm(
                            np.asarray(camera["terrain_screen_translation_xy"])
                            - label.terrain_translation_xy
                        )
                    )
                )

    def stats(values):
        return {
            "mean": float(np.mean(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }

    return {
        "protocol": f"{split} independent pixel-label evaluation",
        "split": split,
        "source_frames": len(selected),
        "sessions": len({item.session_id for item in selected}),
        "evidence_domains": sorted({item.evidence_domain for item in selected}),
        "wheel_labeled_frames": wheel_eligible,
        "wheel_valid_frames": wheel_valid,
        "wheel_coverage": wheel_valid / wheel_eligible if wheel_eligible else None,
        "wheel_center_error_pixels": stats(wheel_errors),
        "axle_pitch_error_radians": stats(pitch_errors),
        "visibility_labeled_frames": visible_labeled,
        "false_wheel_pair_detections": false_wheel_pair,
        "terrain_labeled_points": terrain_eligible,
        "terrain_valid_points": len(terrain_errors),
        "terrain_coverage": len(terrain_errors) / terrain_eligible if terrain_eligible else None,
        "terrain_y_error_pixels": stats(terrain_errors),
        "camera_labeled_intervals": camera_eligible,
        "camera_valid_intervals": len(camera_errors),
        "camera_translation_error_pixels": stats(camera_errors),
        "label_wheel_uncertainty_pixels": stats(uncertainty),
        "accuracy_claim": "Only the declared split and labels; no world-scale or UI validation",
    }
