"""Declared screen-pixel geometry for construction pose replay, never physics truth.

Reviewed labels need a frozen protocol and source-image reference. An annotation
attests visible geometry; its hash does not establish accuracy. Missing geometry
stays absent. No vehicle/map names, collision surface, metres or dynamics follow.
"""

from __future__ import annotations

import math
from datetime import datetime
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, model_validator

from gradientclimb.datasets.demonstrations import EvidenceRef
from gradientclimb.environments.profiles import ImmutableRecord

Point = tuple[FiniteFloat, FiniteFloat]
Positive = Annotated[FiniteFloat, Field(gt=0)]
Uncertainty = Annotated[FiniteFloat, Field(ge=0)]
Name = Annotated[str, Field(min_length=1)]


class BodyOutline(ImmutableRecord):
    vertices: tuple[Point, ...] = Field(min_length=3, max_length=64)
    uncertainty_pixels: Uncertainty
    note: Name

    @model_validator(mode="after")
    def nondegenerate(self) -> Self:
        area = sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(self.vertices, (*self.vertices[1:], self.vertices[0]))
        )
        if abs(area) < 1:
            raise ValueError("Body outline must have nonzero pixel area")
        return self


class WheelOutline(ImmutableRecord):
    wheel_id: Name
    center: Point
    radius_pixels: Positive
    uncertainty_pixels: Uncertainty
    note: Name


class TerrainOutline(ImmutableRecord):
    """One visible strip; separate strips never imply terrain across their gap."""

    points: tuple[Point, ...] = Field(min_length=2, max_length=256)
    uncertainty_pixels: Uncertainty
    note: Name

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if any(a[0] >= b[0] for a, b in zip(self.points, self.points[1:])):
            raise ValueError("A terrain strip requires strictly increasing screen x")
        return self


class DirectedLandmark(ImmutableRecord):
    """An explicitly identified visible feature, not an inferred orientation label."""

    role: Literal["front", "rear", "roof", "head"]
    point: Point
    uncertainty_pixels: Uncertainty
    note: Name


class ScreenGeometry(ImmutableRecord):
    schema_version: Literal["screen-geometry-labels-3.0"] = "screen-geometry-labels-3.0"
    label_id: Name
    coordinate_system: Literal["screen_pixels_x_right_y_down"] = "screen_pixels_x_right_y_down"
    image_size: tuple[Annotated[int, Field(ge=32, le=2048, strict=True)], ...] = Field(
        min_length=2, max_length=2
    )
    provenance: Literal["synthetic_fixture", "reviewed_construction"]
    source_image: EvidenceRef | None = None
    label_protocol: EvidenceRef | None = None
    reviewer: Name | None = None
    reviewed_at: datetime | None = None
    body: BodyOutline | None = None
    wheels: tuple[WheelOutline, ...] = Field(default=(), max_length=8)
    terrain: tuple[TerrainOutline, ...] = Field(default=(), max_length=16)
    landmarks: tuple[DirectedLandmark, ...] = Field(default=(), max_length=4)
    occluders: tuple[BodyOutline, ...] = Field(default=(), max_length=16)
    missing_geometry: tuple[Name, ...] = ()
    note: Name

    @model_validator(mode="after")
    def evidence_and_extent(self) -> Self:
        if self.provenance == "reviewed_construction":
            if not all((self.source_image, self.label_protocol, self.reviewer, self.reviewed_at)):
                raise ValueError("Reviewed geometry requires source, protocol and reviewer receipt")
            if self.reviewed_at.utcoffset() is None:
                raise ValueError("Geometry review time requires a timezone")
        elif any((self.source_image, self.label_protocol, self.reviewer, self.reviewed_at)):
            raise ValueError("Synthetic fixtures cannot claim reviewed image provenance")
        if self.body is None and not self.missing_geometry:
            raise ValueError("Absent body geometry requires an explicit missingness note")
        if len({w.wheel_id for w in self.wheels}) != len(self.wheels):
            raise ValueError("Wheel identities must be unique")
        if len({p.role for p in self.landmarks}) != len(self.landmarks):
            raise ValueError("Directed landmark roles must be unique")
        width, height = self.image_size
        points = [
            p
            for shape in (*self.occluders, *((self.body,) if self.body else ()))
            for p in shape.vertices
        ]
        points += [p for strip in self.terrain for p in strip.points]
        points += [p.point for p in self.landmarks]
        points += [w.center for w in self.wheels]
        if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            raise ValueError("Label coordinates must lie within the declared image")
        if any(w.radius_pixels > min(width, height) / 2 for w in self.wheels):
            raise ValueError("Wheel radius exceeds the bounded screen geometry")
        extents = [(strip.points[0][0], strip.points[-1][0]) for strip in self.terrain]
        if any(b[0] < a[1] for a, b in pairwise(extents)):
            raise ValueError("Terrain strips must be ordered and nonoverlapping")
        return self

    def diagnostic_directed_angle(self) -> float | None:
        """Annotation-only angle, never supplied to the actor or shared measurer."""
        points = {p.role: p.point for p in self.landmarks}
        if not {"front", "rear"} <= points.keys() or points["front"] == points["rear"]:
            return None
        rear, front = points["rear"], points["front"]
        return math.atan2(-(front[1] - rear[1]), front[0] - rear[0])


class PoseFrame(ImmutableRecord):
    geometry: ScreenGeometry
    timestamp_ns: Annotated[int, Field(gt=0, strict=True)]
    continuity_id: Name


class PoseReplay(ImmutableRecord):
    schema_version: Literal["screen-pose-replay-3.0"] = "screen-pose-replay-3.0"
    replay_id: Name
    frames: tuple[PoseFrame, ...] = Field(min_length=1, max_length=120)
    source_partition: Literal["synthetic", "construction", "development", "train"]
    note: Name

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len({f.geometry.label_id for f in self.frames}) != len(self.frames):
            raise ValueError("Every pose frame needs a unique label identity")
        if len({f.geometry.image_size for f in self.frames}) != 1:
            raise ValueError("A replay requires one fixed image size")
        for previous, current in zip(self.frames, self.frames[1:]):
            if current.timestamp_ns <= previous.timestamp_ns:
                raise ValueError("Replay timestamps must strictly increase")
        synthetic = all(f.geometry.provenance == "synthetic_fixture" for f in self.frames)
        if synthetic != (self.source_partition == "synthetic"):
            raise ValueError("Synthetic and reviewed construction provenance cannot be mixed")
        if not synthetic and any(
            f.geometry.provenance != "reviewed_construction" for f in self.frames
        ):
            raise ValueError("Every construction pose needs reviewed image provenance")
        return self
