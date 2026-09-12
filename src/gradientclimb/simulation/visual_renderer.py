"""Original procedural screen-pixel art, without copied game assets or physics."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field

from gradientclimb.environments.profiles import ImmutableRecord

from .visual_labels import ScreenGeometry

RENDERER_VERSION = "procedural-screen-pose-3.0"
Channel = Annotated[int, Field(ge=0, le=255, strict=True)]
Color = tuple[Channel, Channel, Channel]


class RenderStyle(ImmutableRecord):
    """Frozen original palette; changing appearance is a new declared comparison."""

    version: Literal["procedural-screen-pose-3.0"] = RENDERER_VERSION
    sky: Color = (160, 222, 240)
    soil: Color = (113, 78, 43)
    turf: Color = (111, 218, 29)
    body: Color = (205, 30, 22)
    tire: Color = (40, 43, 45)
    hub: Color = (176, 189, 193)
    occluder: Color = (55, 55, 55)
    turf_width_pixels: int = Field(default=6, ge=1, le=16, strict=True)


def render_pose(geometry: ScreenGeometry, style: RenderStyle | None = None) -> np.ndarray:
    """Draw only declared primitives; missing terrain and body regions remain absent.

    Wheel interiors are original decoration, not labels of wheel rotation.
    Directed landmarks are annotation metadata and never appear in actor pixels.
    The existing shared bridge therefore still cannot recover full orientation.
    """
    geometry = ScreenGeometry.model_validate(geometry)
    style = RenderStyle.model_validate(style) if style is not None else RenderStyle()
    width, height = geometry.image_size
    image = Image.new("RGB", (width, height), style.sky)
    draw = ImageDraw.Draw(image)
    for strip in geometry.terrain:
        points = list(strip.points)
        draw.polygon([*points, (points[-1][0], height), (points[0][0], height)], fill=style.soil)
        draw.line(points, fill=style.turf, width=style.turf_width_pixels)
    if geometry.body:
        draw.polygon(geometry.body.vertices, fill=style.body)
    for wheel in geometry.wheels:
        x, y = wheel.center
        radius = wheel.radius_pixels
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=style.tire)
        hub = radius * 0.68
        draw.ellipse((x - hub, y - hub, x + hub, y + hub), fill=style.hub)
    for shape in geometry.occluders:
        draw.polygon(shape.vertices, fill=style.occluder)
    return np.array(image, dtype=np.uint8, copy=True)
