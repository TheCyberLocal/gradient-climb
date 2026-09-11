"""Profile-specific offline pixel measurements, never a game-state oracle.

Coordinates are screen pixels (x right, y down). Quality scores report measured
visual support, not calibrated correctness probabilities. World scale, front/back,
wheel contact and physical velocity are deliberately not inferred.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .pixels import _color_mask, _cv2, _region, _rgb


@dataclass(frozen=True)
class MeasurementProfile:
    profile_id: str
    expected_size: tuple[int, int]
    scene_roi: tuple[float, float, float, float]
    body_hsv: tuple = (((0, 130, 65), (8, 255, 255)), ((172, 130, 65), (179, 255, 255)))
    turf_hsv: tuple = (((30, 110, 85), (55, 255, 255)),)
    soil_hsv: tuple = (((8, 50, 30), (25, 220, 170)),)
    exclusion_rois: tuple = ()
    body_area_pixels: tuple[int, int] = (300, 14000)
    wheel_radius_pixels: tuple[int, int] = (10, 25)
    axle_length_pixels: tuple[float, float] = (55, 160)
    terrain_samples: int = 48
    camera_band_depth_pixels: int = 100
    static_ground_assumption_verified: bool = False

    def __post_init__(self):
        if not self.profile_id or len(self.expected_size) != 2:
            raise ValueError("A named profile and exact width/height are required")
        if any(type(v) is not int or v < 2 for v in self.expected_size):
            raise ValueError("Frame dimensions must be positive integers")
        object.__setattr__(self, "expected_size", tuple(self.expected_size))
        if type(self.static_ground_assumption_verified) is not bool:
            raise ValueError("Ground anchoring verification must be an explicit boolean")
        for pair in (self.body_area_pixels, self.wheel_radius_pixels, self.axle_length_pixels):
            if (
                len(pair) != 2
                or not all(math.isfinite(v) for v in pair)
                or not 0 < pair[0] < pair[1]
            ):
                raise ValueError("Profile measurement bounds must be finite ordered positive pairs")
        if self.terrain_samples < 3 or self.camera_band_depth_pixels < 5:
            raise ValueError("Insufficient terrain sampling or camera mask depth")
        probe = np.empty((self.expected_size[1], self.expected_size[0], 3), dtype=np.uint8)
        _region(probe, self.scene_roi)
        for roi in self.exclusion_rois:
            _region(probe, roi)


@dataclass
class PixelMeasurement:
    profile_id: str
    image_size: tuple[int, int]
    coordinate_system: str = "screen_pixels"
    body: dict = field(default_factory=dict)
    wheels: dict = field(default_factory=dict)
    terrain: dict = field(default_factory=dict)
    limitations: tuple[str, ...] = (
        "Profile-specific visual quality is not measured generalization accuracy",
        "Axle pitch is modulo pi; front/back and rollover remain unresolved",
        "No world meters, contact state, fuel or UI authorization is inferred",
    )

    def as_dict(self) -> dict:
        return asdict(self)


def _invalid(reason: str) -> dict:
    return {"valid": False, "quality": 0.0, "reason": reason}


class HCRPixelMeasurer:
    """Color body + independently supported circular wheel pair + turf boundary.

    The caller supplies an appearance/geometry profile. Detection is permitted on
    offline frames without asserting that the game is running or input is safe.
    """

    def __init__(self, profile: MeasurementProfile):
        self.profile = profile

    def _scene_mask(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        region, x, y = _region(frame, self.profile.scene_roi)
        mask[y : y + region.shape[0], x : x + region.shape[1]] = 255
        for roi in self.profile.exclusion_rois:
            region, x, y = _region(frame, roi)
            mask[y : y + region.shape[0], x : x + region.shape[1]] = 0
        return mask

    def measure(self, frame: np.ndarray) -> PixelMeasurement:
        frame = _rgb(frame)
        size = (frame.shape[1], frame.shape[0])
        result = PixelMeasurement(self.profile.profile_id, size)
        if size != self.profile.expected_size:
            result.body = result.wheels = result.terrain = _invalid("Profile geometry mismatch")
            return result
        mask = self._scene_mask(frame)
        result.body = self._body(frame, mask)
        result.wheels = self._wheels(frame, result.body)
        result.terrain = self._terrain(frame, mask)
        return result

    def _body(self, frame, scene_mask):
        cv2 = _cv2()
        mask = _color_mask(frame, self.profile.body_hsv) & scene_mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        low, high = self.profile.body_area_pixels
        candidates = [i for i in range(1, count) if low <= stats[i, cv2.CC_STAT_AREA] <= high]
        candidates.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        if not candidates:
            return _invalid("No body-colored component within profile area bounds")
        i = candidates[0]
        if (
            len(candidates) > 1
            and stats[candidates[1], cv2.CC_STAT_AREA] > 0.5 * stats[i, cv2.CC_STAT_AREA]
        ):
            return _invalid("Multiple plausible body-colored components")
        x, y, width, height, area = (int(v) for v in stats[i])
        fill = area / (width * height)
        if fill < 0.2:
            return _invalid("Body color support too sparse")
        yy, xx = np.nonzero(labels == i)
        _, vectors = np.linalg.eigh(np.cov(np.column_stack((xx, yy)), rowvar=False))
        axis = vectors[:, -1]
        if axis[0] < 0:
            axis = -axis
        return {
            "valid": True,
            "center_xy": centers[i].tolist(),
            "bbox": [x, y, x + width, y + height],
            "area_pixels": area,
            "fill_fraction": fill,
            "quality": fill,
            "axis_mod_pi_radians": math.atan2(-axis[1], axis[0]),
            "reason": "Largest isolated body-colored component; centroid is not center of mass",
        }

    def _wheels(self, frame, body):
        if not body["valid"]:
            return _invalid("Wheel search requires an identified body component")
        cv2 = _cv2()
        min_radius, max_radius = self.profile.wheel_radius_pixels
        x0, y0, x1, y1 = body["bbox"]
        margin = 2 * max_radius
        left, top = max(0, x0 - margin), max(0, y0 - margin)
        right, bottom = min(frame.shape[1], x1 + margin), min(frame.shape[0], y1 + margin)
        image = frame[top:bottom, left:right]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(gray, (5, 5), 1),
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=2 * min_radius,
            param1=100,
            param2=15,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return _invalid("No circular wheel candidates")
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        yy, xx = np.indices(gray.shape)
        edges = cv2.Canny(gray, 60, 130)
        edge_distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        candidates = []
        for cx, cy, radius in circles[0]:
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * 0.75) ** 2
            hub_support = float(((hsv[..., 1] < 85) & (hsv[..., 2] > 70))[disk].mean())
            angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            rx, ry = (
                np.rint(cx + radius * np.cos(angles)).astype(int),
                np.rint(cy + radius * np.sin(angles)).astype(int),
            )
            if (
                (rx < 0).any()
                or (ry < 0).any()
                or (rx >= gray.shape[1]).any()
                or (ry >= gray.shape[0]).any()
            ):
                continue
            ring_support = float((edge_distance[ry, rx] <= 2.5).mean())
            bx, by = float(cx + left), float(cy + top)
            body_distance = math.hypot(max(x0 - bx, 0, bx - x1), max(y0 - by, 0, by - y1))
            if hub_support < 0.45 or ring_support < 0.65 or body_distance > max_radius * 1.6:
                continue
            candidates.append(
                {
                    "center_xy": [bx, by],
                    "radius_pixels": float(radius),
                    "hub_support": hub_support,
                    "ring_support": ring_support,
                    "quality": min(hub_support, ring_support),
                }
            )
        pairs = []
        for a, b in itertools.combinations(candidates, 2):
            vector = np.asarray(b["center_xy"]) - a["center_xy"]
            distance = float(np.linalg.norm(vector))
            radii = [a["radius_pixels"], b["radius_pixels"]]
            center = (np.asarray(a["center_xy"]) + b["center_xy"]) / 2
            axle_pitch = math.atan2(-vector[1], vector[0])
            axis_difference = abs(
                (axle_pitch - body["axis_mod_pi_radians"] + math.pi / 2) % math.pi - math.pi / 2
            )
            perpendicular_offset = abs(
                float(np.dot(center - body["center_xy"], [-vector[1], vector[0]]))
            ) / max(distance, 1)
            if (
                not self.profile.axle_length_pixels[0]
                <= distance
                <= self.profile.axle_length_pixels[1]
                or max(radii) / min(radii) > 1.35
                or np.linalg.norm(center - body["center_xy"]) > distance * 0.45
                or axis_difference > 0.35
                or not distance * 0.05 <= perpendicular_offset <= distance * 0.45
            ):
                continue
            pairs.append((min(a["quality"], b["quality"]) * min(radii) / max(radii), a, b))
        pairs.sort(key=lambda row: row[0], reverse=True)
        if not pairs:
            return _invalid(f"No supported wheel pair ({len(candidates)} circular candidates)")
        if len(pairs) > 1 and pairs[0][0] - pairs[1][0] < 0.08:
            return _invalid("Wheel pair ambiguous among similar circular candidates")
        quality, a, b = pairs[0]
        a, b = sorted((a, b), key=lambda wheel: tuple(wheel["center_xy"]))
        delta = np.asarray(b["center_xy"]) - a["center_xy"]
        return {
            "valid": True,
            "left": a,
            "right": b,
            "center_xy": ((np.asarray(a["center_xy"]) + b["center_xy"]) / 2).tolist(),
            "axle_length_pixels": float(np.linalg.norm(delta)),
            "pitch_mod_pi_radians": math.atan2(-delta[1], delta[0]),
            "orientation_resolved": False,
            "quality": quality,
            "reason": "Two image circles with hub/ring/body support; not identified front/back wheels",
        }

    def _terrain(self, frame, scene_mask):
        cv2 = _cv2()
        turf = _color_mask(frame, self.profile.turf_hsv) & scene_mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(turf, connectivity=8)
        scene, scene_x, _ = _region(frame, self.profile.scene_roi)
        plausible = [
            i for i in range(1, count) if stats[i, cv2.CC_STAT_WIDTH] >= scene.shape[1] * 0.4
        ]
        if not plausible:
            return _invalid("No horizontally extensive turf component")
        label = max(plausible, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        mask = labels == label
        soil = _color_mask(frame, self.profile.soil_hsv) > 0
        xs = np.rint(
            np.linspace(scene_x, scene_x + scene.shape[1] - 1, self.profile.terrain_samples)
        ).astype(int)
        points = []
        for x in xs:
            nearby = []
            for column in range(max(0, x - 5), min(frame.shape[1], x + 6)):
                rows = np.flatnonzero(mask[:, column])
                if len(rows) < 3:
                    continue
                # The lower turf boundary suppresses decorative foliage above the road.
                # It is a visible material boundary, not a known collision surface.
                y = int(rows[-1]) + 1
                # Brown soil below the green strip distinguishes distant scenery/objects.
                lower = soil[min(y + 2, frame.shape[0]) : min(y + 24, frame.shape[0]), column]
                if len(lower) and lower.mean() >= 0.4:
                    nearby.append(y)
            if len(nearby) >= 3:
                points.append(
                    {
                        "x": int(x),
                        "y": float(np.median(nearby)),
                        "valid": True,
                        "support_columns": len(nearby),
                    }
                )
            else:
                points.append(
                    {"x": int(x), "y": None, "valid": False, "support_columns": len(nearby)}
                )
        coverage = sum(point["valid"] for point in points) / len(points)
        return {
            "valid": coverage >= 0.25,
            "points": points,
            "coverage": coverage,
            "surface_kind": "visible_turf_soil_boundary",
            "quality": coverage,
            "reason": "Median turf/soil boundary; offset from the game's physical collision surface is unknown",
        }

    def camera_mask(self, frame: np.ndarray, measurement: PixelMeasurement) -> np.ndarray:
        """Use near-ground soil texture; exclude vehicle and configured HUD/overlays."""
        frame = _rgb(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if measurement.image_size != self.profile.expected_size or not measurement.terrain.get(
            "valid"
        ):
            return mask
        points = [p for p in measurement.terrain["points"] if p["valid"]]
        if len(points) < 3:
            return mask
        for x in range(points[0]["x"], points[-1]["x"] + 1):
            y = int(np.interp(x, [p["x"] for p in points], [p["y"] for p in points]))
            mask[y + 15 : min(frame.shape[0], y + self.profile.camera_band_depth_pixels), x] = 255
        mask &= _color_mask(frame, self.profile.soil_hsv) & self._scene_mask(frame)
        if measurement.body.get("valid"):
            x0, y0, x1, y1 = measurement.body["bbox"]
            radius = 2 * self.profile.wheel_radius_pixels[1]
            mask[max(0, y0 - radius) : y1 + radius, max(0, x0 - radius) : x1 + radius] = 0
        return mask


class TerrainMotionTracker:
    """Relative translation from static near-ground texture, in pixels only.

    Scale/rotation/poorly distributed tracks are rejected. A failed interval breaks
    continuity; no global distance is filled in across that gap. Repeated still
    images can yield valid zero translation but cannot establish capture freshness.
    """

    def __init__(self, measurer: HCRPixelMeasurer, *, max_interval_seconds=0.5, min_tracks=8):
        if not math.isfinite(max_interval_seconds) or max_interval_seconds <= 0 or min_tracks < 4:
            raise ValueError("Invalid tracking limits")
        self.measurer, self.max_interval_seconds, self.min_tracks = (
            measurer,
            max_interval_seconds,
            min_tracks,
        )
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, frame: np.ndarray, measurement: PixelMeasurement, timestamp_ns: int) -> dict:
        frame = _rgb(frame)
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("A nonnegative monotonic capture timestamp is required")
        cv2 = _cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mask = self.measurer.camera_mask(frame, measurement)
        current = (gray, mask, measurement, timestamp_ns)
        previous, self.previous = self.previous, current
        if previous is None:
            return _invalid("First frame has no camera-motion interval")
        old, old_mask, old_measurement, old_time = previous
        dt = (timestamp_ns - old_time) / 1e9
        if not 0 < dt <= self.max_interval_seconds or old.shape != gray.shape:
            return _invalid(
                "Nonpositive/long capture interval or geometry change; continuity broken"
            )
        if not mask.any() or not old_mask.any():
            return _invalid("Static ground mask unavailable")
        features = cv2.goodFeaturesToTrack(
            old, maxCorners=400, qualityLevel=0.01, minDistance=7, mask=old_mask
        )
        if features is None or len(features) < self.min_tracks:
            return _invalid("Insufficient ground texture features")
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            old, gray, features, None, winSize=(21, 21), maxLevel=3
        )
        if moved is None:
            return _invalid("Forward optical flow failed")
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, old, moved, None, winSize=(21, 21), maxLevel=3
        )
        if back is None:
            return _invalid("Backward optical flow failed")
        p, q = features[:, 0], moved[:, 0]
        keep = (
            status[:, 0].astype(bool)
            & back_status[:, 0].astype(bool)
            & (np.linalg.norm(back[:, 0] - p, axis=1) <= 1.0)
        )
        coords = np.rint(q).astype(int)
        inside = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < gray.shape[1])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < gray.shape[0])
        )
        keep &= inside
        keep[inside] &= mask[coords[inside, 1], coords[inside, 0]] > 0
        p, q = p[keep], q[keep]
        if len(p) < self.min_tracks:
            return _invalid("Insufficient consistent ground tracks after forward/back checks")
        matrix, inliers = cv2.estimateAffinePartial2D(
            p, q, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=2000, confidence=0.99
        )
        if matrix is None or inliers is None:
            return _invalid("Ground motion model could not be identified")
        selected = inliers[:, 0].astype(bool)
        ratio = float(selected.mean())
        scale, rotation = (
            math.hypot(matrix[0, 0], matrix[1, 0]),
            math.atan2(matrix[1, 0], matrix[0, 0]),
        )
        shifts = q[selected] - p[selected]
        translation = np.median(shifts, axis=0)
        residual = float(np.median(np.linalg.norm(shifts - translation, axis=1)))
        spread = np.ptp(p[selected], axis=0)
        diagnostics = {
            "tracked_features": len(p),
            "inlier_tracks": int(selected.sum()),
            "inlier_fraction": ratio,
            "median_residual_pixels": residual,
            "scale": scale,
            "rotation_radians": rotation,
            "support_span_pixels": spread.tolist(),
            "interval_seconds": dt,
        }
        if (
            selected.sum() < self.min_tracks
            or ratio < 0.7
            or residual > 1.2
            or abs(scale - 1) > 0.01
            or abs(rotation) > 0.015
            or spread[0] < 60
            or spread[1] < 15
        ):
            return {
                **_invalid("Ground tracks violate translation/support model; no compensation"),
                **diagnostics,
            }
        # Texture may be screen anchored in a 2D game. Check the independently
        # extracted turf edge, then require documented anchoring verification before
        # interpreting this image translation as camera/vehicle displacement.
        old_points = old_measurement.terrain.get("points", [])
        edge_errors = []
        for point in measurement.terrain.get("points", []):
            if not point["valid"]:
                continue
            x = point["x"] - translation[0]
            for first, second in itertools.pairwise(old_points):
                if first["valid"] and second["valid"] and first["x"] <= x <= second["x"]:
                    expected = (
                        np.interp(x, [first["x"], second["x"]], [first["y"], second["y"]])
                        + translation[1]
                    )
                    edge_errors.append(abs(point["y"] - expected))
                    break
        diagnostics["turf_edge_support_points"] = len(edge_errors)
        diagnostics["turf_edge_median_error_pixels"] = (
            float(np.median(edge_errors)) if edge_errors else None
        )
        diagnostics["measured_texture_translation_xy"] = translation.tolist()
        if len(edge_errors) < 8 or np.median(edge_errors) > 2.5:
            return {
                **_invalid("Texture motion disagrees with visible road boundary"),
                **diagnostics,
            }
        if not self.measurer.profile.static_ground_assumption_verified:
            return {
                **_invalid(
                    "Texture translation measured; static ground anchoring remains unverified"
                ),
                **diagnostics,
                "camera_translation_xy": None,
                "vehicle_displacement_xy": None,
            }
        vehicle_delta = None
        if measurement.wheels.get("valid") and old_measurement.wheels.get("valid"):
            vehicle_delta = (
                np.asarray(measurement.wheels["center_xy"])
                - old_measurement.wheels["center_xy"]
                - translation
            ).tolist()
        return {
            "valid": True,
            "quality": ratio * max(0, 1 - residual / 1.2),
            **diagnostics,
            "coordinate_system": "relative_ground_plane_pixels",
            "terrain_screen_translation_xy": translation.tolist(),
            "camera_translation_xy": (-translation).tolist(),
            "vehicle_displacement_xy": vehicle_delta,
            "reason": "Static near-ground translation model; no world scale, global position or parallax guarantee",
        }
