"""Generated visual fixtures only; no real capture, input, or game accuracy claims."""

import hashlib
from dataclasses import replace

import cv2
import numpy as np
import pytest

from gradientclimb.perception.annotations import (
    FrameAnnotation,
    evaluate_measurements,
    validate_partition,
)
from gradientclimb.perception.hud import HUDDigitReader
from gradientclimb.perception.measurements import (
    HCRPixelMeasurer,
    MeasurementProfile,
    TerrainMotionTracker,
)


def fixture_scene(texture=True):
    image = np.full((240, 400, 3), (145, 220, 250), dtype=np.uint8)
    image[135:145] = (100, 205, 40)
    image[145:] = (110, 82, 52)
    if texture:
        rng = np.random.default_rng(44)
        for x, y, radius in zip(
            rng.integers(5, 395, 160), rng.integers(155, 232, 160), rng.integers(2, 5, 160)
        ):
            cv2.circle(image, (int(x), int(y)), int(radius), (85, 65, 43), -1)
    cv2.rectangle(image, (80, 87), (188, 118), (210, 25, 25), -1)
    for center in [(99, 124), (168, 124)]:
        cv2.circle(image, center, 14, (25, 25, 25), -1)
        cv2.circle(image, center, 10, (180, 185, 190), -1)
        cv2.line(image, (center[0] - 8, center[1]), (center[0] + 8, center[1]), (120, 120, 120), 2)
    return image


def measurer():
    return HCRPixelMeasurer(
        MeasurementProfile(
            "synthetic-known-scene",
            (400, 240),
            (0, 0.2, 1, 1),
            wheel_radius_pixels=(10, 18),
            axle_length_pixels=(55, 90),
            static_ground_assumption_verified=True,
        )
    )


def test_supported_wheel_pair_and_material_boundary_on_generated_image():
    result = measurer().measure(fixture_scene())
    assert result.body["valid"] and result.wheels["valid"]
    np.testing.assert_allclose(result.wheels["left"]["center_xy"], [99, 124], atol=2)
    np.testing.assert_allclose(result.wheels["right"]["center_xy"], [168, 124], atol=2)
    assert not result.wheels["orientation_resolved"]
    assert result.terrain["surface_kind"] == "visible_turf_soil_boundary"
    assert result.terrain["coverage"] > 0.9
    assert all(abs(point["y"] - 145) <= 1 for point in result.terrain["points"] if point["valid"])


def test_absent_or_ambiguous_body_and_geometry_fail_explicitly():
    detector = measurer()
    blank = np.zeros((240, 400, 3), np.uint8)
    assert not detector.measure(blank).wheels["valid"]
    ambiguous = fixture_scene()
    cv2.rectangle(ambiguous, (240, 87), (348, 118), (210, 25, 25), -1)
    assert not detector.measure(ambiguous).body["valid"]
    assert not detector.measure(blank[:100]).terrain["valid"]


def test_driver_head_cannot_replace_a_missing_wheel():
    image = fixture_scene()
    cv2.circle(image, (99, 124), 16, (145, 220, 250), -1)
    cv2.circle(image, (140, 65), 14, (25, 25, 25), -1)
    cv2.circle(image, (140, 65), 10, (180, 185, 190), -1)
    assert not measurer().measure(image).wheels["valid"]


def test_static_ground_translation_is_recovered_and_vehicle_delta_compensated():
    detector, image = measurer(), fixture_scene()
    moved = cv2.warpAffine(
        image, np.float32([[1, 0, -5], [0, 1, 2]]), (400, 240), borderMode=cv2.BORDER_REFLECT
    )
    tracker = TerrainMotionTracker(detector)
    assert not tracker.update(image, detector.measure(image), 1_000_000_000)["valid"]
    camera = tracker.update(moved, detector.measure(moved), 1_050_000_000)
    assert camera["valid"], camera
    np.testing.assert_allclose(camera["terrain_screen_translation_xy"], [-5, 2], atol=0.3)
    np.testing.assert_allclose(camera["vehicle_displacement_xy"], [0, 0], atol=1)


def test_camera_rejects_missing_texture_large_scale_and_time_gaps():
    detector = measurer()
    blank_ground = fixture_scene(texture=False)
    tracker = TerrainMotionTracker(detector)
    tracker.update(blank_ground, detector.measure(blank_ground), 1_000_000_000)
    assert not tracker.update(blank_ground, detector.measure(blank_ground), 1_050_000_000)["valid"]
    image = fixture_scene()
    tracker.reset()
    tracker.update(image, detector.measure(image), 1_000_000_000)
    scaled = cv2.warpAffine(image, cv2.getRotationMatrix2D((200, 170), 0, 1.04), (400, 240))
    assert not tracker.update(scaled, detector.measure(scaled), 1_050_000_000)["valid"]
    assert not tracker.update(image, detector.measure(image), 3_000_000_000)["valid"]


def test_unverified_ground_texture_does_not_produce_camera_position():
    detector = HCRPixelMeasurer(
        replace(measurer().profile, static_ground_assumption_verified=False)
    )
    image = fixture_scene()
    tracker = TerrainMotionTracker(detector)
    tracker.update(image, detector.measure(image), 1_000_000_000)
    result = tracker.update(image, detector.measure(image), 1_050_000_000)
    assert not result["valid"]
    assert result["camera_translation_xy"] is None
    assert result["vehicle_displacement_xy"] is None
    assert "measured_texture_translation_xy" in result


def annotation(**changes):
    data = {
        "annotation_id": "one",
        "source_sha256": hashlib.sha256(b"one").hexdigest(),
        "session_id": "session-one",
        "split": "heldout",
        "evidence_domain": "synthetic",
        "width": 400,
        "height": 240,
        "annotator": "fixture author",
        "annotation_method": "synthetic_known_transform",
        "wheel_pair_visible": True,
        "left_wheel": {"x": 99, "y": 124},
        "right_wheel": {"x": 168, "y": 124},
    }
    data.update(changes)
    return FrameAnnotation(**data)


def test_annotation_partition_and_missing_output_coverage():
    first = annotation()
    second = annotation(annotation_id="two", source_sha256=hashlib.sha256(b"two").hexdigest())
    predictions = {first.source_sha256: measurer().measure(fixture_scene()).as_dict()}
    report = evaluate_measurements(predictions, [first, second])
    assert report["wheel_coverage"] == 0.5
    assert report["wheel_labeled_frames"] == 2
    with pytest.raises(ValueError, match="session_id overlap"):
        validate_partition([first, second.model_copy(update={"split": "train"})])
    with pytest.raises(ValueError, match="without viewing"):
        annotation(viewed_model_prediction=True)
    with pytest.raises(ValueError, match="not real-game"):
        annotation(evidence_domain="real_game")


def test_hud_requires_complete_alphabet_and_reads_generated_digits():
    def rendered(text):
        image = np.zeros((60, 160, 3), dtype=np.uint8)
        cv2.putText(image, text, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        return image

    reader = HUDDigitReader(expected_size=(160, 60), roi=(0, 0, 1, 1))
    for digit in "012345678":
        reader.add_labeled_region(
            rendered(digit),
            digit,
            (0, 0, 1, 1),
            source_sha256=hashlib.sha256(digit.encode()).hexdigest(),
            annotation_id=f"synthetic-{digit}",
        )
    assert reader.read(rendered("26"))["missing_digits"] == ["9"]
    reader.add_labeled_region(
        rendered("9"),
        "9",
        (0, 0, 1, 1),
        source_sha256=hashlib.sha256(b"9").hexdigest(),
        annotation_id="synthetic-9",
    )
    reading = reader.read(rendered("290"))
    assert reading["valid"], reading
    assert reading["hud_displayed_progress_meters"] == 290
    assert not reader.read(np.zeros((60, 160, 3), dtype=np.uint8))["valid"]
