"""Feature semantics checked with explicit synthetic evidence, never game accuracy."""

import copy
import math

import numpy as np
import pytest

from gradientclimb.perception.measurements import PixelMeasurement
from gradientclimb.perception.screen_features import FEATURE_NAMES, ScreenFeatureBridge


def evidence(angle=0, shift=0):
    return PixelMeasurement(
        "synthetic",
        (400, 240),
        body={
            "valid": True,
            "center_xy": [130 + shift, 100],
            "bbox": [80 + shift, 85, 180 + shift, 115],
            "axis_mod_pi_radians": angle,
            "quality": 0.8,
        },
        wheels={
            "valid": True,
            "center_xy": [130 + shift, 120],
            "axle_length_pixels": 60,
            "pitch_mod_pi_radians": angle,
            "quality": 0.9,
            "left": {"center_xy": [100 + shift, 120], "radius_pixels": 10},
            "right": {"center_xy": [160 + shift, 120], "radius_pixels": 10},
        },
        terrain={
            "valid": True,
            "points": [{"x": x, "y": 140 + 0.1 * x, "valid": True} for x in range(0, 401, 20)],
        },
    )


def hud(progress):
    return {"valid": progress is not None, "hud_displayed_progress_meters": progress}


def feature(observation, name):
    index = FEATURE_NAMES.index(name)
    return observation.values[index], observation.valid[index]


def test_relative_geometry_mask_order_and_action_bits():
    bridge = ScreenFeatureBridge(None, None, history=2)
    result = bridge.update(evidence(), hud(5), 1_000_000_000, previous_action_code=3)
    assert result.geometry_valid and not result.temporal_contiguous
    assert feature(result, "body_to_terrain_axles")[0] == pytest.approx(53 / 60)
    assert feature(result, "terrain_slope_image_right")[0] == pytest.approx(0.1)
    assert feature(result, "previous_os_gas") == (1, True)
    assert feature(result, "previous_os_brake") == (1, True)
    assert not feature(result, "previous_os_state_age_seconds")[1]
    assert not feature(result, "terrain_y_relative_axles_at_6")[1]
    assert not result.hud_increment["valid"]
    np.testing.assert_array_equal(result.vector[: 2 * len(FEATURE_NAMES)], 0)
    np.testing.assert_array_equal(result.vector[-len(FEATURE_NAMES) :], result.valid)
    assert result.vector.shape == (bridge.observation_dim,)


def test_modulo_pi_derivative_is_continuous_and_image_velocity_is_named():
    bridge = ScreenFeatureBridge(None, None)
    bridge.update(evidence(math.pi / 2 - 0.05), hud(2), 1_000_000_000)
    result = bridge.update(evidence(-math.pi / 2 + 0.05, shift=4), hud(3), 1_100_000_000)
    assert feature(result, "image_angular_rate_radians_per_second")[0] == pytest.approx(1)
    assert feature(result, "image_axle_vx_widths_per_second")[0] == pytest.approx(0.1)
    assert result.hud_increment["displayed_meters"] == 1


def test_missing_frames_do_not_fill_terrain_or_reward_gaps():
    bridge = ScreenFeatureBridge(None, None)
    missing = evidence()
    missing.terrain["points"][6]["valid"] = False  # x120 breaks interpolation at axle x130.
    bridge.update(evidence(), hud(1), 1_000_000_000)
    result = bridge.update(missing, hud(None), 1_100_000_000)
    assert not feature(result, "terrain_y_relative_axles_at_0")[1]
    assert not result.hud_increment["valid"]
    result = bridge.update(evidence(), hud(5), 1_200_000_000)
    assert not result.hud_increment["valid"]  # No invented per-action reward over unknown OCR.


def test_long_gap_resets_history_and_derivatives_and_reset_does_not_infer_ocr():
    bridge = ScreenFeatureBridge(None, None, history=2)
    bridge.update(evidence(), hud(12), 1_000_000_000)
    result = bridge.update(evidence(), hud(0), 2_000_000_000)
    assert not result.temporal_contiguous
    assert not feature(result, "image_angular_rate_radians_per_second")[1]
    np.testing.assert_array_equal(result.vector[: 2 * len(FEATURE_NAMES)], 0)
    result = bridge.update(evidence(), hud(5), 2_100_000_000)
    assert not result.hud_increment["valid"]  # Long gaps do not authorize episode resets.
    result = bridge.update(evidence(), hud(1), 2_200_000_000)
    assert not result.hud_increment["valid"]
    assert any("decreased" in reason for reason in result.failures)
    bridge.reset()
    assert not bridge.update(evidence(), hud(0), 3_000_000_000).temporal_contiguous


def test_failed_geometry_never_claims_valid_policy_observation():
    bridge = ScreenFeatureBridge(None, None)
    missing = copy.deepcopy(evidence())
    missing.body = missing.wheels = {"valid": False, "reason": "No object"}
    result = bridge.update(missing, hud(None), 1_000_000_000)
    assert not result.geometry_valid
    assert not result.valid[:15].any()
    assert np.isfinite(result.vector).all()
    assert result.as_dict()["input_authorized"] is False
    with pytest.raises(ValueError, match="Nonmonotonic"):
        bridge.update(evidence(), hud(0), 1_000_000_000)
    with pytest.raises(ValueError):
        bridge.update(evidence(), hud(0), 2_000_000_000, previous_action_code=4)


def test_body_subset_remains_explicit_when_wheels_are_missing():
    bridge = ScreenFeatureBridge(None, None)
    scene = evidence()
    scene.wheels = {"valid": False, "reason": "Occluded wheels"}
    first = bridge.update(scene, hud(0), 1_000_000_000)
    assert not first.geometry_valid
    assert first.supports(["body_sin_2angle", "body_cos_2angle", "body_to_terrain_spans"])
    assert not first.supports(["sin_2angle"])
    assert feature(first, "body_to_terrain_spans")[0] == pytest.approx(0.53)
    second = bridge.update(scene, hud(0), 1_100_000_000)
    assert feature(second, "body_image_angular_rate_radians_per_second") == (0, True)
    with pytest.raises(ValueError, match="required feature"):
        second.supports([])
