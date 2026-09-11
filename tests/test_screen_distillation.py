"""Pure semantic checks; not distillation training or real pixel accuracy."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gradientclimb.algorithms.screen_distillation import (
    BODY_FEATURES,
    BODY_POLYGON,
    ScreenBodyStudent,
    SurrogateBodyBridge,
    evaluate_student,
    pack_frames,
    polygon_moments,
    project_body_geometry,
)
from gradientclimb.perception.measurements import PixelMeasurement
from gradientclimb.perception.screen_features import FEATURE_NAMES, ScreenFeatureBridge


def test_polygon_moments_rectangle_known_solution():
    center, covariance = polygon_moments([[-2, -1], [2, -1], [2, 1], [-2, 1]])
    np.testing.assert_allclose(center, [0, 0], atol=1e-12)
    np.testing.assert_allclose(covariance, np.diag([4 / 3, 1 / 3]), atol=1e-12)


def test_projection_matches_screen_bridge_geometry_sign_and_span():
    x, y, theta = np.array([9.0]), np.array([2.0]), np.array([0.4])
    values, angles, valid = project_body_geometry(x, y, theta, lambda points: 0.1 * points)
    center, _ = polygon_moments(BODY_POLYGON)
    rotation = np.array(
        [[np.cos(theta[0]), -np.sin(theta[0])], [np.sin(theta[0]), np.cos(theta[0])]]
    )
    points = BODY_POLYGON @ rotation.T + [x[0], y[0]]
    centroid = rotation @ center + [x[0], y[0]]
    image_points = (points - [1, 6.5]) * [32, -32]
    image_center = (centroid - [1, 6.5]) * [32, -32]
    terrain = [
        {"x": float(px), "y": float((6.5 - 0.1 * (px / 32 + 1)) * 32), "valid": True}
        for px in range(0, 961, 10)
    ]
    evidence = PixelMeasurement(
        "synthetic_projection",
        (960, 540),
        body={
            "valid": True,
            "quality": 1,
            "center_xy": image_center.tolist(),
            "bbox": [*image_points.min(axis=0), *image_points.max(axis=0)],
            "axis_mod_pi_radians": float(angles[0]),
        },
        terrain={"valid": True, "points": terrain},
    )
    bridge = ScreenFeatureBridge(None, None)
    actual = bridge.update(evidence, {}, 1_000_000_000)
    indices = [FEATURE_NAMES.index(name) for name in BODY_FEATURES]
    np.testing.assert_array_equal(actual.valid[indices], valid[0])
    np.testing.assert_allclose(actual.values[indices], values[0], rtol=1e-5, atol=1e-6)
    assert values[0, 4] == pytest.approx(-0.1)


def test_projection_censors_terrain_outside_viewport():
    values, _, valid = project_body_geometry([0], [30], [0], lambda q: np.zeros_like(q))
    assert valid[0, :2].all()
    assert not valid[0, 3:].any()
    assert not values[0, 3:].any()


def test_temporal_derivative_reset_and_no_privileged_velocity_reads():
    # Reading omega, vx, fuel, contacts or teacher observations would raise.
    env = SimpleNamespace(
        num_envs=2,
        x=np.array([0.0, 1.0]),
        y=np.ones(2),
        theta=np.array([0.0, 0.0]),
        action_duration=0.06,
        ground=lambda q: (np.zeros_like(q), np.zeros_like(q)),
    )
    bridge = SurrogateBodyBridge(2)
    first = bridge.observe(env).reshape(2, 4, 16)
    assert not first[:, :3].any()
    assert not first[:, -1, 10].any()
    env.theta += 0.06
    second = bridge.observe(env).reshape(2, 4, 16)
    np.testing.assert_allclose(second[:, -1, 2], 0.25, atol=1e-5)
    assert second[:, -1, 10].all()
    third = bridge.observe(env, [True, False]).reshape(2, 4, 16)
    assert not third[0, :3].any()
    assert not third[0, -1, 10]
    assert third[1, -1, 10]


def observation(student, frames):
    return SimpleNamespace(schema_id=student.schema_id, vector=frames.reshape(-1))


def test_actor_selects_only_body_feature_history_and_masks():
    student = ScreenBodyStudent("test-schema", history=2)
    count = len(FEATURE_NAMES)
    frames = np.zeros((2, 2 * count), np.float32)
    columns = [FEATURE_NAMES.index(name) for name in BODY_FEATURES]
    frames[:, columns] = 0.2
    frames[:, count + np.array(columns)] = 1
    original = student.restricted_vector(observation(student, frames))
    excluded = sorted(set(range(count)) - set(columns))
    frames[:, excluded] = np.nan  # Unrelated inputs cannot affect actor.
    frames[:, count + np.array(excluded)] = 1
    np.testing.assert_array_equal(student.restricted_vector(observation(student, frames)), original)
    frames[0, columns[0]] = 0.8
    assert student.restricted_vector(observation(student, frames))[0] != original[0]
    with pytest.raises(ValueError, match="schemas differ"):
        student.action(SimpleNamespace(schema_id="wrong", vector=frames.reshape(-1)))
    frames[-1, count + columns[0]] = 0
    assert student.action(observation(student, frames)) == 0


def test_masked_nan_is_missing_valid_nan_rejected_and_schema_roundtrip(tmp_path):
    values, valid = np.full((2, 8), np.nan), np.zeros((2, 8), bool)
    assert not pack_frames(values, valid).any()
    valid[0, 0] = True
    with pytest.raises(ValueError, match="nonfinite"):
        pack_frames(values, valid)
    model = ScreenBodyStudent("real-bridge-schema-hash", history=4)
    path = tmp_path / "student.pt"
    model.save(path)
    loaded = ScreenBodyStudent.load(path)
    inputs = torch.zeros(2, 64)
    torch.testing.assert_close(model(inputs), loaded(inputs))
    assert loaded.schema_id == model.schema_id
    assert loaded.config["qualifies_real_game"] is False


def test_modulo_pi_alias_rejected_and_gap_clears_history():
    env = SimpleNamespace(
        num_envs=1,
        x=np.zeros(1),
        y=np.ones(1),
        theta=np.zeros(1),
        action_duration=0.06,
        ground=lambda q: (np.zeros_like(q), np.zeros_like(q)),
    )
    bridge = SurrogateBodyBridge(1)
    bridge.observe(env)
    env.theta += 1.2
    alias = bridge.observe(env).reshape(1, 4, 16)
    assert not alias[0, -1, 10]
    env.action_duration = 0.7
    gap = bridge.observe(env).reshape(1, 4, 16)
    assert not gap[:, :3].any()
    assert not gap[0, -1, 10]


def test_evaluation_counts_first_episodes_only_with_mock_dynamics(monkeypatch):
    class FakeEnv:
        def __init__(self, *args, **kwargs):
            self.num_envs, self.action_duration = 2, 0.06
            self.x, self.y, self.theta = np.zeros(2), np.ones(2), np.zeros(2)
            self.step_number = 0

        def reset(self, seeds):
            assert seeds == [30000, 30001]
            return np.zeros((2, 96), np.float32), {}

        def ground(self, points):
            return np.zeros_like(points), np.zeros_like(points)

        def step(self, actions):
            self.step_number += 1
            rows = [{"env_index": 0, "distance": 3.0, "termination": "crash"}]
            done = np.array([True, False])
            if self.step_number == 2:
                # Automatic reset's second env0 episode must never enter evaluation.
                rows = [
                    {"env_index": 0, "distance": 999.0, "termination": "time_limit"},
                    {"env_index": 1, "distance": 4.0, "termination": "time_limit"},
                ]
                done[:] = True
            return (
                np.zeros((2, 96), np.float32),
                np.zeros(2),
                done,
                np.zeros(2, bool),
                {"episodes": rows},
            )

    import gradientclimb.simulation

    monkeypatch.setattr(gradientclimb.simulation, "VectorHillEnv", FakeEnv)
    teacher = SimpleNamespace(
        config={"stack": 4}, eval=lambda: None, act=lambda observations: np.ones(2, dtype=int)
    )
    student = SimpleNamespace(
        history=4,
        max_interval_seconds=0.5,
        eval=lambda: None,
        act_vectors=lambda vectors: np.array([1, 0]),
    )
    result = evaluate_student(student, teacher, [30000, 30001])
    assert result["mean_distance"] == 3.5
    assert result["agreement_observations"] == 3
    assert result["joint_action_agreement"] == pytest.approx(1 / 3)
    assert result["pedal_bit_agreement"] == pytest.approx(2 / 3)
    assert result["action_counts"] == [2, 1, 0, 0]
