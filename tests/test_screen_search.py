"""Screen policy contracts and actual optimizer convergence, with no game input."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from gradientclimb.algorithms.screen_search import EpisodeCEM, ScreenLinearPolicy


def test_independent_heads_and_missing_observations():
    obs = SimpleNamespace(schema_id="screen-test", values=np.array([0.2]), valid=np.array([True]))
    for code in range(4):
        weights = np.zeros((2, 3))
        weights[:, -1] = [1 if code & 1 else -1, 1 if code & 2 else -1]
        policy = ScreenLinearPolicy("screen-test", ("angle",), weights, [1])
        assert policy.action(obs, ["angle"]) == code
        restored = ScreenLinearPolicy.from_dict(json.loads(json.dumps(policy.as_dict())))
        assert restored.action(obs, ["angle"]) == code
    obs.valid[:] = False
    assert policy.action(obs, ["angle"]) == 0
    obs.schema_id = "privileged-surrogate"
    with pytest.raises(ValueError, match="schemas differ"):
        policy.action(obs, ["angle"])


def test_mask_is_an_input_and_missing_value_is_not_a_measurement():
    weights = np.zeros((2, 5))
    weights[0, 0] = 10
    weights[1, 2] = 1  # Validity of the first feature drives brake.
    policy = ScreenLinearPolicy("s", ("a", "b"), weights, [1, 1])
    obs = SimpleNamespace(schema_id="s", values=[1e6, 0], valid=[False, True])
    assert policy.action(obs, ["a", "b"]) == 0
    obs.values, obs.valid = [0, 0], [True, True]
    assert policy.action(obs, ["a", "b"]) == 2
    obs.values = [float("nan"), 0]
    with pytest.raises(ValueError, match="nonfinite"):
        policy.action(obs, ["a", "b"])


def test_search_improves_a_known_objective_and_retains_episode_cost():
    optimizer = EpisodeCEM(np.zeros((2, 2)), seed=16, population=16, elite_count=4)
    target = np.array([[1.0, -0.8], [0.6, -0.4]])
    original_error = np.square(target).sum()
    for _ in range(25 * 16):
        candidate_id, candidate = optimizer.ask()
        optimizer.tell(candidate_id, -float(np.square(candidate - target).sum()))
    assert np.square(optimizer.incumbent - target).sum() < original_error / 20
    assert optimizer.generation == 25
    assert optimizer.completed_episodes == 400


def test_mid_generation_snapshot_resumes_identically():
    original = EpisodeCEM(np.zeros((2, 2)), seed=9, population=6, elite_count=2)
    for _ in range(3):
        identifier, weights = original.ask()
        original.tell(identifier, -float(np.square(weights - 1).sum()))
    restored = EpisodeCEM.from_state_dict(json.loads(json.dumps(original.state_dict())))
    for _ in range(15):
        left_id, left = original.ask()
        right_id, right = restored.ask()
        assert left_id == right_id
        np.testing.assert_array_equal(left, right)
        score = -float(np.square(left - 1).sum())
        assert original.tell(left_id, score) == restored.tell(right_id, score)
    assert original.state_dict() == restored.state_dict()


def test_search_rejects_wrong_candidate_or_nonfinite_score_without_updating():
    optimizer = EpisodeCEM(np.zeros((2, 1)))
    identifier, _ = optimizer.ask()
    for wrong_id, wrong_score in (("stale:0", 1), (identifier, float("nan")), (identifier, True)):
        with pytest.raises(ValueError):
            optimizer.tell(wrong_id, wrong_score)
    assert optimizer.completed_episodes == 0
    optimizer.tell(identifier, 1.0)
    with pytest.raises(ValueError):
        optimizer.tell(identifier, 2.0)
    state = optimizer.state_dict()
    state["completed_episodes"] = 50
    with pytest.raises(ValueError, match="counts disagree"):
        EpisodeCEM.from_state_dict(state)
