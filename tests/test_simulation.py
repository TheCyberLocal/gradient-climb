import numpy as np
import pytest

from gradientclimb.simulation import VectorHillEnv


def test_seeded_rollouts_are_identical():
    left = VectorHillEnv(8, seed=31, randomization=True)
    right = VectorHillEnv(8, seed=31, randomization=True)
    rng = np.random.default_rng(12)
    for _ in range(120):
        actions = rng.integers(0, 4, 8)
        lhs, rhs = left.step(actions), right.step(actions)
        for i in range(4):
            np.testing.assert_array_equal(lhs[i], rhs[i])


def test_pedals_are_independent_and_encoded_consistently():
    encoded = VectorHillEnv(4)
    bits = VectorHillEnv(4)
    actions = np.arange(4)
    pedals = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    for _ in range(20):
        encoded.step(actions)
        bits.step(pedals)
    np.testing.assert_array_equal(encoded.x, bits.x)
    np.testing.assert_array_equal(encoded.pedals, pedals)
    assert encoded.x[1] > encoded.x[0]
    assert encoded.x[2] < encoded.x[0]
    assert encoded.x[3] != encoded.x[1]
    assert encoded.action_duration == pytest.approx(0.06)


def test_observations_preserve_input_order():
    first = VectorHillEnv(1)
    second = VectorHillEnv(1)
    first.step(np.array([1]))
    second.step(np.array([2]))
    a = first.step(np.array([3]))[0]
    b = second.step(np.array([3]))[0]
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(first.history[0, -2, 21:23], [1, 0])
    np.testing.assert_array_equal(second.history[0, -2, 21:23], [0, 1])
    np.testing.assert_array_equal(first.history[0, -1, 21:23], [1, 1])


def test_same_step_reset_retains_final_observation_and_time_limit():
    env = VectorHillEnv(3, max_steps=2)
    env.step(np.ones(3, dtype=int))
    observations, rewards, terminated, truncated, info = env.step(np.ones(3, dtype=int))
    assert truncated.all() and not terminated.any()
    assert len(info["episodes"]) == 3
    assert all(e["length"] == 2 and e["termination"] == "time_limit" for e in info["episodes"])
    assert np.all(env.steps == 0)
    assert not np.array_equal(observations, info["final_observation"])
    assert np.isfinite(rewards).all()


def test_physical_death_and_terrain_gradient():
    env = VectorHillEnv(2)
    env.fuel[0] = 0
    _, _, terminated, truncated, info = env.step(np.array([0, 1]))
    assert terminated[0] and not truncated[0]
    assert info["episodes"][0]["termination"] == "fuel"
    x = np.array([17.0, 29.0])
    _, slope = env.ground(x)
    numerical = (env.ground(x + 1e-5)[0] - env.ground(x - 1e-5)[0]) / 2e-5
    np.testing.assert_allclose(slope, numerical, atol=1e-7)


def test_profile_shift_and_seed_list_reproducibility():
    env = VectorHillEnv(2, profile="heavy", terrain="rough")
    obs, _ = env.reset([91, 12])
    single = VectorHillEnv(1, profile="heavy", terrain="rough")
    other, _ = single.reset([12])
    np.testing.assert_array_equal(obs[1], other[0])
    assert env.parameters["mass"][0] == 1.55
    assert env.render(0, 320, 180).size == (320, 180)


@pytest.mark.parametrize(
    "actions", [np.array([4]), np.array([1.2]), np.array([[1, 2]]), np.zeros(3)]
)
def test_invalid_actions_are_rejected(actions):
    with pytest.raises(ValueError):
        VectorHillEnv(1).step(actions)
