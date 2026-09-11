import numpy as np
import pytest
import torch

from gradientclimb.algorithms import (
    ActorCritic,
    AlwaysGasPolicy,
    RandomPolicy,
    load_policy,
    train_ppo,
)
from gradientclimb.algorithms.ppo import compute_gae
from gradientclimb.evaluation import compare_paired, evaluate, summarize


def test_gae_bootstraps_time_limit_but_not_physical_terminal():
    rewards = torch.tensor([[1.0, 1.0], [99.0, 99.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[5.0, 5.0], [0.0, 0.0]])
    terminated = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    done = torch.ones_like(rewards)
    advantage, returns = compute_gae(rewards, values, next_values, terminated, done, gamma=0.9)
    torch.testing.assert_close(advantage[0], torch.tensor([1.0, 5.5]))
    torch.testing.assert_close(returns, advantage)


def test_checkpoint_round_trip_and_four_actions(tmp_path):
    torch.set_num_threads(1)
    model = ActorCritic(96, 16)
    model.config = {"seed": 123, "stack": 4}
    observations = np.random.default_rng(42).normal(size=(100, 96)).astype(np.float32)
    path = tmp_path / "policy.pt"
    model.save(path)
    loaded = load_policy(path)
    np.testing.assert_array_equal(model.act(observations), loaded.act(observations))
    assert loaded.config["seed"] == 123
    assert set(model.act(np.zeros((1000, 96), np.float32), deterministic=False)) == {0, 1, 2, 3}


def test_wall_clock_training_produces_finite_updates():
    # PyTorch lazily imports its optimizer stack on first construction. Warm that
    # runtime for this short CI test; governed research runs count the cold start.
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    calls = []
    result = train_ppo(
        2.0,
        seed=9,
        num_envs=8,
        rollout_steps=16,
        hidden_size=16,
        minibatch_size=64,
        epochs=2,
        max_steps=30,
        checkpoint_times=[0.5, 1.0],
        callback=lambda metrics, model, elapsed: calls.append(metrics.copy()),
    )
    assert result.environment_steps > 0 and result.optimizer_updates > 0
    assert result.elapsed >= 2.0
    assert result.elapsed < 8.0
    assert all(torch.isfinite(p).all() for p in result.model.parameters())
    assert calls[-1]["final"]
    assert {
        row["requested_checkpoint_seconds"] for row in calls if row["requested_checkpoint_seconds"]
    } == {0.5, 1.0}


def test_evaluation_is_seeded_and_paired():
    first = evaluate(AlwaysGasPolicy(), [10, 11, 12], max_steps=40)
    second = evaluate(AlwaysGasPolicy(), [10, 11, 12], max_steps=40)
    assert first["episodes"] == second["episodes"]
    assert sum(first["action_counts"]) == first["environment_steps"]
    assert compare_paired(first, second)["distance_difference"]["mean"] == 0
    random = evaluate(RandomPolicy(8), [10, 11, 12], max_steps=40)
    assert all(count > 0 for count in random["action_counts"])
    assert len(random["episodes"]) == 3


def test_summary_rejects_missing_and_nonfinite_values():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize([float("nan")])
    assert summarize([2.0])["ci95_low"] is None
    assert summarize([2.0, 2.0])["ci95_low"] == 2.0
