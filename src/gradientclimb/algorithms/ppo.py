"""Original clipped PPO implementation with GAE and explicit time governance.

References are recorded in research/literature. Independent Bernoulli outputs
allow both pedals at once. This implementation consumes idealized simulator
observations; it is not by itself a validated real-game perception policy.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli

from gradientclimb.simulation import SIMULATOR_VERSION, VectorHillEnv

DEFAULT_CHECKPOINT_TIMES = (300, 600, 1200, 1800, 2700, 3600)


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_size: int = 64, device: str = "cpu"):
        super().__init__()
        self.observation_dim, self.hidden_size = observation_dim, hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, 2)
        self.critic = nn.Linear(hidden_size, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        self.config: dict = {}
        self.training_state: dict = {}
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def distribution_value(self, observations: torch.Tensor):
        hidden = self.encoder(observations)
        return Bernoulli(logits=self.actor(hidden)), self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def act(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        distribution, _ = self.distribution_value(torch.as_tensor(observations, device=self.device))
        bits = (distribution.logits >= 0).long() if deterministic else distribution.sample().long()
        return (bits[:, 0] + 2 * bits[:, 1]).cpu().numpy()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "algorithm": "ppo",
                "config": self.config,
                "observation_dim": self.observation_dim,
                "hidden_size": self.hidden_size,
                "model_state": self.state_dict(),
                "training_state": self.training_state,
            },
            path,
        )


@dataclass
class TrainingResult:
    model: Any
    metrics: list[dict]
    config: dict
    elapsed: float
    environment_steps: int
    episodes: int
    optimizer_updates: int


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    done: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap time limits from their final state, but never across a reset."""
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        accumulator = delta + gamma * gae_lambda * (1 - done[t]) * accumulator
        advantages[t] = accumulator
    return advantages, advantages + values


def train_ppo(
    seconds: float,
    seed: int = 0,
    num_envs: int = 64,
    callback: Callable[[dict, ActorCritic, float], None] | None = None,
    checkpoint_times=DEFAULT_CHECKPOINT_TIMES,
    profile: str = "default",
    terrain: str = "train",
    randomization: bool = False,
    stack: int = 4,
    max_steps: int = 1000,
    hidden_size: int = 64,
    rollout_steps: int = 128,
    minibatch_size: int = 512,
    epochs: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    entropy_coef: float = 0.01,
    device: str = "cpu",
    torch_threads: int = 1,
    log_interval: float = 5.0,
    parent_checkpoint: str | None = None,
    snapshot_interval: float | None = None,
    snapshot: Callable[[dict], None] | None = None,
) -> TrainingResult:
    """Train until a monotonic deadline, counting initialization and callback time.

    Callback metrics contain ``checkpoint`` and ``checkpoint_target_seconds``.
    Periodic logs also invoke callbacks, with checkpoint=False. Final callback has
    final=True. At a scheduled boundary the last fully updated model is emitted;
    actual elapsed time is authoritative and a partial rollout is not optimized.
    No rollout or optimizer minibatch starts after the deadline. The unavoidable
    final serialization callback can complete slightly after it and is recorded.

    parent_checkpoint warm-starts weights and optimizer for adaptation. It does
    not restore exact in-flight simulator state and is not bitwise continuation.

    ``snapshot`` is an observation side channel, not a hyperparameter: it is not
    part of the returned ``config``. Every ``snapshot_interval`` seconds of the
    governed clock (first at the first step boundary, so a warm start is visible
    immediately) the callback receives a dict with a detached CPU copy of the
    weights (``state_dict``), the best-so-far copy (``best_state_dict``,
    ``best_index`` and its provenance ``best_snapshot``) selected by the highest
    mean distance of the most recent training episodes at snapshot time (a
    training-signal selector, not held-out evaluation), ``is_best``, ``index``,
    ``elapsed``, ``optimizer_updates``, ``environment_steps``, ``episodes`` and
    ``mean_episode_distance``. The selector lags the weights: the rolling window
    holds up to 100 episodes completed *before* the snapshot, produced by the
    weights of earlier iterations, so a snapshot taken right after a degrading
    update can still rank best. Best-related values live only in the payload;
    metric rows are identical with or without a snapshot callback apart from the
    counters below. The copy and callback time count inside the governed clock
    and are reported per row as ``snapshot_copy_seconds`` /
    ``snapshot_callback_seconds``. A callback error is counted in
    ``snapshot_errors`` and never interrupts training.
    """
    if seconds <= 0 or rollout_steps < 1 or num_envs < 1 or epochs < 1 or minibatch_size < 1:
        raise ValueError("Budget, rollout dimensions, epochs and batch size must be positive")
    if snapshot is not None and (snapshot_interval is None or not snapshot_interval > 0):
        raise ValueError("A snapshot callback requires a positive snapshot_interval")
    start = time.monotonic()
    deadline = start + seconds
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = VectorHillEnv(num_envs, seed, profile, terrain, randomization, stack, max_steps)
    observations, _ = env.reset(seed)
    model = ActorCritic(env.observation_dim, hidden_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    config = {
        "algorithm": "ppo",
        "algorithm_version": "original-0.1.0",
        "seed": seed,
        "seconds": seconds,
        "num_envs": num_envs,
        "profile": profile,
        "terrain": terrain,
        "randomization": randomization,
        "stack": stack,
        "max_steps": max_steps,
        "hidden_size": hidden_size,
        "rollout_steps": rollout_steps,
        "minibatch_size": minibatch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_coef": clip_coef,
        "entropy_coef": entropy_coef,
        "device": device,
        "torch_threads": torch_threads,
        "checkpoint_times": list(checkpoint_times),
        "log_interval": log_interval,
        "parent_checkpoint": parent_checkpoint,
        "simulator_version": SIMULATOR_VERSION,
        "calibration_version": "uncalibrated",
        "observation_source": "idealized_simulator_state",
        "action_space": "two_independent_bernoulli_pedals",
    }
    if parent_checkpoint:
        from .policies import load_policy

        parent = load_policy(parent_checkpoint, device=device)
        model.load_state_dict(parent.state_dict())
        if "optimizer" in parent.training_state:
            optimizer.load_state_dict(parent.training_state["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
    model.config = config
    metrics: list[dict] = []
    recent: deque = deque(maxlen=100)
    targets = deque(sorted({float(t) for t in checkpoint_times if 0 < t <= seconds}))
    environment_steps = episodes = updates = iterations = 0
    next_log = log_interval
    total_inference = total_environment = total_optimizer = 0.0
    last_loss = last_entropy = last_kl = 0.0
    next_snapshot = 0.0
    snapshot_count = snapshot_errors = 0
    snapshot_copy_seconds = snapshot_callback_seconds = 0.0
    best_mean: float | None = None
    best_state: dict | None = None
    best_snapshot: dict | None = None

    def emit(final=False, target=None):
        elapsed = time.monotonic() - start
        row = {
            "wall_clock_seconds": elapsed,
            "environment_steps": environment_steps,
            "env_steps": environment_steps,
            "training_steps": environment_steps,
            "episodes": episodes,
            "optimizer_updates": updates,
            "iterations": iterations,
            "environment_steps_per_second": environment_steps / max(elapsed, 1e-9),
            "mean_episode_distance": float(np.mean([e["distance"] for e in recent]))
            if recent
            else None,
            "mean_episode_return": float(np.mean([e["return_"] for e in recent]))
            if recent
            else None,
            "policy_loss": last_loss,
            "entropy": last_entropy,
            "approx_kl": last_kl,
            "inference_seconds": total_inference,
            "environment_seconds": total_environment,
            "optimizer_seconds": total_optimizer,
            "checkpoint": target is not None or final,
            "checkpoint_target_seconds": target,
            "requested_checkpoint_seconds": target,
            "final": final,
            "snapshot_count": snapshot_count,
            "snapshot_copy_seconds": snapshot_copy_seconds,
            "snapshot_callback_seconds": snapshot_callback_seconds,
            "snapshot_errors": snapshot_errors,
        }
        metrics.append(row)
        model.training_state = {
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "environment_steps": environment_steps,
            "episodes": episodes,
            "optimizer_updates": updates,
            "elapsed": elapsed,
        }
        if callback:
            callback(row, model, elapsed)

    def take_snapshot(elapsed):
        nonlocal snapshot_count, snapshot_errors, snapshot_copy_seconds
        nonlocal snapshot_callback_seconds, best_mean, best_state, best_snapshot, next_snapshot
        t = time.perf_counter()
        # A fresh detached CPU copy: the observer never aliases live parameters.
        state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        mean = float(np.mean([e["distance"] for e in recent])) if recent else None
        is_best = mean is not None and (best_mean is None or mean > best_mean)
        provenance = {
            "index": snapshot_count,
            "elapsed": elapsed,
            "optimizer_updates": updates,
            "environment_steps": environment_steps,
            "episodes": episodes,
            "iterations": iterations,
            "mean_episode_distance": mean,
        }
        if is_best:
            best_mean, best_state, best_snapshot = mean, state, provenance
        snapshot_copy_seconds += time.perf_counter() - t
        payload = {
            **provenance,
            "state_dict": state,
            "best_state_dict": best_state,
            "best_index": best_snapshot["index"] if best_snapshot else None,
            "best_snapshot": best_snapshot,
            "is_best": is_best,
            "best_mean_episode_distance": best_mean,
            "algorithm_version": config["algorithm_version"],
            "seed": seed,
        }
        t = time.perf_counter()
        try:
            snapshot(payload)
        except Exception:
            snapshot_errors += 1
            logging.getLogger(__name__).warning("Snapshot callback failed", exc_info=True)
        snapshot_callback_seconds += time.perf_counter() - t
        snapshot_count += 1
        next_snapshot = elapsed + snapshot_interval

    def boundaries():
        nonlocal next_log
        elapsed = time.monotonic() - start
        emitted = False
        while targets and elapsed >= targets[0]:
            emit(target=targets.popleft())
            emitted = True
            elapsed = time.monotonic() - start
        if elapsed >= next_log:
            if not emitted:
                emit()
            next_log = elapsed + log_interval
        if snapshot is not None and elapsed >= next_snapshot:
            take_snapshot(elapsed)

    while time.monotonic() < deadline:
        obs_buffer, bit_buffer, logp_buffer, value_buffer = [], [], [], []
        reward_buffer, next_value_buffer, term_buffer, done_buffer = [], [], [], []
        for _ in range(rollout_steps):
            boundaries()
            if time.monotonic() >= deadline:
                break
            t = time.monotonic()
            obs_tensor = torch.as_tensor(observations, device=device)
            with torch.no_grad():
                distribution, values = model.distribution_value(obs_tensor)
                bits = distribution.sample()
                logp = distribution.log_prob(bits).sum(-1)
            actions = (bits[:, 0].long() + 2 * bits[:, 1].long()).cpu().numpy()
            total_inference += time.monotonic() - t
            t = time.monotonic()
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            total_environment += time.monotonic() - t
            done = terminated | truncated
            bootstrap_obs = next_obs.copy()
            if done.any():
                bootstrap_obs[done] = info["final_observation"][done]
            t = time.monotonic()
            with torch.no_grad():
                _, next_values = model.distribution_value(
                    torch.as_tensor(bootstrap_obs, device=device)
                )
            total_inference += time.monotonic() - t
            obs_buffer.append(obs_tensor)
            bit_buffer.append(bits)
            logp_buffer.append(logp)
            value_buffer.append(values)
            reward_buffer.append(torch.as_tensor(rewards, device=device))
            next_value_buffer.append(next_values)
            term_buffer.append(torch.as_tensor(terminated, device=device, dtype=torch.float32))
            done_buffer.append(torch.as_tensor(done, device=device, dtype=torch.float32))
            recent.extend(info["episodes"])
            episodes += len(info["episodes"])
            environment_steps += num_envs
            observations = next_obs
        if not obs_buffer or time.monotonic() >= deadline:
            break
        t = time.monotonic()
        advantages, returns = compute_gae(
            torch.stack(reward_buffer),
            torch.stack(value_buffer),
            torch.stack(next_value_buffer),
            torch.stack(term_buffer),
            torch.stack(done_buffer),
            gamma,
            gae_lambda,
        )
        obs_batch = torch.stack(obs_buffer).flatten(0, 1)
        bits_batch = torch.stack(bit_buffer).flatten(0, 1)
        old_logp = torch.stack(logp_buffer).flatten()
        advantages, returns = advantages.flatten(), returns.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        count = len(advantages)
        for _ in range(epochs):
            order = rng.permutation(count)
            for offset in range(0, count, minibatch_size):
                if time.monotonic() >= deadline:
                    break
                indices = torch.as_tensor(order[offset : offset + minibatch_size], device=device)
                distribution, values = model.distribution_value(obs_batch[indices])
                new_logp = distribution.log_prob(bits_batch[indices]).sum(-1)
                log_ratio = new_logp - old_logp[indices]
                ratio = log_ratio.exp()
                policy_loss = torch.maximum(
                    -advantages[indices] * ratio,
                    -advantages[indices] * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                ).mean()
                value_loss = 0.5 * (values - returns[indices]).square().mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = policy_loss + value_loss * 0.5 - entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                updates += 1
                last_loss, last_entropy = float(policy_loss.detach()), float(entropy.detach())
                last_kl = float(((ratio - 1) - log_ratio).mean().detach())
            if last_kl > 0.03 or time.monotonic() >= deadline:
                break
        total_optimizer += time.monotonic() - t
        iterations += 1
        boundaries()
    boundaries()
    emit(final=True)
    model.eval()
    return TrainingResult(
        model, metrics, config, time.monotonic() - start, environment_steps, episodes, updates
    )
