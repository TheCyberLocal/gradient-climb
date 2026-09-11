"""Restricted body-feature student and an uncalibrated analytic render projection.

The teacher may inspect privileged simulator observations. The student only sees
eight body features, validity bits and their history. Analytic projected features
are not measured pixels and schema compatibility is not evidence of calibration.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gradientclimb.perception.screen_features import FEATURE_NAMES

BODY_FEATURES = (
    "body_sin_2angle",
    "body_cos_2angle",
    "body_image_angular_rate_radians_per_second",
    "body_to_terrain_spans",
    "body_terrain_slope_image_right",
    "body_terrain_y_relative_spans_at_1",
    "body_terrain_y_relative_spans_at_2",
    "body_terrain_y_relative_spans_at_3",
)
BODY_SCALES = (1.0, 1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 2.0)
PROJECTION_VERSION = "surrogate-continuous-chassis-body-1"
# The original polygon drawn by VectorHillEnv.render, in nominal local units.
BODY_POLYGON = np.array([[-0.9, -0.12], [0.9, -0.12], [0.65, 0.3], [-0.55, 0.3]])


def polygon_moments(vertices):
    """Uniform filled-polygon centroid/covariance; no raster or pixel fit."""
    p = np.asarray(vertices, dtype=float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = cross.sum() / 2
    center = ((p + q) * cross[:, None]).sum(axis=0) / (6 * area)
    xx = ((p[:, 0] ** 2 + p[:, 0] * q[:, 0] + q[:, 0] ** 2) * cross).sum() / (12 * area)
    yy = ((p[:, 1] ** 2 + p[:, 1] * q[:, 1] + q[:, 1] ** 2) * cross).sum() / (12 * area)
    xy = (
        (2 * p[:, 0] * p[:, 1] + p[:, 0] * q[:, 1] + q[:, 0] * p[:, 1] + 2 * q[:, 0] * q[:, 1])
        * cross
    ).sum() / (24 * area)
    covariance = np.array([[xx, xy], [xy, yy]]) - np.outer(center, center)
    return center, covariance


_CENTER, _COVARIANCE = polygon_moments(BODY_POLYGON)
_AXIS = np.linalg.eigh(_COVARIANCE)[1][:, -1]
_AXIS_ANGLE = math.atan2(_AXIS[1], _AXIS[0])


def project_body_geometry(x, y, theta, ground):
    """Continuous orthographic counterpart of body centroid/PCA/bbox features.

    Image y increases downwards: (surface_y - body_y)/span maps to
    (world_body_y - world_ground_y)/world_span. Uniform camera translation and
    scale cancel. ground receives a (batch, 4) set of x coordinates. No velocity,
    angular velocity, fuel, contacts, reward or previous actions are consumed.
    """
    x, y, theta = (np.asarray(a, dtype=float) for a in (x, y, theta))
    if x.ndim != 1 or y.shape != x.shape or theta.shape != x.shape:
        raise ValueError("Geometry must have matching one-dimensional batches")
    if not all(np.isfinite(a).all() for a in (x, y, theta)):
        raise ValueError("Geometry must be finite")
    cs, sn = np.cos(theta), np.sin(theta)
    center_x = x + cs * _CENTER[0] - sn * _CENTER[1]
    center_y = y + sn * _CENTER[0] + cs * _CENTER[1]
    rotated_x = cs[:, None] * BODY_POLYGON[:, 0] - sn[:, None] * BODY_POLYGON[:, 1]
    rotated_y = sn[:, None] * BODY_POLYGON[:, 0] + cs[:, None] * BODY_POLYGON[:, 1]
    span = np.maximum(np.ptp(rotated_x, axis=1), np.ptp(rotated_y, axis=1))
    surface = np.asarray(ground(center_x[:, None] + span[:, None] * np.arange(4)))
    if surface.shape != (len(x), 4) or not np.isfinite(surface).all():
        raise ValueError("Projected terrain must supply four finite heights per body")
    relative = (center_y[:, None] - surface) / span[:, None]
    angle = theta + _AXIS_ANGLE
    values = np.column_stack(
        (
            np.sin(2 * angle),
            np.cos(2 * angle),
            np.zeros(len(x)),
            relative[:, 0],
            relative[:, 1] - relative[:, 0],
            relative[:, 1:],
        )
    ).astype(np.float32)
    # The selected renderer has a 960x540, 30-unit-wide camera. Out-of-frame
    # terrain is missing, rather than privileged lookahead.
    screen_x = center_x[:, None] - x[:, None] + 8 + span[:, None] * np.arange(4)
    screen_y = y[:, None] + 4.5 - surface
    visible = (screen_x >= 0) & (screen_x <= 30) & (screen_y >= 0) & (screen_y <= 16.875)
    valid = np.ones_like(values, dtype=bool)
    valid[:, 2] = False
    valid[:, 3] = visible[:, 0]
    valid[:, 4] = visible[:, 0] & visible[:, 1]
    valid[:, 5:] = visible[:, 1:]
    values[~valid] = 0
    return values, angle, valid


def pack_frames(values, valid):
    """Normalize ONLY the eight selected features; preserve missingness explicitly."""
    values, valid = np.asarray(values), np.asarray(valid)
    if values.shape != valid.shape or values.shape[-1] != len(BODY_FEATURES):
        raise ValueError("Expected eight feature values and masks per frame")
    if not np.isin(valid, [0, 1]).all():
        raise ValueError("Validity masks must be binary")
    valid = valid.astype(bool)
    if not np.isfinite(values[valid]).all():
        raise ValueError("A supposedly valid body feature is nonfinite")
    normalized = np.where(valid, values / np.asarray(BODY_SCALES), 0.0)
    return np.concatenate((np.clip(normalized, -3, 3), valid), axis=-1).astype(np.float32)


class SurrogateBodyBridge:
    """History of analytic observable counterparts; resets isolate episodes."""

    def __init__(self, num_envs, history=4, max_interval_seconds=0.5):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("A positive integer batch size is required")
        if type(history) is not int or not 1 <= history <= 16:
            raise ValueError("History must be an integer in 1..16")
        if not math.isfinite(max_interval_seconds) or not 0.01 <= max_interval_seconds <= 2:
            raise ValueError("Invalid maximum observation interval")
        self.history, self.max_interval_seconds = history, max_interval_seconds
        self.frames = np.zeros((num_envs, history, 16), dtype=np.float32)
        self.previous_angle = np.zeros(num_envs)
        self.previous_valid = np.zeros(num_envs, dtype=bool)

    def observe(self, env, reset_mask=None):
        count = len(self.frames)
        reset = np.zeros(count, bool) if reset_mask is None else np.asarray(reset_mask, bool)
        if reset.shape != (count,) or env.num_envs != count:
            raise ValueError("Environment/reset batch differs from the bridge")
        dt = env.action_duration
        contiguous = np.isfinite(dt) and 0.01 <= dt <= self.max_interval_seconds
        if not contiguous:
            reset = np.ones(count, bool)
        self.frames[reset] = 0
        self.previous_valid[reset] = False
        values, angle, valid = project_body_geometry(
            env.x, env.y, env.theta, lambda q: env.ground(q)[0]
        )
        difference = angle - self.previous_angle
        change = 0.5 * np.arctan2(np.sin(2 * difference), np.cos(2 * difference))
        valid[:, 2] = self.previous_valid & (np.abs(change) < np.pi / 3) & contiguous
        values[:, 2] = np.where(valid[:, 2], change / dt if contiguous else 0, 0)
        self.frames[:, :-1] = self.frames[:, 1:]
        self.frames[:, -1] = pack_frames(values, valid)
        self.previous_angle = angle
        self.previous_valid[:] = True
        return self.frames.reshape(count, -1).copy()


class ScreenBodyStudent(nn.Module):
    """Independent pedal heads over eight masked body features and their history."""

    def __init__(self, schema_id, history=4, hidden_size=64, max_interval_seconds=0.5):
        super().__init__()
        if not isinstance(schema_id, str) or not schema_id:
            raise ValueError("An explicit destination ScreenObservation.schema_id is required")
        if type(history) is not int or not 1 <= history <= 16:
            raise ValueError("History must be an integer in 1..16")
        if type(hidden_size) is not int or not 1 <= hidden_size <= 1024:
            raise ValueError("Invalid student hidden size")
        if not math.isfinite(max_interval_seconds) or not 0.01 <= max_interval_seconds <= 2:
            raise ValueError("Invalid maximum observation interval")
        self.schema_id, self.history, self.hidden_size = schema_id, history, hidden_size
        self.max_interval_seconds = max_interval_seconds
        self.network = nn.Sequential(
            nn.Linear(16 * history, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        self.config = {
            "algorithm": "screen_body_distillation",
            "schema_id": schema_id,
            "history": history,
            "hidden_size": hidden_size,
            "feature_names": list(BODY_FEATURES),
            "feature_scales": list(BODY_SCALES),
            "max_interval_seconds": max_interval_seconds,
            "projection_version": PROJECTION_VERSION,
            "actor_inputs": "eight body features plus masks, oldest-first history",
            "training_observation_source": "analytic_uncalibrated_surrogate_projection",
            "qualifies_real_game": False,
        }

    def forward(self, vectors):
        if vectors.ndim != 2 or vectors.shape[1] != 16 * self.history:
            raise ValueError("Student input shape must match restricted masked history")
        return self.network(vectors)

    @torch.no_grad()
    def act_vectors(self, vectors):
        inputs = torch.as_tensor(vectors, dtype=torch.float32)
        logits = self(inputs)
        bits = (logits >= 0).to(torch.int64)
        actions = (bits[:, 0] + 2 * bits[:, 1]).numpy()
        # Current orientation is required even when older history remains valid.
        current = inputs.reshape(-1, self.history, 16)[:, -1]
        actions[~(current[:, 8:10] == 1).all(dim=1).numpy()] = 0
        return actions

    def restricted_vector(self, observation, available_names=FEATURE_NAMES):
        if observation.schema_id != self.schema_id:
            raise ValueError("The student and screen observation schemas differ")
        names = tuple(available_names)
        if len(set(names)) != len(names) or any(name not in names for name in BODY_FEATURES):
            raise ValueError("Observation feature names do not identify the body schema")
        frames = np.asarray(observation.vector)
        if frames.shape != (self.history * 2 * len(names),):
            raise ValueError("Screen history shape differs from the trained student")
        frames = frames.reshape(self.history, 2 * len(names))
        columns = np.array([names.index(name) for name in BODY_FEATURES])
        return pack_frames(frames[:, columns], frames[:, len(names) + columns]).reshape(-1)

    def action(self, observation, available_names=FEATURE_NAMES):
        """Native adapters retain sole authority for UI/freshness/input leases."""
        return int(self.act_vectors(self.restricted_vector(observation, available_names)[None])[0])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "screen-body-student-1",
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        config = data["config"]
        if data.get("format") != "screen-body-student-1":
            raise ValueError("Unsupported body-student checkpoint")
        if config["feature_names"] != list(BODY_FEATURES) or config["feature_scales"] != list(
            BODY_SCALES
        ):
            raise ValueError("Checkpoint feature schema is incompatible")
        student = cls(
            config["schema_id"],
            config["history"],
            config["hidden_size"],
            config["max_interval_seconds"],
        )
        student.load_state_dict(data["state_dict"])
        student.config = config
        student.eval()
        return student


def train_student(
    teacher,
    *,
    schema_id,
    seconds,
    seed=500,
    num_envs=64,
    history=4,
    max_interval_seconds=0.5,
    rollout_steps=32,
    epochs=4,
    learning_rate=0.001,
    callback=None,
):
    """Teacher-occupancy soft-label imitation; no reinforcement-learning update.

    Clock includes initialization, data collection, optimization and callback I/O.
    Each teacher target is two detached Bernoulli probabilities. No privileged
    teacher observations are concatenated with, or passed to, the student.
    Imports/checkpoint loading by the caller and held-out evaluation are separate.
    """
    from gradientclimb.simulation import VectorHillEnv

    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Distillation budget must be finite and positive")
    for value in (num_envs, history, rollout_steps, epochs):
        if type(value) is not int or value < 1:
            raise ValueError("Training dimensions must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    start = time.monotonic()
    deadline = start + seconds
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    teacher.eval()
    teacher.requires_grad_(False)
    student = ScreenBodyStudent(schema_id, history, max_interval_seconds=max_interval_seconds)
    student.config.update(
        seed=seed,
        seconds=seconds,
        teacher_config=dict(teacher.config),
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        occupancy_policy="deterministic_teacher",
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    env = VectorHillEnv(num_envs=num_envs, seed=seed, stack=teacher.config.get("stack", 4))
    observations, _ = env.reset(seed)
    bridge = SurrogateBodyBridge(num_envs, history, max_interval_seconds)
    reset = np.ones(num_envs, bool)
    steps = episodes = updates = 0
    last_loss = None
    next_report = start + 5
    while time.monotonic() < deadline:
        vectors, targets = [], []
        for _ in range(rollout_steps):
            if time.monotonic() >= deadline:
                break
            vector = bridge.observe(env, reset)
            with torch.no_grad():
                distribution, _ = teacher.distribution_value(torch.as_tensor(observations))
                target = distribution.probs.detach().clone()
                bits = (target >= 0.5).long()
                actions = (bits[:, 0] + 2 * bits[:, 1]).numpy()
            vectors.append(vector)
            targets.append(target)
            observations, _, terminated, truncated, info = env.step(actions)
            reset = terminated | truncated
            steps += num_envs
            episodes += len(info["episodes"])
        if vectors:
            inputs = torch.as_tensor(np.concatenate(vectors))
            labels = torch.cat(targets)
            for _ in range(epochs):
                if time.monotonic() >= deadline:
                    break
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(student(inputs), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                updates += 1
                last_loss = float(loss.detach())
        now = time.monotonic()
        if callback and now >= next_report:
            callback(
                {
                    "environment_steps": steps,
                    "episodes": episodes,
                    "optimizer_updates": updates,
                    "bce": last_loss,
                },
                student,
                now - start,
            )
            next_report = now + 5
    elapsed = time.monotonic() - start
    student.eval()
    return student, {
        "training_clock_seconds": elapsed,
        "environment_steps": steps,
        "episodes": episodes,
        "optimizer_updates": updates,
        "bce": last_loss,
    }


def evaluate_student(
    student,
    teacher,
    seeds,
    *,
    occupancy="student",
    profile="default",
    terrain="train",
    max_steps=1000,
):
    """Paired action agreement and distance on first held-out episodes only."""
    from gradientclimb.evaluation.benchmark import summarize
    from gradientclimb.simulation import SIMULATOR_VERSION, VectorHillEnv

    seeds = list(seeds)
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Evaluation needs unique nonnegative integer episode seeds")
    if occupancy not in {"student", "teacher"}:
        raise ValueError("Occupancy must identify teacher or student")
    start = time.monotonic()
    env = VectorHillEnv(
        len(seeds),
        seeds[0],
        profile,
        terrain,
        stack=teacher.config.get("stack", 4),
        max_steps=max_steps,
    )
    observations, _ = env.reset(seeds)
    bridge = SurrogateBodyBridge(len(seeds), student.history, student.max_interval_seconds)
    active, reset = np.ones(len(seeds), bool), np.ones(len(seeds), bool)
    episodes, action_counts = [], np.zeros(4, int)
    joint_agreement = bit_agreement = observations_count = 0
    teacher.eval()
    student.eval()
    while active.any():
        student_actions = student.act_vectors(bridge.observe(env, reset))
        teacher_actions = teacher.act(observations)
        joint_agreement += int((student_actions[active] == teacher_actions[active]).sum())
        for bit in (1, 2):
            bit_agreement += int(
                ((student_actions[active] & bit) == (teacher_actions[active] & bit)).sum()
            )
        observations_count += int(active.sum())
        actions = student_actions if occupancy == "student" else teacher_actions
        np.add.at(action_counts, actions[active], 1)
        observations, _, terminated, truncated, info = env.step(actions)
        reset = terminated | truncated
        for episode in info["episodes"]:
            index = episode["env_index"]
            if active[index]:
                episodes.append({**episode, "seed": seeds[index]})
                active[index] = False
    episodes.sort(key=lambda item: seeds.index(item["seed"]))
    distance = summarize([episode["distance"] for episode in episodes])
    return {
        "scope": "uncalibrated_analytic_body_projection",
        "qualifies_real_game": False,
        "simulator_version": SIMULATOR_VERSION,
        "projection_version": PROJECTION_VERSION,
        "occupancy": occupancy,
        "seeds": seeds,
        "profile": profile,
        "terrain": terrain,
        "max_steps": max_steps,
        "episodes": episodes,
        "summary": {"distance": distance},
        "mean_distance": distance["mean"],
        "median_distance": distance["median"],
        "joint_action_agreement": joint_agreement / observations_count,
        "pedal_bit_agreement": bit_agreement / (2 * observations_count),
        "agreement_observations": observations_count,
        "action_counts": action_counts.tolist(),
        "termination_counts": dict(Counter(e["termination"] for e in episodes)),
        "wall_clock_seconds": time.monotonic() - start,
    }
