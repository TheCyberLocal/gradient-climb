"""A vectorized two-contact hill vehicle surrogate, not a recreation of HCR.

Units are nominal metres, seconds and kilograms. Parameters are hypotheses until
calibrated against independently measured screen trajectories. Observations are
idealized state estimates and must not be called validated pixel observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SIMULATOR_VERSION = "surrogate-0.1.0"


@dataclass(frozen=True)
class Vehicle:
    mass: float = 1.0
    inertia: float = 0.48
    wheelbase: float = 1.35
    engine: float = 11.0
    brake: float = 15.0
    stiffness: float = 95.0
    damping: float = 8.0
    traction: float = 1.15
    air_torque: float = 2.8


VEHICLES = {
    "default": Vehicle(),
    "heavy": Vehicle(
        mass=1.55,
        inertia=0.9,
        wheelbase=1.65,
        engine=10.2,
        stiffness=135.0,
        damping=12.0,
        traction=0.95,
        air_torque=2.0,
    ),
    "agile": Vehicle(
        mass=0.8,
        inertia=0.29,
        wheelbase=1.1,
        engine=12.0,
        stiffness=85.0,
        damping=7.0,
        traction=1.3,
        air_torque=3.8,
    ),
}

TERRAINS = {
    "train": ((1.8, 1.0, 0.40), (0.085, 0.23, 0.57)),
    "rolling": ((3.4, 1.7, 0.30), (0.055, 0.16, 0.43)),
    "rough": ((2.4, 1.25, 0.75), (0.09, 0.29, 0.72)),
}

FEATURE_NAMES = (
    "vx",
    "vy",
    "sin_pitch",
    "cos_pitch",
    "angular_velocity",
    "fuel",
    "ground_clearance",
    "rear_contact",
    "front_contact",
    "rear_slope",
    "front_slope",
    "height_ahead_2",
    "height_ahead_4",
    "height_ahead_7",
    "height_ahead_11",
    "height_ahead_16",
    "slope_ahead_2",
    "slope_ahead_4",
    "slope_ahead_7",
    "slope_ahead_11",
    "slope_ahead_16",
    "previous_gas",
    "previous_brake",
    "episode_fraction",
)


class VectorHillEnv:
    """NumPy batch environment with same-step automatic episode reset.

    Actions are integer joint states: 0=00, 1=gas, 2=brake, 3=both;
    alternatively pass a (num_envs, 2) array of independent gas/brake bits.
    Each action lasts exactly ``dt * substeps`` seconds. A previous-pedal state
    in every temporal observation preserves order; repeated steps create holds.

    ``step`` follows the Gymnasium five-result convention. Terminated/truncated
    flags and episode records describe the episode that just ended, whereas the
    returned observation already contains the reset episode. Terminal states are
    in ``info['final_observation']`` with ``info['_final_observation']`` as mask.
    A time-limit truncation is distinct from physical termination for bootstraps.
    This is intentionally a compact batch API, not a Gymnasium VectorEnv subclass.
    """

    def __init__(
        self,
        num_envs: int = 64,
        seed: int = 0,
        profile: str = "default",
        terrain: str = "train",
        randomization: bool = False,
        stack: int = 4,
        max_steps: int = 1000,
        dt: float = 0.02,
        substeps: int = 3,
    ):
        if num_envs < 1 or stack < 1 or max_steps < 1 or dt <= 0 or substeps < 1:
            raise ValueError("Environment dimensions and timesteps must be positive")
        if profile not in VEHICLES or terrain not in TERRAINS:
            raise ValueError(f"Unknown profile/terrain: {profile}/{terrain}")
        self.num_envs, self.stack, self.max_steps = num_envs, stack, max_steps
        self.profile, self.terrain, self.randomization = profile, terrain, randomization
        self.dt, self.substeps = dt, substeps
        self.action_duration = dt * substeps
        self.observation_dim = len(FEATURE_NAMES) * stack
        self.config = {
            "num_envs": num_envs,
            "seed": seed,
            "profile": profile,
            "terrain": terrain,
            "randomization": randomization,
            "stack": stack,
            "max_steps": max_steps,
            "dt": dt,
            "substeps": substeps,
        }
        self._rng = np.random.default_rng(seed)
        for name in (
            "x",
            "y",
            "vx",
            "vy",
            "theta",
            "omega",
            "fuel",
            "best_x",
            "return_",
            "next_fuel",
            "stalled",
        ):
            setattr(self, name, np.zeros(num_envs, dtype=np.float64))
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.pedals = np.zeros((num_envs, 2), dtype=np.float64)
        self.contacts = np.zeros((num_envs, 2), dtype=bool)
        self.phase = np.zeros((num_envs, 3))
        self.amplitude = np.zeros((num_envs, 3))
        self.frequency = np.zeros((num_envs, 3))
        self.parameters = {key: np.zeros(num_envs) for key in Vehicle.__dataclass_fields__}
        self.history = np.zeros((num_envs, stack, len(FEATURE_NAMES)), dtype=np.float32)
        self.episode_count = 0
        self.reset()

    def reset(self, seed: int | list[int] | None = None) -> tuple[np.ndarray, dict]:
        if isinstance(seed, (list, tuple, np.ndarray)):
            if len(seed) != self.num_envs:
                raise ValueError("Per-environment seed count must equal num_envs")
            for index, item in enumerate(seed):
                self._rng = np.random.default_rng(int(item))
                self._reset_indices(np.array([index]))
            return self.history.reshape(self.num_envs, -1).copy(), {
                "simulator_version": SIMULATOR_VERSION
            }
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_indices(np.arange(self.num_envs))
        return self.history.reshape(self.num_envs, -1).copy(), {
            "simulator_version": SIMULATOR_VERSION
        }

    def _reset_indices(self, indices: np.ndarray) -> None:
        count = len(indices)
        if not count:
            return
        self.phase[indices] = self._rng.uniform(0, 2 * np.pi, (count, 3))
        amplitudes, frequencies = TERRAINS[self.terrain]
        self.amplitude[indices] = np.asarray(amplitudes) * self._rng.uniform(0.75, 1.25, (count, 3))
        self.frequency[indices] = np.asarray(frequencies) * self._rng.uniform(0.8, 1.2, (count, 3))
        for key, value in vars(VEHICLES[self.profile]).items():
            self.parameters[key][indices] = value * (
                self._rng.uniform(0.8, 1.2, count) if self.randomization else 1.0
            )
        for name in ("x", "vx", "vy", "theta", "omega", "best_x", "return_", "stalled"):
            getattr(self, name)[indices] = 0
        self.y[indices] = 0.62
        self.fuel[indices] = 1
        self.next_fuel[indices] = 80
        self.steps[indices] = 0
        self.pedals[indices] = 0
        self.contacts[indices] = True
        frame = self._observe_frame()[indices]
        self.history[indices] = frame[:, None, :]

    def ground(
        self, positions: np.ndarray, indices: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Terrain height and analytic gradient; first five metres are flat."""
        positions = np.asarray(positions)
        phase = self.phase if indices is None else self.phase[indices]
        amp = self.amplitude if indices is None else self.amplitude[indices]
        freq = self.frequency if indices is None else self.frequency[indices]
        # positions shape: (environments,) or (environments, sample_points).
        extra = positions.ndim - 1
        expanded = (slice(None),) + (None,) * extra + (slice(None),)
        phase, amp, freq = phase[expanded], amp[expanded], freq[expanded]
        q = np.maximum(positions - 5.0, 0)
        wave = q[..., None] * freq + phase
        base = np.sum(amp * (np.sin(wave) - np.sin(phase)), axis=-1)
        derivative = np.sum(amp * freq * np.cos(wave), axis=-1)
        blend = 1 - np.exp(-q / 4)
        slope = derivative * blend + base * np.exp(-q / 4) / 4
        return base * blend, np.where(positions > 5, slope, 0.0)

    def _observe_frame(self) -> np.ndarray:
        ground, _ = self.ground(self.x)
        half = self.parameters["wheelbase"][:, None] / 2
        wheel_x = self.x[:, None] + np.array([-1, 1]) * half * np.cos(self.theta[:, None])
        _, wheel_slope = self.ground(wheel_x)
        ahead, slopes = self.ground(self.x[:, None] + np.array([2, 4, 7, 11, 16]))
        frame = np.column_stack(
            (
                self.vx / 12,
                self.vy / 8,
                np.sin(self.theta),
                np.cos(self.theta),
                self.omega / 5,
                self.fuel,
                (self.y - ground) / 3,
                self.contacts,
                wheel_slope,
                (ahead - ground[:, None]) / 6,
                slopes,
                self.pedals,
                self.steps / self.max_steps,
            )
        )
        return np.clip(frame, -5, 5).astype(np.float32)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions)
        if actions.shape == (self.num_envs,):
            if not np.all((actions == actions.astype(int)) & (actions >= 0) & (actions <= 3)):
                raise ValueError("Joint actions must be integers in [0, 3]")
            actions = actions.astype(np.int64)
            pedals = np.column_stack((actions & 1, (actions >> 1) & 1))
        elif actions.shape == (self.num_envs, 2) and np.all((actions == 0) | (actions == 1)):
            pedals = actions
        else:
            raise ValueError(
                f"Expected ({self.num_envs},) joint actions or ({self.num_envs},2) pedal bits"
            )
        self.pedals[:] = pedals
        gas, brake = self.pedals.T
        p = self.parameters
        half = p["wheelbase"][:, None] * np.array([-0.5, 0.5])
        old_best = self.best_x.copy()
        for _ in range(self.substeps):
            sn, cs = np.sin(self.theta[:, None]), np.cos(self.theta[:, None])
            rx, ry = half * cs + 0.22 * sn, half * sn - 0.22 * cs
            gx, slope = self.ground(self.x[:, None] + rx)
            compression = gx + 0.27 - (self.y[:, None] + ry)
            normalizer = np.sqrt(1 + slope * slope)
            nx, ny = -slope / normalizer, 1 / normalizer
            wheel_vx = self.vx[:, None] - self.omega[:, None] * ry
            wheel_vy = self.vy[:, None] + self.omega[:, None] * rx
            normal_velocity = wheel_vx * nx + wheel_vy * ny
            normal = np.where(
                compression > 0,
                np.maximum(
                    0,
                    p["stiffness"][:, None] * compression - p["damping"][:, None] * normal_velocity,
                ),
                0,
            )
            normal = np.minimum(normal, 70 * p["mass"][:, None])
            self.contacts[:] = compression > -0.025
            tangent_velocity = wheel_vx * ny - wheel_vy * nx
            # Brake opposes forward motion, becoming reverse torque near rest.
            motor = (
                p["mass"][:, None]
                * (
                    gas[:, None] * p["engine"][:, None]
                    - brake[:, None] * p["brake"][:, None] * np.tanh((tangent_velocity + 0.6) / 1.4)
                )
                / 2
            )
            friction = np.clip(
                motor - 0.035 * tangent_velocity,
                -p["traction"][:, None] * normal,
                p["traction"][:, None] * normal,
            )
            fx = normal * nx + friction * ny
            fy = normal * ny - friction * nx
            self.vx += self.dt * (
                np.sum(fx, axis=1) / p["mass"] - 0.008 * self.vx * np.abs(self.vx)
            )
            self.vy += self.dt * (np.sum(fy, axis=1) / p["mass"] - 9.81 - 0.02 * self.vy)
            torque = np.sum(rx * fy - ry * fx, axis=1)
            airborne = ~np.any(self.contacts, axis=1)
            torque += airborne * p["air_torque"] * (gas - brake)
            self.omega += self.dt * (torque / p["inertia"] - 0.25 * self.omega)
            self.vx = np.clip(self.vx, -18, 30)
            self.vy = np.clip(self.vy, -30, 30)
            self.omega = np.clip(self.omega, -15, 15)
            self.x += self.vx * self.dt
            self.y += self.vy * self.dt
            self.theta = (self.theta + self.omega * self.dt + np.pi) % (2 * np.pi) - np.pi
        self.steps += 1
        self.best_x = np.maximum(self.best_x, self.x)
        progress = self.best_x - old_best
        self.stalled = np.where(progress > 0.01, 0, self.stalled + self.action_duration)
        self.fuel -= self.action_duration * (0.004 + 0.009 * gas + 0.002 * brake)
        refuel = self.x >= self.next_fuel
        self.fuel = np.minimum(1, self.fuel + 0.45 * refuel)
        self.next_fuel += 80 * refuel
        ground, _ = self.ground(self.x)
        crash = ((np.cos(self.theta) < 0.15) & (self.y - ground < 0.9)) | (self.y - ground < 0.14)
        out_of_fuel = self.fuel <= 0
        reversed_out = self.x < -8
        stalled = self.stalled > 12
        terminated = crash | out_of_fuel | reversed_out | stalled
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated
        reward = progress * 0.1 - 0.0004 - terminated.astype(float) * 1.0
        self.return_ += reward
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = self._observe_frame()
        final_observation = self.history.reshape(self.num_envs, -1).copy() if done.any() else None
        episodes = []
        for i in np.flatnonzero(done):
            reason = (
                "crash"
                if crash[i]
                else "fuel"
                if out_of_fuel[i]
                else "reverse_boundary"
                if reversed_out[i]
                else "stalled"
                if stalled[i]
                else "time_limit"
            )
            episodes.append(
                {
                    "env_index": int(i),
                    "return_": float(self.return_[i]),
                    "distance": float(self.best_x[i]),
                    "length": int(self.steps[i]),
                    "seconds": float(self.steps[i] * self.action_duration),
                    "termination": reason,
                    "terminated": bool(terminated[i]),
                    "truncated": bool(truncated[i]),
                }
            )
        self.episode_count += len(episodes)
        info = {
            "episodes": episodes,
            "final_observation": final_observation,
            "_final_observation": done.copy(),
            "simulator_version": SIMULATOR_VERSION,
        }
        self._reset_indices(np.flatnonzero(done))
        return (
            self.history.reshape(self.num_envs, -1).copy(),
            reward.astype(np.float32),
            terminated,
            truncated,
            info,
        )

    def render(self, index: int = 0, width: int = 960, height: int = 540):
        """Render one selected environment as a PIL image; never called by step."""
        from PIL import Image, ImageDraw

        if not 0 <= index < self.num_envs:
            raise IndexError(index)
        image = Image.new("RGB", (width, height), "#101927")
        draw = ImageDraw.Draw(image)
        scale = width / 30
        left = self.x[index] - 8
        base_y = self.y[index] + 4.5
        positions = np.linspace(left, left + 30, width // 3)
        heights, _ = self.ground(positions[None, :], np.array([index]))
        points = [
            (int((x - left) * scale), int((base_y - y) * scale))
            for x, y in zip(positions, heights[0])
        ]
        draw.polygon(points + [(width, height), (0, height)], fill="#315143")
        draw.line(points, fill="#a8d991", width=3)
        theta = self.theta[index]
        cs, sn = np.cos(theta), np.sin(theta)

        def xy(dx, dy):
            return (
                (self.x[index] + cs * dx - sn * dy - left) * scale,
                (base_y - self.y[index] - sn * dx - cs * dy) * scale,
            )

        body = [xy(-0.9, -0.12), xy(0.9, -0.12), xy(0.65, 0.3), xy(-0.55, 0.3)]
        draw.polygon(body, fill="#fbba5b")
        for wheel in (-0.5, 0.5):
            x, y = xy(wheel * self.parameters["wheelbase"][index], -0.22)
            r = scale * 0.27
            draw.ellipse((x - r, y - r, x + r, y + r), fill="#101318", outline="#b7c6d7", width=3)
        draw.text((22, 20), "GRADIENTCLIMB | UNCALIBRATED RESEARCH SURROGATE", fill="#e2eaf7")
        draw.text(
            (22, 44),
            f"distance {self.best_x[index]:.1f} m  velocity {self.vx[index]:.1f} m/s  "
            f"fuel {self.fuel[index]:.0%}  gas {int(self.pedals[index, 0])}  "
            f"brake {int(self.pedals[index, 1])}  step {self.steps[index]}",
            fill="#b7c6d7",
        )
        return image
