"""Small independently parameterized screen policy and sequential episode CEM.

Only externally measured episode scores update the search. This module performs
no screen capture or input and does not accept privileged surrogate checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScreenLinearPolicy:
    schema_id: str
    feature_names: tuple[str, ...]
    weights: np.ndarray
    feature_scales: np.ndarray

    def __post_init__(self):
        self.feature_names = tuple(self.feature_names)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.feature_scales = np.asarray(self.feature_scales, dtype=np.float64)
        count = len(self.feature_names)
        if not self.schema_id or not 1 <= count <= 64 or len(set(self.feature_names)) != count:
            raise ValueError("A named schema and 1..64 unique features are required")
        if self.weights.shape != (2, 2 * count + 1) or not np.isfinite(self.weights).all():
            raise ValueError("Expected two finite linear heads over values, masks, and bias")
        if (
            self.feature_scales.shape != (count,)
            or not np.isfinite(self.feature_scales).all()
            or (self.feature_scales <= 0).any()
        ):
            raise ValueError("Feature scales must be finite and positive")

    def action(self, observation, available_names):
        """Return independent pedal bits; UI/focus authorization remains external.

        Validity bits are observed inputs. Missing measurements are represented
        by zero value and false mask, never asserted as measured zeros. If none
        of the selected features is available, release both pedals.
        """
        if observation.schema_id != self.schema_id:
            raise ValueError("The policy and screen observation schemas differ")
        index = {name: i for i, name in enumerate(available_names)}
        if any(name not in index for name in self.feature_names):
            raise ValueError("Required feature names are absent from this schema")
        columns = [index[name] for name in self.feature_names]
        valid = np.asarray(observation.valid, dtype=bool)[columns]
        if not valid.any():
            return 0
        values = np.asarray(observation.values, dtype=np.float64)[columns]
        if not np.isfinite(values[valid]).all():
            raise ValueError("A supposedly valid feature is nonfinite")
        values = np.where(valid, values / self.feature_scales, 0.0)
        inputs = np.r_[np.clip(values, -3, 3), valid.astype(float), 1.0]
        pressed = self.weights @ inputs > 0
        return int(pressed[0]) | (int(pressed[1]) << 1)

    def as_dict(self):
        return {
            "type": "screen-linear-independent-pedals-1",
            "schema_id": self.schema_id,
            "feature_names": list(self.feature_names),
            "feature_scales": self.feature_scales.tolist(),
            "weights": self.weights.tolist(),
            "observation_source": "masked_rendered_pixel_measurements",
            "action_order": ["gas", "brake"],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "screen-linear-independent-pedals-1":
            raise ValueError("Unsupported screen policy format")
        return cls(
            data["schema_id"], tuple(data["feature_names"]), data["weights"], data["feature_scales"]
        )


class EpisodeCEM:
    """Sequential candidates, complete-generation updates, explicit noisy scores.

    The incumbent and distribution mean are reevaluated in every population.
    The incumbent is selected within the latest complete generation; an isolated
    all-time maximum is not treated as a noise-free estimate. Invalid or aborted
    episodes must not be passed to tell(). Snapshot the pending candidate before
    a rollout so interruption can retain both its identity and experiment cost.
    """

    def __init__(
        self,
        initial,
        *,
        seed=0,
        population=8,
        elite_count=3,
        initial_std=0.5,
        minimum_std=0.08,
        smoothing=0.5,
    ):
        initial = np.asarray(initial, dtype=np.float64)
        if (
            initial.ndim != 2
            or not initial.size
            or initial.size > 258
            or not np.isfinite(initial).all()
        ):
            raise ValueError("Initial parameter array must be a finite small matrix")
        if type(population) is not int or not 4 <= population <= 64:
            raise ValueError("Population must be an integer in 4..64")
        if type(elite_count) is not int or not 2 <= elite_count < population:
            raise ValueError("At least two elites and one nonelite are required")
        if not (0 < minimum_std <= initial_std <= 5 and 0 < smoothing <= 1):
            raise ValueError("Invalid CEM scale or smoothing")
        self.rng = np.random.default_rng(seed)
        self.population, self.elite_count = population, elite_count
        self.minimum_std, self.smoothing = minimum_std, smoothing
        self.mean, self.incumbent = initial.copy(), initial.copy()
        self.std = np.full_like(initial, initial_std)
        self.generation = 0
        self.completed_episodes = 0
        self.incumbent_score = None
        self._candidates = None
        self._scores = []

    def ask(self):
        if self._candidates is None:
            self._candidates = (
                self.mean + self.rng.standard_normal((self.population, *self.mean.shape)) * self.std
            )
            self._candidates[0] = self.incumbent
            self._candidates[1] = self.mean
            self._scores = []
        index = len(self._scores)
        return f"{self.generation}:{index}", self._candidates[index].copy()

    def tell(self, candidate_id, score):
        if self._candidates is None or candidate_id != f"{self.generation}:{len(self._scores)}":
            raise ValueError("Score does not identify the pending candidate")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score):
            raise ValueError("An observed finite score is required")
        self._scores.append(float(score))
        self.completed_episodes += 1
        if len(self._scores) != self.population:
            return None
        scores = np.asarray(self._scores)
        order = np.argsort(scores, kind="stable")
        elite = self._candidates[order[-self.elite_count :]]
        rate = self.smoothing
        self.mean = (1 - rate) * self.mean + rate * elite.mean(axis=0)
        self.std = np.clip((1 - rate) * self.std + rate * elite.std(axis=0), self.minimum_std, 5.0)
        # The first tied maximum preserves the reevaluated incumbent.
        winner = int(scores.argmax())
        self.incumbent = self._candidates[winner].copy()
        self.incumbent_score = float(scores[winner])
        summary = {
            "generation": self.generation,
            "scores": self._scores.copy(),
            "selected_candidate": f"{self.generation}:{winner}",
            "selected_training_score": self.incumbent_score,
            "mean_score": float(scores.mean()),
            "mean_search_std": float(self.std.mean()),
        }
        self.generation += 1
        self._candidates, self._scores = None, []
        return summary

    def state_dict(self):
        return {
            "type": "sequential-episode-cem-1",
            "population": self.population,
            "elite_count": self.elite_count,
            "minimum_std": self.minimum_std,
            "smoothing": self.smoothing,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "incumbent": self.incumbent.tolist(),
            "incumbent_score": self.incumbent_score,
            "generation": self.generation,
            "completed_episodes": self.completed_episodes,
            "rng_state": self.rng.bit_generator.state,
            "pending_candidates": None if self._candidates is None else self._candidates.tolist(),
            "pending_scores": self._scores.copy(),
        }

    @classmethod
    def from_state_dict(cls, data):
        if data.get("type") != "sequential-episode-cem-1":
            raise ValueError("Unsupported optimizer snapshot")
        optimizer = cls(
            data["mean"],
            population=data["population"],
            elite_count=data["elite_count"],
            initial_std=data["minimum_std"],
            minimum_std=data["minimum_std"],
            smoothing=data["smoothing"],
        )
        for name in ("std", "incumbent"):
            value = np.asarray(data[name], dtype=float)
            if value.shape != optimizer.mean.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid snapshot {name}")
            setattr(optimizer, name, value.copy())
        if ((optimizer.std < optimizer.minimum_std) | (optimizer.std > 5)).any():
            raise ValueError("Snapshot deviation is out of bounds")
        for name in ("generation", "completed_episodes"):
            value = data[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid snapshot {name}")
            setattr(optimizer, name, value)
        score = data["incumbent_score"]
        if score is not None and (type(score) not in (int, float) or not np.isfinite(score)):
            raise ValueError("Invalid incumbent score")
        optimizer.incumbent_score = score
        pending = data["pending_candidates"]
        scores = data["pending_scores"]
        if (
            not isinstance(scores, list)
            or len(scores) >= optimizer.population
            or any(type(s) not in (int, float) or not np.isfinite(s) for s in scores)
        ):
            raise ValueError("Invalid pending scores")
        if pending is not None:
            pending = np.asarray(pending, dtype=float)
            if (
                pending.shape != (optimizer.population, *optimizer.mean.shape)
                or not np.isfinite(pending).all()
            ):
                raise ValueError("Invalid pending population")
            optimizer._candidates = pending.copy()
        elif scores:
            raise ValueError("Pending scores require pending candidates")
        if optimizer.completed_episodes != optimizer.generation * optimizer.population + len(
            scores
        ):
            raise ValueError("Snapshot episode and generation counts disagree")
        optimizer._scores = scores.copy()
        if (
            not isinstance(data["rng_state"], dict)
            or data["rng_state"].get("bit_generator") != "PCG64"
        ):
            raise ValueError("Unexpected random generator")
        optimizer.rng.bit_generator.state = data["rng_state"]
        return optimizer
