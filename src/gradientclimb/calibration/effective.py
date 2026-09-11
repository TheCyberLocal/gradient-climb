"""Fit a restricted input-response model from observed position and pitch.

This is an effective model for controlled flat-ground or airborne probes, not a
full vehicle/contact simulator. It cannot identify mass, suspension, fuel, slope
forces, or collision dynamics. Pixel x must be camera-compensated; raw car screen
x is not progress. The model fits separate gas, brake and simultaneous-input terms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

from gradientclimb.experiments.schemas import CalibrationRecord

MODEL_VERSION = "effective-input-response-1"
PARAMETER_NAMES = (
    "gas_acceleration",
    "brake_acceleration",
    "both_acceleration",
    "linear_drag",
    "gas_angular_acceleration",
    "brake_angular_acceleration",
    "both_angular_acceleration",
    "angular_drag",
)


@dataclass(frozen=True)
class MotionTrajectory:
    trajectory_id: str
    source_artifact: str
    timestamps_seconds: np.ndarray
    position: np.ndarray
    pitch_radians: np.ndarray
    gas: np.ndarray
    brake: np.ndarray
    coordinate_system: str
    synthetic: bool = False
    initial_velocity: float | None = None
    initial_angular_velocity: float | None = None

    def __post_init__(self):
        if not self.trajectory_id or not self.source_artifact:
            raise ValueError("Trajectories require stable IDs and source artifact references")
        if self.coordinate_system not in {
            "world_meters",
            "camera_compensated_pixels",
            "synthetic_meters",
        }:
            raise ValueError("Position must have declared world or camera-compensated coordinates")
        if self.coordinate_system == "synthetic_meters" and not self.synthetic:
            raise ValueError("Synthetic coordinates must be explicitly labeled synthetic")
        names = ("timestamps_seconds", "position", "pitch_radians", "gas", "brake")
        for name in names:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError("Trajectory columns must be finite one-dimensional arrays")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        n = len(self.timestamps_seconds)
        if n < 6 or any(len(getattr(self, name)) != n for name in names):
            raise ValueError("Trajectory columns must have equal lengths of at least six")
        if np.any(np.diff(self.timestamps_seconds) <= 0):
            raise ValueError("Trajectory timestamps must be strictly increasing")
        if not np.isin(self.gas, [0, 1]).all() or not np.isin(self.brake, [0, 1]).all():
            raise ValueError("Gas and brake columns must be independent binary values")
        for value in (self.initial_velocity, self.initial_angular_velocity):
            if value is not None and not np.isfinite(value):
                raise ValueError("Initial velocities must be finite")

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.coordinate_system.encode())
        digest.update(str(self.synthetic).encode())
        digest.update(str((self.initial_velocity, self.initial_angular_velocity)).encode())
        for value in (
            self.timestamps_seconds - self.timestamps_seconds[0],
            self.position,
            self.pitch_radians,
            self.gas,
            self.brake,
        ):
            digest.update(np.asarray(value, dtype="<f8").tobytes())
        return digest.hexdigest()


def load_trajectory(path: str | Path) -> MotionTrajectory:
    """Read a JSON trajectory; actions[i] are held from timestamp[i] to [i+1]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MotionTrajectory(**data)


def _advance(position: float, velocity: float, force: float, drag: float, dt: float):
    # Exact integration of dv/dt = force - drag*v for a held input interval.
    if abs(drag * dt) < 1e-5:
        integral = dt - drag * dt * dt / 2 + drag * drag * dt**3 / 6
        double_integral = dt * dt / 2 - drag * dt**3 / 6 + drag * drag * dt**4 / 24
    else:
        integral = -np.expm1(-drag * dt) / drag
        double_integral = (dt - integral) / drag
    next_position = position + velocity * integral + force * double_integral
    next_velocity = velocity * np.exp(-drag * dt) + force * integral
    return next_position, next_velocity


def predict_trajectory(parameters: dict[str, float] | np.ndarray, trajectory: MotionTrajectory):
    """Return position, unwrapped pitch; no ground contact or hidden state queried."""
    p = np.array(
        [parameters[name] for name in PARAMETER_NAMES]
        if isinstance(parameters, dict)
        else parameters,
        dtype=float,
    )
    if p.shape != (8,) or not np.isfinite(p).all() or p[3] < 0 or p[7] < 0:
        raise ValueError("Expected eight finite parameters and nonnegative drag")
    t = trajectory.timestamps_seconds
    observed_pitch = np.unwrap(trajectory.pitch_radians)
    x, pitch = np.empty(len(t)), np.empty(len(t))
    x[0], pitch[0] = trajectory.position[0], observed_pitch[0]
    velocity = trajectory.initial_velocity
    omega = trajectory.initial_angular_velocity
    if velocity is None:
        velocity = float(np.gradient(trajectory.position[:3], t[:3], edge_order=2)[0])
    if omega is None:
        omega = float(np.gradient(observed_pitch[:3], t[:3], edge_order=2)[0])
    for i, dt in enumerate(np.diff(t)):
        gas, brake = trajectory.gas[i], trajectory.brake[i]
        inputs = np.array([gas, brake, gas * brake])
        x[i + 1], velocity = _advance(x[i], velocity, float(p[:3] @ inputs), p[3], dt)
        pitch[i + 1], omega = _advance(pitch[i], omega, float(p[4:7] @ inputs), p[7], dt)
    return x, pitch


def _metrics(parameters, trajectories: list[MotionTrajectory]) -> dict[str, float]:
    dx, dp, endx, endp = [], [], [], []
    for trajectory in trajectories:
        x, pitch = predict_trajectory(parameters, trajectory)
        dx.extend((x[1:] - trajectory.position[1:]).tolist())
        dp.extend((pitch[1:] - np.unwrap(trajectory.pitch_radians)[1:]).tolist())
        endx.append(abs(float(x[-1] - trajectory.position[-1])))
        endp.append(abs(float(pitch[-1] - np.unwrap(trajectory.pitch_radians)[-1])))
    return {
        "trajectories": float(len(trajectories)),
        "predicted_samples": float(len(dx)),
        "position_rmse": float(np.sqrt(np.mean(np.square(dx)))),
        "pitch_rmse_radians": float(np.sqrt(np.mean(np.square(dp)))),
        "endpoint_position_mae": float(np.mean(endx)),
        "endpoint_pitch_mae_radians": float(np.mean(endp)),
    }


@dataclass(frozen=True)
class CalibrationFit:
    record: CalibrationRecord
    report: dict
    sha256: str

    def save(self, directory: str | Path) -> Path:
        """Write a new immutable calibration directory, never overwrite an old fit."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "calibration.json").write_text(
            self.record.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (directory / "fit-report.json").write_text(
            json.dumps(self.report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (directory / "sha256.txt").write_text(self.sha256 + "\n", encoding="ascii")
        return directory


def fit_calibration(
    train_trajectories: list[MotionTrajectory],
    heldout_trajectories: list[MotionTrajectory],
    *,
    run_id: str,
    position_scale: float = 1.0,
    pitch_scale: float = 1.0,
    max_nfev: int = 250,
) -> CalibrationFit:
    """Fit on train episodes only; evaluate frozen parameters on held-out episodes.

    Position/pitch scales set residual weights and must be selected before seeing
    held-out results. Synthetic and real data cannot be mixed in a calibration.
    All four pedal states should be probed; independent gas/brake/both excitation
    is required to identify the simultaneous-input term.
    """
    if not train_trajectories or not heldout_trajectories or not run_id:
        raise ValueError("A fit requires training data, held-out data, and a producing run ID")
    if (
        not np.isfinite([position_scale, pitch_scale]).all()
        or min(position_scale, pitch_scale) <= 0
    ):
        raise ValueError("Residual scales must be positive and finite")
    if type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    all_trajectories = train_trajectories + heldout_trajectories
    ids = [t.trajectory_id for t in all_trajectories]
    artifacts = [t.source_artifact for t in all_trajectories]
    hashes = [t.content_hash for t in all_trajectories]
    if (
        len(set(ids)) != len(ids)
        or len(set(artifacts)) != len(artifacts)
        or len(set(hashes)) != len(hashes)
    ):
        raise ValueError(
            "Train/held-out trajectories must have disjoint IDs, artifacts, and content"
        )
    if len({(t.coordinate_system, t.synthetic) for t in all_trajectories}) != 1:
        raise ValueError(
            "All trajectories must share coordinate units and real/synthetic provenance"
        )
    design = np.concatenate(
        [
            np.column_stack((t.gas[:-1], t.brake[:-1], t.gas[:-1] * t.brake[:-1]))
            for t in train_trajectories
        ]
    )
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Training actions do not independently identify gas, brake, and both")

    def residual(parameters):
        parts = []
        for trajectory in train_trajectories:
            x, pitch = predict_trajectory(parameters, trajectory)
            parts.extend(
                (
                    (x[1:] - trajectory.position[1:]) / position_scale,
                    (pitch[1:] - np.unwrap(trajectory.pitch_radians)[1:]) / pitch_scale,
                )
            )
        return np.concatenate(parts)

    lower = [-np.inf, -np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf, 0]
    upper = [np.inf] * 8
    fitted = least_squares(
        residual,
        [5, -5, 0, 0.1, 2, -2, 0, 0.1],
        bounds=(lower, upper),
        loss="soft_l1",
        max_nfev=max_nfev,
    )
    singular = np.linalg.svd(fitted.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(fitted.jac))
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Effective calibration did not converge: {fitted.message}")
    if rank < len(PARAMETER_NAMES):
        raise ValueError("Effective model is not identifiable from these training trajectories")
    parameters = dict(zip(PARAMETER_NAMES, map(float, fitted.x)))
    synthetic = train_trajectories[0].synthetic
    limitations = [
        "Restricted effective model; not a calibrated suspension/contact/fuel simulator.",
        "World progress or camera-compensated position is required; raw screen x is invalid.",
        "Ground slopes, collisions, menus, and changing camera scale violate model assumptions.",
        "Pitch must be physically oriented; a color principal axis modulo pi is insufficient alone.",
        "Validation errors do not establish control-policy transfer.",
    ]
    if synthetic:
        limitations.insert(
            0, "SYNTHETIC validation only; this is not measured Hill Climb Racing calibration."
        )
    if any(
        t.initial_velocity is None or t.initial_angular_velocity is None for t in all_trajectories
    ):
        limitations.append(
            "Some initial velocities were estimated from the first three observations."
        )
    report = {
        "model_version": MODEL_VERSION,
        "synthetic": synthetic,
        "coordinate_system": train_trajectories[0].coordinate_system,
        "position_scale": position_scale,
        "pitch_scale": pitch_scale,
        "scipy_version": scipy.__version__,
        "optimizer": "scipy.optimize.least_squares/soft_l1",
        "nfev": int(fitted.nfev),
        "cost": float(fitted.cost),
        "jacobian_rank": rank,
        "jacobian_condition": float(singular[0] / singular[-1]),
        "train": [
            {"id": t.trajectory_id, "artifact": t.source_artifact, "sha256": t.content_hash}
            for t in train_trajectories
        ],
        "heldout": [
            {"id": t.trajectory_id, "artifact": t.source_artifact, "sha256": t.content_hash}
            for t in heldout_trajectories
        ],
        "parameters": parameters,
        "fit_metrics": _metrics(parameters, train_trajectories),
        "validation_metrics": _metrics(parameters, heldout_trajectories),
    }
    fingerprint = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    record = CalibrationRecord(
        calibration_id=f"cal-{fingerprint[:16]}",
        version=f"{MODEL_VERSION}-{fingerprint[:12]}",
        run_id=run_id,
        source_trajectory_artifacts=[t.source_artifact for t in train_trajectories],
        simulator_version=MODEL_VERSION,
        parameters=parameters,
        held_out_trajectory_artifacts=[t.source_artifact for t in heldout_trajectories],
        fit_metrics=report["fit_metrics"],
        validation_metrics=report["validation_metrics"],
        limitations=limitations,
    )
    return CalibrationFit(record, report, fingerprint)
