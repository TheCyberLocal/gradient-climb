"""Synthetic identifiability and held-out isolation tests; no real-game evidence."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gradientclimb.calibration import MotionTrajectory, fit_calibration
from gradientclimb.calibration.effective import PARAMETER_NAMES

TRUE_PARAMETERS = np.array([6.0, -4.0, 1.25, 0.45, 0.8, -0.6, 0.35, 0.55])


def synthetic_trajectory(seed: int, count: int = 75) -> MotionTrajectory:
    """Generate independent reference data with solve_ivp rather than fitted integrator."""
    rng = np.random.default_rng(seed)
    times = np.r_[0.0, np.cumsum(rng.uniform(0.04, 0.09, count - 1))]
    actions = np.repeat(rng.integers(0, 4, size=count), 4)[:count]
    gas, brake = actions & 1, (actions >> 1) & 1
    state = np.array([0.1 * seed, 0.3, 0.1, -0.1])
    observed = [state.copy()]
    p = TRUE_PARAMETERS
    for index, dt in enumerate(np.diff(times)):
        g, b = gas[index], brake[index]

        def derivative(_, s, g=g, b=b):
            return [
                s[1],
                p[0] * g + p[1] * b + p[2] * g * b - p[3] * s[1],
                s[3],
                p[4] * g + p[5] * b + p[6] * g * b - p[7] * s[3],
            ]

        state = solve_ivp(derivative, (0, dt), state, rtol=1e-10, atol=1e-12).y[:, -1]
        observed.append(state.copy())
    observed = np.asarray(observed)
    return MotionTrajectory(
        f"synthetic-{seed}",
        f"synthetic-artifact-{seed}",
        times,
        observed[:, 0],
        observed[:, 2],
        gas,
        brake,
        "synthetic_meters",
        True,
        0.3,
        -0.1,
    )


def test_recovers_effective_parameters_with_disjoint_validation(tmp_path):
    fitted = fit_calibration(
        [synthetic_trajectory(1), synthetic_trajectory(2)],
        [synthetic_trajectory(3)],
        run_id="synthetic-test",
    )
    recovered = [fitted.record.parameters[name] for name in PARAMETER_NAMES]
    np.testing.assert_allclose(recovered, TRUE_PARAMETERS, rtol=1e-4, atol=1e-5)
    assert fitted.record.validation_metrics["position_rmse"] < 1e-5
    assert fitted.record.validation_metrics["pitch_rmse_radians"] < 1e-5
    assert fitted.report["synthetic"] is True
    assert any("SYNTHETIC" in limitation for limitation in fitted.record.limitations)
    assert len(fitted.sha256) == 64
    output = fitted.save(tmp_path / "calibration")
    assert (output / "calibration.json").exists()
    with pytest.raises(FileExistsError):
        fitted.save(output)


def test_heldout_measurements_cannot_change_fitted_parameters():
    train = [synthetic_trajectory(1), synthetic_trajectory(2)]
    original = synthetic_trajectory(3)
    changed = replace(original, position=original.position + 0.2 * original.timestamps_seconds**2)
    fit_a = fit_calibration(train, [original], run_id="synthetic-a")
    fit_b = fit_calibration(train, [changed], run_id="synthetic-b")
    assert fit_a.record.parameters == fit_b.record.parameters
    assert fit_b.record.validation_metrics["position_rmse"] > 0.5
    assert fit_a.sha256 != fit_b.sha256


def test_rejects_renamed_duplicate_trajectories():
    trajectory = synthetic_trajectory(1)
    duplicated = replace(trajectory, trajectory_id="different-id", source_artifact="other-artifact")
    with pytest.raises(ValueError, match="disjoint"):
        fit_calibration([trajectory], [duplicated], run_id="synthetic")


def test_rejects_unidentified_simultaneous_controls():
    trajectory = synthetic_trajectory(1)
    no_brake = replace(trajectory, brake=np.zeros_like(trajectory.brake))
    with pytest.raises(ValueError, match="independently identify"):
        fit_calibration([no_brake], [synthetic_trajectory(3)], run_id="synthetic")


def test_rejects_raw_screen_coordinates_and_invalid_timing():
    trajectory = synthetic_trajectory(1)
    with pytest.raises(ValueError, match="camera-compensated"):
        replace(trajectory, coordinate_system="raw_screen_pixels")
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(trajectory, timestamps_seconds=np.zeros_like(trajectory.timestamps_seconds))
    with pytest.raises(ValueError, match="explicitly labeled"):
        replace(trajectory, synthetic=False)


def test_requires_real_synthetic_provenance_separation():
    real_labeled_fixture = replace(
        synthetic_trajectory(3), coordinate_system="world_meters", synthetic=False
    )
    with pytest.raises(ValueError, match="provenance"):
        fit_calibration([synthetic_trajectory(1)], [real_labeled_fixture], run_id="synthetic")
