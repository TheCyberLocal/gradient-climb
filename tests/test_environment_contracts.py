"""Generation isolation, immutable provenance and exact legacy adapter behavior."""

import copy

import numpy as np
import pytest
from pydantic import ValidationError

from gradientclimb.algorithms.policies import AlwaysGasPolicy
from gradientclimb.environments import (
    ENVIRONMENTS,
    RANDOMIZED_ENVIRONMENT_ID,
    REFERENCE_ENVIRONMENT_ID,
    EnvironmentUnavailableError,
    Scenario,
    environment_from_config,
    make_environment,
)
from gradientclimb.environments.profiles import EvidenceReference
from gradientclimb.evaluation import compare_paired, evaluate
from gradientclimb.simulation import VectorHillEnv


@pytest.mark.parametrize(
    "profile,terrain,randomization",
    [
        ("default", "train", False),
        ("heavy", "rough", True),
        ("agile", "rolling", False),
    ],
)
def test_factory_is_exact_legacy_adapter(profile, terrain, randomization):
    settings = {
        "num_envs": 4,
        "seed": 39,
        "profile": profile,
        "terrain": terrain,
        "randomization": randomization,
        "stack": 2,
        "max_steps": 17,
        "dt": 0.01,
        "substeps": 6,
    }
    legacy, factory = VectorHillEnv(**settings), make_environment(**settings)
    assert legacy.config == factory.config
    assert factory.scenario.map.distance_unit == "surrogate_unit"
    assert not factory.scenario.observation.deployable_from_game
    actions = np.random.default_rng(81)
    for index in range(40):
        if index == 25:
            left = legacy.reset([11, 22, 33, 44])
            right = factory.reset([11, 22, 33, 44])
            np.testing.assert_array_equal(left[0], right[0])
        controls = actions.integers(0, 4, 4)
        left, right = legacy.step(controls), factory.step(controls)
        for output in range(4):
            np.testing.assert_array_equal(left[output], right[output])
        assert left[4]["episodes"] == right[4]["episodes"]
        np.testing.assert_array_equal(left[4]["final_observation"], right[4]["final_observation"])
    assert (
        legacy.render(width=240, height=135).tobytes()
        == factory.render(width=240, height=135).tobytes()
    )


@pytest.mark.parametrize("generation", [REFERENCE_ENVIRONMENT_ID, RANDOMIZED_ENVIRONMENT_ID])
def test_unimplemented_generations_never_fall_back(generation):
    assert not ENVIRONMENTS[generation].available
    with pytest.raises(EnvironmentUnavailableError, match=generation):
        make_environment(generation)
    with pytest.raises(EnvironmentUnavailableError):
        environment_from_config({"simulator_version": generation})


def test_unknown_and_contradictory_versions_fail_closed():
    with pytest.raises(ValueError, match="Unknown environment"):
        make_environment("typo-3.0")
    with pytest.raises(ValueError, match="disagree"):
        environment_from_config(
            {
                "environment_id": REFERENCE_ENVIRONMENT_ID,
                "simulator_version": "surrogate-0.1.0",
            }
        )
    with pytest.raises(TypeError):
        ENVIRONMENTS["new"] = ENVIRONMENTS["surrogate-0.1.0"]
    with pytest.raises(ValueError, match="observation source"):
        environment_from_config({"observation_source": "rendered_pixels"})
    with pytest.raises(ValueError, match="calibration"):
        environment_from_config({"calibration_version": "measured-1"})


def test_profile_roundtrip_is_content_addressed_and_deeply_immutable():
    scenario = make_environment().scenario
    restored = Scenario.model_validate_json(scenario.model_dump_json())
    assert restored == scenario and restored.sha256 == scenario.sha256
    with pytest.raises(ValidationError):
        scenario.vehicle.bodies[0].role = "wheel"
    changed = scenario.model_dump()
    changed["vehicle"]["parameters"][0]["value"] += 0.1
    assert Scenario.model_validate(changed).sha256 != scenario.sha256


def test_topology_and_controls_support_unusual_vehicles_without_route_scripts():
    data = make_environment().scenario.model_dump(mode="json")
    data["vehicle"]["bodies"] = [
        {"body_id": "body", "role": "chassis"},
        *({"body_id": f"wheel{i}", "role": "wheel"} for i in range(3)),
    ]
    data["vehicle"]["joints"] = [
        {"joint_id": f"joint{i}", "kind": "wheel", "body_a": "body", "body_b": f"wheel{i}"}
        for i in range(3)
    ]
    data["vehicle"]["required_input_channels"] = ["gas", "brake", "boost"]
    data["vehicle"]["mechanisms"] = ["three_wheels", "airborne_thrust"]
    data["controls"]["input_channels"].append("boost")
    data["controls"]["joint_states"] = [[False, False, False], [False, False, True]]
    assert len(Scenario.model_validate(data).vehicle.bodies) == 4
    data["vehicle"]["route_script"] = "boost at metre 120"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


@pytest.mark.parametrize("case", ["build", "upgrade", "channel", "joint", "privilege", "coverage"])
def test_incompatible_profiles_and_unmeasured_claims_are_rejected(case):
    data = make_environment().scenario.model_dump(mode="json")
    if case == "build":
        data["map"]["game_build"] = "different-build"
    elif case == "upgrade":
        data["upgrades"]["vehicle_profile_id"] = "different-vehicle"
    elif case == "channel":
        data["vehicle"]["required_input_channels"].append("boost")
    elif case == "joint":
        data["vehicle"]["joints"] = [
            {"joint_id": "invalid", "kind": "wheel", "body_a": "chassis", "body_b": "missing"}
        ]
    elif case == "privilege":
        data["observation"]["deployable_from_game"] = True
    else:
        data["map"]["measured_coverage"] = [[0, 2000]]
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_measured_provenance_requires_artifact_hash():
    with pytest.raises(ValidationError, match="SHA-256"):
        EvidenceReference(source="local-run/frame.png", kind="measured", note="Observed car")


def test_evaluation_resolves_checkpoint_scenario_and_explicit_transfer_override():
    policy = AlwaysGasPolicy()
    policy.config.update(profile="heavy", terrain="rough", max_steps=3, dt=0.01, substeps=2)
    result = evaluate(policy, [1, 2])
    explicit = evaluate(policy, [1, 2], profile="default", terrain="train", max_steps=4)
    assert (result["profile"], result["terrain"], result["max_steps"]) == ("heavy", "rough", 3)
    assert result["action_duration_seconds"] == 0.02
    assert result["distance_unit"] == "surrogate_unit"
    assert len(result["scenario_hash"]) == 64
    assert (explicit["profile"], explicit["terrain"], explicit["max_steps"]) == (
        "default",
        "train",
        4,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "real_game"),
        ("simulator_version", "reference-0.1.0"),
        ("calibration_version", "measured"),
        ("distance_unit", "game_metre"),
        ("action_duration_seconds", 0.1),
        ("dt", 0.03),
        ("substeps", 2),
        ("randomization", True),
        ("vehicle_profile_hash", "different"),
        ("map_profile_hash", "different"),
        ("deterministic", False),
    ],
)
def test_package_pairing_rejects_incomparable_measurements(field, value):
    first = evaluate(AlwaysGasPolicy(), [1, 2], max_steps=2)
    second = copy.deepcopy(first)
    second[field] = value
    with pytest.raises(ValueError, match="domains, units"):
        compare_paired(first, second)


def test_pairing_rejects_unknown_units_and_duplicate_seeds():
    first = evaluate(AlwaysGasPolicy(), [1, 2], max_steps=2)
    second = copy.deepcopy(first)
    second["episodes"][1]["seed"] = second["episodes"][0]["seed"]
    with pytest.raises(ValueError, match="unique"):
        compare_paired(first, second)
    del first["distance_unit"]
    with pytest.raises(ValueError, match="distance units"):
        compare_paired(first, first)


def test_historical_pairing_uses_only_the_named_legacy_protocol():
    result = evaluate(AlwaysGasPolicy(), [1, 2], max_steps=2)
    historical = {
        key: result[key]
        for key in (
            "scope",
            "simulator_version",
            "calibration_version",
            "profile",
            "terrain",
            "max_steps",
            "deterministic",
            "episodes",
        )
    }
    historical["benchmark_version"] = "surrogate-evaluation-0.1.0"
    assert compare_paired(historical, historical)["distance_difference"]["mean"] == 0
    historical["simulator_version"] = "unknown-1"
    with pytest.raises(ValueError, match="distance units"):
        compare_paired(historical, historical)


def test_replay_uses_the_checkpoint_scenario_and_is_explicitly_visualization(monkeypatch):
    pytest.importorskip("torch")
    import gradientclimb.algorithms
    from gradientclimb.visualization.replay import watch
    from gradientclimb.visualization.sinks import NullSink

    policy = AlwaysGasPolicy()
    policy.config.update(profile="heavy", terrain="rough", dt=0.01, substeps=2, max_steps=7)
    monkeypatch.setattr(gradientclimb.algorithms, "load_policy", lambda _: policy)
    result = watch("pretend-checkpoint", seconds=0.08, sink=NullSink())
    assert result["frames"] == 4
    assert result["environment_config"]["profile"] == "heavy"
    assert result["environment_config"]["terrain"] == "rough"
    assert result["environment_config"]["max_steps"] == 7
    assert result["qualification"] is False


def test_observer_uses_checkpoint_cadence_without_changing_learner_rng():
    torch = pytest.importorskip("torch")
    from gradientclimb.visualization.observer import ObserverConfig, TrainingObserver
    from gradientclimb.visualization.sinks import NullSink

    torch.manual_seed(52)
    state = torch.get_rng_state().clone()
    observer = TrainingObserver(
        ObserverConfig(display="none"),
        {"profile": "heavy", "terrain": "rough", "dt": 0.01, "substeps": 2},
        [NullSink()],
    )
    assert observer.env.action_duration == 0.02
    assert (observer.env.profile, observer.env.terrain) == ("heavy", "rough")
    torch.testing.assert_close(state, torch.get_rng_state())


def test_cli_preserves_omitted_checkpoint_scenario_and_explicit_overrides(monkeypatch, capsys):
    from gradientclimb.benchmarks import runner
    from gradientclimb.cli import main

    calls = []
    monkeypatch.setattr(
        runner, "run_evaluation", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    main(["evaluate", "--checkpoint", "model.pt"])
    main(["evaluate", "--checkpoint", "model.pt", "--profile", "agile", "--terrain", "rough"])
    assert calls[0][0][5:7] == (None, None)
    assert calls[1][0][5:7] == ("agile", "rough")
    capsys.readouterr()
