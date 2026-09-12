"""Tests of scientific distinctions, not historical metric migrations."""

import copy
import json

import pytest

from gradientclimb.experiments.efficiency import (
    CostVector,
    LearningEfficiencyStudy,
    analyze_study,
    compare_studies,
    load_studies,
)


def ref(digit="a"):
    return {"path": "evidence.json", "sha256": digit * 64}


def study():
    return {
        "study_id": "run-a",
        "training_run_id": "training-a",
        "training_seed": 31,
        "started_at": "2026-09-13T00:00:00Z",
        "real_scenario": ref(),
        "profile_id": "reference-v3",
        "simulator_fidelity_id": "legacy-lightweight-0.1.0",
        "method": "candidate",
        "evaluation_split": "qualification",
        "prior_roots": [],
        "policy_prior_roots": [],
        "prior_costs": [],
        "lineage_declared_complete": True,
        "protocol": {
            "protocol_id": "prospective-3",
            "registered_at": "2026-09-12T00:00:00Z",
            "training_budget_seconds": 3600,
            "evaluation_episodes": 2,
            "evaluation_procedure": "Frozen weights, whole independent sessions",
            "resource_accounting": "per-process CPU counters; GPU unavailable",
            "experience_accounting": "all env lifecycles, including vectorized resets",
            "compute_comparison_scope": "same-cpu-and-process-scope",
            "prior_class": "cold-start",
            "evidence": ref(),
        },
        "checkpoints": [
            {
                "checkpoint_sha256": digit * 64,
                "checkpoint_evidence": ref(digit),
                "command_clock_evidence": ref(),
                "resource_evidence": [ref()],
                "elapsed_seconds": seconds,
                "cost": {
                    "cpu_core_seconds": seconds * 2,
                    "simulator_transitions": steps,
                    "simulator_episodes": episodes,
                    "real_game_interaction_seconds": 0,
                },
            }
            for digit, seconds, steps, episodes in [
                ("b", 300, 10000, 1000),
                ("c", 600, 30000, 3000),
            ]
        ],
        "evaluations": [
            {
                "evaluation_run_id": f"eval-{digit}",
                "checkpoint_sha256": digit * 64,
                "protocol_id": "prospective-3",
                "profile_id": "reference-v3",
                "scenario_sha256": "a" * 64,
                "policy_kind": "learned",
                "split": "qualification",
                "horizon_seconds": 900,
                "verification_elapsed_seconds": 1500,
                "evaluation_elapsed_seconds": 200,
                "evidence": [ref()],
                "evaluation_cost": {"real_game_interaction_seconds": 100},
                "episodes": [
                    {
                        "episode_id": f"{digit}-{i}",
                        "distance_m": distance,
                        "eligible": True,
                        "endpoint": "natural",
                    }
                    for i in range(2)
                ],
            }
            for digit, distance in [("b", 400), ("c", 1200)]
        ],
    }


def test_time_is_first_observed_real_pass_not_episode_count():
    report = analyze_study(study())
    first, second, third = report["thresholds"]
    assert first["time_seconds"] == second["time_seconds"] == 600
    assert first["interval_lower_seconds"] == 300
    assert first["own_cost"]["simulator_episodes"] == 3000
    assert first["own_cost"]["gpu_utilization_equivalent_seconds"] is None
    assert third["status"] == "right_censored"
    assert third["time_seconds"] is None
    assert third["censor_seconds"] == 600
    assert third["budget_seconds"] == 3600
    assert report["evaluation_cost"]["real_game_interaction_seconds"] == 200


def test_no_evaluation_is_unknown_and_incomplete_attempts_not_filtered():
    item = study()
    item["evaluations"] = []
    assert all(x["status"] == "not_evaluable" for x in analyze_study(item)["thresholds"])
    item = study()
    item["evaluations"][1]["episodes"][0]["distance_m"] = None
    result = analyze_study(item)["thresholds"][0]
    assert result["status"] == "right_censored" and result["censor_seconds"] == 300


@pytest.mark.parametrize(
    "field,value",
    [
        ("domain", "simulator"),
        ("frozen", False),
        ("independent", False),
        ("learner_updates_during_evaluation", 1),
    ],
)
def test_intermediate_or_adaptive_results_cannot_be_competence(field, value):
    item = study()
    item["evaluations"][0][field] = value
    with pytest.raises(ValueError):
        analyze_study(item)


def test_no_cherry_picking_evaluations_or_reusing_executions():
    item = study()
    duplicate = copy.deepcopy(item["evaluations"][1])
    duplicate["evaluation_run_id"] += "-repeat"
    for episode in duplicate["episodes"]:
        episode["episode_id"] += "-repeat"
    item["evaluations"].append(duplicate)
    assert analyze_study(item)["thresholds"][0]["censor_seconds"] == 300
    item["evaluations"][-1]["episodes"][0]["episode_id"] = "b-0"
    with pytest.raises(ValueError, match="counted twice"):
        analyze_study(item)


def add_priors(item):
    item["protocol"]["prior_class"] = "fine-tuning"
    item["prior_roots"] = ["teacher", "student"]
    item["policy_prior_roots"] = ["teacher", "student"]
    item["prior_costs"] = [
        {
            "cost_id": name,
            "kind": "teacher-training",
            "parents": parents,
            "elapsed_seconds": 7200,
            "cost": {"cpu_core_seconds": cpu},
            "evidence": [ref()],
        }
        for name, parents, cpu in [
            ("base", [], 100),
            ("teacher", ["base"], 200),
            ("student", ["base"], 300),
        ]
    ]


def test_lineage_deduplicates_shared_ancestors_keeps_adaptation_separate():
    item = study()
    add_priors(item)
    report = analyze_study(item)
    threshold = report["thresholds"][0]
    assert threshold["time_seconds"] == 600
    assert threshold["prior_cost"]["cpu_core_seconds"] == 600
    assert threshold["combined_cost"]["cpu_core_seconds"] == 1800
    assert len(report["prior_ledger"]) == 3
    assert "elapsed_seconds" not in threshold["combined_cost"]
    item["prior_costs"][0]["parents"] = ["unmeasured-pretraining"]
    report = analyze_study(item)
    assert not report["lineage_complete"]
    assert report["thresholds"][0]["prior_cost"]["cpu_core_seconds"] is None


def test_cyclic_hidden_and_mislabeled_priors_rejected():
    item = study()
    add_priors(item)
    item["prior_costs"][0]["parents"] = ["teacher"]
    with pytest.raises(ValueError, match="cycle"):
        analyze_study(item)
    item["prior_costs"][0]["parents"] = []
    item["protocol"]["prior_class"] = "cold-start"
    with pytest.raises(ValueError, match="cold-start"):
        analyze_study(item)
    item = study()
    add_priors(item)
    item["prior_roots"] = ["teacher"]
    item["policy_prior_roots"] = ["teacher"]
    with pytest.raises(ValueError, match="reachable"):
        analyze_study(item)


def test_pareto_keeps_time_compute_tradeoff_and_prior_classes_separate():
    original = study()
    fast = copy.deepcopy(original)
    fast["study_id"] = "fast-expensive"
    fast["checkpoints"][1]["elapsed_seconds"] = 450
    fast["checkpoints"][1]["cost"]["cpu_core_seconds"] = 3000
    slow = copy.deepcopy(original)
    slow["study_id"] = "dominated"
    slow["checkpoints"][1]["elapsed_seconds"] = 700
    slow["checkpoints"][1]["cost"]["cpu_core_seconds"] = 1500
    assisted = copy.deepcopy(slow)
    assisted["study_id"] = "separate-prior"
    add_priors(assisted)
    reports = [analyze_study(x) for x in [original, fast, slow, assisted]]
    comparison = compare_studies(reports, 1000, axes=["time_seconds", "cpu_core_seconds"])
    rows = {x["study_id"]: x for x in comparison["comparisons"]}
    assert rows["run-a"]["pareto_frontier"]
    assert rows["fast-expensive"]["pareto_frontier"]
    assert not rows["dominated"]["pareto_frontier"]
    assert rows["separate-prior"]["pareto_frontier"]
    assert len(compare_studies(reports, 2000)["excluded"]) == 4
    assert (
        len(compare_studies(reports, 500, axes=["gpu_utilization_equivalent_seconds"])["excluded"])
        == 4
    )


@pytest.mark.parametrize("change", ["naive", "retroactive", "late", "decreased", "hash"])
def test_new_registration_clock_and_identity_constraints(change):
    item = study()
    if change == "naive":
        item["started_at"] = "2026-09-13T00:00:00"
    elif change == "retroactive":
        item["started_at"] = "2026-09-11T00:00:00Z"
    elif change == "late":
        item["checkpoints"][-1]["elapsed_seconds"] = 3601
    elif change == "decreased":
        item["checkpoints"][-1]["cost"]["simulator_transitions"] = 1
    else:
        item["checkpoints"][0]["checkpoint_evidence"] = ref("f")
    with pytest.raises(ValueError):
        LearningEfficiencyStudy.model_validate(item)


def test_old_runs_are_not_inferred(tmp_path):
    old = tmp_path / "runs" / "historical"
    old.mkdir(parents=True)
    (old / "run.json").write_text('{"summary":{"episodes":1000}}')
    assert load_studies(tmp_path) == []
    (old / "learning-efficiency.json").write_text("{}")
    assert "error" in load_studies(tmp_path)[0]


def test_invalid_resource_values_not_accepted():
    for value in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            CostVector(cpu_core_seconds=value)


def test_cold_policy_can_inherit_costly_system_construction():
    item = study()
    add_priors(item)
    item["protocol"]["prior_class"] = "cold-start"
    item["policy_prior_roots"] = []
    report = analyze_study(item)
    assert report["thresholds"][0]["prior_cost"]["cpu_core_seconds"] == 600
    assert report["prior_class"] == "cold-start"


def test_first_evaluation_pass_has_no_measured_lower_bracket():
    item = study()
    for episode in item["evaluations"][0]["episodes"]:
        episode["distance_m"] = 600
    point = analyze_study(item)["thresholds"][0]
    assert point["time_seconds"] == 300 and point["interval_lower_seconds"] is None


def materialize(item, root):
    from gradientclimb.artifacts import sha256_file

    replacements = {}

    def walk(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                old = value["sha256"]
                name = f"evidence-{old[:2]}.json"
                path = root / name
                path.write_text(old, encoding="utf-8")
                replacements[old] = sha256_file(path)
                value["path"] = name
            else:
                for child in value.values():
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(item)
    serialized = json.dumps(item)
    for old, new in replacements.items():
        serialized = serialized.replace(old, new)
    return json.loads(serialized)


def test_publish_load_and_tamper_detection_without_rewriting_sources(tmp_path):
    from gradientclimb.experiments.efficiency import publish_study

    item = materialize(study(), tmp_path)
    path = tmp_path / "evidence-aa.json"
    original = path.read_bytes()
    record = publish_study(tmp_path, item)
    report = load_studies(tmp_path)[0]
    assert record["status"] == "completed"
    assert report["provenance_status"] == "sealed_envelope_and_referenced_bytes_verified"
    assert report["thresholds"][0]["time_seconds"] == 600
    assert path.read_bytes() == original
    path.write_text("changed", encoding="utf-8")
    assert "hash mismatch" in load_studies(tmp_path)[0]["error"]


def test_declared_real_scenario_and_verification_time_must_match():
    item = study()
    item["evaluations"][1]["scenario_sha256"] = "f" * 64
    assert analyze_study(item)["thresholds"][0]["censor_seconds"] == 300
    item = study()
    item["evaluations"][1]["verification_elapsed_seconds"] = 700
    with pytest.raises(ValueError, match="completion"):
        analyze_study(item)


def test_different_registered_budgets_do_not_establish_pareto_dominance():
    first = study()
    second = study()
    second["study_id"] = "different-budget"
    second["protocol"]["training_budget_seconds"] = 7200
    second["checkpoints"][1]["elapsed_seconds"] = 650
    reports = [analyze_study(item) for item in [first, second]]
    assert reports[0]["comparison_contract_sha256"] != reports[1]["comparison_contract_sha256"]
    result = compare_studies(reports, 500)
    assert all(row["pareto_frontier"] for row in result["comparisons"])
