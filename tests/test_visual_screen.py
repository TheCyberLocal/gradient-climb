import io
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from demo_dataset_fixtures import make_demo_source, reference, write_json
from PIL import Image, UnidentifiedImageError
from test_visual_replay import geometry, profile

import gradientclimb.experiments.visual_screen as screen
import gradientclimb.simulation.visual_replay as replay_module
from gradientclimb.experiments import verify_run
from gradientclimb.simulation.visual_renderer import render_pose


def fixture(tmp_path, *, bad_image=None):
    def image_bytes(index):
        if index == bad_image:
            return b"Sealed synthetic undecodable image"
        stream = io.BytesIO()
        Image.fromarray(render_pose(geometry())).save(stream, format="PNG")
        return stream.getvalue()

    root, plan, review, ledger, freeze, frames, _controls = make_demo_source(
        tmp_path, frame_bytes=image_bytes
    )
    write_json(root / "research/profile.json", asdict(profile()))
    write_json(
        root / "research/prior.json", {"purpose": "Synthetic prior diagnostic", "cost": None}
    )
    protocol = json.loads(
        (
            Path(__file__).parents[1]
            / "experiments/definitions/cycle-3-visual-pose-screen-001.json"
        ).read_bytes()
    )
    protocol.update(
        experiment_id="synthetic-visual-screen",
        source_run_id=review["source"]["run_id"],
        window_plan=reference(root, root / "research/plan.json"),
        prior_diagnostic=reference(root, root / "research/prior.json"),
        perception_profile=reference(root, root / "research/profile.json"),
    )
    protocol["annotation"]["image_size"] = [256, 160]
    folder = Path(review["source"]["frames"]["path"]).parent
    protocol["selected_frames"] = [
        {"frame_index": i, "source_image": reference(root, root / folder / frames[i]["path"])}
        for i in range(4)
    ]
    write_json(root / "research/protocol.json", protocol)
    protocol_ref = reference(root, root / "research/protocol.json")
    labels = {
        "replay_id": "synthetic-reviewed-labels",
        "source_partition": "train",
        "note": "Synthetic construction integration fixture",
        "frames": [],
    }
    for i, selected in enumerate(protocol["selected_frames"]):
        g = geometry(
            str(i),
            provenance="reviewed_construction",
            source_image=selected["source_image"],
            label_protocol=protocol_ref,
            reviewer="synthetic-fixture",
            reviewed_at=datetime(2026, 9, 12, 20, tzinfo=UTC),
        )
        labels["frames"].append(
            {
                "geometry": g.model_dump(mode="json"),
                "timestamp_ns": frames[i]["timestamp_ns"],
                "continuity_id": str(i),
            }
        )
    write_json(root / "research/labels.json", labels)
    return root, protocol, labels, review, plan, ledger, freeze


def publish(root):
    return screen.publish_visual_pose_screen(root, "research/protocol.json", "research/labels.json")


def derived(root, source_id):
    dirs = [p for p in (root / "artifacts/runs").iterdir() if p.name != source_id]
    assert len(dirs) == 1
    directory = dirs[0]
    return directory, json.loads((directory / "run.json").read_bytes())


def operations(directory):
    return [
        json.loads(line)
        for line in (directory / "visual-pose-operations.jsonl").read_text().splitlines()
    ]


def assert_zero(record):
    for key in ("environment_steps", "training_steps", "episodes", "optimizer_updates"):
        assert record[key] == 0
    assert record["evaluation_results"] == []
    assert record["summary"]["evaluation_episodes"] == record["summary"]["policy_decisions"] == 0
    assert record["summary"]["annotation_labor_seconds"] is None
    assert record["summary"]["resources"]["version"] == "resources-3.0"


def test_canonical_four_pairs_have_exact_retained_pixels_and_early_partition_binding(
    tmp_path, monkeypatch
):
    pytest.importorskip("cv2")
    root, _, _, review, *_ = fixture(tmp_path)
    source_id = review["source"]["run_id"]
    seal = root / "artifacts/runs" / source_id / "seal.json"
    old_seal = seal.read_bytes()
    registration = []
    actual_register = screen.RunRecorder.register_artifact

    def register(self, path, kind, metadata=None):
        registration.append(kind)
        return actual_register(self, path, kind, metadata)

    monkeypatch.setattr(screen.RunRecorder, "register_artifact", register)
    actual_assemble = screen.assemble_windows

    def assemble(*args):
        assert registration.index("whole_session_partition_ledger") < registration.index(
            "visual_screen_protocol"
        )
        assert {
            "visual_screen_protocol",
            "visual_geometry_labels",
            "measurement_profile",
            "segmentation_review",
        } <= set(registration)
        assert registration.count("selected_source_image") == 4
        return actual_assemble(*args)

    monkeypatch.setattr(screen, "assemble_windows", assemble)
    renders = []

    def render(*args):
        rgb = render_pose(*args)
        renders.append(rgb.copy())
        return rgb

    monkeypatch.setattr(replay_module, "render_pose", render)
    result = publish(root)
    directory = Path(result["report"]).parent
    report = json.loads(Path(result["report"]).read_bytes())
    record = json.loads((directory / "run.json").read_bytes())
    assert result["verification"]["valid"] and record["status"] == "completed"
    assert len(renders) == len(report["rows"]) == 4
    assert report["render_replay_identity"]["renderer_version"] == "procedural-screen-pose-3.0"
    assert len(report["render_replay_identity"]["render_style_sha256"]) == 64
    assert record["summary"]["source_image_decodes"] == 4
    for i, row in enumerate(report["rows"]):
        with Image.open(directory / f"actor-pose-{i:03d}.png") as image:
            assert np.array_equal(np.asarray(image), renders[i])
        assert row["rendered"]["history_reset"]
        for side in ("source", "rendered"):
            observation = row[side]["observation"]
            assert not observation["temporal_contiguous"] and not observation["input_authorized"]
        assert row["rendered_discrepancies"]["body"] is not None
    assert len([e for e in operations(directory) if e["event"] == "frame_completed"]) == 4
    assert_zero(record)
    assert record["summary"]["inherited_sources"][0]["prior_cost"]["capture_wall_seconds"] == 1
    assert seal.read_bytes() == old_seal and verify_run(root / "artifacts", source_id)["valid"]
    with pytest.raises(ValueError, match="already has a canonical attempt"):
        publish(root)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("timestamp", "timestamp"),
        ("order", "source order"),
        ("review_time", "post-registration"),
        ("continuity", "continuity"),
        ("partition", "partitions"),
    ],
)
def test_rejects_mismatched_frozen_label_contract_before_rendering(
    tmp_path, monkeypatch, mutation, match
):
    root, _, labels, *_ = fixture(tmp_path)
    if mutation == "timestamp":
        labels["frames"][0]["timestamp_ns"] += 1
    elif mutation == "order":
        labels["frames"][0]["geometry"]["source_image"] = labels["frames"][1]["geometry"][
            "source_image"
        ]
    elif mutation == "review_time":
        labels["frames"][0]["geometry"]["reviewed_at"] = "2026-09-12T10:00:00Z"
    elif mutation == "continuity":
        labels["frames"][1]["continuity_id"] = labels["frames"][0]["continuity_id"]
    else:
        labels["source_partition"] = "development"
    write_json(root / "research/labels.json", labels)
    monkeypatch.setattr(
        replay_module, "render_pose", lambda *_: pytest.fail("Rejected contract rendered pixels")
    )
    with pytest.raises(ValueError, match=match):
        publish(root)


@pytest.mark.parametrize("failure", ["decode", "render", "interrupt"])
def test_operation_failure_preserves_sealed_progress_and_prior_costs(
    tmp_path, monkeypatch, failure
):
    root, _, _, review, *_ = fixture(tmp_path, bad_image=2 if failure == "decode" else None)
    source_id = review["source"]["run_id"]
    calls = []
    exception = {
        "decode": UnidentifiedImageError,
        "render": RuntimeError,
        "interrupt": KeyboardInterrupt,
    }[failure]

    def render(*args):
        if len(calls) == 2:
            directory = next(p for p in (root / "artifacts/runs").iterdir() if p.name != source_id)
            assert sum(e["event"] == "frame_completed" for e in operations(directory)) == 2
            raise exception("Synthetic third-render failure")
        calls.append(1)
        return render_pose(*args)

    if failure != "decode":
        monkeypatch.setattr(replay_module, "render_pose", render)
    with pytest.raises(exception):
        publish(root)
    directory, record = derived(root, source_id)
    assert record["status"] == ("cancelled" if failure == "interrupt" else "failed")
    assert operations(directory)[-1]["event"] == "screen_failed"
    failure_receipt = operations(directory)[-1]
    assert failure_receipt["source_failed_pose_index"] == (2 if failure == "decode" else None)
    assert failure_receipt["source_unattempted_pose_indices"] == (
        [3] if failure == "decode" else []
    )
    assert failure_receipt["render_failed_pose_index"] == (None if failure == "decode" else 2)
    assert failure_receipt["render_unattempted_pose_indices"] == (
        list(range(4)) if failure == "decode" else [3]
    )
    progress = record["summary"]["visual_progress"]
    assert progress["source_completed"] == (2 if failure == "decode" else 4)
    assert progress["render_completed"] == (0 if failure == "decode" else 2)
    assert verify_run(root / "artifacts", directory.name)["valid"]
    assert any(a["kind"] == "visual_pose_operation_journal" for a in record["artifact_manifest"])
    assert_zero(record)
    assert not (directory / "visual-pose-screen.json").exists()


def test_terrain_comparison_preserves_gaps_occlusion_and_missingness():
    g = geometry(
        terrain=[
            {"points": points, "uncertainty_pixels": 6, "note": "Visible lower turf boundary"}
            for points in (((0, 125), (45, 125)), ((210, 125), (255, 125)))
        ]
    )
    measured = {
        "body": {"valid": False},
        "wheels": {"valid": False},
        "terrain": {
            "valid": True,
            "points": [{"x": x, "y": 129, "valid": True} for x in (20, 120, 230)],
        },
    }
    result = screen.geometry_discrepancies(g, measured)
    assert result["body"] is None and result["wheels"] is None
    assert [p["x"] for p in result["terrain"]] == [20, 230]
    assert all(
        p["delta_y_pixels"] == 4 and p["label_uncertainty_pixels"] == 6 for p in result["terrain"]
    )
    hidden = g.model_dump(mode="json")
    hidden["occluders"] = [
        {
            "vertices": [[10, 120], [30, 120], [30, 135], [10, 135]],
            "uncertainty_pixels": 1,
            "note": "Synthetic visible overlay",
        }
    ]
    assert [
        p["x"]
        for p in screen.geometry_discrepancies(type(g).model_validate(hidden), measured)["terrain"]
    ] == [230]
    measured["terrain"]["valid"] = False
    assert screen.geometry_discrepancies(g, measured)["terrain"] is None


def test_budget_failure_is_sealed_before_any_pixel_operation(tmp_path, monkeypatch):
    root, protocol, labels, review, *_ = fixture(tmp_path)
    protocol["budgets"]["maximum_output_bytes"] = 100
    write_json(root / "research/protocol.json", protocol)
    for row in labels["frames"]:
        row["geometry"]["label_protocol"] = reference(root, root / "research/protocol.json")
    write_json(root / "research/labels.json", labels)
    monkeypatch.setattr(
        replay_module, "render_pose", lambda *_: pytest.fail("Budget failure rendered pixels")
    )
    with pytest.raises(RuntimeError, match="budget"):
        publish(root)
    directory, record = derived(root, review["source"]["run_id"])
    assert record["summary"]["visual_progress"]["source_attempted"] == 0
    assert record["status"] == "failed" and verify_run(root / "artifacts", directory.name)["valid"]


def test_wall_budget_counts_preflight_and_stops_before_source_decode(tmp_path, monkeypatch):
    root, _, _, review, *_ = fixture(tmp_path)
    clocks = iter((0, 121))
    monkeypatch.setattr(screen, "time", SimpleNamespace(perf_counter=lambda: next(clocks)))
    with pytest.raises(RuntimeError, match="budget"):
        publish(root)
    directory, record = derived(root, review["source"]["run_id"])
    assert record["summary"]["visual_progress"]["source_attempted"] == 0
    assert verify_run(root / "artifacts", directory.name)["valid"]


def test_prior_partition_assignment_cannot_be_rewritten(tmp_path, monkeypatch):
    root, _, _, _, plan, ledger, freeze = fixture(tmp_path)
    # A prior failed publisher still reserves the session's original split.
    with (
        screen.partition_publication(root, plan) as binding,
        screen.RunRecorder(
            root / "artifacts",
            "prior-failed-publication",
            binding.configuration({}),
            source_root=root,
            telemetry_interval_seconds=0,
        ) as run,
    ):
        binding.register(run)
        run.finalize(status="failed", episodes=0)
    ledger["sessions"][0]["split"] = "development"
    plan["split"] = "development"
    freeze()
    protocol = json.loads((root / "research/protocol.json").read_bytes())
    protocol["source_partition"] = "development"
    protocol["window_plan"] = reference(root, root / "research/plan.json")
    write_json(root / "research/protocol.json", protocol)
    labels = json.loads((root / "research/labels.json").read_bytes())
    labels["source_partition"] = "development"
    for row in labels["frames"]:
        row["geometry"]["label_protocol"] = reference(root, root / "research/protocol.json")
    write_json(root / "research/labels.json", labels)
    monkeypatch.setattr(
        screen,
        "assemble_windows",
        lambda *_: pytest.fail("Conflicting partition reached source consumption"),
    )
    with pytest.raises(ValueError, match="reassignment"):
        publish(root)


def test_legacy_string_protocol_in_canonical_history_does_not_break_screen(tmp_path):
    root, *_ = fixture(tmp_path)
    with screen.RunRecorder(
        root / "artifacts",
        "legacy-fixture",
        {"protocol": "cycle-1-legacy-string"},
        source_root=root,
        telemetry_interval_seconds=0,
    ) as legacy:
        legacy.finalize(status="completed", episodes=0)
    original = (legacy.directory / "seal.json").read_bytes()
    result = publish(root)
    assert result["verification"]["valid"]
    assert (legacy.directory / "seal.json").read_bytes() == original
    assert verify_run(root / "artifacts", legacy.run_id)["valid"]


def operational_successor(root, original):
    write_json(
        root / "research/failed-dispatch.json",
        {
            "stage": "pre-recorder",
            "renders": 0,
            "source_decodes": 0,
            "reason": "Synthetic operational failure receipt",
        },
    )
    successor = json.loads(json.dumps(original))
    successor.update(
        experiment_id="synthetic-visual-operational-successor",
        status="Registered operational correction, original labels frozen",
        created_at="2026-09-12T21:00:00Z",
        construction_protocol=reference(root, root / "research/protocol.json"),
        operational_successor={
            "frozen_replay": reference(root, root / "research/labels.json"),
            "failure_evidence": reference(root, root / "research/failed-dispatch.json"),
            "reason": "Retry only the diagnosed pre-recorder failure with identical construction inputs",
        },
    )
    write_json(root / "research/successor.json", successor)
    return successor


def test_operational_successor_uses_original_label_bytes_and_original_review_cutoff(tmp_path):
    root, original, *_ = fixture(tmp_path)
    old_protocol = (root / "research/protocol.json").read_bytes()
    old_labels = (root / "research/labels.json").read_bytes()
    successor = operational_successor(root, original)
    result = screen.publish_visual_pose_screen(
        root, "research/successor.json", "research/labels.json"
    )
    assert result["verification"]["valid"]
    report = json.loads(Path(result["report"]).read_bytes())
    assert report["summary"]["retained_paired_poses"] == 4
    assert report["input_contract"]["construction_protocol"] == successor["construction_protocol"]
    assert all(
        row["label"]["label_protocol"] == successor["construction_protocol"]
        for row in report["rows"]
    )
    record = json.loads((Path(result["report"]).parent / "run.json").read_bytes())
    assert {"original_construction_protocol", "prior_operational_failure"} <= {
        a["kind"] for a in record["artifact_manifest"]
    }
    assert (root / "research/protocol.json").read_bytes() == old_protocol
    assert (root / "research/labels.json").read_bytes() == old_labels


@pytest.mark.parametrize(
    "field", ["selected_frames", "budgets", "annotation", "renderer", "perception_profile"]
)
def test_operational_successor_cannot_change_substantive_inputs(tmp_path, monkeypatch, field):
    root, original, *_ = fixture(tmp_path)
    successor = operational_successor(root, original)
    if field == "selected_frames":
        successor[field].reverse()
    elif field == "budgets":
        successor[field]["maximum_wall_seconds"] = 119
    elif field == "annotation":
        successor[field]["image_size"][0] += 1
    elif field == "renderer":
        successor[field]["assets"] = "Different original art"
    else:
        successor[field]["sha256"] = "a" * 64
    write_json(root / "research/successor.json", successor)
    monkeypatch.setattr(
        replay_module, "render_pose", lambda *_: pytest.fail("Changed successor rendered pixels")
    )
    with pytest.raises(ValueError, match="substantive construction inputs or budgets"):
        screen.publish_visual_pose_screen(root, "research/successor.json", "research/labels.json")


def test_operational_successor_cannot_substitute_changed_label_bytes(tmp_path):
    root, original, labels, *_ = fixture(tmp_path)
    operational_successor(root, original)
    labels["note"] = "Changed after frozen operational receipt"
    write_json(root / "research/labels.json", labels)
    with pytest.raises(ValueError, match="exact frozen original replay bytes"):
        screen.publish_visual_pose_screen(root, "research/successor.json", "research/labels.json")
