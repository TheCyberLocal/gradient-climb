import hashlib
import math
from datetime import UTC, datetime
from typing import ClassVar

import numpy as np
import pytest

from gradientclimb.artifacts import sha256_file
from gradientclimb.perception.measurements import MeasurementProfile, PixelMeasurement
from gradientclimb.perception.screen_features import FEATURE_NAMES
from gradientclimb.simulation.visual_labels import PoseReplay, ScreenGeometry
from gradientclimb.simulation.visual_renderer import RenderStyle, render_pose
from gradientclimb.simulation.visual_replay import replay_poses


def geometry(label_id="fixed-pose", **changes):
    value = {
        "label_id": label_id,
        "image_size": (256, 160),
        "provenance": "synthetic_fixture",
        "body": {
            "vertices": ((65, 72), (153, 72), (168, 100), (58, 100)),
            "uncertainty_pixels": 0,
            "note": "Original synthetic polygon",
        },
        "wheels": [
            {
                "wheel_id": str(index),
                "center": (x, 111),
                "radius_pixels": 12,
                "uncertainty_pixels": 0,
                "note": "Synthetic circle, no physical rotation",
            }
            for index, x in enumerate((80, 150))
        ],
        "terrain": [
            {
                "points": ((0, 125), (255, 125)),
                "uncertainty_pixels": 0,
                "note": "Visible strip, no collision surface",
            }
        ],
        "note": "Fixed original test fixture; not fitted to gameplay",
    }
    return ScreenGeometry.model_validate(value | changes)


def plan(*frames, partition="synthetic"):
    return PoseReplay(
        replay_id="fixed-test",
        frames=tuple(
            {
                "geometry": frame,
                "timestamp_ns": 1_000_000_000 + i * 100_000_000,
                "continuity_id": "one",
            }
            for i, frame in enumerate(frames)
        ),
        source_partition=partition,
        note="Synthetic test, no measured campaign",
    )


def profile():
    return MeasurementProfile(
        "fixed-test",
        (256, 160),
        (0, 0, 1, 1),
        body_area_pixels=(50, 14000),
        wheel_radius_pixels=(8, 16),
        axle_length_pixels=(40, 120),
        terrain_samples=12,
    )


def test_renderer_is_deterministic_and_does_not_bridge_missing_terrain():
    strips = [
        {"points": points, "uncertainty_pixels": 0, "note": "Separate support extent"}
        for points in (((0, 125), (45, 125)), ((210, 125), (255, 125)))
    ]
    pose = geometry(terrain=strips)
    before = pose.sha256
    first, second = render_pose(pose), render_pose(pose)
    assert first.shape == (160, 256, 3) and first.dtype == np.uint8
    assert np.array_equal(first, second) and pose.sha256 == before
    assert tuple(first[145, 120]) == RenderStyle().sky
    assert tuple(first[145, 20]) == RenderStyle().soil
    first[:] = 0
    assert np.array_equal(render_pose(pose), second)


def test_explicit_occluder_and_absent_geometry_do_not_invent_shapes():
    hidden = geometry(
        body=None,
        wheels=(),
        terrain=(),
        missing_geometry=("All scene geometry unobserved",),
    )
    image = render_pose(hidden)
    assert np.all(image == np.asarray(RenderStyle().sky))
    occluded = geometry(
        occluders=(
            {
                "vertices": ((0, 0), (255, 0), (255, 159), (0, 159)),
                "uncertainty_pixels": 0,
                "note": "Synthetic opaque overlay",
            },
        )
    )
    assert np.all(render_pose(occluded) == np.asarray(RenderStyle().occluder))


@pytest.mark.parametrize(
    "changes",
    [
        {"body": None},
        {"image_size": (4096, 160)},
        {"coordinate_system": "game_metres"},
        {
            "body": {
                "vertices": ((0, 0), (1, 1), (2, 2)),
                "uncertainty_pixels": 0,
                "note": "degenerate",
            }
        },
        {
            "terrain": [
                {"points": ((50, 100), (20, 100)), "uncertainty_pixels": 0, "note": "reversed"}
            ]
        },
        {
            "body": {
                "vertices": ((-1, 0), (20, 0), (20, 20)),
                "uncertainty_pixels": 0,
                "note": "outside",
            }
        },
        {"provenance": "reviewed_construction"},
    ],
)
def test_label_contract_rejects_unsupported_or_unreviewed_geometry(changes):
    with pytest.raises(ValueError):
        geometry(**changes)


def test_directed_landmark_angle_is_annotation_only_and_requires_both_ends():
    def landmark(role, point):
        return {
            "role": role,
            "point": point,
            "uncertainty_pixels": 1,
            "note": "Synthetic semantic mark",
        }

    forward = geometry(landmarks=(landmark("front", (145, 80)), landmark("rear", (75, 80))))
    reverse = geometry(landmarks=(landmark("rear", (145, 80)), landmark("front", (75, 80))))
    assert forward.diagnostic_directed_angle() == 0
    assert abs(reverse.diagnostic_directed_angle()) == math.pi
    assert geometry(landmarks=(landmark("front", (145, 80)),)).diagnostic_directed_angle() is None
    assert np.array_equal(render_pose(forward), render_pose(reverse))
    assert forward.sha256 != reverse.sha256


class MissingMeasurer:
    received: ClassVar[list] = []

    def __init__(self, measurement_profile):
        self.profile = measurement_profile

    def measure(self, rgb):
        self.received.append(rgb.copy())
        invalid = {"valid": False, "reason": "Fixed detector abstains"}
        return PixelMeasurement(
            self.profile.profile_id,
            self.profile.expected_size,
            body=invalid,
            wheels=invalid,
            terrain=invalid,
        )


def test_rgb_is_only_actor_geometry_route_and_missing_context_stays_masked(monkeypatch):
    MissingMeasurer.received = []
    monkeypatch.setattr("gradientclimb.simulation.visual_replay.HCRPixelMeasurer", MissingMeasurer)
    pose = geometry()
    result = replay_poses(plan(pose), profile())
    assert np.array_equal(MissingMeasurer.received[0], render_pose(pose))
    observed = result["rows"][0]["observation"]
    # Complete annotation geometry cannot override an abstaining pixel measurer.
    assert not any(observed["valid"])
    assert not any(observed["vector"])
    assert "directed_angle" not in observed
    assert not observed["input_authorized"]
    assert result["summary"]["physics_substeps"] == result["summary"]["simulator_seconds"] == 0
    assert result["summary"]["policy_decisions"] == result["summary"]["gameplay_episodes"] == 0
    assert result["summary"]["rendered_observations"] == 1


def test_history_resets_and_failed_measurement_emits_completed_prefix(monkeypatch):
    class FailsThird(MissingMeasurer):
        def measure(self, rgb):
            if len(self.received) == 2:
                raise RuntimeError("Synthetic detector failure")
            return super().measure(rgb)

    FailsThird.received = []
    monkeypatch.setattr("gradientclimb.simulation.visual_replay.HCRPixelMeasurer", FailsThird)
    sequence = plan(*(geometry(str(i)) for i in range(3))).model_dump(mode="json")
    sequence["frames"][1]["continuity_id"] = "new-segment"
    events = []
    with pytest.raises(RuntimeError, match="Synthetic detector failure"):
        replay_poses(sequence, profile(), on_event=events.append)
    completed = [e["result"] for e in events if e["event"] == "frame_completed"]
    assert len(completed) == 2 and completed[1]["history_reset"]
    assert not completed[1]["observation"]["temporal_contiguous"]
    assert events[-1]["event"] == "frame_failed" and events[-1]["frame_index"] == 2


def test_reviewed_replay_verifies_pinned_sources_before_rendering(tmp_path, monkeypatch):
    source, protocol = tmp_path / "source.png", tmp_path / "protocol.json"
    source.write_bytes(b"Opaque source fixture; renderer must not decode this")
    protocol.write_text('{"purpose":"synthetic protocol fixture"}', encoding="utf-8")
    pose = geometry(
        provenance="reviewed_construction",
        source_image={"path": source.name, "sha256": sha256_file(source)},
        label_protocol={"path": protocol.name, "sha256": sha256_file(protocol)},
        reviewer="fixture",
        reviewed_at=datetime(2026, 9, 12, tzinfo=UTC),
    )
    monkeypatch.setattr("gradientclimb.simulation.visual_replay.HCRPixelMeasurer", MissingMeasurer)
    sequence = plan(pose, partition="construction")
    result = replay_poses(sequence, profile(), project_root=tmp_path)
    assert len(result["verified_evidence"]) == 2
    source.write_bytes(b"Changed source")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_poses(sequence, profile(), project_root=tmp_path)
    with pytest.raises(ValueError, match="requires a project root"):
        replay_poses(sequence, profile())


def test_existing_real_pixel_pipeline_runs_on_original_rendered_shapes():
    pytest.importorskip("cv2")
    result = replay_poses(plan(geometry("one"), geometry("two")), profile())
    observed = result["rows"][-1]["observation"]
    assert observed["measurement"]["body"]["valid"]
    assert observed["temporal_contiguous"]
    for name in ("hud_displayed_progress_div_1000", "previous_os_gas", "episode_elapsed_div_60"):
        assert not observed["valid"][FEATURE_NAMES.index(name)]
    assert result["summary"]["declared_contiguous_pose_interval_seconds"] == 0.1
    assert result["summary"]["process_cpu_core_seconds"] >= 0
    assert result["summary"]["sampled_peak_process_rss_bytes"] > 0


def test_landmark_annotation_changes_cannot_change_actor_pixels_or_features():
    pytest.importorskip("cv2")
    ordinary = geometry()
    annotated = geometry(
        landmarks=tuple(
            {
                "role": role,
                "point": point,
                "uncertainty_pixels": uncertainty,
                "note": f"Diagnostic-only annotation {role}",
            }
            for role, point, uncertainty in (
                ("front", (10, 10), 1),
                ("rear", (240, 140), 12),
                ("head", (100, 45), 3),
                ("roof", (100, 70), 5),
            )
        )
    )
    a = replay_poses(plan(ordinary), profile())["rows"][0]
    b = replay_poses(plan(annotated), profile())["rows"][0]
    assert a["label_sha256"] != b["label_sha256"]
    assert a["rgb_sha256"] == b["rgb_sha256"]
    assert a["observation"] == b["observation"]


def test_pixel_callback_receives_exact_copy_after_perception_without_rerender(monkeypatch):
    import gradientclimb.simulation.visual_replay as replay_module

    MissingMeasurer.received = []
    monkeypatch.setattr(replay_module, "HCRPixelMeasurer", MissingMeasurer)
    rendered_images = []

    def counted_render(*args):
        rgb = render_pose(*args)
        rendered_images.append(rgb)
        return rgb

    monkeypatch.setattr(replay_module, "render_pose", counted_render)
    delivered = []

    def retain_and_mutate(index, rgb):
        assert len(MissingMeasurer.received) == index + 1
        assert np.array_equal(rgb, MissingMeasurer.received[index])
        assert not np.shares_memory(rgb, rendered_images[index])
        delivered.append((index, hashlib.sha256(rgb.tobytes()).hexdigest()))
        rgb[:] = 0

    result = replay_poses(
        plan(geometry("one"), geometry("two")), profile(), on_pixels=retain_and_mutate
    )
    assert len(rendered_images) == 2
    assert delivered == [(row["frame_index"], row["rgb_sha256"]) for row in result["rows"]]
    assert all(np.any(image) for image in rendered_images)
    assert all(not row["observation"]["measurement"]["body"]["valid"] for row in result["rows"])


def test_pixel_retention_failure_emits_failure_and_preserves_completed_prefix(monkeypatch):
    monkeypatch.setattr("gradientclimb.simulation.visual_replay.HCRPixelMeasurer", MissingMeasurer)
    events = []

    def retention(index, _rgb):
        if index == 1:
            raise OSError("Synthetic image publication failure")

    with pytest.raises(OSError, match="Synthetic image publication failure"):
        replay_poses(
            plan(*(geometry(str(i)) for i in range(3))),
            profile(),
            on_event=events.append,
            on_pixels=retention,
        )
    assert [
        event["result"]["frame_index"] for event in events if event["event"] == "frame_completed"
    ] == [0]
    assert events[-1]["event"] == "frame_failed"
    assert events[-1]["frame_index"] == 1
    assert not any(event["event"] == "replay_completed" for event in events)
