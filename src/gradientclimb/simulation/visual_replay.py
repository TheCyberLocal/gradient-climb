"""Bounded pose replay through RGB and the existing real-screen feature bridge.

This is an offline construction library, not a training environment or registered
experiment runner. A caller must preregister reviewed labels before dispatch and
persist emitted operations when running a measured study. No actor sees labels,
source images, directed annotation angles, collision state or privileged values.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import psutil

from gradientclimb.artifacts import canonical_json, sha256_file
from gradientclimb.artifacts.archive import _inside, _no_links
from gradientclimb.perception.measurements import HCRPixelMeasurer, MeasurementProfile
from gradientclimb.perception.screen_features import ScreenFeatureBridge

from .visual_labels import PoseReplay
from .visual_renderer import RENDERER_VERSION, RenderStyle, render_pose

REPLAY_VERSION = "visual-shared-perception-replay-3.0"


class _UnavailableHUD:
    def read(self, _rgb):
        return {"valid": False, "reason": "Construction renderer contains no HUD evidence"}


def _verify_sources(replay, project_root):
    if replay.source_partition == "synthetic":
        return []
    if project_root is None:
        raise ValueError("Reviewed pose replay requires a project root for source verification")
    root = Path(project_root).absolute()
    _no_links(root)
    references = {}
    for frame in replay.frames:
        for ref in (frame.geometry.source_image, frame.geometry.label_protocol):
            if ref.path in references and references[ref.path] != ref.sha256:
                raise ValueError("Conflicting hashes for one geometry evidence path")
            references[ref.path] = ref.sha256
    for path, digest in references.items():
        if sha256_file(_inside(root, path)) != digest:
            raise ValueError(f"Geometry evidence hash mismatch: {path}")
    return [{"path": path, "sha256": digest} for path, digest in sorted(references.items())]


def replay_poses(
    replay: PoseReplay,
    profile: MeasurementProfile,
    *,
    project_root=None,
    style: RenderStyle | None = None,
    history=4,
    maximum_frame_interval_seconds=0.5,
    on_event=None,
    on_pixels=None,
):
    """Return pixel-derived observations and separate diagnostic cost receipts.

    Timestamps are the declared pose timeline, not capture timestamps or measured
    simulator time. History resets across continuity IDs. Missing HUD, previous
    OS input and episode clocks stay masked; geometry is never copied into features.
    The event callback receives completed rows immediately for external journaling.
    The optional pixel callback receives ``(index, rgb.copy())`` after observation
    and image hashing. It can retain exactly the measured pixels without another
    render, and cannot mutate the image supplied to perception. Its time is included
    in whole-function costs, outside component render and perception timers.
    """
    started, cpu_started = time.perf_counter(), time.process_time()
    replay = PoseReplay.model_validate(replay)
    style = RenderStyle() if style is None else RenderStyle.model_validate(style)
    if tuple(profile.expected_size) != replay.frames[0].geometry.image_size:
        raise ValueError("Shared measurer and rendered image dimensions must agree")
    emit = on_event if on_event is not None else lambda _event: None
    emit({"event": "replay_started", "planned_frames": len(replay.frames)})
    evidence = _verify_sources(replay, project_root)
    bridge = ScreenFeatureBridge(
        HCRPixelMeasurer(profile),
        _UnavailableHUD(),
        history=history,
        max_interval_seconds=maximum_frame_interval_seconds,
    )
    process = psutil.Process()
    rss_samples = [process.memory_info().rss]
    rows = []
    previous = None
    render_seconds = perception_seconds = pose_interval_seconds = 0.0
    for index, frame in enumerate(replay.frames):
        identity = {"frame_index": index, "label_id": frame.geometry.label_id}
        emit({"event": "frame_started", **identity})
        try:
            reset = previous is None or previous.continuity_id != frame.continuity_id
            if reset:
                bridge.reset()
            elif frame.timestamp_ns - previous.timestamp_ns <= maximum_frame_interval_seconds * 1e9:
                pose_interval_seconds += (frame.timestamp_ns - previous.timestamp_ns) / 1e9
            render_started = time.perf_counter()
            rgb = render_pose(frame.geometry, style)
            rendered = time.perf_counter()
            # The only route from declared scene geometry to observation is RGB.
            observation = bridge.observe(rgb, frame.timestamp_ns)
            measured = time.perf_counter()
            render_seconds += rendered - render_started
            perception_seconds += measured - rendered
            rss_samples.append(process.memory_info().rss)
            row = {
                **identity,
                "label_sha256": frame.geometry.sha256,
                "timestamp_ns": frame.timestamp_ns,
                "continuity_id": frame.continuity_id,
                "history_reset": reset,
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "render_seconds": rendered - render_started,
                "perception_seconds": measured - rendered,
                "observation": observation.as_dict(),
            }
            canonical_json(row)
            if on_pixels is not None:
                on_pixels(index, rgb.copy())
            emit({"event": "frame_completed", "result": row})
        except BaseException as error:
            try:
                emit(
                    {
                        "event": "frame_failed",
                        **identity,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            except BaseException as journal_error:  # noqa: BLE001 - preserve the detector failure
                error.add_note(f"Replay failure journal also failed: {journal_error}")
            raise
        rows.append(row)
        previous = frame
    result = {
        "schema_version": REPLAY_VERSION,
        "replay_sha256": replay.sha256,
        "renderer_version": RENDERER_VERSION,
        "render_style_sha256": style.sha256,
        "source_partition": replay.source_partition,
        "verified_evidence": evidence,
        "shared_observation_schema": bridge.schema,
        "shared_observation_schema_id": bridge.schema_id,
        "rows": rows,
        "summary": {
            "fidelity_level": "declared_screen_pixel_pose_replay",
            "coordinate_system": "screen_pixels_x_right_y_down",
            "rendered_observations": len(rows),
            "extracted_observations": len(rows),
            "physics_substeps": 0,
            "simulator_seconds": 0,
            "declared_contiguous_pose_interval_seconds": pose_interval_seconds,
            "policy_decisions": 0,
            "optimizer_updates": 0,
            "gameplay_episodes": 0,
            "new_real_interaction_seconds": 0,
            "qualifies_real_game": False,
            "render_wall_seconds": render_seconds,
            "perception_wall_seconds": perception_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "process_cpu_core_seconds": time.process_time() - cpu_started,
            "sampled_peak_process_rss_bytes": max(rss_samples),
            "rss_sample_count": len(rss_samples),
            "cost_scope": "Function entry until summary clocks are read, including validation, source hashing and per-frame callbacks. Excludes imports, final replay_completed callback, post-return publication and inherited annotation/development costs. Component render/perception timers exclude callbacks and hashing. CPU covers this process's threads, excludes children. RSS sampled before loop and after each measured frame before its callback; brief peaks may be missed.",
        },
        "limitations": [
            "Original procedural shapes test shared visual observability, not game appearance fidelity.",
            "Directed landmark annotation is available only for diagnostic comparison; existing perception retains modulo-pi orientation.",
            "Labels, their uncertainty and source-image bytes never enter actor observation values directly.",
            "Source hashes verify bytes, not annotation accuracy, partition governance or preregistration chronology; the experiment runner must bind those before dispatch.",
            "No physical evolution, collision calibration, real-metre scale, control effect, learning or competence is measured.",
        ],
    }
    emit({"event": "replay_completed", "summary": result["summary"]})
    return result
