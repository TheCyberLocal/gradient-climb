import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gradientclimb.artifacts import sha256_file
from gradientclimb.experiments import RunRecorder, load_run


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Optional installed media tools",
)
def test_sparse_review_retains_original_pts_and_sealed_frames(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("owner_review", scripts / "review_owner_video.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ffmpeg, ffprobe = (Path(shutil.which(x)) for x in ("ffmpeg", "ffprobe"))
    store = tmp_path / "artifacts"
    with RunRecorder(store, "synthetic-media-fixture", {}, source_root=tmp_path) as source:
        video = source.directory / "input.mp4"
        subprocess.run(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=10:duration=2",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
        )
        source.register_artifact(video, "synthetic_media")
        source.finalize()
    plan = {
        "schema_version": "owner-video-review-3.0",
        "status": "registered",
        "experiment_id": "synthetic-sparse-review",
        "requested_timestamps_seconds": [0, 0.5, 1.5],
        "wall_budget_seconds": 120,
        "output_budget_bytes": 128 * 1024**2,
        "source_run_id": source.run_id,
        "source_video": {
            "path": video.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(video),
        },
        "ffmpeg": {"path": str(ffmpeg), "sha256": sha256_file(ffmpeg)},
        "ffprobe": {"path": str(ffprobe), "sha256": sha256_file(ffprobe)},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    result = module.review(tmp_path, "plan.json")
    assert result["verification"]["valid"]
    directory = store / "runs" / result["run_id"]
    report = json.loads((directory / "review/review.json").read_bytes())
    assert [row["source_pts_seconds"] for row in report["frames"]] == [0, 0.5, 1.5]
    assert report["probe_video_frames_read"] == 20
    assert len(list((directory / "review").glob("frame-*.png"))) == 3
    run = load_run(store, result["run_id"])
    assert run["episodes"] == 0 and run["summary"]["training_eligible"] is False
    with pytest.raises(ValueError, match="already consumed"):
        module.review(tmp_path, "plan.json")
