"""Canonical synthetic receipt/callback tests; no native input or inference."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gradientclimb.artifacts import sha256_file
from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect
from gradientclimb.experiments import RunRecorder, load_run, verify_run
from gradientclimb.experiments import reader_collection3 as collection
from gradientclimb.experiments.reader_receipts3 import (
    publish_session_declaration3,
    verify_session_declaration3,
)
from gradientclimb.experiments.reader_units3 import ReaderUnitResolver3, result_selection_coverage

ROOT = Path(__file__).resolve().parents[1]


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8", newline="\n")
    return {"path": name, "sha256": sha256_file(path)}


def recorder(root, experiment, config):
    return RunRecorder(
        root / "artifacts",
        experiment,
        config,
        environment="synthetic",
        source_root=root,
        telemetry_interval_seconds=0,
    )


def setup(root, *, stages=(0,), role="native-session-3"):
    protocol = write(
        root,
        "protocol.json",
        {
            "schema_version": "reader-validation-3.0",
            "status": "registered",
            "registered_at": "2020-01-01T00:00:00+00:00",
            "environment": "synthetic",
        },
    )
    selection = write(
        root,
        "selection.json",
        {
            "schema_version": "reader-source-selection-3.0",
            "result_endpoint_rule": "first_retained_terminal_per_attempt",
            "non_result_units": [
                {
                    "source_config_id": role,
                    "unit_id": f"tune-{i}",
                    "stage": f"attempt:{stage}:start:tune",
                    "capture_ordinal": i,
                }
                for i, stage in enumerate(stages)
            ],
        },
    )
    with recorder(root, "reader-candidate-freeze-3.0", {"protocol": protocol}) as run:
        run.finalize(episodes=0)
    receipt = publish_session_declaration3(
        root,
        "artifacts",
        protocol,
        "new-whole-session",
        "heldout",
        {"reader_sampling_source_id": role},
        candidate_freeze_run_id=run.run_id,
        selection_ref=selection,
    )
    return protocol, receipt


def prepare(root, protocol, receipt, **kwargs):
    return collection.prepare_native_reader_collection(
        root,
        "artifacts",
        receipt["run_id"],
        "native-session-3",
        expected_protocol=protocol,
        attempts=12,
        environment="synthetic",
        **kwargs,
    )


def observation(stamp, *, state="result", value=0):
    return SimpleNamespace(
        state=state,
        variant="synthetic",
        confidence=1.0,
        frame=CapturedFrame(
            np.full((4, 4, 3), value, dtype=np.uint8),
            stamp,
            stamp + 20,
            "synthetic",
            ClientRect(0, 0, 4, 4),
            "synthetic",
            0.00002,
        ),
    )


def module():
    spec = importlib.util.spec_from_file_location(
        "reader_native_test", ROOT / collection.SOURCE_FILES[0]
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_actual_sealed_preflight_binds_config_and_uses_frozen_selection(tmp_path):
    protocol, receipt = setup(tmp_path)
    write(tmp_path, "selection.json", {"changed_after_declaration": True})
    prepared = prepare(tmp_path, protocol, receipt)
    assert prepared.selection.non_result_units[0].stage == "attempt:0:start:tune"
    assert prepared.bind({"baseline": "always_gas"}) == {
        **receipt["required_source_configuration"],
        "baseline": "always_gas",
    }
    assert prepared.preflight_wall_seconds >= 0 and prepared.preflight_cpu_core_seconds >= 0
    with pytest.raises(ValueError, match="conflicts"):
        prepared.bind({"reader_sampling_source_id": "overridden"})


@pytest.mark.parametrize("stages", [(12,), (0, 0)])
def test_out_of_budget_or_duplicate_opportunities_rejected(tmp_path, stages):
    protocol, receipt = setup(tmp_path, stages=stages)
    with pytest.raises(ValueError, match="opportunities"):
        prepare(tmp_path, protocol, receipt)


def test_receipt_tamper_rejected_before_source_creation(tmp_path):
    protocol, receipt = setup(tmp_path)
    directory = tmp_path / "artifacts/runs" / receipt["run_id"]
    payload = directory / "reader-session-declaration.json"
    payload.write_bytes(payload.read_bytes() + b" ")
    with pytest.raises(ValueError, match="seal failed"):
        prepare(tmp_path, protocol, receipt)


def test_wrong_source_role_and_protocol_rejected(tmp_path):
    protocol, receipt = setup(tmp_path, role="different")
    with pytest.raises(ValueError, match="role"):
        prepare(tmp_path, protocol, receipt)
    with pytest.raises(ValueError, match="protocol"):
        prepare(tmp_path, {**protocol, "sha256": "0" * 64}, receipt)


def test_canonical_no_reader_frames_resolve_and_keep_out_of_attempt_diagnostics(tmp_path):
    protocol, receipt = setup(tmp_path)
    prepared = prepare(tmp_path, protocol, receipt)
    with recorder(tmp_path, "synthetic-native-source", prepared.bind({})) as run:
        with collection.NativeReaderCollection(run, prepared) as sink:
            terminal = module().ResultScoreCollector(None, capture_sink=sink)
            assert terminal(observation(100)) is True
            sink.start_attempt(0)
            sink.observe_start(observation(200, state="playing"))
            sink.observe_start(observation(300, state="tune"))
            sink.observe_start(observation(400, state="tune"))  # never replace first
            assert terminal(observation(500)) is True  # start-reset, not attempt zero
            start = len(terminal.readings)
            sink.activate_endpoint(0)
            assert terminal(observation(600, value=1)) is True
            assert terminal(observation(700, value=2)) is True
        run.finalize(
            episodes=1,
            episode_summaries=[
                {"attempt": {"index": 0}, "terminal_readings": terminal.readings[start:]},
                {"attempt": {"index": 1}, "terminal_readings": []},
            ],
        )
    assert verify_run(tmp_path / "artifacts", run.run_id)["valid"]
    record = load_run(tmp_path / "artifacts", run.run_id)
    verified = verify_session_declaration3(
        tmp_path,
        "artifacts",
        receipt["run_id"],
        protocol,
        "new-whole-session",
        "heldout",
        record["start_time"],
        [record],
    )
    resolver = ReaderUnitResolver3(tmp_path)
    terminal_artifacts = [a for a in record["artifact_manifest"] if a["kind"] == "terminal_frame"]
    assert len(terminal_artifacts) == 2
    for i, artifact in enumerate(terminal_artifacts):
        frame = {
            "path": f"artifacts/runs/{run.run_id}/{artifact['path']}",
            "sha256": artifact["sha256"],
        }
        arguments = {
            "source_run_id": run.run_id,
            "label_is_result": True,
            "purpose": "heldout",
            "selection_document": verified["selection_document"],
        }
        if i == 0:
            unit = resolver.resolve(frame, **arguments)
            assert unit["attempt_index"] == 0 and not unit["qualification_eligible"]
        else:
            with pytest.raises(ValueError, match="first retained"):
                resolver.resolve(frame, **arguments)
    negative = next(
        a for a in record["artifact_manifest"] if a["kind"] == "reader_non_result_frame"
    )
    assert negative["metadata"]["started_ns"] == 300
    resolver.resolve(
        {"path": f"artifacts/runs/{run.run_id}/{negative['path']}", "sha256": negative["sha256"]},
        source_run_id=run.run_id,
        label_is_result=False,
        purpose="heldout",
        selection_document=verified["selection_document"],
    )
    coverage = result_selection_coverage([record])
    assert len(coverage["expected_result_unit_ids"]) == 1
    assert coverage["attempts_without_retained_terminal"][0]["attempt_index"] == 1
    summary = record["summary"]["reader_collection"]
    assert summary["out_of_attempt_terminal_frames"] == 2
    assert summary["negative_opportunities_completed"] == 1
    assert not terminal.frames  # no delayed duplicate terminal artifact publication


def test_numeric_failure_keeps_terminal_identity_and_registers_failure_journal(tmp_path):
    protocol, receipt = setup(tmp_path, stages=(0, 3))
    prepared = prepare(tmp_path, protocol, receipt)
    with recorder(tmp_path, "synthetic-native-source", prepared.bind({})) as run:
        sink = collection.NativeReaderCollection(run, prepared)

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic numeric failure")

        loaded = module()
        terminal = loaded.ResultScoreCollector(SimpleNamespace(read=fail), capture_sink=sink)
        with pytest.raises(RuntimeError, match="numeric failure"), sink:
            sink.start_attempt(0)
            sink.activate_endpoint(0)
            terminal(observation(100))
        summary = {"attempt": {"index": 0}, "distance": None}
        loaded.annotate_terminal_score(summary, terminal, 0, 0)
        run.finalize(
            status="failed",
            episodes=1,
            episode_summaries=[summary],
        )
    assert verify_run(tmp_path / "artifacts", run.run_id)["valid"]
    record = load_run(tmp_path / "artifacts", run.run_id)
    assert record["summary"]["episode_summaries"][0]["terminal_readings"][0]["timestamp_ns"] == 110
    assert terminal.readings[0]["timestamp_ns"] == 110
    assert terminal.readings[0]["distance_meters"] is None
    assert len(result_selection_coverage([record])["expected_result_unit_ids"]) == 1
    journal = next(
        a for a in record["artifact_manifest"] if a["kind"] == "reader_collection_sampling_journal"
    )
    rows = [
        json.loads(line) for line in (run.directory / journal["path"]).read_bytes().splitlines()
    ]
    assert rows[-1]["event"] == "collection_closed" and rows[-1]["failed"]
    assert rows[-1]["summary"]["negative_opportunities_started"] == 1
    assert len(rows[-1]["summary"]["negative_opportunities_missing"]) == 2


def test_storage_failure_journals_started_but_not_completed(tmp_path, monkeypatch):
    protocol, receipt = setup(tmp_path)
    prepared = prepare(tmp_path, protocol, receipt)
    with recorder(tmp_path, "synthetic-native-source", prepared.bind({})) as run:
        sink = collection.NativeReaderCollection(run, prepared)
        monkeypatch.setattr(
            collection.Image.Image,
            "save",
            lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic storage failure")),
        )
        with pytest.raises(OSError, match="storage"), sink:
            sink.start_attempt(0)
            sink.observe_start(observation(100, state="tune"))
        run.finalize(status="failed", episodes=0)
    rows = [json.loads(line) for line in sink.journal.read_bytes().splitlines()]
    assert any(row["event"] == "capture_started" for row in rows)
    assert not any(row["event"] == "capture_completed" for row in rows)
    assert rows[-1]["summary"]["negative_opportunities_completed"] == 0
    assert verify_run(tmp_path / "artifacts", run.run_id)["valid"]


def test_prediction_safe_console_has_only_allowlisted_fields():
    assert collection.prediction_safe_console(
        run_id="source", attempt_index=8, failed=True, completed=9
    ) == {
        "reader_collection": collection.VERSION,
        "run_id": "source",
        "attempt_index": 8,
        "status": "halted",
        "completed_attempts": 9,
        "predictions_withheld_until_annotations_sealed": True,
    }


@pytest.mark.parametrize(
    "protocol_file",
    [
        "experiments/drafts/reader-native-collection-3/amendment.json",
        "experiments/definitions/cycle-3-native-reliability-2.3.json",
    ],
)
def test_draft_or_legacy_native_protocol_cannot_dispatch_instrumented_mode(
    monkeypatch, protocol_file
):
    loaded = module()
    native_calls = []
    monkeypatch.setattr(loaded, "discover_windows", lambda: native_calls.append("discovery"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_screen_episodes.py",
            "--hud",
            "unused.json",
            "--protocol",
            str(ROOT / protocol_file),
            "--project-root",
            str(ROOT),
            "--reader-session-receipt",
            "unused",
            "--reader-sampling-source-id",
            "future",
        ],
    )
    with pytest.raises(ValueError):
        loaded.main()
    assert not native_calls


def test_registered_amendment_requires_exact_source_and_carry_forward(tmp_path, monkeypatch):
    document = json.loads(
        (ROOT / "experiments/drafts/reader-native-collection-3/amendment.json").read_bytes()
    )
    document.update(status="registered", registered_at=datetime.now(UTC).isoformat())
    protocol = write(
        tmp_path,
        "protocol.json",
        {
            "schema_version": "reader-validation-3.0",
            "status": "registered",
            "registered_at": "2020-01-01T00:00:00+00:00",
        },
    )
    document["reader_protocol"] = protocol
    parent = tmp_path / document["native_protocol"]["path"]
    parent.parent.mkdir(parents=True)
    parent.write_bytes((ROOT / document["native_protocol"]["path"]).read_bytes())
    document["implementation_files"] = []
    for name in collection.SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        document["implementation_files"].append({"path": name, "sha256": sha256_file(target)})
    validated = []
    monkeypatch.setattr(
        collection, "validate_protocol", lambda parent, args: validated.append(parent)
    )
    args = SimpleNamespace(exploratory=False)
    amendment, parent = collection.validate_native_reader_amendment(tmp_path, document, args)
    assert validated == [parent] and amendment.maximum_new_sessions == 1
    document["prior_session_run_ids"] = list(collection.PRIOR_RUNS[:1])
    with pytest.raises(ValueError, match="prior reliability"):
        collection.validate_native_reader_amendment(tmp_path, document, args)
    document["prior_session_run_ids"] = list(collection.PRIOR_RUNS)
    document["implementation_files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        collection.validate_native_reader_amendment(tmp_path, document, args)


@pytest.mark.parametrize("failed", [False, True])
def test_remaining_budget_cannot_be_renamed_or_retry_failures(tmp_path, monkeypatch, failed):
    # No native operation: synthetic records carry the exact native budget marker.
    prior_ids = []
    for _ in range(2):
        with recorder(tmp_path, "synthetic-prior", {}) as run:
            run.finalize(episodes=0)
        prior_ids.append(run.run_id)
    monkeypatch.setattr(collection, "PRIOR_RUNS", tuple(prior_ids))
    collection.assert_reliability_capacity(tmp_path, "artifacts")
    with recorder(
        tmp_path,
        "an-arbitrary-renamed-experiment",
        {
            "reader_native_collection_version": collection.VERSION,
        },
    ) as run:
        with pytest.raises(ValueError, match="already consumed"):
            collection.assert_reliability_capacity(tmp_path, "artifacts")
        run.finalize(status="failed" if failed else "completed", episodes=0)
    with pytest.raises(ValueError, match="already consumed"):
        collection.assert_reliability_capacity(tmp_path, "artifacts")


@pytest.mark.parametrize("fault", ["amendment", "constructor", "adapter"])
def test_runner_setup_failure_closes_both_owners_without_native_actions(
    tmp_path,
    monkeypatch,
    fault,
):
    protocol, receipt = setup(tmp_path)
    prepared = prepare(tmp_path, protocol, receipt)
    loaded = module()
    events = []
    amendment = SimpleNamespace(
        reader_protocol=SimpleNamespace(model_dump=lambda: protocol),
        model_dump=lambda **kwargs: {"synthetic": True},
    )
    monkeypatch.setattr(collection, "validate_native_reader_amendment", lambda *a: (amendment, {}))
    monkeypatch.setattr(collection, "prepare_native_reader_collection", lambda *a, **k: prepared)
    monkeypatch.setattr(collection, "assert_reliability_capacity", lambda *a: None)
    for name in ("hud.json", "ui.json", "measurement.json", "amendment.json"):
        write(tmp_path, name, {})
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_screen_episodes.py",
            "--hud",
            str(tmp_path / "hud.json"),
            "--ui-profile",
            str(tmp_path / "ui.json"),
            "--measurement-profile",
            str(tmp_path / "measurement.json"),
            "--protocol",
            str(tmp_path / "amendment.json"),
            "--root",
            str(tmp_path / "artifacts"),
            "--project-root",
            str(tmp_path),
            "--reader-session-receipt",
            receipt["run_id"],
            "--reader-sampling-source-id",
            "native-session-3",
        ],
    )
    target = SimpleNamespace(as_dict=lambda: {"synthetic": True})
    monkeypatch.setattr(
        loaded, "HostBudgetGuard", lambda *a: SimpleNamespace(check=lambda **k: None)
    )
    monkeypatch.setattr(loaded, "discover_windows", lambda: [target])
    monkeypatch.setattr(loaded, "MeasurementProfile", lambda **k: None)
    monkeypatch.setattr(loaded, "HCRPixelMeasurer", lambda *a: None)
    monkeypatch.setattr(loaded.HUDDigitReader, "from_manifest", lambda *a: None)
    monkeypatch.setattr(loaded.PausedDistanceReader, "from_gameplay_manifest", lambda *a: None)
    monkeypatch.setattr(
        loaded,
        "ScreenFeatureBridge",
        lambda *a: SimpleNamespace(
            schema={}, schema_id="synthetic", hud_reader=SimpleNamespace(glyphs=[])
        ),
    )
    backend = SimpleNamespace(close=lambda: events.append("backend_closed"))
    controller = SimpleNamespace(
        close=lambda: events.append("controller_closed"),
        release=lambda: events.append("release_called"),
    )
    monkeypatch.setattr(loaded, "WindowsPedalBackend", lambda *a, **k: backend)
    monkeypatch.setattr(loaded, "PedalController", lambda *a, **k: controller)

    def adapter(*args, **kwargs):
        if fault == "adapter":
            raise OSError("injected adapter failure")
        return SimpleNamespace(recognizer=SimpleNamespace(profile=SimpleNamespace(variants=[])))

    monkeypatch.setattr(loaded, "NativeGameAdapter", adapter)
    published = []

    def synthetic_recorder(*args, **kwargs):
        kwargs.update(source_root=tmp_path, telemetry_interval_seconds=0, environment="synthetic")
        run = RunRecorder(*args, **kwargs)
        published.append(run)
        return run

    monkeypatch.setattr(loaded, "RunRecorder", synthetic_recorder)
    original = RunRecorder.register_artifact

    def fail_copy(run, path, kind, metadata=None):
        if (fault == "constructor" and kind == "reader_collection_declaration_source") or (
            fault == "amendment" and kind == "reader_native_acquisition_amendment"
        ):
            raise OSError(f"injected {fault} failure")
        return original(run, path, kind, metadata)

    monkeypatch.setattr(RunRecorder, "register_artifact", fail_copy)
    with pytest.raises(OSError, match="injected"):
        loaded.main()
    assert events == ["controller_closed", "backend_closed"]
    if fault == "adapter":
        assert not published
    else:
        record = load_run(tmp_path / "artifacts", published[0].run_id)
        assert record["status"] == "failed"
        journal = next(
            a
            for a in record["artifact_manifest"]
            if a["kind"] == "reader_collection_sampling_journal"
        )
        rows = [
            json.loads(line)
            for line in (published[0].directory / journal["path"]).read_bytes().splitlines()
        ]
        assert rows[0]["event"] == "collection_opened"
        assert not any(row["event"] == "capture_started" for row in rows)
        assert verify_run(tmp_path / "artifacts", published[0].run_id)["valid"]


def test_setup_cleanup_preserves_primary_failure_and_transfer():
    events = []

    def controller_close():
        events.append("controller_closed")
        raise RuntimeError("cleanup failure")

    controller = SimpleNamespace(close=controller_close)
    backend = SimpleNamespace(close=lambda: events.append("backend_closed"))
    with (
        pytest.raises(OSError, match="primary") as error,
        collection.ReaderSetupOwnership(
            controller,
            backend,
        ),
    ):
        raise OSError("primary setup failure")
    assert events == ["controller_closed", "backend_closed"]
    assert any("cleanup failure" in note for note in error.value.__notes__)
    events.clear()
    with collection.ReaderSetupOwnership(controller, backend) as ownership:
        ownership.transfer()
    assert not events
