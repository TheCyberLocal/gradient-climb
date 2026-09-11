"""Retained scoped source evidence; public OS/hardware queries are mocked."""

import hashlib
from types import SimpleNamespace

from gradientclimb.artifacts import canonical_json, sha256_file
from gradientclimb.telemetry import provenance


def test_source_state_retains_exact_diff_and_scoped_untracked_hashes(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    file = tmp_path / "src/new.py"
    file.write_text("print('source')\n")
    patch = "diff --git a/src/a.py b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    calls = []

    def command(args, cwd=None, **kwargs):
        calls.append(args)
        if args[1] == "rev-parse":
            return "a" * 40
        if args[1] == "status":
            return "?? src/new.py"
        if args[1] == "diff":
            assert kwargs["strip_output"] is False
            return patch
        if args[1] == "ls-files":
            return "src/new.py\0"
        raise AssertionError(args)

    monkeypatch.setattr(provenance, "_command", command)
    monkeypatch.setattr(provenance, "_nvidia_smi", lambda: None)
    state = {}
    record = provenance.capture_provenance(tmp_path, source_state=state)
    assert state["tracked_diff"] == patch
    assert state["untracked_source_sha256"] == {"src/new.py": sha256_file(file)}
    assert (
        record["source_diff_sha256"]
        == hashlib.sha256(
            canonical_json({"diff": patch, "untracked": state["untracked_source_sha256"]}).encode()
        ).hexdigest()
    )
    untracked_call = next(args for args in calls if args[1] == "ls-files")
    assert untracked_call[untracked_call.index("--") + 1 :] == [
        "src",
        "tests",
        "scripts",
        "configs",
    ]
    assert "--exclude-standard" in untracked_call
    assert state["untracked_contents_included"] is False
    assert state["collection_is_atomic"] is False


def test_unavailable_source_query_does_not_claim_a_complete_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(
        provenance, "_command", lambda args, *a, **k: "a" * 40 if args[1] == "rev-parse" else None
    )
    monkeypatch.setattr(provenance, "_nvidia_smi", lambda: None)
    state = {}
    record = provenance.capture_provenance(tmp_path, source_state=state)
    assert record["source_diff_sha256"] is None
    assert state["tracked_diff_available"] is False
    assert state["untracked_manifest_available"] is False


def test_machine_fingerprint_excludes_imported_framework_state(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(provenance, "_nvidia_smi", lambda: None)
    monkeypatch.setattr(provenance, "psutil", None)
    monkeypatch.setattr(provenance.os, "cpu_count", lambda: 8)
    monkeypatch.setitem(provenance.sys.modules, "torch", None)
    before = provenance.capture_provenance(tmp_path)
    monkeypatch.setitem(
        provenance.sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(is_available=lambda: True),
        ),
    )
    after = provenance.capture_provenance(tmp_path)
    assert before["machine_fingerprint"] == after["machine_fingerprint"]
    assert after["hardware"]["machine_fingerprint_basis"] == "hardware-observation-v2"
    assert "pytorch_cuda_available" not in before["hardware"]
    assert after["hardware"]["pytorch_cuda_available"] is True
    assert before["cuda"] is None and after["cuda"] == "12.8"
    monkeypatch.setattr(provenance.os, "cpu_count", lambda: 16)
    changed_hardware = provenance.capture_provenance(tmp_path)
    assert changed_hardware["machine_fingerprint"] != after["machine_fingerprint"]
