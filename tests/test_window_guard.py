"""Window guard evidence for foreground loss; no real window handles are used."""

from types import SimpleNamespace

import pytest

from gradientclimb.capture.windows import ClientRect, WindowGuard, WindowTarget, WindowUnavailable

TARGET = WindowTarget(42, 7, 1.0, "Observed game", "observed.exe", ClientRect(100, 200, 800, 600))


def test_foreground_loss_names_the_window_that_took_focus():
    api = SimpleNamespace(
        describe=lambda hwnd: TARGET,
        foreground=lambda: 99,
        foreground_summary=lambda: {"hwnd": 99, "title": "Store page", "executable": "chrome.exe"},
    )
    guard = WindowGuard(TARGET, api=api)
    with pytest.raises(WindowUnavailable, match=r"foreground is 'Store page' \(chrome.exe\)"):
        guard.validate()
    assert guard.validate(require_foreground=False) == TARGET


def test_foreground_evidence_failure_never_masks_the_fault():
    def boom():
        raise OSError("no summary")

    api = SimpleNamespace(
        describe=lambda hwnd: TARGET, foreground=lambda: 99, foreground_summary=boom
    )
    with pytest.raises(WindowUnavailable, match="foreground unknown"):
        WindowGuard(TARGET, api=api).validate()
