"""Synthetic pixel fixtures and mock Windows APIs; never capture or send live input."""

import ctypes
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest

from gradientclimb.capture.screen import WindowCapture, crop_rectangle
from gradientclimb.capture.windows import ClientRect, WindowGuard, WindowTarget, WindowUnavailable
from gradientclimb.control.windows import (
    INPUT,
    KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY,
    KEYEVENTF_KEYUP,
    WindowsPedalBackend,
)
from gradientclimb.perception.pixels import (
    ColorGeometryEstimator,
    ColorProfile,
    TemplateRecognizer,
    UITemplate,
    geometry_validation_metrics,
    state_validation_metrics,
)
from gradientclimb.perception.states import StateEvidence, UIState


def patch(seed=1):
    return np.random.default_rng(seed).integers(0, 256, (8, 12, 3), dtype=np.uint8)


def scene():
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[5:13, 5:17] = patch()
    return image


def template(state=UIState.PLAYING, label="playing", role="state"):
    return UITemplate(label, state, patch(), (0, 0, 0.4, 0.4), role=role)


def test_template_requires_distinct_state_evidence_and_exact_geometry():
    recognizer = TemplateRecognizer([template()], expected_size=(100, 60))
    assert recognizer.classify(scene()).state == UIState.PLAYING
    assert recognizer.classify(np.zeros((60, 100, 3), dtype=np.uint8)).state == UIState.UNEXPECTED
    assert recognizer.classify(np.zeros((120, 200, 3), dtype=np.uint8)).state == UIState.UNEXPECTED
    empty = TemplateRecognizer([], expected_size=(100, 60))
    assert empty.classify(scene()).state == UIState.UNEXPECTED


def test_ambiguous_templates_fail_closed():
    recognizer = TemplateRecognizer(
        [template(), template(UIState.PAUSED, "paused")], expected_size=(100, 60)
    )
    result = recognizer.classify(scene())
    assert result == StateEvidence(UIState.UNEXPECTED, 0.0)


def test_advertisement_requires_separate_close_evidence():
    ad = template(UIState.ADVERTISEMENT, "ad")
    recognizer = TemplateRecognizer([ad], expected_size=(100, 60))
    evidence = recognizer.classify(scene())
    assert evidence.state == UIState.ADVERTISEMENT
    assert not evidence.legitimate_close_visible
    close = UITemplate(
        "close", UIState.ADVERTISEMENT, patch(2), (0.7, 0, 1, 0.4), role="legitimate_ad_close"
    )
    with_close = TemplateRecognizer([ad, close], expected_size=(100, 60))
    assert not with_close.classify(scene()).legitimate_close_visible
    image = scene()
    image[5:13, 80:92] = patch(2)
    assert with_close.classify(image).legitimate_close_visible


def test_synthetic_color_geometry_and_metrics_count_missing_frames():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[30:38, 35:75] = [255, 0, 0]
    image[0:8, 0:110] = [255, 0, 0]  # HUD-colored distractor excluded by vehicle ROI.
    for x in range(120):
        image[55 + x // 20 :, x] = [0, 255, 0]
    profile = ColorProfile(
        (120, 80),
        (0, 0.15, 1, 0.6),
        (((0, 200, 200), (5, 255, 255)),),
        (0, 0.5, 1, 1),
        (((50, 200, 200), (70, 255, 255)),),
    )
    estimator = ColorGeometryEstimator(profile)
    vehicle = estimator.vehicle(image)
    assert vehicle.valid
    np.testing.assert_allclose(vehicle.center_xy, (54.5, 33.5))
    assert abs(vehicle.pitch_mod_pi) < 1e-8
    terrain = estimator.terrain(image, samples=12)
    assert terrain.coverage == 1.0
    np.testing.assert_allclose(terrain.y_pixels, 55 + terrain.x_pixels // 20)
    missing = estimator.vehicle(np.zeros_like(image))
    metrics = geometry_validation_metrics(
        [vehicle, missing], [{"center_xy": [54.5, 33.5], "pitch_mod_pi": 0}] * 2
    )
    assert metrics["coverage"] == 0.5
    assert metrics["center_error_pixels_mean"] == 0
    states = state_validation_metrics([StateEvidence(UIState.UNEXPECTED, 0)], [UIState.PLAYING])
    assert states["accuracy"] == 0 and states["unknown_fraction"] == 1


TARGET = WindowTarget(
    42, 100, 12.5, "Hill Climb Racing", r"C:\PlayGames\crosvm.exe", ClientRect(20, 30, 100, 60)
)


class FakeWindows:
    target = TARGET
    focused = 42

    def describe(self, hwnd):
        return self.target

    def foreground(self):
        return self.focused

    def physical_pixels(self):
        return nullcontext()


class FakeSender:
    def __init__(self):
        self.events = []
        self.fail_next = False

    def is_down(self, key):
        return False

    def send(self, events):
        self.events.extend((int(e.ki.wVk), int(e.ki.dwFlags)) for e in events)
        if self.fail_next:
            self.fail_next = False
            return 0
        return len(events)


def test_windows_input_struct_layout_and_independent_transition_order():
    assert ctypes.sizeof(INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
    assert ctypes.sizeof(KEYBDINPUT) == (24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16)
    api, sender = FakeWindows(), FakeSender()
    backend = WindowsPedalBackend(
        TARGET, gas_vk=0x27, brake_vk=0x25, guard=WindowGuard(TARGET, api=api), sender=sender
    )
    backend.set_pedals(True, False)
    backend.set_pedals(True, True)
    backend.set_pedals(True, True)
    backend.set_pedals(False, True)
    backend.set_pedals(False, False)
    assert sender.events == [
        (0x27, KEYEVENTF_EXTENDEDKEY),
        (0x25, KEYEVENTF_EXTENDEDKEY),
        (0x27, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP),
        (0x25, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP),
    ]


def test_focus_loss_releases_owned_keys_and_refuses_new_down():
    api, sender = FakeWindows(), FakeSender()
    backend = WindowsPedalBackend(
        TARGET, gas_vk=0x27, brake_vk=0x25, guard=WindowGuard(TARGET, api=api), sender=sender
    )
    backend.set_pedals(True, False)
    api.focused = 99
    with pytest.raises(WindowUnavailable):
        backend.set_pedals(True, True)
    assert sender.events[-1] == (0x27, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP)
    assert len(sender.events) == 2


def test_reused_handle_or_restarted_process_is_refused():
    api = FakeWindows()
    api.target = replace(TARGET, process_created_at=99)
    with pytest.raises(WindowUnavailable, match="identity changed"):
        WindowGuard(TARGET, api=api).validate()


def test_partial_send_releases_all_possible_keys_and_disables_backend():
    api, sender = FakeWindows(), FakeSender()
    sender.fail_next = True
    backend = WindowsPedalBackend(
        TARGET, gas_vk=0x27, brake_vk=0x25, guard=WindowGuard(TARGET, api=api), sender=sender
    )
    with pytest.raises(RuntimeError, match="only part"):
        backend.set_pedals(True, True)
    assert sender.events[-2:] == [
        (0x25, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP),
        (0x27, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP),
    ]
    with pytest.raises(RuntimeError, match="faulted"):
        backend.set_pedals(True, False)


def test_crop_and_capture_use_verified_geometry_without_live_capture(monkeypatch, tmp_path):
    api = FakeWindows()
    capture = WindowCapture(
        WindowGuard(TARGET, api=api),
        backend="pillow",
        crop=(0.1, 0.2, 0.9, 0.8),
        output_size=(20, 10),
    )
    assert crop_rectangle(TARGET.client_rect, capture.crop) == ClientRect(30, 42, 80, 36)
    monkeypatch.setattr(
        capture, "_read", lambda rect: np.zeros((rect.height, rect.width, 3), dtype=np.uint8)
    )
    frame = capture.grab()
    assert frame.rgb.shape == (10, 20, 3)
    assert frame.started_ns <= frame.timestamp_ns <= frame.completed_ns
    result = capture.benchmark(frame_count=3, output_dir=tmp_path / "capture", max_record_bytes=1)
    assert result["captured_frames"] == 3
    assert result["recording_limit_hit"] is True
    assert result["recorded_frames"] == 0
    assert result["source_dropped_frames"] is None
    assert (tmp_path / "capture" / "frames.jsonl").read_text().count("\n") == 3
