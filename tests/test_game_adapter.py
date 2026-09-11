"""Functional reset sequences with synthetic references and mocked native I/O."""

import hashlib
import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gradientclimb.capture.screen import CapturedFrame
from gradientclimb.capture.windows import ClientRect, WindowTarget
from gradientclimb.control.game_adapter import (
    GameUIProfile,
    GameUIRecognizer,
    NativeGameAdapter,
    _NativeMenuInput,
    mouse_click_events,
)

TARGET = WindowTarget(42, 7, 1.0, "Observed game", "observed.exe", ClientRect(100, 200, 800, 600))
ROLES = {
    "playing": "pause_episode",
    "paused": "restart_paused",
    "revive_offer": "decline_revive_offer",
    "result": "continue_result",
    "bonus_offer": "decline_bonus_offer",
    "tune": "start_episode",
    "advertisement": "legitimate_ad_close",
}


@pytest.fixture
def dataset(tmp_path):
    frames, variants = {}, []
    rng = np.random.default_rng(7)
    for state, action in ROLES.items():
        rgb = np.full((60, 80, 3), 20, dtype=np.uint8)
        for left, top, right, bottom in [(2, 2, 14, 14), (18, 2, 30, 14), (5, 25, 25, 45)]:
            rgb[top:bottom, left:right] = rng.integers(
                0, 256, (bottom - top, right - left, 3), dtype=np.uint8
            )
        path = tmp_path / f"{state}.png"
        Image.fromarray(rgb).save(path)
        frames[state] = rgb
        variants.append(
            {
                "label": state,
                "state": state,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "anchors": [[2, 2, 14, 14], [18, 2, 30, 14]],
                "controls": {action: [5, 25, 25, 45]},
            }
        )
    frames["unknown"] = np.zeros((60, 80, 3), dtype=np.uint8)
    profile = {
        "version": 1,
        "profile_id": "synthetic",
        "expected_size": [80, 60],
        "margin_pixels": 0,
        "threshold": 0.97,
        "variants": variants,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return tmp_path, path, frames, profile


class Clock:
    now = 1.0

    def __call__(self):
        return self.now

    def sleep(self, value):
        self.now += value


def adapter_fixture(dataset, sequence):
    root, path, frames, _ = dataset
    clock, current, released, clicks = Clock(), [TARGET], [], []

    class Capture:
        index = 0

        def grab(self):
            clock.now += 0.02
            state = sequence[min(self.index, len(sequence) - 1)]
            self.index += 1
            if state == "capture_failure":
                raise OSError("mock capture failure")
            return CapturedFrame(
                frames[state].copy(),
                int((clock.now - 0.01) * 1e9),
                int(clock.now * 1e9),
                "2026-09-11T00:00:00Z",
                TARGET.client_rect,
                "mock",
                10.0,
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    sender = SimpleNamespace(is_down=lambda key: False, click=lambda point: clicks.append(point))
    adapter = NativeGameAdapter(
        TARGET,
        path,
        root,
        guard=SimpleNamespace(validate=lambda **kwargs: current[0]),
        capture=Capture(),
        sender=sender,
        release_pedals=lambda: released.append(clock()),
        clock=clock,
        sleep=clock.sleep,
    )
    return adapter, clock, current, released, clicks


@pytest.mark.parametrize(
    "sequence,names",
    [
        (["playing", "paused", "playing"], ["pause_episode", "restart_paused"]),
        (["paused", "paused", "unknown", "playing"], ["restart_paused"]),
        (
            ["revive_offer", "result", "tune", "playing"],
            ["decline_revive_offer", "continue_result", "start_episode"],
        ),
        (
            ["result", "bonus_offer", "result", "tune", "playing"],
            ["continue_result", "decline_bonus_offer", "continue_result", "start_episode"],
        ),
    ],
)
def test_bounded_reset_sequences_have_only_verified_clicks(dataset, sequence, names):
    adapter, _, _, released, clicks = adapter_fixture(dataset, sequence)
    results, observed = [], []
    outcome = adapter.reset(
        truncate=sequence[0] == "playing",
        on_terminal=results.append,
        on_observation=observed.append,
    )
    assert outcome.state == "playing"
    assert [row["name"] for row in adapter.trace] == names
    assert len(clicks) == len(names)
    assert all(row["accepted"] for row in adapter.trace)
    assert len(results) == int("result" in sequence)
    assert released
    assert observed[-1].state == "playing"


@pytest.mark.parametrize(
    "sequence,expected",
    [
        (["unknown"], "Unrecognized"),
        (["tune"], "Unrecognized"),
        (["paused", "paused"], "timed-out"),
        (["result"], "Terminal-frame"),
        (["capture_failure"], "mock capture"),
    ],
)
def test_unknown_incomplete_or_failed_flow_stops_and_releases(dataset, sequence, expected):
    adapter, _, _, released, clicks = adapter_fixture(dataset, sequence)
    with pytest.raises((RuntimeError, OSError), match=expected):
        adapter.reset(transition_seconds=0.2)
    assert adapter._stopped.is_set() and released
    assert len(clicks) <= 1


@pytest.mark.parametrize(
    "failure",
    ["focus", "geometry", "escape", "stale", "future", "forged", "missing_control", "sender"],
)
def test_no_unverified_click_and_fault_is_latched(dataset, failure):
    adapter, clock, current, released, clicks = adapter_fixture(dataset, ["paused"])
    observed = adapter.observe()
    if failure == "focus":

        def fail(**kwargs):
            raise RuntimeError("mock focus loss")

        adapter.guard.validate = fail
    if failure == "geometry":
        current[0] = replace(TARGET, client_rect=ClientRect(101, 200, 800, 600))
    if failure == "escape":
        adapter.sender.is_down = lambda key: True
    if failure == "stale":
        clock.now += 1
    if failure == "future":
        observed = replace(
            observed, frame=replace(observed.frame, started_ns=int((clock.now + 1) * 1e9))
        )
    if failure == "forged":
        observed = replace(observed, state="result")
    if failure == "missing_control":
        observed.frame.rgb[25:45, 5:25] = 0
    if failure == "sender":

        def fail(point):
            raise OSError("mock sender failure")

        adapter.sender.click = fail
    with pytest.raises((RuntimeError, OSError)):
        adapter.click_verified("restart_paused", observed)
    assert not clicks and released and adapter._stopped.is_set()


def test_repeated_frame_click_and_purchase_names_are_refused(dataset):
    adapter, _, _, _, clicks = adapter_fixture(dataset, ["paused"])
    observed = adapter.observe()
    adapter.click_verified("restart_paused", observed)
    with pytest.raises(RuntimeError, match="already clicked"):
        adapter.click_verified("restart_paused", observed)
    assert len(clicks) == 1
    for name in ("buy_upgrade", "get_revived", "get_reward", "legitimate_ad_close", "exit"):
        adapter, _, _, _, clicks = adapter_fixture(dataset, ["paused"])
        with pytest.raises(RuntimeError):
            adapter.click_verified(name, adapter.observe())
        assert not clicks


def test_terminal_callback_failure_preserves_result_before_any_click(dataset):
    adapter, _, _, released, clicks = adapter_fixture(dataset, ["result"])

    def fail(observation):
        raise ValueError("save failed")

    with pytest.raises(ValueError, match="save failed"):
        adapter.reset(on_terminal=fail)
    assert released and not clicks


def test_reference_hash_and_role_allowlist_enforced(dataset):
    root, _, _, profile = dataset
    profile["variants"][0]["controls"] = {"buy_upgrade": [5, 25, 25, 45]}
    with pytest.raises(ValueError, match="not permitted"):
        GameUIProfile.model_validate(profile)
    profile["variants"][0]["controls"] = {"pause_episode": [5, 25, 25, 45]}
    profile["variants"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        GameUIRecognizer(GameUIProfile.model_validate(profile), root)


def test_mouse_events_use_physical_virtual_desktop_coordinates():
    events = mouse_click_events((0, 0), ClientRect(-1920, 0, 3840, 1080))
    assert [event.mi.dwFlags for event in events] == [0xC001, 2, 4]
    assert events[0].mi.dx == round(1920 * 65535 / 3839)
    assert events[0].mi.dy == 0
    with pytest.raises(ValueError):
        mouse_click_events((1920, 0), ClientRect(-1920, 0, 3840, 1080))


@pytest.mark.parametrize("backend", ["dxcam", "mss", "pillow"])
def test_explicit_capture_backend_constructs_only_selected_reader(dataset, backend, monkeypatch):
    root, path, _, _ = dataset
    created = []

    def capture(guard, **kwargs):
        created.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("gradientclimb.control.game_adapter.WindowCapture", capture)
    adapter = NativeGameAdapter(
        TARGET,
        path,
        root,
        guard=SimpleNamespace(),
        sender=SimpleNamespace(),
        capture_backend=backend,
        release_pedals=lambda: None,
    )
    assert adapter.capture_backend_name == backend
    assert created == [{"backend": backend, "output_size": (80, 60)}]


def test_capture_backend_has_no_automatic_fallback(dataset, monkeypatch):
    root, path, _, _ = dataset
    attempts = []

    def failing_capture(guard, **kwargs):
        attempts.append(kwargs["backend"])
        raise RuntimeError("chosen capture unavailable")

    monkeypatch.setattr("gradientclimb.control.game_adapter.WindowCapture", failing_capture)
    with pytest.raises(RuntimeError, match="chosen capture unavailable"):
        NativeGameAdapter(
            TARGET,
            path,
            root,
            guard=SimpleNamespace(),
            sender=SimpleNamespace(),
            capture_backend="mss",
            release_pedals=lambda: None,
        )
    assert attempts == ["mss"]
    with pytest.raises(ValueError, match="Explicit capture backend"):
        NativeGameAdapter(TARGET, path, root, capture_backend="automatic")


def test_stale_watchdog_releases_without_permanent_stop_and_recovers(dataset):
    adapter, clock, _, _, _ = adapter_fixture(dataset, ["playing"])
    adapter.observe()
    assert adapter.is_playing()
    clock.now += 0.5
    assert not adapter.is_playing()
    assert not adapter._stopped.is_set()
    before = len(adapter.guard_trace)
    assert not adapter.is_playing()
    assert len(adapter.guard_trace) == before
    adapter.observe()
    assert adapter.is_playing()
    assert any("maximum observation age" in row["reason"] for row in adapter.guard_trace)


def test_cold_stale_capture_is_discarded_then_fresh_frame_required(dataset):
    adapter, clock, _, released, _ = adapter_fixture(dataset, ["playing"])
    grab = adapter.capture_backend.grab
    calls = []

    def cold_grab():
        frame = grab()
        calls.append(frame)
        if len(calls) == 1:
            clock.sleep(0.6)
        return frame

    adapter.capture_backend.grab = cold_grab
    observed = adapter.observe()
    assert len(calls) == 2 and observed.frame is calls[1]
    assert adapter.capture_trace[0]["discarded"] and released
    assert not adapter._stopped.is_set() and adapter.is_playing()


def test_stale_capture_retry_is_bounded_and_does_not_return_old_pixels(dataset):
    adapter, clock, _, released, _ = adapter_fixture(dataset, ["playing"])
    grab = adapter.capture_backend.grab

    def stale_grab():
        frame = grab()
        clock.sleep(0.6)
        return frame

    adapter.capture_backend.grab = stale_grab
    with pytest.raises(RuntimeError, match="maximum observation age"):
        adapter.observe()
    assert len(adapter.capture_trace) == 2 and adapter.latest is None
    assert not adapter._stopped.is_set() and released


def test_watchdog_focus_fault_remains_latched(dataset):
    adapter, _, current, _, _ = adapter_fixture(dataset, ["playing"])
    adapter.observe()
    current[0] = replace(TARGET, client_rect=ClientRect(0, 0, 800, 600))
    assert not adapter.is_playing()
    current[0] = TARGET
    assert not adapter.is_playing()
    assert adapter._stopped.is_set()
    assert any(row["latched"] for row in adapter.guard_trace)


@pytest.mark.parametrize(
    "sequence,names,expected",
    [
        (["playing", "paused"], ["pause_episode"], "paused"),
        (["revive_offer", "result", "tune"], ["decline_revive_offer", "continue_result"], "tune"),
    ],
)
def test_stationary_handoff_does_not_start_next_episode(dataset, sequence, names, expected):
    adapter, _, _, _, _ = adapter_fixture(dataset, sequence)
    observation = adapter.reset(
        truncate=True, start_next=False, on_terminal=lambda observation: None
    )
    assert observation.state == expected
    assert [row["name"] for row in adapter.trace] == names


def test_explicit_initial_start_and_next_restart(dataset):
    for first in ("tune", "paused"):
        adapter, _, _, _, _ = adapter_fixture(dataset, [first, "playing"])
        assert adapter.reset(allow_initial_start=True).state == "playing"
        assert len(adapter.trace) == 1


def test_known_ad_waits_for_own_close_then_handles_second_phase(dataset):
    root, path, frames, profile = dataset
    frames["ad_wait"] = frames["advertisement"].copy()
    frames["ad_wait"][25:45, 5:25] = 0
    frames["second_ad"] = np.roll(frames["advertisement"], 1, axis=2)
    second = root / "second_ad.png"
    Image.fromarray(frames["second_ad"]).save(second)
    profile["variants"].append(
        {
            **profile["variants"][-1],
            "label": "second_ad",
            "file": second.name,
            "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
        }
    )
    path.write_text(json.dumps(profile))
    sequence = [
        "result",
        *(["ad_wait"] * 620),
        "advertisement",
        "unknown",
        "second_ad",
        "tune",
        "playing",
    ]
    adapter, clock, _, _, _ = adapter_fixture(dataset, sequence)
    assert adapter.reset(on_terminal=lambda observation: None).state == "playing"
    assert clock.now > 30
    assert [row["name"] for row in adapter.trace] == [
        "continue_result",
        "legitimate_ad_close",
        "legitimate_ad_close",
        "start_episode",
    ]


def test_ad_unknown_creative_waits_bounded_without_generic_close(dataset):
    _, _, frames, _ = dataset
    frames["unknown_creative"] = np.full((60, 80, 3), 220, dtype=np.uint8)
    adapter, _, _, _, clicks = adapter_fixture(dataset, ["advertisement", "unknown_creative"])
    with pytest.raises((RuntimeError, TimeoutError)):
        adapter.reset(allow_initial_start=True, max_seconds=0.2)
    assert len(clicks) == 1


def test_result_continue_unknown_video_wait_then_only_registered_close(dataset):
    _, _, frames, _ = dataset
    frames["unknown_video"] = np.full((60, 80, 3), 220, dtype=np.uint8)
    sequence = ["result", *(["unknown_video"] * 410), "advertisement", "unknown_video", "tune"]
    adapter, clock, _, _, _ = adapter_fixture(dataset, sequence)
    assert adapter.reset(start_next=False, on_terminal=lambda observation: True).state == "tune"
    assert clock.now > 20
    assert [row["name"] for row in adapter.trace] == ["continue_result", "legitimate_ad_close"]


def test_known_ad_without_close_times_out_without_click(dataset):
    _, _, frames, _ = dataset
    frames["ad_wait"] = frames["advertisement"].copy()
    frames["ad_wait"][25:45, 5:25] = 0
    adapter, _, _, released, clicks = adapter_fixture(dataset, ["ad_wait"])
    with pytest.raises(TimeoutError):
        adapter.reset(max_seconds=0.2, transition_seconds=0.1)
    assert released and not clicks


def test_slow_callback_cannot_send_after_reset_budget(dataset):
    adapter, clock, _, _, clicks = adapter_fixture(dataset, ["paused"])
    with pytest.raises(TimeoutError):
        adapter.reset(
            max_seconds=0.2,
            transition_seconds=0.1,
            on_observation=lambda observation: clock.sleep(1),
        )
    assert not clicks


def test_partial_native_click_releases_mouse_and_records_both_attempts():
    sender = _NativeMenuInput.__new__(_NativeMenuInput)
    packets = []

    def send(events):
        packets.append([event.mi.dwFlags for event in events])
        return 2 if len(packets) == 1 else 1

    sender.api = SimpleNamespace(
        is_down=lambda key: False,
        send=send,
        user32=SimpleNamespace(
            GetSystemMetrics=lambda index: {76: 0, 77: 0, 78: 1920, 79: 1080}[index]
        ),
    )
    sender.guard = SimpleNamespace(api=SimpleNamespace(physical_pixels=nullcontext))
    sender.trace = []
    with pytest.raises(RuntimeError, match="complete menu click"):
        sender.click((200, 300))
    assert packets == [[0xC001, 2, 4], [4]]
    assert sender.trace[0]["inserted"] == 2 and sender.trace[0]["cleanup_inserted"] == 1
    sender.api.is_down = lambda key: True
    with pytest.raises(RuntimeError, match="already held"):
        sender.click((200, 300))
    assert len(packets) == 2


def test_pointer_parking_is_a_separate_move_only_packet_and_trace():
    sender = _NativeMenuInput.__new__(_NativeMenuInput)
    packets = []

    def send(events):
        packets.append([event.mi.dwFlags for event in events])
        return len(events)

    sender.api = SimpleNamespace(
        is_down=lambda key: False,
        send=send,
        user32=SimpleNamespace(
            GetSystemMetrics=lambda index: {76: 0, 77: 0, 78: 1920, 79: 1080}[index]
        ),
    )
    sender.guard = SimpleNamespace(api=SimpleNamespace(physical_pixels=nullcontext))
    sender.trace = []
    sender.click((200, 300))
    sender.park_pointer((520, 18))
    assert packets == [[0xC001, 2, 4], [0xC001]]
    assert [row["kind"] for row in sender.trace] == ["menu_click", "pointer_park"]


def test_pointer_parking_rechecks_focus_after_successful_menu_click(dataset):
    adapter, _, current, _, clicks = adapter_fixture(dataset, ["paused"])
    adapter.pointer_park_xy = (40, 2)
    parked = []
    adapter.sender.park_pointer = parked.append

    def click(point):
        clicks.append(point)
        current[0] = replace(TARGET, client_rect=ClientRect(101, 200, 800, 600))

    adapter.sender.click = click
    with pytest.raises(RuntimeError, match="geometry"):
        adapter.click_verified("restart_paused", adapter.observe())
    assert len(clicks) == 1 and not parked


def test_result_callback_can_require_fresh_confirmation_before_dismissal(dataset):
    adapter, _, _, _, clicks = adapter_fixture(dataset, ["result", "result", "tune"])
    observations = []

    def confirm(observation):
        observations.append(observation.frame.timestamp_ns)
        assert not clicks
        return len(observations) >= 2

    assert adapter.reset(start_next=False, on_terminal=confirm).state == "tune"
    assert len(observations) == 2 and observations[0] < observations[1]
    assert [row["name"] for row in adapter.trace] == ["continue_result"]


def test_result_without_visible_continue_waits_without_repeating_terminal_callback(dataset):
    _, _, frames, _ = dataset
    frames["result_blink"] = frames["result"].copy()
    frames["result_blink"][25:45, 5:25] = 0
    adapter, _, _, _, clicks = adapter_fixture(dataset, ["result_blink", "result", "tune"])
    results = []
    assert adapter.reset(start_next=False, on_terminal=results.append).state == "tune"
    assert len(results) == 1 and len(clicks) == 1


def test_unconfirmed_result_callback_is_bounded_without_click(dataset):
    adapter, _, _, released, clicks = adapter_fixture(dataset, ["result"])
    with pytest.raises(TimeoutError):
        adapter.reset(max_seconds=0.2, on_terminal=lambda observation: False)
    assert not clicks and released


def test_known_result_transient_unknown_waits_without_click_before_recovery(dataset):
    adapter, _, _, _, clicks = adapter_fixture(dataset, ["result", "unknown", "result", "tune"])
    accepted = []

    def confirm(observation):
        accepted.append(observation.frame.timestamp_ns)
        assert not clicks
        return len(accepted) == 2

    assert adapter.reset(start_next=False, on_terminal=confirm).state == "tune"
    assert len(accepted) == 2 and len(clicks) == 1


def test_result_outlined_glyphs_ignore_background_but_require_ink_and_outline(dataset):
    root, path, frames, profile = dataset
    variant = next(v for v in profile["variants"] if v["state"] == "result")
    rgb = np.full((60, 80, 3), 90, dtype=np.uint8)
    boxes = [*variant["anchors"], *variant["controls"].values()]
    for index, (left, top, right, bottom) in enumerate(boxes):
        # Synthetic distinctive outlined white glyph: shape varies by region.
        x = left + 2 + index
        rgb[top + 1 : top + 11, x : x + 6] = 0
        rgb[top + 2 : top + 10, x + 1 : x + 5] = 255
    file = root / variant["file"]
    Image.fromarray(rgb).save(file)
    variant["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    variant["matching"] = "outlined_white"
    path.write_text(json.dumps(profile))
    frames["result_changed_background"] = rgb.copy()
    background = np.all(rgb == 90, axis=2)
    frames["result_changed_background"][background] = [130, 180, 220]
    adapter, _, _, _, _ = adapter_fixture(dataset, ["result_changed_background"])
    observed = adapter.observe()
    assert observed.state == "result" and observed.controls[0].name == "continue_result"
    # Blank white/black areas and missing glyph cores cannot impersonate text.
    for fill in (0, 255):
        altered = observed.frame.rgb.copy()
        for left, top, right, bottom in boxes:
            altered[top:bottom, left:right] = fill
        checked = adapter.recognizer.observe(replace(observed.frame, rgb=altered))
        assert checked.state == "unknown" and not checked.controls
