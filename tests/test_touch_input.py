"""Public touch structure and safety tests; no native desktop API is called."""

import ctypes
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gradientclimb.capture.windows import ClientRect, WindowTarget
from gradientclimb.control.pedals import PedalAction, PedalController
from gradientclimb.control.touch import (
    CANCELED,
    DOWN,
    INCONTACT,
    INRANGE,
    POINTER_INFO,
    POINTER_TOUCH_INFO,
    TOUCH_FEEDBACK_NONE,
    UP,
    UPDATE,
    TouchPedalProfile,
    TouchPositionEvidence,
    WindowsTouchPedalBackend,
    _NativeTouchInput,
    touch_contact,
)

TARGET = WindowTarget(
    42, 100, 10.0, "Observed game", "observed.exe", ClientRect(724, 294, 2581, 1449)
)
PROFILE = TouchPedalProfile(TARGET.client_rect, (1034, 581), (925, 480), (170, 480), "a" * 64)


class Sender:
    def __init__(self):
        self.frames = []
        self.failure = False

    def inject(self, contacts):
        self.frames.append(
            [
                (
                    c.pointerInfo.pointerId,
                    c.pointerInfo.pointerFlags,
                    c.pointerInfo.ptPixelLocation.x,
                    c.pointerInfo.ptPixelLocation.y,
                )
                for c in contacts
            ]
        )
        if self.failure:
            self.failure = False
            raise OSError("mock touch failure")
        return True


def fixture():
    sender, current = Sender(), [TARGET]
    evidence = [
        TouchPositionEvidence(PROFILE.sha256, TARGET.client_rect, 1_000_000_000, True, True)
    ]
    backend = WindowsTouchPedalBackend(
        TARGET,
        PROFILE,
        lambda: evidence[0],
        guard=SimpleNamespace(validate=lambda **kwargs: current[0]),
        sender=sender,
        clock=lambda: 1.1,
    )
    return backend, sender, current, evidence


def test_structure_layout_matches_windows_x64_abi():
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(POINTER_INFO) == 96
        assert POINTER_INFO.PerformanceCount.offset == 80
        assert ctypes.sizeof(POINTER_TOUCH_INFO) == 144
        assert POINTER_TOUCH_INFO.rcContact.offset == 104


def test_physical_geometry_mapping_and_explicit_profile():
    assert PROFILE.physical_point(True) == (3033, 1491)
    assert PROFILE.physical_point(False) == (1148, 1491)
    with pytest.raises(ValueError):
        replace(PROFILE, gas_xy=(1034, 480))
    with pytest.raises(ValueError):
        replace(PROFILE, brake_xy=PROFILE.gas_xy)
    with pytest.raises(ValueError):
        replace(PROFILE, reference_sha256="")


def test_native_context_disables_only_injected_contact_feedback_without_native_calls():
    calls = []
    sender = object.__new__(_NativeTouchInput)
    sender.initialized = False
    sender.feedback_mode = TOUCH_FEEDBACK_NONE
    sender.user32 = SimpleNamespace(
        InitializeTouchInjection=lambda count, mode: calls.append((count, mode)) or True,
        InjectTouchInput=lambda count, contacts: True,
    )
    assert sender.inject([touch_contact(1, (100, 100), DOWN | INRANGE | INCONTACT)]) is True
    assert calls == [(2, 3)]
    backend, _, _, _ = fixture()
    assert backend.input_encoding["touch_feedback_mode"] == 3
    assert backend.input_encoding["touch_feedback_name"] == "none"


def test_all_four_states_and_held_contact_updates():
    backend, sender, _, _ = fixture()
    assert sender.frames == []
    backend.set_pedals(False, False)
    backend.set_pedals(True, False)
    backend.set_pedals(True, True)
    backend.set_pedals(False, True)
    backend.set_pedals(False, True)
    backend.set_pedals(False, False)
    assert [[(k, f) for k, f, _, _ in row] for row in sender.frames] == [
        [(1, DOWN | INRANGE | INCONTACT)],
        [(1, UPDATE | INRANGE | INCONTACT), (2, DOWN | INRANGE | INCONTACT)],
        [(1, UP), (2, UPDATE | INRANGE | INCONTACT)],
        [(2, UPDATE | INRANGE | INCONTACT)],
        [(2, UP)],
    ]
    assert all(row["accepted"] is True for row in backend.trace)
    backend.close()


@pytest.mark.parametrize(
    "failure",
    [
        "stale",
        "future",
        "profile",
        "positions",
        "playing",
        "geometry",
        "callback",
        "focus",
        "send",
        "action",
    ],
)
def test_failures_cancel_owned_contacts_and_latch(failure):
    backend, sender, current, evidence = fixture()
    backend.set_pedals(True, True)
    if failure == "stale":
        evidence[0] = replace(evidence[0], frame_started_ns=0)
    if failure == "future":
        evidence[0] = replace(evidence[0], frame_started_ns=2_000_000_000)
    if failure == "profile":
        evidence[0] = replace(evidence[0], profile_sha256="b" * 64)
    if failure == "positions":
        evidence[0] = replace(evidence[0], positions_verified=False)
    if failure == "playing":
        evidence[0] = replace(evidence[0], playing=False)
    if failure == "geometry":
        current[0] = replace(TARGET, client_rect=ClientRect(725, 294, 2581, 1449))
    if failure == "send":
        sender.failure = True

    def fail():
        raise OSError("mock guard failure")

    if failure == "callback":
        backend.evidence = fail
    if failure == "focus":
        backend.guard.validate = lambda **kwargs: fail()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        backend.set_pedals(1 if failure == "action" else True, True)
    assert [(k, f) for k, f, _, _ in sender.frames[-1]] == [(1, UP | CANCELED), (2, UP | CANCELED)]
    assert backend._held == set()
    with pytest.raises(RuntimeError, match="faulted"):
        backend.set_pedals(True, False)
    backend.close()


def test_watchdog_releases_two_contacts_when_fresh_gameplay_guard_fails():
    backend, sender, _, _ = fixture()
    allowed = [True]
    with PedalController(backend, lambda: allowed[0]) as controller:
        controller.submit(PedalAction(True, True, 0.1))
        allowed[0] = False
        deadline = time.perf_counter() + 0.5
        while backend._held and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert backend._held == set()
        assert [(k, f) for k, f, _, _ in sender.frames[-1]] == [(1, UP), (2, UP)]


def test_expired_controller_lease_releases_without_renewal():
    backend, _sender, _, _ = fixture()
    with PedalController(backend, lambda: True) as controller:
        controller.submit(PedalAction(True, True, 0.02))
        deadline = time.perf_counter() + 0.5
        while backend._held and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert backend._held == set()
