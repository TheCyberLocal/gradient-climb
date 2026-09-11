import time

import pytest

from gradientclimb.control import PedalAction, PedalController
from gradientclimb.perception.states import StateEvidence, UIState, permitted_action


class MemoryBackend:
    def __init__(self):
        self.events = []

    def set_pedals(self, gas, brake):
        self.events.append((gas, brake))


def test_joint_states_and_order():
    assert [PedalAction.from_code(i).code for i in range(4)] == [0, 1, 2, 3]
    backend = MemoryBackend()
    with PedalController(backend, lambda: True) as controller:
        for code in [1, 3, 0, 2, 3]:
            controller.submit(PedalAction.from_code(code))
        assert [e["gas"] for e in controller.trace] == [True, True, False, False, True]
        assert backend.events[1:6] == [
            (True, False),
            (True, True),
            (False, False),
            (False, True),
            (True, True),
        ]
    assert backend.events[-1] == (False, False)


def test_expired_lease_releases_and_unknown_refuses():
    backend = MemoryBackend()
    with PedalController(backend, lambda: True) as controller:
        controller.submit(PedalAction(True, True, 0.02))
        time.sleep(0.08)
        assert backend.events[-1] == (False, False)
    with PedalController(backend, lambda: False) as controller, pytest.raises(RuntimeError):
        controller.submit(PedalAction(True, True, 0.1))
    assert backend.events[-1] == (False, False)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), 3])
def test_invalid_duration(duration):
    with pytest.raises(ValueError):
        PedalAction(True, False, duration)


def test_advertisement_and_unknown_do_not_click():
    assert permitted_action(StateEvidence(UIState.ADVERTISEMENT, 1)) == "release_and_wait"
    assert permitted_action(StateEvidence(UIState.UNEXPECTED, 1)) == "release_and_halt"
    assert permitted_action(StateEvidence(UIState.PLAYING, 0.5)) == "release_and_halt"
    assert (
        permitted_action(StateEvidence(UIState.RESULT, 1, restart_visible=True))
        == "verified_restart"
    )


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_confidence_never_permits_policy(confidence):
    with pytest.raises(ValueError):
        StateEvidence(UIState.PLAYING, confidence)


def test_watchdog_latches_failed_release():
    class FailingRelease(MemoryBackend):
        def set_pedals(self, gas, brake):
            if self.events and not gas and not brake:
                raise OSError("Input delivery failed")
            super().set_pedals(gas, brake)

    controller = PedalController(FailingRelease(), lambda: True)
    controller.submit(PedalAction(True, True, 0.02))
    time.sleep(0.08)
    assert controller._stop.is_set()
    assert controller.fault is not None and controller.release_failure is not None
    with pytest.raises(RuntimeError):
        controller.submit(PedalAction(True, False, 0.1))
    with pytest.raises(OSError):
        controller.close()


def test_state_callback_exception_latches_input():
    def broken_classifier():
        raise ValueError("No valid pixels")

    controller = PedalController(MemoryBackend(), broken_classifier)
    with pytest.raises(ValueError):
        controller.submit(PedalAction(True, False, 0.1))
    assert controller._stop.is_set()
    assert controller.backend.events[-1] == (False, False)
    controller.close()
