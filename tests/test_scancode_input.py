"""Keyboard encoding and release safety with no native input or API calls."""

from types import SimpleNamespace

import pytest

from gradientclimb.control.windows import WindowsPedalBackend, keyboard_event


class Sender:
    def __init__(self):
        self.events = []
        self.failure = None

    def scan_code(self, key):
        return {0x27: 0xE04D, 0x25: 0xE04B}[key]

    def is_down(self, key):
        return False

    def send(self, events):
        self.events.extend((e.ki.wVk, e.ki.wScan, e.ki.dwFlags) for e in events)
        failure, self.failure = self.failure, None
        if failure == "exception":
            raise OSError("mock send failure")
        return 0 if failure == "partial" else len(events)


@pytest.mark.parametrize("key,scan", [(0x27, 0xE04D), (0x25, 0xE04B), (0x27, 0x4D), (0x25, 0x4B)])
def test_extended_arrow_down_and_up_encoding(key, scan):
    down, up = [keyboard_event(key, pressed, scan_code=scan) for pressed in (True, False)]
    assert (down.ki.wVk, down.ki.wScan, down.ki.dwFlags) == (0, scan & 0xFF, 0x09)
    assert (up.ki.wVk, up.ki.wScan, up.ki.dwFlags) == (0, scan & 0xFF, 0x0B)


@pytest.mark.parametrize("scan", [0, 0xE11D, 0xE000, 0xE24D, -1, True, 0x10000])
def test_unmapped_or_unsupported_scan_code_refused(scan):
    with pytest.raises(ValueError):
        keyboard_event(0x27, True, scan_code=scan)


def test_nonextended_scan_code_does_not_get_arrow_flag():
    event = keyboard_event(0x41, True, scan_code=0x1E)
    assert (event.ki.wVk, event.ki.wScan, event.ki.dwFlags) == (0, 0x1E, 8)


def test_independent_scancode_transitions_and_mode_trace():
    sender = Sender()
    backend = WindowsPedalBackend(
        None,
        gas_vk=0x27,
        brake_vk=0x25,
        input_mode="scancode",
        sender=sender,
        guard=SimpleNamespace(validate=lambda **kwargs: None),
    )
    assert sender.events == []
    backend.set_pedals(True, False)
    backend.set_pedals(True, True)
    backend.set_pedals(True, True)
    backend.set_pedals(False, True)
    backend.close()
    assert sender.events == [(0, 0x4D, 9), (0, 0x4B, 9), (0, 0x4D, 11), (0, 0x4B, 11)]
    assert backend.input_encoding["scan_codes"] == {"39": 0xE04D, "37": 0xE04B}
    assert all(row["input_mode"] == "scancode" for row in backend.trace)
    assert backend.trace[0]["encoded_events"] == [{"wVk": 0, "wScan": 0x4D, "dwFlags": 9}]


@pytest.mark.parametrize("failure", ["partial", "exception", "focus"])
def test_scancode_failures_release_owned_keys_and_preserve_trace(failure):
    sender, focused = Sender(), [True]

    def validate(**kwargs):
        if not focused[0]:
            raise RuntimeError("mock focus loss")

    backend = WindowsPedalBackend(
        None,
        gas_vk=0x27,
        brake_vk=0x25,
        input_mode="scancode",
        sender=sender,
        guard=SimpleNamespace(validate=validate),
    )
    backend.set_pedals(True, False)
    if failure == "focus":
        focused[0] = False
    else:
        sender.failure = failure
    with pytest.raises((RuntimeError, OSError)):
        backend.set_pedals(True, True)
    assert sender.events[-1] == (0, 0x4D, 11)
    assert backend._held == set()
    if failure == "exception":
        assert backend.trace[-2]["delivered"] is None
        assert "mock send failure" in backend.trace[-2]["error"]
    elif failure == "partial":
        assert backend.trace[-2]["delivered"] == 0
    backend.close()


def test_mapping_failure_sends_nothing():
    sender = Sender()
    sender.scan_code = lambda key: 0
    with pytest.raises(ValueError):
        WindowsPedalBackend(
            None,
            gas_vk=39,
            brake_vk=37,
            input_mode="scancode",
            sender=sender,
            guard=SimpleNamespace(validate=lambda **kwargs: None),
        )
    assert sender.events == []
