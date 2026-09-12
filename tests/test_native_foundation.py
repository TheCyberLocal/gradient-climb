"""Native foundation failures exercised without game capture or gameplay input."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_screen_episodes import episode_module, fixture

from gradientclimb.control.host import HostBudgetGuard, HostLimits, NativeInputLease
from gradientclimb.control.native_protocol import (
    longest_scored_streak,
    validate_limits,
    validate_protocol,
)
from gradientclimb.control.pedals import PedalAction, PedalController


def test_unscored_breaks_scored_success_streak():
    outcomes = [
        {"classification": c}
        for c in (
            ["success_natural_scored"] * 4
            + ["success_unscored"]
            + ["success_truncated_scored"] * 6
            + ["not_attempted"]
        )
    ]
    assert longest_scored_streak(outcomes) == 6


def test_session_deadline_is_not_full_episode_horizon():
    module = episode_module()
    adapter, controller, backend, bridge, first, sent, released = fixture(module)
    summary, _, _ = module.collect_episode(
        adapter, controller, backend, bridge, lambda _: 1, first, seconds=60, deadline=1.12
    )
    assert sent and released
    assert summary["reason"] == "session_deadline"
    assert module.terminal_cause(summary) == "session_deadline"
    assert (
        module.classify_attempt(summary, SimpleNamespace(state="paused"))
        == "administrative_interruption"
    )


def test_observed_natural_result_precedes_simultaneous_clock_limit():
    module = episode_module()
    adapter, controller, backend, bridge, first, _, released = fixture(module)
    first.state = "result"
    summary, _, _ = module.collect_episode(
        adapter, controller, backend, bridge, lambda _: 1, first, seconds=60, deadline=1.0
    )
    assert released and summary["reason"] == "observed_result"


def test_long_run_bounds_are_explicit_and_finite():
    validate_limits(20, 900, 20000, long_run=True)
    with pytest.raises(ValueError):
        validate_limits(20, 900, 20000)
    for bad in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            validate_limits(20, bad, 1000, long_run=True)


def test_host_disk_pressure_latches_even_after_space_recovers():
    space = [40 * 1024**3]
    guard = HostBudgetGuard(".", disk_usage=lambda _: SimpleNamespace(free=space[0]))
    guard.check(force=True)
    space[0] = 10 * 1024**3
    with pytest.raises(RuntimeError, match="margin"):
        guard.check(force=True)
    space[0] = 50 * 1024**3
    with pytest.raises(RuntimeError, match="margin"):
        guard.check(force=True)
    assert guard.trace[-1]["fault"]


def test_host_growth_and_process_identity_are_checked():
    space = [50 * 1024**3]
    guard = HostBudgetGuard(
        ".",
        limits=HostLimits(max_drive_growth_gib=2),
        disk_usage=lambda _: SimpleNamespace(free=space[0]),
    )
    guard.check(force=True)
    space[0] -= 3 * 1024**3
    with pytest.raises(RuntimeError, match="growth"):
        guard.check(force=True)
    guard = HostBudgetGuard(
        ".",
        target=SimpleNamespace(pid=1, process_created_at=2),
        process_factory=lambda _: SimpleNamespace(is_running=lambda: True, create_time=lambda: 3),
        disk_usage=lambda _: SimpleNamespace(free=50 * 1024**3),
    )
    with pytest.raises(RuntimeError, match="identity"):
        guard.check(force=True)


def test_ctrl_c_after_press_releases_and_preserves_primary_exception():
    states = []

    def send(gas, brake):
        states.append((gas, brake))
        if gas:
            raise KeyboardInterrupt("injected after press")

    controller = PedalController(SimpleNamespace(set_pedals=send), lambda: True)
    try:
        with pytest.raises(KeyboardInterrupt, match="injected"):
            controller.submit(PedalAction(True, False, 0.1))
        assert states[-1] == (False, False)
        assert "KeyboardInterrupt" in controller.fault
    finally:
        controller.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows named ownership mutex")
def test_native_ownership_excludes_concurrent_collectors_and_releases():
    with NativeInputLease(), pytest.raises(RuntimeError, match="owns input"), NativeInputLease():
        pytest.fail("second collector entered")
    with NativeInputLease():
        pass


def test_registered_protocol_rejects_default_command_before_native_access(tmp_path):
    protocol = json.loads(
        Path("experiments/definitions/cycle-3-native-reliability-2.3.json").read_text()
    )
    args = SimpleNamespace(exploratory=False, episodes=2)
    with pytest.raises(ValueError, match="episodes"):
        validate_protocol(protocol, args)


def test_historical_registration_is_not_silently_reexecuted():
    with pytest.raises(ValueError, match="2.3"):
        validate_protocol(
            {"protocol_version": "native-reliability-2.2"}, SimpleNamespace(exploratory=False)
        )
