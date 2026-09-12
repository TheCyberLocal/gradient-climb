"""Prospective native protocol binding, before capture or input construction."""

from __future__ import annotations

import math
from pathlib import Path

from gradientclimb.artifacts import sha256_file

SCORED_SUCCESSES = frozenset({"success_natural_scored", "success_truncated_scored"})


def longest_scored_streak(outcomes):
    longest = current = 0
    for outcome in outcomes:
        current = current + 1 if outcome["classification"] in SCORED_SUCCESSES else 0
        longest = max(longest, current)
    return longest


def validate_protocol(protocol, args):
    """Reject unsupported registrations and mismatches before game discovery.

    Exploratory runs may reference a protocol, but never qualify as study data.
    Old registrations remain readable artifacts, not silently corrected reruns.
    """
    if args.exploratory:
        return
    if protocol.get("protocol_version") != "native-reliability-2.3":
        raise ValueError(
            "Registered execution requires native-reliability-2.3; amend other protocols"
        )
    expected = {
        "episodes": protocol["attempts_per_session"],
        "episode_seconds": protocol["episode_horizon_seconds"],
        "max_seconds": protocol["session_budget_seconds"],
        "seed": protocol["policies"]["random_seed"],
        "capture_backend": protocol["capture_backend"],
        "baseline": "alternating_gas_random",
        "reset_seconds": protocol["reset_deadline_seconds"],
        "restart_seconds": protocol["recovery"]["max_seconds"],
        "max_restarts": protocol["recovery"]["max_restarts_per_session"],
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"Protocol mismatch: {name} must be {value!r}")
    if args.inspect or args.probe_restart or args.policy or args.long_run:
        raise ValueError("Reliability requires full scripted attempts, not inspect/probe/policy")
    if args.restart_shortcut is None or not args.restart_shortcut.is_file():
        raise ValueError("Registered restart shortcut required")
    for name, digest in protocol["implementation_inputs"].items():
        path = getattr(args, name)
        if path is None or sha256_file(Path(path)) != digest:
            raise ValueError(f"Protocol mismatch: {name} content hash")
    if protocol["maximum_lease_seconds"] != 0.4:
        raise ValueError("Runner lease differs from protocol")
    if protocol["maximum_observation_age_seconds"] != 0.45:
        raise ValueError("Runner observation freshness differs from protocol")


def validate_limits(episodes, episode_seconds, max_seconds, *, long_run=False):
    if type(episodes) is not int or not 1 <= episodes <= 30:
        raise ValueError("Episode count must be in 1..30")
    for name, value, limit in (
        ("episode seconds", episode_seconds, 900 if long_run else 120),
        ("session seconds", max_seconds, 28800 if long_run else 10800),
    ):
        if not math.isfinite(value) or not 1 <= value <= limit:
            raise ValueError(f"{name} must be finite and in 1..{limit}")
