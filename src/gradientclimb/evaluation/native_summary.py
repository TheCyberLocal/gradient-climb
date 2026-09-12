"""Descriptive native observations, with controls and missing scores kept visible."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

VERSION = "native-descriptive-summary-3.0"


def _distance(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if len(a) > 1 else None,
        "minimum": float(a.min()),
        "maximum": float(a.max()),
        **{f"p{q:02}": float(np.percentile(a, q)) for q in (5, 25, 75, 95)},
        "ci95_low": None,
        "ci95_high": None,
        "ci_method": "not_estimated; descriptive observed native scores only",
        "scope": "available frozen-reader results; missing scores are not imputed",
    }


def describe_native_attempts(episodes, attempts_requested, *, attempt_outcomes=None):
    """Do not infer learning, independent terrain seeds or qualification from scores.

    A result reading can survive a later parking fault. Keep its descriptive value
    alongside the attempt classification; it cannot silently become a successful
    lifecycle or a qualified evaluation episode.
    """
    if attempt_outcomes is not None:
        by_index = {row["attempt"]["index"]: row for row in episodes}
        if len(by_index) != len(episodes):
            raise ValueError("Episode summaries must have unique attempt identities")
        indices = [row["index"] for row in attempt_outcomes]
        if len(set(indices)) != len(indices):
            raise ValueError("Attempt outcomes must have unique identities")
        inventory = []
        for outcome in attempt_outcomes:
            row = by_index.pop(outcome["index"], None)
            if row is not None and row["attempt"] != outcome:
                raise ValueError("Episode and attempt outcome disagree")
            inventory.append(
                row if row is not None else {"attempt": outcome, "summary_missing": True}
            )
        if by_index:
            raise ValueError("Episode summary is missing its authoritative attempt outcome")
        episodes = inventory
    if type(attempts_requested) is not int or attempts_requested < len(episodes):
        raise ValueError("Requested count must cover every observed attempt")
    groups = {}
    for episode in episodes:
        attempt = episode["attempt"]
        policy = attempt["policy"]
        if not isinstance(policy, str) or not policy:
            raise ValueError("Every attempt needs its actual control identity")
        score = episode.get("distance")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < 0
        ):
            raise ValueError("Observed scores must be finite nonnegative numbers or null")
        group = groups.setdefault(policy, [])
        group.append((score, attempt["classification"], bool(episode.get("summary_missing"))))

    def describe(rows):
        scores = [score for score, _, _ in rows if score is not None]
        return {
            "attempts": len(rows),
            "observed_scores": len(scores),
            "unknown_scores": len(rows) - len(scores),
            "classifications": dict(sorted(Counter(c for _, c, _ in rows).items())),
            "attempts_missing_episode_summary": sum(missing for _, _, missing in rows),
            "observed_result_distance": _distance(scores),
        }

    return {
        "version": VERSION,
        "attempts_requested": attempts_requested,
        "not_attempted": attempts_requested - len(episodes),
        **describe([row for rows in groups.values() for row in rows]),
        "by_control": {name: describe(rows) for name, rows in sorted(groups.items())},
        "pooled_scope": "descriptive inventory only; mixtures do not rank a single controller",
        "independent_game_seed_control": False,
        "qualification_inferred": False,
        "learning_speed_inferred": False,
    }
