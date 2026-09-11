"""Fresh-process import boundaries; no GPU, capture, or native input invocation."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "code",
    [
        "from gradientclimb.algorithms.screen_search import EpisodeCEM, ScreenLinearPolicy",
        "from gradientclimb.algorithms import AlwaysGasPolicy, RandomPolicy, LinearPolicy, load_policy",
        "import runpy; runpy.run_path('scripts/train_screen_cem.py', run_name='import_check')",
    ],
)
def test_numpy_screen_and_baseline_imports_do_not_load_torch(code):
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; {code}; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_public_exports_resolve_to_existing_implementations():
    from gradientclimb import algorithms
    from gradientclimb.algorithms.policies import AlwaysGasPolicy
    from gradientclimb.algorithms.ppo import ActorCritic

    assert algorithms.AlwaysGasPolicy is AlwaysGasPolicy
    assert algorithms.ActorCritic is ActorCritic
    assert set(algorithms.__all__).issubset(dir(algorithms))
    with pytest.raises(AttributeError):
        _ = algorithms.not_an_algorithm
