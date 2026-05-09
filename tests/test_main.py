"""Smoke tests covering all (case, da_method) combinations of scripts/main.py."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = REPO_ROOT / "scripts" / "main.py"

CASES = ["lorenz_63", "lorenz_96", "kuramoto"]
DA_METHODS = ["enkf", "agmf", "pff", "particle_filter"]


@pytest.mark.parametrize("case", CASES)  # type: ignore[misc]
@pytest.mark.parametrize("da_method", DA_METHODS)  # type: ignore[misc]
def test_main_combination(case: str, da_method: str) -> None:
    """Run scripts/main.py end-to-end for every (case, da_method) pair."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [
            sys.executable,
            str(MAIN_SCRIPT),
            f"case={case}",
            f"da_method={da_method}",
            "data_assimilation_steps=10",
            "model_integration_steps=5",
            "ensemble_size=50",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"case={case} da_method={da_method} failed (exit={result.returncode})\n"
        f"--- stdout ---\n{result.stdout.decode(errors='replace')}\n"
        f"--- stderr ---\n{result.stderr.decode(errors='replace')}"
    )
