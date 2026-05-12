from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "runtime" / script_name
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_backtest_help() -> None:
    result = _run_help("run_backtest.py")
    assert result.returncode == 0
    assert "--start" in result.stdout
    assert "--frequency" in result.stdout


def test_local_backtest_runner_help() -> None:
    result = _run_help("local_backtest_runner.py")
    assert result.returncode == 0
    assert "--api-file" in result.stdout
    assert "--runs-dir" in result.stdout
