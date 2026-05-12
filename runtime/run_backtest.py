from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_ROOT.parent
FACTORS_ROOT = PROJECT_ROOT / "factors"
LEGACY_ROOT = PROJECT_ROOT.parent / "backtest"
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != LEGACY_ROOT.resolve()]
sys.path[:0] = [str(PROJECT_ROOT), str(RUNTIME_ROOT), str(FACTORS_ROOT)]

from local_backtest_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
