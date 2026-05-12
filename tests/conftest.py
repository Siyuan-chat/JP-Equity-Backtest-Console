from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "runtime"
FACTORS_ROOT = REPO_ROOT / "factors"

for path in [str(REPO_ROOT), str(RUNTIME_ROOT), str(FACTORS_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)
