from __future__ import annotations

import json
from pathlib import Path

from runtime.local_backtest_runner import _load_config, build_backtest_config, build_rules


def test_example_config_is_valid_json() -> None:
    path = Path(__file__).resolve().parents[1] / "local_backtest_config.example.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "backtest" in data


def test_builders_accept_example_config() -> None:
    path = Path(__file__).resolve().parents[1] / "local_backtest_config.example.json"
    config = _load_config(path)
    rules = build_rules(config)
    backtest_config = build_backtest_config(config)

    assert rules.market_cap.rank_min == 200
    assert backtest_config.composite_formula
