from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.defaults import DEFAULT_BACKTEST, DEFAULT_CACHE


@dataclass
class BacktestCommand:
    program: str
    arguments: list[str]
    config_path: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def backtest_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return backtest_dir() / "data_cache"


def default_runs_dir() -> Path:
    return backtest_dir() / "backtest_runs"


def python_executable() -> Path:
    local = backtest_dir() / ".venv" / "Scripts" / "python.exe"
    if local.exists():
        return local
    current = Path(sys.executable)
    if current.name.lower() == "pythonw.exe":
        sibling = current.with_name("python.exe")
        if sibling.exists():
            return sibling
    return current


def build_runtime_config(params: dict[str, Any]) -> dict[str, Any]:
    market_cap_mode = params["market_cap_mode"] if params.get("use_market_cap_filter", True) else "none"
    min_avg_daily_volume = int(params["min_avg_daily_volume"]) if params.get("use_liquidity_filter", False) else 0
    max_annualized_volatility = (
        float(params["max_annualized_volatility"]) if params.get("use_volatility_filter", False) else 999.0
    )
    rules = {
        "market_cap": {
            "mode": market_cap_mode,
            "rank_min": int(params["rank_min"]),
            "rank_max": int(params["rank_max"]),
            "min_billion_jpy": float(params["min_market_cap"]),
            "max_billion_jpy": float(params["max_market_cap"]),
        },
        "liquidity": {
            "lookback_days": int(params["liquidity_lookback"]),
            "min_avg_daily_volume": min_avg_daily_volume,
        },
        "volatility": {
            "lookback_days": int(params["volatility_lookback"]),
            "max_annualized_volatility": max_annualized_volatility,
        },
        "selection": {
            "top_n_if_over_200": int(params["selection_top_n"]),
            "sort_by": params["selection_sort"],
        },
    }
    if params.get("use_max_lot_cost_filter", False):
        rules["selection"]["max_lot_cost_jpy"] = float(params["max_lot_cost_jpy"])

    portfolio_constraints: dict[str, Any] = {}
    if params.get("use_sector_constraints", True):
        portfolio_constraints = {
            "max_names_per_sector": int(params["max_names_per_sector"]),
            "sector_cap_mode": str(params["sector_cap_mode"]),
        }

    backtest = {
        "target_count": int(params["target_count"]),
        "candidate_pool_count": int(DEFAULT_BACKTEST["candidate_pool_count"]),
        "min_execution_price_coverage": float(DEFAULT_BACKTEST["min_execution_price_coverage"]),
        "min_execution_price_count": int(DEFAULT_BACKTEST["min_execution_price_count"]),
        "rebuild_targets_after_price_lookup": bool(DEFAULT_BACKTEST["rebuild_targets_after_price_lookup"]),
        "rescale_partial_target_weights": bool(DEFAULT_BACKTEST["rescale_partial_target_weights"]),
        "record_missing_execution_prices": bool(DEFAULT_BACKTEST["record_missing_execution_prices"]),
        "max_gross_exposure": float(params["max_gross_exposure"]),
        "enable_short_leg": bool(params["enable_short_leg"]),
        "short_gross_exposure": float(params["short_gross_exposure"]),
        "short_target_count": int(params["short_target_count"]),
        "short_construction_mode": str(params["short_construction_mode"]),
        "short_weighting_mode": str(params["short_weighting_mode"]),
        "short_expansion_mode": str(params["short_expansion_mode"]),
        "short_adv_threshold": (float(params["short_adv_threshold"]) if float(params["short_adv_threshold"]) > 0 else None),
        "short_sector_cap": (float(params["short_sector_cap"]) if float(params["short_sector_cap"]) > 0 else None),
        "short_min_names": int(params["short_min_names"]),
        "short_max_single_weight": float(params["short_max_single_weight"]),
        "transaction_cost_bps": float(params["transaction_cost_bps"]),
        "slippage_bps": float(params["slippage_bps"]),
        "stop_loss_pct": float(params.get("stop_loss_pct", DEFAULT_BACKTEST["stop_loss_pct"])),
        "stop_loss_check_frequency": str(params.get("stop_loss_check_frequency", DEFAULT_BACKTEST["stop_loss_check_frequency"])),
        "stop_loss_cash_penalty_pct": float(params.get("stop_loss_cash_penalty_pct", DEFAULT_BACKTEST["stop_loss_cash_penalty_pct"])),
        "stop_loss_cooldown_rebalances": int(params.get("stop_loss_cooldown_rebalances", DEFAULT_BACKTEST["stop_loss_cooldown_rebalances"])),
        "execution_price": "close",
        "signal_execution_lag_days": 1,
        "lot_size": int(params["lot_size"]),
        "round_lots": bool(params["round_lots"]),
        "allow_yfinance_fallback": False,
        "risk_index_code": "TOPIX",
        "use_risk_gating": bool(params["use_risk_gating"]),
        "benchmark_index_codes": ["TOPIX", "NIKKEI225"],
        "include_sp500_benchmark": bool(params["include_sp500_benchmark"]),
        "sp500_yfinance_ticker": "^GSPC",
        "enabled_factors": [key for key, enabled in params["enabled_factors"].items() if enabled],
        "composite_weights": params["composite_weights"],
        "composite_formula": params["formula"],
        "weight_source_mode": params["weight_source_mode"],
        "regime_weight_map": params["regime_weight_map"],
        "regime_config": params["regime_config"],
        "allocation_gate_config": {
            "enabled": bool(params["allocation_gate_enabled"]),
            "mode": str(params["allocation_gate_mode"]),
            "risk_override_threshold": int(params["allocation_gate_risk_override_threshold"]),
            "severe_risk_score_threshold": int(params["allocation_gate_severe_risk_score_threshold"]),
            "defensive_regime": str(params["allocation_gate_defensive_regime"]),
        },
        "portfolio_constraints": portfolio_constraints,
        "post_score_adjustments": {
            **dict(DEFAULT_BACKTEST.get("post_score_adjustments", {})),
            **dict(params.get("post_score_adjustments", {})),
        },
        "factor_configs": {
            "risk": {
                "lookback_high": int(params["risk_lookback_high"]),
                "rv_window": int(params["risk_rv_window"]),
                "rv_q_window": int(params["risk_rv_q_window"]),
                "risk_off_score": int(params["risk_off_score"]),
                "trend_short_ma": int(params["risk_trend_short_ma"]),
                "trend_long_ma": int(params["risk_trend_long_ma"]),
                "trend_confirm_days": int(params["risk_trend_confirm_days"]),
                "upgrade_confirm_days": int(params["risk_upgrade_confirm_days"]),
                "downgrade_confirm_days": int(params["risk_downgrade_confirm_days"]),
            }
        },
    }
    cache = dict(DEFAULT_CACHE)
    cache["force_rebuild_cache"] = bool(params["force_rebuild_cache"])
    return {
        "initial_capital": float(params["initial_capital"]),
        "history_overrides": {
            "enabled_factors": backtest["enabled_factors"],
            "use_risk_gating": backtest["use_risk_gating"],
        },
        "cache": cache,
        "rules": rules,
        "backtest": backtest,
        "gui": {
            "formula": params["formula"],
        },
    }


def write_runtime_config(params: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gui_backtest_config.json"
    path.write_text(json.dumps(build_runtime_config(params), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_powershell_command(params: dict[str, Any], config_path: Path) -> BacktestCommand:
    script = backtest_dir() / "runtime" / "run_backtest.py"
    program = str(python_executable())
    args = [
        str(script),
        "--start",
        params["start_date"].toString("yyyy-MM-dd"),
        "--end",
        params["end_date"].toString("yyyy-MM-dd"),
        "--api-file",
        str(Path(params["api_file"]).resolve()),
        "--config",
        str(config_path.resolve()),
        "--data-dir",
        str(Path(params.get("data_dir") or default_data_dir()).resolve()),
        "--runs-dir",
        str(Path(params.get("runs_dir") or default_runs_dir()).resolve()),
        "--frequency",
        str(params["frequency"]),
    ]
    if params["force_rebuild_cache"]:
        args.append("--force-rebuild-cache")
    return BacktestCommand(program, args, config_path)
