"""Default GUI parameter definitions.

Values mirror the existing local_backtest_config.example.json and dataclass
defaults in the backtest runtime.  The GUI writes these values into a temporary
JSON config consumed by `runtime/run_backtest.py`; it does not implement model logic.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_START = "2019-04-01"
DEFAULT_END = "2023-03-31"


MARKET_CAP_MODES = {
    "rank": "Rank range",
    "range_billion_jpy": "Market-cap range",
    "none": "No market-cap filter",
}

SORT_OPTIONS = {
    "market_cap_desc": "Market cap descending",
    "avg_daily_trading_value_desc": "Trading value descending",
    "annualized_volatility_asc": "Volatility ascending",
}

FREQUENCIES = {"monthly": "Monthly", "weekly": "Weekly"}

DEFAULT_RULES = {
    "market_cap": {"mode": "rank", "rank_min": 200, "rank_max": 500, "min_billion_jpy": 50.0, "max_billion_jpy": 5000.0},
    "liquidity": {"lookback_days": 60, "min_avg_daily_volume": 0},
    "volatility": {"lookback_days": 60, "max_annualized_volatility": 1.20},
    "selection": {"top_n_if_over_200": 500, "sort_by": "market_cap_desc"},
}

DEFAULT_BACKTEST = {
    "initial_capital": 100_000_000,
    "target_count": 10,
    "candidate_pool_count": 50,
    "min_execution_price_coverage": 0.80,
    "min_execution_price_count": 10,
    "rebuild_targets_after_price_lookup": True,
    "rescale_partial_target_weights": False,
    "record_missing_execution_prices": True,
    "max_gross_exposure": 1.0,
    "enable_short_leg": True,
    "short_gross_exposure": 0.50,
    "short_target_count": 21,
    "short_construction_mode": "advanced_bottom_decile",
    "short_weighting_mode": "score_proportional",
    "short_expansion_mode": "expand_to_bottom20",
    "short_adv_threshold": None,
    "short_sector_cap": None,
    "short_min_names": 8,
    "short_max_single_weight": 0.075,
    "transaction_cost_bps": 10.0,
    "slippage_bps": 5.0,
    "stop_loss_pct": 0.18,
    "stop_loss_check_frequency": "daily",
    "stop_loss_cash_penalty_pct": 0.25,
    "stop_loss_cooldown_rebalances": 3,
    "execution_price": "close",
    "signal_execution_lag_days": 1,
    "lot_size": 100,
    "round_lots": False,
    "allow_yfinance_fallback": False,
    "risk_index_code": "TOPIX",
    "use_risk_gating": True,
    "benchmark_index_codes": ["TOPIX", "NIKKEI225"],
    "include_sp500_benchmark": True,
    "sp500_yfinance_ticker": "^GSPC",
    "portfolio_constraints": {"max_names_per_sector": 2, "sector_cap_mode": "hard"},
    "post_score_adjustments": {
        "regime_sector_bias_regime_source": "active_regime",
        "trend_lowvol_sector_bias_enabled": False,
        "trend_lowvol_sector_bonus": 0.0,
        "trend_lowvol_sector_penalty": 0.0,
        "range_lowvol_sector_bias_enabled": False,
        "range_lowvol_sector_bonus": 0.0,
        "range_lowvol_sector_penalty": 0.0,
        "riskoff_highvol_sector_bias_enabled": False,
        "riskoff_highvol_sector_bonus": 0.0,
        "riskoff_highvol_sector_penalty": 0.0,
        "hot_riskon_sector_bias_enabled": False,
        "hot_riskon_sector_bonus": 0.0,
        "hot_riskon_sector_penalty": 0.0,
        "macro_reflation_sector_alignment_enabled": False,
        "macro_reflation_size_tilt_enabled": False,
        "macro_reflation_sector_bonus": 0.0,
        "macro_reflation_sector_penalty": 0.0,
        "macro_reflation_large_cap_bonus": 0.0,
        "macro_reflation_small_cap_penalty": 0.0,
        "macro_reflation_low_liquidity_penalty": 0.0,
        "macro_reflation_large_cap_rank_threshold": 250,
        "macro_reflation_small_cap_rank_threshold": 700,
        "macro_reflation_liquidity_floor": 1000000.0,
        "macro_reflation_leader_bias_enabled": False,
        "macro_reflation_leader_core_bonus": 0.0,
        "macro_reflation_trend_confirmation_bonus": 0.0,
        "macro_reflation_leader_combo_bonus": 0.0,
        "macro_reflation_weak_trend_penalty": 0.0,
        "macro_reflation_non_leader_penalty": 0.0,
        "macro_reflation_mgate_floor": 0.0,
        "macro_reflation_value_floor": 0.0,
    },
}

DEFAULT_RISK = {
    "lookback_high": 252,
    "rv_window": 20,
    "rv_q_window": 252,
    "risk_off_score": 4,
    "trend_short_ma": 50,
    "trend_long_ma": 200,
    "trend_confirm_days": 3,
    "upgrade_confirm_days": 1,
    "downgrade_confirm_days": 2,
}

DEFAULT_ALLOCATION_GATE = {
    "enabled": True,
    "mode": "matrix",
    "risk_override_threshold": 3,
    "severe_risk_score_threshold": 4,
    "defensive_regime": "riskoff_highvol",
}

DEFAULT_CACHE = {
    "save_format": "parquet",
    "include_margin": True,
    "margin_optional": True,
    "include_financials": True,
    "include_topix_index": True,
    "include_nikkei_index": True,
    "additional_index_codes": ["0028", "002A", "002D", "0070", "0081", "0084", "0085", "0086", "008A", "008D", "8100", "8200"],
    "allow_missing_market_cap": False,
    "market_cap_fallback": "disabled",
    "force_rebuild_cache": False,
}


@dataclass(frozen=True)
class FactorDefinition:
    key: str
    label: str
    variable: str
    description: str
    default_enabled: bool = True


FACTORS = [
    FactorDefinition("quality", "Quality", "Q", "Profitability and balance-sheet quality from disclosed financial summary."),
    FactorDefinition("value", "Value", "V", "Valuation-style score from point-in-time financial summary inputs."),
    FactorDefinition("residual_momentum", "12-1 Momentum", "M", "Classic 12-1 price momentum signal with the most recent month skipped."),
    FactorDefinition("dual_ma", "Dual MA", "TS", "Trend strength signal based on short/long moving average structure."),
    FactorDefinition("reversal", "Reversal", "REV", "Short-term skip-window reversal signal from local prices."),
    FactorDefinition("behaviour", "Behaviour", "B", "Margin/credit behaviour signal from J-Quants market data."),
    FactorDefinition("attention", "Attention", "T", "Attention/liquidity activity signal from price and capitalization inputs."),
]

DEFAULT_COMPOSITE_WEIGHTS = {
    "w_q": 0.20,
    "w_v": 0.20,
    "w_m": 0.20,
    "w_ts": 0.15,
    "w_b": 0.10,
    "w_t": 0.10,
    "w_r": 0.10,
}

SECTOR_CAP_MODES = {
    "hard": "Hard",
    "soft_penalty": "Soft penalty",
    "exception": "Exception",
}

SHORT_CONSTRUCTION_MODES = {
    "simple_bottom_n": "Simple bottom N",
    "advanced_bottom_decile": "Advanced bottom decile",
}

SHORT_WEIGHTING_MODES = {
    "equal_weight": "Equal weight",
    "rank_weighted": "Rank weighted",
    "score_proportional": "Score proportional",
}

SHORT_EXPANSION_MODES = {
    "bottom_decile_only": "Bottom decile only",
    "expand_to_bottom20": "Expand to bottom 20%",
    "expand_to_bottom30": "Expand to bottom 30%",
}

DEFAULT_FILTER_ENABLES = {
    "market_cap": True,
    "liquidity": False,
    "volatility": True,
    "max_lot_cost": False,
    "sector_constraints": True,
}

WEIGHT_DESCRIPTIONS = {
    "w_q": "Quality factor weight.",
    "w_v": "Value factor weight.",
    "w_m": "12-1 momentum factor weight.",
    "w_ts": "Dual-MA / trend-strength factor weight.",
    "w_b": "Behaviour factor weight.",
    "w_t": "Attention factor weight.",
    "w_r": "Reversal module weight.",
}

DEFAULT_FORMULA = "w_q*Q + w_v*V + w_m*M + w_ts*TS + w_b*B + w_t*T + w_r*REV"

WEIGHT_SOURCE_MODES = {
    "manual": "Manual weights",
}

REGIME_SECTOR_BIAS_SOURCES = {
    "active_regime": "Active regime",
    "weight_regime": "Weight regime",
    "position_regime": "Position regime",
    "raw_regime": "Raw regime",
}

ALLOCATION_GATE_MODES = {
    "matrix": "Unified matrix",
    "legacy": "Legacy stacked",
}

REGIME_NAMES = [
    "trend_lowvol",
    "range_lowvol",
    "riskoff_highvol",
    "hot_riskon",
    "macro_reflation",
]

DEFAULT_REGIME_CONFIG = {
    "trend_strong_threshold": 0.8,
    "trend_weak_threshold": 0.3,
    "trend_recovery_threshold": 0.55,
    "trend_recovery_slope_t_threshold": 8.0,
    "trend_recovery_ret120_threshold": 0.03,
    "trend_recovery_breadth_floor": -0.08,
    "trend_lowvol_exit_t_threshold": 0.2,
    "trend_lowvol_exit_ret120_threshold": 0.015,
    "trend_lowvol_exit_slope_t_threshold": 4.0,
    "trend_lowvol_exit_breadth_threshold": -0.2,
    "trend_lowvol_fast_exit_ret120_threshold": 0.0,
    "trend_lowvol_fast_exit_breadth_threshold": -0.35,
    "vol_high_enter": 0.65,
    "vol_high_exit": 0.55,
    "vol_low_enter": 0.35,
    "vol_low_exit": 0.45,
    "switch_confirm_bars": 3,
    "overlay_confirm_bars": 2,
    "high_to_low_trend_switch_bars": 1,
    "trend_lowvol_exit_bars": 2,
    "range_lowvol_exit_bars": 1,
    "range_lowvol_trend_fast_t_threshold": 0.75,
    "range_lowvol_trend_fast_breadth_threshold": 0.15,
    "range_lowvol_riskoff_fast_breadth_threshold": -0.25,
    "macro_reflation_trend_threshold": 0.9,
    "macro_reflation_f_score_threshold": 0.85,
    "macro_reflation_breadth_threshold": 0.05,
    "macro_reflation_range_base_breadth_threshold": 0.15,
    "macro_reflation_vmg_threshold": 0.0,
    "macro_reflation_range_base_vmg_threshold": 0.03,
    "macro_reflation_cycdef_threshold": 0.0,
    "macro_reflation_ret120_threshold": 0.04,
    "macro_reflation_r_score_floor": 0.0,
    "macro_reflation_risk_score_floor": -0.2,
    "macro_reflation_max_vol_pct": 0.5,
    "macro_reflation_exit_bars": 1,
    "macro_reflation_exit_breadth_threshold": 0.08,
}

WEIGHT_DISPLAY_NAMES = {
    "w_q": "w<sub>Q</sub>",
    "w_v": "w<sub>V</sub>",
    "w_m": "w<sub>M</sub>",
    "w_ts": "w<sub>TS</sub>",
    "w_b": "w<sub>B</sub>",
    "w_t": "w<sub>T</sub>",
    "w_r": "w<sub>REV</sub>",
}

COMPOSITE_FORMULA_HTML = """
<div style="line-height: 145%;">
<b>Composite score</b><br>
Score = w<sub>Q</sub> × Q + w<sub>V</sub> × V + w<sub>M</sub> × M + w<sub>TS</sub> × TS + w<sub>B</sub> × B + w<sub>T</sub> × T + w<sub>REV</sub> × REV
</div>
"""

COMPOSITE_EXPLANATION_HTML = """
<div style="line-height: 145%;">
<b>How the composite score is built</b><br>
Each enabled factor is normalized onto the shared composite scale before scoring.<br>
The default model is a simple linear combination of Quality (Q), Value (V), 12-1 Momentum (M), trend strength (TS), Behaviour (B), Attention (T), and Reversal (REV).<br>
You can change both the weights and the formula directly in the GUI.
</div>
"""

FORMULA_VARIABLES = {
    "Q",
    "V",
    "M",
    "TS",
    "REV",
    "B",
    "T",
    *DEFAULT_COMPOSITE_WEIGHTS.keys(),
}

VARIABLE_TO_FACTOR = {
    "Q": "quality",
    "V": "value",
    "M": "residual_momentum",
    "TS": "dual_ma",
    "REV": "reversal",
    "B": "behaviour",
    "T": "attention",
}
