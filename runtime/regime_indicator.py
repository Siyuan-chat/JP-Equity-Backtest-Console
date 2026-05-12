from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from historical_data import JpxHistoricalProvider, normalize_code


INDEX_CODE_ALIASES = {
    "topix": "TOPIX",
    "nikkei225": "NIKKEI225",
    "topix_small": "002D",
    "topix_value": "8100",
    "topix_growth": "8200",
    "cyclical_basket": "__cyclical__",
    "defensive_basket": "__defensive__",
}

CYCLICAL_INDEX_CODES = ("0081", "0085", "0086")
DEFENSIVE_INDEX_CODES = ("0084", "008A", "008D")
HBLV_HIGH_BETA_CODES = ("0070", "002D")
HBLV_LOW_VOL_CODES = ("0028", "002A")
CYCLICAL_SECTOR_NAMES = ("銀行業", "機械", "輸送用機器", "卸売業", "海運業", "非鉄金属", "鉄鋼")
DEFENSIVE_SECTOR_NAMES = ("医薬品", "食料品", "小売業", "電気・ガス業", "陸運業", "情報･通信業")
VALUE_PROXY_SECTOR_NAMES = ("銀行業", "保険業", "卸売業", "機械", "輸送用機器", "鉄鋼")
GROWTH_PROXY_SECTOR_NAMES = ("情報･通信業", "サービス業", "精密機器", "医薬品", "電気機器")


DEFAULT_REGIME_CONFIG: dict[str, Any] = {
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
    "zscore_window": 252,
    "vol_quantile_window": 252,
    "trend_lookback": 120,
    "breadth_window": 60,
    "turnover_window": 20,
    "risk_window": 20,
    "reflation_window": 60,
    "breadth_min_obs": 20,
    "beta_window": 60,
    "group_quantile": 0.3,
    "stale_series_max_age_days": 10,
}


@dataclass
class RegimeState:
    active_regime: str = ""
    candidate_regime: str = ""
    candidate_count: int = 0
    overlay_candidate_regime: str = ""
    overlay_candidate_count: int = 0
    last_vol_bucket: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_regime": self.active_regime,
            "candidate_regime": self.candidate_regime,
            "candidate_count": int(self.candidate_count),
            "overlay_candidate_regime": self.overlay_candidate_regime,
            "overlay_candidate_count": int(self.overlay_candidate_count),
            "last_vol_bucket": self.last_vol_bucket,
        }


def _cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_REGIME_CONFIG)
    out.update(dict(config or {}))
    return out


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _zscore_last(series: pd.Series, window: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    tail = clean.tail(max(5, int(window)))
    if len(tail) < 5:
        return float("nan")
    std = float(tail.std(ddof=0))
    if not np.isfinite(std) or abs(std) < 1e-12:
        return float("nan")
    return float((tail.iloc[-1] - tail.mean()) / std)


def _quantile_last(series: pd.Series, window: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    tail = clean.tail(max(20, int(window)))
    if len(tail) < 20:
        return float("nan")
    rank = tail.rank(pct=True).iloc[-1]
    return float(rank) if pd.notna(rank) else float("nan")


def _log_return(close: pd.Series, window: int) -> pd.Series:
    work = pd.to_numeric(close, errors="coerce")
    return np.log(work).diff(window)


def _rolling_slope_t(close: pd.Series, window: int) -> pd.Series:
    log_close = np.log(pd.to_numeric(close, errors="coerce"))

    def _calc(values: np.ndarray) -> float:
        y = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < max(20, window // 3):
            return np.nan
        y = y[mask]
        x = np.arange(len(y), dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        sxx = float(np.dot(x, x))
        if sxx <= 0:
            return np.nan
        slope = float(np.dot(x, y) / sxx)
        resid = y - slope * x
        dof = len(y) - 2
        if dof <= 0:
            return np.nan
        sigma2 = float(np.dot(resid, resid) / dof)
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return np.nan
        se = float(np.sqrt(sigma2 / sxx))
        if se <= 0:
            return np.nan
        return slope / se

    return log_close.rolling(window).apply(_calc, raw=True)


def _annualized_realized_vol(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(pd.to_numeric(close, errors="coerce")).diff()
    return log_ret.rolling(window).std(ddof=0) * np.sqrt(252.0)


def _price_series(provider: JpxHistoricalProvider, index_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        return provider.load_index_close(index_code, start, end, {})
    except Exception:
        return pd.Series(dtype=float)


def _fresh_series_or_empty(series: pd.Series, signal_date: pd.Timestamp, max_age_days: int) -> pd.Series:
    if not isinstance(series, pd.Series) or series.empty:
        return pd.Series(dtype=float)
    latest = pd.to_datetime(series.index, errors="coerce").max()
    if pd.isna(latest):
        return pd.Series(dtype=float)
    if latest < signal_date - pd.Timedelta(days=max(1, int(max_age_days))):
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").dropna().sort_index()


def _mean_of_series(series_list: Iterable[pd.Series]) -> pd.Series:
    frames = [s.rename(idx) for idx, s in enumerate(series_list) if isinstance(s, pd.Series) and not s.empty]
    if not frames:
        return pd.Series(dtype=float)
    merged = pd.concat(frames, axis=1).sort_index()
    return merged.mean(axis=1, skipna=True)


def _compute_turnover_series(provider: JpxHistoricalProvider, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    prices = provider.table("prices", copy=False)
    work = prices.loc[prices["date"].between(start, end, inclusive="both")].copy()
    if work.empty:
        return pd.Series(dtype=float)
    trade_value_col = "trading_value" if "trading_value" in work.columns else "value"
    if trade_value_col in work.columns:
        value = pd.to_numeric(work[trade_value_col], errors="coerce")
    else:
        price_col = "adjustment_close" if "adjustment_close" in work.columns else "close"
        value = pd.to_numeric(work.get(price_col), errors="coerce") * pd.to_numeric(work.get("volume"), errors="coerce")
    work["turnover_value"] = value
    daily = work.groupby("date", as_index=True)["turnover_value"].sum(min_count=1).sort_index()
    return daily


def _latest_snapshot_by_code(frame: pd.DataFrame, date_col: str, signal_date: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or date_col not in frame.columns or "code" not in frame.columns:
        return pd.DataFrame()
    work = frame.loc[frame[date_col] <= signal_date].copy()
    if work.empty:
        return pd.DataFrame()
    work = work.dropna(subset=[date_col, "code"]).sort_values(["code", date_col])
    return work.groupby("code", as_index=False).tail(1)


def _active_universe_codes(provider: JpxHistoricalProvider, signal_date: pd.Timestamp) -> set[str]:
    try:
        universe = provider.table("universe", required=False, copy=False)
    except Exception:
        return set()
    latest = _latest_snapshot_by_code(universe, "asof_date", signal_date)
    if latest.empty or "in_universe" not in latest.columns:
        return set()
    in_universe = latest["in_universe"]
    if in_universe.dtype == bool:
        mask = in_universe
    else:
        mask = in_universe.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
    return set(latest.loc[mask, "code"].dropna().astype(str))


def _prepare_equity_panel(
    provider: JpxHistoricalProvider,
    start: pd.Timestamp,
    end: pd.Timestamp,
    universe_codes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    prices = provider.table("prices", copy=False)
    work = prices.loc[prices["date"].between(start, end, inclusive="both")].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    if universe_codes:
        work = work.loc[work["code"].isin(universe_codes)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    price_col = "adjustment_close" if "adjustment_close" in work.columns else "close"
    work["close"] = pd.to_numeric(work[price_col], errors="coerce")
    if "trading_value" in work.columns:
        work["turnover_value"] = pd.to_numeric(work["trading_value"], errors="coerce")
    else:
        work["turnover_value"] = pd.to_numeric(work["close"], errors="coerce") * pd.to_numeric(work.get("volume"), errors="coerce")
    work = work.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"])
    close_pivot = work.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    turnover = work.groupby("date", as_index=True)["turnover_value"].sum(min_count=1).sort_index()
    daily_log_returns = np.log(close_pivot).diff()
    return work, close_pivot, turnover


def _market_proxy_close(close_pivot: pd.DataFrame) -> pd.Series:
    if close_pivot.empty:
        return pd.Series(dtype=float)
    daily_log_returns = np.log(close_pivot).diff()
    market_log_return = daily_log_returns.mean(axis=1, skipna=True)
    market_log_return = pd.to_numeric(market_log_return, errors="coerce").dropna()
    if market_log_return.empty:
        return pd.Series(dtype=float)
    proxy = np.exp(market_log_return.cumsum()) * 100.0
    proxy.name = "market_proxy"
    return proxy.sort_index()


def _compute_breadth_from_close_pivot(
    close_pivot: pd.DataFrame,
    signal_date: pd.Timestamp,
    window: int,
    min_obs: int,
) -> tuple[float, dict[str, Any]]:
    if close_pivot.empty:
        return float("nan"), {"breadth_code_count": 0, "breadth_obs_count": 0}
    latest_date = close_pivot.index[close_pivot.index <= signal_date]
    if len(latest_date) == 0:
        return float("nan"), {"breadth_code_count": 0, "breadth_obs_count": 0}
    ma = close_pivot.rolling(window, min_periods=min_obs).mean()
    last_date = latest_date[-1]
    latest = close_pivot.loc[last_date]
    latest_ma = ma.loc[last_date]
    valid = latest.notna() & latest_ma.notna()
    if valid.sum() <= 0:
        return float("nan"), {"breadth_code_count": 0, "breadth_obs_count": 0}
    pct_above = float((latest[valid] > latest_ma[valid]).mean())
    breadth60 = 2.0 * (pct_above - 0.5)
    return breadth60, {"breadth_code_count": int(valid.sum()), "breadth_obs_count": int(close_pivot.shape[0])}


def _latest_cross_section(series: pd.Series, quantile: float, high: bool) -> pd.Index:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return pd.Index([])
    q = min(max(float(quantile), 0.05), 0.45)
    threshold = clean.quantile(1.0 - q if high else q)
    selected = clean[clean >= threshold] if high else clean[clean <= threshold]
    return pd.Index(selected.index.astype(str))


def _cross_section_spread(
    values: pd.Series,
    left_codes: pd.Index,
    right_codes: pd.Index,
) -> float:
    clean = pd.to_numeric(values, errors="coerce")
    left = clean.loc[clean.index.intersection(left_codes)].dropna()
    right = clean.loc[clean.index.intersection(right_codes)].dropna()
    if left.empty or right.empty:
        return float("nan")
    return float(left.mean() - right.mean())


def _sector_spread(
    values: pd.Series,
    sector_map: pd.Series,
    left_names: tuple[str, ...],
    right_names: tuple[str, ...],
) -> float:
    if values.empty or sector_map.empty:
        return float("nan")
    sectors = sector_map.astype(str)
    left_codes = sectors.loc[sectors.isin(left_names)].index.astype(str)
    right_codes = sectors.loc[sectors.isin(right_names)].index.astype(str)
    return _cross_section_spread(values, left_codes, right_codes)


def _compute_breadth60(provider: JpxHistoricalProvider, signal_date: pd.Timestamp, window: int, min_obs: int) -> tuple[float, dict[str, Any]]:
    start = signal_date - pd.Timedelta(days=max(220, window * 4))
    prices = provider.table("prices", copy=False)
    work = prices.loc[prices["date"].between(start, signal_date, inclusive="both")].copy()
    if work.empty:
        return float("nan"), {"breadth_code_count": 0, "breadth_obs_count": 0}
    price_col = "adjustment_close" if "adjustment_close" in work.columns else "close"
    work["close"] = pd.to_numeric(work[price_col], errors="coerce")
    pivot = work.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    if pivot.empty:
        return float("nan"), {"breadth_code_count": 0, "breadth_obs_count": 0}
    ma60 = pivot.rolling(window, min_periods=min_obs).mean()
    latest_date = pivot.index[pivot.index <= signal_date]
    if len(latest_date) == 0:
        return float("nan"), {"breadth_code_count": 0, "breadth_obs_count": 0}
    last_date = latest_date[-1]
    latest = pivot.loc[last_date]
    latest_ma = ma60.loc[last_date]
    valid = latest.notna() & latest_ma.notna()
    if valid.sum() <= 0:
        return float("nan"), {"breadth_code_count": 0, "breadth_obs_count": 0}
    pct_above = float((latest[valid] > latest_ma[valid]).mean())
    breadth60 = 2.0 * (pct_above - 0.5)
    return breadth60, {"breadth_code_count": int(valid.sum()), "breadth_obs_count": int(pivot.shape[0])}


def _classify_unified_regime(features: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, str, str]:
    t = _safe_float(features.get("T"))
    vol_pct = _safe_float(features.get("vol_pct"))
    breadth60 = _safe_float(features.get("breadth60"))
    ret120 = _safe_float(features.get("ret120"))
    slope_t = _safe_float(features.get("slope_t"))
    r_score = _safe_float(features.get("R"))
    f_score = _safe_float(features.get("F"))
    vmg60 = _safe_float(features.get("vmg60"))
    cycdef60 = _safe_float(features.get("cycdef60"))

    strong = float(cfg["trend_strong_threshold"])
    weak = float(cfg["trend_weak_threshold"])
    recovery = float(cfg["trend_recovery_threshold"])
    recovery_slope = float(cfg["trend_recovery_slope_t_threshold"])
    recovery_ret120 = float(cfg["trend_recovery_ret120_threshold"])
    recovery_breadth_floor = float(cfg["trend_recovery_breadth_floor"])
    high = float(cfg["vol_high_enter"])
    low = float(cfg["vol_low_enter"])
    low_exit = float(cfg["vol_low_exit"])
    macro_breadth = float(cfg["macro_reflation_breadth_threshold"])
    macro_range_base_breadth = float(cfg["macro_reflation_range_base_breadth_threshold"])
    macro_trend = float(cfg["macro_reflation_trend_threshold"])
    macro_f_score = float(cfg["macro_reflation_f_score_threshold"])
    macro_vmg = float(cfg["macro_reflation_vmg_threshold"])
    macro_range_base_vmg = float(cfg["macro_reflation_range_base_vmg_threshold"])
    macro_cycdef = float(cfg["macro_reflation_cycdef_threshold"])
    macro_ret120 = float(cfg["macro_reflation_ret120_threshold"])
    macro_r_score_floor = float(cfg["macro_reflation_r_score_floor"])
    macro_risk_floor = float(cfg["macro_reflation_risk_score_floor"])
    macro_max_vol_pct = float(cfg["macro_reflation_max_vol_pct"])

    recovery_uptrend = (
        np.isfinite(t)
        and np.isfinite(vol_pct)
        and vol_pct <= low
        and (
            t >= recovery
            or (
                np.isfinite(ret120)
                and np.isfinite(slope_t)
                and ret120 >= recovery_ret120
                and slope_t >= recovery_slope
                and (not np.isfinite(breadth60) or breadth60 >= recovery_breadth_floor)
            )
        )
    )

    base_regime = "range_lowvol"
    if np.isfinite(t) and np.isfinite(vol_pct):
        if recovery_uptrend or (t >= strong and vol_pct <= low):
            base_regime = "trend_lowvol"
        elif abs(t) <= weak and vol_pct <= low:
            base_regime = "range_lowvol"
        elif t <= -strong and vol_pct >= 0.5:
            base_regime = "riskoff_highvol"
        elif t >= strong:
            base_regime = "trend_lowvol"
        elif vol_pct <= low_exit:
            base_regime = "range_lowvol"

    overlay = ""
    if np.isfinite(vol_pct) and np.isfinite(r_score) and np.isfinite(breadth60):
        if vol_pct >= high and r_score <= -0.7 and breadth60 < -0.2:
            overlay = "riskoff_highvol"
    if not overlay and np.isfinite(t) and np.isfinite(f_score):
        breadth_threshold = macro_range_base_breadth if base_regime == "range_lowvol" else macro_breadth
        vmg_threshold = macro_range_base_vmg if base_regime == "range_lowvol" else macro_vmg
        trend_ok = t >= max(strong, macro_trend)
        f_score_ok = f_score >= macro_f_score
        breadth_ok = not np.isfinite(breadth60) or breadth60 >= breadth_threshold
        vmg_ok = np.isfinite(vmg60) and vmg60 >= vmg_threshold
        cycdef_ok = not np.isfinite(cycdef60) or cycdef60 > macro_cycdef
        ret_ok = not np.isfinite(ret120) or ret120 >= macro_ret120
        style_risk_ok = not np.isfinite(r_score) or r_score >= macro_r_score_floor
        risk_ok = not np.isfinite(r_score) or r_score >= macro_risk_floor
        vol_ok = not np.isfinite(vol_pct) or vol_pct < macro_max_vol_pct
        if trend_ok and f_score_ok and breadth_ok and vmg_ok and cycdef_ok and ret_ok and style_risk_ok and risk_ok and vol_ok:
            overlay = "macro_reflation"
    if not overlay and np.isfinite(t) and np.isfinite(r_score):
        hot_t = max(0.6, recovery)
        hot_r = 0.45 if np.isfinite(vol_pct) and vol_pct <= 0.45 else 0.7
        if t >= hot_t and r_score >= hot_r:
            if not np.isfinite(f_score) or f_score < 0.9 or r_score > f_score:
                overlay = "hot_riskon"
    if not overlay and recovery_uptrend and np.isfinite(r_score) and r_score >= 0.6 and np.isfinite(breadth60) and breadth60 >= 0.1:
        if not np.isfinite(f_score) or r_score >= f_score - 0.1:
            overlay = "hot_riskon"

    final_regime = overlay or base_regime
    if not overlay and base_regime == "range_lowvol":
        if np.isfinite(vol_pct) and vol_pct >= high and np.isfinite(t) and t <= 0:
            final_regime = "riskoff_highvol"
        elif np.isfinite(t) and t >= recovery:
            final_regime = "trend_lowvol"
    return base_regime, overlay, final_regime


def _regime_vol_family(regime: str) -> str:
    text = str(regime or "")
    if text == "riskoff_highvol":
        return "high"
    if text in {"trend_lowvol", "range_lowvol"}:
        return "low"
    return "mid"


def _current_vol_bucket(vol_pct: float, state: RegimeState, cfg: dict[str, Any]) -> str:
    if not np.isfinite(vol_pct):
        return state.last_vol_bucket or ""
    last = str(state.last_vol_bucket or "")
    high_enter = float(cfg["vol_high_enter"])
    high_exit = float(cfg["vol_high_exit"])
    low_enter = float(cfg["vol_low_enter"])
    low_exit = float(cfg["vol_low_exit"])
    if last == "high":
        if vol_pct >= high_exit:
            return "high"
        if vol_pct <= low_enter:
            return "low"
        return "mid"
    if last == "low":
        if vol_pct <= low_exit:
            return "low"
        if vol_pct >= high_enter:
            return "high"
        return "mid"
    if vol_pct >= high_enter:
        return "high"
    if vol_pct <= low_enter:
        return "low"
    return "mid"


def _apply_regime_state(
    raw_regime: str,
    overlay_regime: str,
    state: RegimeState,
    cfg: dict[str, Any],
    *,
    vol_bucket: str = "",
    features: dict[str, Any] | None = None,
) -> RegimeState:
    next_state = RegimeState(**state.to_dict())
    confirm_bars = int(cfg["switch_confirm_bars"])
    overlay_confirm = int(cfg["overlay_confirm_bars"])
    high_to_low_trend_switch_bars = int(cfg["high_to_low_trend_switch_bars"])
    trend_lowvol_exit_bars = int(cfg["trend_lowvol_exit_bars"])
    range_lowvol_exit_bars = int(cfg["range_lowvol_exit_bars"])
    if vol_bucket:
        next_state.last_vol_bucket = vol_bucket

    if overlay_regime:
        active_is_range = next_state.active_regime == "range_lowvol"
        effective_overlay_confirm = 1 if active_is_range else overlay_confirm
        if overlay_regime == "riskoff_highvol" and next_state.active_regime != "riskoff_highvol":
            effective_overlay_confirm = 1
        if overlay_regime == next_state.active_regime:
            next_state.overlay_candidate_regime = ""
            next_state.overlay_candidate_count = 0
        elif overlay_regime == next_state.overlay_candidate_regime:
            next_state.overlay_candidate_count += 1
        else:
            next_state.overlay_candidate_regime = overlay_regime
            next_state.overlay_candidate_count = 1
        if next_state.overlay_candidate_count >= effective_overlay_confirm:
            next_state.active_regime = overlay_regime
            next_state.candidate_regime = ""
            next_state.candidate_count = 0
            next_state.overlay_candidate_regime = ""
            next_state.overlay_candidate_count = 0
        return next_state

    next_state.overlay_candidate_regime = ""
    next_state.overlay_candidate_count = 0
    active_vol = _regime_vol_family(next_state.active_regime)
    candidate_vol = _regime_vol_family(next_state.candidate_regime)
    feature_map = dict(features or {})
    t = _safe_float(feature_map.get("T"))
    breadth60 = _safe_float(feature_map.get("breadth60"))
    ret120 = _safe_float(feature_map.get("ret120"))
    slope_t = _safe_float(feature_map.get("slope_t"))
    range_lowvol_trend_fast_t = float(cfg["range_lowvol_trend_fast_t_threshold"])
    range_lowvol_trend_fast_breadth = float(cfg["range_lowvol_trend_fast_breadth_threshold"])
    range_lowvol_riskoff_fast_breadth = float(cfg["range_lowvol_riskoff_fast_breadth_threshold"])
    macro_reflation_exit_bars = int(cfg["macro_reflation_exit_bars"])
    macro_reflation_exit_breadth = float(cfg["macro_reflation_exit_breadth_threshold"])
    trend_lowvol_exit = (
        next_state.active_regime == "trend_lowvol"
        and raw_regime == "range_lowvol"
        and vol_bucket in {"low", "mid"}
        and (
            (np.isfinite(t) and t < float(cfg["trend_lowvol_exit_t_threshold"]))
            or (
                np.isfinite(ret120)
                and np.isfinite(slope_t)
                and ret120 < float(cfg["trend_lowvol_exit_ret120_threshold"])
                and slope_t < float(cfg["trend_lowvol_exit_slope_t_threshold"])
            )
            or (np.isfinite(breadth60) and breadth60 <= float(cfg["trend_lowvol_exit_breadth_threshold"]))
        )
    )
    trend_lowvol_fast_exit = (
        next_state.active_regime == "trend_lowvol"
        and raw_regime == "range_lowvol"
        and (
            (np.isfinite(ret120) and ret120 < float(cfg["trend_lowvol_fast_exit_ret120_threshold"]))
            or (np.isfinite(breadth60) and breadth60 <= float(cfg["trend_lowvol_fast_exit_breadth_threshold"]))
        )
    )
    if trend_lowvol_exit:
        if next_state.candidate_regime == raw_regime:
            next_state.candidate_count += 1
        else:
            next_state.candidate_regime = raw_regime
            next_state.candidate_count = 1
        required_exit_bars = 1 if trend_lowvol_fast_exit else max(1, trend_lowvol_exit_bars)
        if next_state.candidate_count >= required_exit_bars:
            next_state.active_regime = raw_regime
            next_state.candidate_regime = ""
            next_state.candidate_count = 0
        return next_state
    range_lowvol_fast_exit = (
        next_state.active_regime == "range_lowvol"
        and raw_regime != "range_lowvol"
        and (
            (
                raw_regime == "trend_lowvol"
                and np.isfinite(t)
                and t >= range_lowvol_trend_fast_t
                and (not np.isfinite(breadth60) or breadth60 >= range_lowvol_trend_fast_breadth)
            )
            or (
                raw_regime == "riskoff_highvol"
                and (
                    vol_bucket == "high"
                    or (np.isfinite(breadth60) and breadth60 <= range_lowvol_riskoff_fast_breadth)
                )
            )
        )
    )
    if range_lowvol_fast_exit:
        if next_state.candidate_regime == raw_regime:
            next_state.candidate_count += 1
        else:
            next_state.candidate_regime = raw_regime
            next_state.candidate_count = 1
        if next_state.candidate_count >= max(1, range_lowvol_exit_bars):
            next_state.active_regime = raw_regime
            next_state.candidate_regime = ""
            next_state.candidate_count = 0
        return next_state
    macro_reflation_fast_exit = (
        next_state.active_regime == "macro_reflation"
        and raw_regime != "macro_reflation"
        and (
            raw_regime == "riskoff_highvol"
            or (
                raw_regime == "range_lowvol"
                and (not np.isfinite(breadth60) or breadth60 < macro_reflation_exit_breadth)
            )
        )
    )
    if macro_reflation_fast_exit:
        if next_state.candidate_regime == raw_regime:
            next_state.candidate_count += 1
        else:
            next_state.candidate_regime = raw_regime
            next_state.candidate_count = 1
        if next_state.candidate_count >= max(1, macro_reflation_exit_bars):
            next_state.active_regime = raw_regime
            next_state.candidate_regime = ""
            next_state.candidate_count = 0
        return next_state
    if (
        next_state.active_regime
        and vol_bucket in {"high", "low"}
        and active_vol in {"high", "low"}
        and vol_bucket != active_vol
    ):
        if candidate_vol == vol_bucket:
            next_state.candidate_regime = raw_regime
            next_state.candidate_count += 1
        else:
            next_state.candidate_regime = raw_regime
            next_state.candidate_count = 1
        switch_bars = max(2, confirm_bars - 1)
        if active_vol == "high" and vol_bucket == "low" and raw_regime == "trend_lowvol":
            switch_bars = max(1, high_to_low_trend_switch_bars)
        if next_state.candidate_count >= switch_bars:
            next_state.active_regime = raw_regime
            next_state.candidate_regime = ""
            next_state.candidate_count = 0
        return next_state
    if (
        next_state.active_regime
        and vol_bucket == "mid"
        and candidate_vol in {"high", "low"}
        and active_vol in {"high", "low"}
        and candidate_vol != active_vol
    ):
        return next_state
    if raw_regime == next_state.active_regime:
        next_state.candidate_regime = ""
        next_state.candidate_count = 0
        return next_state
    if raw_regime == next_state.candidate_regime:
        next_state.candidate_count += 1
    else:
        next_state.candidate_regime = raw_regime
        next_state.candidate_count = 1
    if next_state.candidate_count >= confirm_bars:
        next_state.active_regime = raw_regime
        next_state.candidate_regime = ""
        next_state.candidate_count = 0
    return next_state


def compute_regime_features(
    date: str | pd.Timestamp,
    market_data: JpxHistoricalProvider,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _cfg(config)
    signal_date = pd.Timestamp(date).normalize()
    z_window = int(cfg["zscore_window"])
    vol_quantile_window = int(cfg["vol_quantile_window"])
    trend_lookback = int(cfg["trend_lookback"])
    breadth_window = int(cfg["breadth_window"])
    risk_window = int(cfg["risk_window"])
    reflation_window = int(cfg["reflation_window"])
    beta_window = int(cfg["beta_window"])
    group_quantile = float(cfg["group_quantile"])
    stale_series_max_age_days = int(cfg["stale_series_max_age_days"])
    start = signal_date - pd.Timedelta(days=max(900, z_window * 4))

    universe_codes = _active_universe_codes(market_data, signal_date)
    _, close_pivot, turnover_series = _prepare_equity_panel(market_data, start, signal_date, universe_codes or None)
    market_proxy = _market_proxy_close(close_pivot)

    topix = _fresh_series_or_empty(
        _price_series(market_data, INDEX_CODE_ALIASES["topix"], start, signal_date),
        signal_date,
        stale_series_max_age_days,
    )
    market_close = topix if not topix.empty else market_proxy
    market_source = "topix" if not topix.empty else "equity_proxy"

    log_close_pivot = np.log(close_pivot) if not close_pivot.empty else pd.DataFrame()
    daily_log_returns = log_close_pivot.diff() if not log_close_pivot.empty else pd.DataFrame()
    market_log_returns = np.log(market_close).diff() if not market_close.empty else pd.Series(dtype=float)

    market_cap_table = market_data.table("market_cap", required=False, copy=False)
    market_caps = _latest_snapshot_by_code(market_cap_table, "asof_date", signal_date)
    market_caps["market_cap"] = pd.to_numeric(market_caps.get("market_cap"), errors="coerce")
    market_cap_series = market_caps.set_index("code")["market_cap"] if not market_caps.empty else pd.Series(dtype=float)

    sector_table = market_data.table("sector", required=False, copy=False)
    sector_snapshot = _latest_snapshot_by_code(sector_table, "asof_date", signal_date)
    sector_map = sector_snapshot.set_index("code")["sector"] if not sector_snapshot.empty else pd.Series(dtype=object)

    breadth60, breadth_meta = _compute_breadth_from_close_pivot(
        close_pivot,
        signal_date,
        breadth_window,
        int(cfg["breadth_min_obs"]),
    )
    ret120_series = _log_return(market_close, trend_lookback)
    slope_t_series = _rolling_slope_t(market_close, trend_lookback)
    rv20_series = _annualized_realized_vol(market_close, 20)
    rv60_series = _annualized_realized_vol(market_close, 60)
    turnover20_series = np.log(pd.to_numeric(turnover_series, errors="coerce")).diff(risk_window)

    smb20_series = pd.Series(dtype=float)
    hblv20_series = pd.Series(dtype=float)
    vmg60_series = pd.Series(dtype=float)
    cycdef60_series = pd.Series(dtype=float)
    smb20 = float("nan")
    hblv20 = float("nan")
    vmg60 = float("nan")
    cycdef60 = float("nan")

    if not close_pivot.empty:
        ret20_cross = log_close_pivot.diff(risk_window)
        ret60_cross = log_close_pivot.diff(reflation_window)
        last_row_20 = ret20_cross.iloc[-1] if not ret20_cross.empty else pd.Series(dtype=float)
        last_row_60 = ret60_cross.iloc[-1] if not ret60_cross.empty else pd.Series(dtype=float)

        if not market_cap_series.empty:
            small_codes = _latest_cross_section(market_cap_series, group_quantile, high=False)
            large_codes = _latest_cross_section(market_cap_series, group_quantile, high=True)
            smb20 = _cross_section_spread(last_row_20, small_codes, large_codes)
            valid_smb = ret20_cross.loc[:, ret20_cross.columns.intersection(small_codes.union(large_codes))].mean(axis=1, skipna=True)
            invalid_large = ret20_cross.loc[:, ret20_cross.columns.intersection(large_codes)].mean(axis=1, skipna=True)
            smb20_series = (valid_smb - invalid_large).dropna()

        if not daily_log_returns.empty and not market_log_returns.empty:
            trailing_stock = daily_log_returns.tail(max(beta_window, risk_window + 5))
            trailing_market = market_log_returns.reindex(trailing_stock.index).dropna()
            trailing_stock = trailing_stock.loc[trailing_market.index]
            if len(trailing_market) >= max(20, beta_window // 2):
                market_var = float(trailing_market.var(ddof=0))
                if np.isfinite(market_var) and market_var > 1e-12:
                    min_points = max(20, beta_window // 2)
                    valid_mask = trailing_stock.notna()
                    valid_counts = valid_mask.sum(axis=0)
                    stock_means = trailing_stock.mean(axis=0, skipna=True)
                    demeaned_stock = trailing_stock.sub(stock_means, axis=1)
                    demeaned_market = trailing_market - float(trailing_market.mean())
                    covariance_sum = demeaned_stock.mul(demeaned_market, axis=0).where(valid_mask).sum(axis=0, skipna=True)
                    beta_series = ((covariance_sum / valid_counts.where(valid_counts > 0)) / market_var).loc[valid_counts >= min_points].dropna()
                    vol_series = trailing_stock.std(axis=0, ddof=0, skipna=True).loc[valid_counts >= min_points].dropna()
                    high_beta_codes = _latest_cross_section(beta_series, group_quantile, high=True)
                    low_vol_codes = _latest_cross_section(vol_series, group_quantile, high=False)
                    hblv20 = _cross_section_spread(last_row_20, high_beta_codes, low_vol_codes)
                    if not beta_series.empty and not vol_series.empty:
                        high_beta_mean = ret20_cross.loc[:, ret20_cross.columns.intersection(high_beta_codes)].mean(axis=1, skipna=True)
                        low_vol_mean = ret20_cross.loc[:, ret20_cross.columns.intersection(low_vol_codes)].mean(axis=1, skipna=True)
                        hblv20_series = (high_beta_mean - low_vol_mean).dropna()

        if not sector_map.empty:
            vmg60 = _sector_spread(last_row_60, sector_map, VALUE_PROXY_SECTOR_NAMES, GROWTH_PROXY_SECTOR_NAMES)
            cycdef60 = _sector_spread(last_row_60, sector_map, CYCLICAL_SECTOR_NAMES, DEFENSIVE_SECTOR_NAMES)
            left_value = ret60_cross.loc[:, ret60_cross.columns.intersection(sector_map.loc[sector_map.isin(VALUE_PROXY_SECTOR_NAMES)].index.astype(str))].mean(axis=1, skipna=True)
            right_growth = ret60_cross.loc[:, ret60_cross.columns.intersection(sector_map.loc[sector_map.isin(GROWTH_PROXY_SECTOR_NAMES)].index.astype(str))].mean(axis=1, skipna=True)
            vmg60_series = (left_value - right_growth).dropna()
            cyc_mean = ret60_cross.loc[:, ret60_cross.columns.intersection(sector_map.loc[sector_map.isin(CYCLICAL_SECTOR_NAMES)].index.astype(str))].mean(axis=1, skipna=True)
            def_mean = ret60_cross.loc[:, ret60_cross.columns.intersection(sector_map.loc[sector_map.isin(DEFENSIVE_SECTOR_NAMES)].index.astype(str))].mean(axis=1, skipna=True)
            cycdef60_series = (cyc_mean - def_mean).dropna()

    ret120 = _safe_float(ret120_series.iloc[-1] if not ret120_series.empty else np.nan)
    slope_t = _safe_float(slope_t_series.iloc[-1] if not slope_t_series.empty else np.nan)
    rv20 = _safe_float(rv20_series.iloc[-1] if not rv20_series.empty else np.nan)
    rv60 = _safe_float(rv60_series.iloc[-1] if not rv60_series.empty else np.nan)
    smb20 = _safe_float(smb20 if np.isfinite(smb20) else (smb20_series.iloc[-1] if not smb20_series.empty else np.nan))
    hblv20 = _safe_float(hblv20 if np.isfinite(hblv20) else (hblv20_series.iloc[-1] if not hblv20_series.empty else np.nan))
    turnover20 = _safe_float(turnover20_series.iloc[-1] if not turnover20_series.empty else np.nan)
    vmg60 = _safe_float(vmg60 if np.isfinite(vmg60) else (vmg60_series.iloc[-1] if not vmg60_series.empty else np.nan))
    cycdef60 = _safe_float(cycdef60 if np.isfinite(cycdef60) else (cycdef60_series.iloc[-1] if not cycdef60_series.empty else np.nan))

    breadth_component = breadth60 if np.isfinite(breadth60) else float("nan")
    t_score = 0.4 * _zscore_last(ret120_series, z_window) + 0.4 * _zscore_last(slope_t_series, z_window) + 0.2 * breadth_component
    r_score = 0.35 * _zscore_last(smb20_series, z_window) + 0.35 * _zscore_last(hblv20_series, z_window) + 0.30 * _zscore_last(turnover20_series, z_window)

    if np.isfinite(cycdef60):
        f_score = 0.6 * _zscore_last(vmg60_series, z_window) + 0.4 * _zscore_last(cycdef60_series, z_window)
    else:
        f_score = _zscore_last(vmg60_series, z_window)

    vol_pct = _quantile_last(rv20_series, vol_quantile_window)
    features = {
        "signal_date": signal_date,
        "ret120": ret120,
        "slope_t": slope_t,
        "breadth60": breadth60,
        "T": _safe_float(t_score),
        "rv20": rv20,
        "rv60": rv60,
        "vol_pct": _safe_float(vol_pct),
        "smb20": smb20,
        "hblv20": hblv20,
        "turnover20": turnover20,
        "R": _safe_float(r_score),
        "vmg60": vmg60,
        "cycdef60": cycdef60,
        "F": _safe_float(f_score),
        "market_source": market_source,
        "feature_available_topix": not topix.empty,
        "feature_available_hblv20": not hblv20_series.empty or np.isfinite(hblv20),
        "feature_available_cycdef60": not cycdef60_series.empty or np.isfinite(cycdef60),
        "feature_available_vmg60": not vmg60_series.empty or np.isfinite(vmg60),
        "universe_code_count": int(len(universe_codes)),
        "price_panel_code_count": int(close_pivot.shape[1]) if not close_pivot.empty else 0,
    }
    features.update(breadth_meta)
    return features


def detect_regime_from_features(
    features: dict[str, Any],
    config: dict[str, Any] | None = None,
    state: RegimeState | None = None,
) -> dict[str, Any]:
    cfg = _cfg(config)
    current_state = state or RegimeState()
    base_regime, overlay_regime, raw_regime = _classify_unified_regime(features, cfg)
    vol_bucket = _current_vol_bucket(_safe_float(features.get("vol_pct")), current_state, cfg)
    next_state = _apply_regime_state(
        raw_regime,
        overlay_regime,
        current_state,
        cfg,
        vol_bucket=vol_bucket,
        features=features,
    )
    active_regime = next_state.active_regime or raw_regime
    return {
        "signal_date": pd.Timestamp(features.get("signal_date")).normalize(),
        "base_regime": base_regime,
        "overlay_regime": overlay_regime,
        "raw_regime": raw_regime,
        "active_regime": active_regime,
        "features": features,
        "state": next_state,
        "state_dict": next_state.to_dict(),
        "warnings": [],
    }


def detect_regime(
    date: str | pd.Timestamp,
    market_data: JpxHistoricalProvider,
    config: dict[str, Any] | None = None,
    state: RegimeState | None = None,
) -> dict[str, Any]:
    features = compute_regime_features(date, market_data, config)
    return detect_regime_from_features(features, config=config, state=state)
