"""
Runtime wrapper for the JP equity behaviour / credit crowding factor.

Behaviour requires margin/credit data. There is intentionally no yfinance
fallback here: pass config["behaviour_loader"] or config["jquants_client"] so
the local backtest console can provide J-Quants-backed inputs via
jquants_api_key.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


CODE_COLUMN_CANDIDATES = ["code", "Code", "LocalCode", "stock_code", "StockCode"]
TICKER_COLUMN_CANDIDATES = ["ticker", "Ticker", "symbol", "Symbol", "code", "Code"]

DEFAULT_CONFIG: dict[str, Any] = {
    "data_source": "jquants",
    "lookback_weeks": 26,
    "winsor_lower_q": 0.02,
    "winsor_upper_q": 0.98,
    "stale_days_limit": 14,
    "save_minimal_path": None,
    "save_detail_path": None,
    "verbose": True,
}

BehaviourInputLoader = Callable[[pd.DataFrame, pd.Timestamp, pd.Timestamp, Optional[str], dict[str, Any]], pd.DataFrame]


def normalize_jp_code(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"(\d{4})", text)
    return match.group(1) if match else None


def to_yfinance_ticker(value: object) -> Optional[str]:
    code = normalize_jp_code(value)
    return f"{code}.T" if code is not None else None


def first_existing(columns: pd.Index | list[str], candidates: list[str]) -> Optional[str]:
    lower_map = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def zscore_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - valid.mean()) / std


def winsorize_series(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return values
    return values.clip(valid.quantile(lower_q), valid.quantile(upper_q))


def numeric_detail_series(detail: pd.DataFrame, column: str) -> pd.Series:
    if column in detail.columns:
        return pd.to_numeric(detail[column], errors="coerce")
    return pd.Series(np.nan, index=detail.index, dtype=float)


def coerce_universe_frame(universe: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(universe, pd.DataFrame):
        return universe.copy()
    return pd.read_csv(universe)


def extract_universe_codes(universe: pd.DataFrame | str | Path) -> pd.DataFrame:
    frame = coerce_universe_frame(universe)
    code_col = first_existing(frame.columns, CODE_COLUMN_CANDIDATES)
    ticker_col = first_existing(frame.columns, TICKER_COLUMN_CANDIDATES)
    if code_col is None and ticker_col is None:
        raise ValueError("Universe must contain code or ticker.")
    out = frame.copy()
    out["code"] = out[code_col].map(normalize_jp_code) if code_col is not None else out[ticker_col].map(normalize_jp_code)
    out["ticker"] = out["code"].map(to_yfinance_ticker)
    out = out.loc[out["code"].notna()].drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return out


def load_behaviour_inputs_jquants(
    universe_codes: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    jquants_api_key: Optional[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    client = config.get("jquants_client")
    if client is not None and hasattr(client, "load_behaviour_inputs"):
        return client.load_behaviour_inputs(universe_codes, start_date, end_date, config=config)
    raise NotImplementedError(
        "Behaviour factor requires J-Quants credit/margin data. Pass "
        "config['behaviour_loader'] or config['jquants_client'] with "
        "load_behaviour_inputs(...)."
    )


def normalize_behaviour_input_frame(inputs: pd.DataFrame) -> pd.DataFrame:
    lower_map = {str(col).strip().lower(): col for col in inputs.columns}

    def col(*names: str) -> Optional[str]:
        for name in names:
            found = lower_map.get(name.lower())
            if found is not None:
                return found
        return None

    code_col = col("code", "Code", "LocalCode", "stock_code")
    date_col = col("date", "Date", "credit_data_date", "PublishedDate")
    buy_balance_col = col("margin_buy_balance", "MarginBuyBalance", "long_margin_balance", "credit_buy_balance")
    sell_balance_col = col("margin_sell_balance", "MarginSellBalance", "short_margin_balance", "credit_sell_balance")
    buy_change_col = col("margin_buy_change", "MarginBuyChange", "long_margin_change", "credit_buy_change")
    sell_change_col = col("margin_sell_change", "MarginSellChange", "short_margin_change", "credit_sell_change")
    market_cap_col = col("market_cap_jpy", "MarketCap")
    adv_col = col("avg_daily_trading_value", "AvgDailyTradingValue")

    missing = [
        name
        for name, value in [
            ("code", code_col),
            ("date", date_col),
            ("margin_buy_balance", buy_balance_col),
            ("margin_sell_balance", sell_balance_col),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(f"Behaviour input frame is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "code": inputs[code_col].map(normalize_jp_code),
            "date": pd.to_datetime(inputs[date_col], errors="coerce"),
            "margin_buy_balance": pd.to_numeric(inputs[buy_balance_col], errors="coerce"),
            "margin_sell_balance": pd.to_numeric(inputs[sell_balance_col], errors="coerce"),
            "margin_buy_change": pd.to_numeric(inputs[buy_change_col], errors="coerce") if buy_change_col else np.nan,
            "margin_sell_change": pd.to_numeric(inputs[sell_change_col], errors="coerce") if sell_change_col else np.nan,
            "market_cap_jpy": pd.to_numeric(inputs[market_cap_col], errors="coerce") if market_cap_col else np.nan,
            "avg_daily_trading_value": pd.to_numeric(inputs[adv_col], errors="coerce") if adv_col else np.nan,
        }
    )
    return out.dropna(subset=["code", "date"])


def compute_latest_behaviour_rows(
    universe_codes: pd.DataFrame,
    input_frame: pd.DataFrame,
    asof_date: pd.Timestamp,
    rebalance_date: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    inputs = normalize_behaviour_input_frame(input_frame)
    inputs = inputs.loc[inputs["date"] <= asof_date].sort_values(["code", "date"])
    latest = inputs.groupby("code", as_index=False).tail(1).copy()

    rows = []
    for _, item in universe_codes.iterrows():
        code = item["code"]
        ticker = item["ticker"]
        hit = latest.loc[latest["code"].eq(code)]
        if hit.empty:
            rows.append(
                {
                    "code": code,
                    "ticker": ticker,
                    "status": "missing_credit_input",
                    "rebalance_date": rebalance_date,
                    "asof_date": asof_date,
                }
            )
            continue
        row = hit.iloc[0].to_dict()
        row.update({"ticker": ticker, "status": "ok", "rebalance_date": rebalance_date, "asof_date": asof_date})
        rows.append(row)

    detail = pd.DataFrame(rows)
    if "date" in detail.columns:
        detail["credit_data_date"] = pd.to_datetime(detail["date"], errors="coerce")
    else:
        detail["credit_data_date"] = pd.NaT

    buy = numeric_detail_series(detail, "margin_buy_balance")
    sell = numeric_detail_series(detail, "margin_sell_balance")
    buy_chg = numeric_detail_series(detail, "margin_buy_change")
    sell_chg = numeric_detail_series(detail, "margin_sell_change")
    mcap = numeric_detail_series(detail, "market_cap_jpy")
    adv = numeric_detail_series(detail, "avg_daily_trading_value")

    detail["credit_pressure"] = np.log1p(buy.clip(lower=0)) - np.log1p(sell.clip(lower=0))
    detail["credit_change"] = zscore_series(buy_chg) - zscore_series(sell_chg)
    detail["margin_buy_to_mcap"] = buy / mcap.replace(0, np.nan)
    detail["margin_sell_to_mcap"] = sell / mcap.replace(0, np.nan)
    detail["margin_buy_to_adv"] = buy / adv.replace(0, np.nan)

    raw_components = [
        zscore_series(winsorize_series(detail["credit_pressure"], float(config["winsor_lower_q"]), float(config["winsor_upper_q"]))),
        zscore_series(detail["credit_change"]),
    ]
    if detail["margin_buy_to_mcap"].notna().sum() >= 2:
        raw_components.append(zscore_series(detail["margin_buy_to_mcap"]))
    detail["credit_raw"] = pd.concat(raw_components, axis=1).mean(axis=1, skipna=True)
    detail["credit_residual_factor"] = zscore_series(detail["credit_raw"])
    detail.loc[~detail["status"].eq("ok"), "credit_residual_factor"] = np.nan
    detail["signal_date"] = detail["credit_data_date"]
    detail["data_end_date"] = detail["credit_data_date"]
    detail["lag_days"] = (rebalance_date - detail["data_end_date"]).dt.days
    detail["stale_flag"] = detail["lag_days"].gt(int(config["stale_days_limit"])).fillna(False).astype(int)
    return detail


def build_behaviour_minimal(detail: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": detail["code"],
            "factor_name": "behaviour",
            "factor_value": detail["credit_residual_factor"],
            "signal_date": pd.to_datetime(detail["signal_date"], errors="coerce"),
            "data_end_date": pd.to_datetime(detail["data_end_date"], errors="coerce"),
            "rebalance_date": rebalance_date,
            "credit_residual_factor": detail["credit_residual_factor"],
        }
    )


def run_behaviour_factor(
    universe: pd.DataFrame | str | Path,
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    asof_date = pd.Timestamp(cfg.get("asof_date", rebalance_date)).normalize()
    start_date = asof_date - pd.Timedelta(days=int(cfg["lookback_weeks"]) * 7 + 14)
    universe_codes = extract_universe_codes(universe)

    loader: BehaviourInputLoader = cfg.get("behaviour_loader") or load_behaviour_inputs_jquants
    input_frame = loader(universe_codes, start_date, asof_date, jquants_api_key, cfg)
    detail = compute_latest_behaviour_rows(universe_codes, input_frame, asof_date, rebalance_ts, cfg)
    minimal = build_behaviour_minimal(detail, rebalance_ts)

    summary = {
        "factor_name": "behaviour",
        "rebalance_date": rebalance_ts,
        "asof_date": asof_date,
        "input_universe_rows": int(len(coerce_universe_frame(universe))),
        "usable_universe_rows": int(len(universe_codes)),
        "detail_rows": int(len(detail)),
        "minimal_rows": int(len(minimal)),
        "ok_rows": int(detail["status"].eq("ok").sum()) if "status" in detail else 0,
        "missing_factor_rows": int(minimal["factor_value"].isna().sum()),
        "stale_rows": int(detail.get("stale_flag", pd.Series(dtype=int)).sum()),
        "max_lag_days": float(pd.to_numeric(detail.get("lag_days"), errors="coerce").max()) if "lag_days" in detail else np.nan,
        "median_lag_days": float(pd.to_numeric(detail.get("lag_days"), errors="coerce").median()) if "lag_days" in detail else np.nan,
        "data_source": str(cfg.get("data_source", "jquants")),
        "loader_mode": "behaviour_loader" if cfg.get("behaviour_loader") is not None else ("jquants_client" if cfg.get("jquants_client") is not None else "unwired_jquants"),
        "jquants_api_key_supplied": bool(jquants_api_key),
    }

    if cfg.get("save_detail_path"):
        detail.to_csv(cfg["save_detail_path"], index=False, encoding="utf-8-sig")
    if cfg.get("save_minimal_path"):
        minimal.to_csv(cfg["save_minimal_path"], index=False, encoding="utf-8-sig")

    return {"minimal": minimal, "detail": detail, "summary": summary}
