"""
Runtime wrapper for the JP equity dual-MA soft trend gate.

This is a stock-level soft gate for residual momentum:

    M_gate_i = M_i * dual_ma_gate_i

The gate is mapped to [0, 1].  Missing or invalid price history defaults to
0.5, representing neutral trend information.
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
    "short_ma_window": 20,
    "long_ma_window": 60,
    "sigma_window": 120,
    "temperature": 1.5,
    "history_buffer_calendar_days": 320,
    "neutral_gate_value": 0.5,
    "price_column_preference": ["adjustment_close", "AdjustmentClose", "adjusted_close", "Adj Close", "close", "Close"],
    "save_minimal_path": None,
    "save_detail_path": None,
    "verbose": True,
}

PriceLoader = Callable[[pd.DataFrame, pd.Timestamp, pd.Timestamp, Optional[str], dict[str, Any]], pd.DataFrame]


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
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if "." in text:
        return text.upper()
    code = normalize_jp_code(text)
    return f"{code}.T" if code is not None else text.upper()


def first_existing(columns: pd.Index | list[str], candidates: list[str]) -> Optional[str]:
    lower_map = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


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
    out["ticker"] = out[ticker_col].map(to_yfinance_ticker) if ticker_col is not None else out["code"].map(to_yfinance_ticker)
    out = out.loc[out["code"].notna() & out["ticker"].notna()].drop_duplicates(subset=["code"], keep="last")
    return out[["code", "ticker"]].reset_index(drop=True)


def load_price_inputs_jquants(
    universe_codes: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    jquants_api_key: Optional[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    client = config.get("jquants_client")
    if client is not None and hasattr(client, "load_price_history"):
        return client.load_price_history(universe_codes, start_date, end_date, config=config)
    if client is not None and hasattr(client, "load_dual_ma_prices"):
        return client.load_dual_ma_prices(universe_codes, start_date, end_date, config=config)
    raise NotImplementedError(
        "J-Quants dual-MA price loader is not wired yet. Pass config['price_loader'] "
        "or config['jquants_client'] with load_price_history(...)."
    )


def normalize_price_input_frame(price_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    lower_map = {str(col).strip().lower(): col for col in price_frame.columns}

    def col(*names: str) -> Optional[str]:
        for name in names:
            found = lower_map.get(name.lower())
            if found is not None:
                return found
        return None

    code_col = col("code", "Code", "LocalCode", "stock_code")
    date_col = col("date", "Date", "signal_date")
    price_col = None
    for candidate in list(config["price_column_preference"]):
        found = first_existing(price_frame.columns, [candidate])
        if found is None:
            continue
        values = pd.to_numeric(price_frame[found], errors="coerce")
        if values.notna().any():
            price_col = found
            break
    missing = [
        name
        for name, value in [("code", code_col), ("date", date_col), ("price", price_col)]
        if value is None
    ]
    if missing:
        raise ValueError(f"Dual-MA price input frame is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "code": price_frame[code_col].map(normalize_jp_code),
            "date": pd.to_datetime(price_frame[date_col], errors="coerce"),
            "close": pd.to_numeric(price_frame[price_col], errors="coerce"),
            "price_column_used": price_col,
        }
    )
    return out.dropna(subset=["code", "date"]).sort_values(["code", "date"])


def compute_one_dual_ma_gate(
    code: str,
    ticker: str,
    prices: pd.DataFrame,
    asof_date: pd.Timestamp,
    rebalance_date: pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, Any]:
    short_window = int(config["short_ma_window"])
    long_window = int(config["long_ma_window"])
    sigma_window = int(config["sigma_window"])
    temperature = float(config["temperature"])
    neutral = float(config["neutral_gate_value"])
    required_obs = max(short_window, long_window) + sigma_window

    base = {
        "code": code,
        "ticker": ticker,
        "status": "ok",
        "rebalance_date": rebalance_date,
        "asof_date": asof_date,
        "close": np.nan,
        "ma_short": np.nan,
        "ma_long": np.nan,
        "ma_spread_pct": np.nan,
        "ma_spread_sigma": np.nan,
        "trend_z": np.nan,
        "dual_ma_gate_raw": np.nan,
        "dual_ma_gate": neutral,
        "signal_date": pd.NaT,
        "data_end_date": pd.NaT,
        "available_price_obs": 0,
        "required_price_obs": int(required_obs),
        "price_column_used": pd.NA,
    }

    if prices is None or prices.empty:
        base["status"] = "missing_price"
        return base

    work = prices.loc[(prices["code"].eq(code)) & (prices["date"] <= asof_date)].copy()
    work = work.dropna(subset=["close"]).sort_values("date")
    base["available_price_obs"] = int(len(work))
    if work.empty:
        base["status"] = "missing_price"
        return base
    if len(work) < required_obs:
        base["status"] = "insufficient_history"
        base["signal_date"] = work["date"].iloc[-1]
        base["data_end_date"] = work["date"].iloc[-1]
        base["close"] = float(work["close"].iloc[-1])
        base["price_column_used"] = work["price_column_used"].iloc[-1] if "price_column_used" in work else pd.NA
        return base

    close = pd.Series(work["close"].to_numpy(dtype=float), index=pd.to_datetime(work["date"]))
    ma_short = close.rolling(short_window).mean()
    ma_long = close.rolling(long_window).mean()
    spread_pct = ma_short / ma_long - 1.0
    spread_sigma = spread_pct.rolling(sigma_window).std(ddof=0)

    latest_date = close.index[-1]
    latest_close = close.iloc[-1]
    latest_short = ma_short.iloc[-1]
    latest_long = ma_long.iloc[-1]
    latest_spread = spread_pct.iloc[-1]
    latest_sigma = spread_sigma.iloc[-1]

    base.update(
        {
            "signal_date": latest_date,
            "data_end_date": latest_date,
            "close": float(latest_close) if pd.notna(latest_close) else np.nan,
            "ma_short": float(latest_short) if pd.notna(latest_short) else np.nan,
            "ma_long": float(latest_long) if pd.notna(latest_long) else np.nan,
            "ma_spread_pct": float(latest_spread) if pd.notna(latest_spread) else np.nan,
            "ma_spread_sigma": float(latest_sigma) if pd.notna(latest_sigma) else np.nan,
            "price_column_used": work["price_column_used"].iloc[-1] if "price_column_used" in work else pd.NA,
        }
    )

    if pd.isna(latest_close):
        base["status"] = "missing_price"
        return base
    if pd.isna(latest_short) or pd.isna(latest_long) or pd.isna(latest_spread):
        base["status"] = "invalid_ma"
        return base
    if pd.isna(latest_sigma) or latest_sigma == 0:
        base["status"] = "invalid_sigma"
        return base

    trend_z = latest_spread / latest_sigma
    gate_raw = 0.5 * (1.0 + np.tanh(trend_z / temperature))
    gate = float(np.clip(gate_raw, 0.0, 1.0))
    base.update(
        {
            "trend_z": float(trend_z),
            "dual_ma_gate_raw": float(gate_raw),
            "dual_ma_gate": gate,
        }
    )
    return base


def compute_dual_ma_detail(
    universe_codes: pd.DataFrame,
    price_frame: pd.DataFrame,
    asof_date: pd.Timestamp,
    rebalance_date: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    prices = normalize_price_input_frame(price_frame, config)
    rows = [
        compute_one_dual_ma_gate(
            code=row["code"],
            ticker=row["ticker"],
            prices=prices,
            asof_date=asof_date,
            rebalance_date=rebalance_date,
            config=config,
        )
        for _, row in universe_codes.iterrows()
    ]
    return pd.DataFrame(rows)


def build_dual_ma_minimal(detail: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": detail["code"],
            "factor_name": "dual_ma",
            "factor_value": detail["dual_ma_gate"],
            "signal_date": pd.to_datetime(detail["signal_date"], errors="coerce"),
            "data_end_date": pd.to_datetime(detail["data_end_date"], errors="coerce"),
            "rebalance_date": rebalance_date,
            "dual_ma_gate": detail["dual_ma_gate"],
        }
    )


def run_dual_ma_factor(
    universe: pd.DataFrame | str | Path,
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    asof_date = pd.Timestamp(cfg.get("asof_date", rebalance_date)).normalize()
    start_date = asof_date - pd.Timedelta(days=int(cfg["history_buffer_calendar_days"]))
    universe_codes = extract_universe_codes(universe)

    loader: PriceLoader = cfg.get("price_loader") or load_price_inputs_jquants
    price_frame = loader(universe_codes, start_date, asof_date, jquants_api_key, cfg)
    detail = compute_dual_ma_detail(universe_codes, price_frame, asof_date, rebalance_ts, cfg)
    minimal = build_dual_ma_minimal(detail, rebalance_ts)

    status_counts = detail["status"].value_counts(dropna=False).to_dict() if "status" in detail else {}
    neutral_rows = int(detail["dual_ma_gate_raw"].isna().sum()) if "dual_ma_gate_raw" in detail else 0
    summary = {
        "factor_name": "dual_ma",
        "rebalance_date": rebalance_ts,
        "asof_date": asof_date,
        "input_universe_rows": int(len(coerce_universe_frame(universe))),
        "usable_universe_rows": int(len(universe_codes)),
        "detail_rows": int(len(detail)),
        "minimal_rows": int(len(minimal)),
        "gate_mean": float(detail["dual_ma_gate"].mean()) if len(detail) else np.nan,
        "gate_median": float(detail["dual_ma_gate"].median()) if len(detail) else np.nan,
        "gate_min": float(detail["dual_ma_gate"].min()) if len(detail) else np.nan,
        "gate_max": float(detail["dual_ma_gate"].max()) if len(detail) else np.nan,
        "neutral_default_rows": neutral_rows,
        "missing_raw_gate_rows": neutral_rows,
        "status_counts": status_counts,
        "short_ma_window": int(cfg["short_ma_window"]),
        "long_ma_window": int(cfg["long_ma_window"]),
        "sigma_window": int(cfg["sigma_window"]),
        "temperature": float(cfg["temperature"]),
        "data_source": str(cfg.get("data_source", "jquants")),
        "loader_mode": "price_loader" if cfg.get("price_loader") is not None else ("jquants_client" if cfg.get("jquants_client") is not None else "unwired_jquants"),
        "jquants_api_key_supplied": bool(jquants_api_key),
    }

    if cfg.get("save_detail_path"):
        detail.to_csv(cfg["save_detail_path"], index=False, encoding="utf-8-sig")
    if cfg.get("save_minimal_path"):
        minimal.to_csv(cfg["save_minimal_path"], index=False, encoding="utf-8-sig")

    return {"minimal": minimal, "detail": detail, "summary": summary}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate JP equity dual-MA soft trend gate.")
    parser.add_argument("--input", required=True, help="Universe CSV path.")
    parser.add_argument("--rebalance-date", required=True, help="Rebalance/as-of date.")
    parser.add_argument("--detail-output", default=None, help="Optional detail output CSV path.")
    parser.add_argument("--minimal-output", default=None, help="Optional minimal output CSV path.")
    args = parser.parse_args()

    run_dual_ma_factor(
        universe=args.input,
        rebalance_date=args.rebalance_date,
        config={
            "save_detail_path": args.detail_output,
            "save_minimal_path": args.minimal_output,
        },
    )
