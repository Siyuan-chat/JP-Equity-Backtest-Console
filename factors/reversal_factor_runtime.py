"""
Runtime wrapper for the JP equity short-term reversal factor.

This module is intentionally small and close to reverse.ipynb. It preserves
the notebook's skip-5 / 20-day reversal calculation while adding a stable
function entry point and a minimal/detail output contract for orchestration
from the local GUI and CLI workflows in this repository.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


CODE_COLUMN_CANDIDATES = [
    "code",
    "Code",
    "ticker",
    "Ticker",
    "symbol",
    "stock_code",
    "StockCode",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "skip_recent_days": 5,
    "lookback_window": 20,
    "history_buffer_calendar_days": 140,
    "sleep_seconds": 0.0,
    "save_minimal_path": None,
    "save_detail_path": None,
    "verbose": True,
}

PriceLoader = Callable[..., pd.DataFrame]


def normalize_jp_code(value: object) -> Optional[str]:
    """Normalize JP equity identifiers to a 4-digit code when possible."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 4:
        return digits[:4]
    return text.upper()


def find_code_column(df: pd.DataFrame) -> str:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in CODE_COLUMN_CANDIDATES:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return str(df.columns[0])


def coerce_universe_frame(universe: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(universe, pd.DataFrame):
        return universe.copy()
    return pd.read_csv(universe)


def extract_universe_codes(universe: pd.DataFrame | str | Path) -> pd.DataFrame:
    frame = coerce_universe_frame(universe)
    code_col = find_code_column(frame)
    out = frame.copy()
    out["code"] = out[code_col].map(normalize_jp_code)
    out = out.loc[out["code"].notna()].copy()
    out = out.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return out


def zscore_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - valid.mean()) / std


def _flatten_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        out = frame.copy()
        out.columns = [col[0] if isinstance(col, tuple) else col for col in out.columns]
        return out
    return frame


def _normalize_panel_price_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    if price_frame is None or price_frame.empty:
        return pd.DataFrame(columns=["code", "Date", "Open", "High", "Low", "Close", "Volume"])
    work = price_frame.copy()
    rename_map = {}
    for col in work.columns:
        text = str(col).strip()
        lower = text.lower()
        if lower == "date":
            rename_map[col] = "Date"
        elif lower == "code":
            rename_map[col] = "code"
        elif lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower == "close":
            rename_map[col] = "Close"
        elif lower == "volume":
            rename_map[col] = "Volume"
    work = work.rename(columns=rename_map)
    if "Date" not in work.columns or "code" not in work.columns:
        return pd.DataFrame(columns=["code", "Date", "Open", "High", "Low", "Close", "Volume"])
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.tz_localize(None)
    work["code"] = work["code"].map(normalize_jp_code)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    keep = [col for col in ["code", "Date", "Open", "High", "Low", "Close", "Volume"] if col in work.columns]
    return work.loc[work["Date"].notna() & work["code"].notna(), keep].sort_values(["code", "Date"]).reset_index(drop=True)


def compute_reversal_from_panel(
    universe_codes: pd.DataFrame,
    price_frame: pd.DataFrame,
    *,
    rebalance_ts: pd.Timestamp,
    asof_date: pd.Timestamp,
    skip_recent_days: int,
    lookback_window: int,
    sleep_seconds: float,
    verbose: bool,
) -> pd.DataFrame:
    normalized = _normalize_panel_price_frame(price_frame)
    grouped = {str(code): group for code, group in normalized.groupby("code", sort=False)} if not normalized.empty else {}
    rows: list[dict[str, Any]] = []
    for _, item in universe_codes.iterrows():
        code = item["code"]
        if verbose:
            print(f"Processing reversal factor: {code}", flush=True)
        hist = grouped.get(str(code), pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"]))
        try:
            metrics = calc_skip_reversal_detail(
                hist,
                skip_recent_days=skip_recent_days,
                lookback_window=lookback_window,
                asof_date=asof_date,
            )
        except Exception as exc:  # pragma: no cover - depends on data source
            metrics = {
                "reversal_factor": np.nan,
                "status": "download_or_calculation_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        rows.append(
            {
                "code": code,
                "stock_code": code,
                "rebalance_date": rebalance_ts,
                "asof_date": asof_date,
                **metrics,
            }
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows)


def calc_skip_reversal_detail(
    hist: pd.DataFrame,
    skip_recent_days: int = 5,
    lookback_window: int = 20,
    asof_date: Optional[pd.Timestamp] = None,
) -> dict[str, Any]:
    if hist is None or hist.empty:
        return {"reversal_factor": np.nan, "status": "empty_history"}

    work = _flatten_price_columns(hist).copy()
    if "Close" not in work.columns:
        return {"reversal_factor": np.nan, "status": "missing_close"}

    if not isinstance(work.index, pd.DatetimeIndex):
        date_col = "Date" if "Date" in work.columns else None
        if date_col is None:
            return {"reversal_factor": np.nan, "status": "missing_date_index"}
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.set_index(date_col)

    work.index = pd.to_datetime(work.index, errors="coerce").tz_localize(None)
    work = work.loc[work.index.notna()].sort_index()
    if asof_date is not None:
        work = work.loc[work.index <= asof_date]

    close = pd.to_numeric(work["Close"], errors="coerce").dropna()
    required_len = skip_recent_days + lookback_window + 1
    if len(close) < required_len:
        return {
            "reversal_factor": np.nan,
            "status": "insufficient_history",
            "available_close_count": int(len(close)),
            "required_close_count": int(required_len),
            "data_end_date": close.index.max() if len(close) else pd.NaT,
        }

    end_idx = -(skip_recent_days + 1)
    start_idx = -(skip_recent_days + lookback_window + 1)
    end_price = close.iloc[end_idx]
    start_price = close.iloc[start_idx]
    end_price_date = close.index[end_idx]
    start_price_date = close.index[start_idx]

    if pd.isna(end_price) or pd.isna(start_price) or start_price == 0:
        return {
            "reversal_factor": np.nan,
            "status": "invalid_price",
            "start_price": start_price,
            "end_price": end_price,
            "start_price_date": start_price_date,
            "end_price_date": end_price_date,
            "data_end_date": close.index.max(),
        }

    past_return = end_price / start_price - 1.0
    return {
        "reversal_factor": -past_return,
        "past_return": past_return,
        "status": "ok",
        "start_price": start_price,
        "end_price": end_price,
        "start_price_date": start_price_date,
        "end_price_date": end_price_date,
        "data_end_date": close.index.max(),
        "available_close_count": int(len(close)),
        "required_close_count": int(required_len),
    }


def build_reversal_minimal(
    detail: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    factor_name: str = "reversal",
) -> pd.DataFrame:
    minimal = pd.DataFrame(
        {
            "code": detail["code"],
            "factor_name": factor_name,
            "factor_value": detail["reversal_factor_zscore"],
            "signal_date": pd.to_datetime(detail["signal_date"], errors="coerce"),
            "data_end_date": pd.to_datetime(detail["data_end_date"], errors="coerce"),
            "freshness_date": pd.to_datetime(detail["data_end_date"], errors="coerce"),
            "rebalance_date": rebalance_date,
            "reversal_factor_zscore": detail["reversal_factor_zscore"],
        }
    )
    return minimal


def run_reversal_factor(
    universe: pd.DataFrame | str | Path,
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Calculate the short-term reversal factor for one rebalance date.

    Parameters follow the shared factor-function contract.  Historical runs
    must inject config["price_loader"].  The local/J-Quants path uses plain
    security codes such as 7203; this module no longer creates external
    ticker symbols in the main data flow.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    asof_date = pd.Timestamp(cfg.get("asof_date", rebalance_date)).tz_localize(None)
    rebalance_ts = pd.Timestamp(rebalance_date).tz_localize(None)
    start_date = asof_date - pd.Timedelta(days=int(cfg["history_buffer_calendar_days"]))
    end_date = asof_date
    skip_recent_days = int(cfg["skip_recent_days"])
    lookback_window = int(cfg["lookback_window"])
    sleep_seconds = float(cfg["sleep_seconds"])
    verbose = bool(cfg["verbose"])
    price_loader: PriceLoader | None = cfg.get("price_loader")
    panel_price_loader: PriceLoader | None = cfg.get("panel_price_loader")
    if price_loader is None:
        price_loader = panel_price_loader
    if price_loader is None and panel_price_loader is None:
        raise ValueError("price_loader is required for reversal runs; yfinance fallback is disabled.")

    universe_codes = extract_universe_codes(universe)
    detail = pd.DataFrame()
    if panel_price_loader is not None:
        try:
            if verbose:
                print(f"Loading reversal price panel for {len(universe_codes)} codes", flush=True)
            panel = panel_price_loader(universe_codes, start_date, end_date, jquants_api_key, cfg)
            detail = compute_reversal_from_panel(
                universe_codes,
                panel,
                rebalance_ts=rebalance_ts,
                asof_date=asof_date,
                skip_recent_days=skip_recent_days,
                lookback_window=lookback_window,
                sleep_seconds=sleep_seconds,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"Reversal panel loader failed, falling back to single-name loader: {exc!r}", flush=True)

    if detail.empty:
        rows: list[dict[str, Any]] = []
        for _, item in universe_codes.iterrows():
            code = item["code"]
            if verbose:
                print(f"Processing reversal factor: {code}", flush=True)

            try:
                hist = price_loader(code, start_date, end_date, cfg)
                metrics = calc_skip_reversal_detail(
                    hist,
                    skip_recent_days=skip_recent_days,
                    lookback_window=lookback_window,
                    asof_date=asof_date,
                )
            except Exception as exc:  # pragma: no cover - depends on data source
                metrics = {
                    "reversal_factor": np.nan,
                    "status": "download_or_calculation_failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

            row = {
                "code": code,
                "stock_code": code,
                "rebalance_date": rebalance_ts,
                "asof_date": asof_date,
                **metrics,
            }
            rows.append(row)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        detail = pd.DataFrame(rows)
    if detail.empty:
        detail = pd.DataFrame(
            columns=[
                "code",
                "stock_code",
                "rebalance_date",
                "asof_date",
                "reversal_factor",
                "status",
            ]
        )

    detail["reversal_factor"] = pd.to_numeric(detail["reversal_factor"], errors="coerce")
    detail["reversal_factor_zscore"] = zscore_series(detail["reversal_factor"])
    detail["signal_date"] = pd.to_datetime(
        detail.get("end_price_date", pd.Series(pd.NaT, index=detail.index)),
        errors="coerce",
    )
    detail["data_end_date"] = pd.to_datetime(
        detail.get("data_end_date", pd.Series(pd.NaT, index=detail.index)),
        errors="coerce",
    )

    minimal = build_reversal_minimal(detail, rebalance_date=rebalance_ts)
    summary = {
        "factor_name": "reversal",
        "rebalance_date": rebalance_ts,
        "asof_date": asof_date,
        "input_universe_rows": int(len(coerce_universe_frame(universe))),
        "usable_universe_rows": int(len(universe_codes)),
        "detail_rows": int(len(detail)),
        "minimal_rows": int(len(minimal)),
        "ok_rows": int((detail.get("status") == "ok").sum()) if "status" in detail else 0,
        "missing_factor_rows": int(detail["reversal_factor_zscore"].isna().sum()),
        "skip_recent_days": skip_recent_days,
        "lookback_window": lookback_window,
        "data_source": "custom_price_loader" if config and config.get("price_loader") else "missing_price_loader",
        "jquants_api_key_supplied": bool(jquants_api_key),
    }

    if cfg.get("save_detail_path"):
        detail.to_csv(cfg["save_detail_path"], index=False, encoding="utf-8-sig")
    if cfg.get("save_minimal_path"):
        minimal.to_csv(cfg["save_minimal_path"], index=False, encoding="utf-8-sig")

    return {"minimal": minimal, "detail": detail, "summary": summary}


def run_from_csv(
    input_csv: str | Path,
    output_csv: str | Path = "reversal_factor_output.csv",
    rebalance_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compatibility helper that behaves like the original notebook CSV flow."""
    anchor = pd.Timestamp.today().normalize() if rebalance_date is None else pd.Timestamp(rebalance_date)
    result = run_reversal_factor(
        universe=input_csv,
        rebalance_date=anchor,
        config={"save_detail_path": output_csv},
    )
    return result["detail"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate JP equity reversal factor.")
    parser.add_argument("--input", default="stock_pool.csv", help="Universe CSV path.")
    parser.add_argument("--output", default="reversal_factor_output.csv", help="Detail output CSV path.")
    parser.add_argument("--minimal-output", default=None, help="Optional minimal output CSV path.")
    parser.add_argument("--rebalance-date", required=True, help="Rebalance/as-of date, e.g. 2026-03-17.")
    args = parser.parse_args()

    run_reversal_factor(
        universe=args.input,
        rebalance_date=args.rebalance_date,
        config={
            "save_detail_path": args.output,
            "save_minimal_path": args.minimal_output,
        },
    )
