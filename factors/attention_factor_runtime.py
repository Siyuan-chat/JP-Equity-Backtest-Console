"""
Runtime wrapper for the JP equity attention / crowding factor.

This module is a thin extraction from attention.ipynb.  It keeps the original
logic based on abnormal monthly volume and turnover, while adding the standard
factor entry point and minimal/detail outputs for a main Colab control notebook.
"""

from __future__ import annotations

import re
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
    "history_buffer_calendar_days": 520,
    "winsor_lower_q": 0.05,
    "winsor_upper_q": 0.95,
    "save_minimal_path": None,
    "save_detail_path": None,
    "verbose": True,
}

PriceLoader = Callable[[str, pd.Timestamp, pd.Timestamp, dict[str, Any]], pd.DataFrame]
SharesLoader = Callable[[str, dict[str, Any]], float]
AttentionInputLoader = Callable[[pd.DataFrame, pd.Timestamp, pd.Timestamp, Optional[str], dict[str, Any]], pd.DataFrame]


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


def to_yfinance_ticker(code: object) -> Optional[str]:
    normalized = normalize_jp_code(code)
    if normalized is None:
        return None
    if "." in normalized:
        return normalized.upper()
    if normalized.isdigit():
        return f"{normalized}.T"
    return normalized.upper()


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
    out["yf_ticker"] = out["code"].map(to_yfinance_ticker)
    out = out.loc[out["code"].notna() & out["yf_ticker"].notna()].copy()
    out = out.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return out


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        out = frame.copy()
        out.columns = [col[0] if isinstance(col, tuple) else col for col in out.columns]
        return out
    return frame


def download_price_history_yfinance(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("yfinance is required unless config['price_loader'] is supplied.") from exc

    hist = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _flatten_yfinance_columns(hist)


def get_shares_outstanding_yfinance(ticker: str, config: dict[str, Any]) -> float:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("yfinance is required unless config['shares_loader'] is supplied.") from exc

    stock = yf.Ticker(ticker)

    try:
        fast_info = stock.fast_info
        if fast_info is not None:
            shares = fast_info.get("shares")
            if isinstance(shares, (int, float)) and shares > 0:
                return float(shares)
    except Exception:
        pass

    try:
        info = stock.info
        shares = info.get("sharesOutstanding")
        if isinstance(shares, (int, float)) and shares > 0:
            return float(shares)
    except Exception:
        pass

    return np.nan


def winsorize_series(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return values
    lower = valid.quantile(lower_q)
    upper = valid.quantile(upper_q)
    return values.clip(lower=lower, upper=upper)


def zscore_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - valid.mean()) / std


def _prepare_price_frame(price_df: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = _flatten_yfinance_columns(price_df).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = "Date" if "Date" in df.columns else None
        if date_col is None:
            return pd.DataFrame()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    df = df.loc[df.index.notna()].sort_index()
    df = df.loc[df.index <= asof_date]
    df = df.reset_index().rename(columns={"index": "Date"})

    required = {"Date", "Close", "Volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df[["Date", "Close", "Volume"]].copy()


def compute_attention_metrics(
    price_df: pd.DataFrame,
    shares_outstanding: float,
    asof_date: pd.Timestamp,
) -> dict[str, Any]:
    df = _prepare_price_frame(price_df, asof_date)
    if df.empty:
        return {"status": "missing_price_history"}

    df = df.dropna(subset=["Date", "Close", "Volume"]).copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.loc[df["Volume"] > 0].copy()
    if df.empty:
        return {"status": "invalid_volume"}

    df["month"] = df["Date"].dt.to_period("M")
    df["turnover_value_daily"] = df["Close"] * df["Volume"]
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            monthly_volume=("Volume", "sum"),
            monthly_turnover_value=("turnover_value_daily", "sum"),
            data_end_date=("Date", "max"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    if len(monthly) < 13:
        return {
            "status": "insufficient_monthly_history",
            "available_months": int(len(monthly)),
            "required_months": 13,
            "data_end_date": df["Date"].max(),
        }

    current = monthly.iloc[-1]
    prev12 = monthly.iloc[-13:-1]
    avg_prev12_volume = prev12["monthly_volume"].mean()

    if pd.isna(avg_prev12_volume) or avg_prev12_volume <= 0:
        return {
            "status": "invalid_prev12_volume",
            "available_months": int(len(monthly)),
            "data_end_date": current["data_end_date"],
        }

    avol = current["monthly_volume"] / avg_prev12_volume
    if pd.isna(shares_outstanding) or shares_outstanding <= 0:
        turnover = np.nan
    else:
        turnover = current["monthly_volume"] / shares_outstanding

    return {
        "status": "ok",
        "signal_month": str(current["month"]),
        "signal_date": current["data_end_date"],
        "data_end_date": current["data_end_date"],
        "available_months": int(len(monthly)),
        "monthly_volume": float(current["monthly_volume"]),
        "avg_prev12_monthly_volume": float(avg_prev12_volume),
        "avol": float(avol),
        "turnover": float(turnover) if not pd.isna(turnover) else np.nan,
        "monthly_turnover_value": float(current["monthly_turnover_value"]),
    }


def add_attention_scores(
    detail: pd.DataFrame,
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    out = detail.copy()
    ok_mask = out["status"].eq("ok") if "status" in out.columns else pd.Series(False, index=out.index)
    if not ok_mask.any():
        for col in [
            "avol_w",
            "turnover_w",
            "log_avol",
            "log_turnover",
            "z_log_avol",
            "z_log_turnover",
            "attention_raw",
            "attention_factor",
        ]:
            out[col] = np.nan
        return out

    ok = out.loc[ok_mask].copy()
    ok["avol_w"] = winsorize_series(ok["avol"], lower_q, upper_q)
    ok["turnover_w"] = winsorize_series(ok["turnover"], lower_q, upper_q)
    ok["log_avol"] = np.log(ok["avol_w"])
    ok["log_turnover"] = np.log1p(ok["turnover_w"])
    ok["z_log_avol"] = zscore_series(ok["log_avol"])
    ok["z_log_turnover"] = zscore_series(ok["log_turnover"])
    ok["attention_raw"] = ok["z_log_avol"] + ok["z_log_turnover"]
    ok["attention_factor"] = -ok["attention_raw"]

    score_cols = [
        "avol_w",
        "turnover_w",
        "log_avol",
        "log_turnover",
        "z_log_avol",
        "z_log_turnover",
        "attention_raw",
        "attention_factor",
    ]
    out = out.merge(ok[["code", *score_cols]], on="code", how="left")
    return out


def build_attention_minimal(
    detail: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    factor_name: str = "attention",
) -> pd.DataFrame:
    minimal = pd.DataFrame(
        {
            "code": detail["code"],
            "factor_name": factor_name,
            "factor_value": detail["attention_raw"],
            "signal_date": pd.to_datetime(detail["signal_date"], errors="coerce"),
            "data_end_date": pd.to_datetime(detail["data_end_date"], errors="coerce"),
            "rebalance_date": rebalance_date,
            "attention_raw": detail["attention_raw"],
            "attention_factor": detail["attention_factor"],
        }
    )
    return minimal


def normalize_attention_input_frame(inputs: pd.DataFrame) -> pd.DataFrame:
    lower_map = {str(col).strip().lower(): col for col in inputs.columns}

    def col(*names: str) -> Optional[str]:
        for name in names:
            found = lower_map.get(name.lower())
            if found is not None:
                return found
        return None

    code_col = col("code", "Code", "LocalCode", "stock_code")
    date_col = col("date", "Date", "signal_date")
    close_col = col("close", "Close", "adjustment_close", "AdjustmentClose", "adjusted_close")
    volume_col = col("volume", "Volume", "turnover_volume")
    shares_col = col("shares_outstanding", "SharesOutstanding", "listed_shares", "ListedShares")

    missing = [
        name
        for name, value in [
            ("code", code_col),
            ("date", date_col),
            ("close", close_col),
            ("volume", volume_col),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(f"Attention input frame is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "code": inputs[code_col].map(normalize_jp_code),
            "Date": pd.to_datetime(inputs[date_col], errors="coerce"),
            "Close": pd.to_numeric(inputs[close_col], errors="coerce"),
            "Volume": pd.to_numeric(inputs[volume_col], errors="coerce"),
        }
    )
    if shares_col is not None:
        out["shares_outstanding"] = pd.to_numeric(inputs[shares_col], errors="coerce")
    else:
        out["shares_outstanding"] = np.nan
    return out.dropna(subset=["code", "Date"])


def load_attention_inputs_jquants(
    universe_codes: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    jquants_api_key: Optional[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    client = config.get("jquants_client")
    if client is not None and hasattr(client, "load_attention_inputs"):
        return client.load_attention_inputs(universe_codes, start_date, end_date, config=config)
    raise NotImplementedError(
        "J-Quants attention loader is not wired yet. Pass config['attention_loader'] "
        "or config['jquants_client'] with load_attention_inputs(...)."
    )


def compute_attention_from_inputs(
    universe_codes: pd.DataFrame,
    input_frame: pd.DataFrame,
    asof_date: pd.Timestamp,
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    inputs = normalize_attention_input_frame(input_frame)
    rows: list[dict[str, Any]] = []
    ticker_map = dict(zip(universe_codes["code"], universe_codes["yf_ticker"]))

    for code, group in inputs.groupby("code"):
        ticker = ticker_map.get(code, to_yfinance_ticker(code))
        shares_series = pd.to_numeric(group.get("shares_outstanding", pd.Series(dtype=float)), errors="coerce").dropna()
        shares = float(shares_series.iloc[-1]) if not shares_series.empty else np.nan
        metrics = compute_attention_metrics(group[["Date", "Close", "Volume"]], shares, asof_date=asof_date)
        rows.append(
            {
                "code": code,
                "stock_code": code,
                "ticker": ticker,
                "yf_ticker": ticker,
                "shares_outstanding": shares,
                "rebalance_date": rebalance_date,
                "asof_date": asof_date,
                **metrics,
            }
        )

    seen_codes = {row["code"] for row in rows}
    for _, item in universe_codes.loc[~universe_codes["code"].isin(seen_codes)].iterrows():
        rows.append(
            {
                "code": item["code"],
                "stock_code": item["code"],
                "ticker": item["yf_ticker"],
                "yf_ticker": item["yf_ticker"],
                "shares_outstanding": np.nan,
                "rebalance_date": rebalance_date,
                "asof_date": asof_date,
                "status": "missing_attention_input",
            }
        )
    return pd.DataFrame(rows)


def run_attention_factor(
    universe: pd.DataFrame | str | Path,
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Calculate the attention / crowding factor for one rebalance date.

    jquants_api_key is accepted for interface consistency. Historical backtests
    should inject attention_loader or J-Quants-compatible loaders; yfinance is
    retained only behind allow_yfinance_fallback=True for debug compatibility.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    asof_date = pd.Timestamp(cfg.get("asof_date", rebalance_date)).tz_localize(None)
    rebalance_ts = pd.Timestamp(rebalance_date).tz_localize(None)
    start_date = asof_date - pd.Timedelta(days=int(cfg["history_buffer_calendar_days"]))
    end_date = asof_date
    explicit_data_source = cfg.get("data_source")
    has_jquants_loader = cfg.get("attention_loader") is not None or cfg.get("jquants_client") is not None
    # An API key alone does not mean the J-Quants attention loader is wired.
    # Fall back to the legacy yfinance loaders when the caller permits it.
    if explicit_data_source is None:
        data_source = "jquants" if has_jquants_loader else "unwired"
    else:
        data_source = str(explicit_data_source).lower()
    price_loader: PriceLoader | None = cfg.get("price_loader")
    shares_loader: SharesLoader | None = cfg.get("shares_loader")
    verbose = bool(cfg["verbose"])

    universe_codes = extract_universe_codes(universe)

    use_jquants_path = has_jquants_loader and data_source == "jquants"
    if use_jquants_path:
        loader: AttentionInputLoader = cfg.get("attention_loader") or load_attention_inputs_jquants
        input_frame = loader(universe_codes, start_date, end_date, jquants_api_key, cfg)
        detail = compute_attention_from_inputs(universe_codes, input_frame, asof_date, rebalance_ts)
    else:
        if data_source != "yfinance" and not bool(cfg.get("allow_yfinance_fallback", False)):
            raise ValueError(
                "Unsupported or unwired attention data_source. Pass attention_loader/"
                "jquants_client, or set allow_yfinance_fallback=True for debug yfinance."
            )
        if data_source in {"jquants", "unwired"} and bool(cfg.get("allow_yfinance_fallback", False)):
            data_source = "yfinance"
        if price_loader is None:
            price_loader = download_price_history_yfinance
        if shares_loader is None:
            shares_loader = get_shares_outstanding_yfinance
        rows: list[dict[str, Any]] = []
        for _, item in universe_codes.iterrows():
            code = item["code"]
            ticker = item["yf_ticker"]
            if verbose:
                print(f"Processing attention factor: {code} -> {ticker}")

            try:
                hist = price_loader(ticker, start_date, end_date, cfg)
                shares = shares_loader(ticker, cfg)
                metrics = compute_attention_metrics(hist, shares, asof_date=asof_date)
            except Exception as exc:  # pragma: no cover - depends on data source
                shares = np.nan
                metrics = {
                    "status": "download_or_calculation_failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

            rows.append(
                {
                    "code": code,
                    "stock_code": code,
                    "ticker": ticker,
                    "yf_ticker": ticker,
                    "shares_outstanding": shares,
                    "rebalance_date": rebalance_ts,
                    "asof_date": asof_date,
                    **metrics,
                }
            )
        detail = pd.DataFrame(rows)
    if detail.empty:
        detail = pd.DataFrame(
            columns=[
                "code",
                "stock_code",
                "ticker",
                "yf_ticker",
                "shares_outstanding",
                "rebalance_date",
                "asof_date",
                "status",
            ]
        )

    for col in ["signal_date", "data_end_date"]:
        if col not in detail.columns:
            detail[col] = pd.NaT
        detail[col] = pd.to_datetime(detail[col], errors="coerce")

    detail = add_attention_scores(
        detail,
        lower_q=float(cfg["winsor_lower_q"]),
        upper_q=float(cfg["winsor_upper_q"]),
    )
    minimal = build_attention_minimal(detail, rebalance_date=rebalance_ts)

    summary = {
        "factor_name": "attention",
        "rebalance_date": rebalance_ts,
        "asof_date": asof_date,
        "input_universe_rows": int(len(coerce_universe_frame(universe))),
        "usable_universe_rows": int(len(universe_codes)),
        "detail_rows": int(len(detail)),
        "minimal_rows": int(len(minimal)),
        "ok_rows": int((detail.get("status") == "ok").sum()) if "status" in detail else 0,
        "missing_factor_rows": int(detail["attention_raw"].isna().sum()),
        "data_source": data_source,
        "loader_mode": "attention_loader" if cfg.get("attention_loader") is not None else ("jquants_client" if cfg.get("jquants_client") is not None else "legacy_yfinance_loaders"),
        "shares_source": "custom_shares_loader" if config and config.get("shares_loader") else ("debug_yfinance_fallback" if data_source == "yfinance" else "attention_loader"),
        "jquants_api_key_supplied": bool(jquants_api_key),
    }

    if cfg.get("save_detail_path"):
        detail.to_csv(cfg["save_detail_path"], index=False, encoding="utf-8-sig")
    if cfg.get("save_minimal_path"):
        minimal.to_csv(cfg["save_minimal_path"], index=False, encoding="utf-8-sig")

    return {"minimal": minimal, "detail": detail, "summary": summary}


def run_from_csv(
    input_csv: str | Path,
    output_csv: str | Path = "attention_factor_output.csv",
    rebalance_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compatibility helper for the original input CSV -> output CSV flow."""
    anchor = pd.Timestamp.today().normalize() if rebalance_date is None else pd.Timestamp(rebalance_date)
    result = run_attention_factor(
        universe=input_csv,
        rebalance_date=anchor,
        config={"save_detail_path": output_csv},
    )
    return result["detail"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate JP equity attention factor.")
    parser.add_argument("--input", required=True, help="Universe CSV path.")
    parser.add_argument("--output", default="attention_factor_output.csv", help="Detail output CSV path.")
    parser.add_argument("--minimal-output", default=None, help="Optional minimal output CSV path.")
    parser.add_argument("--rebalance-date", required=True, help="Rebalance/as-of date, e.g. 2026-03-17.")
    args = parser.parse_args()

    run_attention_factor(
        universe=args.input,
        rebalance_date=args.rebalance_date,
        config={
            "save_detail_path": args.output,
            "save_minimal_path": args.minimal_output,
        },
    )
