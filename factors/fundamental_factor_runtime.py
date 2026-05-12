"""
Runtime wrappers for JP equity quality and value factors.

These functions are thin, stable-entry adaptations of the realtime quality and
value notebooks.  Historical backtests should pass config["fundamental_loader"]
so point-in-time J-Quants style statement summaries are used.  yfinance remains
available only as an explicit debug fallback.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


TICKER_COLUMN_CANDIDATES = ["ticker", "Ticker", "symbol", "Symbol", "code", "Code"]
CODE_COLUMN_CANDIDATES = ["code", "Code", "stock_code", "StockCode"]
INDUSTRY_COLUMN_CANDIDATES = ["industry", "Industry", "sector", "Sector"]

FundamentalLoader = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]

DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "historical_loader",
    "allow_yfinance_fallback": False,
    "statement_frequency": "annual",
    "request_pause_sec": 0.05,
    "retry_count": 2,
    "winsor_lower_q": 0.02,
    "winsor_upper_q": 0.98,
    "quality_min_metrics": 4,
    "value_min_metrics": 3,
    "save_minimal_path": None,
    "save_detail_path": None,
    "verbose": True,
}

FIELD_CANDIDATES = {
    "TotalRevenue": ["Total Revenue", "TotalRevenue"],
    "GrossProfit": ["Gross Profit", "GrossProfit"],
    "OperatingIncome": ["Operating Income", "OperatingIncome", "EBIT"],
    "EBIT": ["EBIT", "Operating Income", "OperatingIncome"],
    "EBITDA": ["EBITDA"],
    "NetIncome": ["Net Income", "NetIncome"],
    "TotalAssets": ["Total Assets", "TotalAssets"],
    "StockholdersEquity": ["Stockholders Equity", "StockholdersEquity", "Total Equity Gross Minority Interest"],
    "TotalDebt": ["Total Debt", "TotalDebt"],
    "CashAndCashEquivalents": ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash Cash Equivalents And Short Term Investments"],
    "OperatingCashFlow": ["Operating Cash Flow", "OperatingCashFlow", "Cash Flow From Continuing Operating Activities"],
    "FreeCashFlow": ["Free Cash Flow", "FreeCashFlow"],
}


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


def clean_string_series(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA, "NaN": pd.NA})


def safe_div(a: object, b: object) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    try:
        b_float = float(b)
        if b_float == 0:
            return np.nan
        return float(a) / b_float
    except Exception:
        return np.nan


def normalize_name(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def value_from_candidates(data_dict: dict[str, float], candidates: list[str]) -> float:
    if not data_dict:
        return np.nan
    norm_map = {normalize_name(k): v for k, v in data_dict.items()}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in norm_map:
            return norm_map[key]
    for candidate in candidates:
        key = normalize_name(candidate)
        for existing_key, existing_value in norm_map.items():
            if key in existing_key or existing_key in key:
                return existing_value
    return np.nan


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


def coerce_universe_frame(universe: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(universe, pd.DataFrame):
        return universe.copy()
    return pd.read_csv(universe)


def prepare_universe(universe: pd.DataFrame | str | Path) -> pd.DataFrame:
    source = coerce_universe_frame(universe)
    if source.empty:
        raise ValueError("Universe is empty.")
    ticker_col = first_existing(source.columns, TICKER_COLUMN_CANDIDATES)
    code_col = first_existing(source.columns, CODE_COLUMN_CANDIDATES)
    industry_col = first_existing(source.columns, INDUSTRY_COLUMN_CANDIDATES)
    if ticker_col is None and code_col is None:
        raise ValueError("Universe must include ticker or code.")

    out = source.copy()
    out["Ticker"] = out[ticker_col].map(to_yfinance_ticker) if ticker_col is not None else out[code_col].map(to_yfinance_ticker)
    out["code"] = out[code_col].map(normalize_jp_code) if code_col is not None else out["Ticker"].map(normalize_jp_code)
    if industry_col is not None:
        out["industry"] = clean_string_series(out[industry_col])
    elif "industry" not in out.columns:
        out["industry"] = pd.NA
    out = out.loc[out["Ticker"].notna()].drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)
    return out


def statement_to_dict(statement_df: pd.DataFrame) -> tuple[dict[str, float], object]:
    if statement_df is None or statement_df.empty:
        return {}, pd.NA
    cols = list(statement_df.columns)
    try:
        cols = sorted(cols, reverse=True)
    except Exception:
        pass
    if not cols:
        return {}, pd.NA
    return statement_df[cols[0]].to_dict(), cols[0]


def fetch_fundamentals_yfinance(universe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required unless config['fundamental_loader'] is supplied.") from exc

    freq = str(config.get("statement_frequency", "annual"))
    pause_sec = float(config.get("request_pause_sec", 0.05))
    retry_count = int(config.get("retry_count", 2))
    rows: list[dict[str, Any]] = []

    for _, item in universe.iterrows():
        ticker = item["Ticker"]
        base = {
            "Ticker": ticker,
            "code": item.get("code", normalize_jp_code(ticker)),
            "industry": item.get("industry", pd.NA),
            "FetchError": pd.NA,
            "MarketCap": np.nan,
            "StatementDate": pd.NA,
        }
        for field in FIELD_CANDIDATES:
            base[field] = np.nan

        last_error = None
        for _ in range(retry_count + 1):
            try:
                tk = yf.Ticker(ticker)
                try:
                    base["MarketCap"] = tk.fast_info.get("market_cap", np.nan)
                except Exception:
                    pass
                info = {}
                if pd.isna(base["MarketCap"]) or pd.isna(base["industry"]):
                    try:
                        info = tk.info if isinstance(tk.info, dict) else {}
                    except Exception:
                        info = {}
                if pd.isna(base["MarketCap"]):
                    base["MarketCap"] = info.get("marketCap", np.nan)
                if pd.isna(base["industry"]):
                    base["industry"] = info.get("industry") or info.get("sector") or pd.NA

                if freq == "quarterly":
                    fin_df = getattr(tk, "quarterly_financials", pd.DataFrame())
                    bal_df = getattr(tk, "quarterly_balance_sheet", pd.DataFrame())
                    cfs_df = getattr(tk, "quarterly_cashflow", pd.DataFrame())
                else:
                    fin_df = getattr(tk, "financials", pd.DataFrame())
                    bal_df = getattr(tk, "balance_sheet", pd.DataFrame())
                    cfs_df = getattr(tk, "cashflow", pd.DataFrame())

                fin, fin_date = statement_to_dict(fin_df)
                bal, bal_date = statement_to_dict(bal_df)
                cfs, cfs_date = statement_to_dict(cfs_df)
                dates = [x for x in [fin_date, bal_date, cfs_date] if pd.notna(x)]
                base["StatementDate"] = max(dates) if dates else pd.NA
                for field, candidates in FIELD_CANDIDATES.items():
                    if field in {"TotalRevenue", "GrossProfit", "OperatingIncome", "EBIT", "EBITDA", "NetIncome"}:
                        base[field] = value_from_candidates(fin, candidates)
                    elif field in {"TotalAssets", "StockholdersEquity", "TotalDebt", "CashAndCashEquivalents"}:
                        base[field] = value_from_candidates(bal, candidates)
                    else:
                        base[field] = value_from_candidates(cfs, candidates)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(pause_sec)
        if last_error is not None and pd.isna(base["StatementDate"]):
            base["FetchError"] = str(last_error)
        rows.append(base)
        if pause_sec > 0:
            time.sleep(pause_sec)

    return pd.DataFrame(rows)


def load_fundamental_panel(universe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    loader: FundamentalLoader | None = config.get("fundamental_loader")
    if loader is None:
        if not bool(config.get("allow_yfinance_fallback", False)):
            raise ValueError(
                "fundamental_loader is required for historical runs. "
                "Set allow_yfinance_fallback=True only for debug yfinance compatibility."
            )
        loader = fetch_fundamentals_yfinance
    panel = loader(universe, config)
    if "Ticker" not in panel.columns and "ticker" in panel.columns:
        panel = panel.rename(columns={"ticker": "Ticker"})
    if "code" not in panel.columns:
        panel["code"] = panel["Ticker"].map(normalize_jp_code)
    return panel.copy()


def compute_quality_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["ROE"] = out.apply(lambda x: safe_div(x.get("NetIncome"), x.get("StockholdersEquity")), axis=1)
    out["ROA"] = out.apply(lambda x: safe_div(x.get("NetIncome"), x.get("TotalAssets")), axis=1)
    out["GrossMargin"] = out.apply(lambda x: safe_div(x.get("GrossProfit"), x.get("TotalRevenue")), axis=1)
    out["OperatingMargin"] = out.apply(lambda x: safe_div(x.get("OperatingIncome"), x.get("TotalRevenue")), axis=1)
    out["FCF_to_Assets"] = out.apply(lambda x: safe_div(x.get("FreeCashFlow"), x.get("TotalAssets")), axis=1)
    out["CFO_to_NetIncome"] = out.apply(lambda x: safe_div(x.get("OperatingCashFlow"), x.get("NetIncome")), axis=1)
    out["Debt_to_Equity"] = out.apply(lambda x: safe_div(x.get("TotalDebt"), x.get("StockholdersEquity")), axis=1)
    metrics = ["ROE", "ROA", "GrossMargin", "OperatingMargin", "FCF_to_Assets", "CFO_to_NetIncome", "Debt_to_Equity"]
    out["MetricNonNullCount"] = out[metrics].notna().sum(axis=1)
    return out


def compute_value_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["EnterpriseValue_calc"] = out["MarketCap"] + out["TotalDebt"] - out["CashAndCashEquivalents"]
    out["E_to_P"] = out.apply(lambda x: safe_div(x.get("NetIncome"), x.get("MarketCap")), axis=1)
    out["B_to_P"] = out.apply(lambda x: safe_div(x.get("StockholdersEquity"), x.get("MarketCap")), axis=1)
    out["S_to_P"] = out.apply(lambda x: safe_div(x.get("TotalRevenue"), x.get("MarketCap")), axis=1)
    out["CFO_to_P"] = out.apply(lambda x: safe_div(x.get("OperatingCashFlow"), x.get("MarketCap")), axis=1)
    out["FCF_to_P"] = out.apply(lambda x: safe_div(x.get("FreeCashFlow"), x.get("MarketCap")), axis=1)
    out["EBIT_to_EV"] = out.apply(lambda x: safe_div(x.get("EBIT"), x.get("EnterpriseValue_calc")), axis=1)
    metrics = ["E_to_P", "B_to_P", "S_to_P", "CFO_to_P", "FCF_to_P", "EBIT_to_EV"]
    out["MetricNonNullCount"] = out[metrics].notna().sum(axis=1)
    return out


def build_weighted_score(
    df: pd.DataFrame,
    metrics: list[str],
    directions: dict[str, float],
    min_metrics: int,
    score_name: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        winsor = winsorize_series(out[metric], float(config["winsor_lower_q"]), float(config["winsor_upper_q"]))
        out[f"{metric}_z"] = zscore_series(winsor) * float(directions.get(metric, 1.0))

    z_cols = [f"{m}_z" for m in metrics]
    valid_counts = out[z_cols].notna().sum(axis=1)
    out[f"{score_name}_raw_pre_global_zscore"] = out[z_cols].mean(axis=1, skipna=True)
    out.loc[valid_counts < int(min_metrics), f"{score_name}_raw_pre_global_zscore"] = np.nan
    out[score_name] = zscore_series(out[f"{score_name}_raw_pre_global_zscore"])
    out[f"{score_name}Rank_desc_target"] = out[score_name].rank(ascending=False, method="dense")
    out["UsedMetricCount"] = valid_counts
    return out


def build_minimal(detail: pd.DataFrame, rebalance_date: pd.Timestamp, factor_name: str, score_col: str) -> pd.DataFrame:
    date_col = "StatementDate" if "StatementDate" in detail.columns else None
    signal_date = pd.to_datetime(detail[date_col], errors="coerce") if date_col else rebalance_date
    return pd.DataFrame(
        {
            "code": detail["code"],
            "factor_name": factor_name,
            "factor_value": detail[score_col],
            "signal_date": signal_date,
            "data_end_date": signal_date,
            "rebalance_date": rebalance_date,
            score_col: detail[score_col],
        }
    )


def run_quality_factor(
    universe: pd.DataFrame | str | Path,
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    prepared = prepare_universe(universe)
    panel = load_fundamental_panel(prepared, cfg)
    detail = compute_quality_metrics(panel)
    directions = {
        "ROE": 1.0,
        "ROA": 1.0,
        "GrossMargin": 1.0,
        "OperatingMargin": 1.0,
        "FCF_to_Assets": 1.0,
        "CFO_to_NetIncome": 1.0,
        "Debt_to_Equity": -1.0,
    }
    metrics = list(directions)
    detail = build_weighted_score(detail, metrics, directions, int(cfg["quality_min_metrics"]), "QualityScore", cfg)
    minimal = build_minimal(detail, rebalance_ts, "quality", "QualityScore")
    summary = {
        "factor_name": "quality",
        "rebalance_date": rebalance_ts,
        "input_rows": int(len(coerce_universe_frame(universe))),
        "detail_rows": int(len(detail)),
        "minimal_rows": int(len(minimal)),
        "missing_factor_rows": int(minimal["factor_value"].isna().sum()),
        "mode": cfg["mode"],
        "jquants_api_key_supplied": bool(jquants_api_key),
    }
    if cfg.get("save_detail_path"):
        detail.to_csv(cfg["save_detail_path"], index=False, encoding="utf-8-sig")
    if cfg.get("save_minimal_path"):
        minimal.to_csv(cfg["save_minimal_path"], index=False, encoding="utf-8-sig")
    return {"minimal": minimal, "detail": detail, "summary": summary}


def run_value_factor(
    universe: pd.DataFrame | str | Path,
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    prepared = prepare_universe(universe)
    panel = load_fundamental_panel(prepared, cfg)
    detail = compute_value_metrics(panel)
    metrics = ["E_to_P", "B_to_P", "S_to_P", "CFO_to_P", "FCF_to_P", "EBIT_to_EV"]
    directions = {metric: 1.0 for metric in metrics}
    detail = build_weighted_score(detail, metrics, directions, int(cfg["value_min_metrics"]), "ValueScore", cfg)
    minimal = build_minimal(detail, rebalance_ts, "value", "ValueScore")
    summary = {
        "factor_name": "value",
        "rebalance_date": rebalance_ts,
        "input_rows": int(len(coerce_universe_frame(universe))),
        "detail_rows": int(len(detail)),
        "minimal_rows": int(len(minimal)),
        "missing_factor_rows": int(minimal["factor_value"].isna().sum()),
        "mode": cfg["mode"],
        "jquants_api_key_supplied": bool(jquants_api_key),
    }
    if cfg.get("save_detail_path"):
        detail.to_csv(cfg["save_detail_path"], index=False, encoding="utf-8-sig")
    if cfg.get("save_minimal_path"):
        minimal.to_csv(cfg["save_minimal_path"], index=False, encoding="utf-8-sig")
    return {"minimal": minimal, "detail": detail, "summary": summary}
