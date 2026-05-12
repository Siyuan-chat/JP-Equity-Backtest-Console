"""
Runtime wrapper for the JP equity 12-1 momentum factor.

This repository uses standard 12-1 price momentum:
12-month lookback with the most recent 1 month skipped.
Legacy function names are kept for compatibility with the surrounding engine.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd


MARKET_BENCHMARK = "1306.T"

CANONICAL_INDUSTRY_TO_ETF: dict[str, str] = {
    "Foods": "1617.T",
    "Energy Resources": "1618.T",
    "Construction & Materials": "1619.T",
    "Raw Materials & Chemicals": "1620.T",
    "Pharmaceutical": "1621.T",
    "Automobiles & Transportation Equipment": "1622.T",
    "Steel & Nonferrous Metals": "1623.T",
    "Machinery": "1624.T",
    "Electric Appliances & Precision Instruments": "1625.T",
    "IT & Services, Others": "1626.T",
    "Electric Power & Gas": "1627.T",
    "Transportation & Logistics": "1628.T",
    "Commercial & Wholesale Trade": "1629.T",
    "Retail Trade": "1630.T",
    "Banks": "1631.T",
    "Financials (ex Banks)": "1632.T",
    "Real Estate": "1633.T",
}

RAW_ALIAS_MAP: dict[str, str] = {
    "Foods": "Foods",
    "Fishery, Agriculture & Forestry": "Foods",
    "Fishery, Agriculture and Forestry": "Foods",
    "Energy Resources": "Energy Resources",
    "Mining": "Energy Resources",
    "Oil & Coal Products": "Energy Resources",
    "Construction & Materials": "Construction & Materials",
    "Construction": "Construction & Materials",
    "Glass & Ceramics Products": "Construction & Materials",
    "Glass and Ceramics Products": "Construction & Materials",
    "Metal Products": "Construction & Materials",
    "Raw Materials & Chemicals": "Raw Materials & Chemicals",
    "Chemicals": "Raw Materials & Chemicals",
    "Textiles & Apparels": "Raw Materials & Chemicals",
    "Pulp & Paper": "Raw Materials & Chemicals",
    "Pulp and Paper": "Raw Materials & Chemicals",
    "Rubber Products": "Raw Materials & Chemicals",
    "Pharmaceutical": "Pharmaceutical",
    "Pharmaceuticals": "Pharmaceutical",
    "Automobiles & Transportation Equipment": "Automobiles & Transportation Equipment",
    "Transportation Equipment": "Automobiles & Transportation Equipment",
    "Steel & Nonferrous Metals": "Steel & Nonferrous Metals",
    "Steel": "Steel & Nonferrous Metals",
    "Iron and Steel": "Steel & Nonferrous Metals",
    "Iron & Steel": "Steel & Nonferrous Metals",
    "Nonferrous Metals": "Steel & Nonferrous Metals",
    "Machinery": "Machinery",
    "Electric Appliances & Precision Instruments": "Electric Appliances & Precision Instruments",
    "Electric Appliances": "Electric Appliances & Precision Instruments",
    "Precision Instruments": "Electric Appliances & Precision Instruments",
    "IT & Services, Others": "IT & Services, Others",
    "Information & Communication": "IT & Services, Others",
    "Services": "IT & Services, Others",
    "Electric Power & Gas": "Electric Power & Gas",
    "Electric Power & Gas Services": "Electric Power & Gas",
    "Transportation & Logistics": "Transportation & Logistics",
    "Land Transportation": "Transportation & Logistics",
    "Marine Transportation": "Transportation & Logistics",
    "Air Transportation": "Transportation & Logistics",
    "Warehousing & Harbor Transportation Services": "Transportation & Logistics",
    "Commercial & Wholesale Trade": "Commercial & Wholesale Trade",
    "Wholesale Trade": "Commercial & Wholesale Trade",
    "Retail Trade": "Retail Trade",
    "Retail": "Retail Trade",
    "Banks": "Banks",
    "Banks & Other Financial Institutions": "Banks",
    "Securities & Commodities Futures": "Financials (ex Banks)",
    "Insurance": "Financials (ex Banks)",
    "Other Financing Business": "Financials (ex Banks)",
    "Real Estate": "Real Estate",
}

CODE_COLUMN_CANDIDATES = ["code", "Code", "stock_code", "StockCode"]
TICKER_COLUMN_CANDIDATES = ["ticker", "Ticker", "symbol", "Symbol"]
INDUSTRY_COLUMN_CANDIDATES = ["industry", "Industry", "canonical_industry"]

CloseLoader = Callable[[Iterable[str], pd.Timestamp, pd.Timestamp, dict[str, Any]], pd.DataFrame]


@dataclass
class ResidualMomentumParams:
    history_buffer_calendar_days: int = 520
    beta_window: int = 60
    min_reg_obs: int = 40
    signal_window: int = 60
    skip_recent: int = 20
    min_signal_coverage: float = 0.80
    winsor_lower: float = 0.02
    winsor_upper: float = 0.98
    industry_min_count_for_z: int = 5
    build_industry_control_version: bool = True
    build_second_stage_version: bool = True
    allow_market_only_fallback: bool = True
    second_stage_features: tuple[str, ...] = (
        "log_market_cap_billion_jpy",
        "log_avg_daily_trading_value",
        "annualized_volatility",
    )
    save_minimal_path: Optional[str] = None
    save_detail_path: Optional[str] = None
    verbose: bool = True


def normalize_jp_code(value: object) -> Optional[str]:
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
    return None


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


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_list = list(columns)
    lower_map = {str(col).strip().lower(): col for col in col_list}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def normalize_industry_text(text: object) -> Optional[str]:
    if pd.isna(text):
        return None
    value = str(text).strip()
    if not value:
        return None
    return " ".join(value.replace(" and ", " & ").split()).casefold()


INDUSTRY_ALIASES = {normalize_industry_text(k): v for k, v in RAW_ALIAS_MAP.items()}
for _canonical in CANONICAL_INDUSTRY_TO_ETF:
    INDUSTRY_ALIASES[normalize_industry_text(_canonical)] = _canonical


def canonicalize_industry(industry: object) -> Optional[str]:
    normalized = normalize_industry_text(industry)
    return None if normalized is None else INDUSTRY_ALIASES.get(normalized)


def safe_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if pd.isna(std) or std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (values - valid.mean()) / std


def winsorize_series(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return values.copy()
    return values.clip(valid.quantile(lower_q), valid.quantile(upper_q))


def compute_rank(series: pd.Series, ascending: bool = False) -> pd.Series:
    return series.rank(ascending=ascending, method="dense")


def coerce_universe_frame(universe: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(universe, pd.DataFrame):
        return universe.copy()
    return pd.read_csv(universe)


def prepare_universe(universe: pd.DataFrame | str | Path) -> pd.DataFrame:
    df = coerce_universe_frame(universe)
    if df.empty:
        return pd.DataFrame(columns=["code", "ticker", "industry", "canonical_industry", "sector_etf"])

    ticker_col = first_existing(df.columns, TICKER_COLUMN_CANDIDATES)
    code_col = first_existing(df.columns, CODE_COLUMN_CANDIDATES)
    industry_col = first_existing(df.columns, INDUSTRY_COLUMN_CANDIDATES)
    if ticker_col is None and code_col is None:
        raise ValueError("Universe must contain a ticker/code column.")
    out = df.copy()
    if ticker_col is None:
        out["ticker"] = out[code_col].map(to_yfinance_ticker)
    else:
        out["ticker"] = out[ticker_col].map(to_yfinance_ticker)
    if code_col is None:
        out["code"] = out["ticker"].map(normalize_jp_code)
    else:
        out["code"] = out[code_col].map(normalize_jp_code)
    if industry_col is None:
        out["industry"] = pd.NA
        out["canonical_industry"] = pd.NA
        out["sector_etf"] = pd.NA
    else:
        out["industry"] = out[industry_col]
        out["canonical_industry"] = out["industry"].apply(canonicalize_industry)
        out["sector_etf"] = out["canonical_industry"].map(CANONICAL_INDUSTRY_TO_ETF)

    if "market_cap_billion_jpy" in out.columns:
        out["log_market_cap_billion_jpy"] = np.log(pd.to_numeric(out["market_cap_billion_jpy"], errors="coerce").replace(0, np.nan))
    if "avg_daily_trading_value" in out.columns:
        out["log_avg_daily_trading_value"] = np.log(pd.to_numeric(out["avg_daily_trading_value"], errors="coerce").replace(0, np.nan))

    out = out.loc[out["ticker"].notna()].drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return out


def download_close_prices_yfinance(
    tickers: Iterable[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required unless config['close_loader'] is supplied.") from exc

    ticker_list = sorted(set(tickers))
    data = yf.download(
        tickers=ticker_list,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data is None or data.empty:
        raise RuntimeError("No data returned by yfinance.")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": ticker_list[0]})
    close = close.sort_index().dropna(axis=1, how="all")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def rolling_dual_neutral_residuals(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    sector_ret: pd.Series,
    beta_window: int,
    min_reg_obs: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    aligned = pd.concat(
        [stock_ret.rename("stock"), market_ret.rename("market"), sector_ret.rename("sector")],
        axis=1,
        join="inner",
    ).sort_index()
    residual = pd.Series(index=aligned.index, dtype=float)
    beta_market = pd.Series(index=aligned.index, dtype=float)
    beta_sector = pd.Series(index=aligned.index, dtype=float)

    for pos in range(beta_window, len(aligned)):
        train = aligned.iloc[pos - beta_window : pos].dropna()
        current = aligned.iloc[pos]
        if len(train) < min_reg_obs or current.isna().any():
            continue
        x = np.column_stack(
            [np.ones(len(train)), train["market"].to_numpy(), train["sector"].to_numpy()]
        )
        coef, *_ = np.linalg.lstsq(x, train["stock"].to_numpy(), rcond=None)
        current_x = np.array([1.0, current["market"], current["sector"]])
        residual.iloc[pos] = current["stock"] - float(current_x @ coef)
        beta_market.iloc[pos] = coef[1]
        beta_sector.iloc[pos] = coef[2]

    return residual, beta_market, beta_sector


def rolling_market_neutral_residuals(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    beta_window: int,
    min_reg_obs: int,
) -> tuple[pd.Series, pd.Series]:
    aligned = pd.concat([stock_ret.rename("stock"), market_ret.rename("market")], axis=1, join="inner").sort_index()
    residual = pd.Series(index=aligned.index, dtype=float)
    beta_market = pd.Series(index=aligned.index, dtype=float)

    for pos in range(beta_window, len(aligned)):
        train = aligned.iloc[pos - beta_window : pos].dropna()
        current = aligned.iloc[pos]
        if len(train) < min_reg_obs or current.isna().any():
            continue
        x = np.column_stack([np.ones(len(train)), train["market"].to_numpy()])
        coef, *_ = np.linalg.lstsq(x, train["stock"].to_numpy(), rcond=None)
        current_x = np.array([1.0, current["market"]])
        residual.iloc[pos] = current["stock"] - float(current_x @ coef)
        beta_market.iloc[pos] = coef[1]

    return residual, beta_market


def rolling_market_neutral_residuals_panel(
    stock_ret: pd.DataFrame,
    market_ret: pd.Series,
    beta_window: int,
    min_reg_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock = stock_ret.sort_index().astype(float)
    market = pd.to_numeric(market_ret.reindex(stock.index), errors="coerce").astype(float)
    index = stock.index
    columns = stock.columns
    y = stock.to_numpy(dtype=float)
    m = market.to_numpy(dtype=float)
    t_len, n_cols = y.shape
    residual = np.full((t_len, n_cols), np.nan, dtype=float)
    beta_market = np.full((t_len, n_cols), np.nan, dtype=float)

    for pos in range(beta_window, t_len):
        y_train = y[pos - beta_window : pos, :]
        m_train = m[pos - beta_window : pos]
        current_y = y[pos, :]
        current_m = m[pos]
        train_valid = np.isfinite(y_train) & np.isfinite(m_train)[:, None]
        train_count = train_valid.sum(axis=0)
        current_valid = np.isfinite(current_y) & np.isfinite(current_m)
        valid_cols = (train_count >= min_reg_obs) & current_valid
        if not valid_cols.any():
            continue

        col_idx = np.flatnonzero(valid_cols)
        yv = np.where(train_valid[:, valid_cols], y_train[:, valid_cols], 0.0)
        mv = np.where(train_valid[:, valid_cols], m_train[:, None], 0.0)
        cnt = train_count[valid_cols].astype(float)
        mean_y = yv.sum(axis=0) / cnt
        mean_m = mv.sum(axis=0) / cnt
        cov_ym = (yv * mv).sum(axis=0) / cnt - mean_y * mean_m
        var_m = (mv * mv).sum(axis=0) / cnt - mean_m * mean_m
        stable = np.isfinite(var_m) & (var_m > 1e-12)
        if not stable.any():
            continue

        good_idx = col_idx[stable]
        beta = cov_ym[stable] / var_m[stable]
        alpha = mean_y[stable] - beta * mean_m[stable]
        residual[pos, good_idx] = current_y[good_idx] - alpha - beta * current_m
        beta_market[pos, good_idx] = beta

    return (
        pd.DataFrame(residual, index=index, columns=columns),
        pd.DataFrame(beta_market, index=index, columns=columns),
    )


def rolling_dual_neutral_residuals_panel(
    stock_ret: pd.DataFrame,
    market_ret: pd.Series,
    sector_ret: pd.Series,
    beta_window: int,
    min_reg_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock = stock_ret.sort_index().astype(float)
    market = pd.to_numeric(market_ret.reindex(stock.index), errors="coerce").astype(float)
    sector = pd.to_numeric(sector_ret.reindex(stock.index), errors="coerce").astype(float)
    index = stock.index
    columns = stock.columns
    y = stock.to_numpy(dtype=float)
    m = market.to_numpy(dtype=float)
    s = sector.to_numpy(dtype=float)
    t_len, n_cols = y.shape
    residual = np.full((t_len, n_cols), np.nan, dtype=float)
    beta_market = np.full((t_len, n_cols), np.nan, dtype=float)
    beta_sector = np.full((t_len, n_cols), np.nan, dtype=float)

    for pos in range(beta_window, t_len):
        y_train = y[pos - beta_window : pos, :]
        m_train = m[pos - beta_window : pos]
        s_train = s[pos - beta_window : pos]
        current_y = y[pos, :]
        current_m = m[pos]
        current_s = s[pos]
        train_valid = np.isfinite(y_train) & np.isfinite(m_train)[:, None] & np.isfinite(s_train)[:, None]
        train_count = train_valid.sum(axis=0)
        current_valid = np.isfinite(current_y) & np.isfinite(current_m) & np.isfinite(current_s)
        valid_cols = (train_count >= min_reg_obs) & current_valid
        if not valid_cols.any():
            continue

        col_idx = np.flatnonzero(valid_cols)
        yv = np.where(train_valid[:, valid_cols], y_train[:, valid_cols], 0.0)
        mv = np.where(train_valid[:, valid_cols], m_train[:, None], 0.0)
        sv = np.where(train_valid[:, valid_cols], s_train[:, None], 0.0)
        cnt = train_count[valid_cols].astype(float)
        mean_y = yv.sum(axis=0) / cnt
        mean_m = mv.sum(axis=0) / cnt
        mean_s = sv.sum(axis=0) / cnt
        cov_ym = (yv * mv).sum(axis=0) / cnt - mean_y * mean_m
        cov_ys = (yv * sv).sum(axis=0) / cnt - mean_y * mean_s
        var_m = (mv * mv).sum(axis=0) / cnt - mean_m * mean_m
        var_s = (sv * sv).sum(axis=0) / cnt - mean_s * mean_s
        cov_ms = (mv * sv).sum(axis=0) / cnt - mean_m * mean_s
        det = var_m * var_s - cov_ms * cov_ms
        stable = np.isfinite(det) & (np.abs(det) > 1e-12)
        if not stable.any():
            continue

        good_idx = col_idx[stable]
        beta_m = (cov_ym[stable] * var_s[stable] - cov_ys[stable] * cov_ms[stable]) / det[stable]
        beta_s = (cov_ys[stable] * var_m[stable] - cov_ym[stable] * cov_ms[stable]) / det[stable]
        alpha = mean_y[stable] - beta_m * mean_m[stable] - beta_s * mean_s[stable]
        residual[pos, good_idx] = current_y[good_idx] - alpha - beta_m * current_m - beta_s * current_s
        beta_market[pos, good_idx] = beta_m
        beta_sector[pos, good_idx] = beta_s

    return (
        pd.DataFrame(residual, index=index, columns=columns),
        pd.DataFrame(beta_market, index=index, columns=columns),
        pd.DataFrame(beta_sector, index=index, columns=columns),
    )


def select_signal_window(residuals: pd.Series, params: ResidualMomentumParams) -> Optional[pd.Series]:
    residuals = residuals.sort_index()
    if len(residuals) < params.signal_window + params.skip_recent:
        return None
    end_pos = len(residuals) - 1 - params.skip_recent
    start_pos = end_pos - params.signal_window + 1
    if start_pos < 0:
        return None
    window = residuals.iloc[start_pos : end_pos + 1]
    min_obs = int(np.ceil(params.signal_window * params.min_signal_coverage))
    return window if int(window.notna().sum()) >= min_obs else None


def compute_signal_metrics(signal_slice: pd.Series) -> dict[str, float]:
    valid = signal_slice.dropna()
    obs = len(valid)
    if obs == 0:
        return {
            "resid_cumret": np.nan,
            "resid_ir": np.nan,
            "resid_tstat": np.nan,
            "residual_vol": np.nan,
            "resid_positive_ratio": np.nan,
            "signal_obs": 0,
        }
    resid_std = valid.std(ddof=0)
    mean_ret = valid.mean()
    resid_ir = np.nan if pd.isna(resid_std) or resid_std < 1e-12 else mean_ret / resid_std * np.sqrt(252.0)
    resid_tstat = np.nan if pd.isna(resid_std) or resid_std < 1e-12 else mean_ret / (resid_std / np.sqrt(obs))
    return {
        "resid_cumret": (1.0 + valid).prod() - 1.0,
        "resid_ir": resid_ir,
        "resid_tstat": resid_tstat,
        "residual_vol": resid_std,
        "resid_positive_ratio": (valid > 0).mean(),
        "signal_obs": obs,
    }


def compute_residual_momentum_batch(
    prepared: pd.DataFrame,
    returns_df: pd.DataFrame,
    params: ResidualMomentumParams,
    market_benchmark: str,
) -> pd.DataFrame:
    base_columns = [
        "code",
        "ticker",
        "industry",
        "canonical_industry",
        "sector_etf",
        "log_market_cap_billion_jpy",
        "log_avg_daily_trading_value",
        "annualized_volatility",
    ]
    out = prepared.copy()
    for col in [
        "beta_market_latest",
        "beta_sector_latest",
        "resid_cumret",
        "resid_ir",
        "resid_tstat",
        "residual_vol",
        "resid_positive_ratio",
        "signal_obs",
    ]:
        out[col] = np.nan
    out["signal_start"] = pd.NaT
    out["signal_end"] = pd.NaT
    out["data_status"] = "ok"
    out["neutralization_model"] = pd.NA

    if market_benchmark not in returns_df.columns:
        out["data_status"] = f"missing_data_columns:{market_benchmark}"
        return out

    ticker_present = out["ticker"].isin(returns_df.columns)
    out.loc[~ticker_present, "data_status"] = out.loc[~ticker_present, "ticker"].map(lambda x: f"missing_data_columns:{x}")

    missing_sector_mapping = out["canonical_industry"].isna() | out["sector_etf"].isna()
    missing_sector_prices = ~out["sector_etf"].isin(returns_df.columns)
    dual_eligible = ticker_present & ~missing_sector_mapping & ~missing_sector_prices
    market_only = ticker_present & ~dual_eligible
    if not params.allow_market_only_fallback:
        out.loc[ticker_present & missing_sector_mapping, "data_status"] = "industry_mapping_missing"
        out.loc[ticker_present & ~missing_sector_mapping & missing_sector_prices, "data_status"] = (
            out.loc[ticker_present & ~missing_sector_mapping & missing_sector_prices, "sector_etf"]
            .map(lambda x: f"missing_data_columns:{x}")
        )
        market_only = pd.Series(False, index=out.index)

    min_obs = int(np.ceil(params.signal_window * params.min_signal_coverage))
    end_pos = len(returns_df.index) - 1 - params.skip_recent
    start_pos = end_pos - params.signal_window + 1
    if start_pos < 0:
        out.loc[ticker_present, "data_status"] = "insufficient_signal_window"
        return out
    latest_signal_date = returns_df.index[end_pos]
    out.loc[ticker_present, "signal_start"] = returns_df.index[start_pos]
    out.loc[ticker_present, "signal_end"] = latest_signal_date

    market_series = returns_df[market_benchmark]

    def _apply_metrics(idx: pd.Index, residual: pd.DataFrame, beta_market: pd.DataFrame, beta_sector: pd.DataFrame, model_name: str) -> None:
        if residual.empty:
            return
        signal = residual.iloc[start_pos : end_pos + 1]
        obs = signal.notna().sum(axis=0)
        cumret = (1.0 + signal).prod(axis=0, min_count=1) - 1.0
        mean = signal.mean(axis=0)
        std = signal.std(axis=0, ddof=0)
        ir = mean.divide(std).replace([np.inf, -np.inf], np.nan) * np.sqrt(252.0)
        tstat = mean.divide(std.divide(np.sqrt(obs))).replace([np.inf, -np.inf], np.nan)
        positive_ratio = signal.gt(0).sum(axis=0).divide(obs).replace([np.inf, -np.inf], np.nan)
        beta_market_latest = beta_market.ffill().iloc[end_pos]
        beta_sector_latest = beta_sector.ffill().iloc[end_pos] if not beta_sector.empty else pd.Series(np.nan, index=residual.columns)
        has_any_residual = residual.notna().any(axis=0)
        enough_signal = obs >= min_obs

        by_ticker = pd.Series(idx, index=out.loc[idx, "ticker"])
        target_idx = by_ticker.reindex(residual.columns).dropna().astype(int)
        cols = target_idx.index
        rows = target_idx.to_numpy()
        out.loc[rows, "neutralization_model"] = model_name
        out.loc[rows, "beta_market_latest"] = beta_market_latest.reindex(cols).to_numpy()
        out.loc[rows, "beta_sector_latest"] = beta_sector_latest.reindex(cols).to_numpy()
        out.loc[rows, "resid_cumret"] = cumret.reindex(cols).to_numpy()
        out.loc[rows, "resid_ir"] = ir.reindex(cols).to_numpy()
        out.loc[rows, "resid_tstat"] = tstat.reindex(cols).to_numpy()
        out.loc[rows, "residual_vol"] = std.reindex(cols).to_numpy()
        out.loc[rows, "resid_positive_ratio"] = positive_ratio.reindex(cols).to_numpy()
        out.loc[rows, "signal_obs"] = obs.reindex(cols).to_numpy()
        out.loc[rows[~has_any_residual.reindex(cols).fillna(False).to_numpy()], "data_status"] = "no_residuals"
        out.loc[rows[has_any_residual.reindex(cols).fillna(False).to_numpy() & ~enough_signal.reindex(cols).fillna(False).to_numpy()], "data_status"] = "insufficient_signal_window"

    dual_frame = out.loc[dual_eligible].copy()
    if not dual_frame.empty:
        for sector_etf, group in dual_frame.groupby("sector_etf", sort=False):
            tickers = group["ticker"].dropna().tolist()
            residual, beta_market, beta_sector = rolling_dual_neutral_residuals_panel(
                returns_df[tickers],
                market_series,
                returns_df[sector_etf],
                params.beta_window,
                params.min_reg_obs,
            )
            _apply_metrics(group.index, residual, beta_market, beta_sector, "market_sector")

    market_only_frame = out.loc[market_only].copy()
    if not market_only_frame.empty:
        tickers = market_only_frame["ticker"].dropna().tolist()
        residual, beta_market = rolling_market_neutral_residuals_panel(
            returns_df[tickers],
            market_series,
            params.beta_window,
            params.min_reg_obs,
        )
        beta_sector = pd.DataFrame(np.nan, index=residual.index, columns=residual.columns)
        _apply_metrics(market_only_frame.index, residual, beta_market, beta_sector, "market_only")

    return out



def compute_single_stock_factor(
    row: pd.Series,
    returns_df: pd.DataFrame,
    params: ResidualMomentumParams,
    market_benchmark: str,
) -> dict[str, Any]:
    out = row.to_dict()
    out.update(
        {
            "beta_market_latest": np.nan,
            "beta_sector_latest": np.nan,
            "resid_cumret": np.nan,
            "resid_ir": np.nan,
            "resid_tstat": np.nan,
            "residual_vol": np.nan,
            "resid_positive_ratio": np.nan,
            "signal_obs": np.nan,
            "signal_start": pd.NaT,
            "signal_end": pd.NaT,
            "data_status": "ok",
        }
    )

    ticker = row["ticker"]
    sector_etf = row.get("sector_etf")
    if ticker not in returns_df.columns:
        out["data_status"] = f"missing_data_columns:{ticker}"
        return out
    if market_benchmark not in returns_df.columns:
        out["data_status"] = f"missing_data_columns:{market_benchmark}"
        return out

    use_market_only = False
    if pd.isna(row.get("canonical_industry")) or pd.isna(sector_etf):
        if not params.allow_market_only_fallback:
            out["data_status"] = "industry_mapping_missing"
            return out
        use_market_only = True

    missing_cols = [] if use_market_only else [col for col in [sector_etf] if col not in returns_df.columns]
    if missing_cols:
        if not params.allow_market_only_fallback:
            out["data_status"] = f"missing_data_columns:{','.join(missing_cols)}"
            return out
        use_market_only = True

    if use_market_only:
        residuals, beta_market = rolling_market_neutral_residuals(
            returns_df[ticker],
            returns_df[market_benchmark],
            params.beta_window,
            params.min_reg_obs,
        )
        beta_sector = pd.Series(np.nan, index=residuals.index, dtype=float)
        out["neutralization_model"] = "market_only"
    else:
        residuals, beta_market, beta_sector = rolling_dual_neutral_residuals(
            returns_df[ticker],
            returns_df[market_benchmark],
            returns_df[sector_etf],
            params.beta_window,
            params.min_reg_obs,
        )
        out["neutralization_model"] = "market_sector"
    if residuals.dropna().empty:
        out["data_status"] = "no_residuals"
        return out

    signal_slice = select_signal_window(residuals, params)
    if signal_slice is None:
        out["data_status"] = "insufficient_signal_window"
        return out

    latest_signal_date = signal_slice.index[-1]
    out.update(compute_signal_metrics(signal_slice))
    beta_market_latest = beta_market.loc[:latest_signal_date].dropna()
    beta_sector_latest = beta_sector.loc[:latest_signal_date].dropna()
    out["beta_market_latest"] = beta_market_latest.iloc[-1] if not beta_market_latest.empty else np.nan
    out["beta_sector_latest"] = beta_sector_latest.iloc[-1] if not beta_sector_latest.empty else np.nan
    out["signal_start"] = signal_slice.index[0]
    out["signal_end"] = signal_slice.index[-1]
    return out


def apply_cross_section_template(
    df: pd.DataFrame,
    raw_col: str,
    version_name: str,
    params: ResidualMomentumParams,
    industry_adjust: bool,
) -> pd.DataFrame:
    result = df.copy()
    valid_mask = result["data_status"].eq("ok") & result[raw_col].notna()
    winsor_col = f"{version_name}_winsor"
    adjusted_col = f"{version_name}_adjusted"
    score_col = f"{version_name}_score"
    rank_col = f"{version_name}_rank"
    rank_ind_col = f"{version_name}_rank_in_industry"

    for col in [winsor_col, adjusted_col, score_col, rank_col, rank_ind_col]:
        result[col] = np.nan
    if not valid_mask.any():
        return result

    result.loc[valid_mask, winsor_col] = winsorize_series(
        result.loc[valid_mask, raw_col].astype(float),
        params.winsor_lower,
        params.winsor_upper,
    )
    if industry_adjust:
        for _, group in result.loc[valid_mask].groupby("canonical_industry"):
            idx = group.index
            values = result.loc[idx, winsor_col].astype(float)
            if len(values) >= params.industry_min_count_for_z and values.std(ddof=0) > 1e-12:
                result.loc[idx, adjusted_col] = safe_zscore(values)
            else:
                result.loc[idx, adjusted_col] = values - values.mean()
    else:
        result.loc[valid_mask, adjusted_col] = result.loc[valid_mask, winsor_col]

    result.loc[valid_mask, score_col] = safe_zscore(result.loc[valid_mask, adjusted_col].astype(float))
    result.loc[valid_mask, rank_col] = compute_rank(result.loc[valid_mask, score_col], ascending=False)
    for _, group in result.loc[valid_mask].groupby("canonical_industry"):
        idx = group.index
        result.loc[idx, rank_ind_col] = compute_rank(result.loc[idx, score_col], ascending=False)
    return result


def second_stage_neutralize(
    df: pd.DataFrame,
    raw_col: str,
    version_name: str,
    params: ResidualMomentumParams,
) -> pd.DataFrame:
    result = df.copy()
    residual_col = f"{version_name}_raw"
    score_col = f"{version_name}_score"
    rank_col = f"{version_name}_rank"
    rank_ind_col = f"{version_name}_rank_in_industry"
    for col in [residual_col, score_col, rank_col, rank_ind_col]:
        result[col] = np.nan

    needed = [col for col in params.second_stage_features if col in result.columns]
    valid_mask = result["data_status"].eq("ok") & result[raw_col].notna()
    if needed:
        valid_mask = valid_mask & result[needed].notna().all(axis=1)
    if len(needed) == 0 or valid_mask.sum() < len(needed) + 2:
        return result

    y = winsorize_series(result.loc[valid_mask, raw_col].astype(float), params.winsor_lower, params.winsor_upper)
    x_raw = result.loc[valid_mask, needed].astype(float)
    x = ((x_raw - x_raw.mean()) / x_raw.std(ddof=0).replace(0, np.nan)).fillna(0.0)
    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef, *_ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)
    residual_signal = y.to_numpy() - design @ coef
    result.loc[valid_mask, residual_col] = residual_signal
    result.loc[valid_mask, score_col] = safe_zscore(pd.Series(residual_signal, index=y.index))
    result.loc[valid_mask, rank_col] = compute_rank(result.loc[valid_mask, score_col], ascending=False)
    for _, group in result.loc[valid_mask].groupby("canonical_industry"):
        idx = group.index
        result.loc[idx, rank_ind_col] = compute_rank(result.loc[idx, score_col], ascending=False)
    return result


def build_metric_versions(df: pd.DataFrame, metric: str, params: ResidualMomentumParams) -> pd.DataFrame:
    enriched = apply_cross_section_template(df, metric, f"{metric}_raw", params, industry_adjust=False)
    if params.build_industry_control_version:
        enriched = apply_cross_section_template(enriched, metric, f"{metric}_industry", params, industry_adjust=True)
    if params.build_second_stage_version:
        enriched = second_stage_neutralize(enriched, metric, f"{metric}_neutralized", params)
    return enriched


def build_residual_momentum_minimal(detail: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    factor_col = "momentum_12_1_score"
    if factor_col not in detail.columns or detail[factor_col].isna().all():
        factor_col = "momentum_12_1_raw"
    factor_value = detail[factor_col]
    minimal = pd.DataFrame(
        {
            "code": detail["code"],
            "factor_name": "momentum_12_1",
            "factor_value": factor_value,
            "signal_date": pd.to_datetime(detail["signal_end"], errors="coerce"),
            "data_end_date": pd.to_datetime(detail["asof_date"], errors="coerce"),
            "freshness_date": pd.to_datetime(detail["asof_date"], errors="coerce"),
            "rebalance_date": rebalance_date,
            "momentum_score": factor_value,
            "momentum_12_1_score": factor_value,
            "resid_ir_neutralized_score": factor_value,
            "resid_tstat_neutralized_score": detail.get("momentum_12_1_raw", pd.Series(np.nan, index=detail.index)),
            "resid_cumret_neutralized_score": detail.get("momentum_12_1_raw", pd.Series(np.nan, index=detail.index)),
        }
    )
    return minimal


def run_residual_momentum_factor(
    universe: pd.DataFrame | str | Path,
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Calculate standard 12-1 momentum for one rebalance date.

    Historical backtests must inject config["close_loader"] so closes come from
    local JPX/J-Quants style history. yfinance is retained only behind
    allow_yfinance_fallback=True for debug compatibility.
    """
    cfg = config or {}
    params = ResidualMomentumParams(**{k: v for k, v in cfg.items() if k in ResidualMomentumParams.__dataclass_fields__})
    rebalance_ts = pd.Timestamp(rebalance_date).tz_localize(None)
    asof_date = pd.Timestamp(cfg.get("asof_date", rebalance_date)).tz_localize(None)
    start_date = asof_date - pd.Timedelta(days=params.history_buffer_calendar_days)
    close_loader: CloseLoader | None = cfg.get("close_loader")
    if close_loader is None:
        if not bool(cfg.get("allow_yfinance_fallback", False)):
            raise ValueError(
                "close_loader is required for historical 12-1 momentum runs. "
                "Set allow_yfinance_fallback=True only for debug yfinance compatibility."
            )
        close_loader = download_close_prices_yfinance

    prepared = prepare_universe(universe)
    tickers = prepared["ticker"].dropna().tolist()
    all_tickers = sorted(set(tickers))
    if params.verbose:
        print(f"12-1 momentum: downloading/using close prices for {len(all_tickers)} tickers")

    close = close_loader(all_tickers, start_date, asof_date, cfg)
    close = close.loc[close.index <= asof_date].sort_index()
    lookback = int(cfg.get("momentum_lookback_trading_days", 252))
    skip_recent = int(cfg.get("momentum_skip_recent_trading_days", 21))
    min_required = int(max(2, round((lookback + skip_recent + 1) * float(params.min_signal_coverage))))
    rows: list[dict[str, Any]] = []
    asof_actual = close.index.max() if not close.empty else pd.NaT
    signal_end = close.index[-(skip_recent + 1)] if len(close.index) > skip_recent else pd.NaT
    for _, row in prepared.iterrows():
        ticker = row.get("ticker")
        series = close.get(ticker) if ticker in close.columns else None
        if series is None:
            rows.append(
                {
                    "code": row.get("code"),
                    "ticker": ticker,
                    "data_status": "missing_price_history",
                    "signal_end": signal_end,
                    "asof_date": asof_actual,
                    "momentum_12_1_raw": np.nan,
                }
            )
            continue
        valid = pd.to_numeric(series, errors="coerce").dropna()
        if len(valid) < min_required or len(valid) <= lookback + skip_recent:
            rows.append(
                {
                    "code": row.get("code"),
                    "ticker": ticker,
                    "data_status": "insufficient_history",
                    "signal_end": signal_end,
                    "asof_date": asof_actual,
                    "momentum_12_1_raw": np.nan,
                }
            )
            continue
        end_price = float(valid.iloc[-(skip_recent + 1)])
        start_price = float(valid.iloc[-(lookback + skip_recent + 1)])
        raw_value = np.nan if start_price <= 0 else (end_price / start_price) - 1.0
        rows.append(
            {
                "code": row.get("code"),
                "ticker": ticker,
                "data_status": "ok" if np.isfinite(raw_value) else "invalid_return",
                "signal_end": signal_end,
                "asof_date": asof_actual,
                "momentum_12_1_raw": raw_value,
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        detail = pd.DataFrame(columns=["code", "ticker", "data_status", "signal_end", "asof_date", "momentum_12_1_raw"])
    detail["momentum_12_1_score"] = np.nan
    valid_mask = detail["data_status"].eq("ok") & pd.to_numeric(detail["momentum_12_1_raw"], errors="coerce").notna()
    if valid_mask.any():
        detail.loc[valid_mask, "momentum_12_1_score"] = safe_zscore(detail.loc[valid_mask, "momentum_12_1_raw"].astype(float))
    detail["momentum_12_1_rank"] = np.nan
    if valid_mask.any():
        detail.loc[valid_mask, "momentum_12_1_rank"] = compute_rank(detail.loc[valid_mask, "momentum_12_1_score"], ascending=False)
    detail["benchmark"] = "not_used_in_public_12_1_mode"
    detail["asof_date"] = asof_actual
    detail["rebalance_date"] = rebalance_ts
    detail["resid_cumret"] = detail["momentum_12_1_raw"]
    detail["resid_ir"] = detail["momentum_12_1_raw"]
    detail["resid_tstat"] = detail["momentum_12_1_raw"]
    detail["resid_cumret_raw_score"] = detail["momentum_12_1_score"]
    detail["resid_ir_raw_score"] = detail["momentum_12_1_score"]
    detail["resid_tstat_raw_score"] = detail["momentum_12_1_score"]
    detail["resid_cumret_neutralized_score"] = detail["momentum_12_1_score"]
    detail["resid_ir_neutralized_score"] = detail["momentum_12_1_score"]
    detail["resid_tstat_neutralized_score"] = detail["momentum_12_1_score"]

    minimal = build_residual_momentum_minimal(detail, rebalance_date=rebalance_ts)
    summary = {
        "factor_name": "momentum_12_1",
        "rebalance_date": rebalance_ts,
        "asof_date": asof_date,
        "input_universe_rows": int(len(coerce_universe_frame(universe))),
        "usable_universe_rows": int(len(prepared)),
        "detail_rows": int(len(detail)),
        "minimal_rows": int(len(minimal)),
        "ok_rows": int(detail["data_status"].eq("ok").sum()) if "data_status" in detail else 0,
        "missing_factor_rows": int(minimal["factor_value"].isna().sum()),
        "lookback_trading_days": lookback,
        "skip_recent_trading_days": skip_recent,
        "price_columns": int(close.shape[1]) if isinstance(close, pd.DataFrame) else 0,
        "data_source": "custom_close_loader" if cfg.get("close_loader") else "debug_yfinance_fallback",
        "jquants_api_key_supplied": bool(jquants_api_key),
        "params": asdict(params),
    }

    if params.save_detail_path:
        detail.to_csv(params.save_detail_path, index=False, encoding="utf-8-sig")
    if params.save_minimal_path:
        minimal.to_csv(params.save_minimal_path, index=False, encoding="utf-8-sig")

    return {"minimal": minimal, "detail": detail, "summary": summary}


def run_from_csv(
    input_csv: str | Path,
    output_csv: str | Path = "topix_residual_momentum_output.csv",
    rebalance_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    anchor = pd.Timestamp.today().normalize() if rebalance_date is None else pd.Timestamp(rebalance_date)
    result = run_residual_momentum_factor(
        universe=input_csv,
        rebalance_date=anchor,
        config={"save_detail_path": output_csv},
    )
    return result["detail"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate JP equity 12-1 momentum factor.")
    parser.add_argument("--input", required=True, help="Universe CSV path.")
    parser.add_argument("--output", default="topix_residual_momentum_output.csv", help="Detail output CSV path.")
    parser.add_argument("--minimal-output", default=None, help="Optional minimal output CSV path.")
    parser.add_argument("--rebalance-date", required=True, help="Rebalance/as-of date, e.g. 2026-03-17.")
    args = parser.parse_args()

    run_residual_momentum_factor(
        universe=args.input,
        rebalance_date=args.rebalance_date,
        config={
            "save_detail_path": args.output,
            "save_minimal_path": args.minimal_output,
        },
    )
