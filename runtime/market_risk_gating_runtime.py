"""
Runtime wrapper for market-index risk gating.

This module extracts a portfolio-level risk score from a configurable
reference index. It is not a stock-level factor; it produces a single-row
risk overlay that the main backtest engine can apply to portfolio exposure.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


DEFAULT_CONFIG: dict[str, Any] = {
    "risk_index_ticker": "TOPIX",
    "start_date": "2014-01-01",
    "lookback_high": 252,
    "rv_window": 20,
    "rv_q_window": 252,
    "dd_thresholds": [0.10, 0.15, 0.20],
    "rv_q_thresholds": [0.82, 0.90, 0.96],
    "score_cap": 5,
    "upgrade_confirm_days": 1,
    "downgrade_confirm_days": 2,
    "use_refined_rv_gate": True,
    "trend_short_ma": 50,
    "trend_long_ma": 200,
    "trend_confirm_days": 3,
    "risk_off_score": 4,
    "exposure_map": {
        0: 1.00,
        1: 1.00,
        2: 0.85,
        3: 0.70,
        4: 0.50,
        5: 0.30,
    },
    "save_minimal_path": None,
    "save_detail_path": None,
}

RiskPriceLoader = Callable[[str, pd.Timestamp, pd.Timestamp, Optional[str], dict[str, Any]], pd.Series | pd.DataFrame]


def _merged_config(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    if "risk_index_ticker" not in merged or not merged.get("risk_index_ticker"):
        merged["risk_index_ticker"] = merged.get("nikkei_ticker", DEFAULT_CONFIG["risk_index_ticker"])
    # Keep the legacy key populated for backward compatibility with older outputs/configs.
    merged["nikkei_ticker"] = merged["risk_index_ticker"]
    return merged


def realized_vol_annualized(close: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(pd.to_numeric(close, errors="coerce")).diff()
    return log_ret.rolling(window).std() * np.sqrt(252.0)


def apply_confirmation(score: pd.Series, up_days: int = 2, down_days: int = 3) -> pd.Series:
    clean = pd.to_numeric(score, errors="coerce").dropna().astype(int)
    if clean.empty:
        return pd.Series(dtype="Int64", name="score_effective")

    effective: list[int] = []
    current = int(clean.iloc[0])
    effective.append(current)

    for i in range(1, len(clean)):
        candidate = int(clean.iloc[i])
        if candidate > current:
            if i - up_days + 1 >= 0 and (clean.iloc[i - up_days + 1 : i + 1] >= candidate).all():
                current = candidate
        elif candidate < current:
            if i - down_days + 1 >= 0 and (clean.iloc[i - down_days + 1 : i + 1] <= candidate).all():
                current = candidate
        effective.append(current)

    return pd.Series(effective, index=clean.index, name="score_effective", dtype="Int64")


def apply_trend_confirmation(raw_trend: pd.Series, confirm_days: int = 3) -> pd.Series:
    clean = pd.to_numeric(raw_trend, errors="coerce").dropna().astype(int)
    if clean.empty:
        return pd.Series(dtype="Int64", name="trend_effective")

    effective: list[int] = []
    current = int(clean.iloc[0])
    effective.append(current)
    for i in range(1, len(clean)):
        candidate = int(clean.iloc[i])
        if candidate != current:
            window = clean.iloc[max(0, i - confirm_days + 1) : i + 1]
            if len(window) >= confirm_days and (window == candidate).all():
                current = candidate
        effective.append(current)
    return pd.Series(effective, index=clean.index, name="trend_effective", dtype="Int64")


def compute_score_table(close: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame({"close": pd.to_numeric(close, errors="coerce")}).dropna()
    df = df.sort_index()

    lookback_high = int(config["lookback_high"])
    rv_window = int(config["rv_window"])
    rv_q_window = int(config["rv_q_window"])
    dd_thresholds = [float(x) for x in config["dd_thresholds"]]
    rv_q_thresholds = [float(x) for x in config["rv_q_thresholds"]]

    roll_high = df["close"].rolling(lookback_high).max()
    drawdown = (df["close"] / roll_high - 1.0).clip(lower=-1.0)

    dd_points = pd.Series(0, index=df.index, dtype=int)
    for threshold in dd_thresholds:
        dd_points += (drawdown <= -threshold).astype(int)

    rv = realized_vol_annualized(df["close"], rv_window)
    rv_quantiles = [rv.rolling(rv_q_window).quantile(q) for q in rv_q_thresholds]
    rv_points_ungated = sum((rv > q).astype(int) for q in rv_quantiles).astype("Int64")

    if bool(config["use_refined_rv_gate"]):
        gate_deep = drawdown <= -dd_thresholds[-1]
        gate_panic_raw = (drawdown <= -dd_thresholds[0]) & (rv > rv_quantiles[-1])
        gate_panic = gate_panic_raw.rolling(2).sum() >= 2
        rv_gate = gate_deep | gate_panic
        rv_points = rv_points_ungated.where(rv_gate, 0).astype("Int64")
    else:
        rv_gate = pd.Series(True, index=df.index)
        rv_points = rv_points_ungated

    raw_score = (dd_points.astype("Int64") + rv_points).astype("Int64")
    score = raw_score.clip(upper=int(config["score_cap"]))

    out = pd.DataFrame(
        {
            "close": df["close"],
            "roll_high_252": roll_high,
            "drawdown": drawdown,
            "dd_points": dd_points.astype("Int64"),
            "rv20_ann": rv,
            "rv_q_low": rv_quantiles[0],
            "rv_q_mid": rv_quantiles[1],
            "rv_q_high": rv_quantiles[2],
            "rv_gate": rv_gate.astype("boolean"),
            "rv_points": rv_points,
            "raw_score": raw_score,
            "score_0_5": score,
        }
    )
    out["score_effective"] = apply_confirmation(
        out["score_0_5"],
        up_days=int(config["upgrade_confirm_days"]),
        down_days=int(config["downgrade_confirm_days"]),
    )

    ma_short = out["close"].rolling(int(config["trend_short_ma"])).mean()
    ma_long = out["close"].rolling(int(config["trend_long_ma"])).mean()
    raw_trend = ((out["close"] > ma_long) & (ma_short > ma_long)).astype(int)
    out["ma_short"] = ma_short
    out["ma_long"] = ma_long
    out["trend_raw"] = raw_trend.astype("Int64")
    out["trend_effective"] = apply_trend_confirmation(raw_trend, int(config["trend_confirm_days"]))
    return out


def download_risk_index_close_yfinance(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    jquants_api_key: Optional[str],
    config: dict[str, Any],
) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required unless config['risk_price_loader'] is supplied.") from exc

    data = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][ticker] if ticker in data["Close"].columns else data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    close.name = ticker
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def load_risk_index_close(
    rebalance_date: pd.Timestamp,
    jquants_api_key: Optional[str],
    config: dict[str, Any],
) -> pd.Series:
    ticker = str(config["risk_index_ticker"])
    start_date = pd.Timestamp(config["start_date"]).normalize()
    end_date = pd.Timestamp(config.get("asof_date", rebalance_date)).normalize()

    loader: RiskPriceLoader | None = config.get("risk_price_loader")
    if loader is None:
        if not bool(config.get("allow_yfinance_fallback", False)):
            raise ValueError(
                "risk_price_loader is required for historical risk gating. "
                "Set allow_yfinance_fallback=True only for debug yfinance compatibility."
            )
        loader = download_risk_index_close_yfinance
    raw = loader(ticker, start_date, end_date, jquants_api_key, config)
    if isinstance(raw, pd.DataFrame):
        if "close" in raw.columns:
            close = raw["close"]
        elif "Close" in raw.columns:
            close = raw["Close"]
        else:
            close = raw.iloc[:, 0]
    else:
        close = raw
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.loc[close.index <= end_date].sort_index()


def exposure_multiplier(score_effective: int, config: dict[str, Any]) -> float:
    exposure_map = {int(k): float(v) for k, v in dict(config["exposure_map"]).items()}
    return float(exposure_map.get(int(score_effective), exposure_map.get(int(config["score_cap"]), 0.0)))


def build_minimal(score_row: pd.Series, rebalance_date: pd.Timestamp, config: dict[str, Any]) -> pd.DataFrame:
    score_effective = int(score_row["score_effective"])
    multiplier = exposure_multiplier(score_effective, config)
    risk_on = bool(score_effective < int(config["risk_off_score"]))
    trend_on = bool(int(score_row.get("trend_effective", 0)) == 1)
    signal_date = pd.Timestamp(score_row.name).normalize()
    ticker = str(config["risk_index_ticker"])
    return pd.DataFrame(
        [
            {
                "risk_model_name": "market_risk_gating",
                "signal_date": signal_date,
                "data_end_date": signal_date,
                "rebalance_date": rebalance_date,
                "risk_score_0_5": int(score_row["score_0_5"]),
                "risk_score_effective": score_effective,
                "risk_on": risk_on,
                "trend_on": trend_on,
                "gross_exposure_multiplier": multiplier,
                "equity_exposure_multiplier": multiplier,
                "cash_weight_floor": max(0.0, 1.0 - multiplier),
                "risk_index_ticker": ticker,
                "nikkei_ticker": ticker,
            }
        ]
    )


def run_market_risk_gating(
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = _merged_config(config)
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    asof_date = pd.Timestamp(cfg.get("asof_date", rebalance_ts)).normalize()
    close = load_risk_index_close(asof_date, jquants_api_key, cfg)
    detail = compute_score_table(close, cfg)
    detail = detail.loc[detail.index <= asof_date].copy()
    usable = detail.dropna(subset=["score_0_5", "score_effective"])
    if usable.empty:
        raise ValueError("No usable market risk score rows. Check price history and lookback windows.")

    latest = usable.iloc[-1]
    minimal = build_minimal(latest, rebalance_ts, cfg)
    ticker = str(cfg["risk_index_ticker"])
    summary = {
        "risk_model_name": "market_risk_gating",
        "rebalance_date": rebalance_ts,
        "asof_date": asof_date,
        "data_end_date": pd.Timestamp(latest.name).normalize(),
        "detail_rows": int(len(detail)),
        "minimal_rows": int(len(minimal)),
        "risk_score_0_5": int(minimal["risk_score_0_5"].iloc[0]),
        "risk_score_effective": int(minimal["risk_score_effective"].iloc[0]),
        "risk_on": bool(minimal["risk_on"].iloc[0]),
        "trend_on": bool(minimal["trend_on"].iloc[0]),
        "gross_exposure_multiplier": float(minimal["gross_exposure_multiplier"].iloc[0]),
        "risk_index_ticker": ticker,
        "nikkei_ticker": ticker,
        "data_source": "custom_loader" if config and config.get("risk_price_loader") else "debug_yfinance_fallback",
        "jquants_api_key_supplied": bool(jquants_api_key),
    }

    if cfg.get("save_detail_path"):
        detail.to_csv(cfg["save_detail_path"], index=True, encoding="utf-8-sig")
    if cfg.get("save_minimal_path"):
        minimal.to_csv(cfg["save_minimal_path"], index=False, encoding="utf-8-sig")

    return {"minimal": minimal, "detail": detail, "summary": summary}


def run_nikkei_risk_gating(
    rebalance_date: str | pd.Timestamp,
    jquants_api_key: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return run_market_risk_gating(rebalance_date=rebalance_date, jquants_api_key=jquants_api_key, config=config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate market-index portfolio-level risk gating.")
    parser.add_argument("--rebalance-date", required=True, help="Rebalance/as-of date.")
    parser.add_argument("--minimal-output", default=None, help="Optional minimal output CSV path.")
    parser.add_argument("--detail-output", default=None, help="Optional detail output CSV path.")
    args = parser.parse_args()

    run_market_risk_gating(
        rebalance_date=args.rebalance_date,
        config={
            "save_minimal_path": args.minimal_output,
            "save_detail_path": args.detail_output,
        },
    )
