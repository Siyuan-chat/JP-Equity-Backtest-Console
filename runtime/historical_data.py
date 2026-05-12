from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from topix_pool_runtime import HistoricalDataProvider, RunReport, _normalize_code


SCHEMAS: dict[str, set[str]] = {
    "prices": {"date", "code", "open", "high", "low", "close", "volume"},
    "universe": {"asof_date", "code", "in_universe"},
    "sector": {"asof_date", "code", "sector"},
    "market_cap": {"asof_date", "code", "market_cap"},
    "financial_summary": {"disclosed_date", "code"},
    "index_prices": {"date", "index_code", "close"},
    "margin": {"date", "code", "margin_buy_balance", "margin_sell_balance"},
}

ADJUSTMENT_RATIO_JUMP_UPPER = 3.0
ADJUSTMENT_RATIO_JUMP_LOWER = 1.0 / ADJUSTMENT_RATIO_JUMP_UPPER


@dataclass
class LocalDataPaths:
    """Local JPX/J-Quants style cache paths used by JpxHistoricalProvider.

    Each path may point to parquet or csv.  When omitted, the provider looks for
    `<name>.parquet`, then `<name>.csv`, under data_dir.
    """

    data_dir: str | Path = "data"
    prices: str | Path | None = None
    universe: str | Path | None = None
    sector: str | Path | None = None
    market_cap: str | Path | None = None
    financial_summary: str | Path | None = None
    index_prices: str | Path | None = None
    margin: str | Path | None = None


def normalize_code(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 4 and text.isdigit():
            return text
        if len(text) >= 4 and text[:4].isdigit() and all(ch in {".", "-", "_"} or ch.isalpha() for ch in text[4:]):
            return text[:4]
    elif isinstance(value, (int, np.integer)):
        if 1000 <= int(value) <= 9999:
            return f"{int(value):04d}"
    elif isinstance(value, float) and np.isfinite(value) and float(value).is_integer():
        int_value = int(value)
        if 1000 <= int_value <= 9999:
            return f"{int_value:04d}"
    return _normalize_code(value)


def _lower_rename(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(col).strip() for col in out.columns]
    lower_map = {col: col.strip().lower() for col in out.columns}
    out = out.rename(columns=lower_map)
    aliases = {
        "localcode": "code",
        "local_code": "code",
        "stock_code": "code",
        "date": "date",
        "asofdate": "asof_date",
        "as_of_date": "asof_date",
        "discloseddate": "disclosed_date",
        "disclosure_date": "disclosed_date",
        "published_date": "disclosed_date",
        "sector33": "sector",
        "industry": "sector",
        "market_cap_jpy": "market_cap",
        "marketcapitalization": "market_cap",
        "adjustmentclose": "adjustment_close",
        "adjusted_close": "adjustment_close",
        "sharesoutstanding": "shares_outstanding",
        "listedshares": "shares_outstanding",
        "index_name": "index_code",
    }
    return out.rename(columns={col: aliases.get(col, col) for col in out.columns})


def read_local_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, dtype=str, low_memory=False)
    raise ValueError(f"Unsupported local data file type: {path}")


def _flag_unreliable_adjusted_prices(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "code" not in frame.columns:
        return frame
    if "adjustment_close" not in frame.columns or "close" not in frame.columns:
        out = frame.copy()
        out["adjustment_price_unreliable"] = False
        out["adjustment_price_issue"] = pd.NA
        return out

    out = frame.sort_values(["code", "date"]).copy()
    raw_close = pd.to_numeric(out["close"], errors="coerce")
    adj_close = pd.to_numeric(out["adjustment_close"], errors="coerce")
    adjustment_factor = (
        pd.to_numeric(out["adjustment_factor"], errors="coerce")
        if "adjustment_factor" in out.columns
        else pd.Series(1.0, index=out.index, dtype=float)
    )
    ratio = (adj_close / raw_close).where((raw_close > 0) & (adj_close > 0))
    prev_ratio = ratio.groupby(out["code"]).shift(1)
    ratio_jump = (ratio / prev_ratio).where(prev_ratio > 0)
    bad_jump = ratio_jump.gt(ADJUSTMENT_RATIO_JUMP_UPPER) | ratio_jump.lt(ADJUSTMENT_RATIO_JUMP_LOWER)
    supported_corporate_action = adjustment_factor.ne(1.0) & adjustment_factor.notna() & adjustment_factor.gt(0)
    forward_action_support = (
        supported_corporate_action.groupby(out["code"]).shift(-1, fill_value=False)
        | supported_corporate_action.groupby(out["code"]).shift(-2, fill_value=False)
        | supported_corporate_action.groupby(out["code"]).shift(-3, fill_value=False)
        | supported_corporate_action.groupby(out["code"]).shift(-4, fill_value=False)
        | supported_corporate_action.groupby(out["code"]).shift(-5, fill_value=False)
    )
    supported_corporate_action = supported_corporate_action | forward_action_support
    bad_without_action = bad_jump.fillna(False) & ~supported_corporate_action
    unreliable_codes = set(out.loc[bad_without_action, "code"].astype(str))

    out["adjustment_price_unreliable"] = out["code"].astype(str).isin(unreliable_codes)
    out["adjustment_price_issue"] = pd.NA
    if unreliable_codes:
        out.loc[out["adjustment_price_unreliable"], "adjustment_price_issue"] = "adjustment_ratio_jump"
        for col in [
            "adjustment_open",
            "adjustment_high",
            "adjustment_low",
            "adjustment_close",
            "adjustment_volume",
        ]:
            if col in out.columns:
                out.loc[out["adjustment_price_unreliable"], col] = np.nan
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def validate_schema(frame: pd.DataFrame, schema_name: str, path: str | Path | None = None) -> None:
    required = SCHEMAS[schema_name]
    missing = sorted(required - set(frame.columns))
    if missing:
        location = f" in {path}" if path is not None else ""
        raise ValueError(f"{schema_name} table is missing required columns{location}: {missing}")


def validate_nonempty_table(frame: pd.DataFrame, schema_name: str, path: str | Path | None = None) -> None:
    if frame.empty:
        location = f" in {path}" if path is not None else ""
        raise ValueError(f"{schema_name} table is empty{location}. Rebuild the J-Quants cache and inspect cache diagnostics.")
    if "code" in frame.columns and frame["code"].notna().sum() == 0:
        location = f" in {path}" if path is not None else ""
        raise ValueError(f"{schema_name} table has no valid code values{location}.")


def _coerce_date_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["date", "asof_date", "disclosed_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.tz_localize(None)
    if "code" in out.columns:
        out["code"] = out["code"].map(normalize_code)
    return out


def _deduplicate_table(frame: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    out = frame.copy()
    if schema_name == "prices" and {"date", "code"}.issubset(out.columns):
        out = out.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last")
    elif schema_name in {"universe", "sector", "market_cap"} and {"asof_date", "code"}.issubset(out.columns):
        out = out.sort_values(["asof_date", "code"]).drop_duplicates(subset=["asof_date", "code"], keep="last")
    elif schema_name == "financial_summary" and {"disclosed_date", "code"}.issubset(out.columns):
        out = out.sort_values(["disclosed_date", "code"]).drop_duplicates(subset=["disclosed_date", "code"], keep="last")
    elif schema_name == "index_prices" and {"date", "index_code"}.issubset(out.columns):
        out = out.sort_values(["date", "index_code"]).drop_duplicates(subset=["date", "index_code"], keep="last")
    elif schema_name == "margin" and {"date", "code"}.issubset(out.columns):
        out = out.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last")
    return out.reset_index(drop=True)


def _latest_per_code(frame: pd.DataFrame, date_col: str, asof_date: pd.Timestamp, codes: Iterable[str] | None = None) -> pd.DataFrame:
    work = frame.loc[frame[date_col] <= asof_date].copy()
    if codes is not None:
        code_set = set(codes)
        work = work.loc[work["code"].isin(code_set)]
    if work.empty:
        return work
    return work.sort_values(["code", date_col]).groupby("code", as_index=False).tail(1)


@dataclass
class JpxHistoricalProvider(HistoricalDataProvider):
    """Point-in-time historical provider backed by local parquet/csv caches.

    The class intentionally avoids web scraping and yfinance.  It expects local
    monthly/daily snapshots exported from JPX/J-Quants compatible sources.
    Universe defaults to `latest_snapshot`: use the most recent constituent
    snapshot whose asof_date is <= signal date, then keep rows where
    in_universe is true.
    """

    paths: LocalDataPaths | Mapping[str, Any] = field(default_factory=LocalDataPaths)
    universe_mode: str = "latest_snapshot"
    provider_name: str = "jpx-local-historical"
    supports_historical: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.paths, Mapping):
            self.paths = LocalDataPaths(**dict(self.paths))
        self.data_dir = Path(self.paths.data_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def _resolve_path(self, name: str) -> Path:
        configured = getattr(self.paths, name)
        if configured is not None:
            path = Path(configured)
            return path if path.is_absolute() else self.data_dir / path
        for suffix in (".parquet", ".csv"):
            candidate = self.data_dir / f"{name}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No local {name} cache found under {self.data_dir}")

    def table(self, name: str, *, required: bool = True, copy: bool = True) -> pd.DataFrame:
        if name in self._cache:
            frame = self._cache[name]
            return frame.copy(deep=False) if copy else frame
        try:
            path = self._resolve_path(name)
            frame = _coerce_date_columns(_lower_rename(read_local_table(path)))
            frame = _deduplicate_table(frame, name)
            validate_schema(frame, name, path)
            if name in {"universe", "sector", "market_cap", "prices"}:
                validate_nonempty_table(frame, name, path)
        except FileNotFoundError:
            if required:
                raise
            return pd.DataFrame()
        self._cache[name] = frame
        return frame.copy(deep=False) if copy else frame

    def get_universe(self, asof_date: pd.Timestamp | None, report: RunReport) -> list[str]:
        asof = pd.Timestamp(asof_date).normalize()
        uni = self.table("universe", copy=False)
        if self.universe_mode == "latest_by_code":
            latest = _latest_per_code(uni, "asof_date", asof)
        else:
            snapshots = uni.loc[uni["asof_date"] <= asof, "asof_date"].dropna()
            if snapshots.empty:
                raise ValueError(f"No universe snapshot on or before {asof.date()}")
            snapshot_date = snapshots.max()
            latest = uni.loc[uni["asof_date"].eq(snapshot_date)].copy()
            report.metadata["universe_snapshot_date"] = str(snapshot_date.date())
        latest["in_universe"] = latest["in_universe"].astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
        codes = latest.loc[latest["in_universe"] & latest["code"].notna(), "code"].drop_duplicates().tolist()
        report.bump("universe_codes", len(codes))
        return codes

    def get_sector_map(self, codes: Iterable[str], asof_date: pd.Timestamp | None, report: RunReport) -> dict[str, str]:
        asof = pd.Timestamp(asof_date).normalize()
        sector = self.table("sector", copy=False)
        latest = _latest_per_code(sector, "asof_date", asof, codes)
        out = dict(zip(latest["code"], latest["sector"]))
        missing = [code for code in codes if code not in out]
        if missing:
            report.bump("sector_missing", len(missing))
            report.add_issue("sector", "warning", "Missing point-in-time sector rows.", details={"sample_codes": missing[:10]})
        return out

    def get_market_caps(self, codes: Iterable[str], asof_date: pd.Timestamp | None, report: RunReport) -> pd.DataFrame:
        asof = pd.Timestamp(asof_date).normalize()
        caps = self.table("market_cap", copy=False)
        latest = _latest_per_code(caps, "asof_date", asof, codes)
        if latest.empty:
            raise ValueError(f"No market-cap rows on or before {asof.date()}")
        latest = latest.rename(columns={"market_cap": "market_cap_jpy"}).copy()
        latest["market_cap_jpy"] = pd.to_numeric(latest["market_cap_jpy"], errors="coerce")
        latest = latest.dropna(subset=["code"]).sort_values("market_cap_jpy", ascending=False, na_position="last")
        latest["market_cap_rank"] = np.arange(1, len(latest) + 1)
        latest["market_cap_billion_jpy"] = latest["market_cap_jpy"] / 1e9
        report.bump("market_cap_available", len(latest))
        cols = ["code", "market_cap_jpy", "market_cap_billion_jpy", "market_cap_rank"]
        if "shares_outstanding" in latest.columns:
            cols.append("shares_outstanding")
        return latest[cols].reset_index(drop=True)

    def price_frame(self, codes: Iterable[str], start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        prices = self.table("prices", copy=False)
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        code_set = set(codes)
        work = prices.loc[prices["code"].isin(code_set)].copy()
        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjustment_factor",
            "adjustment_open",
            "adjustment_high",
            "adjustment_low",
            "adjustment_close",
            "adjustment_volume",
        ]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        work = _flag_unreliable_adjusted_prices(work)
        work = work.loc[work["date"].between(start_ts, end_ts, inclusive="both")].copy()
        return work.sort_values(["code", "date"]).reset_index(drop=True)

    def get_price_panel(self, codes: Iterable[str], start: str, end: str, report: RunReport) -> dict[str, pd.DataFrame]:
        prices = self.price_frame(codes, start, end)
        out: dict[str, pd.DataFrame] = {}
        rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        for code, group in prices.groupby("code"):
            sub = group.set_index("date").rename(columns=rename)
            keep = [
                col
                for col in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "adjustment_close",
                    "adjustment_factor",
                    "adjustment_price_unreliable",
                    "adjustment_price_issue",
                ]
                if col in sub.columns
            ]
            out[str(code)] = sub[keep].sort_index()
        report.bump("ohlcv_available", len(out))
        return out

    def load_price_history(
        self,
        universe_codes: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        jquants_api_key: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        codes = universe_codes["code"].dropna().astype(str).tolist()
        return self.price_frame(codes, start_date, end_date)

    def load_attention_inputs(
        self,
        universe_codes: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        jquants_api_key: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        prices = self.price_frame(universe_codes["code"], start_date, end_date)
        caps = self.table("market_cap", required=False, copy=False)
        if caps.empty:
            prices["shares_outstanding"] = np.nan
            return prices
        caps = caps.loc[caps["asof_date"] <= pd.Timestamp(end_date).normalize()].sort_values(["code", "asof_date"])
        latest = caps.groupby("code", as_index=False).tail(1)[["code", "market_cap", *([ "shares_outstanding"] if "shares_outstanding" in caps.columns else [])]]
        merged = prices.merge(latest, on="code", how="left")
        if "shares_outstanding" not in merged.columns:
            merged["shares_outstanding"] = np.nan
        missing_shares = merged["shares_outstanding"].isna()
        merged.loc[missing_shares, "shares_outstanding"] = merged.loc[missing_shares, "market_cap"] / merged.loc[missing_shares, "close"].replace(0, np.nan)
        return merged

    def load_behaviour_inputs(
        self,
        universe_codes: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        jquants_api_key: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        codes = universe_codes["code"].dropna().astype(str).tolist()
        margin = self.table("margin", copy=False)
        margin = margin.loc[margin["code"].isin(codes) & margin["date"].between(start_date, end_date, inclusive="both")].copy()
        report = RunReport(mode="provider_internal", anchor_date=str(pd.Timestamp(end_date).date()))
        caps = self.get_market_caps(codes, end_date, report)[["code", "market_cap_jpy"]]
        prices = self.price_frame(codes, pd.Timestamp(end_date) - pd.Timedelta(days=120), end_date)
        if not prices.empty:
            prices["trading_value"] = prices["close"] * prices["volume"]
            adv = prices.groupby("code")["trading_value"].mean().rename("avg_daily_trading_value").reset_index()
            margin = margin.merge(adv, on="code", how="left")
        return margin.merge(caps, on="code", how="left")

    def load_fundamental_panel(self, universe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        asof = pd.Timestamp(config.get("asof_date") or config.get("rebalance_date")).normalize()
        codes = universe["code"].dropna().astype(str).tolist()
        fin = self.table("financial_summary", copy=False)
        fin = _latest_per_code(fin, "disclosed_date", asof, codes)
        if fin.empty:
            out = pd.DataFrame({"code": codes})
            report = RunReport(mode="provider_internal", anchor_date=str(asof.date()))
            try:
                caps = self.get_market_caps(codes, asof, report)[["code", "market_cap_jpy"]].rename(columns={"market_cap_jpy": "MarketCap"})
                out = out.merge(caps, on="code", how="left")
            except Exception:
                out["MarketCap"] = np.nan
            for col in [
                "StatementDate",
                "TotalRevenue",
                "GrossProfit",
                "OperatingIncome",
                "EBIT",
                "EBITDA",
                "NetIncome",
                "TotalAssets",
                "StockholdersEquity",
                "TotalDebt",
                "CashAndCashEquivalents",
                "OperatingCashFlow",
                "FreeCashFlow",
            ]:
                out[col] = np.nan
            return out
        rename = {
            "disclosed_date": "StatementDate",
            "market_cap": "MarketCap",
            "total_revenue": "TotalRevenue",
            "gross_profit": "GrossProfit",
            "operating_income": "OperatingIncome",
            "ebit": "EBIT",
            "ebitda": "EBITDA",
            "net_income": "NetIncome",
            "total_assets": "TotalAssets",
            "stockholders_equity": "StockholdersEquity",
            "total_debt": "TotalDebt",
            "cash_and_cash_equivalents": "CashAndCashEquivalents",
            "operating_cash_flow": "OperatingCashFlow",
            "free_cash_flow": "FreeCashFlow",
        }
        out = fin.rename(columns={k: v for k, v in rename.items() if k in fin.columns})
        if "MarketCap" not in out.columns:
            report = RunReport(mode="provider_internal", anchor_date=str(asof.date()))
            caps = self.get_market_caps(codes, asof, report)[["code", "market_cap_jpy"]].rename(columns={"market_cap_jpy": "MarketCap"})
            out = out.merge(caps, on="code", how="left")
        for col in [
            "StatementDate",
            "TotalRevenue",
            "GrossProfit",
            "OperatingIncome",
            "EBIT",
            "EBITDA",
            "NetIncome",
            "TotalAssets",
            "StockholdersEquity",
            "TotalDebt",
            "CashAndCashEquivalents",
            "OperatingCashFlow",
            "FreeCashFlow",
            "MarketCap",
        ]:
            if col not in out.columns:
                out[col] = np.nan
        return out

    def load_index_close(self, index_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp, config: dict[str, Any]) -> pd.Series:
        idx = self.table("index_prices", copy=False)
        work = idx.loc[
            idx["index_code"].astype(str).str.upper().eq(str(index_code).upper())
            & idx["date"].between(start_date, end_date, inclusive="both")
        ].copy()
        close = pd.to_numeric(work["close"], errors="coerce")
        close.index = pd.to_datetime(work["date"])
        close = close.dropna().sort_index()
        close.name = index_code
        return close

    def load_close_prices(self, tickers: Iterable[str], start_date: pd.Timestamp, end_date: pd.Timestamp, config: dict[str, Any]) -> pd.DataFrame:
        """Return a wide close-price table for residual momentum style loaders.

        Four-digit JP stock tickers are read from prices.  Non-stock identifiers
        are treated as index_code values in index_prices, which lets backtests
        use TOPIX/Nikkei local histories instead of ETF/yfinance substitutes.
        """

        requested = list(dict.fromkeys(str(t) for t in tickers))
        stock_codes = [normalize_code(t) for t in requested if normalize_code(t) is not None]
        stock_prices = self.price_frame(stock_codes, start_date, end_date) if stock_codes else pd.DataFrame()
        series: dict[str, pd.Series] = {}
        ticker_to_code = {t: normalize_code(t) for t in requested}
        if not stock_prices.empty:
            for ticker, code in ticker_to_code.items():
                if code is None:
                    continue
                group = stock_prices.loc[stock_prices["code"].eq(code)].copy()
                if group.empty:
                    continue
                close_col = "adjustment_close" if "adjustment_close" in group.columns and group["adjustment_close"].notna().any() else "close"
                s = pd.to_numeric(group[close_col], errors="coerce")
                s.index = pd.to_datetime(group["date"])
                series[ticker] = s.sort_index()
        for ticker in requested:
            if ticker in series:
                continue
            try:
                series[ticker] = self.load_index_close(ticker, start_date, end_date, config)
            except Exception:
                continue
        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1).sort_index()
