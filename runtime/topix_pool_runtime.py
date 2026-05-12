from __future__ import annotations

import io
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:  # yfinance is only needed by LiveTopixProvider, not historical backtests.
    import yfinance as yf
except ImportError:  # pragma: no cover - depends on optional debug dependency
    yf = None


USER_AGENT = {"User-Agent": "Mozilla/5.0"}


@dataclass
class MarketCapRule:
    mode: str = "rank"
    rank_min: int = 200
    rank_max: int = 500
    min_billion_jpy: float = 50.0
    max_billion_jpy: float = 5000.0


@dataclass
class LiquidityRule:
    lookback_days: int = 60
    min_avg_daily_volume: int = 0
    min_avg_daily_trading_value: float = 0.0


@dataclass
class VolatilityRule:
    lookback_days: int = 60
    max_annualized_volatility: float = 999.0


@dataclass
class SelectionRule:
    top_n_if_over_200: int = 500
    sort_by: str = "market_cap_desc"
    max_lot_cost_jpy: float | None = None


@dataclass
class ScreenRules:
    universe: str = "TOPIX_CURRENT_CONSTITUENTS"
    market_cap: MarketCapRule = field(default_factory=MarketCapRule)
    liquidity: LiquidityRule = field(default_factory=LiquidityRule)
    volatility: VolatilityRule = field(default_factory=VolatilityRule)
    selection: SelectionRule = field(default_factory=SelectionRule)

    def validate(self) -> None:
        if self.market_cap.mode not in {"rank", "range_billion_jpy", "none"}:
            raise ValueError(f"Unsupported market_cap.mode: {self.market_cap.mode}")
        if self.market_cap.rank_min <= 0 or self.market_cap.rank_max <= 0:
            raise ValueError("market_cap rank bounds must be positive")
        if self.market_cap.rank_min > self.market_cap.rank_max:
            raise ValueError("market_cap.rank_min must be <= market_cap.rank_max")
        if self.market_cap.min_billion_jpy > self.market_cap.max_billion_jpy:
            raise ValueError("market_cap.min_billion_jpy must be <= market_cap.max_billion_jpy")
        if self.liquidity.lookback_days <= 0 or self.volatility.lookback_days <= 0:
            raise ValueError("lookback days must be positive")
        if self.liquidity.min_avg_daily_volume < 0:
            raise ValueError("min_avg_daily_volume must be >= 0")
        if self.liquidity.min_avg_daily_trading_value < 0:
            raise ValueError("min_avg_daily_trading_value must be >= 0")
        if self.volatility.max_annualized_volatility <= 0:
            raise ValueError("max_annualized_volatility must be > 0")
        if self.selection.top_n_if_over_200 <= 0:
            raise ValueError("top_n_if_over_200 must be > 0")
        if self.selection.max_lot_cost_jpy is not None and self.selection.max_lot_cost_jpy <= 0:
            raise ValueError("selection.max_lot_cost_jpy must be > 0 when provided")
        if self.selection.sort_by not in {"market_cap_desc", "volume_desc", "volatility_asc"}:
            raise ValueError(f"Unsupported selection.sort_by: {self.selection.sort_by}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunIssue:
    stage: str
    level: str
    message: str
    code: str | None = None
    ticker: str | None = None
    exception_type: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunReport:
    mode: str
    anchor_date: str
    issues: list[RunIssue] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def bump(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + value

    def add_issue(
        self,
        stage: str,
        level: str,
        message: str,
        *,
        code: str | None = None,
        ticker: str | None = None,
        exception: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.issues.append(
            RunIssue(
                stage=stage,
                level=level,
                message=message,
                code=code,
                ticker=ticker,
                exception_type=type(exception).__name__ if exception else None,
                details=details or {},
            )
        )

    def to_summary_dict(self) -> dict[str, Any]:
        by_stage: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for issue in self.issues:
            by_stage[issue.stage] = by_stage.get(issue.stage, 0) + 1
            by_level[issue.level] = by_level.get(issue.level, 0) + 1
        return {
            "mode": self.mode,
            "anchor_date": self.anchor_date,
            "counters": dict(sorted(self.counters.items())),
            "issue_count": len(self.issues),
            "issues_by_stage": by_stage,
            "issues_by_level": by_level,
            "metadata": self.metadata,
        }

    def issues_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(
                columns=["stage", "level", "message", "code", "ticker", "exception_type", "details"]
            )
        rows = []
        for issue in self.issues:
            row = asdict(issue)
            row["details"] = str(row["details"])
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class ScreenRunResult:
    result: pd.DataFrame
    snapshot: pd.DataFrame
    report: RunReport


class MarketDataProvider(ABC):
    provider_name: str = "base"
    supports_historical: bool = False

    @abstractmethod
    def get_universe(self, asof_date: pd.Timestamp | None, report: RunReport) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_sector_map(
        self, codes: Iterable[str], asof_date: pd.Timestamp | None, report: RunReport
    ) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def get_market_caps(
        self, codes: Iterable[str], asof_date: pd.Timestamp | None, report: RunReport
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_price_panel(
        self, codes: Iterable[str], start: str, end: str, report: RunReport
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError


class HistoricalDataProvider(MarketDataProvider):
    provider_name = "historical-placeholder"
    supports_historical = True

    def get_universe(self, asof_date: pd.Timestamp | None, report: RunReport) -> list[str]:
        raise NotImplementedError("Plug in a point-in-time constituent source here.")

    def get_sector_map(
        self, codes: Iterable[str], asof_date: pd.Timestamp | None, report: RunReport
    ) -> dict[str, str]:
        raise NotImplementedError("Plug in a point-in-time sector source here.")

    def get_market_caps(
        self, codes: Iterable[str], asof_date: pd.Timestamp | None, report: RunReport
    ) -> pd.DataFrame:
        raise NotImplementedError("Plug in a point-in-time market-cap source here.")

    def get_price_panel(
        self, codes: Iterable[str], start: str, end: str, report: RunReport
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError("Plug in a historical OHLCV source here.")


def _first_existing(cols: Iterable[Any], candidates: Iterable[str]) -> Any | None:
    cmap = {str(col).strip().lower(): col for col in cols}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cmap:
            return cmap[key]
    return None


def _normalize_code(x: Any) -> str | None:
    match = re.search(r"(\d{4})", str(x).strip())
    return match.group(1) if match else None


def to_yf_ticker(code: str) -> str:
    return f"{code}.T"


def annualized_vol_from_close(close_series: pd.Series) -> float:
    returns = close_series.pct_change().dropna()
    if len(returns) < 10:
        return np.nan
    return float(returns.std() * np.sqrt(252))


def chunked(seq: list[str], n: int) -> Iterable[list[str]]:
    for idx in range(0, len(seq), n):
        yield seq[idx : idx + n]


class LiveTopixProvider(MarketDataProvider):
    provider_name = "live-jpx-yfinance"
    supports_historical = False

    def __init__(
        self,
        *,
        request_timeout: int = 30,
        market_cap_sleep_sec: float = 0.02,
        download_chunk_size: int = 100,
    ) -> None:
        self.request_timeout = request_timeout
        self.market_cap_sleep_sec = market_cap_sleep_sec
        self.download_chunk_size = download_chunk_size

    def get_universe(self, asof_date: pd.Timestamp | None, report: RunReport) -> list[str]:
        if asof_date is not None:
            report.add_issue(
                "universe",
                "warning",
                "Live provider ignores asof_date for constituents and uses current TOPIX members.",
                details={"requested_asof_date": str(asof_date.date())},
            )

        page_url = "https://www.jpx.co.jp/english/markets/indices/topix/"
        response = requests.get(page_url, headers=USER_AGENT, timeout=self.request_timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        csv_url = None
        for anchor in soup.find_all("a", href=True):
            text = " ".join(anchor.get_text(" ", strip=True).split())
            href = requests.compat.urljoin(page_url, anchor["href"])
            if ("Component Stocks Weight" in text or "TOPIX Component Stocks Weight" in text) and ".csv" in href.lower():
                csv_url = href
                break

        if csv_url is None:
            for anchor in soup.find_all("a", href=True):
                href = requests.compat.urljoin(page_url, anchor["href"])
                if href.lower().endswith(".csv") and "topix" in href.lower():
                    csv_url = href
                    break

        if csv_url is None:
            raise RuntimeError("Could not find TOPIX constituents CSV on JPX page.")

        raw = requests.get(csv_url, headers=USER_AGENT, timeout=self.request_timeout)
        raw.raise_for_status()

        frame = None
        for encoding in ("utf-8-sig", "cp932", "shift_jis", None):
            try:
                if encoding is None:
                    frame = pd.read_csv(io.BytesIO(raw.content))
                else:
                    frame = pd.read_csv(io.BytesIO(raw.content), encoding=encoding)
                break
            except Exception:
                continue

        if frame is None:
            raise RuntimeError("Failed to read TOPIX constituents CSV.")

        code_col = _first_existing(
            frame.columns,
            ["Code", "code", "Issue Code", "Local Code", "Security Code"],
        )
        if code_col is None:
            best_col = None
            best_score = -1
            for column in frame.columns:
                score = frame[column].astype(str).str.extract(r"(\d{4})", expand=False).notna().sum()
                if score > best_score:
                    best_score = score
                    best_col = column
            code_col = best_col

        if code_col is None:
            raise RuntimeError("Failed to infer TOPIX code column.")

        codes = (
            frame[code_col]
            .astype(str)
            .map(_normalize_code)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        if not codes:
            raise RuntimeError("Parsed TOPIX code list is empty.")

        report.metadata["topix_csv_url"] = csv_url
        report.bump("universe_codes", len(codes))
        return codes

    def get_sector_map(
        self, codes: Iterable[str], asof_date: pd.Timestamp | None, report: RunReport
    ) -> dict[str, str]:
        if asof_date is not None:
            report.add_issue(
                "sector",
                "warning",
                "Live provider ignores asof_date for sector mapping and uses current JPX issue list.",
                details={"requested_asof_date": str(asof_date.date())},
            )

        page_url = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html"
        response = requests.get(page_url, headers=USER_AGENT, timeout=self.request_timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        xls_url = None
        for anchor in soup.find_all("a", href=True):
            text = " ".join(anchor.get_text(" ", strip=True).split())
            href = requests.compat.urljoin(page_url, anchor["href"])
            if "List of TSE-listed Issues" in text and (href.lower().endswith(".xls") or href.lower().endswith(".xlsx")):
                xls_url = href
                break

        if xls_url is None:
            for anchor in soup.find_all("a", href=True):
                href = requests.compat.urljoin(page_url, anchor["href"])
                if href.lower().endswith(".xls") or href.lower().endswith(".xlsx"):
                    xls_url = href
                    break

        if xls_url is None:
            report.add_issue("sector", "warning", "Could not find JPX sector file; industry column will be blank.")
            return {}

        raw = requests.get(xls_url, headers=USER_AGENT, timeout=self.request_timeout)
        raw.raise_for_status()
        frame = pd.read_excel(io.BytesIO(raw.content), sheet_name=0)
        frame.columns = [str(col).strip() for col in frame.columns]

        code_col = _first_existing(frame.columns, ["Code", "code", "Local Code"])
        industry_col = _first_existing(
            frame.columns,
            ["33 Sector(name)", "17 Sector(name)", "Sector", "Industry", "33 Sector Name", "17 Sector Name"],
        )

        if code_col is None:
            for column in frame.columns:
                if frame[column].astype(str).str.contains(r"\b\d{4}\b", regex=True).any():
                    code_col = column
                    break

        if industry_col is None:
            for column in frame.columns:
                lower = column.lower()
                if "sector" in lower or "industry" in lower:
                    industry_col = column
                    break

        if code_col is None or industry_col is None:
            report.add_issue("sector", "warning", "Could not infer JPX sector columns; industry column will be blank.")
            return {}

        out = frame[[code_col, industry_col]].copy()
        out.columns = ["code", "industry"]
        out["code"] = out["code"].astype(str).map(_normalize_code)
        out = out.dropna().drop_duplicates("code")

        sector_map = dict(zip(out["code"], out["industry"]))
        missing = [code for code in codes if code not in sector_map]
        if missing:
            report.bump("sector_missing", len(missing))
            report.add_issue(
                "sector",
                "warning",
                "Some codes are missing sector mapping.",
                details={"missing_count": len(missing), "sample_codes": missing[:10]},
            )
        report.metadata["sector_xls_url"] = xls_url
        return sector_map

    def get_market_caps(
        self, codes: Iterable[str], asof_date: pd.Timestamp | None, report: RunReport
    ) -> pd.DataFrame:
        if yf is None:
            raise ImportError("yfinance is required only for LiveTopixProvider market-cap fallback.")
        if asof_date is not None:
            report.add_issue(
                "market_cap",
                "warning",
                "Live provider ignores asof_date for market cap and uses current yfinance values.",
                details={"requested_asof_date": str(asof_date.date())},
            )

        rows: list[dict[str, Any]] = []
        codes_list = list(codes)
        for idx, code in enumerate(codes_list, start=1):
            ticker = to_yf_ticker(code)
            market_cap = np.nan
            try:
                ticker_obj = yf.Ticker(ticker)
                try:
                    fast_info = ticker_obj.fast_info
                    market_cap = fast_info.get("market_cap", np.nan)
                except Exception as exc:
                    report.add_issue(
                        "market_cap",
                        "info",
                        "fast_info lookup failed; falling back to info.",
                        code=code,
                        ticker=ticker,
                        exception=exc,
                    )

                if pd.isna(market_cap):
                    try:
                        info = ticker_obj.info
                        market_cap = info.get("marketCap", np.nan)
                    except Exception as exc:
                        report.add_issue(
                            "market_cap",
                            "error",
                            "Failed to load market cap from yfinance.",
                            code=code,
                            ticker=ticker,
                            exception=exc,
                        )
            except Exception as exc:
                report.add_issue(
                    "market_cap",
                    "error",
                    "Failed to initialize yfinance ticker object.",
                    code=code,
                    ticker=ticker,
                    exception=exc,
                )

            if pd.isna(market_cap):
                report.bump("market_cap_missing")
            else:
                rows.append({"code": code, "market_cap_jpy": float(market_cap)})

            if idx % 100 == 0:
                print(f"[market cap] {idx}/{len(codes_list)} done")
            time.sleep(self.market_cap_sleep_sec)

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("No market caps were fetched.")

        frame = frame.sort_values("market_cap_jpy", ascending=False).reset_index(drop=True)
        frame["market_cap_rank"] = np.arange(1, len(frame) + 1)
        frame["market_cap_billion_jpy"] = frame["market_cap_jpy"] / 1e9
        report.bump("market_cap_available", len(frame))
        return frame

    def get_price_panel(
        self, codes: Iterable[str], start: str, end: str, report: RunReport
    ) -> dict[str, pd.DataFrame]:
        if yf is None:
            raise ImportError("yfinance is required only for LiveTopixProvider price fallback.")
        tickers = [to_yf_ticker(code) for code in codes]
        chunks: list[pd.DataFrame] = []

        for chunk in chunked(tickers, self.download_chunk_size):
            try:
                data = yf.download(
                    tickers=chunk,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                chunks.append(data)
            except Exception as exc:
                report.add_issue(
                    "ohlcv",
                    "error",
                    "Failed to download OHLCV chunk from yfinance.",
                    exception=exc,
                    details={"chunk_size": len(chunk), "sample_tickers": chunk[:10]},
                )
            time.sleep(0.2)

        if not chunks:
            raise RuntimeError("No OHLCV data was downloaded.")

        full = pd.concat(chunks, axis=1)
        out: dict[str, pd.DataFrame] = {}
        for code in codes:
            ticker = to_yf_ticker(code)
            try:
                if isinstance(full.columns, pd.MultiIndex):
                    sub = full[ticker].copy()
                else:
                    sub = full.copy()
                sub = sub.dropna(how="all")
                if sub.empty:
                    report.bump("ohlcv_missing")
                    report.add_issue("ohlcv", "warning", "Downloaded OHLCV frame is empty.", code=code, ticker=ticker)
                    continue
                out[code] = sub
            except Exception as exc:
                report.bump("ohlcv_missing")
                report.add_issue(
                    "ohlcv",
                    "error",
                    "Ticker OHLCV could not be extracted from the downloaded panel.",
                    code=code,
                    ticker=ticker,
                    exception=exc,
                )
        report.bump("ohlcv_available", len(out))
        return out


def _build_snapshot_vectorized(
    codes: list[str],
    sector_map: dict[str, str],
    market_cap_frame: pd.DataFrame,
    anchor_date: pd.Timestamp,
    rules: ScreenRules,
    price_frame: pd.DataFrame,
    report: RunReport,
    verbose: bool = True,
) -> pd.DataFrame:
    liq_lb = int(rules.liquidity.lookback_days)
    vol_lb = int(rules.volatility.lookback_days)
    min_required_rows = max(liq_lb, vol_lb) + 5

    def _log_snapshot(message: str) -> None:
        if verbose:
            print(f"[snapshot-debug] {message}", flush=True)

    _log_snapshot(
        "start "
        f"anchor_date={pd.Timestamp(anchor_date).date()} "
        f"initial_codes={len(codes)} sector_keys={len(sector_map)} "
        f"market_cap_rows={len(market_cap_frame)} price_frame_rows={len(price_frame)} "
        f"required_price_rows_per_code={min_required_rows}"
    )

    if price_frame is None or price_frame.empty:
        raise RuntimeError("Snapshot is empty after feature construction. price_frame is empty.")

    work = price_frame.copy()
    work["code"] = work["code"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.tz_localize(None)
    work = work.loc[work["code"].isin(set(codes)) & work["date"].notna()].copy()
    work = work.loc[work["date"] <= pd.Timestamp(anchor_date).normalize()].copy()

    if work.empty:
        raise RuntimeError(
            "Snapshot is empty after feature construction. "
            f"anchor_date={pd.Timestamp(anchor_date).date()}, all filtered price rows are empty."
        )

    price_date_min = work["date"].min()
    price_date_max = work["date"].max()

    # Keep only the minimal columns needed for the screening snapshot.
    keep_cols = ["code", "date"]
    for col in ["close", "volume"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            keep_cols.append(col)
    for col in ["adjustment_price_unreliable", "adjustment_price_issue"]:
        if col in work.columns:
            keep_cols.append(col)
    work = work[keep_cols].sort_values(["code", "date"]).reset_index(drop=True)

    code_order = pd.Index(pd.Series(codes, dtype="string").dropna().astype(str).unique(), name="code")
    diagnostics = {
        "initial_codes": len(codes),
        "after_sector_merge": 0,
        "after_market_cap_merge": 0,
        "after_price_feature_merge": 0,
        "after_history_length": 0,
        "after_close_volume": 0,
        "after_volatility": 0,
        "after_feature_construction": 0,
    }

    base = pd.DataFrame({"code": code_order.astype(str)})
    sector_df = pd.DataFrame({"code": list(sector_map.keys()), "industry": list(sector_map.values())})
    sector_df["code"] = sector_df["code"].astype(str)
    base = base.merge(sector_df, on="code", how="left")
    diagnostics["after_sector_merge"] = int(base["industry"].notna().sum())
    _log_snapshot(f"after sector merge rows={len(base)} matched_industry={diagnostics['after_sector_merge']}")

    cap_cols = [col for col in ["code", "market_cap_jpy", "market_cap_billion_jpy", "market_cap_rank"] if col in market_cap_frame.columns]
    caps = market_cap_frame[cap_cols].copy()
    caps["code"] = caps["code"].astype(str)
    base = base.merge(caps, on="code", how="left")
    diagnostics["after_market_cap_merge"] = int(base["market_cap_jpy"].notna().sum()) if "market_cap_jpy" in base.columns else 0
    _log_snapshot(f"after market cap merge rows={len(base)} matched_market_cap={diagnostics['after_market_cap_merge']}")

    per_code_rows = work.groupby("code").size().rename("price_rows")
    sufficient_history = per_code_rows[per_code_rows >= min_required_rows]
    diagnostics["after_history_length"] = int(sufficient_history.shape[0])

    close_nonnull = work.loc[work["close"].notna(), ["code", "date", "close"]].copy()
    volume_nonnull = work.loc[work["volume"].notna(), ["code", "date", "volume"]].copy()

    close_counts = close_nonnull.groupby("code").size().rename("close_rows")
    volume_counts = volume_nonnull.groupby("code").size().rename("volume_rows")
    eligible_close = set(close_counts[close_counts >= vol_lb].index.astype(str))
    eligible_volume = set(volume_counts[volume_counts >= liq_lb].index.astype(str))
    eligible_codes = sorted(set(sufficient_history.index.astype(str)) & eligible_close & eligible_volume)
    diagnostics["after_close_volume"] = len(eligible_codes)

    close_tail = close_nonnull.groupby("code", group_keys=False).tail(vol_lb + 5).copy()
    close_tail["return"] = close_tail.groupby("code")["close"].pct_change()
    volatility = (
        close_tail.groupby("code")["return"].std().mul(np.sqrt(252)).rename("annualized_volatility").reset_index()
    )
    valid_volatility = volatility.loc[volatility["annualized_volatility"].notna(), "code"].astype(str)
    diagnostics["after_volatility"] = int(valid_volatility.nunique())

    latest_close = close_nonnull.groupby("code", group_keys=False).tail(1)[["code", "close"]].rename(columns={"close": "close_latest"})
    avg_daily_volume = (
        volume_nonnull.groupby("code", group_keys=False).tail(liq_lb).groupby("code")["volume"].mean().rename("avg_daily_volume").reset_index()
    )
    trading_value_rows = work.loc[work["close"].notna() & work["volume"].notna(), ["code", "date", "close", "volume"]].copy()
    trading_value_rows["trading_value"] = trading_value_rows["close"] * trading_value_rows["volume"]
    avg_daily_trading_value = (
        trading_value_rows.groupby("code", group_keys=False)
        .tail(liq_lb)
        .groupby("code")["trading_value"]
        .mean()
        .rename("avg_daily_trading_value")
        .reset_index()
    )

    features = latest_close.merge(avg_daily_volume, on="code", how="outer")
    features = features.merge(avg_daily_trading_value, on="code", how="outer")
    features = features.merge(volatility, on="code", how="outer")
    if "adjustment_price_unreliable" in work.columns:
        unreliable_flags = (
            work.groupby("code")["adjustment_price_unreliable"]
            .max()
            .rename("adjustment_price_unreliable")
            .reset_index()
        )
        features = features.merge(unreliable_flags, on="code", how="left")
        features["adjustment_price_unreliable"] = features["adjustment_price_unreliable"].fillna(False).astype(bool)
    features["code"] = features["code"].astype(str)
    diagnostics["after_price_feature_merge"] = int(features["close_latest"].notna().sum())
    _log_snapshot(
        "after price feature merge "
        f"rows={len(features)} matched_close={int(features['close_latest'].notna().sum())} "
        f"matched_adv={int(features['avg_daily_trading_value'].notna().sum())} "
        f"matched_vol={int(features['annualized_volatility'].notna().sum())}"
    )

    frame = base.merge(features, on="code", how="left")
    frame["anchor_date"] = pd.Timestamp(anchor_date).strftime("%Y-%m-%d")
    frame["ticker"] = frame["code"].astype(str).map(to_yf_ticker)
    frame = frame.rename(columns={"close_latest": "close"})

    if eligible_codes:
        frame = frame.loc[frame["code"].isin(eligible_codes)].copy()
    else:
        frame = frame.iloc[0:0].copy()
    if "adjustment_price_unreliable" in frame.columns:
        frame = frame.loc[~frame["adjustment_price_unreliable"].fillna(False)].copy()
    frame = frame.loc[frame["annualized_volatility"].notna()].copy()
    diagnostics["after_feature_construction"] = len(frame)

    _log_snapshot(
        "stage counts "
        f"initial={diagnostics['initial_codes']} "
        f"after_sector_merge={diagnostics['after_sector_merge']} "
        f"after_market_cap_merge={diagnostics['after_market_cap_merge']} "
        f"after_price_feature_merge={diagnostics['after_price_feature_merge']} "
        f"after_history_length={diagnostics['after_history_length']} "
        f"after_close_volume={diagnostics['after_close_volume']} "
        f"after_volatility={diagnostics['after_volatility']} "
        f"after_feature_construction={diagnostics['after_feature_construction']}"
    )
    _log_snapshot(f"price_panel_date_range={price_date_min}..{price_date_max}")

    if not frame.empty:
        core_features = [
            "industry",
            "market_cap_jpy",
            "market_cap_billion_jpy",
            "market_cap_rank",
            "close",
            "avg_daily_volume",
            "avg_daily_trading_value",
            "annualized_volatility",
        ]
        stats = []
        for col in core_features:
            if col in frame.columns:
                non_null = int(frame[col].notna().sum())
                stats.append({"feature": col, "non_null": non_null, "nan_ratio": float(frame[col].isna().mean())})
        _log_snapshot(f"feature_missing_stats={stats}")
        dropna_cols = [
            "market_cap_jpy",
            "market_cap_rank",
            "close",
            "avg_daily_volume",
            "avg_daily_trading_value",
            "annualized_volatility",
        ]
        before_dropna = len(frame)
        frame = frame.dropna(subset=[col for col in dropna_cols if col in frame.columns]).copy()
        _log_snapshot(f"after dropna rows={len(frame)} from={before_dropna}")

    if frame.empty:
        required_history_start = pd.Timestamp(anchor_date).normalize() - pd.tseries.offsets.BDay(min_required_rows + 5)
        raise RuntimeError(
            "Snapshot is empty after feature construction. "
            f"anchor_date={pd.Timestamp(anchor_date).date()}, "
            f"required_history_start_for_first_anchor~={required_history_start.date()}, "
            f"price_panel_date_range={price_date_min}..{price_date_max}, "
            f"stage_counts={diagnostics}. "
            "Check whether local price cache starts too late, join keys differ, or filters removed all rows."
        )

    frame = frame.sort_values("market_cap_jpy", ascending=False).reset_index(drop=True)
    report.bump("snapshot_rows", len(frame))
    return frame


def build_snapshot(
    codes: list[str],
    sector_map: dict[str, str],
    market_cap_frame: pd.DataFrame,
    anchor_date: pd.Timestamp,
    rules: ScreenRules,
    price_panel: dict[str, pd.DataFrame] | pd.DataFrame,
    report: RunReport,
    verbose: bool = True,
) -> pd.DataFrame:
    if isinstance(price_panel, pd.DataFrame):
        return _build_snapshot_vectorized(
            codes=codes,
            sector_map=sector_map,
            market_cap_frame=market_cap_frame,
            anchor_date=anchor_date,
            rules=rules,
            price_frame=price_panel,
            report=report,
            verbose=verbose,
        )

    liq_lb = int(rules.liquidity.lookback_days)
    vol_lb = int(rules.volatility.lookback_days)
    min_required_rows = max(liq_lb, vol_lb) + 5

    def _log_snapshot(message: str) -> None:
        if verbose:
            print(f"[snapshot-debug] {message}", flush=True)

    _log_snapshot(
        "start "
        f"anchor_date={pd.Timestamp(anchor_date).date()} "
        f"initial_codes={len(codes)} sector_keys={len(sector_map)} "
        f"market_cap_rows={len(market_cap_frame)} price_panel_codes={len(price_panel)} "
        f"required_price_rows_per_code={min_required_rows}"
    )
    if codes:
        _log_snapshot(f"code dtype sample={[type(code).__name__ for code in codes[:5]]} values={codes[:5]}")
    if sector_map:
        sector_keys = list(sector_map.keys())[:5]
        _log_snapshot(f"sector key sample types={[type(key).__name__ for key in sector_keys]} values={sector_keys}")
    if not market_cap_frame.empty and "code" in market_cap_frame.columns:
        cap_sample = market_cap_frame["code"].head(5).tolist()
        _log_snapshot(
            f"market_cap code dtype={market_cap_frame['code'].dtype} sample_types={[type(v).__name__ for v in cap_sample]} sample={cap_sample}"
        )
    if price_panel:
        price_keys = list(price_panel.keys())[:5]
        _log_snapshot(f"price_panel key sample types={[type(key).__name__ for key in price_keys]} values={price_keys}")

    market_cap_map = market_cap_frame.set_index("code").to_dict("index")
    rows: list[dict[str, Any]] = []
    diagnostics = {
        "initial_codes": len(codes),
        "sector_matches": 0,
        "market_cap_matches": 0,
        "price_panel_matches": 0,
        "price_nonempty": 0,
        "after_anchor_nonempty": 0,
        "sufficient_history": 0,
        "complete_close_volume": 0,
        "valid_volatility": 0,
        "rows_built": 0,
    }
    skip_counts: dict[str, int] = {}
    feature_rows: list[dict[str, Any]] = []
    price_date_min: pd.Timestamp | None = None
    price_date_max: pd.Timestamp | None = None

    def _skip(reason: str) -> None:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
        report.bump(f"snapshot_skipped_{reason}")

    for idx, code in enumerate(codes, start=1):
        if code in sector_map:
            diagnostics["sector_matches"] += 1
        if code in market_cap_map:
            diagnostics["market_cap_matches"] += 1
        price_frame = price_panel.get(code)
        if price_frame is None or price_frame.empty:
            _skip("missing_price")
            continue
        diagnostics["price_panel_matches"] += 1

        px = price_frame.copy()
        px.index = pd.to_datetime(px.index).tz_localize(None)
        px = px.loc[px.index <= anchor_date].dropna(how="all")
        if not px.empty:
            diagnostics["price_nonempty"] += 1
            diagnostics["after_anchor_nonempty"] += 1
            local_min = px.index.min()
            local_max = px.index.max()
            price_date_min = local_min if price_date_min is None else min(price_date_min, local_min)
            price_date_max = local_max if price_date_max is None else max(price_date_max, local_max)

        if len(px) < min_required_rows:
            _skip("short_history")
            report.add_issue(
                "snapshot",
                "warning",
                "Price history is shorter than required lookback window.",
                code=code,
                ticker=to_yf_ticker(code),
                details={"available_rows": len(px), "required_rows": min_required_rows},
            )
            continue
        diagnostics["sufficient_history"] += 1

        close = px.get("Close", pd.Series(dtype=float)).dropna()
        volume = px.get("Volume", pd.Series(dtype=float)).dropna()
        if "adjustment_price_unreliable" in px.columns and bool(px["adjustment_price_unreliable"].fillna(False).any()):
            _skip("unreliable_adjusted_price")
            report.add_issue(
                "snapshot",
                "warning",
                "Adjusted price series failed continuity checks and was excluded from trading.",
                code=code,
                ticker=to_yf_ticker(code),
            )
            continue
        if len(close) < vol_lb or len(volume) < liq_lb:
            _skip("incomplete_series")
            report.add_issue(
                "snapshot",
                "warning",
                "Close or volume series is shorter than the configured lookback.",
                code=code,
                ticker=to_yf_ticker(code),
                details={"close_rows": len(close), "volume_rows": len(volume)},
            )
            continue
        diagnostics["complete_close_volume"] += 1

        market_cap_row = market_cap_map.get(code)
        if market_cap_row is None:
            _skip("missing_market_cap")
            continue

        avg_daily_volume = float(volume.tail(liq_lb).mean())
        avg_daily_trading_value = float((close * volume).tail(liq_lb).mean())
        annualized_volatility = annualized_vol_from_close(close.tail(vol_lb + 5))
        if pd.isna(annualized_volatility):
            _skip("nan_volatility")
            report.add_issue(
                "snapshot",
                "warning",
                "Volatility calculation returned NaN.",
                code=code,
                ticker=to_yf_ticker(code),
            )
            continue
        diagnostics["valid_volatility"] += 1

        row = {
                "anchor_date": anchor_date.strftime("%Y-%m-%d"),
                "code": code,
                "ticker": to_yf_ticker(code),
                "industry": sector_map.get(code, np.nan),
                "market_cap_jpy": float(market_cap_row["market_cap_jpy"]),
                "market_cap_billion_jpy": float(market_cap_row["market_cap_billion_jpy"]),
                "market_cap_rank": int(market_cap_row["market_cap_rank"]),
                "close": float(close.iloc[-1]),
                "avg_daily_volume": avg_daily_volume,
                "avg_daily_trading_value": avg_daily_trading_value,
                "annualized_volatility": float(annualized_volatility),
        }
        feature_rows.append(row)
        rows.append(row)
        diagnostics["rows_built"] += 1

        if verbose and idx % 200 == 0:
            print(f"[snapshot] {idx}/{len(codes)} done", flush=True)

    frame = pd.DataFrame(rows)
    _log_snapshot(
        "stage counts "
        f"initial={diagnostics['initial_codes']} "
        f"after_sector_key_match={diagnostics['sector_matches']} "
        f"after_market_cap_key_match={diagnostics['market_cap_matches']} "
        f"after_price_panel_match={diagnostics['price_panel_matches']} "
        f"after_price_nonempty={diagnostics['price_nonempty']} "
        f"after_anchor_filter={diagnostics['after_anchor_nonempty']} "
        f"after_history_length={diagnostics['sufficient_history']} "
        f"after_close_volume={diagnostics['complete_close_volume']} "
        f"after_market_cap_merge={diagnostics['complete_close_volume'] - skip_counts.get('missing_market_cap', 0)} "
        f"after_volatility={diagnostics['valid_volatility']} "
        f"after_feature_construction={len(frame)}"
    )
    _log_snapshot(f"skip_counts={skip_counts}")
    _log_snapshot(f"price_panel_date_range={price_date_min}..{price_date_max}")
    if not frame.empty:
        core_features = [
            "industry",
            "market_cap_jpy",
            "market_cap_billion_jpy",
            "market_cap_rank",
            "close",
            "avg_daily_volume",
            "avg_daily_trading_value",
            "annualized_volatility",
        ]
        stats = []
        for col in core_features:
            if col in frame.columns:
                non_null = int(frame[col].notna().sum())
                stats.append({"feature": col, "non_null": non_null, "nan_ratio": float(frame[col].isna().mean())})
        _log_snapshot(f"feature_missing_stats={stats}")
        dropna_cols = [
            "market_cap_jpy",
            "market_cap_rank",
            "close",
            "avg_daily_volume",
            "avg_daily_trading_value",
            "annualized_volatility",
        ]
        before_dropna = len(frame)
        after_dropna = int(frame.dropna(subset=[col for col in dropna_cols if col in frame.columns]).shape[0])
        _log_snapshot(f"after_dropna_core_features={after_dropna}/{before_dropna}")
    if frame.empty:
        required_history_start = pd.Timestamp(anchor_date).normalize() - pd.tseries.offsets.BDay(min_required_rows + 5)
        raise RuntimeError(
            "Snapshot is empty after feature construction. "
            f"anchor_date={pd.Timestamp(anchor_date).date()}, "
            f"required_history_start_for_first_anchor~={required_history_start.date()}, "
            f"price_panel_date_range={price_date_min}..{price_date_max}, "
            f"stage_counts={diagnostics}, skip_counts={skip_counts}. "
            "Check whether local price cache starts too late, join keys differ, or filters removed all rows."
        )

    frame = frame.sort_values("market_cap_jpy", ascending=False).reset_index(drop=True)
    report.bump("snapshot_rows", len(frame))
    return frame


def apply_rules(snapshot: pd.DataFrame, rules: ScreenRules, report: RunReport | None = None, verbose: bool = True) -> pd.DataFrame:
    df = snapshot.copy()
    if verbose:
        print(f"[snapshot-debug] apply_rules start rows={len(df)}", flush=True)
    if "adjustment_price_unreliable" in df.columns:
        df = df.loc[~df["adjustment_price_unreliable"].fillna(False)].copy()
        if verbose:
            print(f"[snapshot-debug] after data-quality filter rows={len(df)}", flush=True)

    if rules.market_cap.mode == "rank":
        df = df[
            (df["market_cap_rank"] >= int(rules.market_cap.rank_min))
            & (df["market_cap_rank"] <= int(rules.market_cap.rank_max))
        ]
        if verbose:
            print(f"[snapshot-debug] after market_cap rank filter rows={len(df)}", flush=True)
    elif rules.market_cap.mode == "range_billion_jpy":
        df = df[
            (df["market_cap_billion_jpy"] >= float(rules.market_cap.min_billion_jpy))
            & (df["market_cap_billion_jpy"] <= float(rules.market_cap.max_billion_jpy))
        ]
        if verbose:
            print(f"[snapshot-debug] after market_cap range filter rows={len(df)}", flush=True)

    df = df[df["avg_daily_volume"] >= int(rules.liquidity.min_avg_daily_volume)]
    if verbose:
        print(f"[snapshot-debug] after liquidity filter rows={len(df)}", flush=True)
    df = df[df["avg_daily_trading_value"] >= float(rules.liquidity.min_avg_daily_trading_value)]
    if verbose:
        print(f"[snapshot-debug] after trading-value filter rows={len(df)}", flush=True)
    df = df[df["annualized_volatility"] <= float(rules.volatility.max_annualized_volatility)]
    if verbose:
        print(f"[snapshot-debug] after volatility filter rows={len(df)}", flush=True)
    if rules.selection.max_lot_cost_jpy is not None:
        df = df[(pd.to_numeric(df["close"], errors="coerce") * 100.0) <= float(rules.selection.max_lot_cost_jpy)]
        if verbose:
            print(f"[snapshot-debug] after lot-cost filter rows={len(df)}", flush=True)

    if rules.selection.sort_by == "market_cap_desc":
        df = df.sort_values("market_cap_jpy", ascending=False)
    elif rules.selection.sort_by == "volume_desc":
        df = df.sort_values("avg_daily_trading_value", ascending=False)
    elif rules.selection.sort_by == "volatility_asc":
        df = df.sort_values("annualized_volatility", ascending=True)

    df = df.reset_index(drop=True)
    if len(df) > int(rules.selection.top_n_if_over_200):
        df = df.head(int(rules.selection.top_n_if_over_200)).copy()
    if verbose:
        print(f"[snapshot-debug] after final screening rules rows={len(df)}", flush=True)

    df["selection_rank"] = np.arange(1, len(df) + 1)
    if report is not None:
        report.bump("selected_rows", len(df))
    return df


def run_realtime_screen(
    *,
    rules: ScreenRules | None = None,
    provider: MarketDataProvider | None = None,
    anchor_date: str | pd.Timestamp | None = None,
    recent_anchor_days: int = 3,
    debug_small_sample: bool = False,
    debug_sample_size: int = 300,
    verbose: bool = True,
) -> ScreenRunResult:
    provider = provider or LiveTopixProvider()
    rules = rules or ScreenRules()
    rules.validate()

    if anchor_date is None:
        anchor_ts = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(recent_anchor_days))
    else:
        anchor_ts = pd.Timestamp(anchor_date).normalize()

    mode = "historical_point_in_time" if provider.supports_historical else "realtime_screen"
    report = RunReport(mode=mode, anchor_date=anchor_ts.strftime("%Y-%m-%d"))
    report.metadata.update(
        {
            "provider_name": provider.provider_name,
            "supports_historical": provider.supports_historical,
            "rules": rules.to_dict(),
        }
    )

    def _log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    _log("1) Loading universe ...")
    codes = provider.get_universe(anchor_ts, report)
    _log(f"Universe size: {len(codes)}")

    _log("2) Loading sector map ...")
    sector_map = provider.get_sector_map(codes, anchor_ts, report)
    _log(f"Sector rows: {len(sector_map)}")

    _log("3) Loading market caps ...")
    market_cap_frame = provider.get_market_caps(codes, anchor_ts, report)
    _log(f"Market caps available: {len(market_cap_frame)}")

    if debug_small_sample:
        sample_codes = market_cap_frame["code"].head(int(debug_sample_size)).tolist()
        codes = sample_codes
        market_cap_frame = market_cap_frame[market_cap_frame["code"].isin(sample_codes)].copy()
        report.bump("debug_sample_codes", len(sample_codes))
        report.add_issue(
            "debug",
            "info",
            "Debug sample mode is enabled; only the first market-cap-ranked codes are used.",
            details={"sample_size": len(sample_codes)},
        )
        _log(f"Debug sample size: {len(sample_codes)}")

    liq_lb = int(rules.liquidity.lookback_days)
    vol_lb = int(rules.volatility.lookback_days)
    lookback = max(liq_lb, vol_lb) + 40
    start = (anchor_ts - pd.Timedelta(days=int(lookback * 1.8))).strftime("%Y-%m-%d")
    end = (anchor_ts + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    required_price_rows = max(liq_lb, vol_lb) + 5
    required_history_start_for_anchor = anchor_ts - pd.tseries.offsets.BDay(required_price_rows + 5)
    if verbose:
        print(
            "[snapshot-debug] price request plan "
            f"anchor_date={anchor_ts.date()} request_start={start} request_end={end} "
            f"liquidity_lookback={liq_lb} volatility_lookback={vol_lb} "
            f"required_rows_per_code={required_price_rows} "
            f"required_history_start_for_first_anchor~={required_history_start_for_anchor.date()}",
            flush=True,
        )

    _log(f"4) Loading prices up to anchor_date={anchor_ts.date()} ...")
    price_panel: dict[str, pd.DataFrame] | pd.DataFrame
    if provider.supports_historical and hasattr(provider, "price_frame"):
        price_panel = provider.price_frame(codes, start=start, end=end)
        nonempty_codes = int(price_panel["code"].nunique()) if not price_panel.empty and "code" in price_panel.columns else 0
        report.bump("ohlcv_available", nonempty_codes)
        if verbose:
            row_counts = price_panel.groupby("code").size().tolist() if not price_panel.empty else []
            print(
                "[snapshot-debug] price_frame coverage "
                f"codes_requested={len(codes)} codes_returned={nonempty_codes} nonempty_codes={nonempty_codes} "
                f"date_range={(price_panel['date'].min() if not price_panel.empty else None)}..{(price_panel['date'].max() if not price_panel.empty else None)} "
                f"min_rows_per_code={(min(row_counts) if row_counts else 0)} "
                f"median_rows_per_code={(float(np.median(row_counts)) if row_counts else 0.0)}",
                flush=True,
            )
    else:
        price_panel = provider.get_price_panel(codes, start=start, end=end, report=report)
        if price_panel:
            min_dates = []
            max_dates = []
            nonempty_codes = 0
            row_counts = []
            for px in price_panel.values():
                if px is None or px.empty:
                    continue
                idx = pd.to_datetime(px.index).tz_localize(None)
                if len(idx):
                    nonempty_codes += 1
                    min_dates.append(idx.min())
                    max_dates.append(idx.max())
                    row_counts.append(len(px))
            if verbose:
                print(
                    "[snapshot-debug] price_panel coverage "
                    f"codes_requested={len(codes)} codes_returned={len(price_panel)} nonempty_codes={nonempty_codes} "
                    f"date_range={(min(min_dates) if min_dates else None)}..{(max(max_dates) if max_dates else None)} "
                    f"min_rows_per_code={(min(row_counts) if row_counts else 0)} "
                    f"median_rows_per_code={(float(np.median(row_counts)) if row_counts else 0.0)}",
                    flush=True,
                )
        elif verbose:
            print("[snapshot-debug] price_panel coverage codes_returned=0", flush=True)

    _log("5) Building snapshot ...")
    snapshot = build_snapshot(codes, sector_map, market_cap_frame, anchor_ts, rules, price_panel, report, verbose=verbose)
    _log(f"Snapshot rows: {len(snapshot)}")

    _log("6) Applying rules ...")
    result = apply_rules(snapshot, rules, report=report, verbose=verbose)
    _log(f"Selected rows: {len(result)}")

    return ScreenRunResult(result=result, snapshot=snapshot, report=report)


def export_run_outputs(
    run: ScreenRunResult,
    *,
    output_prefix: str,
    output_dir: str = ".",
) -> dict[str, str]:
    result_path = f"{output_dir}/{output_prefix}_result.csv"
    snapshot_path = f"{output_dir}/{output_prefix}_snapshot.csv"
    issues_path = f"{output_dir}/{output_prefix}_issues.csv"
    summary_path = f"{output_dir}/{output_prefix}_summary.json"

    run.result.to_csv(result_path, index=False, encoding="utf-8-sig")
    run.snapshot.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    run.report.issues_frame().to_csv(issues_path, index=False, encoding="utf-8-sig")
    pd.Series(run.report.to_summary_dict()).to_json(summary_path, force_ascii=False, indent=2)

    return {
        "result_csv": result_path,
        "snapshot_csv": snapshot_path,
        "issues_csv": issues_path,
        "summary_json": summary_path,
    }


def build_universe_minimal(
    selected: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build the stable downstream universe contract from the selected pool."""
    out = pd.DataFrame(index=selected.index)
    out["code"] = selected["code"].astype(str)
    # Historical J-Quants/local-cache joins use normalized JP stock codes as
    # keys.  Yahoo-style ".T" tickers are kept inside explicit yfinance fallback
    # loaders only, so they do not leak into the main backtest data contract.
    out["ticker"] = out["code"]
    out["industry"] = selected["industry"] if "industry" in selected.columns else pd.NA
    out["selection_rank"] = selected["selection_rank"] if "selection_rank" in selected.columns else np.arange(1, len(selected) + 1)
    out["anchor_date"] = pd.to_datetime(selected["anchor_date"], errors="coerce") if "anchor_date" in selected.columns else rebalance_date
    out["rebalance_date"] = rebalance_date

    optional_cols = [
        "market_cap_jpy",
        "market_cap_billion_jpy",
        "market_cap_rank",
        "close",
        "avg_daily_volume",
        "avg_daily_trading_value",
        "annualized_volatility",
    ]
    for col in optional_cols:
        if col in selected.columns:
            out[col] = selected[col]
    return out.reset_index(drop=True)


def run_universe(
    universe: pd.DataFrame | None = None,
    rebalance_date: str | pd.Timestamp | None = None,
    jquants_api_key: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a single-period JP equity universe with the standard runtime contract.

    The current production path wraps run_realtime_screen and uses the existing
    live JPX + yfinance provider.  Pass config["provider"] to inject a J-Quants
    or cached point-in-time provider later.  The universe argument is accepted
    for interface consistency and may be used as a prebuilt selected pool when
    config["mode"] == "from_frame".
    """
    cfg = config or {}
    mode = str(cfg.get("mode", "screen"))
    rebalance_ts = (
        pd.Timestamp.today().normalize()
        if rebalance_date is None
        else pd.Timestamp(rebalance_date).normalize()
    )

    if mode == "from_frame":
        if universe is None:
            raise ValueError("universe must be supplied when config['mode'] == 'from_frame'.")
        selected = universe.copy()
        if "code" not in selected.columns and "ticker" not in selected.columns:
            raise ValueError("from_frame universe must include code or ticker.")
        if "code" not in selected.columns:
            selected["code"] = selected["ticker"].map(_normalize_code)
        if "ticker" not in selected.columns:
            selected["ticker"] = selected["code"].map(to_yf_ticker)
        if "selection_rank" not in selected.columns:
            selected["selection_rank"] = np.arange(1, len(selected) + 1)
        if "anchor_date" not in selected.columns:
            selected["anchor_date"] = rebalance_ts.strftime("%Y-%m-%d")
        report = RunReport(mode="from_frame", anchor_date=rebalance_ts.strftime("%Y-%m-%d"))
        report.metadata.update(
            {
                "provider_name": "prebuilt_frame",
                "supports_historical": True,
                "jquants_api_key_supplied": bool(jquants_api_key),
            }
        )
        report.bump("selected_rows", len(selected))
        snapshot = selected.copy()
        run = ScreenRunResult(result=selected, snapshot=snapshot, report=report)
    else:
        rules = cfg.get("rules")
        if rules is None:
            rules = ScreenRules(
                market_cap=MarketCapRule(**cfg.get("market_cap", {})),
                liquidity=LiquidityRule(**cfg.get("liquidity", {})),
                volatility=VolatilityRule(**cfg.get("volatility", {})),
                selection=SelectionRule(**cfg.get("selection", {})),
            )
        provider = cfg.get("provider") or LiveTopixProvider(
            request_timeout=int(cfg.get("request_timeout", 30)),
            market_cap_sleep_sec=float(cfg.get("market_cap_sleep_sec", 0.02)),
            download_chunk_size=int(cfg.get("download_chunk_size", 100)),
        )
        run = run_realtime_screen(
            rules=rules,
            provider=provider,
            # Keep rebalance_date as the portfolio date; only pin the data
            # anchor when the user explicitly asks for one.
            anchor_date=cfg.get("anchor_date"),
            recent_anchor_days=int(cfg.get("recent_anchor_days", 3)),
            debug_small_sample=bool(cfg.get("debug_small_sample", False)),
            debug_sample_size=int(cfg.get("debug_sample_size", 300)),
        )
        run.report.metadata["jquants_api_key_supplied"] = bool(jquants_api_key)

    minimal = build_universe_minimal(run.result, rebalance_date=rebalance_ts)
    detail = run.result.copy()
    issues = run.report.issues_frame()
    summary = run.report.to_summary_dict()
    summary.update(
        {
            "universe_name": "topix_screen",
            "rebalance_date": rebalance_ts.strftime("%Y-%m-%d"),
            "minimal_rows": int(len(minimal)),
            "detail_rows": int(len(detail)),
            "snapshot_rows": int(len(run.snapshot)),
            "jquants_api_key_supplied": bool(jquants_api_key),
        }
    )

    output_paths: dict[str, str] = {}
    if bool(cfg.get("export", False)):
        output_prefix = str(cfg.get("output_prefix") or f"topix_live_{run.report.anchor_date.replace('-', '')}")
        output_dir = str(cfg.get("output_dir", "."))
        output_paths = export_run_outputs(run, output_prefix=output_prefix, output_dir=output_dir)

    return {
        "minimal": minimal,
        "detail": detail,
        "summary": summary,
        "snapshot": run.snapshot,
        "issues": issues,
        "report": run.report,
        "output_paths": output_paths,
    }
