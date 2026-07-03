from __future__ import annotations

import os
import re
import time
import threading
import zipfile
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from historical_data import JpxHistoricalProvider, LocalDataPaths, normalize_code


JQUANTS_CACHE_BUILDER_VERSION = "2026-04-18-full-month-cache"


EQUITIES_DAILY_BARS_ENDPOINT = "/equities/bars/daily"
MARGIN_DATASET_ENDPOINTS = {"daily_margin_interest", "weekly_margin_interest"}
MARGIN_ENDPOINT_PATHS = {"/markets/margin-alert", "/markets/margin-interest"}


def _cell5_log(message: str) -> None:
    ts = pd.Timestamp.now().strftime("%H:%M:%S")
    print(f"[{ts}] [cell5] {message}", flush=True)


def infer_required_history_days(overrides: dict[str, Any] | None = None) -> int:
    """Return the minimum calendar warmup needed by the current factor stack.

    This keeps the cache window tied to actual model inputs instead of blindly
    extending every run by a hard-coded year-plus buffer.
    """

    overrides = dict(overrides or {})
    enabled_factors = set(overrides.get("enabled_factors") or ())
    use_all_factors = not enabled_factors
    use_risk_gating = bool(overrides.get("use_risk_gating", True))
    def trading_to_calendar(days: int, buffer_calendar_days: int = 20) -> int:
        return int(np.ceil(int(days) * 7.0 / 5.0 + int(buffer_calendar_days)))

    screen_days = max(
        int(overrides.get("screen_history_days", 0)),
        trading_to_calendar(
            max(
                int(overrides.get("screen_liquidity_lookback_days", 60)),
                int(overrides.get("screen_volatility_lookback_days", 60)),
            )
            + 10,
            40,
        ),
    )
    dual_ma_days = max(
        int(overrides.get("dual_ma_history_buffer_calendar_days", 0)),
        trading_to_calendar(int(overrides.get("dual_ma_long_window", 200)) + 10, 30),
    )
    reversal_days = max(
        int(overrides.get("reversal_history_buffer_calendar_days", 0)),
        trading_to_calendar(
            int(overrides.get("reversal_lookback_window", 20)) + int(overrides.get("reversal_skip_recent_days", 5)) + 10,
            30,
        ),
    )
    residual_days = max(
        int(overrides.get("residual_momentum_history_buffer_calendar_days", 0)),
        trading_to_calendar(
            int(overrides.get("residual_momentum_beta_window", 60))
            + int(overrides.get("residual_momentum_signal_window", 60))
            + int(overrides.get("residual_momentum_skip_recent", 20))
            + 20,
            40,
        ),
    )
    attention_days = max(
        int(overrides.get("attention_history_buffer_calendar_days", 0)),
        trading_to_calendar(int(overrides.get("attention_months", 12)) * 21 + 20, 45),
    )
    behaviour_days = max(
        int(overrides.get("behaviour_history_days", 0)),
        int(overrides.get("behaviour_weeks", 26)) * 7 + 14,
    )
    risk_days = max(
        int(overrides.get("risk_gate_history_days", 0)),
        trading_to_calendar(
            max(
                int(overrides.get("risk_gate_lookback_high", 252)),
                int(overrides.get("risk_gate_rv_window", 20)) + int(overrides.get("risk_gate_rv_q_window", 252)),
                int(overrides.get("risk_gate_trend_long_ma", 200)),
            )
            + int(overrides.get("risk_gate_trend_confirm_days", 3)),
            40,
        ),
    )
    all_candidates = {
        "screen_liquidity_volatility": screen_days,
        "dual_ma": dual_ma_days,
        "reversal": reversal_days,
        "attention": attention_days,
        "residual_momentum": residual_days,
        "behaviour": behaviour_days,
        "risk_gate": risk_days,
    }
    candidates = {"screen_liquidity_volatility": all_candidates["screen_liquidity_volatility"]}
    for name in ("dual_ma", "reversal", "attention", "residual_momentum", "behaviour"):
        if use_all_factors or name in enabled_factors:
            candidates[name] = all_candidates[name]
    if use_risk_gating:
        candidates["risk_gate"] = all_candidates["risk_gate"]
    required = max(candidates.values())
    _cell5_log(f"required history window days={required} components={candidates}")
    return required


def _is_month_start(value: pd.Timestamp) -> bool:
    ts = pd.Timestamp(value).normalize()
    return ts.day == 1


def _is_month_end(value: pd.Timestamp) -> bool:
    ts = pd.Timestamp(value).normalize()
    return ts == (ts + pd.offsets.MonthEnd(0)).normalize()


def _floor_month_start(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize().replace(day=1)


def _ceil_to_next_month_start(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value).normalize()
    if _is_month_start(ts):
        return ts
    return (ts + pd.offsets.MonthBegin(1)).normalize()


def _floor_to_previous_month_end(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value).normalize()
    if _is_month_end(ts):
        return ts
    return (ts - pd.offsets.MonthEnd(1)).normalize()


def normalize_full_month_backtest_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert arbitrary user dates to a non-empty full-month backtest range."""

    user_start = pd.Timestamp(start).normalize()
    user_end = pd.Timestamp(end).normalize()
    effective_start = _ceil_to_next_month_start(user_start)
    effective_end = _floor_to_previous_month_end(user_end)
    if effective_start > effective_end:
        raise ValueError("当前输入区间无法形成至少一个完整月份，请调整回测日期")
    return effective_start, effective_end


def assert_full_month_range(start: str | pd.Timestamp, end: str | pd.Timestamp, *, context: str) -> None:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if not _is_month_start(start_ts) or not _is_month_end(end_ts):
        raise ValueError(
            f"{context} requires a full-month range: from must be month start and to must be month end. "
            f"Got from={start_ts.date()} to={end_ts.date()}."
        )


def bulk_month_labels(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[str]:
    assert_full_month_range(start, end, context="bulk_month_labels")
    return [pd.Timestamp(d).strftime("%Y-%m") for d in pd.date_range(_floor_month_start(pd.Timestamp(start)), pd.Timestamp(end), freq="MS")]


@contextmanager
def _cell5_stage(stage: str, heartbeat_sec: float = 20.0):
    start = time.perf_counter()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(float(heartbeat_sec)):
            elapsed = time.perf_counter() - start
            _cell5_log(f"still running: stage={stage} elapsed={elapsed:.1f}s")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    _cell5_log(f"{stage} start")
    try:
        yield
    finally:
        stop.set()
        elapsed = time.perf_counter() - start
        _cell5_log(f"{stage} done elapsed={elapsed:.1f}s")


def validate_equities_daily_params(params: dict[str, Any] | None) -> str:
    """Validate and classify J-Quants V2 equities daily bars parameters.

    Returns one of:
    - ``single_code_range``: ``code`` is present, optional ``from``/``to``.
    - ``market_single_day``: ``date`` is present and no ``code`` is present.
    - ``market_range``: no ``code``/``date`` but ``from`` or ``to`` is present;
      this must be routed through Bulk API.
    """

    params = dict(params or {})
    has_code = bool(params.get("code"))
    has_date = bool(params.get("date"))
    has_range = bool(params.get("from") or params.get("to"))
    if has_code:
        return "single_code_range"
    if has_date:
        return "market_single_day"
    if has_range:
        return "market_range"
    raise ValueError(
        "J-Quants /equities/bars/daily requires either code + optional from/to, "
        "date for a full-market single-day request, or from/to routed through Bulk API."
    )


def _is_equities_daily_bars_path(path: str) -> bool:
    return str(path).rstrip("/").endswith(EQUITIES_DAILY_BARS_ENDPOINT)


def _is_margin_path(path: str) -> bool:
    return str(path).rstrip("/") in MARGIN_ENDPOINT_PATHS


def validate_margin_params(endpoint_name: str, params: dict[str, Any] | None) -> str:
    """Validate and classify J-Quants V2 margin endpoint parameters.

    V2 ``/markets/margin-alert`` and ``/markets/margin-interest`` require
    either ``code`` or ``date`` for REST calls.  A market-wide range expressed
    as only ``from``/``to`` is not a legal direct REST request.
    """

    if endpoint_name not in MARGIN_DATASET_ENDPOINTS:
        raise ValueError(f"Unsupported margin endpoint_name={endpoint_name!r}")
    params = dict(params or {})
    has_code = bool(params.get("code"))
    has_date = bool(params.get("date"))
    has_range = bool(params.get("from") or params.get("to"))
    if has_code:
        return "single_code_range"
    if has_date:
        return "market_single_day"
    if has_range:
        return "market_range"
    raise ValueError(
        "J-Quants V2 margin endpoints require code + optional from/to, "
        "or date. Market-wide ranges must not be sent to direct REST. "
        f"endpoint_name={endpoint_name!r}, params={params!r}"
    )


def _request_mode(path: str, params: dict[str, Any] | None) -> str:
    if _is_equities_daily_bars_path(path):
        return validate_equities_daily_params(params)
    if _is_margin_path(path):
        # Path-only classification cannot know whether this is alert or weekly
        # balance data, but the parameter legality is identical for both.
        return validate_margin_params("weekly_margin_interest", params)
    return "generic"


@dataclass
class JQuantsApiConfig:
    """Minimal J-Quants API/cache configuration for notebook orchestration.

    Defaults target J-Quants API v2.  Endpoint overrides are available because
    the public legacy docs still show v1 paths in many places; if an endpoint
    name differs in your v2 contract, override only that logical name.
    """

    api_key: str | None = None
    refresh_token: str | None = None
    id_token: str | None = None
    api_version: str = "v2"
    base_url: str = "https://api.jquants.com/v2"
    # Standard J-Quants public docs still use v1 for token refresh.  Keep data
    # endpoints on v2, but authenticate through the documented Standard URL.
    auth_url: str = "https://api.jquants.com/v1/token/auth_refresh"
    endpoint_overrides: dict[str, str] = field(default_factory=dict)
    auth_header_name: str = "x-api-key"
    use_bulk_download: bool = True
    bulk_list_path: str = "/bulk/list"
    bulk_download_path: str = "/bulk/get"
    request_sleep_sec: float = 1.0
    timeout_sec: int = 60
    max_retries: int = 3
    rate_limit_sleep_sec: float = 300.0
    rate_limit_max_cooldowns: int = 3
    save_format: str = "parquet"  # parquet preferred, csv fallback if parquet engine is missing
    include_margin: bool = True
    # Behaviour/crowding factor expects credit balance history, so the default
    # margin cache uses weekly margin interest (/markets/margin-interest), not
    # daily margin-alert issue flags.
    margin_dataset: str = "weekly_margin_interest"
    margin_optional: bool = True
    include_financials: bool = True
    include_topix_index: bool = True
    include_nikkei_index: bool = True
    additional_index_codes: tuple[str, ...] = ()
    universe_from_listed_info: bool = True
    market_cap_source: str = "listed_shares_if_available"
    allow_missing_market_cap: bool = False
    market_cap_fallback: str = "liquidity_proxy"  # debug-only fallback when J-Quants master lacks shares/mcap
    force_rebuild_cache: bool = False
    extra_history_days: int | None = None
    listed_info_frequency: str = "ME"  # month-end snapshots for PIT universe/sector
    bulk_only_datasets: tuple[str, ...] = ("daily_quotes",)
    fins_summary_bulk_workers: int = 4

    def effective_history_days(self) -> int:
        if self.extra_history_days is not None:
            return int(self.extra_history_days)
        return infer_required_history_days()

    def endpoint(self, name: str) -> str:
        defaults = {
            "listed_info": "/equities/master",
            "daily_quotes": "/equities/bars/daily",
            "statements": "/fins/summary",
            "topix": "/indices/bars/daily/topix",
            "indices": "/indices/bars/daily",
            "nikkei225": "/indices/nikkei225",
            "daily_margin_interest": "/markets/margin-alert",
            "weekly_margin_interest": "/markets/margin-interest",
        }
        return self.endpoint_overrides.get(name, defaults[name])

    def validate_endpoint_mapping(self) -> None:
        validate_endpoint_mapping(self)


def validate_endpoint_mapping(config: JQuantsApiConfig | dict[str, Any] | None = None) -> None:
    """Fail fast on endpoint mappings that are known to be invalid."""

    if config is None:
        config = JQuantsApiConfig()
    if isinstance(config, dict):
        config = JQuantsApiConfig(**config)
    expected = {
        "daily_quotes": "/equities/bars/daily",
        "topix": "/indices/bars/daily/topix",
        "indices": "/indices/bars/daily",
        "daily_margin_interest": "/markets/margin-alert",
        "weekly_margin_interest": "/markets/margin-interest",
    }
    for name, path in expected.items():
        actual = config.endpoint(name)
        if not str(actual).startswith(path):
            raise ValueError(f"Invalid J-Quants endpoint mapping: endpoint_name={name!r}, expected_prefix={path!r}, actual={actual!r}")
    if config.endpoint("topix").rstrip("/") == "/indices/topix":
        raise ValueError("Invalid J-Quants endpoint mapping: topix must not resolve to '/indices/topix'.")
    for name in MARGIN_DATASET_ENDPOINTS:
        old_path = f"/markets/{name}"
        if config.endpoint(name).rstrip("/") == old_path:
            raise ValueError(f"Invalid J-Quants endpoint mapping: {name} must not resolve to V1 path {old_path!r}.")


class JQuantsApiClient:
    """Small REST client covering the Standard-plan datasets used by backtests."""

    def __init__(self, config: JQuantsApiConfig | dict[str, Any] | None = None) -> None:
        if config is None:
            config = JQuantsApiConfig()
        if isinstance(config, dict):
            config = JQuantsApiConfig(**config)
        self.config = config
        self.config.validate_endpoint_mapping()
        self.session = requests.Session()
        self._id_token = config.id_token

    def _get_with_rate_limit(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        endpoint: str,
        endpoint_name: str | None = None,
        mode: str | None = None,
        signed_url: bool = False,
    ) -> requests.Response:
        """GET with J-Quants 429 cooldown recovery.

        J-Quants Standard can return 429 when the rolling request budget is
        exhausted.  Treat that as a temporary cooldown event and resume the
        exact same request after the configured sleep, instead of letting
        callers write empty shard caches.
        """

        cooldowns = 0
        clean_params = {k: v for k, v in dict(params or {}).items() if v is not None}
        log_endpoint = "<signed-bulk-url>" if signed_url else endpoint
        while True:
            if signed_url:
                resp = requests.get(url, timeout=self.config.timeout_sec, allow_redirects=True)
            else:
                resp = self.session.get(url, params=clean_params, headers=headers, timeout=self.config.timeout_sec)
            if resp.status_code != 429:
                time.sleep(float(self.config.request_sleep_sec))
                if cooldowns:
                    _cell5_log(
                        "[rate-limit] success after cooldown "
                        f"endpoint={log_endpoint} endpoint_name={endpoint_name} mode={mode} params={clean_params}"
                    )
                return resp
            if cooldowns >= int(self.config.rate_limit_max_cooldowns):
                return resp
            cooldowns += 1
            sleep_sec = float(self.config.rate_limit_sleep_sec)
            _cell5_log(
                "[rate-limit] 429 hit "
                f"endpoint={log_endpoint} endpoint_name={endpoint_name} mode={mode} "
                f"params={clean_params} sleep={sleep_sec:g}s retry={cooldowns}/{int(self.config.rate_limit_max_cooldowns)}"
            )
            time.sleep(sleep_sec)
            _cell5_log(
                "[rate-limit] resumed "
                f"endpoint={log_endpoint} endpoint_name={endpoint_name} mode={mode} params={clean_params}"
            )

    def id_token(self) -> str:
        direct_token = self._id_token or self.config.api_key or os.getenv("JQUANTS_API_KEY")
        if direct_token:
            self._id_token = str(direct_token)
            return self._id_token
        refresh_token = self.config.refresh_token or os.getenv("JQUANTS_REFRESH_TOKEN")
        if not refresh_token:
            raise ValueError(
                "Set JQUANTS_API_KEY for direct Bearer-token use, or pass "
                "JQuantsApiConfig(api_key=...). Refresh-token auth is optional."
            )
        resp = self._post_auth_refresh(self.config.auth_url, refresh_token)
        if resp.status_code == 403 and "/v2/" in self.config.auth_url and "api.jquants.com" in self.config.auth_url:
            # Standard API users may request v2 data while still using the v1
            # token-refresh endpoint.  Retry once with the documented Standard
            # auth URL, without exposing the refresh token in an HTTPError URL.
            resp = self._post_auth_refresh("https://api.jquants.com/v1/token/auth_refresh", refresh_token)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"J-Quants auth_refresh failed with HTTP {resp.status_code}. "
                "Check that the refresh token is current, the subscription is active, "
                "and the auth_url matches your Standard/Pro contract."
            )
        payload = resp.json()
        token = payload.get("idToken")
        if not token:
            raise RuntimeError(f"J-Quants auth response did not include idToken: {payload}")
        self._id_token = str(token)
        return self._id_token

    def auth_headers(self) -> dict[str, str]:
        key = self.id_token()
        if self.config.auth_header_name.lower() == "authorization":
            return {"Authorization": f"Bearer {key}"}
        return {self.config.auth_header_name: key}

    def _post_auth_refresh(self, auth_url: str, refresh_token: str) -> requests.Response:
        return self.session.post(
            auth_url,
            params={"refreshtoken": refresh_token},
            timeout=self.config.timeout_sec,
        )

    def get(self, path: str, params: dict[str, Any] | None = None, result_key: str | None = None, endpoint_name: str | None = None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        params = {k: v for k, v in dict(params or {}).items() if v is not None}
        pagination_key: str | None = None
        while True:
            request_params = dict(params)
            if pagination_key:
                request_params["pagination_key"] = pagination_key
            mode = _request_mode(path, request_params)
            if _is_equities_daily_bars_path(path) and mode == "market_range":
                raise ValueError(
                    "Missing code/date for /equities/bars/daily. A full-market range "
                    "request must use Bulk API, not direct REST. "
                    f"endpoint={path!r}, mode={mode}, params={request_params!r}"
                )
            if _is_margin_path(path) and mode == "market_range":
                raise ValueError(
                    "Missing code/date for J-Quants V2 margin endpoint. "
                    "Direct REST margin requests must use code + optional from/to, "
                    "or date. Market-wide ranges require a bulk/date fallback or "
                    "must be skipped as optional cache data. "
                    f"endpoint_name={endpoint_name!r}, endpoint={path!r}, mode={mode}, params={request_params!r}"
                )
            payload = self._get_json(path, request_params, mode=mode, endpoint_name=endpoint_name)
            key = result_key if result_key in payload else self._infer_result_key(payload)
            rows.extend(payload.get(key, []) or [])
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                break
        return pd.DataFrame(rows)

    def _get_json(self, path: str, params: dict[str, Any], mode: str | None = None, endpoint_name: str | None = None) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = self.auth_headers()
        last_error: Exception | None = None
        last_status: int | None = None
        last_body: str | None = None
        mode = mode or _request_mode(path, params)
        for attempt in range(int(self.config.max_retries)):
            try:
                resp = self._get_with_rate_limit(
                    url,
                    params=params,
                    headers=headers,
                    endpoint=path,
                    endpoint_name=endpoint_name,
                    mode=mode,
                )
                if resp.status_code == 401 and attempt == 0:
                    self._id_token = None
                    headers = self.auth_headers()
                    resp = self._get_with_rate_limit(
                        url,
                        params=params,
                        headers=headers,
                        endpoint=path,
                        endpoint_name=endpoint_name,
                        mode=mode,
                    )
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:  # pragma: no cover - network dependent
                last_error = exc
                response = exc.response
                last_status = response.status_code if response is not None else None
                last_body = (response.text if response is not None else "")[:1000]
                time.sleep(float(self.config.request_sleep_sec) * (attempt + 1))
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                time.sleep(float(self.config.request_sleep_sec) * (attempt + 1))
        raise RuntimeError(
            "J-Quants request failed: "
            f"status={last_status}, endpoint_name={endpoint_name}, path={path}, "
            f"mode={mode}, params={params}, body={last_body}"
        ) from last_error

    def get_dataset(self, endpoint_name: str, params: dict[str, Any] | None = None, result_key: str | None = None) -> pd.DataFrame:
        endpoint = self.config.endpoint(endpoint_name)
        if endpoint_name == "daily_quotes":
            return self.get_equities_daily_bars(params or {}, result_key=result_key)
        if endpoint_name == "statements":
            return self.get_financial_summary_dataset(params or {}, result_key=result_key)
        if endpoint_name in MARGIN_DATASET_ENDPOINTS:
            return self.get_margin_dataset(endpoint_name, params or {}, result_key=result_key)
        bulk_error: Exception | None = None
        if self.config.use_bulk_download:
            try:
                bulk = self.get_bulk(endpoint, params=params)
                if not bulk.empty:
                    return bulk
            except Exception as exc:
                # Bulk availability and response envelopes differ by contract.
                # Fall back to the regular JSON endpoint instead of treating a
                # JSON error body as a gzip CSV.
                bulk_error = exc
        if endpoint_name in set(self.config.bulk_only_datasets):
            raise RuntimeError(
                f"J-Quants bulk download failed for {endpoint_name!r}; not falling back "
                f"to REST because this V2 dataset is configured as bulk-only. "
                f"endpoint={endpoint!r} params={params!r}"
            ) from bulk_error
        return self.get(endpoint, params=params, result_key=result_key, endpoint_name=endpoint_name)

    def get_financial_summary_dataset(self, params: dict[str, Any], result_key: str | None = None) -> pd.DataFrame:
        """Route /fins/summary according to V2 rules.

        V2 financial summary requires either ``code`` or ``date``.  It does not
        support a direct full-market ``from``/``to`` REST request.
        """

        clean = {k: v for k, v in dict(params or {}).items() if v is not None}
        has_code = bool(clean.get("code"))
        has_date = bool(clean.get("date"))
        if not has_code and not has_date:
            raise ValueError(
                "Missing code/date for J-Quants V2 /fins/summary. "
                "Use code for one-name history or date for all disclosures on one day; "
                f"params={clean!r}"
            )
        if not has_code and ("from" in clean or "to" in clean):
            raise ValueError(
                "Do not call /fins/summary with only from/to. "
                "Build a full-market range by requesting date=YYYYMMDD shards. "
                f"params={clean!r}"
            )
        return self.get(self.config.endpoint("statements"), params=clean, result_key=result_key or "data", endpoint_name="statements")

    def get_margin_dataset(self, endpoint_name: str, params: dict[str, Any], result_key: str | None = None) -> pd.DataFrame:
        """Route V2 margin endpoints without allowing illegal range-only REST calls."""

        mode = validate_margin_params(endpoint_name, params)
        endpoint = self.config.endpoint(endpoint_name)
        if mode == "market_range":
            raise ValueError(
                "Missing code/date for J-Quants V2 margin endpoint. "
                "Do not call REST with only from/to. "
                f"endpoint_name={endpoint_name!r}, path={endpoint!r}, mode={mode}, params={params!r}"
            )
        return self.get(endpoint, params=params, result_key=result_key or endpoint_name, endpoint_name=endpoint_name)

    def get_equities_daily_bars(self, params: dict[str, Any], result_key: str | None = None) -> pd.DataFrame:
        """Route /equities/bars/daily according to J-Quants V2 parameter rules."""

        mode = validate_equities_daily_params(params)
        endpoint = self.config.endpoint("daily_quotes")
        if mode == "market_range":
            return self.build_market_daily_bars_cache_via_bulk(params)
        return self.get(endpoint, params=params, result_key=result_key or "daily_quotes", endpoint_name="daily_quotes")

    def build_market_daily_bars_cache_via_bulk(self, params: dict[str, Any]) -> pd.DataFrame:
        """Fetch full-market daily bars for a date range through Bulk API."""

        mode = validate_equities_daily_params(params)
        if mode != "market_range":
            raise ValueError(f"Bulk market-range loader received non-market-range params: mode={mode}, params={params!r}")
        assert_full_month_range(params.get("from"), params.get("to"), context="/equities/bars/daily full-market Bulk API")
        _cell5_log(f"daily_quotes bulk full months={bulk_month_labels(params.get('from'), params.get('to'))}")
        endpoint = self.config.endpoint("daily_quotes")
        if not self.config.use_bulk_download:
            raise ValueError(
                "Full-market daily bars range requires Bulk API. "
                f"Set use_bulk_download=True or pass code/date. params={params!r}"
            )
        frame = self.get_bulk(endpoint, params=params)
        if frame.empty:
            raise RuntimeError(
                "Bulk API returned no daily bar rows for full-market range. "
                f"endpoint={endpoint!r}, mode={mode}, params={params!r}"
            )
        return frame

    def build_fins_summary_cache_via_bulk(
        self,
        params: dict[str, Any],
        data_dir: str | Path,
        *,
        save_format: str,
        codes: Iterable[str] | None = None,
        existing_main: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fetch /fins/summary monthly bulk files once, with per-month cache."""

        data_path = Path(data_dir)
        endpoint = self.config.endpoint("statements")
        month_keys = _months_for_params(params)
        month_set = set(month_keys)
        _cell5_log(
            "fins_summary bulk plan "
            f"months={month_keys}, month_count={len(month_keys)}, "
            f"workers={int(self.config.fins_summary_bulk_workers)}"
        )
        totals = {
            "bulk_get": 0.0,
            "download": 0.0,
            "decompress": 0.0,
            "read_csv": 0.0,
            "cache_write": 0.0,
        }
        overall_start = time.perf_counter()
        month_frames: dict[str, pd.DataFrame] = {}
        pending: list[tuple[str, str, str]] = []
        existing = existing_main.copy() if existing_main is not None else pd.DataFrame()
        if not existing.empty and "disclosed_date" in existing.columns:
            existing["disclosed_date"] = pd.to_datetime(existing["disclosed_date"], errors="coerce").dt.tz_localize(None)
            existing = existing.dropna(subset=["disclosed_date"]).reset_index(drop=True)
            if "code" in existing.columns:
                existing["code"] = existing["code"].map(normalize_code).astype("string")

        list_params = {"endpoint": endpoint}
        for key in ["from", "to", "date"]:
            if key in params and params[key] is not None:
                list_params[key] = params[key]
        with _cell5_stage("fins_summary bulk_list"):
            list_payload = self._get_json(self.config.bulk_list_path, list_params, mode=f"bulk_list:{endpoint}", endpoint_name="bulk/list")
        bulk_items = _extract_bulk_items(list_payload)
        if not bulk_items:
            return pd.DataFrame()

        by_month: dict[str, dict[str, Any]] = {}
        for item in bulk_items:
            key = str(item.get("Key") or item.get("key") or "")
            month = _bulk_key_month(key)
            if month and month in month_set:
                by_month.setdefault(month, item)
        if not by_month:
            _cell5_log("fins_summary bulk plan found no exact monthly keys; falling back to generic bulk reader")
            return self.get_bulk(endpoint, params=params)

        for idx, month in enumerate(month_keys, start=1):
            cached = _read_fins_summary_month_cache(data_path, month)
            if cached is not None:
                _cell5_log(
                    f"fins_summary month={_month_label(month)} cache hit "
                    f"rows={len(cached)} cols={len(cached.columns)}"
                )
                month_frames[month] = cached
                _cell5_log(
                    f"fins_summary progress {idx}/{len(month_keys)} cache_hit "
                    f"total_elapsed={time.perf_counter() - overall_start:.1f}s"
                )
                continue
            if not existing.empty:
                month_from_main = _slice_financial_summary_month_from_main(existing, month)
                if not month_from_main.empty:
                    _write_fins_summary_month_cache(month_from_main, data_path, month, save_format)
                    _cell5_log(
                        f"fins_summary month={_month_label(month)} main-table reuse "
                        f"rows={len(month_from_main)}"
                    )
                    month_frames[month] = month_from_main
                    _cell5_log(
                        f"fins_summary progress {idx}/{len(month_keys)} main_table_reuse "
                        f"total_elapsed={time.perf_counter() - overall_start:.1f}s"
                    )
                    continue
            item = by_month.get(month)
            if item is None:
                _cell5_log(f"fins_summary month={_month_label(month)} no bulk item found; skipping")
                continue
            key = str(item.get("Key") or item.get("key") or "")
            urls = _extract_urls(item)
            signed_url = urls[0] if urls else self._bulk_get_signed_url(key, month=month, totals=totals)
            pending.append((month, key, signed_url))

        completed = 0
        if pending:
            workers = max(1, min(int(self.config.fins_summary_bulk_workers), len(pending)))
            _cell5_log(f"fins_summary concurrent processing start pending={len(pending)} workers={workers}")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _download_parse_write_fins_summary_month,
                        month,
                        key,
                        signed_url,
                        data_path,
                        save_format,
                        self.config.timeout_sec,
                        self.config.request_sleep_sec,
                        self.config.rate_limit_sleep_sec,
                        self.config.rate_limit_max_cooldowns,
                    ): month
                    for month, key, signed_url in pending
                }
                for future in as_completed(futures):
                    month = futures[future]
                    result = future.result()
                    month_frames[month] = result["frame"]
                    for total_key, value in result["timings"].items():
                        totals[total_key] += float(value)
                    completed += 1
                    done_count = len(month_frames)
                    _cell5_log(
                        f"fins_summary progress {done_count}/{len(month_keys)} done "
                        f"month={_month_label(month)} total_elapsed={time.perf_counter() - overall_start:.1f}s"
                    )

        frames = [month_frames[m] for m in month_keys if m in month_frames and not month_frames[m].empty]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["disclosed_date", "code"])
        if codes and not out.empty and "code" in out.columns:
            code_set = {normalize_code(c) for c in codes if normalize_code(c) is not None}
            out = out.loc[out["code"].isin(code_set)].reset_index(drop=True)
        out = _filter_financial_summary_frame(out, params)
        _cell5_log(
            "fins_summary summary "
            f"months_requested={len(month_keys)} months_processed={len(month_frames)} "
            f"bulk_get_total_sec={totals['bulk_get']:.1f} "
            f"download_total_sec={totals['download']:.1f} "
            f"decompress_total_sec={totals['decompress']:.1f} "
            f"read_csv_total_sec={totals['read_csv']:.1f} "
            f"cache_write_total_sec={totals['cache_write']:.1f} "
            f"total_elapsed={time.perf_counter() - overall_start:.1f}s"
        )
        return out

    def _bulk_get_signed_url(self, key: str, *, month: str, totals: dict[str, float]) -> str:
        url = f"{self.config.base_url.rstrip('/')}/{self.config.bulk_download_path.lstrip('/')}"
        request_params = {"key": key}
        start = time.perf_counter()
        _cell5_log(f"fins_summary month={_month_label(month)} bulk_get start key={key} params={request_params}")
        resp = self._get_with_rate_limit(
            url,
            params=request_params,
            headers=self.auth_headers(),
            endpoint=self.config.bulk_download_path,
            endpoint_name="bulk/get",
            mode="bulk_get:fins_summary",
        )
        body_preview = (resp.text if hasattr(resp, "text") else "")[:1000]
        elapsed = time.perf_counter() - start
        totals["bulk_get"] += elapsed
        _cell5_log(
            f"fins_summary month={_month_label(month)} bulk_get done "
            f"status={resp.status_code} elapsed={elapsed:.1f}s body_preview={body_preview!r}"
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"bulk/get failed: params={request_params}, status={resp.status_code}, body={body_preview}"
            )
        payload = _json_payload(resp.content, resp.headers.get("Content-Type", ""))
        urls = _extract_urls(payload)
        _cell5_log(f"fins_summary month={_month_label(month)} bulk_get payload parsed signed_url_found={bool(urls)}")
        if not urls:
            raise RuntimeError(f"Bulk file URL was not found for key={key!r}. Response={payload!r}")
        return urls[0]

    def get_bulk(self, endpoint: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Load a V2 bulk dataset listed by /bulk/list.

        The list response may vary by contract.  This method extracts URL-like
        fields recursively and reads CSV/ZIP payloads into one DataFrame.
        """

        params = params or {}
        list_params = {"endpoint": endpoint}
        for key in ["from", "to", "date"]:
            if key in params and params[key] is not None:
                list_params[key] = params[key]
        list_payload = self._get_json(self.config.bulk_list_path, list_params, mode=f"bulk_list:{endpoint}", endpoint_name="bulk/list")
        bulk_items = _extract_bulk_items(list_payload)
        if not bulk_items:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for item in bulk_items:
            key = item.get("Key") or item.get("key")
            if key and not _bulk_key_overlaps_params(str(key), params):
                continue
            frame = self._read_bulk_item(item)
            if frame.empty:
                continue
            frames.append(frame)
            time.sleep(float(self.config.request_sleep_sec))
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        return _filter_bulk_frame(data, params)

    def _read_bulk_item(self, item: dict[str, Any]) -> pd.DataFrame:
        for url in _extract_urls(item):
            return self._read_bulk_url(url)
        key = item.get("Key") or item.get("key")
        if not key:
            return pd.DataFrame()
        return self._read_bulk_key(str(key))

    def _read_bulk_key(self, key: str) -> pd.DataFrame:
        url = f"{self.config.base_url.rstrip('/')}/{self.config.bulk_download_path.lstrip('/')}"
        request_params = {"key": key}
        _cell5_log(f"bulk get start key={key} params={request_params}")
        resp = self._get_with_rate_limit(
            url,
            params=request_params,
            headers=self.auth_headers(),
            endpoint=self.config.bulk_download_path,
            endpoint_name="bulk/get",
            mode="bulk_get",
        )
        body_preview = (resp.text if hasattr(resp, "text") else "")[:1000]
        _cell5_log(f"bulk get response status={resp.status_code} key={key} body_preview={body_preview!r}")
        if resp.status_code >= 400:
            raise RuntimeError(
                f"bulk/get failed: params={request_params}, status={resp.status_code}, body={body_preview}"
            )
        payload = _json_payload(resp.content, resp.headers.get("Content-Type", ""))
        if payload is not None:
            urls = _extract_urls(payload)
            _cell5_log(f"bulk get payload parsed key={key} signed_url_found={bool(urls)}")
            if urls:
                return self._read_bulk_url(urls[0], depth=1)
            embedded = _extract_tabular_records(payload)
            if embedded is not None:
                return embedded
            raise RuntimeError(f"Bulk file URL was not found for key={key!r}. Response={payload!r}")
        _cell5_log(f"bulk get returned tabular payload directly key={key}")
        return _read_tabular_content(
            resp.content,
            resp.headers.get("Content-Type", ""),
            key,
            url_fetcher=self._read_bulk_url,
            depth=0,
        )

    def _read_bulk_url(self, url: str, depth: int = 0) -> pd.DataFrame:
        if depth > 3:
            raise RuntimeError(f"Bulk payload URL recursion exceeded limit: url={url!r}")
        # Signed URLs should be downloaded without API auth/cookies.  Keeping
        # this request clean avoids accidentally receiving an auth error page.
        resp = self._get_with_rate_limit(
            url,
            endpoint="<signed-bulk-url>",
            endpoint_name="bulk/signed_url",
            mode="bulk_signed_url",
            signed_url=True,
        )
        content = resp.content or b""
        diagnostics = _payload_diagnostics(resp, content, url)
        try:
            resp.raise_for_status()
            return _read_tabular_content(
                content,
                resp.headers.get("Content-Type", ""),
                resp.url or url,
                url_fetcher=self._read_bulk_url,
                depth=depth,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            fmt = detect_payload_format(content, resp.headers.get("Content-Type", ""), resp.url or url)
            raise RuntimeError(
                "Bulk payload parse failed: "
                f"status={diagnostics['status']}, url={diagnostics['final_url']}, "
                f"content_type={diagnostics['content_type']}, "
                f"content_encoding={diagnostics['content_encoding']}, "
                f"content_length={diagnostics['content_length']}, "
                f"detected_format={fmt}, redirected={diagnostics['redirected']}, "
                f"first64_hex={diagnostics['first64_hex']}, "
                f"body_preview={diagnostics['body_preview']!r}"
            ) from exc

    @staticmethod
    def _infer_result_key(payload: dict[str, Any]) -> str:
        for key, value in payload.items():
            if isinstance(value, list):
                return key
        raise RuntimeError(f"Could not infer result list key from response keys: {list(payload)}")


def _extract_urls(obj: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(obj, str):
        if obj.startswith("http://") or obj.startswith("https://"):
            urls.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            urls.extend(_extract_urls(value))
    elif isinstance(obj, list):
        for item in obj:
            urls.extend(_extract_urls(item))
    return list(dict.fromkeys(urls))


def _json_payload(content: bytes, content_type: str = "") -> Any | None:
    text = content[:64].lstrip()
    if "json" not in str(content_type).lower() and not (text.startswith(b"{") or text.startswith(b"[")):
        return None
    try:
        import json

        return json.loads(content.decode("utf-8"))
    except Exception:
        return None


def detect_payload_format(content: bytes, content_type: str = "", name: str = "") -> str:
    """Detect payload format from bytes first, never from filename alone."""

    head = bytes(content[:8])
    stripped = bytes(content).lstrip()
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "json"
    if stripped.startswith(b"<"):
        return "xml_or_html"
    lower_type = str(content_type).lower()
    if "csv" in lower_type or "text/plain" in lower_type:
        return "csv_or_text"
    if _looks_like_csv_text(stripped[:4096]):
        return "csv_or_text"
    return "binary_unknown"


def _looks_like_csv_text(sample: bytes) -> bool:
    if not sample:
        return False
    text = ""
    for encoding in ("utf-8-sig", "cp932"):
        try:
            text = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        # A fixed-size sample may cut a multibyte character at its boundary,
        # which must not be mistaken for binary content; only the first line
        # is inspected, so lenient decoding is safe here.
        text = sample.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text.splitlines() else ""
    return "," in first or "\t" in first


def _text_preview(content: bytes, limit: int = 200) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return content[:limit].decode(encoding, errors="replace")
        except Exception:
            continue
    return ""


def _payload_diagnostics(resp: requests.Response, content: bytes, requested_url: str) -> dict[str, Any]:
    return {
        "requested_url": requested_url,
        "final_url": getattr(resp, "url", requested_url),
        "status": getattr(resp, "status_code", None),
        "content_type": resp.headers.get("Content-Type", ""),
        "content_encoding": resp.headers.get("Content-Encoding", ""),
        "content_length": len(content),
        "first64_hex": content[:64].hex(),
        "body_preview": _text_preview(content, 200),
        "redirected": bool(getattr(resp, "history", None)),
    }


def _extract_tabular_records(obj: Any) -> pd.DataFrame | None:
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in obj):
            return pd.DataFrame(obj)
    if isinstance(obj, dict):
        for value in obj.values():
            frame = _extract_tabular_records(value)
            if frame is not None:
                return frame
    return None


def _extract_bulk_items(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        if "Key" in obj or "key" in obj or _extract_urls(obj):
            return [obj]
        rows: list[dict[str, Any]] = []
        for value in obj.values():
            rows.extend(_extract_bulk_items(value))
        return rows
    if isinstance(obj, list):
        rows: list[dict[str, Any]] = []
        for item in obj:
            rows.extend(_extract_bulk_items(item))
        return rows
    return []


def _bulk_key_overlaps_params(key: str, params: dict[str, Any]) -> bool:
    start = params.get("from") or params.get("start") or params.get("start_date")
    end = params.get("to") or params.get("end") or params.get("end_date")
    if start is None and end is None:
        return True
    months = set()
    start_ts = pd.Timestamp(str(start)).normalize() if start is not None else pd.Timestamp("1900-01-01")
    end_ts = pd.Timestamp(str(end)).normalize() if end is not None else pd.Timestamp("2100-01-01")
    for month in pd.period_range(start_ts, end_ts, freq="M"):
        months.add(month.strftime("%Y%m"))
    years = {m[:4] for m in months}
    digits = "".join(ch if ch.isdigit() else " " for ch in key).split()
    tokens = set(digits)
    return bool(tokens & months) or bool(tokens & years)


FINS_SUMMARY_USECOL_CANDIDATES: tuple[str, ...] = (
    "DisclosedDate",
    "DisclosureDate",
    "DiscDate",
    "disclosed_date",
    "disclosure_date",
    "disc_date",
    "LocalCode",
    "local_code",
    "Code",
    "code",
    "IssueCode",
    "issue_code",
    "NetSales",
    "net_sales",
    "Revenue",
    "revenue",
    "Sales",
    "sales",
    "OperatingRevenue",
    "operating_revenue",
    "OperatingProfit",
    "operating_profit",
    "OperatingIncome",
    "operating_income",
    "OP",
    "op",
    "BusinessProfit",
    "business_profit",
    "OrdinaryProfit",
    "ordinary_profit",
    "OdP",
    "odp",
    "Profit",
    "profit",
    "NetIncome",
    "net_income",
    "NP",
    "np",
    "ProfitAttributableToOwnersOfParent",
    "profit_attributable_to_owners_of_parent",
    "TotalAssets",
    "total_assets",
    "TA",
    "ta",
    "Equity",
    "equity",
    "Eq",
    "eq",
    "NetAssets",
    "net_assets",
    "CashAndEquivalents",
    "cash_and_equivalents",
    "CashEq",
    "casheq",
    "CashFlowsFromOperatingActivities",
    "cash_flows_from_operating_activities",
    "CFO",
    "cfo",
    "CFI",
    "cfi",
    "FreeCashFlow",
    "free_cash_flow",
    "ShOutFY",
    "shoutfy",
    "AvgSh",
    "avgsh",
)


def _month_label(month: str) -> str:
    text = str(month)
    return f"{text[:4]}-{text[4:6]}" if len(text) == 6 else text


def _months_for_params(params: dict[str, Any]) -> list[str]:
    start = params.get("from") or params.get("start") or params.get("start_date")
    end = params.get("to") or params.get("end") or params.get("end_date")
    start_ts = pd.Timestamp(str(start)).normalize() if start is not None else pd.Timestamp("1900-01-01")
    end_ts = pd.Timestamp(str(end)).normalize() if end is not None else pd.Timestamp("2100-01-01")
    return [period.strftime("%Y%m") for period in pd.period_range(start_ts, end_ts, freq="M")]


def _bulk_key_month(key: str) -> str | None:
    matches = re.findall(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)", str(key))
    if not matches:
        return None
    year, month = matches[-1]
    return f"{year}{month}"


def _fins_summary_month_cache_path(data_dir: Path, month: str, save_format: str = "parquet") -> Path:
    suffix = ".parquet" if save_format == "parquet" else ".csv"
    return data_dir / f"financial_summary_{month}{suffix}"


def _read_fins_summary_month_cache(data_dir: Path, month: str) -> pd.DataFrame | None:
    for suffix in (".parquet", ".csv"):
        path = data_dir / f"financial_summary_{month}{suffix}"
        if not path.exists():
            continue
        frame = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path, dtype=str, low_memory=False)
        if frame.empty:
            _cell5_log(f"fins_summary month={_month_label(month)} stale empty cache ignored path={path}")
            try:
                path.unlink()
                _cell5_log(f"fins_summary month={_month_label(month)} stale empty cache deleted path={path}")
            except OSError as exc:
                _cell5_log(f"fins_summary month={_month_label(month)} stale empty cache delete failed path={path} error={exc!r}")
            return None
        return frame
    return None


def _slice_financial_summary_month_from_main(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    if frame.empty or "disclosed_date" not in frame.columns:
        return pd.DataFrame(columns=["disclosed_date", "code"])
    work = frame.copy()
    dates = pd.to_datetime(work["disclosed_date"], errors="coerce").dt.tz_localize(None)
    start = pd.Timestamp(f"{month[:4]}-{month[4:6]}-01")
    end = (start + pd.offsets.MonthEnd(1)).normalize()
    out = work.loc[dates.between(start, end, inclusive="both")].copy()
    if out.empty:
        return pd.DataFrame(columns=work.columns)
    out["disclosed_date"] = pd.to_datetime(out["disclosed_date"], errors="coerce").dt.tz_localize(None)
    return out.drop_duplicates(["disclosed_date", "code"], keep="last").reset_index(drop=True)


def _write_fins_summary_month_cache(frame: pd.DataFrame, data_dir: Path, month: str, save_format: str) -> Path:
    path = _fins_summary_month_cache_path(data_dir, month, save_format)
    if save_format == "parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _fins_summary_usecol(name: object) -> bool:
    wanted = {_canonical_col_name(col) for col in FINS_SUMMARY_USECOL_CANDIDATES}
    return _canonical_col_name(name) in wanted


def _has_canonical_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> bool:
    existing = {_canonical_col_name(col) for col in frame.columns}
    wanted = {_canonical_col_name(col) for col in candidates}
    return bool(existing & wanted)


def _download_parse_write_fins_summary_month(
    month: str,
    key: str,
    signed_url: str,
    data_dir: Path,
    save_format: str,
    timeout_sec: int,
    request_sleep_sec: float,
    rate_limit_sleep_sec: float,
    rate_limit_max_cooldowns: int,
) -> dict[str, Any]:
    timings = {
        "download": 0.0,
        "decompress": 0.0,
        "read_csv": 0.0,
        "cache_write": 0.0,
    }
    label = _month_label(month)

    start = time.perf_counter()
    _cell5_log(f"fins_summary month={label} download start key={key}")
    resp = _signed_get_with_rate_limit(
        signed_url,
        timeout_sec=timeout_sec,
        request_sleep_sec=request_sleep_sec,
        rate_limit_sleep_sec=rate_limit_sleep_sec,
        rate_limit_max_cooldowns=rate_limit_max_cooldowns,
        endpoint_name="bulk/signed_url:fins_summary",
        mode=f"fins_summary_month:{label}",
    )
    content = resp.content or b""
    resp.raise_for_status()
    timings["download"] = time.perf_counter() - start
    size_mb = len(content) / (1024 * 1024)
    _cell5_log(f"fins_summary month={label} download done size_mb={size_mb:.2f} elapsed={timings['download']:.1f}s")

    start = time.perf_counter()
    _cell5_log(f"fins_summary month={label} gzip decompress start")
    csv_bytes = gzip.decompress(content) if content[:2] == b"\x1f\x8b" else content
    timings["decompress"] = time.perf_counter() - start
    _cell5_log(
        f"fins_summary month={label} gzip decompress done "
        f"size_mb={len(csv_bytes) / (1024 * 1024):.2f} elapsed={timings['decompress']:.1f}s"
    )

    start = time.perf_counter()
    _cell5_log(f"fins_summary month={label} read_csv start")
    last_error: Exception | None = None
    raw: pd.DataFrame | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            raw = pd.read_csv(
                BytesIO(csv_bytes),
                encoding=encoding,
                dtype=str,
                low_memory=False,
                usecols=_fins_summary_usecol,
            )
            break
        except Exception as exc:
            last_error = exc
    if raw is None:
        raise RuntimeError(f"Could not parse fins_summary CSV for month={label}, key={key}") from last_error
    if not _has_canonical_columns(raw, ("DisclosedDate", "DisclosureDate", "disclosed_date", "disclosure_date")) or not _has_canonical_columns(raw, ("LocalCode", "Code", "IssueCode", "local_code", "code", "issue_code")):
        _cell5_log(
            f"fins_summary month={label} selected columns missing date/code; "
            f"selected_columns={list(raw.columns)} retrying full header parse"
        )
        raw = None
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp932"):
            try:
                raw = pd.read_csv(
                    BytesIO(csv_bytes),
                    encoding=encoding,
                    dtype=str,
                    low_memory=False,
                )
                break
            except Exception as exc:
                last_error = exc
        if raw is None:
            raise RuntimeError(f"Could not parse full fins_summary CSV for month={label}, key={key}") from last_error
    timings["read_csv"] = time.perf_counter() - start
    _cell5_log(
        f"fins_summary month={label} read_csv done rows={len(raw)} cols={len(raw.columns)} "
        f"columns_sample={list(raw.columns)[:20]} elapsed={timings['read_csv']:.1f}s"
    )

    frame = normalize_jquants_statements(raw)
    if raw is not None and not raw.empty and frame.empty:
        _cell5_log(
            f"fins_summary month={label} normalize produced 0 rows; "
            f"raw_columns={list(raw.columns)[:50]}"
        )
    elif not frame.empty:
        _cell5_log(
            f"fins_summary month={label} normalize done rows={len(frame)} "
            f"non_null_revenue={int(frame.get('total_revenue', pd.Series(dtype=float)).notna().sum()) if 'total_revenue' in frame else 0}"
        )
    start = time.perf_counter()
    _cell5_log(f"fins_summary month={label} cache write start rows={len(frame)} format={save_format}")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = _fins_summary_month_cache_path(data_dir, month, save_format)
    if save_format == "parquet":
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            path = _fins_summary_month_cache_path(data_dir, month, "csv")
            frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    timings["cache_write"] = time.perf_counter() - start
    _cell5_log(f"fins_summary month={label} cache write done path={path} rows={len(frame)} elapsed={timings['cache_write']:.1f}s")
    return {"month": month, "frame": frame, "timings": timings}


def _signed_get_with_rate_limit(
    url: str,
    *,
    timeout_sec: int,
    request_sleep_sec: float,
    rate_limit_sleep_sec: float,
    rate_limit_max_cooldowns: int,
    endpoint_name: str,
    mode: str,
) -> requests.Response:
    """Clean signed-URL GET with the same 429 cooldown policy as the client."""

    cooldowns = 0
    endpoint = "<signed-bulk-url>"
    params: dict[str, Any] = {}
    while True:
        resp = requests.get(url, timeout=timeout_sec, allow_redirects=True)
        if resp.status_code != 429:
            time.sleep(float(request_sleep_sec))
            if cooldowns:
                _cell5_log(
                    "[rate-limit] success after cooldown "
                    f"endpoint={endpoint} endpoint_name={endpoint_name} mode={mode} params={params}"
                )
            return resp
        if cooldowns >= int(rate_limit_max_cooldowns):
            return resp
        cooldowns += 1
        _cell5_log(
            "[rate-limit] 429 hit "
            f"endpoint={endpoint} endpoint_name={endpoint_name} mode={mode} "
            f"params={params} sleep={float(rate_limit_sleep_sec):g}s retry={cooldowns}/{int(rate_limit_max_cooldowns)}"
        )
        time.sleep(float(rate_limit_sleep_sec))
        _cell5_log(
            "[rate-limit] resumed "
            f"endpoint={endpoint} endpoint_name={endpoint_name} mode={mode} params={params}"
        )


def _filter_financial_summary_frame(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if frame.empty or "disclosed_date" not in frame.columns:
        return frame
    out = frame.copy()
    dates = pd.to_datetime(out["disclosed_date"], errors="coerce")
    start = params.get("from") or params.get("start") or params.get("start_date")
    end = params.get("to") or params.get("end") or params.get("end_date")
    if start is not None:
        out = out.loc[dates >= pd.Timestamp(str(start)).normalize()]
    if end is not None:
        out = out.loc[dates <= pd.Timestamp(str(end)).normalize()]
    return out.reset_index(drop=True)


def _read_tabular_content(
    content: bytes,
    content_type: str = "",
    name: str = "",
    url_fetcher: Any | None = None,
    depth: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if depth > 3:
        raise RuntimeError(f"Payload recursion exceeded limit for {name!r}")
    fmt = detect_payload_format(content, content_type, name)
    try:
        if fmt == "zip":
            return _read_zip_csv_content(content, name)
        if fmt == "gzip":
            decompressed = gzip.decompress(content)
            return _read_tabular_content(
                decompressed,
                "",
                f"{name}#gunzip",
                url_fetcher=url_fetcher,
                depth=depth + 1,
                diagnostics=diagnostics,
            )
        if fmt == "json":
            payload = _json_payload(content, content_type)
            urls = _extract_urls(payload)
            if urls:
                if url_fetcher is None:
                    raise RuntimeError(f"JSON payload contains URL but no fetcher is available for {name!r}: {urls[:3]!r}")
                return url_fetcher(urls[0], depth + 1)
            embedded = _extract_tabular_records(payload)
            if embedded is not None:
                return embedded
            raise RuntimeError(f"Non-tabular JSON payload for {name!r}: body_preview={_text_preview(content)!r}")
        if fmt == "xml_or_html":
            raise RuntimeError(f"HTML/XML payload for {name!r}: body_preview={_text_preview(content)!r}")
        if fmt == "csv_or_text":
            return _read_csv_text_content(content, name)
        raise RuntimeError(
            f"Unknown binary payload for {name!r}: first64_hex={content[:64].hex()}, "
            f"body_preview={_text_preview(content)!r}"
        )
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Declared/detected zip payload is invalid for {name!r}: "
            f"first64_hex={content[:64].hex()}, body_preview={_text_preview(content)!r}"
        ) from exc
    except Exception as exc:
        if diagnostics:
            raise RuntimeError(
                f"Tabular payload parse failed for {name!r}: detected_format={fmt}, "
                f"content_type={content_type!r}, content_length={len(content)}, "
                f"first64_hex={content[:64].hex()}, body_preview={_text_preview(content)!r}, "
                f"diagnostics={diagnostics!r}"
            ) from exc
        raise


def _read_zip_csv_content(content: bytes, name: str = "") -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(BytesIO(content)) as zf:
        members = zf.namelist()
        for member in members:
            lower = member.lower()
            if lower.endswith(".csv"):
                with zf.open(member) as fp:
                    frames.append(pd.read_csv(fp, dtype=str, low_memory=False))
            elif lower.endswith(".csv.gz") or lower.endswith(".gz"):
                with zf.open(member) as fp:
                    frames.append(_read_tabular_content(fp.read(), "application/gzip", member))
    if not frames:
        raise RuntimeError(f"Zip payload for {name!r} did not contain a CSV file.")
    return pd.concat(frames, ignore_index=True)


def _read_csv_text_content(content: bytes, name: str = "") -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding, dtype=str, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not parse CSV/text payload for {name!r}: body_preview={_text_preview(content)!r}") from last_error


def _filter_bulk_frame(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    lower = {str(col).lower(): col for col in out.columns}
    date_col = lower.get("date") or lower.get("datetime") or lower.get("local_date")
    code_col = lower.get("code") or lower.get("localcode") or lower.get("local_code")
    start = params.get("from") or params.get("start") or params.get("start_date")
    end = params.get("to") or params.get("end") or params.get("end_date")
    code = params.get("code")
    if date_col is not None and (start is not None or end is not None):
        dates = pd.to_datetime(out[date_col], errors="coerce")
        if start is not None:
            out = out.loc[dates >= pd.Timestamp(str(start)).normalize()]
        if end is not None:
            out = out.loc[dates <= pd.Timestamp(str(end)).normalize()]
    if code_col is not None and code is not None:
        out = out.loc[out[code_col].astype(str).str[:4].eq(str(code)[:4])]
    return out.reset_index(drop=True)


def _date_compact(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _series(frame: pd.DataFrame, *names: str, default: Any = np.nan) -> pd.Series:
    lower = {str(col).strip().lower(): col for col in frame.columns}
    for name in names:
        found = lower.get(str(name).strip().lower())
        if found is not None:
            return frame[found]
    return pd.Series(default, index=frame.index)


def _first_existing_column(frame: pd.DataFrame, *names: str) -> str | None:
    lower = {str(col).strip().lower(): col for col in frame.columns}
    canonical = {_canonical_col_name(col): col for col in frame.columns}
    for name in names:
        found = lower.get(str(name).strip().lower())
        if found is not None:
            return str(found)
        found = canonical.get(_canonical_col_name(name))
        if found is not None:
            return str(found)
    return None


def _series_by_candidates(frame: pd.DataFrame, *names: str, default: Any = np.nan) -> pd.Series:
    found = _first_existing_column(frame, *names)
    if found is not None:
        return frame[found]
    return pd.Series(default, index=frame.index)


def _has_any_column(frame: pd.DataFrame, names: Iterable[str]) -> bool:
    lower = {str(col).strip().lower() for col in frame.columns}
    return any(str(name).strip().lower() in lower for name in names)


def _present_columns(frame: pd.DataFrame, names: Iterable[str]) -> list[str]:
    lower = {str(col).strip().lower(): str(col) for col in frame.columns}
    out = []
    for name in names:
        found = lower.get(str(name).strip().lower())
        if found is not None:
            out.append(found)
    return out


def _canonical_col_name(value: object) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _matching_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    candidate_keys = {_canonical_col_name(c) for c in candidates}
    return [str(col) for col in frame.columns if _canonical_col_name(col) in candidate_keys]


def _first_matching_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    matches = _matching_columns(frame, candidates)
    return matches[0] if matches else None


def _month_end_snapshots(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.date_range(start=start, end=end, freq="ME")
    dates = sorted(set([start.normalize(), end.normalize(), *[d.normalize() for d in months]]))
    return dates


def _listed_info_snapshot_request_date(
    client: "JQuantsApiClient",
    snapshot: pd.Timestamp,
    *,
    max_lookback_days: int = 10,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Fetch /equities/master without accepting next-business-day lookahead."""

    target = pd.Timestamp(snapshot).normalize()
    requested = target
    last_raw = pd.DataFrame()
    for _ in range(max_lookback_days + 1):
        raw = client.get_dataset("listed_info", {"date": _date_compact(requested)}, "info")
        last_raw = raw
        if raw.empty:
            requested = requested - pd.Timedelta(days=1)
            continue
        response_dates = pd.to_datetime(
            _series_by_candidates(raw, "Date", "date", default=pd.NaT),
            errors="coerce",
        ).dropna()
        if response_dates.empty:
            return requested, raw
        effective = response_dates.dt.normalize().max()
        if effective <= target:
            return effective, raw
        requested = requested - pd.Timedelta(days=1)
    raise RuntimeError(
        "Could not fetch listed_info snapshot without next-business-day lookahead. "
        f"target={target.date()} last_requested={requested.date()} last_rows={len(last_raw)}"
    )


def _non_null_stats(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in columns:
        if col not in frame.columns:
            out[col] = {"exists": False, "non_null": 0, "nan_ratio": None, "dtype": None}
            continue
        non_null = int(frame[col].notna().sum())
        out[col] = {
            "exists": True,
            "non_null": non_null,
            "nan_ratio": float(frame[col].isna().mean()) if len(frame) else 0.0,
            "dtype": str(frame[col].dtype),
        }
    return out


def _sample_rows_for_debug(frame: pd.DataFrame, *, code_col: str | None, date_col: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame.head(0)
    work = frame
    if code_col and code_col in work.columns:
        code_norm = work[code_col].map(normalize_code)
        exact = work.loc[code_norm.eq("1301")]
        if not exact.empty:
            work = exact
    if date_col and date_col in work.columns:
        dates = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
        exact = work.loc[dates.eq(pd.Timestamp("2025-04-30"))]
        if not exact.empty:
            work = exact
    return work.head(3)


def log_cache_value_diagnostics(stage: str, table_name: str, frame: pd.DataFrame) -> None:
    """Plain-text diagnostics for raw/normalized/cache value integrity."""

    if table_name == "prices":
        cols = ["open", "high", "low", "close", "volume", "adjustment_close", "O", "H", "L", "C", "Vo"]
        code_col = _first_existing_column(frame, "code", "Code", "LocalCode")
        date_col = _first_existing_column(frame, "date", "Date")
    elif table_name == "index_prices":
        cols = ["open", "high", "low", "close", "volume", "O", "H", "L", "C", "Vo"]
        code_col = _first_existing_column(frame, "index_code", "Code", "code")
        date_col = _first_existing_column(frame, "date", "Date")
    elif table_name == "sector":
        cols = ["sector", "S17", "S17Nm", "S33", "S33Nm", "Sector17CodeName", "Sector33CodeName", "Mkt", "MktNm"]
        code_col = _first_existing_column(frame, "code", "Code", "LocalCode")
        date_col = _first_existing_column(frame, "asof_date", "date", "Date")
    else:
        return
    present_cols = [col for col in cols if col in frame.columns]
    _cell5_log(
        f"{stage} diagnostics rows={len(frame)} columns={list(frame.columns)[:40]} "
        f"stats={_non_null_stats(frame, present_cols)}"
    )
    sample = _sample_rows_for_debug(frame, code_col=code_col, date_col=date_col)
    if not sample.empty:
        keep = [col for col in [date_col, code_col, *present_cols] if col and col in sample.columns]
        _cell5_log(f"{stage} sample:\n{sample[keep].head(3).to_string(index=False)}")


def _write_table(frame: pd.DataFrame, data_dir: Path, name: str, save_format: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    for date_col in ("asof_date", "date", "disclosed_date"):
        if date_col in frame.columns:
            dates = pd.to_datetime(frame[date_col], errors="coerce")
            frame[date_col] = np.where(dates.notna(), dates.dt.strftime("%Y-%m-%d"), frame[date_col])
    _cell5_log(f"cache write start table={name} rows={len(frame)} format={save_format}")
    log_cache_value_diagnostics(f"{name} write-before", name, frame)
    if name in {"prices", "index_prices", "sector"}:
        validate_cache_table_contract(frame, name, min_rows=1)
    def _cleanup_stale_alternates(written_path: Path) -> None:
        for suffix in (".parquet", ".csv"):
            candidate = data_dir / f"{name}{suffix}"
            if candidate == written_path:
                continue
            if candidate.exists():
                try:
                    candidate.unlink()
                    _cell5_log(f"cache cleanup removed stale alternate path={candidate}")
                except Exception as exc:
                    _cell5_log(f"cache cleanup skipped stale alternate path={candidate} reason={exc!r}")
    if save_format == "parquet":
        path = data_dir / f"{name}.parquet"
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            pass
        else:
            _cell5_log(f"cache write done table={name} path={path} rows={len(frame)}")
            readback = pd.read_parquet(path)
            log_cache_value_diagnostics(f"{name} read-after-parquet", name, readback)
            if name in {"prices", "index_prices", "sector"}:
                validate_cache_table_contract(readback, name, min_rows=1)
            _cleanup_stale_alternates(path)
            return path
    path = data_dir / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    _cell5_log(f"cache write done table={name} path={path} rows={len(frame)}")
    readback = pd.read_csv(path, dtype=str, low_memory=False)
    log_cache_value_diagnostics(f"{name} read-after-csv", name, readback)
    if name in {"prices", "index_prices", "sector"}:
        validate_cache_table_contract(readback, name, min_rows=1)
    _cleanup_stale_alternates(path)
    return path


def _append_existing(data_dir: Path, name: str, frame: pd.DataFrame, keys: list[str], save_format: str) -> Path:
    existing_frames: list[pd.DataFrame] = []
    for suffix in (".parquet", ".csv"):
        candidate = data_dir / f"{name}{suffix}"
        if candidate.exists():
            existing = pd.read_parquet(candidate) if suffix == ".parquet" else pd.read_csv(candidate, dtype=str, low_memory=False)
            if not existing.empty:
                existing_frames.append(existing)
    if existing_frames:
        frame = pd.concat([*existing_frames, frame], ignore_index=True)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=[k for k in keys if k in frame.columns], keep="last")
    return _write_table(frame, data_dir, name, save_format)


def _monthly_price_shards_present(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp, save_format: str) -> bool:
    for month in _month_starts(start, end):
        key = month.strftime("%Y%m")
        path = _shard_path(data_dir, "prices", key, save_format)
        if not path.exists() and not path.with_suffix(".csv").exists():
            return False
    return True


def _monthly_topix_shards_present(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp, save_format: str) -> bool:
    suffix = ".parquet" if save_format == "parquet" else ".csv"
    for month in _month_starts(start, end):
        key = pd.Timestamp(month).strftime("%Y%m")
        path = data_dir / "index_prices" / f"topix_{key}{suffix}"
        if not path.exists() and not path.with_suffix(".csv").exists():
            return False
    return True


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    start = pd.Timestamp(start).normalize().replace(day=1)
    end = pd.Timestamp(end).normalize().replace(day=1)
    return [pd.Timestamp(d).normalize() for d in pd.date_range(start, end, freq="MS")]


def _month_bounds(month_start: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    m0 = pd.Timestamp(month_start).normalize()
    m1 = (m0 + pd.offsets.MonthEnd(0)).normalize()
    return max(m0, pd.Timestamp(start).normalize()), min(m1, pd.Timestamp(end).normalize())


def _shard_path(data_dir: Path, dataset: str, key: str, save_format: str) -> Path:
    suffix = ".parquet" if save_format == "parquet" else ".csv"
    key_text = str(key)
    if dataset == "prices" and len(key_text) == 6 and key_text.isdigit():
        return data_dir / "daily_quotes" / f"{key_text[:4]}-{key_text[4:]}{suffix}"
    if dataset == "universe" and len(key_text) == 8 and key_text.isdigit():
        return data_dir / "listed_info" / f"{key_text[:4]}-{key_text[4:6]}-{key_text[6:]}{suffix}"
    if dataset == "sector" and len(key_text) == 8 and key_text.isdigit():
        return data_dir / "sector" / f"{key_text[:4]}-{key_text[4:6]}-{key_text[6:]}{suffix}"
    if dataset == "listed_caps" and len(key_text) == 8 and key_text.isdigit():
        return data_dir / "market_cap" / f"{key_text[:4]}-{key_text[4:6]}-{key_text[6:]}{suffix}"
    if dataset == "financial_summary" and len(key_text) == 8 and key_text.isdigit():
        return data_dir / "financial_summary" / f"{key_text[:4]}-{key_text[4:6]}-{key_text[6:]}{suffix}"
    if dataset == "margin" and len(key_text) == 8 and key_text.isdigit():
        return data_dir / "margin" / f"{key_text[:4]}-{key_text[4:6]}-{key_text[6:]}{suffix}"
    return data_dir / "shards" / dataset / f"{key_text}{suffix}"


def _read_shard(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _write_shard(frame: pd.DataFrame, path: Path, save_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = "prices" if "daily_quotes" in str(path) or path.parent.name == "prices" else ("sector" if path.parent.name == "sector" else "")
    if dataset_name:
        log_cache_value_diagnostics(f"{dataset_name} shard write-before {path.stem}", dataset_name, frame)
        validate_cache_table_contract(frame, dataset_name, min_rows=1)
    if save_format == "parquet":
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            path = path.with_suffix(".csv")
        else:
            if dataset_name:
                readback = pd.read_parquet(path)
                log_cache_value_diagnostics(f"{dataset_name} shard read-after-parquet {path.stem}", dataset_name, readback)
                validate_cache_table_contract(readback, dataset_name, min_rows=1)
            return
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    if dataset_name:
        readback = pd.read_csv(path, dtype=str, low_memory=False)
        log_cache_value_diagnostics(f"{dataset_name} shard read-after-csv {path.stem}", dataset_name, readback)
        validate_cache_table_contract(readback, dataset_name, min_rows=1)


def rebuild_corrupt_cache_tables(data_dir: Path) -> None:
    """Delete provider-ready and shard files known to carry normalized market data."""

    targets = [
        data_dir / "prices.parquet",
        data_dir / "prices.csv",
        data_dir / "index_prices.parquet",
        data_dir / "index_prices.csv",
        data_dir / "sector.parquet",
        data_dir / "sector.csv",
        data_dir / "market_cap.parquet",
        data_dir / "market_cap.csv",
        data_dir / "financial_summary.parquet",
        data_dir / "financial_summary.csv",
        data_dir / "margin.parquet",
        data_dir / "margin.csv",
    ]
    for target in targets:
        if target.exists():
            _cell5_log(f"force rebuild delete file={target}")
            target.unlink()
    for dirname in ["daily_quotes", "sector", "market_cap", "financial_summary", "margin", "shards"]:
        path = data_dir / dirname
        if path.exists() and path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in {".parquet", ".csv"}:
                    _cell5_log(f"force rebuild delete shard={child}")
                    child.unlink()
    for pattern in ("financial_summary_*.parquet", "financial_summary_*.csv"):
        for child in data_dir.glob(pattern):
            if child.is_file():
                _cell5_log(f"force rebuild delete shard={child}")
                child.unlink()


def load_or_download_price_shards(
    client: JQuantsApiClient,
    data_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    save_format: str,
) -> pd.DataFrame:
    """Ensure monthly price shards cover [start, end], then return that range."""

    assert_full_month_range(start, end, context="daily_quotes monthly shard cache")
    months = _month_starts(start, end)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    _cell5_log(f"price shard coverage check start months={len(months)} range={start.date()}..{end.date()}")
    for month in months:
        key = month.strftime("%Y%m")
        path = _shard_path(data_path, "prices", key, save_format)
        if path.exists():
            _cell5_log(f"cache hit prices shard={key} path={path}")
        else:
            missing.append(key)
    if missing:
        _cell5_log(f"cache miss prices shards={missing}")
    else:
        _cell5_log("cache hit prices all required monthly shards present")

    for idx, month in enumerate(months, start=1):
        key = month.strftime("%Y%m")
        path = _shard_path(data_path, "prices", key, save_format)
        alt = path.with_suffix(".csv")
        if path.exists() or alt.exists():
            frames.append(_read_shard(path if path.exists() else alt))
            continue
        shard_start, shard_end = _month_bounds(month, start, end)
        assert_full_month_range(shard_start, shard_end, context=f"daily_quotes shard {key}")
        _cell5_log(f"download missing prices shard {idx}/{len(months)} key={key} range={shard_start.date()}..{shard_end.date()}")
        raw = client.get_dataset("daily_quotes", {"from": _date_compact(shard_start), "to": _date_compact(shard_end)}, "daily_quotes")
        norm = normalize_jquants_prices(raw)
        _write_shard(norm, path, save_format)
        _cell5_log(f"downloaded prices shard key={key} rows={len(norm)} path={path}")
        frames.append(norm)

    if not frames:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume"])
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None)
    return data.loc[data["date"].between(start, end, inclusive="both")].reset_index(drop=True)


def load_or_download_listed_snapshot(
    client: JQuantsApiClient,
    data_path: Path,
    snapshot: pd.Timestamp,
    save_format: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = pd.Timestamp(snapshot).strftime("%Y%m%d")
    paths = {
        "universe": _shard_path(data_path, "universe", key, save_format),
        "sector": _shard_path(data_path, "sector", key, save_format),
        "listed_caps": _shard_path(data_path, "listed_caps", key, save_format),
    }
    existing = all(path.exists() or path.with_suffix(".csv").exists() for path in paths.values())
    if existing:
        _cell5_log(f"cache hit listed_info snapshot={key}")
        read = {}
        for name, path in paths.items():
            read[name] = _read_shard(path if path.exists() else path.with_suffix(".csv"))
        return read["universe"], read["sector"], read["listed_caps"]

    _cell5_log(f"cache miss listed_info snapshot={key}; downloading")
    effective_snapshot, raw = _listed_info_snapshot_request_date(client, pd.Timestamp(snapshot))
    _cell5_log(
        "listed_info snapshot raw rows="
        f"{len(raw)} requested_date={pd.Timestamp(snapshot).date()} effective_snapshot={effective_snapshot.date()}"
    )
    uni, sector, caps = normalize_jquants_listed_info(raw, effective_snapshot)
    _write_shard(uni, paths["universe"], save_format)
    _write_shard(sector, paths["sector"], save_format)
    _write_shard(caps, paths["listed_caps"], save_format)
    return uni, sector, caps


def _date_shard_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        return []
    return [pd.Timestamp(d).normalize() for d in pd.date_range(start_ts, end_ts, freq="D")]


def load_or_download_financial_summary_shards(
    client: JQuantsApiClient,
    data_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    save_format: str,
) -> pd.DataFrame:
    """Ensure /fins/summary date shards cover [start, end]."""

    dates = _date_shard_range(start, end)
    frames: list[pd.DataFrame] = []
    _cell5_log(f"financial_summary date shard coverage start days={len(dates)} range={start.date()}..{end.date()}")
    for idx, day in enumerate(dates, start=1):
        key = day.strftime("%Y%m%d")
        path = _shard_path(data_path, "financial_summary", key, save_format)
        alt = path.with_suffix(".csv")
        if path.exists() or alt.exists():
            frame = _read_shard(path if path.exists() else alt)
        else:
            _cell5_log(f"download financial_summary shard {idx}/{len(dates)} date={key}")
            raw = client.get_dataset("statements", {"date": key}, "data")
            frame = normalize_jquants_statements(raw)
            _write_shard(frame, path, save_format)
            _cell5_log(f"downloaded financial_summary shard date={key} raw_rows={len(raw)} rows={len(frame)} path={path}")
        if not frame.empty:
            frames.append(frame)
        if idx == len(dates) or idx % 20 == 0:
            _cell5_log(f"financial_summary shards progress {idx}/{len(dates)} non_empty_days={len(frames)}")
    if not frames:
        return pd.DataFrame(columns=["disclosed_date", "code"])
    out = pd.concat(frames, ignore_index=True)
    out["disclosed_date"] = pd.to_datetime(out["disclosed_date"], errors="coerce").dt.tz_localize(None)
    return out.loc[out["disclosed_date"].between(start, end, inclusive="both")].drop_duplicates(["disclosed_date", "code"], keep="last").reset_index(drop=True)


def load_or_download_margin_shards(
    client: JQuantsApiClient,
    data_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    save_format: str,
    *,
    endpoint_name: str,
) -> pd.DataFrame:
    """Ensure weekly margin-interest/date shards cover [start, end].

    J-Quants V2 margin endpoints require code or date.  For full-market cache
    building we request date shards; empty weekdays are cached as empty files.
    """

    dates = [d for d in _date_shard_range(start, end) if d.weekday() < 5]
    frames: list[pd.DataFrame] = []
    _cell5_log(f"margin date shard coverage start endpoint={endpoint_name} weekdays={len(dates)} range={start.date()}..{end.date()}")
    for idx, day in enumerate(dates, start=1):
        key = day.strftime("%Y%m%d")
        path = _shard_path(data_path, "margin", key, save_format)
        alt = path.with_suffix(".csv")
        if path.exists() or alt.exists():
            frame = _read_shard(path if path.exists() else alt)
        else:
            _cell5_log(f"download margin shard {idx}/{len(dates)} endpoint={endpoint_name} date={key}")
            raw = client.get_dataset(endpoint_name, {"date": key}, endpoint_name)
            frame = normalize_jquants_margin(raw)
            _write_shard(frame, path, save_format)
            _cell5_log(f"downloaded margin shard date={key} raw_rows={len(raw)} rows={len(frame)} path={path}")
        if not frame.empty:
            frames.append(frame)
        if idx == len(dates) or idx % 20 == 0:
            _cell5_log(f"margin shards progress {idx}/{len(dates)} non_empty_days={len(frames)}")
    if not frames:
        return pd.DataFrame(columns=["date", "published_date", "code", "margin_buy_balance", "margin_sell_balance", "margin_buy_change", "margin_sell_change"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])
    for balance_col, change_col in [("margin_buy_balance", "margin_buy_change"), ("margin_sell_balance", "margin_sell_change")]:
        if change_col not in out.columns or pd.to_numeric(out[change_col], errors="coerce").notna().sum() == 0:
            out[change_col] = pd.to_numeric(out[balance_col], errors="coerce").groupby(out["code"]).diff()
    return out.loc[out["date"].between(start, end, inclusive="both")].reset_index(drop=True)


CACHE_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "universe": {"asof_date", "code", "in_universe"},
    "sector": {"asof_date", "code", "sector"},
    "market_cap": {"asof_date", "code", "market_cap"},
    "prices": {"date", "code", "open", "high", "low", "close", "volume"},
    "index_prices": {"date", "index_code", "close"},
}


def _read_cache_table_for_debug(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _cache_file_for(data_dir: str | Path, name: str) -> Path | None:
    base = Path(data_dir)
    for suffix in (".parquet", ".csv"):
        path = base / f"{name}{suffix}"
        if path.exists():
            return path
    return None


def _read_cache_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _existing_table_range(path: Path | None, date_col: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "min": None, "max": None, "rows": 0}
    frame = _read_cache_table(path)
    if date_col not in frame.columns:
        return {"exists": True, "min": None, "max": None, "rows": len(frame), "missing_date_col": date_col}
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    return {
        "exists": True,
        "min": dates.min() if not dates.empty else None,
        "max": dates.max() if not dates.empty else None,
        "rows": len(frame),
    }


def _existing_table_reusable(
    data_dir: str | Path,
    name: str,
    *,
    date_col: str,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
    asof_date: pd.Timestamp | None = None,
) -> tuple[bool, Path | None]:
    path = _cache_file_for(data_dir, name)
    if path is None:
        return False, None
    try:
        frame = _read_cache_table(path)
        validate_cache_table_contract(frame, name, min_rows=1, asof_date=asof_date)
        info = _existing_table_range(path, date_col)
        min_date = info.get("min")
        max_date = info.get("max")
        if min_date is None or max_date is None:
            return False, path
        covers = pd.Timestamp(min_date).normalize() <= pd.Timestamp(required_start).normalize() and pd.Timestamp(max_date).normalize() >= pd.Timestamp(required_end).normalize()
        return bool(covers), path
    except Exception:
        return False, path


def validate_cache_table_contract(frame: pd.DataFrame, name: str, *, min_rows: int = 0, asof_date: pd.Timestamp | None = None) -> None:
    required = CACHE_REQUIRED_COLUMNS.get(name, set())
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Cache table {name!r} is missing required columns: {missing}; columns={list(frame.columns)}")
    if len(frame) < int(min_rows):
        raise RuntimeError(f"Cache table {name!r} has {len(frame)} rows; expected at least {min_rows}.")
    work = frame.copy()
    if "code" in work.columns:
        codes = work["code"].map(normalize_code)
        if len(work) and codes.notna().sum() == 0:
            raise RuntimeError(f"Cache table {name!r} has no normalizable code values. sample={work['code'].head(5).tolist()}")
    date_col = "asof_date" if "asof_date" in work.columns else ("date" if "date" in work.columns else None)
    if date_col:
        dates = pd.to_datetime(work[date_col], errors="coerce")
        if len(work) and dates.notna().sum() == 0:
            raise RuntimeError(f"Cache table {name!r} has no parseable {date_col} values.")
        if asof_date is not None and dates.le(pd.Timestamp(asof_date).normalize()).sum() == 0:
            raise RuntimeError(f"Cache table {name!r} has no rows on or before {pd.Timestamp(asof_date).date()}.")
    if name == "market_cap":
        numeric = pd.to_numeric(work["market_cap"], errors="coerce")
        if len(work) and numeric.notna().sum() == 0:
            raise RuntimeError(
                "Cache table 'market_cap' has no numeric market_cap values. "
                "No usable raw market-cap column was detected, no usable shares_outstanding column was detected, "
                "or no close price existed on/before snapshot dates."
            )
        positive = numeric.gt(0).sum()
        if len(work) and positive == 0:
            raise RuntimeError(
                "Cache table 'market_cap' has no positive market_cap values. "
                f"numeric_rows={int(numeric.notna().sum())}, sample={work[['code', 'asof_date', 'market_cap']].head(5).to_dict('records')}"
            )
        if "market_cap_source" in work.columns:
            sources = work["market_cap_source"].astype(str)
            debug_mask = sources.str.startswith("debug_", na=False)
            if debug_mask.any():
                sample = work.loc[debug_mask, ["code", "asof_date", "market_cap_source"]].head(5).to_dict("records")
                raise RuntimeError(
                    "Cache table 'market_cap' contains debug fallback sources, which are not allowed for formal PIT backtests. "
                    f"sample={sample}"
                )
    if name == "prices":
        close_non_null = int(pd.to_numeric(work["close"], errors="coerce").notna().sum())
        volume_non_null = int(pd.to_numeric(work["volume"], errors="coerce").notna().sum())
        if close_non_null == 0 or volume_non_null == 0:
            raise RuntimeError(
                "Cache table 'prices' has invalid OHLCV values: "
                f"close_non_null={close_non_null}, volume_non_null={volume_non_null}, "
                f"sample={work.head(5).to_dict('records')}"
            )
    if name == "index_prices":
        close_non_null = int(pd.to_numeric(work["close"], errors="coerce").notna().sum())
        if close_non_null == 0:
            raise RuntimeError(
                "Cache table 'index_prices' has no numeric close values. "
                f"sample={work.head(5).to_dict('records')}"
            )
    if name == "sector":
        sector_non_null = int(work["sector"].replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA}).notna().sum())
        if sector_non_null == 0:
            raise RuntimeError(
                "Cache table 'sector' has no usable sector values. "
                f"sample={work.head(5).to_dict('records')}"
            )


def debug_dump_cache_summary(data_dir: str | Path, *, required_tables: Iterable[str] | None = None, asof_date: pd.Timestamp | None = None) -> dict[str, dict[str, Any]]:
    """Print compact cache diagnostics immediately after J-Quants normalization."""

    tables = ["universe", "sector", "market_cap", "prices", "index_prices", "margin", "financial_summary"]
    required = set(required_tables or ())
    summary: dict[str, dict[str, Any]] = {}
    _cell5_log("cache summary start")
    for name in tables:
        path = _cache_file_for(data_dir, name)
        info: dict[str, Any] = {"path": str(path) if path else None, "exists": path is not None}
        if path is None:
            _cell5_log(f"cache summary {name}: missing")
            if name in required:
                raise RuntimeError(f"Required cache table {name!r} is missing under {data_dir}")
            summary[name] = info
            continue
        frame = _read_cache_table_for_debug(path)
        info.update({"rows": len(frame), "columns": list(frame.columns), "dtypes": {c: str(t) for c, t in frame.dtypes.items()}})
        _cell5_log(f"cache summary {name}: path={path}, rows={len(frame)}, columns={list(frame.columns)}")
        if not frame.empty:
            _cell5_log(f"cache summary {name} head:\n{frame.head(3).to_string(index=False)}")
        if name in required:
            validate_cache_table_contract(frame, name, min_rows=1, asof_date=asof_date)
        summary[name] = info
    return summary


def normalize_jquants_prices(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume"])
    _cell5_log(f"daily_quotes normalize start raw_rows={len(raw)}")
    log_cache_value_diagnostics("daily_quotes raw-api-bulk", "prices", raw)
    start = time.perf_counter()
    open_col = _first_existing_column(raw, "Open", "open", "O", "o", "AdjustmentOpen", "AdjOpen")
    high_col = _first_existing_column(raw, "High", "high", "H", "h", "AdjustmentHigh", "AdjHigh")
    low_col = _first_existing_column(raw, "Low", "low", "L", "l", "AdjustmentLow", "AdjLow")
    close_col = _first_existing_column(raw, "Close", "close", "C", "c", "AdjustmentClose", "AdjClose")
    volume_col = _first_existing_column(raw, "Volume", "volume", "Vo", "VO", "V", "TurnoverVolume", "TradingVolume")
    _cell5_log(
        "daily_quotes column mapping "
        f"open={open_col}, high={high_col}, low={low_col}, close={close_col}, volume={volume_col}"
    )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(_series_by_candidates(raw, "Date", "date"), errors="coerce"),
            "code": _series_by_candidates(raw, "Code", "code", "LocalCode", "local_code").map(normalize_code),
            "open": pd.to_numeric(raw[open_col] if open_col else np.nan, errors="coerce"),
            "high": pd.to_numeric(raw[high_col] if high_col else np.nan, errors="coerce"),
            "low": pd.to_numeric(raw[low_col] if low_col else np.nan, errors="coerce"),
            "close": pd.to_numeric(raw[close_col] if close_col else np.nan, errors="coerce"),
            "volume": pd.to_numeric(raw[volume_col] if volume_col else np.nan, errors="coerce"),
            "trading_value": pd.to_numeric(_series_by_candidates(raw, "TradingValue", "trading_value", "Va", "VA"), errors="coerce"),
            "adjustment_factor": pd.to_numeric(_series_by_candidates(raw, "AdjustmentFactor", "adjustment_factor", "AdjFactor"), errors="coerce"),
            "adjustment_open": pd.to_numeric(_series_by_candidates(raw, "AdjustmentOpen", "adjustment_open", "adjusted_open", "AdjOpen", "AdjO"), errors="coerce"),
            "adjustment_high": pd.to_numeric(_series_by_candidates(raw, "AdjustmentHigh", "adjustment_high", "adjusted_high", "AdjHigh", "AdjH"), errors="coerce"),
            "adjustment_low": pd.to_numeric(_series_by_candidates(raw, "AdjustmentLow", "adjustment_low", "adjusted_low", "AdjLow", "AdjL"), errors="coerce"),
            "adjustment_close": pd.to_numeric(_series_by_candidates(raw, "AdjustmentClose", "adjustment_close", "adjusted_close", "AdjClose", "AdjC"), errors="coerce"),
            "adjustment_volume": pd.to_numeric(_series_by_candidates(raw, "AdjustmentVolume", "adjustment_volume", "adjusted_volume", "AdjVolume", "AdjVo"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "code"])
    if (
        not out.empty
        and out["adjustment_factor"].notna().any()
        and not out["adjustment_close"].notna().any()
    ):
        # Bulk CSV files include AdjFactor but not adjusted OHLCV.  J-Quants
        # factors are cumulative backward from the latest row; applying the
        # prior cumulative factor creates a point-in-time adjusted history
        # without using yfinance.
        work = out.sort_values(["code", "date"], ascending=[True, False], kind="mergesort").copy()
        factor = pd.to_numeric(work["adjustment_factor"], errors="coerce").replace(0, np.nan).fillna(1.0)
        work["_cum_factor"] = factor.groupby(work["code"], sort=False).cumprod()
        work["_adj_multiplier"] = work.groupby("code", sort=False)["_cum_factor"].shift(1, fill_value=1.0)
        for src, dst in [
            ("open", "adjustment_open"),
            ("high", "adjustment_high"),
            ("low", "adjustment_low"),
            ("close", "adjustment_close"),
        ]:
            work[dst] = pd.to_numeric(work[src], errors="coerce") * pd.to_numeric(work["_adj_multiplier"], errors="coerce")
        multiplier = pd.to_numeric(work["_adj_multiplier"], errors="coerce").replace(0, np.nan)
        work["adjustment_volume"] = pd.to_numeric(work["volume"], errors="coerce") / multiplier
        out = work.drop(columns=["_cum_factor", "_adj_multiplier"]).sort_values(["code", "date"], kind="mergesort")
        _cell5_log("daily_quotes adjusted OHLCV derived from AdjFactor")
    log_cache_value_diagnostics("daily_quotes normalized", "prices", out)
    _cell5_log(f"daily_quotes normalize done rows={len(out)} elapsed={time.perf_counter() - start:.1f}s")
    return out


def normalize_jquants_listed_info(raw: pd.DataFrame, snapshot_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        empty_uni = pd.DataFrame(columns=["asof_date", "code", "in_universe"])
        empty_sector = pd.DataFrame(columns=["asof_date", "code", "sector"])
        empty_caps = pd.DataFrame(columns=["asof_date", "code", "market_cap", "shares_outstanding"])
        return empty_uni, empty_sector, empty_caps
    _cell5_log(
        "listed_info normalize "
        f"snapshot={pd.Timestamp(snapshot_date).date()}, raw_rows={len(raw)}, raw_columns={list(raw.columns)}"
    )
    log_cache_value_diagnostics("listed_info raw-api", "sector", raw)
    code = _series_by_candidates(raw, "Code", "code", "LocalCode", "local_code").map(normalize_code)
    missing_core = []
    if code.notna().sum() == 0:
        missing_core.append("Code/LocalCode")
    scale_cols = ["ScaleCat", "ScaleCategory", "scale_cat", "scale_category", "TOPIXScaleCategory", "topix_scale_category"]
    market_name_cols = ["MktNm", "MarketCodeName", "market_code_name", "Section", "section", "MarketSegment", "market_segment"]
    market_code_cols = ["Mkt", "MarketCode", "market_code"]
    scale = _series(raw, *scale_cols, default="").astype(str)
    market_name = _series(raw, *market_name_cols, default=pd.NA)
    market_code = _series(raw, *market_code_cols, default=pd.NA)
    s17 = _series(raw, "S17", "Sector17Code", "sector17_code", default=pd.NA)
    s17_name = _series(raw, "S17Nm", "Sector17CodeName", "sector17_code_name", default=pd.NA)
    s33 = _series(raw, "S33", "Sector33Code", "sector33_code", default=pd.NA)
    s33_name = _series(raw, "S33Nm", "Sector33CodeName", "sector33_code_name", default=pd.NA)
    in_topix = scale.str.upper().str.contains("TOPIX", na=False)
    if not in_topix.any():
        raise RuntimeError(
            "listed_info normalize could not find any TOPIX scale classifications in /equities/master. "
            f"available_scale_columns={_present_columns(raw, scale_cols)} sample_scale_values={scale.dropna().astype(str).head(10).tolist()}"
        )
    if missing_core:
        _cell5_log(f"listed_info normalize missing required source columns: {missing_core}")
    asof = pd.Timestamp(snapshot_date).normalize()
    universe = pd.DataFrame(
        {
            "asof_date": asof,
            "code": code,
            "in_universe": in_topix,
            "scale_cat": scale.replace({"": pd.NA}),
            "market_code": market_code,
            "market_name": market_name,
            "s17": s17,
            "s17_name": s17_name,
            "s33": s33,
            "s33_name": s33_name,
        }
    )
    sector_name = _series_by_candidates(
        raw,
        "S33Nm",
        "Sector33CodeName",
        "sector33_code_name",
        "S17Nm",
        "Sector17CodeName",
        "sector17_code_name",
        default=pd.NA,
    )
    if sector_name.isna().all():
        sector_name = _series_by_candidates(raw, "S33", "Sector33Code", "sector33_code", "S17", "Sector17Code", "sector17_code", default=pd.NA).astype(str)
    sector = pd.DataFrame(
        {
            "asof_date": asof,
            "code": code,
            "sector": sector_name,
            "s17": s17,
            "s17_name": s17_name,
            "s33": s33,
            "s33_name": s33_name,
            "scale_cat": scale.replace({"": pd.NA}),
            "market_code": market_code,
            "market_name": market_name,
        }
    )
    log_cache_value_diagnostics("listed_info normalized-sector", "sector", sector)
    caps = pd.DataFrame({"asof_date": asof, "code": code})
    shares_candidates = [
        "ListedShares",
        "listed_shares",
        "ListedShare",
        "listed_share",
        "ListedShareNumber",
        "listed_share_number",
        "NumberOfIssuedAndOutstandingShares",
        "number_of_issued_and_outstanding_shares",
        "NumberOfIssuedShares",
        "number_of_issued_shares",
        "IssuedShares",
        "issued_shares",
        "IssuedShareEquityQuote",
        "SharesOutstanding",
        "shares_outstanding",
        "OutstandingShares",
        "outstanding_shares",
        "NumberOfShares",
        "number_of_shares",
        "CommonStocksNumberOfShares",
        "common_stocks_number_of_shares",
        "発行済株式数",
        "上場株式数",
    ]
    market_cap_candidates = [
        "MarketCapitalization",
        "market_capitalization",
        "MarketCap",
        "market_cap",
        "market_cap_jpy",
        "MktCap",
        "mkt_cap",
        "Capitalization",
        "capitalization",
        "TotalMarketValue",
        "total_market_value",
        "時価総額",
    ]
    shares_matches = _matching_columns(raw, shares_candidates)
    market_cap_matches = _matching_columns(raw, market_cap_candidates)
    shares_col = shares_matches[0] if shares_matches else None
    market_cap_col = market_cap_matches[0] if market_cap_matches else None
    _cell5_log(
        "listed_info market-cap detection "
        f"shares_candidates={shares_candidates}, detected_shares_columns={shares_matches}, chosen_shares_col={shares_col}, "
        f"market_cap_candidates={market_cap_candidates}, detected_market_cap_columns={market_cap_matches}, chosen_market_cap_col={market_cap_col}"
    )
    if shares_col:
        caps["shares_outstanding"] = pd.to_numeric(raw[shares_col], errors="coerce")
    else:
        caps["shares_outstanding"] = np.nan
    caps["market_cap"] = pd.to_numeric(raw[market_cap_col], errors="coerce") if market_cap_col else np.nan
    universe = universe.dropna(subset=["code"]).drop_duplicates(["asof_date", "code"])
    sector = sector.dropna(subset=["code"]).drop_duplicates(["asof_date", "code"])
    caps = caps.dropna(subset=["code"]).drop_duplicates(["asof_date", "code"])
    _cell5_log(
        "listed_info normalize done "
        f"universe_rows={len(universe)}, in_universe_rows={int(universe['in_universe'].sum()) if 'in_universe' in universe else 0}, "
        f"sector_rows={len(sector)}, caps_rows={len(caps)}, "
        f"sector_non_null={int(sector['sector'].notna().sum()) if 'sector' in sector else 0}, "
        f"shares_col={shares_col}, market_cap_col={market_cap_col}"
    )
    return universe, sector, caps


def normalize_jquants_statements(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["disclosed_date", "code"])
    _cell5_log(f"fins_summary normalize start raw_rows={len(raw)} raw_columns={list(raw.columns)}")
    out = pd.DataFrame(
        {
            "disclosed_date": pd.to_datetime(_series(raw, "DiscDate", "DisclosedDate", "DisclosureDate", "disc_date", "disclosed_date", "disclosure_date"), errors="coerce"),
            "disclosed_time": _series(raw, "DiscTime", "disclosed_time", "disc_time", default=pd.NA),
            "disc_no": _series(raw, "DiscNo", "disc_no", default=pd.NA),
            "doc_type": _series(raw, "DocType", "doc_type", default=pd.NA),
            "period_type": _series(raw, "CurPerType", "period_type", "cur_per_type", default=pd.NA),
            "code": _series(raw, "LocalCode", "Code", "IssueCode", "local_code", "code", "issue_code").map(normalize_code),
            "total_revenue": pd.to_numeric(_series(raw, "Sales", "NetSales", "net_sales", "Revenue", "revenue", "sales", "OperatingRevenue", "operating_revenue"), errors="coerce"),
            "operating_income": pd.to_numeric(_series(raw, "OP", "OperatingProfit", "operating_profit", "OperatingIncome", "operating_income", "BusinessProfit", "business_profit"), errors="coerce"),
            "ordinary_income": pd.to_numeric(_series(raw, "OdP", "OrdinaryProfit", "ordinary_profit"), errors="coerce"),
            "net_income": pd.to_numeric(_series(raw, "NP", "Profit", "profit", "NetIncome", "net_income", "ProfitAttributableToOwnersOfParent", "profit_attributable_to_owners_of_parent"), errors="coerce"),
            "eps": pd.to_numeric(_series(raw, "EPS", "eps"), errors="coerce"),
            "bps": pd.to_numeric(_series(raw, "BPS", "bps"), errors="coerce"),
            "total_assets": pd.to_numeric(_series(raw, "TA", "TotalAssets", "total_assets"), errors="coerce"),
            "stockholders_equity": pd.to_numeric(_series(raw, "Eq", "Equity", "equity", "NetAssets", "net_assets"), errors="coerce"),
            "equity_ratio": pd.to_numeric(_series(raw, "EqAR", "EquityRatio", "equity_ratio"), errors="coerce"),
            "cash_and_cash_equivalents": pd.to_numeric(_series(raw, "CashEq", "CashAndEquivalents", "cash_and_equivalents"), errors="coerce"),
            "operating_cash_flow": pd.to_numeric(_series(raw, "CFO", "CashFlowsFromOperatingActivities", "cash_flows_from_operating_activities"), errors="coerce"),
            "investing_cash_flow": pd.to_numeric(_series(raw, "CFI", "CashFlowsFromInvestingActivities", "cash_flows_from_investing_activities"), errors="coerce"),
            "financing_cash_flow": pd.to_numeric(_series(raw, "CFF", "CashFlowsFromFinancingActivities", "cash_flows_from_financing_activities"), errors="coerce"),
            "free_cash_flow": pd.to_numeric(_series(raw, "FreeCashFlow", "free_cash_flow"), errors="coerce"),
            "shares_outstanding": pd.to_numeric(_series(raw, "ShOutFY", "SharesOutstanding", "shares_outstanding", "IssuedShares", "issued_shares"), errors="coerce"),
            "treasury_shares": pd.to_numeric(_series(raw, "TrShFY", "TreasuryShares", "treasury_shares"), errors="coerce"),
            "avg_shares": pd.to_numeric(_series(raw, "AvgSh", "AverageShares", "avg_shares"), errors="coerce"),
            "forecast_sales": pd.to_numeric(_series(raw, "FSales", "forecast_sales"), errors="coerce"),
            "forecast_operating_income": pd.to_numeric(_series(raw, "FOP", "forecast_operating_income"), errors="coerce"),
            "forecast_ordinary_income": pd.to_numeric(_series(raw, "FOdP", "forecast_ordinary_income"), errors="coerce"),
            "forecast_net_income": pd.to_numeric(_series(raw, "FNP", "forecast_net_income"), errors="coerce"),
            "forecast_eps": pd.to_numeric(_series(raw, "FEPS", "forecast_eps"), errors="coerce"),
        }
    )
    missing_fcf = out["free_cash_flow"].isna()
    out.loc[missing_fcf, "free_cash_flow"] = (
        pd.to_numeric(out.loc[missing_fcf, "operating_cash_flow"], errors="coerce")
        + pd.to_numeric(out.loc[missing_fcf, "investing_cash_flow"], errors="coerce")
    )
    out["gross_profit"] = np.nan
    out["ebit"] = out["operating_income"]
    out["ebitda"] = np.nan
    out["total_debt"] = np.nan
    out = out.dropna(subset=["disclosed_date", "code"])
    _cell5_log(
        "fins_summary normalize done "
        f"rows={len(out)} revenue_non_null={int(out['total_revenue'].notna().sum())} "
        f"equity_non_null={int(out['stockholders_equity'].notna().sum())} "
        f"shares_non_null={int(out['shares_outstanding'].notna().sum())}"
    )
    return out


def normalize_jquants_margin(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "code", "margin_buy_balance", "margin_sell_balance"])
    _cell5_log(f"margin normalize start raw_rows={len(raw)} raw_columns={list(raw.columns)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(_series(raw, "Date", "AppDate", "ApplicationDate", "application_date", "date"), errors="coerce"),
            "published_date": pd.to_datetime(_series(raw, "PubDate", "PublishedDate", "published_date"), errors="coerce"),
            "code": _series(raw, "Code", "code", "LocalCode", "local_code").map(normalize_code),
            "margin_buy_balance": pd.to_numeric(_series(raw, "LongVol", "LongOut", "LongMarginOutstanding", "long_margin_outstanding", "margin_buy_balance"), errors="coerce"),
            "margin_sell_balance": pd.to_numeric(_series(raw, "ShrtVol", "ShrtOut", "ShortMarginOutstanding", "short_margin_outstanding", "margin_sell_balance"), errors="coerce"),
            "margin_buy_change": pd.to_numeric(_series(raw, "LongOutChg", "LongVolChg", "DailyChangeLongMarginOutstanding", "daily_change_long_margin_outstanding", "margin_buy_change"), errors="coerce"),
            "margin_sell_change": pd.to_numeric(_series(raw, "ShrtOutChg", "ShrtVolChg", "DailyChangeShortMarginOutstanding", "daily_change_short_margin_outstanding", "margin_sell_change"), errors="coerce"),
            "margin_buy_negotiable": pd.to_numeric(_series(raw, "LongNegVol", "long_neg_vol"), errors="coerce"),
            "margin_sell_negotiable": pd.to_numeric(_series(raw, "ShrtNegVol", "short_neg_vol"), errors="coerce"),
            "margin_buy_standardized": pd.to_numeric(_series(raw, "LongStdVol", "long_std_vol"), errors="coerce"),
            "margin_sell_standardized": pd.to_numeric(_series(raw, "ShrtStdVol", "short_std_vol"), errors="coerce"),
            "short_long_ratio": pd.to_numeric(_series(raw, "SLRatio", "short_long_ratio"), errors="coerce"),
            "issue_type": _series(raw, "IssType", "issue_type", default=pd.NA),
        }
    )
    out = out.dropna(subset=["date", "code"])
    _cell5_log(
        "margin normalize done "
        f"rows={len(out)} buy_non_null={int(out['margin_buy_balance'].notna().sum())} "
        f"sell_non_null={int(out['margin_sell_balance'].notna().sum())}"
    )
    return out


def normalize_jquants_index(raw: pd.DataFrame, index_code: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "index_code", "close"])
    log_cache_value_diagnostics(f"{index_code} raw-api", "index_prices", raw)
    open_col = _first_existing_column(raw, "Open", "open", "O", "o")
    high_col = _first_existing_column(raw, "High", "high", "H", "h")
    low_col = _first_existing_column(raw, "Low", "low", "L", "l")
    close_col = _first_existing_column(raw, "Close", "close", "C", "c")
    _cell5_log(
        f"{index_code} column mapping open={open_col}, high={high_col}, low={low_col}, close={close_col}"
    )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(_series_by_candidates(raw, "Date", "date"), errors="coerce"),
            "index_code": str(index_code).upper(),
            "open": pd.to_numeric(raw[open_col] if open_col else np.nan, errors="coerce"),
            "high": pd.to_numeric(raw[high_col] if high_col else np.nan, errors="coerce"),
            "low": pd.to_numeric(raw[low_col] if low_col else np.nan, errors="coerce"),
            "close": pd.to_numeric(raw[close_col] if close_col else np.nan, errors="coerce"),
        }
    ).dropna(subset=["date", "index_code"])
    log_cache_value_diagnostics(f"{index_code} normalized", "index_prices", out)
    return out


def normalize_jquants_topix(raw: pd.DataFrame) -> pd.DataFrame:
    return normalize_jquants_index(raw, "TOPIX")


def load_or_download_index_code(
    client: JQuantsApiClient,
    data_path: Path,
    *,
    index_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    save_format: str,
) -> pd.DataFrame:
    month_labels = _months_for_params({"from": start, "to": end})
    existing_main_path = _cache_file_for(data_path, "index_prices")
    existing_main = _read_cache_table(existing_main_path) if existing_main_path is not None and existing_main_path.exists() else None
    frames: list[pd.DataFrame] = []
    for month in month_labels:
        cached = _read_index_month_cache(data_path, month, index_code)
        if cached is not None and not cached.empty:
            frames.append(cached)
            continue
        if existing_main is not None and not existing_main.empty:
            month_from_main = _slice_index_month_from_main(existing_main, month, index_code)
            if not month_from_main.empty:
                _write_index_month_cache(month_from_main, data_path, month, index_code, save_format)
                frames.append(month_from_main)
                continue
        month_start = pd.Timestamp(f"{month[:4]}-{month[4:]}-01")
        month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
        raw = client.get_dataset(
            "indices",
            {"code": str(index_code), "from": _date_compact(month_start), "to": _date_compact(month_end)},
            "indices",
        )
        normalized = normalize_jquants_index(raw, str(index_code).upper())
        _write_index_month_cache(normalized, data_path, month, index_code, save_format)
        frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=["date", "index_code", "open", "high", "low", "close"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=[c for c in ["date", "index_code"] if c in out.columns], keep="last")
    return out.sort_values(["index_code", "date"]).reset_index(drop=True)


def build_market_cap_from_prices_and_shares(prices: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    if caps.empty:
        raise RuntimeError("Cannot build market_cap cache: listed master produced no cap/source rows.")
    caps = caps.copy()
    if "market_cap" not in caps.columns:
        caps["market_cap"] = np.nan
    if "shares_outstanding" not in caps.columns:
        caps["shares_outstanding"] = np.nan
    existing_numeric = pd.to_numeric(caps["market_cap"], errors="coerce")
    existing_positive = existing_numeric.gt(0)
    if existing_positive.any():
        out = caps.copy()
        out["market_cap"] = existing_numeric
        out["market_cap_source"] = "raw_market_cap"
        _cell5_log(
            "build_market_cap_from_prices_and_shares: using raw market_cap column. "
            f"input_rows={len(caps)}, positive_market_cap_rows={int(existing_positive.sum())}"
        )
        return out[["asof_date", "code", "market_cap", "shares_outstanding", "market_cap_source"]]
    shares = caps.dropna(subset=["shares_outstanding"]).copy()
    if shares.empty:
        raise RuntimeError(
            "Cannot build market_cap cache: no raw market_cap values and no shares_outstanding column/value detected "
            "in /equities/master. Provide a point-in-time market_cap cache or enable allow_missing_market_cap debug fallback."
        )
    daily = prices.copy()
    daily["asof_date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["code"] = daily["code"].map(normalize_code)
    shares["asof_date"] = pd.to_datetime(shares["asof_date"], errors="coerce").dt.normalize()
    shares["code"] = shares["code"].map(normalize_code)
    close_col = "adjustment_close" if "adjustment_close" in daily.columns and daily["adjustment_close"].notna().any() else "close"
    daily = daily[["asof_date", "code", close_col]].rename(columns={close_col: "close_for_mcap"})
    daily["close_for_mcap"] = pd.to_numeric(daily["close_for_mcap"], errors="coerce")
    daily = daily.dropna(subset=["asof_date", "code", "close_for_mcap"])
    if daily.empty:
        raise RuntimeError(f"Cannot build market_cap cache: no usable {close_col} price rows for snapshot dates.")
    shares = shares.dropna(subset=["asof_date", "code"])
    left = shares.sort_values(["asof_date", "code"], kind="mergesort").reset_index(drop=True)
    right = daily.sort_values(["asof_date", "code"], kind="mergesort").reset_index(drop=True)
    try:
        merged = pd.merge_asof(
            left,
            right,
            on="asof_date",
            by="code",
            direction="backward",
            allow_exact_matches=True,
        )
    except Exception as exc:
        _cell5_log(
            "market_cap vectorized merge_asof failed; falling back to grouped merge_asof "
            f"error={exc!r}"
        )
        parts = []
        for idx, (code, left_group) in enumerate(left.groupby("code", sort=False), start=1):
            right_group = right.loc[right["code"].eq(code), ["asof_date", "close_for_mcap"]].sort_values("asof_date")
            if right_group.empty:
                tmp = left_group.copy()
                tmp["close_for_mcap"] = np.nan
            else:
                tmp = pd.merge_asof(
                    left_group.sort_values("asof_date"),
                    right_group,
                    on="asof_date",
                    direction="backward",
                    allow_exact_matches=True,
                )
                tmp["code"] = code
            parts.append(tmp)
            if idx % 500 == 0:
                _cell5_log(f"market_cap grouped merge_asof {idx}/{left['code'].nunique()} codes done")
        merged = pd.concat(parts, ignore_index=True) if parts else left.copy()
    merged["market_cap"] = pd.to_numeric(merged["shares_outstanding"], errors="coerce") * pd.to_numeric(merged["close_for_mcap"], errors="coerce")
    numeric = pd.to_numeric(merged["market_cap"], errors="coerce")
    if numeric.notna().sum() == 0:
        raise RuntimeError(
            "market_cap computed from shares_outstanding and close price, but all values are NaN. "
            f"share_rows={len(shares)}, price_rows={len(daily)}, close_col={close_col}"
        )
    if numeric.gt(0).sum() == 0:
        raise RuntimeError(
            "market_cap computed from shares_outstanding and close price, but no positive values were produced. "
            f"share_rows={len(shares)}, price_rows={len(daily)}, close_col={close_col}"
        )
    merged["market_cap_source"] = f"shares_outstanding_x_{close_col}"
    _cell5_log(
        "market_cap build from shares done "
        f"price_rows={len(prices)}, share_rows={len(shares)}, output_rows={len(merged)}, "
        f"numeric_market_cap_rows={int(numeric.notna().sum())}, positive_market_cap_rows={int(numeric.gt(0).sum())}, close_col={close_col}"
    )
    return merged[["asof_date", "code", "market_cap", "shares_outstanding", "market_cap_source"]]


def build_market_cap_from_financial_shares(
    prices: pd.DataFrame,
    snapshots: pd.DataFrame,
    financials: pd.DataFrame,
) -> pd.DataFrame:
    """Build point-in-time market cap from J-Quants fins/summary share fields.

    ``/equities/master`` in V2 does not contain shares outstanding or market cap.
    ``/fins/summary`` does expose ShOutFY/AvgSh on the disclosed date.  For each
    universe snapshot this function takes the latest disclosed shares on or
    before that snapshot, then multiplies by the latest close on or before the
    same snapshot.  This keeps the cache point-in-time and avoids the debug
    close*volume proxy when financial share data is available.
    """

    if snapshots.empty:
        raise RuntimeError("Cannot build market_cap from financial shares: no snapshot rows.")
    if financials.empty:
        raise RuntimeError("Cannot build market_cap from financial shares: financial_summary cache is empty.")
    if "shares_outstanding" not in financials.columns and "avg_shares" not in financials.columns:
        raise RuntimeError("Cannot build market_cap from financial shares: no shares_outstanding/avg_shares column.")

    left = snapshots.loc[:, ["asof_date", "code"]].copy()
    left["asof_date"] = pd.to_datetime(left["asof_date"], errors="coerce").dt.normalize()
    left["code"] = left["code"].map(normalize_code)
    left = left.dropna(subset=["asof_date", "code"]).drop_duplicates(["asof_date", "code"])

    shares = financials.copy()
    shares["disclosed_date"] = pd.to_datetime(shares["disclosed_date"], errors="coerce").dt.normalize()
    shares["code"] = shares["code"].map(normalize_code)
    share_value = pd.to_numeric(shares.get("shares_outstanding", np.nan), errors="coerce")
    if "avg_shares" in shares.columns:
        share_value = share_value.fillna(pd.to_numeric(shares["avg_shares"], errors="coerce"))
    shares["shares_outstanding"] = share_value
    shares = shares.dropna(subset=["disclosed_date", "code", "shares_outstanding"])
    shares = shares.loc[shares["shares_outstanding"].gt(0), ["disclosed_date", "code", "shares_outstanding"]]
    if shares.empty:
        raise RuntimeError("Cannot build market_cap from financial shares: no positive disclosed shares values.")

    _cell5_log(
        "market_cap financial shares matching start "
        f"snapshot_rows={len(left)} financial_rows={len(financials)} positive_share_rows={len(shares)}"
    )
    matched = pd.merge_asof(
        left.sort_values(["asof_date", "code"], kind="mergesort"),
        shares.sort_values(["disclosed_date", "code"], kind="mergesort"),
        left_on="asof_date",
        right_on="disclosed_date",
        by="code",
        direction="backward",
        allow_exact_matches=True,
    )
    matched = matched.drop(columns=[c for c in ["disclosed_date"] if c in matched.columns])
    matched = matched.dropna(subset=["shares_outstanding"])
    if matched.empty:
        raise RuntimeError("Cannot build market_cap from financial shares: no snapshot matched to prior disclosed shares.")
    out = build_market_cap_from_prices_and_shares(prices, matched)
    out["market_cap_source"] = out["market_cap_source"].astype(str).str.replace("shares_outstanding", "fins_summary_shares", regex=False)
    _cell5_log(
        "market_cap build from financial shares done "
        f"matched_share_rows={len(matched)} output_rows={len(out)}"
    )
    return out


def _index_month_cache_path(data_dir: Path, month: str, index_code: str, save_format: str = "parquet") -> Path:
    suffix = ".parquet" if save_format == "parquet" else ".csv"
    return data_dir / "index_prices" / f"{str(index_code).lower()}_{month}{suffix}"


def _read_index_month_cache(data_dir: Path, month: str, index_code: str) -> pd.DataFrame | None:
    for suffix in (".parquet", ".csv"):
        path = data_dir / "index_prices" / f"{str(index_code).lower()}_{month}{suffix}"
        if path.exists():
            frame = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path, dtype=str, low_memory=False)
            if not frame.empty:
                return frame
    return None


def _write_index_month_cache(frame: pd.DataFrame, data_dir: Path, month: str, index_code: str, save_format: str) -> Path:
    path = _index_month_cache_path(data_dir, month, index_code, save_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            path = _index_month_cache_path(data_dir, month, index_code, "csv")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _slice_index_month_from_main(frame: pd.DataFrame, month: str, index_code: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["index_code"] = work.get("index_code", pd.Series(index=work.index, dtype="object")).astype(str)
    start = pd.Timestamp(f"{month[:4]}-{month[4:]}-01")
    end = (start + pd.offsets.MonthEnd(0)).normalize()
    out = work.loc[work["date"].between(start, end, inclusive="both") & work["index_code"].astype(str).eq(str(index_code))].copy()
    return out.reset_index(drop=True)


def load_or_download_topix_monthly_shards(
    client: JQuantsApiClient,
    data_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    save_format: str,
) -> pd.DataFrame:
    assert_full_month_range(start, end, context="topix monthly shard cache")
    months = _months_for_params({"from": start, "to": end})
    existing_main_path = _cache_file_for(data_path, "index_prices")
    existing_main = _read_cache_table(existing_main_path) if existing_main_path is not None and existing_main_path.exists() else None
    frames: list[pd.DataFrame] = []
    for month in months:
        cached = _read_index_month_cache(data_path, month, "TOPIX")
        if cached is not None and not cached.empty:
            _cell5_log(f"topix month={_month_label(month)} cache hit rows={len(cached)}")
            frames.append(cached)
            continue
        if existing_main is not None and not existing_main.empty:
            month_from_main = _slice_index_month_from_main(existing_main, month, "TOPIX")
            if not month_from_main.empty:
                _write_index_month_cache(month_from_main, data_path, month, "TOPIX", save_format)
                _cell5_log(f"topix month={_month_label(month)} main-table reuse rows={len(month_from_main)}")
                frames.append(month_from_main)
                continue
        month_start = pd.Timestamp(f"{month[:4]}-{month[4:]}-01")
        month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
        _cell5_log(f"topix month={_month_label(month)} download start")
        raw_topix = client.get_dataset("topix", {"from": _date_compact(month_start), "to": _date_compact(month_end)}, "topix")
        norm = normalize_jquants_topix(raw_topix)
        _write_index_month_cache(norm, data_path, month, "TOPIX", save_format)
        _cell5_log(f"topix month={_month_label(month)} download done rows={len(norm)}")
        frames.append(norm)
    if not frames:
        return pd.DataFrame(columns=["date", "index_code", "open", "high", "low", "close"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=[c for c in ["date", "index_code"] if c in out.columns], keep="last")
    return out.sort_values(["index_code", "date"]).reset_index(drop=True)


def build_market_cap_debug_fallback_from_prices(prices: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    """Create a positive, deterministic proxy so debug backtests can proceed.

    This is not real market capitalization.  For every code/snapshot row, it
    uses the latest available trading day on or before asof_date.  The proxy
    order is close*volume, then close-only, then a constant 1.0 placeholder.
    """

    if caps.empty:
        raise RuntimeError("Cannot build debug market_cap fallback: no universe/cap rows are available.")

    with _cell5_stage("market_cap fallback"):
        with _cell5_stage("market_cap fallback prepare"):
            snapshots = caps.loc[:, ["asof_date", "code"]].copy()
            snapshots["asof_date"] = pd.to_datetime(snapshots["asof_date"], errors="coerce").dt.normalize()
            snapshots["code"] = snapshots["code"].map(normalize_code).astype("string")
            snapshots = snapshots.dropna(subset=["asof_date", "code"])
            if snapshots.empty:
                raise RuntimeError("Cannot build debug market_cap fallback: no valid code/asof_date rows in cap source.")

            if prices.empty:
                _cell5_log("market_cap fallback prices empty; using constant proxy")
                out = snapshots.copy()
                out["market_cap"] = 1.0
                out["shares_outstanding"] = np.nan
                out["market_cap_source"] = "debug_constant_1_missing_prices"
                return out[["asof_date", "code", "market_cap", "shares_outstanding", "market_cap_source"]]

            required_price_cols = [col for col in ["code", "date", "close", "volume"] if col in prices.columns]
            daily = prices.loc[:, required_price_cols].copy()
            if "volume" not in daily.columns:
                daily["volume"] = np.nan
            daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
            daily["code"] = daily["code"].map(normalize_code).astype("string")
            daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
            daily["volume"] = pd.to_numeric(daily["volume"], errors="coerce")

            code_set = set(snapshots["code"].dropna().astype(str).unique())
            min_date = snapshots["asof_date"].min() - pd.Timedelta(days=40)
            max_date = snapshots["asof_date"].max()
            daily = daily.loc[
                daily["code"].astype(str).isin(code_set)
                & daily["date"].between(min_date, max_date, inclusive="both")
            ].dropna(subset=["date", "code"])

            cap_date_min = snapshots["asof_date"].min()
            cap_date_max = snapshots["asof_date"].max()
            price_date_min = daily["date"].min() if not daily.empty else pd.NaT
            price_date_max = daily["date"].max() if not daily.empty else pd.NaT
            _cell5_log(
                "market_cap fallback inputs "
                f"raw_caps_rows={len(caps)} prices_rows={len(prices)} filtered_prices_rows={len(daily)} "
                f"raw_caps_date_range=({cap_date_min}, {cap_date_max}) prices_date_range=({price_date_min}, {price_date_max}) "
                f"raw_caps_unique_codes={snapshots['code'].nunique()} prices_unique_codes={daily['code'].nunique()} "
                f"raw_caps_code_dtype={snapshots['code'].dtype} prices_code_dtype={daily['code'].dtype} "
                f"raw_caps_asof_dtype={snapshots['asof_date'].dtype} prices_date_dtype={daily['date'].dtype}"
            )

        with _cell5_stage("market_cap fallback sorting"):
            # pandas.merge_asof(..., by="code") is picky: the primary "on" key
            # still needs to be globally monotonic. Sorting by date first keeps
            # the vectorized path stable and avoids the slow grouped fallback.
            left = snapshots.sort_values(["asof_date", "code"], kind="mergesort").reset_index(drop=True)
            right = daily.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

        with _cell5_stage("market_cap fallback merge_asof"):
            try:
                merged = pd.merge_asof(
                    left,
                    right,
                    by="code",
                    left_on="asof_date",
                    right_on="date",
                    direction="backward",
                    allow_exact_matches=True,
                )
            except Exception as exc:
                _cell5_log(f"market_cap fallback vectorized merge_asof failed; falling back to grouped merge_asof error={exc!r}")
                parts = []
                for idx, (code, left_group) in enumerate(left.groupby("code", sort=False), start=1):
                    right_group = right.loc[right["code"].eq(code), ["date", "close", "volume"]].sort_values("date")
                    if right_group.empty:
                        tmp = left_group.copy()
                        tmp["date"] = pd.NaT
                        tmp["close"] = np.nan
                        tmp["volume"] = np.nan
                    else:
                        tmp = pd.merge_asof(
                            left_group.sort_values("asof_date"),
                            right_group,
                            left_on="asof_date",
                            right_on="date",
                            direction="backward",
                            allow_exact_matches=True,
                        )
                        tmp["code"] = code
                    parts.append(tmp)
                    if idx % 500 == 0:
                        _cell5_log(f"market_cap fallback grouped merge_asof {idx}/{left['code'].nunique()} codes done")
                merged = pd.concat(parts, ignore_index=True) if parts else left.copy()

        with _cell5_stage("market_cap fallback proxy generation"):
            merged["close"] = pd.to_numeric(merged.get("close", np.nan), errors="coerce")
            merged["volume"] = pd.to_numeric(merged.get("volume", np.nan), errors="coerce")

            close_volume_proxy = merged["close"] * merged["volume"]
            close_volume_positive = close_volume_proxy.gt(0)
            close_positive = merged["close"].gt(0)
            volume_positive = merged["volume"].gt(0)
            close_nonnull = merged["close"].notna()
            volume_nonnull = merged["volume"].notna()
            _cell5_log(
                "market_cap fallback merge stats "
                f"merged_rows={len(merged)} matched_close={int(close_nonnull.sum())} "
                f"matched_volume={int(volume_nonnull.sum())} positive_close={int(close_positive.sum())} "
                f"positive_volume={int(volume_positive.sum())} positive_close_x_volume={int(close_volume_positive.sum())}"
            )
            _cell5_log("market_cap fallback raw_caps sample:\n" + snapshots[["code", "asof_date"]].head(10).to_string(index=False))
            sample_cols = [col for col in ["code", "asof_date", "date", "close", "volume"] if col in merged.columns]
            _cell5_log("market_cap fallback matched sample:\n" + merged[sample_cols].head(10).to_string(index=False))

            out = merged[["asof_date", "code"]].copy()
            out["market_cap"] = 1.0
            out["market_cap_source"] = "debug_constant_1_missing_close"
            out.loc[close_positive, "market_cap"] = merged.loc[close_positive, "close"]
            out.loc[close_positive, "market_cap_source"] = "debug_weak_proxy_close_only"
            out.loc[close_volume_positive, "market_cap"] = close_volume_proxy.loc[close_volume_positive]
            out.loc[close_volume_positive, "market_cap_source"] = "debug_liquidity_proxy_close_x_volume"
            out["shares_outstanding"] = np.nan
            numeric = pd.to_numeric(out["market_cap"], errors="coerce")
            if numeric.notna().sum() == 0 or numeric.gt(0).sum() == 0:
                out["market_cap"] = 1.0
                out["market_cap_source"] = "debug_constant_1_final_guard"
            source_counts = out["market_cap_source"].value_counts(dropna=False).to_dict()
            _cell5_log(
                "market_cap fallback proxy generation done "
                f"source_counts={source_counts}"
            )
            return out[["asof_date", "code", "market_cap", "shares_outstanding", "market_cap_source"]]


def download_jquants_backtest_cache(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    data_dir: str | Path = "data",
    api_config: JQuantsApiConfig | dict[str, Any] | None = None,
    codes: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Download/cache the local tables needed by JpxHistoricalProvider.

    This is the function a master notebook should call after the user selects a
    backtest date range. It fetches API-available data, normalizes schemas, and
    writes provider-ready cache files. Market cap is generated only when the API
    response contains listed shares; otherwise users should supply a separate
    point-in-time market_cap cache or run screening with market_cap.mode="none".
    """

    cfg = api_config if isinstance(api_config, JQuantsApiConfig) else JQuantsApiConfig(**(api_config or {}))
    client = JQuantsApiClient(cfg)
    data_path = Path(data_dir)
    if cfg.force_rebuild_cache:
        rebuild_corrupt_cache_tables(data_path)
    user_start_ts = pd.Timestamp(start).normalize()
    user_end_ts = pd.Timestamp(end).normalize()
    start_ts, end_ts = normalize_full_month_backtest_range(user_start_ts, user_end_ts)
    required_history_days = cfg.effective_history_days()
    raw_fetch_start = start_ts - pd.Timedelta(days=int(required_history_days))
    fetch_start = _floor_month_start(raw_fetch_start)
    assert_full_month_range(fetch_start, end_ts, context="daily_quotes required cache range")
    user_requested_range = {"from": _date_compact(user_start_ts), "to": _date_compact(user_end_ts)}
    effective_backtest_range = {"from": _date_compact(start_ts), "to": _date_compact(end_ts)}
    effective_request_range = {"from": _date_compact(fetch_start), "to": _date_compact(end_ts)}
    _cell5_log(
        "required data range calculated "
        f"user_requested_range={user_requested_range}, "
        f"effective_backtest_range={effective_backtest_range}, "
        f"effective_request_range={effective_request_range}, "
        f"required_history_days={int(required_history_days)}, "
        f"bulk_months={bulk_month_labels(start_ts, end_ts)}"
    )

    outputs: dict[str, Path] = {}

    listed_universe: list[pd.DataFrame] = []
    listed_sector: list[pd.DataFrame] = []
    listed_caps: list[pd.DataFrame] = []
    universe_reusable, universe_path = _existing_table_reusable(
        data_path, "universe", date_col="asof_date", required_start=start_ts, required_end=end_ts, asof_date=start_ts
    )
    sector_reusable, sector_path = _existing_table_reusable(
        data_path, "sector", date_col="asof_date", required_start=start_ts, required_end=end_ts, asof_date=start_ts
    )
    market_cap_reusable, market_cap_path = _existing_table_reusable(
        data_path, "market_cap", date_col="asof_date", required_start=start_ts, required_end=end_ts, asof_date=start_ts
    )
    prices_reusable, prices_path = _existing_table_reusable(
        data_path, "prices", date_col="date", required_start=fetch_start, required_end=end_ts, asof_date=start_ts
    )
    financials_reusable, financials_path = _existing_table_reusable(
        data_path, "financial_summary", date_col="disclosed_date", required_start=fetch_start, required_end=end_ts
    )
    index_reusable, index_path = _existing_table_reusable(
        data_path, "index_prices", date_col="date", required_start=fetch_start, required_end=end_ts, asof_date=start_ts
    )
    margin_reusable, margin_path = _existing_table_reusable(
        data_path, "margin", date_col="date", required_start=fetch_start, required_end=end_ts
    )

    if prices_reusable and not _monthly_price_shards_present(data_path, fetch_start, end_ts, cfg.save_format):
        _cell5_log("prices main table spans requested dates but monthly shards are incomplete; forcing shard assemble/write")
        prices_reusable = False
    if cfg.include_topix_index and index_reusable and not _monthly_topix_shards_present(data_path, fetch_start, end_ts, cfg.save_format):
        _cell5_log("index_prices main table spans requested dates but TOPIX monthly shards are incomplete; forcing shard assemble/write")
        index_reusable = False

    if cfg.universe_from_listed_info and universe_reusable and sector_reusable and market_cap_reusable:
        _cell5_log("listed_info/universe/sector/market_cap main tables reusable; skipping snapshot rebuild")
    elif cfg.universe_from_listed_info:
        snapshots = _month_end_snapshots(start_ts, end_ts)
        _cell5_log(f"listed_info snapshots start count={len(snapshots)}")
        listed_start = time.perf_counter()
        for idx, snapshot in enumerate(snapshots, start=1):
            uni, sector, caps = load_or_download_listed_snapshot(client, data_path, snapshot, cfg.save_format)
            listed_universe.append(uni)
            listed_sector.append(sector)
            listed_caps.append(caps)
            if idx == len(snapshots) or idx % 3 == 0:
                _cell5_log(f"listed_info snapshots {idx}/{len(snapshots)} done elapsed={time.perf_counter() - listed_start:.1f}s")
        universe_cache = pd.concat(listed_universe, ignore_index=True)
        sector_cache = pd.concat(listed_sector, ignore_index=True)
        validate_cache_table_contract(universe_cache, "universe", min_rows=1, asof_date=start_ts)
        validate_cache_table_contract(sector_cache, "sector", min_rows=1, asof_date=start_ts)
        outputs["universe"] = _append_existing(data_path, "universe", universe_cache, ["asof_date", "code"], cfg.save_format)
        outputs["sector"] = _append_existing(data_path, "sector", sector_cache, ["asof_date", "code"], cfg.save_format)
    else:
        if universe_path is not None:
            outputs["universe"] = universe_path
        if sector_path is not None:
            outputs["sector"] = sector_path

    if codes is None and listed_universe:
        uni_all = pd.concat(listed_universe, ignore_index=True)
        codes = uni_all.loc[uni_all["in_universe"].eq(True), "code"].dropna().drop_duplicates().tolist()
    codes = [str(c) for c in (codes or []) if normalize_code(c) is not None]

    if prices_reusable and prices_path is not None:
        _cell5_log(f"prices main table reusable path={prices_path}; skipping shard assemble/write")
        prices = _read_cache_table(prices_path)
        outputs["prices"] = prices_path
    else:
        with _cell5_stage("daily_quotes cache coverage/download"):
            prices = load_or_download_price_shards(client, data_path, fetch_start, end_ts, cfg.save_format)
        _cell5_log(f"daily_quotes local cache assembled rows={len(prices)} columns={list(prices.columns)}")
        validate_cache_table_contract(prices, "prices", min_rows=1, asof_date=start_ts)
        outputs["prices"] = _append_existing(data_path, "prices", prices, ["date", "code"], cfg.save_format)
    if codes:
        prices = prices.loc[prices["code"].isin(set(codes))]

    financials = pd.DataFrame(columns=["disclosed_date", "code"])
    if cfg.include_financials:
        existing_financials = _read_cache_table(financials_path) if financials_path is not None and financials_path.exists() else pd.DataFrame(columns=["disclosed_date", "code"])
        if financials_reusable and financials_path is not None:
            _cell5_log(f"financial_summary main table reusable path={financials_path}; skipping shard coverage scan")
            financials = existing_financials
            outputs["financial_summary"] = financials_path
        else:
            try:
                if cfg.use_bulk_download:
                    with _cell5_stage("fins_summary monthly bulk cache coverage/download"):
                        financials = client.build_fins_summary_cache_via_bulk(
                            {"from": _date_compact(fetch_start), "to": _date_compact(end_ts)},
                            data_path,
                            save_format=cfg.save_format,
                            codes=codes,
                            existing_main=existing_financials,
                        )
                else:
                    with _cell5_stage("fins_summary date-shard cache coverage/download"):
                        financials = load_or_download_financial_summary_shards(
                            client,
                            data_path,
                            fetch_start,
                            end_ts,
                            cfg.save_format,
                        )
                _cell5_log(
                    "fins_summary local cache assembled "
                    f"rows={len(financials)} columns={list(financials.columns)}"
                )
            except Exception as exc:
                _cell5_log(
                    "WARNING: fins_summary cache could not be built from V2 date shards. "
                    "Quality/value factors may become neutral placeholders until financial data is available. "
                    f"error={exc!r}"
                )
                financials = pd.DataFrame(columns=["disclosed_date", "code"])
            outputs["financial_summary"] = _append_existing(data_path, "financial_summary", financials, ["disclosed_date", "code"], cfg.save_format)
        if codes and not financials.empty:
            financials = financials.loc[financials["code"].isin(set(codes))]

    if market_cap_reusable and market_cap_path is not None:
        _cell5_log(f"market_cap main table reusable path={market_cap_path}; skipping rebuild")
        outputs["market_cap"] = market_cap_path
    elif listed_caps:
        raw_caps = pd.concat(listed_caps, ignore_index=True)
        _cell5_log(f"market_cap build start raw_caps_rows={len(raw_caps)} prices_rows={len(prices)}")
        try:
            market_caps = build_market_cap_from_prices_and_shares(prices, raw_caps)
        except Exception as exc:
            _cell5_log(
                "market_cap direct listed-master build failed; trying financial share fields. "
                f"reason={exc!r}"
            )
            try:
                market_caps = build_market_cap_from_financial_shares(prices, raw_caps, financials)
            except Exception as fin_exc:
                if not cfg.allow_missing_market_cap:
                    raise
                _cell5_log(
                    "WARNING: real market_cap cache could not be built; using debug fallback because "
                    f"allow_missing_market_cap=True. listed_reason={exc!r}, financial_reason={fin_exc!r}"
                )
                market_caps = build_market_cap_debug_fallback_from_prices(prices, raw_caps)
                _cell5_log(
                    "WARNING: debug market_cap fallback is for diagnostics only and will fail PIT cache validation. "
                    "Disable allow_missing_market_cap or provide a real shares-based market_cap source for formal backtests."
                )
        validate_cache_table_contract(market_caps, "market_cap", min_rows=1, asof_date=start_ts)
        outputs["market_cap"] = _append_existing(data_path, "market_cap", market_caps, ["asof_date", "code"], cfg.save_format)

    index_frames: list[pd.DataFrame] = []
    if index_reusable and index_path is not None:
        reusable_indexes = _read_cache_table(index_path)
        existing_codes = set(reusable_indexes.get("index_code", pd.Series(dtype=str)).astype(str).str.upper().unique())
        required_codes = {str(code).upper() for code in cfg.additional_index_codes}
        if cfg.include_topix_index:
            required_codes.add("TOPIX")
        if cfg.include_nikkei_index:
            required_codes.add("NIKKEI225")
        missing_codes = sorted(code for code in required_codes if code not in existing_codes)
        if not missing_codes:
            _cell5_log(f"index_prices main table reusable path={index_path}; skipping index download")
            outputs["index_prices"] = index_path
        else:
            _cell5_log(f"index_prices reusable table missing codes={missing_codes}; downloading incremental index history")
            index_frames.append(reusable_indexes)
            for code in missing_codes:
                if code == "TOPIX":
                    topix = load_or_download_topix_monthly_shards(client, data_path, fetch_start, end_ts, cfg.save_format)
                    if not topix.empty:
                        index_frames.append(topix)
                elif code == "NIKKEI225":
                    try:
                        raw_nikkei = client.get_dataset("nikkei225", {"from": _date_compact(fetch_start), "to": _date_compact(end_ts)}, "nikkei225")
                        index_frames.append(normalize_jquants_index(raw_nikkei, "NIKKEI225"))
                    except Exception:
                        pass
                else:
                    try:
                        index_frames.append(
                            load_or_download_index_code(
                                client,
                                data_path,
                                index_code=code,
                                start=fetch_start,
                                end=end_ts,
                                save_format=cfg.save_format,
                            )
                        )
                    except Exception as exc:
                        _cell5_log(f"WARNING: additional index download failed code={code} reason={exc!r}")
            if index_frames:
                indexes = pd.concat(index_frames, ignore_index=True)
                validate_cache_table_contract(indexes, "index_prices", min_rows=1, asof_date=start_ts)
                outputs["index_prices"] = _append_existing(data_path, "index_prices", indexes, ["date", "index_code"], cfg.save_format)
    else:
        if cfg.include_topix_index:
            topix = load_or_download_topix_monthly_shards(client, data_path, fetch_start, end_ts, cfg.save_format)
            if not topix.empty:
                index_frames.append(topix)
        if cfg.include_nikkei_index:
            try:
                raw_nikkei = client.get_dataset("nikkei225", {"from": _date_compact(fetch_start), "to": _date_compact(end_ts)}, "nikkei225")
                index_frames.append(normalize_jquants_index(raw_nikkei, "NIKKEI225"))
            except Exception:
                pass
        for code in cfg.additional_index_codes:
            code_text = str(code).upper()
            if code_text in {"TOPIX", "NIKKEI225"}:
                continue
            try:
                index_frames.append(
                    load_or_download_index_code(
                        client,
                        data_path,
                        index_code=code_text,
                        start=fetch_start,
                        end=end_ts,
                        save_format=cfg.save_format,
                    )
                )
            except Exception as exc:
                _cell5_log(f"WARNING: additional index download failed code={code_text} reason={exc!r}")
        if index_frames:
            indexes = pd.concat(index_frames, ignore_index=True)
            validate_cache_table_contract(indexes, "index_prices", min_rows=1, asof_date=start_ts)
            outputs["index_prices"] = _append_existing(data_path, "index_prices", indexes, ["date", "index_code"], cfg.save_format)

    if cfg.include_margin:
        margin = pd.DataFrame(
            columns=[
                "date",
                "published_date",
                "code",
                "margin_buy_balance",
                "margin_sell_balance",
                "margin_buy_change",
                "margin_sell_change",
            ]
        )
        if margin_reusable and margin_path is not None:
            _cell5_log(f"margin main table reusable path={margin_path}; skipping date-shard coverage scan")
            margin = _read_cache_table(margin_path)
            outputs["margin"] = margin_path
        else:
            try:
                with _cell5_stage(f"{cfg.margin_dataset} date-shard cache coverage/download"):
                    margin = load_or_download_margin_shards(
                        client,
                        data_path,
                        fetch_start,
                        end_ts,
                        cfg.save_format,
                        endpoint_name=cfg.margin_dataset,
                    )
                _cell5_log(
                    "margin local cache assembled "
                    f"dataset={cfg.margin_dataset} rows={len(margin)} columns={list(margin.columns)}"
                )
            except Exception as exc:
                if not cfg.margin_optional:
                    raise
                _cell5_log(
                    "WARNING: skipping optional margin cache after J-Quants V2 date-shard request failure. "
                    f"dataset={cfg.margin_dataset!r}, error={exc!r}"
                )
            outputs["margin"] = _append_existing(data_path, "margin", margin, ["date", "code", "published_date"], cfg.save_format)
        if codes and not margin.empty:
            margin = margin.loc[margin["code"].isin(set(codes))]

    debug_dump_cache_summary(
        data_path,
        required_tables=("universe", "sector", "market_cap", "prices"),
        asof_date=start_ts,
    )
    _cell5_log("provider ready")
    return outputs


def provider_from_jquants_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    data_dir: str | Path = "data",
    api_config: JQuantsApiConfig | dict[str, Any] | None = None,
    codes: Iterable[str] | None = None,
) -> JpxHistoricalProvider:
    """Download/cache a range and return a provider wired to that cache."""

    download_jquants_backtest_cache(start, end, data_dir=data_dir, api_config=api_config, codes=codes)
    return JpxHistoricalProvider(LocalDataPaths(data_dir=data_dir))
