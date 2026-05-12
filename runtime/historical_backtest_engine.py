from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from allocation_gate import resolve_allocation_gate
from attention_factor_runtime import run_attention_factor
from behaviour_factor_runtime import run_behaviour_factor
from dual_ma_factor_runtime import run_dual_ma_factor
from factor_cache import build_factor_cache_lookup, load_factor_cache, save_factor_cache
from factor_composite import (
    WeightConfig,
    build_runtime_config,
    merge_factor_frames,
    normalize_factor_name_for_composite,
    prepare_composite_ready_frame,
    run_factor_composite_from_frames,
    run_factor_composite_from_prepared_frame,
    standardize_factor_frame_for_composite,
)
from fundamental_factor_runtime import run_quality_factor, run_value_factor
from historical_data import JpxHistoricalProvider, normalize_code
from market_risk_gating_runtime import run_market_risk_gating
from regime_indicator import RegimeState, detect_regime
from regime_weights import DEFAULT_WEIGHT_SOURCE_MODE, to_weight_config
from residual_momentum_factor_runtime import run_residual_momentum_factor
from reversal_factor_runtime import run_reversal_factor
from snapshot_cache import build_snapshot_cache_paths, load_snapshot_cache, save_snapshot_cache
from topix_pool_runtime import RunReport, ScreenRules, build_universe_minimal, run_realtime_screen


@dataclass
class BacktestConfig:
    """Historical backtest configuration with optional light short hedge."""

    signal_execution_lag_days: int = 1
    execution_price: str = "close"  # "open" or "close"
    target_count: int = 15
    candidate_pool_count: int = 50
    max_gross_exposure: float = 1.0
    min_execution_price_coverage: float = 0.8
    min_execution_price_count: int | None = None
    rebuild_targets_after_price_lookup: bool = True
    rescale_partial_target_weights: bool = False
    record_missing_execution_prices: bool = True
    enable_short_leg: bool = False
    short_gross_exposure: float = 0.10
    short_target_count: int = 2
    short_construction_mode: str = "simple_bottom_n"
    short_weighting_mode: str = "equal_weight"
    short_expansion_mode: str = "bottom_decile_only"
    short_adv_threshold: float | None = None
    short_sector_cap: float | None = None
    short_min_names: int = 0
    short_max_single_weight: float = 0.10
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    stop_loss_pct: float = 0.18
    stop_loss_check_frequency: str = "daily"  # "monthly", "weekly", or "daily"
    stop_loss_cash_penalty_pct: float = 0.25
    stop_loss_penalty_ignores_risk_gate: bool = True
    stop_loss_cooldown_rebalances: int = 3
    suppress_reversal_when_negative_mgate: bool = False
    f_penalty_when_nonpositive_mgate: float = 1.0
    output_mode: str = "full"  # "full" or "fast"
    lot_size: int = 100
    round_lots: bool = False
    allow_yfinance_fallback: bool = False
    enabled_factors: tuple[str, ...] = ("quality", "value", "residual_momentum", "dual_ma", "reversal", "attention", "behaviour")
    risk_index_code: str = "TOPIX"
    use_risk_gating: bool = True
    benchmark_index_codes: tuple[str, ...] = ("TOPIX", "NIKKEI225")
    include_sp500_benchmark: bool = True
    sp500_yfinance_ticker: str = "^GSPC"
    composite_weights: WeightConfig = field(default_factory=WeightConfig)
    composite_formula: str = "w_q*Q + w_v*V + w_m*M + w_ts*TS + w_b*B + w_t*T + w_r*REV"
    weight_source_mode: str = DEFAULT_WEIGHT_SOURCE_MODE
    regime_weight_map: dict[str, str] = field(default_factory=dict)
    regime_config: dict[str, Any] = field(default_factory=dict)
    allocation_gate_config: dict[str, Any] = field(default_factory=dict)
    factor_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    enable_factor_cache: bool = True
    factor_cache_dir: str | Path = field(default_factory=lambda: Path(__file__).resolve().parent / "factor_cache")
    enable_snapshot_cache: bool = True
    snapshot_cache_dir: str | Path = field(default_factory=lambda: Path(__file__).resolve().parent / "snapshot_cache")
    enable_composite_ready_cache: bool = True
    composite_ready_cache_dir: str | Path = field(default_factory=lambda: Path(__file__).resolve().parent / "composite_ready_cache")
    post_score_adjustments: dict[str, Any] = field(default_factory=dict)
    portfolio_constraints: dict[str, Any] = field(default_factory=lambda: {"max_names_per_sector": 5})


@dataclass
class RebalanceResult:
    rebalance_date: pd.Timestamp
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp | None
    universe: pd.DataFrame
    factor_results: dict[str, dict[str, Any]]
    composite: dict[str, Any]
    target_weights: pd.DataFrame
    risk: dict[str, Any] | None
    report: Any
    timings: dict[str, float] = field(default_factory=dict)
    factor_cache_statuses: dict[str, str] = field(default_factory=dict)
    snapshot_cache_status: str = "disabled"
    composite_ready_cache_status: str = "disabled"
    regime: dict[str, Any] = field(default_factory=dict)
    weight_resolution: dict[str, Any] = field(default_factory=dict)
    allocation_gate: dict[str, Any] = field(default_factory=dict)


def _cfg(config: BacktestConfig | dict[str, Any] | None) -> BacktestConfig:
    if config is None:
        return BacktestConfig()
    if isinstance(config, BacktestConfig):
        return config
    data = dict(config)
    if isinstance(data.get("composite_weights"), dict):
        data["composite_weights"] = WeightConfig(**data["composite_weights"])
    return BacktestConfig(**data)


def _price_loader_for_single(provider: JpxHistoricalProvider) -> Callable[[str, pd.Timestamp, pd.Timestamp, dict[str, Any]], pd.DataFrame]:
    def loader(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, config: dict[str, Any]) -> pd.DataFrame:
        normalized = ticker[:4] if str(ticker)[:4].isdigit() else str(ticker)
        frame = provider.price_frame([normalized], start_date, end_date)
        if frame.empty:
            return pd.DataFrame()
        out = frame.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return out.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    return loader


def _risk_loader(provider: JpxHistoricalProvider) -> Callable[[str, pd.Timestamp, pd.Timestamp, str | None, dict[str, Any]], pd.Series]:
    def loader(index_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp, jquants_api_key: str | None, config: dict[str, Any]) -> pd.Series:
        return provider.load_index_close(index_code, start_date, end_date, config)

    return loader


def _factor_config(base: BacktestConfig, name: str, provider: JpxHistoricalProvider, signal_date: pd.Timestamp) -> dict[str, Any]:
    cfg = dict(base.factor_configs.get(name, {}))
    cfg.setdefault("verbose", False)
    cfg["asof_date"] = signal_date
    cfg["rebalance_date"] = signal_date
    cfg["allow_yfinance_fallback"] = bool(base.allow_yfinance_fallback)
    if name in {"dual_ma"}:
        cfg["price_loader"] = provider.load_price_history
    elif name == "reversal":
        cfg["panel_price_loader"] = provider.load_price_history
        cfg["price_loader"] = _price_loader_for_single(provider)
    elif name == "attention":
        cfg["data_source"] = "jquants"
        cfg["attention_loader"] = provider.load_attention_inputs
    elif name == "behaviour":
        cfg["behaviour_loader"] = provider.load_behaviour_inputs
    elif name in {"quality", "value"}:
        cfg["mode"] = "historical_local"
        cfg["fundamental_loader"] = provider.load_fundamental_panel
    elif name == "residual_momentum":
        cfg["close_loader"] = provider.load_close_prices
        cfg["market_benchmark"] = cfg.get("market_benchmark", base.risk_index_code)
    return cfg


def _profile_log(signal_date: pd.Timestamp, stage: str, elapsed_sec: float, **fields: Any) -> None:
    extra = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {extra}" if extra else ""
    print(f"[profile] signal_date={signal_date.date()} stage={stage} elapsed_sec={elapsed_sec:.3f}{suffix}", flush=True)


def _stable_frame_digest(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "empty"
    work = frame.copy()
    if "code" in work.columns:
        work["code"] = work["code"].astype(str)
        work = work.sort_values([c for c in ["code"] if c in work.columns]).reset_index(drop=True)
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = pd.to_datetime(work[col], errors="coerce").astype("string")
        else:
            work[col] = work[col].astype("string")
    work = work.reindex(sorted(work.columns), axis=1)
    row_hash = pd.util.hash_pandas_object(work, index=False).values.tobytes()
    return hashlib.sha1(row_hash).hexdigest()


def _composite_ready_cache_paths(
    cache_dir: str | Path,
    *,
    signal_date: pd.Timestamp,
    factor_payloads: dict[str, dict[str, Any]],
    do_preprocess: bool,
) -> dict[str, Any]:
    factor_digests: dict[str, str] = {}
    for factor_name in sorted(factor_payloads):
        payload = factor_payloads.get(factor_name, {})
        minimal = payload.get("minimal") if isinstance(payload, dict) else None
        factor_digests[factor_name] = _stable_frame_digest(minimal) if isinstance(minimal, pd.DataFrame) else "missing"
    key_source = {
        "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
        "do_preprocess": bool(do_preprocess),
        "factor_digests": factor_digests,
    }
    cache_key = hashlib.sha1(json.dumps(key_source, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
    root = Path(cache_dir) / pd.Timestamp(signal_date).strftime("%Y%m")
    return {
        "cache_key": cache_key,
        "frame_path": root / f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')}_{cache_key}.parquet",
        "meta_path": root / f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')}_{cache_key}.json",
    }


def _load_composite_ready_cache(cache_paths: dict[str, Any]) -> pd.DataFrame | None:
    frame_path = Path(cache_paths["frame_path"])
    if not frame_path.exists():
        return None
    try:
        return pd.read_parquet(frame_path)
    except Exception:
        return None


def _save_composite_ready_cache(cache_paths: dict[str, Any], prepared: pd.DataFrame) -> None:
    frame_path = Path(cache_paths["frame_path"])
    meta_path = Path(cache_paths["meta_path"])
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(frame_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "cache_key": cache_paths["cache_key"],
                "created_at": pd.Timestamp.now().isoformat(),
                "frame_path": str(frame_path),
                "rows": int(len(prepared)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_factors(
    universe: pd.DataFrame,
    signal_date: pd.Timestamp,
    provider: JpxHistoricalProvider,
    config: BacktestConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, str]]:
    runners: dict[str, Callable[..., dict[str, Any]]] = {
        "quality": run_quality_factor,
        "value": run_value_factor,
        "dual_ma": run_dual_ma_factor,
        "reversal": run_reversal_factor,
        "attention": run_attention_factor,
        "behaviour": run_behaviour_factor,
        "residual_momentum": run_residual_momentum_factor,
    }
    results: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    cache_statuses: dict[str, str] = {}
    for name in config.enabled_factors:
        runner = runners.get(name)
        if runner is None:
            continue
        stage_name = "momentum" if name == "residual_momentum" else name
        started = time.perf_counter()
        factor_cfg = _factor_config(config, name, provider, signal_date)
        cache_lookup = None
        try:
            if config.enable_factor_cache:
                cache_lookup = build_factor_cache_lookup(
                    config.factor_cache_dir,
                    factor_name=name,
                    anchor_date=signal_date,
                    universe=universe,
                    factor_config=factor_cfg,
                    provider=provider,
                )
                cached = load_factor_cache(cache_lookup)
                if cached is not None:
                    results[name] = cached
                    cache_statuses[name] = "hit"
                    minimal = cached.get("minimal")
                    rows = len(minimal) if isinstance(minimal, pd.DataFrame) else 0
                    print(
                        f"[factor-cache] hit factor={name} date={signal_date.date()} "
                        f"key={cache_lookup.cache_key} rows={rows}",
                        flush=True,
                    )
                    continue
                cache_statuses[name] = "miss"
                print(
                    f"[factor-cache] miss factor={name} date={signal_date.date()} "
                    f"reason=no_matching_key",
                    flush=True,
                )
            results[name] = runner(
                universe=universe,
                rebalance_date=signal_date,
                config=factor_cfg,
            )
            if config.enable_factor_cache and cache_lookup is not None:
                before_exists = cache_lookup.exists
                save_factor_cache(cache_lookup, results[name])
                if cache_lookup.exists and not before_exists:
                    cache_statuses[name] = "saved"
                    minimal = results[name].get("minimal")
                    rows = len(minimal) if isinstance(minimal, pd.DataFrame) else 0
                    print(
                        f"[factor-cache] saved factor={name} date={signal_date.date()} "
                        f"key={cache_lookup.cache_key} rows={rows}",
                        flush=True,
                    )
        except NotImplementedError as exc:
            if config.factor_configs.get(name, {}).get("required", False):
                raise
            cache_statuses[name] = "error"
            minimal = pd.DataFrame(
                {
                    "code": universe["code"].astype(str),
                    "factor_name": name,
                    "factor_value": np.nan,
                    "signal_date": signal_date,
                    "data_end_date": pd.NaT,
                    "rebalance_date": signal_date,
                }
            )
            results[name] = {"minimal": minimal, "detail": pd.DataFrame(), "summary": {"factor_name": name, "error": str(exc)}}
        except Exception as exc:
            if config.factor_configs.get(name, {}).get("required", False):
                raise
            cache_statuses[name] = "error"
            minimal = pd.DataFrame(
                {
                    "code": universe["code"].astype(str),
                    "factor_name": name,
                    "factor_value": np.nan,
                    "signal_date": signal_date,
                    "data_end_date": pd.NaT,
                    "rebalance_date": signal_date,
                }
            )
            results[name] = {
                "minimal": minimal,
                "detail": pd.DataFrame(),
                "summary": {"factor_name": name, "error": f"{type(exc).__name__}: {exc}"},
            }
        finally:
            elapsed = time.perf_counter() - started
            timings[f"{stage_name}_sec"] = elapsed
            if name == "residual_momentum":
                timings["residual_momentum_sec"] = elapsed
            payload = results.get(name, {})
            minimal = payload.get("minimal") if isinstance(payload, dict) else None
            minimal_rows = len(minimal) if isinstance(minimal, pd.DataFrame) else 0
            _profile_log(signal_date, stage_name, elapsed, rows=minimal_rows)
    return results, timings, cache_statuses


def _factor_has_signal(payload: dict[str, Any], candidates: tuple[str, ...]) -> bool:
    minimal = payload.get("minimal") if isinstance(payload, dict) else None
    if not isinstance(minimal, pd.DataFrame) or minimal.empty:
        return False
    for col in candidates:
        if col in minimal.columns and pd.to_numeric(minimal[col], errors="coerce").notna().any():
            return True
    return False


def _effective_composite_weights(base: WeightConfig, factors: dict[str, dict[str, Any]]) -> tuple[WeightConfig, dict[str, Any]]:
    """Drop unavailable factor modules from composite weights for historical runs.

    Factor formulas are untouched.  This only prevents unavailable local data
    inputs from turning the whole composite score into NaN.
    """

    availability = {
        "quality": _factor_has_signal(factors.get("quality", {}), ("QualityScore", "quality_score", "Q", "factor_value")),
        "value": _factor_has_signal(factors.get("value", {}), ("ValueScore", "value_score", "V", "factor_value")),
        "momentum": _factor_has_signal(
            factors.get("residual_momentum", factors.get("momentum", {})),
            ("resid_ir_neutralized_score", "resid_tstat_neutralized_score", "resid_cumret_neutralized_score", "M", "factor_value"),
        ),
        "dual_ma": _factor_has_signal(factors.get("dual_ma", {}), ("dual_ma_gate", "dual_ma_signal", "TS", "factor_value")),
        "reversal": _factor_has_signal(factors.get("reversal", {}), ("reversal_factor_zscore", "reversal_zscore", "REV", "factor_value")),
        "behaviour": _factor_has_signal(factors.get("behaviour", {}), ("credit_residual_factor", "behaviour_score", "B", "factor_value")),
        "attention": _factor_has_signal(factors.get("attention", {}), ("attention_raw", "attention_factor", "attention_score", "T", "factor_value")),
    }
    adjusted = WeightConfig(
        w_q=float(base.w_q) if availability["quality"] else 0.0,
        w_v=float(base.w_v) if availability["value"] else 0.0,
        w_m=float(base.w_m) if availability["momentum"] else 0.0,
        w_ts=float(base.w_ts) if availability["dual_ma"] else 0.0,
        w_b=float(base.w_b) if availability["behaviour"] else 0.0,
        w_t=float(base.w_t) if availability["attention"] else 0.0,
        w_r=float(base.w_r) if availability["reversal"] else 0.0,
    )
    if (
        abs(adjusted.w_q)
        + abs(adjusted.w_v)
        + abs(adjusted.w_m)
        + abs(adjusted.w_ts)
        + abs(adjusted.w_b)
        + abs(adjusted.w_t)
        + abs(adjusted.w_r)
    ) < 1e-12:
        adjusted = base
    return adjusted, {"factor_available": availability, "module_available": availability, "effective_weights": asdict(adjusted)}


def _has_valid_execution_prices(
    provider: JpxHistoricalProvider,
    codes: Iterable[str] | None,
    trade_date: pd.Timestamp,
    price_col: str,
    min_coverage: float = 0.8,
    min_count: int | None = None,
) -> bool:
    code_list = [str(normalize_code(code)) for code in codes or [] if str(code)]
    if not code_list:
        prices = provider.table("prices", copy=False)
        work = prices.loc[prices["date"].eq(pd.Timestamp(trade_date).normalize())].copy()
        if work.empty:
            return False
        cols = ["adjustment_open", "open"] if price_col == "open" else ["adjustment_close", "close"]
        for col in cols:
            if col in work.columns:
                values = pd.to_numeric(work[col], errors="coerce")
                if values.gt(0).any():
                    return True
        return False
    prices = _execution_prices(provider, code_list, trade_date, price_col)
    valid = pd.to_numeric(prices.reindex(code_list), errors="coerce")
    valid_count = int(valid.gt(0).sum())
    coverage = valid_count / len(code_list) if code_list else 0.0
    required_count = min_count if min_count is not None else math.ceil(len(code_list) * float(min_coverage))
    required_count = max(0, min(int(required_count), len(code_list)))
    return bool(valid_count >= required_count and coverage >= float(min_coverage))


def _next_trade_date(
    provider: JpxHistoricalProvider,
    date: pd.Timestamp,
    lag_days: int,
    codes: Iterable[str] | None = None,
    price_col: str = "close",
    min_price_coverage: float = 0.8,
    min_price_count: int | None = None,
) -> pd.Timestamp | None:
    prices = provider.table("prices", copy=False)
    dates = pd.Series(pd.to_datetime(prices["date"].dropna().unique())).sort_values()
    after = dates.loc[dates > pd.Timestamp(date).normalize()]
    if after.empty:
        return None
    valid_candidates: list[pd.Timestamp] = []
    for candidate_raw in after:
        candidate = pd.Timestamp(candidate_raw).normalize()
        # A "next trading day" jump across months or years indicates a cache gap,
        # not a valid execution date for a daily JP equity backtest.
        if (candidate - pd.Timestamp(date).normalize()).days > 14:
            break
        if _has_valid_execution_prices(
            provider,
            codes,
            candidate,
            price_col,
            min_coverage=min_price_coverage,
            min_count=min_price_count,
        ):
            valid_candidates.append(candidate)
        if len(valid_candidates) >= max(1, int(lag_days)):
            break
    idx = max(0, int(lag_days) - 1)
    if idx >= len(valid_candidates):
        return None
    return valid_candidates[idx]


def _rescale_target_weights_to_gross(target_weights: pd.DataFrame, long_gross: float, short_gross: float) -> pd.DataFrame:
    if target_weights.empty or "target_weight" not in target_weights.columns:
        return target_weights
    out = target_weights.copy()
    weights = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0)
    long_mask = weights > 0
    short_mask = weights < 0
    long_sum = float(weights.loc[long_mask].sum())
    short_sum = float((-weights.loc[short_mask]).sum())
    if long_mask.any() and long_sum > 0 and long_gross >= 0:
        out.loc[long_mask, "target_weight"] = weights.loc[long_mask] * (float(long_gross) / long_sum)
    if short_mask.any() and short_sum > 0 and short_gross >= 0:
        out.loc[short_mask, "target_weight"] = weights.loc[short_mask] * (float(short_gross) / short_sum)
    return out


def _short_pool_rows(
    scored: pd.DataFrame,
    *,
    blocked_codes: set[str],
    expansion_mode: str,
    adv_threshold: float | None,
) -> pd.DataFrame:
    valid = scored.dropna(subset=["Score"]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["code"] = valid["code"].map(normalize_code).astype(str)
    if blocked_codes:
        valid = valid.loc[~valid["code"].isin(blocked_codes)].copy()
    if valid.empty:
        return pd.DataFrame()
    valid = valid.sort_values(["Score", "code"], ascending=[True, True]).reset_index(drop=True)
    n = len(valid)
    decile_n = max(1, int(math.ceil(n * 0.1)))
    if str(expansion_mode or "bottom_decile_only").lower() == "expand_to_bottom20":
        limit_n = max(decile_n, int(math.ceil(n * 0.2)))
    elif str(expansion_mode or "bottom_decile_only").lower() == "expand_to_bottom30":
        limit_n = max(decile_n, int(math.ceil(n * 0.3)))
    else:
        limit_n = decile_n
    pool = valid.head(limit_n).copy().reset_index(drop=True)
    pool["base_pool_rank"] = np.arange(1, len(pool) + 1)
    pool["added_from_expansion_pool"] = pool["base_pool_rank"] > decile_n
    if adv_threshold is not None and float(adv_threshold) > 0:
        adv = _safe_numeric_column(pool, "avg_adv60")
        pool = pool.loc[adv.ge(float(adv_threshold)) & np.isfinite(adv)].copy().reset_index(drop=True)
    return pool


def _short_raw_weights(rows: pd.DataFrame, weighting_mode: str) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    mode = str(weighting_mode or "equal_weight").lower()
    n = len(rows)
    if mode == "equal_weight":
        raw = np.ones(n, dtype=float)
    elif mode == "rank_weighted":
        raw = np.arange(n, 0, -1, dtype=float)
    elif mode == "score_proportional":
        cutoff = float(pd.to_numeric(rows["Score"], errors="coerce").max())
        raw = np.maximum(cutoff - pd.to_numeric(rows["Score"], errors="coerce").to_numpy(), 0.0)
        if not np.isfinite(raw).any() or float(np.nansum(raw)) <= 0:
            raw = np.arange(n, 0, -1, dtype=float)
    else:
        raw = np.ones(n, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    if float(raw.sum()) <= 0:
        raw = np.ones(n, dtype=float)
    return pd.Series(raw / raw.sum(), index=rows.index, dtype=float)


def _allocate_short_with_cap(raw_weights: pd.Series, total_target: float, cap: float) -> pd.Series:
    if raw_weights.empty or total_target <= 0 or cap <= 0:
        return pd.Series(0.0, index=raw_weights.index, dtype=float)
    capacity = len(raw_weights) * float(cap)
    total = min(float(total_target), float(capacity))
    alloc = pd.Series(0.0, index=raw_weights.index, dtype=float)
    remaining_total = total
    active = set(raw_weights.index.tolist())
    while active and remaining_total > 1e-12:
        active_idx = list(active)
        sub = raw_weights.loc[active_idx].astype(float)
        sub = sub / sub.sum()
        desired = sub * remaining_total
        room = float(cap) - alloc.loc[active_idx]
        capped = desired.clip(upper=room)
        alloc.loc[active_idx] += capped
        newly_full = [idx for idx in active_idx if alloc.loc[idx] >= float(cap) - 1e-12]
        for idx in newly_full:
            active.remove(idx)
        new_remaining_total = total - float(alloc.sum())
        if new_remaining_total >= remaining_total - 1e-12:
            break
        remaining_total = new_remaining_total
    return alloc


def _sector_share_ok(rows: pd.DataFrame, weights_norm: pd.Series, sector_cap: float | None) -> bool:
    if sector_cap is None or rows.empty:
        return True
    if "sector" not in rows.columns:
        return True
    sector_weights = pd.DataFrame({"sector": rows["sector"].astype(str), "w": weights_norm.values}).groupby("sector")["w"].sum()
    return bool((sector_weights <= float(sector_cap) + 1e-12).all())


def build_target_weights(
    scored: pd.DataFrame,
    target_count: int,
    gross_exposure: float,
    *,
    enable_short_leg: bool = False,
    short_target_count: int = 0,
    short_gross_exposure: float = 0.0,
    short_construction_mode: str = "simple_bottom_n",
    short_weighting_mode: str = "equal_weight",
    short_expansion_mode: str = "bottom_decile_only",
    short_adv_threshold: float | None = None,
    short_sector_cap: float | None = None,
    short_min_names: int = 0,
    short_max_single_weight: float = 0.10,
) -> pd.DataFrame:
    valid = scored.dropna(subset=["Score"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    long_book = (
        valid.sort_values(["Score", "code"], ascending=[False, True])
        .head(int(target_count))
        .copy()
    )
    frames: list[pd.DataFrame] = []
    if not long_book.empty:
        long_book["target_weight"] = float(gross_exposure) / len(long_book)
        long_book["side"] = "LONG"
        frames.append(long_book)
    short_mode = str(short_construction_mode or "simple_bottom_n").lower()
    short_enabled = bool(enable_short_leg) and float(short_gross_exposure) > 0 and (
        short_mode != "simple_bottom_n" or int(short_target_count) > 0
    )
    if short_enabled:
        blocked = set(long_book["code"].astype(str)) if "code" in long_book.columns else set()
        if short_mode == "simple_bottom_n":
            short_pool = valid.loc[~valid["code"].astype(str).isin(blocked)].copy()
            short_book = (
                short_pool.sort_values(["Score", "code"], ascending=[True, True])
                .head(int(short_target_count))
                .copy()
            )
            if not short_book.empty:
                short_book["target_weight"] = -float(short_gross_exposure) / len(short_book)
                short_book["side"] = "SHORT"
                frames.append(short_book)
        else:
            pool = _short_pool_rows(
                valid,
                blocked_codes=blocked,
                expansion_mode=short_expansion_mode,
                adv_threshold=short_adv_threshold,
            )
            chosen_idx: list[int] = []
            min_names = max(int(short_min_names or 0), 1)
            max_single = float(short_max_single_weight or 0.0)
            for idx in pool.index.tolist():
                trial = pool.loc[chosen_idx + [idx]].copy()
                raw = _short_raw_weights(trial, short_weighting_mode)
                if not _sector_share_ok(trial, raw, short_sector_cap):
                    continue
                chosen_idx.append(idx)
                if len(chosen_idx) >= min_names and len(chosen_idx) * max_single >= float(short_gross_exposure):
                    continue
            short_book = pool.loc[chosen_idx].copy() if chosen_idx else pd.DataFrame()
            if not short_book.empty and len(short_book) >= min_names:
                raw = _short_raw_weights(short_book, short_weighting_mode)
                alloc = _allocate_short_with_cap(raw, float(short_gross_exposure), max_single)
                effective = float(alloc.sum())
                if effective > 0:
                    short_book["target_weight"] = -alloc.values
                    short_book["side"] = "SHORT"
                    short_book["short_weighting_mode"] = str(short_weighting_mode)
                    short_book["short_expansion_mode"] = str(short_expansion_mode)
                    frames.append(short_book)
    if not frames:
        return pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    combined = pd.concat(frames, ignore_index=True)
    cols = [c for c in ["code", "target_weight", "side", "Score", "TotalRank"] if c in combined.columns]
    return combined[cols].reset_index(drop=True)


def _build_short_target_weights(
    scored: pd.DataFrame,
    *,
    blocked_codes: set[str] | None = None,
    short_gross_exposure: float = 0.0,
    short_target_count: int = 0,
    short_construction_mode: str = "simple_bottom_n",
    short_weighting_mode: str = "equal_weight",
    short_expansion_mode: str = "bottom_decile_only",
    short_adv_threshold: float | None = None,
    short_sector_cap: float | None = None,
    short_min_names: int = 0,
    short_max_single_weight: float = 0.10,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    if not isinstance(scored, pd.DataFrame) or scored.empty or float(short_gross_exposure) <= 0:
        return empty
    valid = scored.dropna(subset=["Score"]).copy()
    if valid.empty or "code" not in valid.columns:
        return empty
    valid["code"] = valid["code"].map(normalize_code).astype(str)
    blocked = {str(normalize_code(code)) for code in (blocked_codes or set()) if str(code)}
    if blocked:
        valid = valid.loc[~valid["code"].isin(blocked)].copy()
    if valid.empty:
        return empty
    short_mode = str(short_construction_mode or "simple_bottom_n").lower()
    if short_mode == "simple_bottom_n":
        short_book = (
            valid.sort_values(["Score", "code"], ascending=[True, True])
            .head(int(short_target_count))
            .copy()
        )
        if short_book.empty:
            return empty
        short_book["target_weight"] = -float(short_gross_exposure) / len(short_book)
        short_book["side"] = "SHORT"
        cols = [c for c in ["code", "target_weight", "side", "Score", "TotalRank"] if c in short_book.columns]
        return short_book[cols].reset_index(drop=True)
    pool = _short_pool_rows(
        valid,
        blocked_codes=set(),
        expansion_mode=short_expansion_mode,
        adv_threshold=short_adv_threshold,
    )
    chosen_idx: list[int] = []
    min_names = max(int(short_min_names or 0), 1)
    max_single = float(short_max_single_weight or 0.0)
    for idx in pool.index.tolist():
        trial = pool.loc[chosen_idx + [idx]].copy()
        raw = _short_raw_weights(trial, short_weighting_mode)
        if not _sector_share_ok(trial, raw, short_sector_cap):
            continue
        chosen_idx.append(idx)
        if len(chosen_idx) >= min_names and len(chosen_idx) * max_single >= float(short_gross_exposure):
            continue
    short_book = pool.loc[chosen_idx].copy() if chosen_idx else pd.DataFrame()
    if short_book.empty or len(short_book) < min_names:
        return empty
    raw = _short_raw_weights(short_book, short_weighting_mode)
    alloc = _allocate_short_with_cap(raw, float(short_gross_exposure), max_single)
    effective = float(alloc.sum())
    if effective <= 0:
        return empty
    short_book["target_weight"] = -alloc.values
    short_book["side"] = "SHORT"
    short_book["short_weighting_mode"] = str(short_weighting_mode)
    short_book["short_expansion_mode"] = str(short_expansion_mode)
    cols = [c for c in ["code", "target_weight", "side", "Score", "TotalRank"] if c in short_book.columns]
    return short_book[cols].reset_index(drop=True)


def _ranked_candidate_codes(
    scored: pd.DataFrame,
    candidate_pool_count: int,
) -> list[str]:
    if not isinstance(scored, pd.DataFrame) or scored.empty or "Score" not in scored.columns or "code" not in scored.columns:
        return []
    valid = scored.dropna(subset=["Score"]).copy()
    if valid.empty:
        return []
    valid["code"] = valid["code"].map(normalize_code).astype(str)
    ranked = valid.sort_values(["Score", "code"], ascending=[False, True]).head(max(int(candidate_pool_count), 0))
    return ranked["code"].astype(str).tolist()


def _safe_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _resolve_sector_labels(
    provider: JpxHistoricalProvider,
    frame: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "code" not in frame.columns:
        return pd.Series(dtype=str)
    codes = frame["code"].astype(str).dropna().unique().tolist()
    sector_map: dict[str, str] = {}
    if codes:
        try:
            report = RunReport(mode="diagnostic", anchor_date=str(pd.Timestamp(signal_date).date()))
            raw = provider.get_sector_map(codes, signal_date, report)
            sector_map = {
                str(normalize_code(code)): str(value)
                for code, value in raw.items()
                if pd.notna(value) and str(value).strip()
            }
        except Exception:
            sector_map = {}
    resolved = frame["code"].astype(str).map(sector_map)
    for fallback_col in ["sector", "industry"]:
        if fallback_col not in frame.columns:
            continue
        fallback = frame[fallback_col].where(frame[fallback_col].notna(), np.nan)
        fallback = fallback.astype("string")
        resolved = resolved.where(resolved.notna() & resolved.astype("string").str.strip().ne(""), fallback)
    resolved = resolved.fillna("UNKNOWN").astype(str)
    resolved = resolved.replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
    return resolved


def _bool_from_cfg(config: dict[str, Any], key: str) -> bool:
    return bool(config.get(key, False))


def _int_from_cfg(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_from_cfg(config: dict[str, Any], key: str) -> float:
    try:
        return float(config.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


REGIME_SECTOR_BIAS_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "trend_lowvol": {
        "positive": ("機械", "輸送用機器", "電気機器", "化学", "情報･通信業", "情報・通信業"),
        "negative": ("電気・ガス業", "陸運業", "小売業"),
    },
    "range_lowvol": {
        "positive": ("サービス業", "情報･通信業", "情報・通信業"),
        "negative": ("海運業", "鉄鋼", "非鉄金属"),
    },
    "riskoff_highvol": {
        "positive": ("医薬品", "食料品", "電気・ガス業", "陸運業"),
        "negative": ("銀行業", "海運業", "鉄鋼", "非鉄金属", "輸送用機器"),
    },
    "hot_riskon": {
        "positive": ("電気機器", "機械", "サービス業", "情報･通信業", "情報・通信業"),
        "negative": ("電気・ガス業", "食料品", "陸運業"),
    },
    "macro_reflation": {
        "positive": ("銀行業", "機械", "輸送用機器", "卸売業", "鉄鋼", "非鉄金属", "海運業", "建設業", "化学", "その他金融業"),
        "negative": ("小売業", "情報・通信業", "情報･通信業", "不動産業", "サービス業", "精密機器"),
    },
}


def _apply_regime_sector_bias(valid: pd.DataFrame, post_cfg: dict[str, Any], selection_regime: str) -> pd.DataFrame:
    regime_name = str(selection_regime or "").strip().lower()
    valid["regime_sector_alignment_bonus"] = 0.0
    valid["regime_sector_alignment_penalty"] = 0.0
    valid["regime_sector_adjustment_total"] = 0.0
    if not regime_name:
        return valid
    regime_bias = REGIME_SECTOR_BIAS_MAP.get(regime_name)
    if not regime_bias:
        return valid
    enabled = bool(post_cfg.get(f"{regime_name}_sector_bias_enabled", False))
    if not enabled:
        return valid
    sector_bonus = float(post_cfg.get(f"{regime_name}_sector_bonus", 0.0) or 0.0)
    sector_penalty = float(post_cfg.get(f"{regime_name}_sector_penalty", 0.0) or 0.0)
    if abs(sector_bonus) < 1e-12 and abs(sector_penalty) < 1e-12:
        return valid
    sector_text = valid["sector"].astype(str).str.strip()
    positive_mask = sector_text.isin(regime_bias.get("positive", ()))
    negative_mask = sector_text.isin(regime_bias.get("negative", ()))
    valid["regime_sector_alignment_bonus"] = np.where(positive_mask, sector_bonus, 0.0)
    valid["regime_sector_alignment_penalty"] = np.where(negative_mask, sector_penalty, 0.0)
    valid["regime_sector_adjustment_total"] = (
        valid["regime_sector_alignment_bonus"] - valid["regime_sector_alignment_penalty"]
    )
    return valid


def _resolve_sector_bias_regime(
    *,
    post_cfg: dict[str, Any] | None,
    regime_payload: dict[str, Any] | None,
    allocation_gate_payload: dict[str, Any] | None,
) -> str:
    post_cfg = dict(post_cfg or {})
    regime_payload = dict(regime_payload or {})
    allocation_gate_payload = dict(allocation_gate_payload or {})
    source = str(post_cfg.get("regime_sector_bias_regime_source", "active_regime") or "active_regime").strip().lower()
    if source == "weight_regime":
        return str(allocation_gate_payload.get("weight_regime", "") or "")
    if source == "position_regime":
        return str(allocation_gate_payload.get("position_regime", "") or "")
    if source == "raw_regime":
        return str(regime_payload.get("raw_regime", "") or "")
    return str(regime_payload.get("active_regime", "") or "")


def _merge_scored_with_universe_metadata(scored: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(scored, pd.DataFrame) or scored.empty or "code" not in scored.columns:
        return scored
    if not isinstance(universe, pd.DataFrame) or universe.empty or "code" not in universe.columns:
        return scored
    meta_cols = [
        c
        for c in [
            "code",
            "industry",
            "market_cap_jpy",
            "market_cap_billion_jpy",
            "market_cap_rank",
            "avg_daily_volume",
            "avg_daily_trading_value",
            "annualized_volatility",
        ]
        if c in universe.columns
    ]
    if not meta_cols:
        return scored
    meta = universe[meta_cols].copy()
    meta["code"] = meta["code"].map(normalize_code).astype(str)
    meta = meta.drop_duplicates("code", keep="last")
    add_cols = [c for c in meta.columns if c == "code" or c not in scored.columns]
    return scored.merge(meta[add_cols], on="code", how="left")


def _build_post_score_adjusted_targets(
    scored: pd.DataFrame,
    *,
    provider: JpxHistoricalProvider,
    signal_date: pd.Timestamp,
    target_count: int,
    gross_exposure: float,
    current_holdings: set[str] | None = None,
    holding_ages: dict[str, int] | None = None,
    blocked_codes: set[str] | None = None,
    prices: pd.Series | None = None,
    rescale_partial: bool = False,
    post_cfg: dict[str, Any] | None = None,
    constraint_cfg: dict[str, Any] | None = None,
    selection_regime: str | None = None,
    enable_short_leg: bool = False,
    short_gross_exposure: float = 0.0,
    short_target_count: int = 0,
    short_construction_mode: str = "simple_bottom_n",
    short_weighting_mode: str = "equal_weight",
    short_expansion_mode: str = "bottom_decile_only",
    short_adv_threshold: float | None = None,
    short_sector_cap: float | None = None,
    short_min_names: int = 0,
    short_max_single_weight: float = 0.10,
) -> dict[str, Any]:
    empty_target = pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    empty_debug = pd.DataFrame(
        columns=[
            "signal_date",
            "code",
            "original_score",
            "adjusted_score",
            "Q",
            "V",
            "M",
            "M_gate",
            "C",
            "sector",
            "selection_regime",
            "market_cap_rank",
            "avg_daily_volume",
            "regime_sector_alignment_bonus",
            "regime_sector_alignment_penalty",
            "regime_sector_adjustment_total",
            "macro_sector_alignment_bonus",
            "macro_large_cap_bonus",
            "macro_leader_core_bonus",
            "macro_trend_confirmation_bonus",
            "macro_leader_combo_bonus",
            "macro_small_cap_penalty",
            "macro_low_liquidity_penalty",
            "macro_weak_trend_penalty",
            "macro_non_leader_penalty",
            "macro_reflation_adjustment_total",
            "was_current_holding",
            "holding_age_months",
            "mgate_negative_penalty",
            "value_trap_penalty",
            "hard_block_reason",
            "final_rank_before_constraints",
            "final_rank_after_constraints",
            "selected_final_top15",
            "selected_reason",
            "exclusion_reason",
            "final_target_weight",
            "max_names_per_sector",
            "sector_names_count_before",
            "sector_names_count_after",
            "skipped_due_to_sector_cap",
            "allowed_by_exception",
            "sector_cap_penalty_applied",
        ]
    )
    if not isinstance(scored, pd.DataFrame) or scored.empty or "Score" not in scored.columns or "code" not in scored.columns:
        return {
            "target_weights": empty_target,
            "debug": empty_debug,
            "sector_exposure": pd.DataFrame(columns=["signal_date", "sector", "names_count", "sector_weight"]),
            "holding_age_debug": pd.DataFrame(columns=["signal_date", "code", "holding_age_months", "M", "M_gate", "score_rank", "exited_due_to_age_rule", "exited_due_to_mgate_rule", "exited_due_to_value_trap_rule"]),
            "selected_codes": [],
        }

    current_holdings = {str(code) for code in (current_holdings or set())}
    holding_ages = {str(code): int(age) for code, age in (holding_ages or {}).items()}
    blocked_codes = {str(code) for code in (blocked_codes or set())}
    post_cfg = dict(post_cfg or {})
    constraint_cfg = dict(constraint_cfg or {})
    target_count = int(target_count)
    max_names_per_sector = _int_from_cfg(constraint_cfg, "max_names_per_sector")
    sector_cap_mode = str(constraint_cfg.get("sector_cap_mode", "hard") or "hard").strip().lower()
    sector_cap_penalty = float(constraint_cfg.get("sector_cap_penalty", 0.0) or 0.0)
    sector_cap_exception_margin = float(constraint_cfg.get("sector_cap_exception_margin", 0.0) or 0.0)
    sector_cap_exception_max = int(constraint_cfg.get("sector_cap_exception_max", 0) or 0)
    mgate_lambda = _float_from_cfg(post_cfg, "lambda_mgate_neg")
    value_trap_penalty_value = _float_from_cfg(post_cfg, "value_trap_penalty")
    value_trap_v_threshold = float(post_cfg.get("value_trap_v_threshold", 1.0) or 1.0)
    hard_block_deterioration = _bool_from_cfg(post_cfg, "holding_deterioration_hard_block")
    age_exit_months_mgate_negative = _int_from_cfg(post_cfg, "holding_age_exit_months_mgate_negative")
    age_rank_months = _int_from_cfg(post_cfg, "holding_age_must_rank_months")
    age_rank_top_n = _int_from_cfg(post_cfg, "holding_age_must_rank_top_n")
    macro_sector_enabled = bool(post_cfg.get("macro_reflation_sector_alignment_enabled", True))
    macro_size_enabled = bool(post_cfg.get("macro_reflation_size_tilt_enabled", True))
    macro_sector_bonus = float(post_cfg.get("macro_reflation_sector_bonus", 0.18) or 0.0)
    macro_sector_penalty = float(post_cfg.get("macro_reflation_sector_penalty", 0.12) or 0.0)
    macro_large_cap_bonus = float(post_cfg.get("macro_reflation_large_cap_bonus", 0.06) or 0.0)
    macro_small_cap_penalty = float(post_cfg.get("macro_reflation_small_cap_penalty", 0.12) or 0.0)
    macro_low_liquidity_penalty = float(post_cfg.get("macro_reflation_low_liquidity_penalty", 0.10) or 0.0)
    macro_large_cap_rank_threshold = int(post_cfg.get("macro_reflation_large_cap_rank_threshold", 250) or 250)
    macro_small_cap_rank_threshold = int(post_cfg.get("macro_reflation_small_cap_rank_threshold", 700) or 700)
    macro_liquidity_floor = float(post_cfg.get("macro_reflation_liquidity_floor", 1_000_000.0) or 0.0)
    macro_leader_bias_enabled = bool(post_cfg.get("macro_reflation_leader_bias_enabled", True))
    macro_leader_core_bonus = float(post_cfg.get("macro_reflation_leader_core_bonus", 0.10) or 0.0)
    macro_trend_confirmation_bonus = float(post_cfg.get("macro_reflation_trend_confirmation_bonus", 0.06) or 0.0)
    macro_leader_combo_bonus = float(post_cfg.get("macro_reflation_leader_combo_bonus", 0.12) or 0.0)
    macro_weak_trend_penalty = float(post_cfg.get("macro_reflation_weak_trend_penalty", 0.04) or 0.0)
    macro_non_leader_penalty = float(post_cfg.get("macro_reflation_non_leader_penalty", 0.04) or 0.0)
    macro_mgate_floor = float(post_cfg.get("macro_reflation_mgate_floor", 0.0) or 0.0)
    macro_value_floor = float(post_cfg.get("macro_reflation_value_floor", 0.0) or 0.0)
    macro_sector_positive = {
        "銀行業",
        "機械",
        "輸送用機器",
        "卸売業",
        "鉄鋼",
        "非鉄金属",
        "海運業",
        "建設業",
        "化学",
        "その他金融業",
    }
    macro_sector_negative = {
        "小売業",
        "情報・通信業",
        "情報･通信業",
        "不動産業",
        "サービス業",
        "精密機器",
    }

    valid = scored.dropna(subset=["Score"]).copy()
    if valid.empty:
        return {
            "target_weights": empty_target,
            "debug": empty_debug,
            "sector_exposure": pd.DataFrame(columns=["signal_date", "sector", "names_count", "sector_weight"]),
            "holding_age_debug": pd.DataFrame(columns=["signal_date", "code", "holding_age_months", "M", "M_gate", "score_rank", "exited_due_to_age_rule", "exited_due_to_mgate_rule", "exited_due_to_value_trap_rule"]),
            "selected_codes": [],
        }
    valid["code"] = valid["code"].map(normalize_code).astype(str)
    valid["signal_date"] = pd.Timestamp(signal_date).normalize()
    valid["original_score"] = pd.to_numeric(valid["Score"], errors="coerce")
    valid["Q"] = _safe_numeric_column(valid, "Q")
    valid["V"] = _safe_numeric_column(valid, "V")
    valid["M"] = _safe_numeric_column(valid, "M")
    valid["M_gate"] = _safe_numeric_column(valid, "M_gate")
    valid["C"] = _safe_numeric_column(valid, "C")
    valid["market_cap_rank"] = _safe_numeric_column(valid, "market_cap_rank")
    valid["avg_daily_volume"] = _safe_numeric_column(valid, "avg_daily_volume")
    valid["sector"] = _resolve_sector_labels(provider, valid, signal_date)
    valid["selection_regime"] = str(selection_regime or "")
    valid["was_current_holding"] = valid["code"].isin(current_holdings)
    valid["holding_age_months"] = valid["code"].map(lambda code: int(holding_ages.get(str(code), 0)))
    valid["mgate_negative_penalty"] = float(mgate_lambda) * np.maximum(-valid["M_gate"].fillna(0.0), 0.0)
    trap_mask = valid["V"].gt(float(value_trap_v_threshold)) & valid["M_gate"].lt(0.0)
    valid["value_trap_penalty"] = np.where(trap_mask, float(value_trap_penalty_value), 0.0)
    valid["macro_sector_alignment_bonus"] = 0.0
    valid["macro_large_cap_bonus"] = 0.0
    valid["macro_leader_core_bonus"] = 0.0
    valid["macro_trend_confirmation_bonus"] = 0.0
    valid["macro_leader_combo_bonus"] = 0.0
    valid["macro_small_cap_penalty"] = 0.0
    valid["macro_low_liquidity_penalty"] = 0.0
    valid["macro_weak_trend_penalty"] = 0.0
    valid["macro_non_leader_penalty"] = 0.0
    valid = _apply_regime_sector_bias(valid, post_cfg, valid["selection_regime"].iloc[0] if not valid.empty else "")
    if str(selection_regime or "").strip().lower() == "macro_reflation":
        sector_text = valid["sector"].astype(str).str.strip()
        sector_positive_mask = sector_text.isin(macro_sector_positive)
        sector_negative_mask = sector_text.isin(macro_sector_negative)
        large_cap_ok = valid["market_cap_rank"].le(float(macro_large_cap_rank_threshold))
        small_cap_flag = valid["market_cap_rank"].ge(float(macro_small_cap_rank_threshold))
        liquidity_ok = valid["avg_daily_volume"].ge(float(macro_liquidity_floor))
        low_liquidity_flag = valid["avg_daily_volume"].lt(float(macro_liquidity_floor))
        trend_ok = valid["M_gate"].fillna(0.0).ge(float(macro_mgate_floor))
        value_ok = valid["V"].fillna(0.0).ge(float(macro_value_floor))
        leader_core_ok = sector_positive_mask & large_cap_ok & liquidity_ok
        leader_combo_ok = leader_core_ok & trend_ok & value_ok
        if macro_sector_enabled:
            valid["macro_sector_alignment_bonus"] = np.where(
                sector_positive_mask,
                float(macro_sector_bonus),
                np.where(sector_negative_mask, -float(macro_sector_penalty), 0.0),
            )
        if macro_size_enabled:
            valid["macro_large_cap_bonus"] = np.where(
                large_cap_ok,
                float(macro_large_cap_bonus),
                0.0,
            )
            valid["macro_small_cap_penalty"] = np.where(
                small_cap_flag,
                float(macro_small_cap_penalty),
                0.0,
            )
            valid["macro_low_liquidity_penalty"] = np.where(
                low_liquidity_flag,
                float(macro_low_liquidity_penalty),
                0.0,
            )
        if macro_leader_bias_enabled:
            valid["macro_leader_core_bonus"] = np.where(leader_core_ok, float(macro_leader_core_bonus), 0.0)
            valid["macro_trend_confirmation_bonus"] = np.where(
                leader_core_ok & trend_ok,
                float(macro_trend_confirmation_bonus),
                0.0,
            )
            valid["macro_leader_combo_bonus"] = np.where(leader_combo_ok, float(macro_leader_combo_bonus), 0.0)
            valid["macro_weak_trend_penalty"] = np.where(~trend_ok, float(macro_weak_trend_penalty), 0.0)
            valid["macro_non_leader_penalty"] = np.where(
                (~sector_positive_mask) & ((~large_cap_ok) | (~liquidity_ok)),
                float(macro_non_leader_penalty),
                0.0,
            )
    valid["macro_reflation_adjustment_total"] = (
        valid["macro_sector_alignment_bonus"]
        + valid["macro_large_cap_bonus"]
        + valid["macro_leader_core_bonus"]
        + valid["macro_trend_confirmation_bonus"]
        + valid["macro_leader_combo_bonus"]
        - valid["macro_small_cap_penalty"]
        - valid["macro_low_liquidity_penalty"]
        - valid["macro_weak_trend_penalty"]
        - valid["macro_non_leader_penalty"]
    )
    valid["adjusted_score"] = (
        valid["original_score"]
        - valid["mgate_negative_penalty"]
        - valid["value_trap_penalty"]
        + valid["regime_sector_adjustment_total"]
        + valid["macro_reflation_adjustment_total"]
    )
    valid = valid.sort_values(["adjusted_score", "code"], ascending=[False, True]).reset_index(drop=True)
    valid["final_rank_before_constraints"] = np.arange(1, len(valid) + 1)
    valid["hard_block_reason"] = ""
    valid["selected_reason"] = ""
    valid["exclusion_reason"] = ""
    valid["selected_final_top15"] = False
    valid["final_rank_after_constraints"] = np.nan
    valid["final_target_weight"] = 0.0
    valid["max_names_per_sector"] = max_names_per_sector
    valid["sector_names_count_before"] = np.nan
    valid["sector_names_count_after"] = np.nan
    valid["skipped_due_to_sector_cap"] = False
    valid["allowed_by_exception"] = False
    valid["sector_cap_penalty_applied"] = 0.0
    if prices is not None:
        px_view = pd.to_numeric(prices.reindex(valid["code"].astype(str).tolist()), errors="coerce")
        valid["valid_execution_price"] = px_view.to_numpy()
        valid["valid_execution_price"] = np.isfinite(valid["valid_execution_price"]) & (valid["valid_execution_price"] > 0)
    else:
        valid["valid_execution_price"] = True

    for idx, row in valid.iterrows():
        code = str(row["code"])
        age = int(row["holding_age_months"])
        if code in blocked_codes:
            valid.at[idx, "hard_block_reason"] = "cooldown_or_stop_block"
            continue
        if hard_block_deterioration and bool(row["was_current_holding"]) and float(row["M_gate"] if pd.notna(row["M_gate"]) else 0.0) < 0.0 and float(row["M"] if pd.notna(row["M"]) else 0.0) < 0.0:
            valid.at[idx, "hard_block_reason"] = "holding_mgate_and_m_negative"
            continue
        if age_exit_months_mgate_negative is not None and bool(row["was_current_holding"]) and age >= int(age_exit_months_mgate_negative) and float(row["M_gate"] if pd.notna(row["M_gate"]) else 0.0) < 0.0:
            valid.at[idx, "hard_block_reason"] = f"holding_age{int(age_exit_months_mgate_negative)}_mgate_negative"
            continue
        if age_rank_months is not None and age_rank_top_n is not None and bool(row["was_current_holding"]) and age >= int(age_rank_months) and int(row["final_rank_before_constraints"]) > int(age_rank_top_n):
            valid.at[idx, "hard_block_reason"] = f"holding_age{int(age_rank_months)}_rank_gt_{int(age_rank_top_n)}"

    selected_idx: list[int] = []
    sector_counts: dict[str, int] = {}
    active_idx = [
        int(idx)
        for idx, row in valid.iterrows()
        if not row["hard_block_reason"] and bool(row["valid_execution_price"])
    ]

    for idx, row in valid.iterrows():
        if row["hard_block_reason"]:
            valid.at[idx, "exclusion_reason"] = row["hard_block_reason"]
        elif not bool(row["valid_execution_price"]):
            valid.at[idx, "exclusion_reason"] = "missing_execution_price"

    selection_rank = 0
    remaining = list(active_idx)
    while remaining and selection_rank < int(target_count):
        candidate_frame = valid.loc[remaining].copy()
        candidate_frame["effective_score"] = candidate_frame["adjusted_score"]
        candidate_frame["sector_names_count_before"] = candidate_frame["sector"].map(lambda s: sector_counts.get(str(s), 0))

        if max_names_per_sector is not None and sector_cap_mode == "soft_penalty" and float(sector_cap_penalty) > 0:
            over_mask = candidate_frame["sector_names_count_before"].ge(int(max_names_per_sector))
            candidate_frame.loc[over_mask, "effective_score"] = candidate_frame.loc[over_mask, "effective_score"] - float(sector_cap_penalty)
            candidate_frame.loc[over_mask, "sector_cap_penalty_applied"] = float(sector_cap_penalty)

        candidate_frame = candidate_frame.sort_values(["effective_score", "code"], ascending=[False, True]).reset_index()
        if candidate_frame.empty:
            break

        chosen_idx: int | None = None
        chosen_reason = "selected_topN"
        for _, cand in candidate_frame.iterrows():
            idx = int(cand["index"])
            sector = str(cand["sector"])
            before_count = int(sector_counts.get(sector, 0))
            valid.at[idx, "sector_names_count_before"] = before_count
            if max_names_per_sector is None:
                chosen_idx = idx
                break
            if before_count < int(max_names_per_sector):
                chosen_idx = idx
                break
            if sector_cap_mode == "hard":
                valid.at[idx, "skipped_due_to_sector_cap"] = True
                if not valid.at[idx, "exclusion_reason"]:
                    valid.at[idx, "exclusion_reason"] = f"sector_cap_{int(max_names_per_sector)}"
                continue
            if sector_cap_mode == "soft_penalty":
                chosen_idx = idx
                chosen_reason = "selected_topN_soft_sector_penalty"
                break
            if sector_cap_mode == "exception":
                if before_count >= int(max(sector_cap_exception_max, max_names_per_sector)):
                    valid.at[idx, "skipped_due_to_sector_cap"] = True
                    if not valid.at[idx, "exclusion_reason"]:
                        valid.at[idx, "exclusion_reason"] = f"sector_cap_exception_max_{int(max(sector_cap_exception_max, max_names_per_sector))}"
                    continue
                alt_candidates = candidate_frame[
                    candidate_frame["sector"].astype(str).map(lambda s: sector_counts.get(str(s), 0) < int(max_names_per_sector))
                ]
                alt_best = float(alt_candidates["effective_score"].iloc[0]) if not alt_candidates.empty else -np.inf
                cand_score = float(cand["effective_score"])
                if cand_score >= alt_best + float(sector_cap_exception_margin):
                    chosen_idx = idx
                    chosen_reason = "selected_topN_sector_exception"
                    valid.at[idx, "allowed_by_exception"] = True
                    break
                valid.at[idx, "skipped_due_to_sector_cap"] = True
                if not valid.at[idx, "exclusion_reason"]:
                    valid.at[idx, "exclusion_reason"] = f"sector_cap_exception_margin_{float(sector_cap_exception_margin):.2f}"
                continue
            valid.at[idx, "skipped_due_to_sector_cap"] = True
            if not valid.at[idx, "exclusion_reason"]:
                valid.at[idx, "exclusion_reason"] = f"sector_cap_{int(max_names_per_sector)}"

        if chosen_idx is None:
            break

        chosen_row = valid.loc[chosen_idx]
        chosen_sector = str(chosen_row["sector"])
        before_count = int(sector_counts.get(chosen_sector, 0))
        after_count = before_count + 1
        selection_rank += 1
        selected_idx.append(chosen_idx)
        sector_counts[chosen_sector] = after_count
        valid.at[chosen_idx, "selected_final_top15"] = True
        valid.at[chosen_idx, "selected_reason"] = chosen_reason
        valid.at[chosen_idx, "final_rank_after_constraints"] = float(selection_rank)
        valid.at[chosen_idx, "sector_names_count_before"] = before_count
        valid.at[chosen_idx, "sector_names_count_after"] = after_count
        remaining = [idx for idx in remaining if idx != chosen_idx]

    for idx, row in valid.iterrows():
        if bool(row["selected_final_top15"]):
            continue
        if not row["exclusion_reason"]:
            valid.at[idx, "exclusion_reason"] = "not_in_topN_after_constraints"

    selected = valid.loc[selected_idx].copy()
    if not selected.empty:
        if len(selected) >= int(target_count) and int(target_count) > 0:
            per_weight = float(gross_exposure) / float(target_count)
        elif bool(rescale_partial):
            per_weight = float(gross_exposure) / float(len(selected))
        else:
            per_weight = float(gross_exposure) / float(max(int(target_count), 1))
        selected["target_weight"] = per_weight
        selected["side"] = "LONG"
        selected["Score"] = selected["adjusted_score"]
        selected["TotalRank"] = selected["final_rank_after_constraints"]
        valid.loc[selected.index, "final_target_weight"] = per_weight
        target_weights = selected[[c for c in ["code", "target_weight", "side", "Score", "TotalRank", "sector"] if c in selected.columns]].reset_index(drop=True)
    else:
        target_weights = empty_target.copy()

    short_weights = pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    if bool(enable_short_leg) and float(short_gross_exposure) > 0:
        short_source = valid.loc[active_idx].copy()
        selected_long_codes = set(selected["code"].astype(str).tolist()) if not selected.empty else set()
        if selected_long_codes:
            short_source = short_source.loc[~short_source["code"].astype(str).isin(selected_long_codes)].copy()
        if not short_source.empty:
            short_source["Score"] = short_source["adjusted_score"]
            short_source["TotalRank"] = short_source["final_rank_before_constraints"]
            short_weights = _build_short_target_weights(
                short_source,
                blocked_codes=selected_long_codes,
                short_gross_exposure=float(short_gross_exposure),
                short_target_count=int(short_target_count),
                short_construction_mode=str(short_construction_mode),
                short_weighting_mode=str(short_weighting_mode),
                short_expansion_mode=str(short_expansion_mode),
                short_adv_threshold=short_adv_threshold,
                short_sector_cap=short_sector_cap,
                short_min_names=int(short_min_names),
                short_max_single_weight=float(short_max_single_weight),
            )
            if not short_weights.empty:
                index_lookup = (
                    valid.reset_index()[["index", "code"]]
                    .drop_duplicates("code", keep="first")
                    .set_index("code")["index"]
                    .to_dict()
                )
                long_count = int(len(selected))
                for offset, row in enumerate(short_weights.itertuples(index=False), start=1):
                    code = str(row.code)
                    idx = index_lookup.get(code)
                    if idx is None:
                        continue
                    valid.at[idx, "selected_final_top15"] = True
                    valid.at[idx, "selected_reason"] = "selected_short_hedge"
                    valid.at[idx, "exclusion_reason"] = ""
                    valid.at[idx, "final_rank_after_constraints"] = float(long_count + offset)
                    valid.at[idx, "final_target_weight"] = float(row.target_weight)
                target_weights = pd.concat([target_weights, short_weights], ignore_index=True)

    holding_debug = valid.loc[valid["was_current_holding"]].copy()
    if holding_debug.empty:
        holding_age_debug = pd.DataFrame(columns=["signal_date", "code", "holding_age_months", "M", "M_gate", "score_rank", "exited_due_to_age_rule", "exited_due_to_mgate_rule", "exited_due_to_value_trap_rule"])
    else:
        holding_age_debug = pd.DataFrame(
            {
                "signal_date": holding_debug["signal_date"],
                "code": holding_debug["code"],
                "holding_age_months": holding_debug["holding_age_months"],
                "M": holding_debug["M"],
                "M_gate": holding_debug["M_gate"],
                "score_rank": holding_debug["final_rank_before_constraints"],
                "exited_due_to_age_rule": holding_debug["hard_block_reason"].astype(str).str.contains("holding_age", na=False),
                "exited_due_to_mgate_rule": holding_debug["hard_block_reason"].astype(str).str.contains("mgate", na=False),
                "exited_due_to_value_trap_rule": (holding_debug["value_trap_penalty"] > 0) & (~holding_debug["selected_final_top15"]),
            }
        )

    if not target_weights.empty and "code" in target_weights.columns and "target_weight" in target_weights.columns:
        sector_lookup = valid.set_index("code")["sector"].to_dict()
        sector_weights = target_weights.copy()
        sector_weights["sector"] = sector_weights["code"].astype(str).map(sector_lookup).fillna("UNKNOWN")
        sector_exposure = (
            sector_weights.groupby("sector", dropna=False)
            .agg(names_count=("code", "count"), sector_weight=("target_weight", "sum"))
            .reset_index()
        )
        sector_exposure.insert(0, "signal_date", pd.Timestamp(signal_date).normalize())
    else:
        sector_exposure = pd.DataFrame(columns=["signal_date", "sector", "names_count", "sector_weight"])

    debug = valid[
        [
            "signal_date",
            "code",
            "original_score",
            "adjusted_score",
            "Q",
            "V",
            "M",
            "M_gate",
            "C",
            "sector",
            "selection_regime",
            "market_cap_rank",
            "avg_daily_volume",
            "macro_sector_alignment_bonus",
            "macro_large_cap_bonus",
            "macro_leader_core_bonus",
            "macro_trend_confirmation_bonus",
            "macro_leader_combo_bonus",
            "macro_small_cap_penalty",
            "macro_low_liquidity_penalty",
            "macro_weak_trend_penalty",
            "macro_non_leader_penalty",
            "macro_reflation_adjustment_total",
            "was_current_holding",
            "holding_age_months",
            "mgate_negative_penalty",
            "value_trap_penalty",
            "hard_block_reason",
            "final_rank_before_constraints",
            "final_rank_after_constraints",
            "selected_final_top15",
            "selected_reason",
            "exclusion_reason",
            "final_target_weight",
            "max_names_per_sector",
            "sector_names_count_before",
            "sector_names_count_after",
            "skipped_due_to_sector_cap",
            "allowed_by_exception",
            "sector_cap_penalty_applied",
        ]
    ].copy()
    return {
        "target_weights": target_weights,
        "debug": debug.reset_index(drop=True),
        "sector_exposure": sector_exposure.reset_index(drop=True),
        "holding_age_debug": holding_age_debug.reset_index(drop=True),
        "selected_codes": selected["code"].astype(str).tolist(),
    }


def build_target_weights_from_executable_candidates(
    scored: pd.DataFrame,
    prices: pd.Series,
    target_count: int,
    gross_exposure: float,
    *,
    candidate_pool_count: int = 50,
    exclude_codes: set[str] | None = None,
    rescale_partial: bool = False,
    enable_short_leg: bool = False,
    short_target_count: int = 0,
    short_gross_exposure: float = 0.0,
    short_construction_mode: str = "simple_bottom_n",
    short_weighting_mode: str = "equal_weight",
    short_expansion_mode: str = "bottom_decile_only",
    short_adv_threshold: float | None = None,
    short_sector_cap: float | None = None,
    short_min_names: int = 0,
    short_max_single_weight: float = 0.10,
) -> pd.DataFrame:
    if enable_short_leg:
        return build_target_weights(
            scored,
            target_count,
            gross_exposure,
            enable_short_leg=enable_short_leg,
            short_target_count=short_target_count,
            short_gross_exposure=short_gross_exposure,
            short_construction_mode=short_construction_mode,
            short_weighting_mode=short_weighting_mode,
            short_expansion_mode=short_expansion_mode,
            short_adv_threshold=short_adv_threshold,
            short_sector_cap=short_sector_cap,
            short_min_names=short_min_names,
            short_max_single_weight=short_max_single_weight,
        )
    if not isinstance(scored, pd.DataFrame) or scored.empty:
        return pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    valid = scored.dropna(subset=["Score"]).copy()
    if valid.empty or "code" not in valid.columns:
        return pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    valid["code"] = valid["code"].map(normalize_code).astype(str)
    blocked = {str(normalize_code(code)) for code in (exclude_codes or set()) if str(code)}
    if blocked:
        valid = valid.loc[~valid["code"].isin(blocked)].copy()
    if valid.empty:
        return pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    ranked = valid.sort_values(["Score", "code"], ascending=[False, True]).head(max(int(candidate_pool_count), 0)).copy()
    if ranked.empty:
        return pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    candidate_codes = ranked["code"].astype(str).tolist()
    candidate_prices = pd.to_numeric(prices.reindex(candidate_codes), errors="coerce")
    executable_codes = candidate_prices.loc[np.isfinite(candidate_prices) & candidate_prices.gt(0)].index.tolist()
    if int(target_count) > 0:
        executable_codes = executable_codes[: int(target_count)]
    if not executable_codes:
        return pd.DataFrame(columns=["code", "target_weight", "side", "Score", "TotalRank"])
    long_book = ranked.loc[ranked["code"].isin(executable_codes)].copy()
    order_map = {code: idx for idx, code in enumerate(executable_codes)}
    long_book["_exec_order"] = long_book["code"].map(order_map)
    long_book = long_book.sort_values("_exec_order").drop(columns="_exec_order")
    if len(long_book) >= int(target_count) and int(target_count) > 0:
        target_weight = float(gross_exposure) / int(target_count)
    elif bool(rescale_partial):
        target_weight = float(gross_exposure) / len(long_book)
    else:
        target_weight = float(gross_exposure) / max(int(target_count), 1)
    long_book["target_weight"] = target_weight
    long_book["side"] = "LONG"
    cols = [c for c in ["code", "target_weight", "side", "Score", "TotalRank"] if c in long_book.columns]
    return long_book[cols].reset_index(drop=True)


def _apply_cash_penalty_to_target_weights(
    target_weights: pd.DataFrame,
    *,
    max_gross_exposure: float,
    cash_penalty_pct: float,
) -> pd.DataFrame:
    if target_weights.empty or "target_weight" not in target_weights.columns:
        return target_weights
    penalty = float(cash_penalty_pct or 0.0)
    if penalty <= 0:
        return target_weights
    adjusted = target_weights.copy()
    gross_target = max(0.0, float(max_gross_exposure) * (1.0 - penalty))
    weights = pd.to_numeric(adjusted["target_weight"], errors="coerce").fillna(0.0)
    current_gross = float(weights.abs().sum())
    if current_gross <= 0:
        return adjusted
    adjusted["target_weight"] = weights * (gross_target / current_gross)
    return adjusted


def _allocate_target_quantities(
    target_weights: pd.DataFrame,
    prices: pd.Series,
    nav_before: float,
    lot_size: int,
    round_lots: bool,
) -> tuple[pd.Series, pd.Series]:
    if target_weights.empty or "code" not in target_weights.columns or "target_weight" not in target_weights.columns:
        empty_qty = pd.Series(dtype=int)
        empty_value = pd.Series(dtype=float)
        return empty_qty, empty_value
    lots = max(int(lot_size), 1) if round_lots else 1
    base_target_value = pd.to_numeric(target_weights.set_index("code")["target_weight"], errors="coerce").fillna(0.0) * float(nav_before)
    desired_qty = pd.Series(0, index=base_target_value.index, dtype="int64")
    allocated_value = pd.Series(0.0, index=base_target_value.index, dtype="float64")
    valid_prices = pd.to_numeric(prices.reindex(base_target_value.index), errors="coerce")

    for code, target_val in base_target_value.items():
        px = float(valid_prices.get(code, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        sign = 1 if float(target_val) > 0 else (-1 if float(target_val) < 0 else 0)
        raw_qty = int(np.floor(abs(float(target_val)) / px))
        if round_lots and lots > 1:
            raw_qty = int(np.floor(raw_qty / lots) * lots)
        if raw_qty <= 0 or sign == 0:
            continue
        desired_qty.at[code] = sign * raw_qty
        allocated_value.at[code] = sign * float(raw_qty) * px

    for sign in (1, -1):
        side_targets = base_target_value[base_target_value * sign > 0]
        if side_targets.empty:
            continue
        remaining_budget = float(side_targets.abs().sum() - allocated_value.reindex(side_targets.index).abs().sum())
        if remaining_budget <= 0:
            continue
        lot_cost = valid_prices.reindex(side_targets.index) * float(lots)
        while True:
            affordable = lot_cost[(lot_cost > 0) & (lot_cost <= remaining_budget)]
            if affordable.empty:
                break
            active_codes = [code for code in affordable.index if code in allocated_value.index]
            if not active_codes:
                break
            code = min(
                active_codes,
                key=lambda c: (
                    float(abs(allocated_value.get(c, 0.0)) / abs(side_targets.get(c, 1.0))) if float(abs(side_targets.get(c, 0.0))) > 0 else float("inf"),
                    float(valid_prices.get(c, np.inf)),
                    str(c),
                ),
            )
            extra_cost = float(lot_cost.get(code, np.nan))
            if not np.isfinite(extra_cost) or extra_cost <= 0 or extra_cost > remaining_budget:
                break
            desired_qty.at[code] = int(desired_qty.get(code, 0)) + sign * lots
            allocated_value.at[code] = float(allocated_value.get(code, 0.0)) + sign * extra_cost
            remaining_budget -= extra_cost

    return desired_qty.astype(int), allocated_value.astype(float)


def validate_provider_readiness(provider: JpxHistoricalProvider, anchor_date: str | pd.Timestamp) -> dict[str, int]:
    """Fail fast when required local historical cache tables are empty."""

    asof = pd.Timestamp(anchor_date).normalize()
    report = RunReport(mode="historical_readiness", anchor_date=str(asof.date()))
    try:
        universe = provider.get_universe(asof, report)
    except Exception as exc:
        raise RuntimeError(f"Historical provider readiness failed: universe unavailable for {asof.date()}: {exc}") from exc
    try:
        sector = provider.get_sector_map(universe, asof, report) if universe else {}
    except Exception as exc:
        raise RuntimeError(f"Historical provider readiness failed: sector map unavailable for {asof.date()}: {exc}") from exc
    try:
        caps = provider.get_market_caps(universe, asof, report) if universe else pd.DataFrame()
    except Exception as exc:
        raise RuntimeError(f"Historical provider readiness failed: market caps unavailable for {asof.date()}: {exc}") from exc
    counts = {"universe": len(universe), "sector_map": len(sector), "market_caps": len(caps)}
    missing = [name for name, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(
            "Historical provider is missing prerequisite cache tables: "
            f"universe={counts['universe']}, sector_map={counts['sector_map']}, market_caps={counts['market_caps']}, "
            f"anchor_date={asof.date()}, missing={missing}"
        )
    return counts


def run_one_rebalance(
    rebalance_date: str | pd.Timestamp,
    provider: JpxHistoricalProvider,
    rules: ScreenRules | None = None,
    config: BacktestConfig | dict[str, Any] | None = None,
    regime_state: RegimeState | None = None,
) -> RebalanceResult:
    """Run one point-in-time historical rebalance signal.

    Signal inputs are cut at rebalance_date.  The returned execution_date is the
    next available trading day by default; portfolio simulation applies orders
    there, keeping signal date and trade date separate.
    """

    cfg = _cfg(config)
    signal_date = pd.Timestamp(rebalance_date).normalize()
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    started = time.perf_counter()
    validate_provider_readiness(provider, signal_date)
    timings["readiness_sec"] = time.perf_counter() - started

    screen_rules = rules or ScreenRules()
    started = time.perf_counter()
    snapshot_cache_status = "disabled"
    screen = None
    snapshot_paths = None
    if cfg.enable_snapshot_cache:
        snapshot_paths = build_snapshot_cache_paths(
            cfg.snapshot_cache_dir,
            anchor_date=signal_date,
            rules=screen_rules,
            provider=provider,
        )
        screen = load_snapshot_cache(snapshot_paths, signal_date)
        if screen is not None:
            snapshot_cache_status = "hit"
            print(
                f"[snapshot-cache] hit date={signal_date.date()} "
                f"key={snapshot_paths['cache_key']} snapshot_rows={len(screen.snapshot)} selected_rows={len(screen.result)}",
                flush=True,
            )
        else:
            snapshot_cache_status = "miss"
            print(f"[snapshot-cache] miss date={signal_date.date()} reason=no_matching_key", flush=True)
    if screen is None:
        screen = run_realtime_screen(rules=screen_rules, provider=provider, anchor_date=signal_date, verbose=False)
        if cfg.enable_snapshot_cache and snapshot_paths is not None:
            save_snapshot_cache(snapshot_paths, screen, signal_date, screen_rules)
            if Path(snapshot_paths["result_path"]).exists():
                snapshot_cache_status = "saved"
                print(
                    f"[snapshot-cache] saved date={signal_date.date()} "
                    f"key={snapshot_paths['cache_key']} snapshot_rows={len(screen.snapshot)} selected_rows={len(screen.result)}",
                    flush=True,
                )
    timings["snapshot_sec"] = time.perf_counter() - started
    snapshot_rows = int(getattr(screen.report, "counters", {}).get("snapshot_rows", 0))
    selected_rows = int(len(screen.result)) if isinstance(screen.result, pd.DataFrame) else 0
    _profile_log(signal_date, "snapshot", timings["snapshot_sec"], snapshot_rows=snapshot_rows, selected_rows=selected_rows)

    universe = build_universe_minimal(screen.result, signal_date)
    factors, factor_timings, factor_cache_statuses = _run_factors(universe, signal_date, provider, cfg)
    timings.update(factor_timings)

    regime_result = detect_regime(signal_date, provider, config=cfg.regime_config, state=regime_state)
    risk = None
    if cfg.use_risk_gating:
        risk_started = time.perf_counter()
        risk_cfg = dict(cfg.factor_configs.get("risk", {}))
        risk_cfg.update(
            {
                "risk_price_loader": _risk_loader(provider),
                "risk_index_ticker": cfg.risk_index_code,
                "start_date": risk_cfg.get("start_date", signal_date - pd.Timedelta(days=500)),
                "asof_date": signal_date,
            }
        )
        risk = run_market_risk_gating(signal_date, config=risk_cfg)
        timings["risk_gate_sec"] = time.perf_counter() - risk_started
        _profile_log(signal_date, "risk_gate", timings["risk_gate_sec"])
    allocation_gate = resolve_allocation_gate(
        signal_date=signal_date,
        regime_result=regime_result,
        risk=risk,
        weight_source_mode=cfg.weight_source_mode,
        gui_weights=cfg.composite_weights,
        regime_weight_map=cfg.regime_weight_map,
        default_weights=WeightConfig(),
        allocation_gate_config=cfg.allocation_gate_config,
    )
    weight_resolution = dict(allocation_gate.weight_resolution)
    resolved_weight_config = to_weight_config(weight_resolution["resolved_weights"])
    effective_weights, weight_debug = _effective_composite_weights(resolved_weight_config, factors)
    print(
        "[regime-debug] "
        f"signal_date={signal_date.date()} "
        f"raw_regime={regime_result.get('raw_regime')} "
        f"active_regime={regime_result.get('active_regime')} "
        f"weight_regime={allocation_gate.weight_regime} "
        f"gate_state={allocation_gate.gate_state} "
        f"weight_source={weight_resolution.get('source')} "
        f"json_path={weight_resolution.get('json_path', '')} "
        f"warnings={weight_resolution.get('warnings', [])}",
        flush=True,
    )
    print(
        "[allocation-gate] "
        f"signal_date={signal_date.date()} "
        f"mode={allocation_gate.gate_mode} "
        f"gate_state={allocation_gate.gate_state} "
        f"risk_score={allocation_gate.risk_score_effective} "
        f"gross_multiplier={allocation_gate.gross_multiplier:.2f} "
        f"short_gross_multiplier={allocation_gate.short_gross_multiplier:.2f} "
        f"position_profile={allocation_gate.position_profile} "
        f"override={allocation_gate.risk_override_applied}",
        flush=True,
    )
    print(
        "[composite-debug] "
        f"signal_date={signal_date.date()} "
        f"factor_available={weight_debug['factor_available']} "
        f"module_available={weight_debug['module_available']} "
        f"effective_weights={weight_debug['effective_weights']}",
        flush=True,
    )
    composite_runtime_overrides = {
        "suppress_reversal_when_negative_mgate": bool(cfg.suppress_reversal_when_negative_mgate),
        "f_penalty_when_nonpositive_mgate": float(cfg.f_penalty_when_nonpositive_mgate),
        "score_formula": str(cfg.composite_formula),
    }
    composite_cache_status = "disabled"
    prepared_frame = None
    composite_started = time.perf_counter()
    runtime_config = build_runtime_config(
        strict_freshness_mode=False,
        strict_missing_mode=False,
        runtime_overrides=composite_runtime_overrides,
    )
    if cfg.enable_composite_ready_cache:
        cache_paths = _composite_ready_cache_paths(
            cfg.composite_ready_cache_dir,
            signal_date=signal_date,
            factor_payloads=factors,
            do_preprocess=True,
        )
        prepared_frame = _load_composite_ready_cache(cache_paths)
        if prepared_frame is not None:
            composite_cache_status = "hit"
            print(
                f"[composite-ready-cache] hit date={signal_date.date()} key={cache_paths['cache_key']} rows={len(prepared_frame)}",
                flush=True,
            )
        else:
            composite_cache_status = "miss"
            print(f"[composite-ready-cache] miss date={signal_date.date()} reason=no_matching_key", flush=True)
    if prepared_frame is None:
        standardized_frames: dict[str, pd.DataFrame] = {}
        for factor_name, payload in factors.items():
            composite_name = factor_name
            clean, _ = standardize_factor_frame_for_composite(normalize_factor_name_for_composite(factor_name), payload)
            standardized_frames[normalize_factor_name_for_composite(factor_name)] = clean
        merged = merge_factor_frames(standardized_frames)
        prepared_frame, _ = prepare_composite_ready_frame(
            merged=merged,
            rebalance_date=signal_date,
            config=runtime_config,
            do_preprocess=True,
        )
        if cfg.enable_composite_ready_cache:
            _save_composite_ready_cache(cache_paths, prepared_frame)
            composite_cache_status = "saved"
            print(
                f"[composite-ready-cache] saved date={signal_date.date()} key={cache_paths['cache_key']} rows={len(prepared_frame)}",
                flush=True,
            )
    composite = run_factor_composite_from_prepared_frame(
        prepared_df=prepared_frame,
        rebalance_date=signal_date.strftime("%Y-%m-%d"),
        weights=effective_weights,
        strict_freshness_mode=False,
        strict_missing_mode=False,
        runtime_overrides=composite_runtime_overrides,
    )
    timings["composite_sec"] = time.perf_counter() - composite_started
    _profile_log(signal_date, "composite", timings["composite_sec"])

    composite["factor_availability"] = weight_debug["factor_available"]
    composite["module_availability"] = weight_debug["module_available"]
    composite["effective_weights"] = pd.DataFrame([weight_debug["effective_weights"]])
    composite["weight_resolution"] = pd.DataFrame([{k: v for k, v in weight_resolution.items() if k != "resolved_weights"}])
    composite["resolved_weights"] = pd.DataFrame([weight_resolution["resolved_weights"]])
    composite["allocation_gate"] = pd.DataFrame([allocation_gate.to_dict()])
    composite["regime"] = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "base_regime": regime_result.get("base_regime"),
                "overlay_regime": regime_result.get("overlay_regime"),
                "raw_regime": regime_result.get("raw_regime"),
                "active_regime": regime_result.get("active_regime"),
                **{
                    key: value
                    for key, value in dict(regime_result.get("features", {})).items()
                    if key != "signal_date"
                },
            }
        ]
    )
    scored = composite.get("scored_df")
    if isinstance(scored, pd.DataFrame) and "Score" in scored.columns:
        composite["scored_df"] = _merge_scored_with_universe_metadata(scored, universe)
        scored = composite["scored_df"]
        print(
            "[composite-debug] "
            f"signal_date={signal_date.date()} scored_rows={len(scored)} "
            f"score_non_null={int(scored['Score'].notna().sum())}",
            flush=True,
        )
    gross = float(cfg.max_gross_exposure)
    short_gross = float(cfg.short_gross_exposure)
    gross *= min(float(allocation_gate.gross_multiplier), float(allocation_gate.equity_multiplier))
    gross = min(gross, float(cfg.max_gross_exposure) * max(0.0, 1.0 - float(allocation_gate.cash_floor)))
    short_gross *= float(allocation_gate.short_gross_multiplier)
    if not bool(allocation_gate.allow_new_longs):
        gross = 0.0
    if not bool(allocation_gate.allow_new_shorts):
        short_gross = 0.0

    started = time.perf_counter()
    target_weights = build_target_weights(
        composite["scored_df"],
        cfg.target_count,
        gross,
        enable_short_leg=bool(cfg.enable_short_leg),
        short_target_count=int(cfg.short_target_count),
        short_gross_exposure=short_gross,
        short_construction_mode=str(cfg.short_construction_mode),
        short_weighting_mode=str(cfg.short_weighting_mode),
        short_expansion_mode=str(cfg.short_expansion_mode),
        short_adv_threshold=(float(cfg.short_adv_threshold) if cfg.short_adv_threshold is not None else None),
        short_sector_cap=(float(cfg.short_sector_cap) if cfg.short_sector_cap is not None else None),
        short_min_names=int(cfg.short_min_names),
        short_max_single_weight=float(cfg.short_max_single_weight),
    )
    timings["portfolio_construction_sec"] = time.perf_counter() - started
    _profile_log(
        signal_date,
        "portfolio_construction",
        timings["portfolio_construction_sec"],
        rows=int(len(target_weights)) if isinstance(target_weights, pd.DataFrame) else 0,
    )
    scored = composite.get("scored_df")
    if isinstance(scored, pd.DataFrame):
        execution_codes = _ranked_candidate_codes(scored, cfg.candidate_pool_count)
    else:
        execution_codes = target_weights["code"].astype(str).tolist() if isinstance(target_weights, pd.DataFrame) and "code" in target_weights.columns else []
    execution_date = _next_trade_date(
        provider,
        signal_date,
        cfg.signal_execution_lag_days,
        codes=execution_codes,
        price_col=cfg.execution_price,
        min_price_coverage=float(cfg.min_execution_price_coverage),
        min_price_count=(int(cfg.min_execution_price_count) if cfg.min_execution_price_count is not None else int(cfg.target_count)),
    )
    timings["total_rebalance_sec"] = time.perf_counter() - total_started
    _profile_log(signal_date, "total_rebalance", timings["total_rebalance_sec"])
    return RebalanceResult(
        signal_date,
        signal_date,
        execution_date,
        universe,
        factors,
        composite,
        target_weights,
        risk,
        screen.report,
        timings,
        factor_cache_statuses,
        snapshot_cache_status,
        composite_cache_status,
        regime_result,
        weight_resolution,
        allocation_gate.to_dict(),
    )


def _execution_prices(provider: JpxHistoricalProvider, codes: Iterable[str], trade_date: pd.Timestamp, price_col: str) -> pd.Series:
    frame = provider.price_frame(codes, trade_date, trade_date)
    if frame.empty:
        return pd.Series(dtype=float)
    frame = frame.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last").copy()
    if price_col == "open":
        preferred_col = "adjustment_open"
        fallback_col = "open"
    else:
        preferred_col = "adjustment_close"
        fallback_col = "close"
    out = frame.set_index("code")
    preferred = pd.to_numeric(out[preferred_col], errors="coerce") if preferred_col in out.columns else pd.Series(index=out.index, dtype=float)
    fallback = pd.to_numeric(out[fallback_col], errors="coerce") if fallback_col in out.columns else pd.Series(index=out.index, dtype=float)
    return preferred.combine_first(fallback)


def _execution_price_history(
    provider: JpxHistoricalProvider,
    codes: Iterable[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    price_col: str,
) -> pd.DataFrame:
    code_list = [str(code) for code in codes]
    if not code_list:
        return pd.DataFrame()
    frame = provider.price_frame(code_list, start_date, end_date)
    if frame.empty:
        return pd.DataFrame()
    frame = frame.sort_values(["date", "code"]).drop_duplicates(subset=["date", "code"], keep="last").copy()
    if price_col == "open":
        preferred_col = "adjustment_open"
        fallback_col = "open"
    else:
        preferred_col = "adjustment_close"
        fallback_col = "close"
    preferred = pd.to_numeric(frame[preferred_col], errors="coerce") if preferred_col in frame.columns else pd.Series(index=frame.index, dtype=float)
    fallback = pd.to_numeric(frame[fallback_col], errors="coerce") if fallback_col in frame.columns else pd.Series(index=frame.index, dtype=float)
    frame["execution_price"] = preferred.combine_first(fallback)
    pivot = (
        frame.loc[:, ["date", "code", "execution_price"]]
        .pivot(index="date", columns="code", values="execution_price")
        .sort_index()
    )
    pivot.index = pd.to_datetime(pivot.index).normalize()
    return pivot


def _corporate_action_factors(
    provider: JpxHistoricalProvider,
    codes: Iterable[str],
    start_exclusive: pd.Timestamp | None,
    end_inclusive: pd.Timestamp | None,
) -> pd.Series:
    if start_exclusive is None or end_inclusive is None:
        return pd.Series(dtype=float)
    code_list = [str(code) for code in codes]
    if not code_list or pd.Timestamp(end_inclusive) <= pd.Timestamp(start_exclusive):
        return pd.Series(dtype=float)
    try:
        prices = provider.table("prices", copy=False)
    except Exception:
        return pd.Series(dtype=float)
    work = prices.loc[
        prices["code"].astype(str).isin(code_list)
        & (prices["date"] > pd.Timestamp(start_exclusive).normalize())
        & (prices["date"] <= pd.Timestamp(end_inclusive).normalize())
    ].copy()
    if work.empty or "adjustment_factor" not in work.columns:
        return pd.Series(dtype=float)
    work["adjustment_factor"] = pd.to_numeric(work["adjustment_factor"], errors="coerce")
    work = work.loc[
        work["adjustment_factor"].notna()
        & work["adjustment_factor"].gt(0)
        & ~np.isclose(work["adjustment_factor"], 1.0)
    ].copy()
    if work.empty:
        return pd.Series(dtype=float)
    return work.groupby(work["code"].astype(str))["adjustment_factor"].prod().rename("corporate_action_factor")


def _corporate_action_history(
    provider: JpxHistoricalProvider,
    codes: Iterable[str],
    start_exclusive: pd.Timestamp | None,
    end_inclusive: pd.Timestamp | None,
) -> pd.DataFrame:
    if start_exclusive is None or end_inclusive is None:
        return pd.DataFrame(columns=["date", "code", "adjustment_factor"])
    code_list = [str(code) for code in codes]
    if not code_list or pd.Timestamp(end_inclusive) <= pd.Timestamp(start_exclusive):
        return pd.DataFrame(columns=["date", "code", "adjustment_factor"])
    try:
        prices = provider.table("prices", copy=False)
    except Exception:
        return pd.DataFrame(columns=["date", "code", "adjustment_factor"])
    work = prices.loc[
        prices["code"].astype(str).isin(code_list)
        & (prices["date"] > pd.Timestamp(start_exclusive).normalize())
        & (prices["date"] <= pd.Timestamp(end_inclusive).normalize())
    ].copy()
    if work.empty or "adjustment_factor" not in work.columns:
        return pd.DataFrame(columns=["date", "code", "adjustment_factor"])
    work["adjustment_factor"] = pd.to_numeric(work["adjustment_factor"], errors="coerce")
    work = work.loc[
        work["adjustment_factor"].notna()
        & work["adjustment_factor"].gt(0)
        & ~np.isclose(work["adjustment_factor"], 1.0)
    , ["date", "code", "adjustment_factor"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["date", "code", "adjustment_factor"])
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["code"] = work["code"].astype(str)
    return work.sort_values(["date", "code"]).reset_index(drop=True)


def _apply_corporate_actions(
    holdings: dict[str, int],
    cost_basis: dict[str, float],
    action_factors: pd.Series,
    action_date: pd.Timestamp,
    rows: list[dict[str, Any]],
) -> None:
    if action_factors.empty:
        return
    for code, factor in action_factors.items():
        normalized = str(code)
        if normalized not in holdings:
            continue
        current_qty = int(holdings.get(normalized, 0))
        if current_qty == 0:
            continue
        factor_value = float(factor)
        if not np.isfinite(factor_value) or factor_value <= 0 or np.isclose(factor_value, 1.0):
            continue
        adjusted_qty = int(round(current_qty / factor_value))
        current_cost_basis = float(cost_basis.get(normalized, np.nan))
        adjusted_cost_basis = current_cost_basis * factor_value if np.isfinite(current_cost_basis) else np.nan
        holdings[normalized] = adjusted_qty
        if holdings[normalized] == 0:
            holdings.pop(normalized, None)
            cost_basis.pop(normalized, None)
        elif np.isfinite(adjusted_cost_basis) and adjusted_cost_basis > 0:
            cost_basis[normalized] = float(adjusted_cost_basis)
        rows.append(
            {
                "date": pd.Timestamp(action_date).normalize(),
                "code": normalized,
                "factor": factor_value,
                "quantity_before": current_qty,
                "quantity_after": adjusted_qty,
                "cost_basis_before": current_cost_basis,
                "cost_basis_after": adjusted_cost_basis,
            }
        )            


def _apply_stop_loss_cooldown_filter(
    target_weights: pd.DataFrame,
    cooldown_rebalances_remaining: dict[str, int],
    max_gross_exposure: float,
) -> tuple[pd.DataFrame, int]:
    if target_weights.empty or "code" not in target_weights.columns or not cooldown_rebalances_remaining:
        return target_weights, 0
    blocked_codes = set(cooldown_rebalances_remaining)
    blocked_count = int(target_weights["code"].astype(str).isin(blocked_codes).sum())
    if blocked_count <= 0:
        return target_weights, 0
    out = target_weights.loc[~target_weights["code"].astype(str).isin(blocked_codes)].copy()
    if not out.empty and "target_weight" in out.columns:
        original = pd.to_numeric(target_weights["target_weight"], errors="coerce").fillna(0.0)
        original_long_gross = float(original[original > 0].sum())
        original_short_gross = float((-original[original < 0]).sum())
        target_long_gross = min(float(max_gross_exposure), original_long_gross) if original_long_gross > 0 else 0.0
        out = _rescale_target_weights_to_gross(out, target_long_gross, original_short_gross)
    return out, blocked_count


def _portfolio_value(holdings: dict[str, int], prices: pd.Series, cash: float) -> float:
    value = float(cash)
    for code, qty in holdings.items():
        px = prices.get(code, np.nan)
        if pd.notna(px):
            value += int(qty) * float(px)
    return value


def _stop_loss_exit_codes(
    holdings: dict[str, int],
    cost_basis: dict[str, float],
    prices: pd.Series,
    stop_loss_pct: float,
) -> dict[str, dict[str, float]]:
    triggered: dict[str, dict[str, float]] = {}
    if stop_loss_pct <= 0:
        return triggered
    threshold = 1.0 - float(stop_loss_pct)
    for code, qty in holdings.items():
        px = float(prices.get(code, np.nan))
        basis = float(cost_basis.get(code, np.nan))
        if not np.isfinite(px) or px <= 0 or not np.isfinite(basis) or basis <= 0:
            continue
        qty_value = int(qty)
        if qty_value > 0 and px <= basis * threshold:
            triggered[code] = {
                "price": px,
                "cost_basis": basis,
                "drawdown_pct": (px / basis) - 1.0,
                "side": "LONG",
            }
        elif qty_value < 0 and px >= basis * (1.0 + float(stop_loss_pct)):
            triggered[code] = {
                "price": px,
                "cost_basis": basis,
                "drawdown_pct": (basis / px) - 1.0,
                "side": "SHORT",
            }
    return triggered


def _stop_loss_monitor_dates(
    provider: JpxHistoricalProvider,
    start_exclusive: pd.Timestamp | None,
    end_exclusive: pd.Timestamp | None,
    frequency: str,
) -> list[pd.Timestamp]:
    mode = str(frequency or "monthly").lower()
    if mode not in {"weekly", "daily"} or start_exclusive is None or end_exclusive is None:
        return []
    prices = provider.table("prices", copy=False)
    dates = pd.Series(pd.to_datetime(prices["date"].dropna().unique())).sort_values()
    dates = dates.loc[
        dates.gt(pd.Timestamp(start_exclusive).normalize())
        & dates.lt(pd.Timestamp(end_exclusive).normalize())
    ]
    if dates.empty:
        return []
    if mode == "daily":
        return [pd.Timestamp(x).normalize() for x in dates.tolist()]
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    return [pd.Timestamp(x).normalize() for x in frame.groupby("week")["date"].max().tolist()]


def _execute_stop_loss_exits(
    *,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    holdings: dict[str, int],
    cost_basis: dict[str, float],
    prices: pd.Series,
    cfg: BacktestConfig,
    cash: float,
    trade_rows: list[dict[str, Any]],
) -> tuple[float, int, int, int, float, set[str]]:
    stop_loss_exits = _stop_loss_exit_codes(holdings, cost_basis, prices, cfg.stop_loss_pct)
    if not stop_loss_exits:
        return cash, 0, 0, 0, 0.0, set()
    cost_rate = (float(cfg.transaction_cost_bps) + float(cfg.slippage_bps)) / 10000.0
    traded_value = 0.0
    trade_rows_count = 0
    order_rows_count = 0
    stop_loss_exit_rows = 0
    exited_codes: set[str] = set()
    for code, detail in stop_loss_exits.items():
        current_qty = int(holdings.get(code, 0))
        if current_qty == 0:
            continue
        px = float(detail["price"])
        exit_qty = -current_qty
        notional = float(exit_qty) * px
        fees = abs(notional) * cost_rate
        cash -= notional + fees
        traded_value += abs(notional)
        order_rows_count += 1
        trade_rows_count += 1
        stop_loss_exit_rows += 1
        trade_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "code": code,
                "quantity": exit_qty,
                "price": px,
                "notional": notional,
                "cost": fees,
                "reason": f"stop_loss_{str(detail.get('side', 'LONG')).lower()}",
                "cost_basis": float(detail["cost_basis"]),
                "drawdown_pct": float(detail["drawdown_pct"]),
            }
        )
        holdings.pop(code, None)
        cost_basis.pop(code, None)
        exited_codes.add(code)
    return cash, stop_loss_exit_rows, order_rows_count, trade_rows_count, traded_value, exited_codes


def _update_cost_basis_after_trade(
    cost_basis: dict[str, float],
    code: str,
    current_qty: int,
    desired_qty: int,
    delta: int,
    px: float,
) -> None:
    if desired_qty == 0:
        cost_basis.pop(code, None)
        return
    previous_basis = float(cost_basis.get(code, float(px)))
    if current_qty == 0 or current_qty * desired_qty < 0:
        cost_basis[code] = float(px)
        return
    if current_qty > 0 and delta > 0:
        total_cost = current_qty * previous_basis + delta * float(px)
        cost_basis[code] = float(total_cost / desired_qty)
        return
    if current_qty < 0 and delta < 0:
        total_cost = abs(current_qty) * previous_basis + abs(delta) * float(px)
        cost_basis[code] = float(total_cost / abs(desired_qty))
        return
    if code not in cost_basis:
        cost_basis[code] = float(px)


def _run_interim_stop_loss_checks(
    *,
    provider: JpxHistoricalProvider,
    holdings: dict[str, int],
    cost_basis: dict[str, float],
    cash: float,
    start_exclusive: pd.Timestamp | None,
    end_exclusive: pd.Timestamp | None,
    cfg: BacktestConfig,
    nav_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    corporate_action_rows: list[dict[str, Any]],
) -> tuple[float, pd.Timestamp | None, dict[str, int]]:
    monitor_dates = _stop_loss_monitor_dates(provider, start_exclusive, end_exclusive, cfg.stop_loss_check_frequency)
    if not monitor_dates:
        return cash, start_exclusive, {"check_dates": 0, "exit_rows": 0, "order_rows": 0, "trade_rows": 0, "exited_codes": []}
    initial_holdings = sorted(str(code) for code, qty in holdings.items() if int(qty) != 0)
    if not initial_holdings:
        return cash, start_exclusive, {"check_dates": 0, "exit_rows": 0, "order_rows": 0, "trade_rows": 0, "exited_codes": []}
    history_start = pd.Timestamp(monitor_dates[0]).normalize()
    history_end = pd.Timestamp(monitor_dates[-1]).normalize()
    price_history = _execution_price_history(provider, initial_holdings, history_start, history_end, cfg.execution_price)
    action_history = _corporate_action_history(provider, initial_holdings, start_exclusive, history_end)
    checkpoint = pd.Timestamp(start_exclusive).normalize() if start_exclusive is not None else None
    stats = {"check_dates": 0, "exit_rows": 0, "order_rows": 0, "trade_rows": 0, "exited_codes": []}
    for check_date in monitor_dates:
        stats["check_dates"] += 1
        if holdings and checkpoint is not None:
            if action_history.empty:
                action_factors = pd.Series(dtype=float)
            else:
                active_codes = set(str(code) for code in holdings)
                interval_actions = action_history.loc[
                    action_history["code"].isin(active_codes)
                    & action_history["date"].gt(pd.Timestamp(checkpoint).normalize())
                    & action_history["date"].le(pd.Timestamp(check_date).normalize())
                ]
                action_factors = (
                    interval_actions.groupby("code")["adjustment_factor"].prod().rename("corporate_action_factor")
                    if not interval_actions.empty
                    else pd.Series(dtype=float)
                )
            _apply_corporate_actions(holdings, cost_basis, action_factors, check_date, corporate_action_rows)
        checkpoint = check_date
        if not holdings:
            continue
        if price_history.empty or check_date not in price_history.index:
            continue
        prices = pd.to_numeric(price_history.loc[check_date].reindex(sorted(holdings)), errors="coerce")
        prices = prices.dropna()
        if prices.empty:
            continue
        nav_before = _portfolio_value(holdings, prices, cash)
        cash, exit_rows, order_rows, trade_count, traded_value, exited_codes = _execute_stop_loss_exits(
            signal_date=check_date,
            execution_date=check_date,
            holdings=holdings,
            cost_basis=cost_basis,
            prices=prices,
            cfg=cfg,
            cash=cash,
            trade_rows=trade_rows,
        )
        stats["exit_rows"] += exit_rows
        stats["order_rows"] += order_rows
        stats["trade_rows"] += trade_count
        stats["exited_codes"].extend(sorted(exited_codes))
        if exit_rows > 0:
            nav_after = _portfolio_value(holdings, prices, cash)
            turnover = traded_value / nav_before if nav_before else np.nan
            nav_rows.append({"date": check_date, "nav": nav_after, "cash": cash, "turnover": turnover})
    return cash, checkpoint, stats


def _normalised_benchmark_frame(series: pd.Series, name: str, source: str, initial_capital: float) -> pd.DataFrame:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.DataFrame(columns=["date", "benchmark", "close", "nav", "return", "source"])
    base = float(clean.iloc[0])
    if base <= 0:
        return pd.DataFrame(columns=["date", "benchmark", "close", "nav", "return", "source"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(clean.index).normalize(),
            "benchmark": name,
            "close": clean.to_numpy(dtype=float),
            "source": source,
        }
    )
    out["nav"] = out["close"] / base * float(initial_capital)
    out["return"] = out["nav"].pct_change().fillna(out["nav"] / float(initial_capital) - 1.0)
    return out


def _sp500_yfinance_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Download S&P 500 benchmark history.

    yfinance is intentionally isolated to this benchmark helper.  The Japan
    equity backtest data path still uses the historical provider/cache layer.
    """

    try:
        import yfinance as yf
    except Exception:
        return pd.Series(dtype=float, name=ticker)
    try:
        # Add one calendar day because yfinance treats end as exclusive.
        hist = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return pd.Series(dtype=float, name=ticker)
    if hist is None or hist.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(hist.columns, pd.MultiIndex):
        price_frame = hist.xs(ticker, axis=1, level=-1, drop_level=True) if ticker in hist.columns.get_level_values(-1) else hist.droplevel(-1, axis=1)
    else:
        price_frame = hist
    col = "Adj Close" if "Adj Close" in price_frame.columns else "Close"
    if col not in price_frame.columns:
        return pd.Series(dtype=float, name=ticker)
    close = pd.to_numeric(price_frame[col], errors="coerce")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = ticker
    return close.dropna().sort_index()


def build_benchmark_nav(
    provider: JpxHistoricalProvider,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: BacktestConfig | dict[str, Any] | None = None,
    initial_capital: float = 100_000_000.0,
) -> pd.DataFrame:
    """Build normalized TOPIX/Nikkei/local benchmark NAV plus optional S&P 500.

    Local benchmarks are read from ``index_prices`` through the historical
    provider.  S&P 500 may be downloaded from yfinance when
    ``include_sp500_benchmark`` is true.
    """

    cfg = _cfg(config)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    frames: list[pd.DataFrame] = []
    for code in cfg.benchmark_index_codes:
        try:
            series = provider.load_index_close(str(code), start_ts, end_ts, {})
        except Exception:
            continue
        if series.empty:
            continue
        frames.append(_normalised_benchmark_frame(series, str(code), "local_index_prices", initial_capital))
    if cfg.include_sp500_benchmark:
        sp500 = _sp500_yfinance_close(cfg.sp500_yfinance_ticker, start_ts, end_ts)
        if not sp500.empty:
            frames.append(_normalised_benchmark_frame(sp500, "SP500", f"yfinance:{cfg.sp500_yfinance_ticker}", initial_capital))
    if not frames:
        return pd.DataFrame(columns=["date", "benchmark", "close", "nav", "return", "source"])
    return pd.concat(frames, ignore_index=True).sort_values(["benchmark", "date"]).reset_index(drop=True)


def run_backtest(
    rebalance_dates: Iterable[str | pd.Timestamp],
    provider: JpxHistoricalProvider,
    rules: ScreenRules | None = None,
    config: BacktestConfig | dict[str, Any] | None = None,
    initial_capital: float = 100_000_000.0,
) -> dict[str, pd.DataFrame | list[RebalanceResult]]:
    """Run a historical backtest over supplied signal dates."""

    cfg = _cfg(config)
    dates = [pd.Timestamp(d).normalize() for d in rebalance_dates]
    holdings: dict[str, int] = {}
    cost_basis: dict[str, float] = {}
    cash = float(initial_capital)
    nav_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    corporate_action_rows: list[dict[str, Any]] = []
    rebalance_debug_rows: list[dict[str, Any]] = []
    missing_price_rows: list[dict[str, Any]] = []
    post_score_adjustment_rows: list[dict[str, Any]] = []
    sector_exposure_rows: list[dict[str, Any]] = []
    holding_age_debug_rows: list[dict[str, Any]] = []
    top_holdings_rows: list[dict[str, Any]] = []
    results: list[RebalanceResult] = []
    last_execution_date: pd.Timestamp | None = None
    pending_stop_loss_cash_penalty = False
    cooldown_rebalances_remaining: dict[str, int] = {}
    holding_age_months: dict[str, int] = {}
    regime_state = RegimeState()

    for signal_date in dates:
        cooldown_rebalances_remaining = {
            code: int(remaining) - 1
            for code, remaining in cooldown_rebalances_remaining.items()
            if int(remaining) - 1 > 0
        }
        result = run_one_rebalance(signal_date, provider, rules, cfg, regime_state=regime_state)
        if isinstance(result.regime, dict) and isinstance(result.regime.get("state"), RegimeState):
            regime_state = result.regime["state"]
        results.append(result)
        selected_rows = int(len(result.universe)) if isinstance(result.universe, pd.DataFrame) else 0
        target_weights = result.target_weights.copy() if isinstance(result.target_weights, pd.DataFrame) else pd.DataFrame()
        pre_price_target_weights = target_weights.copy()
        weight_values = (
            pd.to_numeric(target_weights["target_weight"], errors="coerce")
            if "target_weight" in target_weights.columns
            else pd.Series(dtype=float)
        )
        final_weight_rows = int(len(target_weights))
        positive_weight_rows = int((weight_values > 0).sum())
        negative_weight_rows = int((weight_values < 0).sum())
        score_non_null_rows = 0
        scored_df = result.composite.get("scored_df") if isinstance(result.composite, dict) else None
        if isinstance(scored_df, pd.DataFrame) and "Score" in scored_df.columns:
            score_non_null_rows = int(pd.to_numeric(scored_df["Score"], errors="coerce").notna().sum())

        debug_row: dict[str, Any] = {
            "rebalance_date": signal_date,
            "signal_date": signal_date,
            "execution_date": result.execution_date,
            "universe_count": int(len(result.universe)) if isinstance(result.universe, pd.DataFrame) else 0,
            "snapshot_count": int(getattr(getattr(result, "report", None), "counters", {}).get("snapshot_rows", 0)),
            "selected_rows": selected_rows,
            "score_non_null_rows": score_non_null_rows,
            "final_weight_rows": final_weight_rows,
            "positive_weight_rows": positive_weight_rows,
            "negative_weight_rows": negative_weight_rows,
            "order_rows": 0,
            "trade_rows": 0,
            "holding_rows": int(len(holdings)),
            "execution_price_rows": 0,
            "valid_execution_price_rows": 0,
            "desired_positive_qty_rows": 0,
            "desired_negative_qty_rows": 0,
            "zero_stage": "ok",
            "snapshot_cache_status": getattr(result, "snapshot_cache_status", "disabled"),
            "composite_ready_cache_status": getattr(result, "composite_ready_cache_status", "disabled"),
            "cooldown_blocked_rows": 0,
            "execution_candidate_rows": 0,
            "execution_candidate_price_rows": 0,
            "execution_candidate_valid_price_rows": 0,
            "pre_price_target_rows": int(len(pre_price_target_weights)),
            "post_price_target_rows": int(len(target_weights)),
            "missing_candidate_price_rows": 0,
            "missing_pre_target_price_rows": 0,
            "target_rebuilt_after_price_lookup": False,
        }
        for timing_name, timing_value in getattr(result, "timings", {}).items():
            debug_row[f"timing_{timing_name}"] = float(timing_value)
        for factor_name, cache_status in getattr(result, "factor_cache_statuses", {}).items():
            debug_row[f"{factor_name}_cache_status"] = cache_status
        debug_row["stop_loss_monitor_check_dates"] = 0
        debug_row["stop_loss_monitor_exit_rows"] = 0
        debug_row["stop_loss_cash_penalty_applied"] = False
        debug_row["target_cash_weight"] = 0.0
        debug_row["regime_raw"] = result.regime.get("raw_regime") if isinstance(result.regime, dict) else ""
        debug_row["regime_active"] = result.regime.get("active_regime") if isinstance(result.regime, dict) else ""
        debug_row["weight_source_mode"] = result.weight_resolution.get("source_mode") if isinstance(result.weight_resolution, dict) else ""
        debug_row["weight_source"] = result.weight_resolution.get("source") if isinstance(result.weight_resolution, dict) else ""
        debug_row["gate_state"] = result.allocation_gate.get("gate_state") if isinstance(result.allocation_gate, dict) else ""
        debug_row["gate_mode"] = result.allocation_gate.get("gate_mode") if isinstance(result.allocation_gate, dict) else ""
        debug_row["gate_weight_regime"] = result.allocation_gate.get("weight_regime") if isinstance(result.allocation_gate, dict) else ""
        debug_row["gate_risk_score_effective"] = result.allocation_gate.get("risk_score_effective") if isinstance(result.allocation_gate, dict) else np.nan
        debug_row["gate_risk_override_applied"] = result.allocation_gate.get("risk_override_applied") if isinstance(result.allocation_gate, dict) else False
        debug_row["gate_gross_multiplier"] = result.allocation_gate.get("gross_multiplier") if isinstance(result.allocation_gate, dict) else 1.0

        if result.execution_date is None:
            debug_row["zero_stage"] = "execution date"
            rebalance_debug_rows.append(debug_row)
            print(
                "[rebalance-debug] "
                f"signal_date={signal_date.date()} execution_date=None "
                f"selected_rows={selected_rows} final_weight_rows={final_weight_rows} "
                f"positive_weight_rows={positive_weight_rows} order_rows=0 trade_rows=0 "
                f"holding_rows={len(holdings)} zero_stage=execution date",
                flush=True,
            )
            continue
        if "code" in target_weights.columns:
            target_weights["code"] = target_weights["code"].map(normalize_code).astype(str)
            pre_price_target_weights = target_weights.copy()
            target_codes = set(target_weights["code"])
        else:
            target_codes = set()
        monitor_stats = {"check_dates": 0, "exit_rows": 0, "order_rows": 0, "trade_rows": 0}
        checkpoint_date = last_execution_date
        if holdings and last_execution_date is not None:
            cash, checkpoint_date, monitor_stats = _run_interim_stop_loss_checks(
                provider=provider,
                holdings=holdings,
                cost_basis=cost_basis,
                cash=cash,
                start_exclusive=last_execution_date,
                end_exclusive=result.execution_date,
                cfg=cfg,
                nav_rows=nav_rows,
                trade_rows=trade_rows,
                corporate_action_rows=corporate_action_rows,
            )
            if int(cfg.stop_loss_cooldown_rebalances) > 0:
                for code in monitor_stats.get("exited_codes", []):
                    cooldown_rebalances_remaining[str(code)] = int(cfg.stop_loss_cooldown_rebalances)
            action_factors = _corporate_action_factors(provider, holdings.keys(), checkpoint_date, result.execution_date)
            _apply_corporate_actions(holdings, cost_basis, action_factors, result.execution_date, corporate_action_rows)
        scored_df = result.composite.get("scored_df") if isinstance(result.composite, dict) else None
        allocation_gate_payload = result.allocation_gate if isinstance(result.allocation_gate, dict) else {}
        gross = float(cfg.max_gross_exposure)
        gross *= min(
            float(allocation_gate_payload.get("gross_multiplier", 1.0)),
            float(allocation_gate_payload.get("equity_multiplier", 1.0)),
        )
        gross = min(gross, float(cfg.max_gross_exposure) * max(0.0, 1.0 - float(allocation_gate_payload.get("cash_floor", 0.0))))
        short_gross = float(cfg.short_gross_exposure) * float(allocation_gate_payload.get("short_gross_multiplier", 1.0))
        if not bool(allocation_gate_payload.get("allow_new_longs", True)):
            gross = 0.0
        if not bool(allocation_gate_payload.get("allow_new_shorts", True)):
            short_gross = 0.0
        debug_row["long_gross_target"] = float(gross)
        debug_row["short_gross_target"] = float(short_gross)
        current_holdings_at_signal = set(str(code) for code in holdings)
        blocked_codes_pre_price = set(str(code) for code in cooldown_rebalances_remaining)
        sector_bias_regime = _resolve_sector_bias_regime(
            post_cfg=cfg.post_score_adjustments,
            regime_payload=result.regime if isinstance(result.regime, dict) else {},
            allocation_gate_payload=allocation_gate_payload,
        )
        pre_selection = _build_post_score_adjusted_targets(
            scored_df,
            provider=provider,
            signal_date=signal_date,
            target_count=cfg.target_count,
            gross_exposure=gross,
            current_holdings=current_holdings_at_signal,
            holding_ages=holding_age_months,
            blocked_codes=blocked_codes_pre_price,
            prices=None,
            rescale_partial=bool(cfg.rescale_partial_target_weights),
            post_cfg=cfg.post_score_adjustments,
            constraint_cfg=cfg.portfolio_constraints,
            selection_regime=sector_bias_regime,
            enable_short_leg=bool(cfg.enable_short_leg),
            short_gross_exposure=short_gross,
            short_target_count=int(cfg.short_target_count),
            short_construction_mode=str(cfg.short_construction_mode),
            short_weighting_mode=str(cfg.short_weighting_mode),
            short_expansion_mode=str(cfg.short_expansion_mode),
            short_adv_threshold=(float(cfg.short_adv_threshold) if cfg.short_adv_threshold is not None else None),
            short_sector_cap=(float(cfg.short_sector_cap) if cfg.short_sector_cap is not None else None),
            short_min_names=int(cfg.short_min_names),
            short_max_single_weight=float(cfg.short_max_single_weight),
        )
        target_weights = pre_selection["target_weights"].copy()
        pre_price_target_weights = target_weights.copy()
        target_codes = set(target_weights["code"]) if "code" in target_weights.columns else set()
        candidate_codes = (
            pre_selection["debug"]
            .sort_values(["final_rank_before_constraints", "code"], ascending=[True, True])["code"]
            .astype(str)
            .tolist()
            if isinstance(pre_selection.get("debug"), pd.DataFrame) and not pre_selection["debug"].empty
            else []
        )
        debug_row["cooldown_blocked_rows"] = int(
            pre_selection["debug"]["hard_block_reason"].astype(str).eq("cooldown_or_stop_block").sum()
        ) if isinstance(pre_selection.get("debug"), pd.DataFrame) and not pre_selection["debug"].empty else 0
        debug_row["execution_candidate_rows"] = int(len(candidate_codes))
        lookup_universe = set(holdings)
        lookup_universe.update(candidate_codes if candidate_codes else target_codes)
        codes = sorted(lookup_universe)
        started = time.perf_counter()
        prices = _execution_prices(provider, codes, result.execution_date, cfg.execution_price)
        execution_price_lookup_sec = time.perf_counter() - started
        result.timings["execution_price_lookup_sec"] = execution_price_lookup_sec
        _profile_log(signal_date, "execution_price_lookup", execution_price_lookup_sec, rows=int(len(prices)))
        debug_row["execution_price_rows"] = int(len(prices))
        debug_row["valid_execution_price_rows"] = int((pd.to_numeric(prices, errors="coerce") > 0).sum())
        candidate_price_view = pd.to_numeric(prices.reindex(candidate_codes), errors="coerce")
        pre_target_codes = pre_price_target_weights["code"].astype(str).tolist() if "code" in pre_price_target_weights.columns else []
        pre_target_price_view = pd.to_numeric(prices.reindex(pre_target_codes), errors="coerce")
        debug_row["execution_candidate_price_rows"] = int(candidate_price_view.notna().sum())
        debug_row["execution_candidate_valid_price_rows"] = int(candidate_price_view.gt(0).sum())
        debug_row["missing_candidate_price_rows"] = int((~candidate_price_view.gt(0)).sum()) if len(candidate_codes) else 0
        debug_row["missing_pre_target_price_rows"] = int((~pre_target_price_view.gt(0)).sum()) if len(pre_target_codes) else 0
        if bool(cfg.record_missing_execution_prices):
            candidate_set = set(candidate_codes)
            pre_target_set = set(pre_target_codes)
            final_target_set_preview = set(target_weights["code"].astype(str)) if "code" in target_weights.columns else set()
            holdings_set = set(str(code) for code in holdings)
            for code, px in candidate_price_view.items():
                valid_px = bool(pd.notna(px) and np.isfinite(px) and float(px) > 0)
                if valid_px:
                    continue
                missing_price_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": result.execution_date,
                        "code": str(code),
                        "reason": "missing_candidate_execution_price",
                        "in_candidate_pool": True,
                        "in_pre_price_target": str(code) in pre_target_set,
                        "in_final_target": str(code) in final_target_set_preview,
                        "in_holdings": str(code) in holdings_set,
                        "price_found": bool(pd.notna(px)),
                        "valid_price": False,
                    }
                )
            for code, px in pre_target_price_view.items():
                valid_px = bool(pd.notna(px) and np.isfinite(px) and float(px) > 0)
                if valid_px:
                    continue
                missing_price_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": result.execution_date,
                        "code": str(code),
                        "reason": "missing_pre_price_target_execution_price",
                        "in_candidate_pool": str(code) in candidate_set,
                        "in_pre_price_target": True,
                        "in_final_target": str(code) in final_target_set_preview,
                        "in_holdings": str(code) in holdings_set,
                        "price_found": bool(pd.notna(px)),
                        "valid_price": False,
                    }
                )
        apply_stop_loss_cash_penalty = bool(monitor_stats["exit_rows"] > 0 or pending_stop_loss_cash_penalty)
        stop_loss_exit_rows = 0
        cost_rate = (float(cfg.transaction_cost_bps) + float(cfg.slippage_bps)) / 10000.0
        traded_value = 0.0
        trade_rows_this_rebalance = 0
        order_rows_this_rebalance = 0
        desired_positive_qty_rows = 0
        desired_negative_qty_rows = 0
        stopped_codes = set(_stop_loss_exit_codes(holdings, cost_basis, prices, cfg.stop_loss_pct))
        cash, stop_loss_exit_rows, stop_loss_order_rows, stop_loss_trade_rows, stop_loss_traded_value, stop_loss_exited_codes = _execute_stop_loss_exits(
            signal_date=signal_date,
            execution_date=result.execution_date,
            holdings=holdings,
            cost_basis=cost_basis,
            prices=prices,
            cfg=cfg,
            cash=cash,
            trade_rows=trade_rows,
        )
        order_rows_this_rebalance += stop_loss_order_rows
        trade_rows_this_rebalance += stop_loss_trade_rows
        traded_value += stop_loss_traded_value
        if int(cfg.stop_loss_cooldown_rebalances) > 0:
            for code in stop_loss_exited_codes:
                cooldown_rebalances_remaining[str(code)] = int(cfg.stop_loss_cooldown_rebalances)
        blocked_codes_final = set(str(code) for code in cooldown_rebalances_remaining)
        blocked_codes_final.update(str(code) for code in stopped_codes)
        final_selection = _build_post_score_adjusted_targets(
            scored_df,
            provider=provider,
            signal_date=signal_date,
            target_count=cfg.target_count,
            gross_exposure=gross,
            current_holdings=current_holdings_at_signal,
            holding_ages=holding_age_months,
            blocked_codes=blocked_codes_final,
            prices=prices,
            rescale_partial=bool(cfg.rescale_partial_target_weights),
            post_cfg=cfg.post_score_adjustments,
            constraint_cfg=cfg.portfolio_constraints,
            selection_regime=sector_bias_regime,
            enable_short_leg=bool(cfg.enable_short_leg),
            short_gross_exposure=short_gross,
            short_target_count=int(cfg.short_target_count),
            short_construction_mode=str(cfg.short_construction_mode),
            short_weighting_mode=str(cfg.short_weighting_mode),
            short_expansion_mode=str(cfg.short_expansion_mode),
            short_adv_threshold=(float(cfg.short_adv_threshold) if cfg.short_adv_threshold is not None else None),
            short_sector_cap=(float(cfg.short_sector_cap) if cfg.short_sector_cap is not None else None),
            short_min_names=int(cfg.short_min_names),
            short_max_single_weight=float(cfg.short_max_single_weight),
        )
        target_weights = final_selection["target_weights"].copy()
        debug_row["target_rebuilt_after_price_lookup"] = True
        combined_gross_for_penalty = float(gross) + float(short_gross)
        if apply_stop_loss_cash_penalty and not target_weights.empty and "target_weight" in target_weights.columns:
            target_weights = _apply_cash_penalty_to_target_weights(
                target_weights,
                max_gross_exposure=combined_gross_for_penalty,
                cash_penalty_pct=cfg.stop_loss_cash_penalty_pct,
            )
            debug_row["stop_loss_cash_penalty_applied"] = True
            debug_row["target_cash_weight"] = float(cfg.stop_loss_cash_penalty_pct)
        nav_before = _portfolio_value(holdings, prices, cash)
        debug_row["post_price_target_rows"] = int(len(target_weights))
        target_codes = set(target_weights["code"]) if "code" in target_weights.columns else set()
        final_debug = final_selection.get("debug", pd.DataFrame()).copy() if isinstance(final_selection, dict) else pd.DataFrame()
        if not final_debug.empty:
            weight_lookup = (
                target_weights.set_index("code")["target_weight"].to_dict()
                if not target_weights.empty and "code" in target_weights.columns and "target_weight" in target_weights.columns
                else {}
            )
            final_debug["final_target_weight"] = final_debug["code"].astype(str).map(weight_lookup).fillna(0.0)
            post_score_adjustment_rows.extend(final_debug.to_dict(orient="records"))
            top_holdings = final_debug[final_debug["selected_final_top15"].fillna(False)].copy()
            if not top_holdings.empty:
                top_holdings = top_holdings.sort_values(["final_rank_after_constraints", "code"], ascending=[True, True])
                for _, holding in top_holdings.iterrows():
                    top_holdings_rows.append(
                        {
                            "signal_date": holding.get("signal_date"),
                            "code": holding.get("code"),
                            "sector": holding.get("sector"),
                            "weight": holding.get("final_target_weight", 0.0),
                            "Q": holding.get("Q"),
                            "V": holding.get("V"),
                            "M": holding.get("M"),
                            "M_gate": holding.get("M_gate"),
                            "I": holding.get("I"),
                            "C": holding.get("C"),
                            "Score": holding.get("original_score"),
                            "adjusted_score": holding.get("adjusted_score"),
                            "final_rank": holding.get("final_rank_after_constraints"),
                        }
                    )
        holding_age_debug = final_selection.get("holding_age_debug", pd.DataFrame()) if isinstance(final_selection, dict) else pd.DataFrame()
        if isinstance(holding_age_debug, pd.DataFrame) and not holding_age_debug.empty:
            holding_age_debug_rows.extend(holding_age_debug.to_dict(orient="records"))
        if not target_weights.empty and "code" in target_weights.columns and "target_weight" in target_weights.columns:
            sector_lookup = {}
            if not final_debug.empty and "sector" in final_debug.columns:
                sector_lookup = final_debug.set_index("code")["sector"].to_dict()
            sector_df = target_weights.copy()
            sector_df["sector"] = sector_df["code"].astype(str).map(sector_lookup).fillna("UNKNOWN")
            sector_df = (
                sector_df.groupby("sector", dropna=False)
                .agg(names_count=("code", "count"), sector_weight=("target_weight", "sum"))
                .reset_index()
            )
            sector_df.insert(0, "signal_date", pd.Timestamp(signal_date).normalize())
            sector_exposure_rows.extend(sector_df.to_dict(orient="records"))
        if target_codes and "target_weight" in target_weights.columns:
            desired_qty_map, target_value = _allocate_target_quantities(
                target_weights=target_weights,
                prices=prices,
                nav_before=nav_before,
                lot_size=cfg.lot_size,
                round_lots=cfg.round_lots,
            )
        else:
            desired_qty_map = pd.Series(dtype=int)
            target_value = pd.Series(dtype=float)
        current_weight_values = (
            pd.to_numeric(target_weights["target_weight"], errors="coerce")
            if "target_weight" in target_weights.columns
            else pd.Series(dtype=float)
        )
        final_weight_rows = int(len(target_weights))
        positive_weight_rows = int((current_weight_values > 0).sum()) if len(current_weight_values) else 0
        negative_weight_rows = int((current_weight_values < 0).sum()) if len(current_weight_values) else 0
        debug_row["final_weight_rows"] = final_weight_rows
        debug_row["positive_weight_rows"] = positive_weight_rows
        debug_row["negative_weight_rows"] = negative_weight_rows
        debug_row["final_long_rows"] = positive_weight_rows
        debug_row["final_short_rows"] = negative_weight_rows
        started = time.perf_counter()
        for code in codes:
            if code in stopped_codes:
                continue
            px = prices.get(code, np.nan)
            if pd.isna(px) or px <= 0:
                continue
            desired_qty = int(pd.to_numeric(desired_qty_map.get(code, 0), errors="coerce") or 0)
            if desired_qty > 0:
                desired_positive_qty_rows += 1
            elif desired_qty < 0:
                desired_negative_qty_rows += 1
            current_qty = int(holdings.get(code, 0))
            delta = desired_qty - current_qty
            target_notional = float(desired_qty) * float(px)
            current_notional = float(current_qty) * float(px)
            delta_notional = float(delta) * float(px)
            target_weight_value = (
                float(target_weights.set_index("code")["target_weight"].get(code, 0.0))
                if not target_weights.empty and "target_weight" in target_weights.columns
                else 0.0
            )
            side = "LONG" if target_weight_value > 0 else ("SHORT" if target_weight_value < 0 else "FLAT")
            position_rows.append(
                {
                    "signal_date": signal_date,
                    "rebalance_date": signal_date,
                    "execution_date": result.execution_date,
                    "code": code,
                    "price": float(px),
                    "current_quantity": current_qty,
                    "current_value": current_notional,
                    "current_weight": (current_notional / nav_before) if nav_before else np.nan,
                    "target_weight": target_weight_value,
                    "target_value": float(target_value.get(code, 0.0)),
                    "target_quantity": desired_qty,
                    "delta_quantity": delta,
                    "delta_value": delta_notional,
                    "side": side,
                    "trade_side": "BUY" if delta > 0 else ("SELL" if delta < 0 else "HOLD"),
                    "estimated_cost": abs(delta_notional) * cost_rate,
                }
            )
            if delta == 0:
                continue
            order_rows_this_rebalance += 1
            notional = float(delta) * float(px)
            fees = abs(notional) * cost_rate
            cash -= notional + fees
            holdings[code] = desired_qty
            if holdings[code] == 0:
                holdings.pop(code, None)
            traded_value += abs(notional)
            trade_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": result.execution_date,
                    "code": code,
                    "quantity": delta,
                    "price": float(px),
                    "notional": notional,
                    "cost": fees,
                    "reason": "rebalance",
                }
            )
            trade_rows_this_rebalance += 1
            _update_cost_basis_after_trade(cost_basis, code, current_qty, desired_qty, delta, float(px))
        sizing_and_orders_sec = time.perf_counter() - started
        result.timings["sizing_and_orders_sec"] = sizing_and_orders_sec
        _profile_log(
            signal_date,
            "sizing_and_orders",
            sizing_and_orders_sec,
            desired_positive_qty_rows=int(desired_positive_qty_rows),
            desired_negative_qty_rows=int(desired_negative_qty_rows),
            order_rows=int(order_rows_this_rebalance),
            trade_rows=int(trade_rows_this_rebalance),
        )
        started = time.perf_counter()
        nav_after = _portfolio_value(holdings, prices, cash)
        turnover = traded_value / nav_before if nav_before else np.nan
        nav_rows.append({"date": result.execution_date, "nav": nav_after, "cash": cash, "turnover": turnover})
        rebalance_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": result.execution_date,
                "nav_before": nav_before,
                "nav_after": nav_after,
                "selected_count": len(target_weights),
                "turnover": turnover,
                "gross_target": float(current_weight_values.abs().sum()) if len(current_weight_values) else 0.0,
                "gross_long": float(current_weight_values[current_weight_values > 0].sum()) if len(current_weight_values) else 0.0,
                "gross_short": float((-current_weight_values[current_weight_values < 0]).sum()) if len(current_weight_values) else 0.0,
                "net_exposure": float(current_weight_values.sum()) if len(current_weight_values) else 0.0,
                "hedge_ratio": (float((-current_weight_values[current_weight_values < 0]).sum()) / float(current_weight_values[current_weight_values > 0].sum()))
                if float(current_weight_values[current_weight_values > 0].sum()) > 0
                else np.nan,
            }
        )
        for code, qty in holdings.items():
            px = prices.get(code, np.nan)
            holding_rows.append(
                {
                    "date": result.execution_date,
                    "code": code,
                    "quantity": qty,
                    "price": px,
                    "market_value": float(qty) * float(px) if pd.notna(px) else np.nan,
                    "weight": (float(qty) * float(px) / nav_after) if pd.notna(px) and nav_after else np.nan,
                    "side": "LONG" if int(qty) > 0 else "SHORT",
                    "gross_exposure_contrib": abs(float(qty) * float(px) / nav_after) if pd.notna(px) and nav_after else np.nan,
                }
            )
        holdings_update_sec = time.perf_counter() - started
        result.timings["holdings_update_sec"] = holdings_update_sec
        _profile_log(signal_date, "holdings_update", holdings_update_sec, holding_rows=int(len(holdings)))
        debug_row["order_rows"] = int(order_rows_this_rebalance)
        debug_row["trade_rows"] = int(trade_rows_this_rebalance)
        debug_row["holding_rows"] = int(len(holdings))
        debug_row["desired_positive_qty_rows"] = int(desired_positive_qty_rows)
        debug_row["desired_negative_qty_rows"] = int(desired_negative_qty_rows)
        debug_row["post_price_target_rows"] = int(len(target_weights))
        debug_row["stop_loss_exit_rows"] = int(stop_loss_exit_rows)
        debug_row["stop_loss_monitor_check_dates"] = int(monitor_stats["check_dates"])
        debug_row["stop_loss_monitor_exit_rows"] = int(monitor_stats["exit_rows"])
        if selected_rows <= 0:
            debug_row["zero_stage"] = "screening"
        elif final_weight_rows <= 0:
            debug_row["zero_stage"] = "portfolio construction"
        elif positive_weight_rows <= 0 and negative_weight_rows <= 0:
            debug_row["zero_stage"] = "weight normalization"
        elif debug_row["valid_execution_price_rows"] <= 0:
            debug_row["zero_stage"] = "execution price lookup"
        elif desired_positive_qty_rows <= 0 and desired_negative_qty_rows <= 0:
            debug_row["zero_stage"] = "position sizing"
        elif order_rows_this_rebalance <= 0:
            debug_row["zero_stage"] = "order generation"
        elif trade_rows_this_rebalance <= 0:
            debug_row["zero_stage"] = "holdings update"
        elif len(holdings) <= 0:
            debug_row["zero_stage"] = "holdings update"
        rebalance_debug_rows.append(debug_row)
        print(
            "[rebalance-debug] "
            f"signal_date={signal_date.date()} execution_date={pd.Timestamp(result.execution_date).date()} "
            f"selected_rows={selected_rows} final_weight_rows={final_weight_rows} "
            f"positive_weight_rows={positive_weight_rows} negative_weight_rows={negative_weight_rows} order_rows={order_rows_this_rebalance} "
            f"trade_rows={trade_rows_this_rebalance} holding_rows={len(holdings)} "
            f"zero_stage={debug_row['zero_stage']}",
            flush=True,
        )
        next_holding_ages: dict[str, int] = {}
        for code in holdings:
            if str(code) in current_holdings_at_signal:
                next_holding_ages[str(code)] = int(holding_age_months.get(str(code), 0)) + 1
            else:
                next_holding_ages[str(code)] = 1
        holding_age_months = next_holding_ages
        last_execution_date = pd.Timestamp(result.execution_date).normalize()
        pending_stop_loss_cash_penalty = bool(stop_loss_exit_rows > 0)

    nav = pd.DataFrame(nav_rows)
    if not nav.empty:
        nav["return"] = nav["nav"].pct_change().fillna(nav["nav"] / float(initial_capital) - 1.0)
    start_for_benchmarks = min(dates) if dates else None
    end_for_benchmarks = max(pd.to_datetime(nav["date"])) if not nav.empty else (max(dates) if dates else None)
    benchmarks = (
        build_benchmark_nav(provider, start_for_benchmarks, end_for_benchmarks, cfg, initial_capital)
        if start_for_benchmarks is not None and end_for_benchmarks is not None
        else pd.DataFrame(columns=["date", "benchmark", "close", "nav", "return", "source"])
    )
    regime_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    allocation_gate_rows: list[dict[str, Any]] = []
    for item in results:
        if isinstance(item.regime, dict):
            row = {
                "signal_date": pd.Timestamp(item.signal_date).normalize(),
                "base_regime": item.regime.get("base_regime"),
                "overlay_regime": item.regime.get("overlay_regime"),
                "raw_regime": item.regime.get("raw_regime"),
                "active_regime": item.regime.get("active_regime"),
            }
            features = item.regime.get("features", {})
            if isinstance(features, dict):
                for key, value in features.items():
                    if key != "signal_date":
                        row[key] = value
            regime_rows.append(row)
        if isinstance(item.weight_resolution, dict):
            row = {
                "signal_date": pd.Timestamp(item.signal_date).normalize(),
                "regime": item.weight_resolution.get("regime"),
                "source_mode": item.weight_resolution.get("source_mode"),
                "source": item.weight_resolution.get("source"),
                "json_path": item.weight_resolution.get("json_path"),
                "used_fallback": item.weight_resolution.get("used_fallback"),
                "warnings": " | ".join(item.weight_resolution.get("warnings", [])),
            }
            resolved = item.weight_resolution.get("resolved_weights", {})
            if isinstance(resolved, dict):
                row.update(resolved)
            weight_rows.append(row)
        if isinstance(item.allocation_gate, dict):
            row = {"signal_date": pd.Timestamp(item.signal_date).normalize(), **item.allocation_gate}
            allocation_gate_rows.append(row)
    return {
        "nav": nav,
        "benchmarks": benchmarks,
        "rebalance_summary": pd.DataFrame(rebalance_rows),
        "trades": pd.DataFrame(trade_rows),
        "holdings": pd.DataFrame(holding_rows),
        "position_snapshots": pd.DataFrame(position_rows),
        "corporate_actions": pd.DataFrame(corporate_action_rows),
        "missing_execution_prices": pd.DataFrame(missing_price_rows),
        "rebalance_debug": pd.DataFrame(rebalance_debug_rows),
        "post_score_adjustment_debug": pd.DataFrame(post_score_adjustment_rows),
        "sector_constraint_debug": pd.DataFrame(post_score_adjustment_rows),
        "sector_exposure_by_rebalance": pd.DataFrame(sector_exposure_rows),
        "holding_age_debug": pd.DataFrame(holding_age_debug_rows),
        "top_holdings_by_rebalance": pd.DataFrame(top_holdings_rows),
        "regime_summary": pd.DataFrame(regime_rows),
        "regime_weight_resolution": pd.DataFrame(weight_rows),
        "allocation_gate_summary": pd.DataFrame(allocation_gate_rows),
        "rebalance_results": results,
        "config": pd.DataFrame([asdict(cfg)]),
    }


def monthly_rebalance_dates(start: str | pd.Timestamp, end: str | pd.Timestamp, provider: JpxHistoricalProvider) -> list[pd.Timestamp]:
    """Return last available trading date in each calendar month."""

    prices = provider.table("prices", copy=False)
    dates = pd.Series(pd.to_datetime(prices["date"].dropna().unique())).sort_values()
    dates = dates.loc[dates.between(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), inclusive="both")]
    if dates.empty:
        return []
    frame = pd.DataFrame({"date": dates})
    frame["month"] = frame["date"].dt.to_period("M")
    return [pd.Timestamp(x).normalize() for x in frame.groupby("month")["date"].max().tolist()]


def weekly_rebalance_dates(start: str | pd.Timestamp, end: str | pd.Timestamp, provider: JpxHistoricalProvider) -> list[pd.Timestamp]:
    """Return last available trading date in each calendar week.

    These are signal dates.  Orders are executed by ``run_backtest`` on the next
    available trading day according to ``BacktestConfig.signal_execution_lag_days``.
    """

    prices = provider.table("prices", copy=False)
    dates = pd.Series(pd.to_datetime(prices["date"].dropna().unique())).sort_values()
    dates = dates.loc[dates.between(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), inclusive="both")]
    if dates.empty:
        return []
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    return [pd.Timestamp(x).normalize() for x in frame.groupby("week")["date"].max().tolist()]


def biweekly_rebalance_dates(start: str | pd.Timestamp, end: str | pd.Timestamp, provider: JpxHistoricalProvider) -> list[pd.Timestamp]:
    """Return every other weekly rebalance date.

    This keeps the same week-ending convention as ``weekly_rebalance_dates``
    and simply subsamples the resulting signal dates in chronological order.
    """

    weekly_dates = weekly_rebalance_dates(start, end, provider)
    return weekly_dates[::2]
