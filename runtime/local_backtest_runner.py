from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

RUNTIME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_ROOT.parent
FACTORS_ROOT = PROJECT_ROOT / "factors"
LEGACY_ROOT = PROJECT_ROOT.parent / "backtest"
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != LEGACY_ROOT.resolve()]
sys.path[:0] = [str(PROJECT_ROOT), str(RUNTIME_ROOT), str(FACTORS_ROOT)]

import pandas as pd
import numpy as np

from factor_composite import WeightConfig
from historical_backtest_engine import (
    BacktestConfig,
    biweekly_rebalance_dates,
    monthly_rebalance_dates,
    run_backtest,
    weekly_rebalance_dates,
)
from historical_data import JpxHistoricalProvider, LocalDataPaths, normalize_code
from jquants_cache_builder import (
    JQuantsApiConfig,
    _month_starts,
    _shard_path,
    bulk_month_labels,
    debug_dump_cache_summary,
    download_jquants_backtest_cache,
    infer_required_history_days,
    normalize_full_month_backtest_range,
)
from topix_pool_runtime import LiquidityRule, MarketCapRule, ScreenRules, SelectionRule, VolatilityRule


class TeeLogger:
    """Write plain-text logs to both terminal and the run log file."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        self._file = None

    def __enter__(self) -> "TeeLogger":
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.log_path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def log(self, message: str) -> None:
        ts = pd.Timestamp.now().strftime("%H:%M:%S")
        line = f"[{ts}] [local] {message}"
        print(line, flush=True)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()


@contextlib.contextmanager
def tee_stdout_to_file(log_path: str | Path):
    """Capture lower-level print logs without changing model/backtest code."""

    class _Tee:
        def __init__(self, *streams) -> None:
            self.streams = streams

        def write(self, data: str) -> None:
            for stream in self.streams:
                try:
                    stream.write(data)
                    stream.flush()
                except OSError:
                    continue

        def flush(self) -> None:
            for stream in self.streams:
                try:
                    stream.flush()
                except OSError:
                    continue

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with path.open("a", encoding="utf-8") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            yield
        finally:
            sys.stdout = original_stdout


def load_api_credentials(path: str | Path = "api.txt") -> dict[str, str]:
    """Load local J-Quants credentials without printing secrets.

    Supported formats:
    - single-line token, treated as api_key
    - key=value lines for api_key, id_token, refresh_token
    """

    api_path = Path(path)
    if not api_path.exists():
        raise FileNotFoundError(f"J-Quants credential file not found: {api_path}")
    text = api_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"J-Quants credential file is empty: {api_path}")

    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if len(lines) == 1 and "=" not in lines[0]:
        token = lines[0].strip()
        if not token:
            raise ValueError(f"J-Quants credential file has an empty token: {api_path}")
        return {"api_key": token}

    allowed = {"api_key", "id_token", "refresh_token"}
    out: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            raise ValueError(
                f"Invalid credential line in {api_path}: expected key=value or a single raw token."
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in allowed:
            raise ValueError(f"Unsupported credential key {key!r} in {api_path}. Allowed keys: {sorted(allowed)}")
        if not value:
            raise ValueError(f"Credential key {key!r} is empty in {api_path}")
        out[key] = value
    if not any(out.get(key) for key in allowed):
        raise ValueError(f"No usable J-Quants credential found in {api_path}")
    return out


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".json", ""}:
        return json.loads(text)
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("PyYAML is required to read YAML config files. Use JSON or install pyyaml.") from exc
        data = yaml.safe_load(text)
        return dict(data or {})
    raise ValueError(f"Unsupported config file type: {cfg_path}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def infer_required_history_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    history_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return user range, full-month backtest range, and cache warmup range."""

    user_start = pd.Timestamp(start).normalize()
    user_end = pd.Timestamp(end).normalize()
    effective_start, effective_end = normalize_full_month_backtest_range(user_start, user_end)
    required_days = infer_required_history_days(history_overrides)
    request_start = (effective_start - pd.Timedelta(days=int(required_days))).replace(day=1)
    # Keep one extra month of market data so the final rebalance can execute on
    # the next trading day even when it falls in the following calendar month.
    request_end = (effective_end + pd.offsets.MonthEnd(1)).normalize()
    return {
        "user_start": user_start,
        "user_end": user_end,
        "backtest_start": effective_start,
        "backtest_end": effective_end,
        "request_start": request_start,
        "request_end": request_end,
        "required_history_days": int(required_days),
        "bulk_months": bulk_month_labels(effective_start, effective_end),
    }


def _table_path(data_dir: Path, name: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        path = data_dir / f"{name}{suffix}"
        if path.exists():
            return path
    return None


def _month_key(month: pd.Timestamp) -> str:
    return pd.Timestamp(month).strftime("%Y%m")


def _range_end_missing(info: dict[str, Any], required_end: pd.Timestamp) -> bool:
    if not info.get("exists"):
        return True
    max_date = info.get("max")
    if max_date is None:
        return True
    return pd.Timestamp(max_date).normalize() < pd.Timestamp(required_end).normalize()


def _index_code_coverage_status(
    data_dir: Path,
    required_codes: Iterable[str],
    required_end: pd.Timestamp,
) -> dict[str, Any]:
    codes = [str(code).upper() for code in required_codes if str(code).strip()]
    if not codes:
        return {"required_codes": [], "missing_codes": [], "stale_codes": {}, "is_covered": True}

    path = _table_path(data_dir, "index_prices")
    if path is None:
        return {"required_codes": codes, "missing_codes": codes, "stale_codes": {}, "is_covered": False}

    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path, columns=["date", "index_code"])
        else:
            frame = pd.read_csv(path, usecols=["date", "index_code"], dtype=str, low_memory=False)
    except Exception as exc:
        return {
            "required_codes": codes,
            "missing_codes": [],
            "stale_codes": {code: f"read_error:{exc!r}" for code in codes},
            "is_covered": False,
        }

    if frame.empty or "date" not in frame.columns or "index_code" not in frame.columns:
        return {"required_codes": codes, "missing_codes": codes, "stale_codes": {}, "is_covered": False}

    work = frame.copy()
    work["index_code"] = work["index_code"].astype(str).str.upper()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    max_by_code = work.dropna(subset=["date"]).groupby("index_code")["date"].max()

    missing_codes: list[str] = []
    stale_codes: dict[str, str] = {}
    for code in codes:
        max_date = max_by_code.get(code)
        if pd.isna(max_date):
            missing_codes.append(code)
            continue
        if pd.Timestamp(max_date).normalize() < pd.Timestamp(required_end).normalize():
            stale_codes[code] = pd.Timestamp(max_date).strftime("%Y-%m-%d")
    return {
        "required_codes": codes,
        "missing_codes": missing_codes,
        "stale_codes": stale_codes,
        "is_covered": not missing_codes and not stale_codes,
    }


def check_local_cache(
    data_dir: str | Path,
    required: dict[str, Any],
    *,
    save_format: str = "parquet",
    strict_validation: bool = False,
    include_financials: bool = False,
    include_margin: bool = False,
    include_topix_index: bool = False,
    include_nikkei_index: bool = False,
    additional_index_codes: Iterable[str] = (),
) -> dict[str, Any]:
    """Inspect local cache coverage without making network requests."""

    data_path = Path(data_dir)
    request_start = pd.Timestamp(required["request_start"])
    request_end = pd.Timestamp(required["request_end"])
    months = _month_starts(request_start, request_end)
    missing_price_shards = []
    existing_price_shards = []
    for month in months:
        key = _month_key(month)
        path = _shard_path(data_path, "prices", key, save_format)
        csv_path = path.with_suffix(".csv")
        if path.exists() or csv_path.exists():
            existing_price_shards.append(key)
        else:
            missing_price_shards.append(key)

    required_tables = ["universe", "sector", "market_cap", "prices"]
    missing_tables = [name for name in required_tables if _table_path(data_path, name) is None]
    table_ranges = {
        "prices": _table_date_range(_table_path(data_path, "prices"), "date"),
        "universe": _table_date_range(_table_path(data_path, "universe"), "asof_date"),
        "sector": _table_date_range(_table_path(data_path, "sector"), "asof_date"),
        "market_cap": _table_date_range(_table_path(data_path, "market_cap"), "asof_date"),
        "index_prices": _table_date_range(_table_path(data_path, "index_prices"), "date"),
        "financial_summary": _table_date_range(_table_path(data_path, "financial_summary"), "disclosed_date"),
        "margin": _table_date_range(_table_path(data_path, "margin"), "date"),
    }
    invalid_tables: list[str] = []
    invalid_datasets: list[str] = []
    invalid_reasons: dict[str, str] = {}
    for name in ["prices", "sector", "market_cap", "index_prices"]:
        path = _table_path(data_path, name)
        table_ranges.setdefault(name, {"exists": False, "min": None, "max": None, "rows": 0})
        if path is None:
            table_ranges[name]["is_valid"] = False
            if name in {"prices", "sector", "market_cap"}:
                invalid_reasons[name] = "missing required cache table"
            continue
        if not strict_validation:
            table_ranges[name]["is_valid"] = bool(table_ranges[name].get("rows", 0) != 0 and "error" not in table_ranges[name])
            continue
        try:
            from jquants_cache_builder import validate_cache_table_contract

            frame = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, dtype=str, low_memory=False)
            validate_cache_table_contract(frame, name, min_rows=1, asof_date=required["backtest_start"])
            table_ranges[name]["is_valid"] = True
        except Exception as exc:
            marker = f"{name}_values"
            if marker not in invalid_tables:
                invalid_tables.append(marker)
            if name not in invalid_datasets:
                invalid_datasets.append(name)
            table_ranges.setdefault(name, {})
            table_ranges[name]["validation_error"] = repr(exc)
            table_ranges[name]["is_valid"] = False
            invalid_reasons[name] = repr(exc)
    price_coverage = _price_cache_coverage_status(
        table_ranges.get("prices", {}),
        request_start,
        request_end,
        missing_price_shards,
    )
    table_ranges["prices"].update(price_coverage)
    if not price_coverage["is_covered"]:
        if "prices_range" not in missing_tables:
            missing_tables.append("prices_range")
    for name in ["universe", "sector", "market_cap"]:
        max_date = table_ranges[name].get("max")
        if max_date is None or pd.Timestamp(max_date).normalize() < pd.Timestamp(required["backtest_start"]).normalize():
            marker = f"{name}_range"
            if marker not in missing_tables:
                missing_tables.append(marker)
    required_index_codes: list[str] = []
    if include_topix_index:
        required_index_codes.append("TOPIX")
    if include_nikkei_index:
        required_index_codes.append("NIKKEI225")
    required_index_codes.extend(str(code).upper() for code in additional_index_codes)
    index_code_coverage = _index_code_coverage_status(data_path, required_index_codes, request_end)
    table_ranges["index_prices"]["required_codes"] = index_code_coverage["required_codes"]
    table_ranges["index_prices"]["missing_codes"] = index_code_coverage["missing_codes"]
    table_ranges["index_prices"]["stale_codes"] = index_code_coverage["stale_codes"]
    table_ranges["index_prices"]["is_covered"] = index_code_coverage["is_covered"]
    if required_index_codes:
        if _range_end_missing(table_ranges["index_prices"], request_end):
            marker = "index_prices_range"
            if marker not in missing_tables:
                missing_tables.append(marker)
        if index_code_coverage["missing_codes"]:
            marker = "index_prices_codes"
            if marker not in missing_tables:
                missing_tables.append(marker)
        if index_code_coverage["stale_codes"]:
            marker = "index_prices_stale_codes"
            if marker not in missing_tables:
                missing_tables.append(marker)
    for marker in invalid_tables:
        if marker not in missing_tables:
            missing_tables.append(marker)
    # These tables feed the quality/value and behaviour factors.  They are not
    # required to instantiate the provider, but a local run should still trigger
    # the data-preparation step when they are absent so those modules do not
    # silently become neutral placeholders.
    for name, enabled in [("financial_summary", include_financials), ("margin", include_margin)]:
        if not enabled:
            continue
        info = table_ranges.get(name, {})
        if not info.get("exists"):
            marker = f"{name}_missing"
            if marker not in missing_tables:
                missing_tables.append(marker)
            continue
        if _range_end_missing(info, request_end):
            marker = f"{name}_range"
            if marker not in missing_tables:
                missing_tables.append(marker)
    status = {
        "data_dir": str(data_path),
        "required_months": [_month_key(m) for m in months],
        "existing_price_shards": existing_price_shards,
        "missing_price_shards": missing_price_shards,
        "missing_tables": missing_tables,
        "table_ranges": table_ranges,
        "invalid_tables": invalid_tables,
        "invalid_datasets": invalid_datasets,
        "invalid_reasons": invalid_reasons,
        "cache_hit": not missing_price_shards and not missing_tables,
    }
    return status


def _is_cache_coverage_gap(reason: str) -> bool:
    """True when validation failed because the cache starts too late.

    A coverage gap should trigger incremental download of missing shards, not a
    destructive force rebuild.  Value/schema corruption is handled separately.
    """

    text = str(reason or "").lower()
    return "has no rows on or before" in text


def _cache_invalid_requires_force_rebuild(cache_status: dict[str, Any]) -> bool:
    """Decide whether invalid cache values are genuinely corrupt.

    Missing date coverage can be repaired by appending older shards.  Only
    schema/value failures should opt into the existing force-rebuild path.
    """

    reasons = cache_status.get("invalid_reasons", {}) or {}
    for reason in reasons.values():
        if not _is_cache_coverage_gap(str(reason)):
            return True
    return False


def _price_cache_coverage_status(
    price_range: dict[str, Any],
    request_start: pd.Timestamp,
    request_end: pd.Timestamp,
    missing_price_shards: list[str],
) -> dict[str, Any]:
    """Return a conservative but non-noisy price cache coverage decision."""

    price_min = price_range.get("min")
    price_max = price_range.get("max")
    valid_values = bool(price_range.get("is_valid", True))
    has_main_range = False
    all_required_shards_present = len(missing_price_shards) == 0
    if price_min is not None and price_max is not None:
        min_ts = pd.Timestamp(price_min).normalize()
        max_ts = pd.Timestamp(price_max).normalize()
        # Full-market bulk data is monthly and starts on the first exchange
        # trading day, not necessarily the calendar month start.  If the
        # required monthly shard is present, allow the main table to start
        # within the first week of the request month instead of rebuilding
        # forever because e.g. Jan 1-3 are holidays.
        start_covered = min_ts <= request_start or (
            all_required_shards_present
            and min_ts.strftime("%Y%m") == request_start.strftime("%Y%m")
            and min_ts <= request_start + pd.Timedelta(days=7)
        )
        has_main_range = bool(start_covered and max_ts >= request_end)
    return {
        "required_start": request_start,
        "required_end": request_end,
        "main_range_covers_required": has_main_range,
        "all_required_monthly_shards_present": all_required_shards_present,
        "is_covered": bool(valid_values and has_main_range and all_required_shards_present),
    }


def download_missing_data(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: str | Path,
    api_config: JQuantsApiConfig,
    logger: TeeLogger,
) -> dict[str, str]:
    """Download or incrementally complete the local J-Quants cache."""

    logger.log("download/cache build start")
    outputs = download_jquants_backtest_cache(start, end, data_dir=data_dir, api_config=api_config)
    logger.log(f"download/cache build done tables={sorted(outputs)}")
    return {key: str(path) for key, path in outputs.items()}


def _read_rows(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return int(len(pd.read_parquet(path, columns=[])))
        return int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
    except Exception:
        return -1


def _table_date_range(path: Path | None, date_col: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "min": None, "max": None, "rows": 0}
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path, columns=[date_col])
        else:
            frame = pd.read_csv(path, usecols=[date_col], dtype=str, low_memory=False)
        if date_col not in frame.columns:
            return {"exists": True, "min": None, "max": None, "rows": len(frame), "missing_date_col": date_col}
        dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
        return {
            "exists": True,
            "min": dates.min() if not dates.empty else None,
            "max": dates.max() if not dates.empty else None,
            "rows": len(frame),
        }
    except Exception as exc:
        return {"exists": True, "min": None, "max": None, "rows": -1, "error": repr(exc)}


def update_manifest(data_dir: str | Path, required: dict[str, Any], cache_status: dict[str, Any], config: dict[str, Any]) -> Path:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name in ["prices", "universe", "sector", "market_cap", "financial_summary", "index_prices", "margin"]:
        path = _table_path(data_path, name)
        tables[name] = {"path": str(path) if path else None, "rows": _read_rows(path)}
    manifest = {
        "updated_at": pd.Timestamp.now().isoformat(),
        "required_range": {key: _json_default(value) for key, value in required.items()},
        "cache_status": cache_status,
        "tables": tables,
        "config": config,
    }
    path = data_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_provider_from_local_cache(data_dir: str | Path) -> JpxHistoricalProvider:
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=data_dir))
    # Fail fast here rather than half-way through the first rebalance.
    for table in ["prices", "universe", "sector", "market_cap"]:
        provider.table(table, copy=False)
    return provider


def build_rules(config: dict[str, Any]) -> ScreenRules:
    rule_cfg = dict(config.get("rules", {}))
    return ScreenRules(
        market_cap=MarketCapRule(**dict(rule_cfg.get("market_cap", {"mode": "rank", "rank_min": 200, "rank_max": 500}))),
        liquidity=LiquidityRule(**dict(rule_cfg.get("liquidity", {"lookback_days": 60, "min_avg_daily_volume": 0}))),
        volatility=VolatilityRule(**dict(rule_cfg.get("volatility", {"lookback_days": 60, "max_annualized_volatility": 999.0}))),
        selection=SelectionRule(**dict(rule_cfg.get("selection", {"top_n_if_over_200": 500, "sort_by": "market_cap_desc"}))),
    )


def build_backtest_config(config: dict[str, Any]) -> BacktestConfig:
    bt_cfg = dict(config.get("backtest", {}))
    gui_cfg = dict(config.get("gui", {}))
    if gui_cfg.get("formula") and "composite_formula" not in bt_cfg:
        bt_cfg["composite_formula"] = str(gui_cfg["formula"])
    weights = bt_cfg.pop("composite_weights", None)
    if isinstance(weights, dict):
        bt_cfg["composite_weights"] = WeightConfig(**weights)
    return BacktestConfig(**bt_cfg)


def create_run_dir(base_dir: str | Path) -> Path:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = root / stamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}_{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def _frame_to_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _weights_from_rebalance_results(results: Iterable[Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        weights = getattr(result, "target_weights", pd.DataFrame())
        if weights is None or weights.empty:
            continue
        frame = weights.copy()
        frame.insert(0, "signal_date", getattr(result, "signal_date", pd.NaT))
        frame.insert(1, "execution_date", getattr(result, "execution_date", pd.NaT))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


EXPECTED_FACTOR_COLUMNS = [
    "quality_raw",
    "value_raw",
    "momentum_raw",
    "dual_ma_raw",
    "reversal_raw",
    "behaviour_raw",
    "attention_raw",
    "Q",
    "V",
    "M",
    "TS",
    "REV",
    "B",
    "T",
    "F",
    "M_gate",
    "I",
    "C",
    "Score",
    "TotalRank",
]


def _normalise_code_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "code" in out.columns:
        out["code"] = out["code"].map(normalize_code).astype("string")
    return out


def _factor_table_for_rebalance(result: Any) -> pd.DataFrame:
    composite = getattr(result, "composite", {}) or {}
    scored = composite.get("scored_df") if isinstance(composite, dict) else None
    if not isinstance(scored, pd.DataFrame):
        scored = pd.DataFrame()
    table = _normalise_code_column(scored)
    if table.empty:
        universe = getattr(result, "universe", pd.DataFrame())
        table = _normalise_code_column(universe if isinstance(universe, pd.DataFrame) else pd.DataFrame())
    universe = getattr(result, "universe", pd.DataFrame())
    if isinstance(universe, pd.DataFrame) and not universe.empty and "code" in universe.columns:
        meta_cols = [
            c
            for c in [
                "code",
                "ticker",
                "stock_name",
                "name",
                "company_name",
                "industry",
                "sector",
                "market_cap_jpy",
                "market_cap_billion_jpy",
                "market_cap_rank",
                "selection_rank",
                "anchor_date",
                "rebalance_date",
            ]
            if c in universe.columns
        ]
        meta = _normalise_code_column(universe[meta_cols]).drop_duplicates("code", keep="last")
        add_cols = [c for c in meta.columns if c == "code" or c not in table.columns]
        table = table.merge(meta[add_cols], on="code", how="left") if "code" in table.columns else table
    if "stock_name" not in table.columns:
        for candidate in ["name", "company_name", "CoName", "co_name"]:
            if candidate in table.columns:
                table["stock_name"] = table[candidate]
                break
        if "stock_name" not in table.columns:
            table["stock_name"] = pd.NA
    if "sector" not in table.columns and "industry" in table.columns:
        table["sector"] = table["industry"]
    if "market_cap" not in table.columns:
        if "market_cap_jpy" in table.columns:
            table["market_cap"] = table["market_cap_jpy"]
        elif "market_cap_billion_jpy" in table.columns:
            table["market_cap"] = pd.to_numeric(table["market_cap_billion_jpy"], errors="coerce") * 1_000_000_000.0
        else:
            table["market_cap"] = pd.NA
    signal_date = getattr(result, "signal_date", pd.NaT)
    rebalance_date = getattr(result, "rebalance_date", signal_date)
    if "anchor_date" not in table.columns:
        table["anchor_date"] = signal_date
    table["rebalance_date"] = rebalance_date
    weights = getattr(result, "target_weights", pd.DataFrame())
    if isinstance(weights, pd.DataFrame) and not weights.empty and "code" in weights.columns:
        w = _normalise_code_column(weights)
        keep = [c for c in ["code", "target_weight", "side"] if c in w.columns]
        table = table.merge(w[keep].drop_duplicates("code", keep="last"), on="code", how="left", suffixes=("", "_target"))
    if "target_weight" not in table.columns:
        table["target_weight"] = np.nan
    if "side" not in table.columns:
        table["side"] = pd.NA
    table["selected"] = pd.to_numeric(table["target_weight"], errors="coerce").fillna(0.0).ne(0)
    for col in EXPECTED_FACTOR_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    leading = [
        "rebalance_date",
        "anchor_date",
        "code",
        "stock_name",
        "sector",
        "market_cap",
        "selected",
        "side",
        "target_weight",
        "TotalRank",
        "Score",
    ]
    ordered = [c for c in leading if c in table.columns] + [c for c in table.columns if c not in leading]
    return table[ordered].sort_values(["selected", "target_weight", "Score", "code"], ascending=[False, False, False, True], na_position="last").reset_index(drop=True)


def save_factor_outputs(result: dict[str, Any], run_dir: Path) -> dict[str, str]:
    factor_dir = run_dir / "factor"
    factor_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    for item in result.get("rebalance_results", []):
        signal_date = pd.Timestamp(getattr(item, "signal_date", getattr(item, "rebalance_date", pd.NaT))).normalize()
        table = _factor_table_for_rebalance(item)
        filename = f"{signal_date.strftime('%Y-%m-%d')}_factors.csv"
        path = factor_dir / filename
        _frame_to_csv(table, path)
        manifest_rows.append(
            {
                "rebalance_date": signal_date,
                "path": str(path),
                "rows": int(len(table)),
                "selected_count": int(table["selected"].sum()) if "selected" in table.columns else 0,
                "non_null_score_count": int(pd.to_numeric(table.get("Score", pd.Series(dtype=float)), errors="coerce").notna().sum()),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = factor_dir / "factor_manifest.csv"
    _frame_to_csv(manifest, manifest_path)
    paths["factor_dir"] = str(factor_dir)
    paths["factor_manifest"] = str(manifest_path)
    return paths


def save_position_outputs(result: dict[str, Any], run_dir: Path) -> dict[str, str]:
    """Save per-rebalance current-vs-target position snapshots."""

    positions_dir = run_dir / "positions"
    positions_dir.mkdir(parents=True, exist_ok=True)
    positions = result.get("position_snapshots", pd.DataFrame())
    paths: dict[str, str] = {"positions_dir": str(positions_dir)}
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        manifest_path = positions_dir / "positions_manifest.csv"
        _frame_to_csv(pd.DataFrame(columns=["rebalance_date", "path", "rows", "buy_rows", "sell_rows", "hold_rows"]), manifest_path)
        paths["positions_manifest"] = str(manifest_path)
        return paths

    table = _normalise_code_column(positions)
    meta_frames: list[pd.DataFrame] = []
    for item in result.get("rebalance_results", []):
        signal_date = pd.Timestamp(getattr(item, "signal_date", getattr(item, "rebalance_date", pd.NaT))).normalize()
        factor_table = _factor_table_for_rebalance(item)
        keep = [c for c in ["code", "stock_name", "sector", "market_cap", "Score", "TotalRank"] if c in factor_table.columns]
        if keep:
            meta = factor_table[keep].copy()
            meta.insert(0, "rebalance_date", signal_date)
            meta_frames.append(meta)
    if meta_frames:
        meta_all = pd.concat(meta_frames, ignore_index=True)
        meta_all = _normalise_code_column(meta_all).drop_duplicates(["rebalance_date", "code"], keep="last")
        table["rebalance_date"] = pd.to_datetime(table["rebalance_date"], errors="coerce").dt.normalize()
        table = table.merge(meta_all, on=["rebalance_date", "code"], how="left")

    manifest_rows: list[dict[str, Any]] = []
    date_col = "rebalance_date" if "rebalance_date" in table.columns else "signal_date"
    table[date_col] = pd.to_datetime(table[date_col], errors="coerce").dt.normalize()
    for rebalance_date, group in table.groupby(date_col, dropna=False):
        date_ts = pd.Timestamp(rebalance_date).normalize()
        out = group.sort_values(["trade_side", "target_weight", "code"], ascending=[True, False, True]).reset_index(drop=True)
        path = positions_dir / f"{date_ts.strftime('%Y-%m-%d')}_positions.csv"
        _frame_to_csv(out, path)
        sides = out.get("trade_side", pd.Series(dtype=object)).astype(str).str.upper()
        manifest_rows.append(
            {
                "rebalance_date": date_ts,
                "path": str(path),
                "rows": int(len(out)),
                "buy_rows": int((sides == "BUY").sum()),
                "sell_rows": int((sides == "SELL").sum()),
                "hold_rows": int((sides == "HOLD").sum()),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = positions_dir / "positions_manifest.csv"
    _frame_to_csv(manifest, manifest_path)
    paths["positions_manifest"] = str(manifest_path)
    return paths


def _metric_row(dates: pd.Series, nav: pd.Series, series_name: str, *, benchmark_name: str | None = None) -> dict[str, Any]:
    work = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "nav": pd.to_numeric(nav, errors="coerce")})
    work = work.dropna().sort_values("date")
    if work.empty:
        return {
            "series": series_name,
            "benchmark_name": benchmark_name,
            "start_date": pd.NaT,
            "end_date": pd.NaT,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "max_drawdown": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
        }
    start_nav = float(work["nav"].iloc[0])
    end_nav = float(work["nav"].iloc[-1])
    total_return = end_nav / start_nav - 1.0 if start_nav > 0 else np.nan
    days = max(1, int((work["date"].iloc[-1] - work["date"].iloc[0]).days))
    annualized_return = (1.0 + total_return) ** (365.25 / days) - 1.0 if pd.notna(total_return) and total_return > -1 else np.nan
    interval_days = work["date"].diff().dt.days
    log_returns = np.log(work["nav"] / work["nav"].shift(1))
    interval_frame = pd.DataFrame(
        {
            "interval_days": pd.to_numeric(interval_days, errors="coerce"),
            "log_return": pd.to_numeric(log_returns, errors="coerce"),
        }
    ).dropna()
    interval_frame = interval_frame.loc[interval_frame["interval_days"] > 0].copy()
    if interval_frame.empty:
        annualized_volatility = np.nan
        downside_deviation = np.nan
    else:
        # The strategy NAV is recorded on irregular event dates rather than a
        # dense daily grid. Dailyizing interval log-returns avoids treating a
        # 30-day move as if it were a 1-day move when annualizing volatility.
        interval_frame["daily_log_return"] = interval_frame["log_return"] / interval_frame["interval_days"]
        daily_log_returns = interval_frame["daily_log_return"]
        annualized_volatility = float(daily_log_returns.std(ddof=0) * np.sqrt(252.0))
        downside_daily_log_returns = daily_log_returns.clip(upper=0.0)
        downside_deviation = float(np.sqrt((downside_daily_log_returns ** 2).mean()) * np.sqrt(252.0))
    sharpe_ratio = float(annualized_return / annualized_volatility) if annualized_volatility and pd.notna(annualized_volatility) else np.nan
    sortino_ratio = float(annualized_return / downside_deviation) if downside_deviation and pd.notna(downside_deviation) else np.nan
    running_max = work["nav"].cummax()
    drawdown = work["nav"] / running_max - 1.0
    return {
        "series": series_name,
        "benchmark_name": benchmark_name,
        "start_date": work["date"].iloc[0],
        "end_date": work["date"].iloc[-1],
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
    }


def _format_percent(value: Any) -> str:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(num) else f"{num:.2%}"


def _format_number(value: Any) -> str:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(num) else f"{num:.2f}"


def compute_performance_outputs(result: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nav = result.get("nav", pd.DataFrame())
    benchmarks = result.get("benchmarks", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    ts_rows: list[pd.DataFrame] = []
    if isinstance(nav, pd.DataFrame) and not nav.empty and {"date", "nav"}.issubset(nav.columns):
        rows.append(_metric_row(nav["date"], nav["nav"], "strategy"))
        strat = nav[["date", "nav"]].copy()
        strat["series"] = "strategy"
        strat["return"] = pd.to_numeric(strat["nav"], errors="coerce").pct_change()
        ts_rows.append(strat[["date", "series", "nav", "return"]])
    if isinstance(benchmarks, pd.DataFrame) and not benchmarks.empty and {"date", "benchmark", "nav"}.issubset(benchmarks.columns):
        for benchmark, group in benchmarks.groupby("benchmark"):
            rows.append(_metric_row(group["date"], group["nav"], str(benchmark), benchmark_name=str(benchmark)))
            bench = group[["date", "benchmark", "nav", "return"]].copy()
            bench = bench.rename(columns={"benchmark": "series"})
            ts_rows.append(bench[["date", "series", "nav", "return"]])
    summary = pd.DataFrame(rows)
    if not summary.empty and "annualized_return" in summary.columns:
        strategy_ann = summary.loc[summary["series"] == "strategy", "annualized_return"]
        strategy_value = float(strategy_ann.iloc[0]) if not strategy_ann.empty and pd.notna(strategy_ann.iloc[0]) else np.nan
        summary["excess_return"] = np.where(
            summary["series"].eq("strategy"),
            np.nan,
            strategy_value - pd.to_numeric(summary["annualized_return"], errors="coerce"),
        )
        summary["outperformed_by_strategy"] = np.where(
            summary["series"].eq("strategy"),
            pd.NA,
            strategy_value > pd.to_numeric(summary["annualized_return"], errors="coerce"),
        )
    timeseries = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame(columns=["date", "series", "nav", "return"])
    if not timeseries.empty:
        timeseries["date"] = pd.to_datetime(timeseries["date"], errors="coerce")
        timeseries = timeseries.sort_values(["series", "date"]).reset_index(drop=True)
        timeseries["drawdown"] = timeseries.groupby("series")["nav"].transform(lambda s: pd.to_numeric(s, errors="coerce") / pd.to_numeric(s, errors="coerce").cummax() - 1.0)
    return summary, timeseries


def _plot_nav_curve(timeseries: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    if timeseries.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _plot_nav_curve_svg(timeseries, summary, path.with_suffix(".svg"))
        return

    plot_df = timeseries.dropna(subset=["date", "nav"]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for series, group in plot_df.groupby("series"):
        ordered = group.sort_values("date")
        base = pd.to_numeric(ordered["nav"], errors="coerce").dropna()
        if base.empty or float(base.iloc[0]) == 0:
            continue
        normalized = pd.to_numeric(ordered["nav"], errors="coerce") / float(base.iloc[0])
        ax.plot(ordered["date"], normalized, label=str(series), linewidth=2 if series == "strategy" else 1.5)
    ax.set_title("Backtest NAV Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1.0")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    if not summary.empty:
        display = summary.copy()
        display["AnnRet"] = display["annualized_return"].map(_format_percent)
        display["MaxDD"] = display["max_drawdown"].map(_format_percent)
        display["Vol"] = display["annualized_volatility"].map(_format_percent)
        display["Sharpe"] = display["sharpe_ratio"].map(_format_number)
        display["Sortino"] = display["sortino_ratio"].map(_format_number)
        display["Excess"] = display["excess_return"].map(_format_percent) if "excess_return" in display.columns else ""
        rows = display[["series", "AnnRet", "MaxDD", "Vol", "Sharpe", "Sortino", "Excess"]].fillna("").values.tolist()
        table = ax.table(
            cellText=rows,
            colLabels=["Series", "AnnRet", "MaxDD", "Vol", "Sharpe", "Sortino", "Excess"],
            loc="bottom",
            bbox=[0.0, -0.45, 1.0, 0.32],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        fig.subplots_adjust(bottom=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_nav_curve_svg(timeseries: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    import html

    plot_df = timeseries.dropna(subset=["date", "nav"]).copy()
    if plot_df.empty:
        return
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"])
    if plot_df.empty:
        return
    width, height = 1100, 720
    left, right, top, chart_bottom = 80, 40, 50, 420
    min_date = plot_df["date"].min()
    max_date = plot_df["date"].max()
    span_days = max(1, int((max_date - min_date).days))
    normalized_frames = []
    for series, group in plot_df.groupby("series"):
        ordered = group.sort_values("date").copy()
        base = pd.to_numeric(ordered["nav"], errors="coerce").dropna()
        if base.empty or float(base.iloc[0]) == 0:
            continue
        ordered["growth"] = pd.to_numeric(ordered["nav"], errors="coerce") / float(base.iloc[0])
        normalized_frames.append(ordered)
    if not normalized_frames:
        return
    norm = pd.concat(normalized_frames, ignore_index=True).dropna(subset=["growth"])
    min_growth = float(min(0.95, norm["growth"].min()))
    max_growth = float(max(1.05, norm["growth"].max()))
    if max_growth <= min_growth:
        max_growth = min_growth + 0.01
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="80" y="30" font-family="Arial" font-size="22" font-weight="bold">Backtest NAV Curve</text>',
        f'<line x1="{left}" y1="{chart_bottom}" x2="{width-right}" y2="{chart_bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{chart_bottom}" stroke="#333"/>',
    ]
    legend_y = 62
    for idx, (series, group) in enumerate(norm.groupby("series")):
        color = colors[idx % len(colors)]
        points = []
        for _, row in group.sort_values("date").iterrows():
            x = left + ((pd.Timestamp(row["date"]) - min_date).days / span_days) * (width - left - right)
            y = chart_bottom - ((float(row["growth"]) - min_growth) / (max_growth - min_growth)) * (chart_bottom - top)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
            lines.append(f'<rect x="{width - 230}" y="{legend_y - 12}" width="14" height="3" fill="{color}"/>')
            lines.append(f'<text x="{width - 210}" y="{legend_y}" font-family="Arial" font-size="13">{html.escape(str(series))}</text>')
            legend_y += 20
    lines.extend(
        [
            f'<text x="{left}" y="{chart_bottom + 25}" font-family="Arial" font-size="12">{min_date.date()}</text>',
            f'<text x="{width - right - 90}" y="{chart_bottom + 25}" font-family="Arial" font-size="12">{max_date.date()}</text>',
            f'<text x="20" y="{top + 5}" font-family="Arial" font-size="12">{max_growth:.2f}</text>',
            f'<text x="20" y="{chart_bottom}" font-family="Arial" font-size="12">{min_growth:.2f}</text>',
        ]
    )
    table_y = 480
    lines.append(f'<text x="80" y="{table_y}" font-family="Arial" font-size="16" font-weight="bold">Performance Summary</text>')
    headers = ["Series", "AnnRet", "MaxDD", "Vol", "Sharpe", "Sortino", "Excess"]
    col_x = [80, 220, 330, 440, 550, 650, 770]
    for x, header in zip(col_x, headers):
        lines.append(f'<text x="{x}" y="{table_y + 28}" font-family="Arial" font-size="13" font-weight="bold">{header}</text>')
    for ridx, (_, row) in enumerate(summary.fillna("").iterrows()):
        y = table_y + 52 + ridx * 22
        values = [
            str(row.get("series", "")),
            _format_percent(row.get("annualized_return")),
            _format_percent(row.get("max_drawdown")),
            _format_percent(row.get("annualized_volatility")),
            _format_number(row.get("sharpe_ratio")),
            _format_number(row.get("sortino_ratio")),
            _format_percent(row.get("excess_return")),
        ]
        for x, value in zip(col_x, values):
            lines.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="12">{html.escape(value)}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_performance_outputs(result: dict[str, Any], run_dir: Path) -> dict[str, str]:
    performance_dir = run_dir / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    summary, timeseries = compute_performance_outputs(result)
    result["_performance_summary"] = summary
    summary_path = performance_dir / "performance_summary.csv"
    timeseries_path = performance_dir / "performance_timeseries.csv"
    chart_path = performance_dir / "nav_curve.png"
    _frame_to_csv(summary, summary_path)
    _frame_to_csv(timeseries, timeseries_path)
    _plot_nav_curve(timeseries, summary, chart_path)
    actual_chart_path = chart_path if chart_path.exists() else chart_path.with_suffix(".svg")
    return {
        "performance_dir": str(performance_dir),
        "performance_summary": str(summary_path),
        "performance_timeseries": str(timeseries_path),
        "nav_curve": str(actual_chart_path),
    }


def save_risk_gate_outputs(result: dict[str, Any], run_dir: Path) -> dict[str, str]:
    """Save per-rebalance risk-gate diagnostics without changing exposure logic."""

    risk_dir = run_dir / "risk_gate"
    risk_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    paths: dict[str, str] = {"risk_gate_dir": str(risk_dir)}
    for item in result.get("rebalance_results", []):
        risk = getattr(item, "risk", None)
        signal_date = pd.Timestamp(getattr(item, "signal_date", getattr(item, "rebalance_date", pd.NaT))).normalize()
        if not isinstance(risk, dict) or not isinstance(risk.get("minimal"), pd.DataFrame) or risk["minimal"].empty:
            summary_rows.append(
                {
                    "rebalance_date": signal_date,
                    "risk_gate_available": False,
                    "risk_gate_active": False,
                    "gross_exposure_multiplier": np.nan,
                    "risk_score_effective": np.nan,
                    "risk_on": pd.NA,
                    "trend_on": pd.NA,
                }
            )
            continue
        minimal = risk["minimal"].copy()
        detail = risk.get("detail", pd.DataFrame())
        minimal_path = risk_dir / f"{signal_date.strftime('%Y-%m-%d')}_risk_minimal.csv"
        detail_path = risk_dir / f"{signal_date.strftime('%Y-%m-%d')}_risk_detail.csv"
        _frame_to_csv(minimal, minimal_path)
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            out_detail = detail.reset_index().rename(columns={"index": "date"})
            _frame_to_csv(out_detail, detail_path)
            detail_rows = int(len(out_detail))
        else:
            detail_path = Path("")
            detail_rows = 0
        row = minimal.iloc[0].to_dict()
        multiplier = pd.to_numeric(pd.Series([row.get("gross_exposure_multiplier")]), errors="coerce").iloc[0]
        risk_active = bool(pd.notna(multiplier) and abs(float(multiplier) - 1.0) > 1e-12)
        summary_rows.append(
            {
                "rebalance_date": signal_date,
                "risk_gate_available": True,
                "risk_gate_active": risk_active,
                "risk_model_name": row.get("risk_model_name"),
                "risk_score_0_5": row.get("risk_score_0_5"),
                "risk_score_effective": row.get("risk_score_effective"),
                "risk_on": row.get("risk_on"),
                "trend_on": row.get("trend_on"),
                "gross_exposure_multiplier": row.get("gross_exposure_multiplier"),
                "equity_exposure_multiplier": row.get("equity_exposure_multiplier"),
                "cash_weight_floor": row.get("cash_weight_floor"),
                "risk_index_ticker": row.get("risk_index_ticker", row.get("nikkei_ticker")),
                "nikkei_ticker": row.get("nikkei_ticker", row.get("risk_index_ticker")),
                "data_end_date": row.get("data_end_date"),
            }
        )
        manifest_rows.append(
            {
                "rebalance_date": signal_date,
                "minimal_path": str(minimal_path),
                "detail_path": str(detail_path) if str(detail_path) else "",
                "minimal_rows": int(len(minimal)),
                "detail_rows": detail_rows,
                "risk_gate_active": risk_active,
            }
        )
    summary = pd.DataFrame(summary_rows)
    manifest = pd.DataFrame(manifest_rows)
    summary_path = risk_dir / "risk_gate_summary.csv"
    manifest_path = risk_dir / "risk_gate_manifest.csv"
    _frame_to_csv(summary, summary_path)
    _frame_to_csv(manifest, manifest_path)
    paths["risk_gate_summary"] = str(summary_path)
    paths["risk_gate_manifest"] = str(manifest_path)
    return paths


def performance_log_summary(performance_summary: pd.DataFrame, started_at: pd.Timestamp, run_dir: Path) -> str:
    elapsed = (pd.Timestamp.now() - started_at).total_seconds()
    lines = [
        "================ Backtest Performance Summary ================",
        f"total_runtime_sec: {elapsed:.1f}",
        f"total_runtime_min: {elapsed / 60.0:.2f}",
        f"result_dir: {run_dir}",
        "",
        "series        ann_return  max_drawdown  volatility  sharpe  sortino  excess_return  outperformed",
    ]
    if performance_summary.empty:
        lines.append("(no performance rows)")
    else:
        for _, row in performance_summary.iterrows():
            series = str(row.get("series", ""))
            excess = "-" if series == "strategy" else _format_percent(row.get("excess_return"))
            outperformed = "-" if series == "strategy" else str(row.get("outperformed_by_strategy"))
            lines.append(
                f"{series:<13} "
                f"{_format_percent(row.get('annualized_return')):<11} "
                f"{_format_percent(row.get('max_drawdown')):<13} "
                f"{_format_percent(row.get('annualized_volatility')):<11} "
                f"{_format_number(row.get('sharpe_ratio')):<7} "
                f"{_format_number(row.get('sortino_ratio')):<8} "
                f"{excess:<14} "
                f"{outperformed}"
            )
    lines.append("==============================================================")
    return "\n".join(lines)


def write_run_status(
    run_dir: str | Path,
    *,
    status: str,
    created_at: str,
    temp_run_dir: str | Path,
    final_run_dir: str | Path | None = None,
    error: str | None = None,
) -> Path:
    path = Path(run_dir) / "run_status.json"
    payload = {
        "status": status,
        "created_at": created_at,
        "finished_at": pd.Timestamp.now().isoformat() if status in {"SUCCESS", "FAILED"} else None,
        "temp_run_dir": str(temp_run_dir),
        "final_run_dir": str(final_run_dir or temp_run_dir),
    }
    if error:
        payload["error"] = error
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


@contextlib.contextmanager
def run_status_context(run_dir: str | Path):
    created_at = pd.Timestamp.now().isoformat()
    write_run_status(run_dir, status="RUNNING", created_at=created_at, temp_run_dir=run_dir, final_run_dir=run_dir)
    try:
        yield
    except Exception as exc:
        write_run_status(
            run_dir,
            status="FAILED",
            created_at=created_at,
            temp_run_dir=run_dir,
            final_run_dir=run_dir,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        write_run_status(run_dir, status="SUCCESS", created_at=created_at, temp_run_dir=run_dir, final_run_dir=run_dir)


def save_backtest_outputs(
    *,
    result: dict[str, Any],
    run_dir: str | Path,
    config_used: dict[str, Any],
    required: dict[str, Any],
    cache_status: dict[str, Any],
) -> dict[str, str]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    output_mode = str(config_used.get("config", {}).get("backtest", {}).get("output_mode", "full")).lower()
    full_output = output_mode != "fast"
    performance_summary, _ = compute_performance_outputs(result)
    result["_performance_summary"] = performance_summary
    weights = _weights_from_rebalance_results(result.get("rebalance_results", []))
    files_to_save: list[tuple[str, str]] = [
        ("nav", "nav.csv"),
        ("rebalance_summary", "rebalance_summary.csv"),
        ("rebalance_debug", "rebalance_debug.csv"),
        ("regime_summary", "regime_summary.csv"),
        ("regime_weight_resolution", "regime_weight_resolution.csv"),
        ("allocation_gate_summary", "allocation_gate_summary.csv"),
        ("benchmarks", "benchmarks.csv"),
        ("config", "engine_config.csv"),
    ]
    if full_output:
        files_to_save = [
            ("nav", "nav.csv"),
            ("trades", "trades.csv"),
            ("holdings", "holdings.csv"),
            ("position_snapshots", "position_snapshots.csv"),
            ("corporate_actions", "corporate_actions.csv"),
            ("missing_execution_prices", "missing_execution_prices.csv"),
            ("post_score_adjustment_debug", "post_score_adjustment_debug.csv"),
            ("sector_constraint_debug", "sector_constraint_debug.csv"),
            ("sector_exposure_by_rebalance", "sector_exposure_by_rebalance.csv"),
            ("holding_age_debug", "holding_age_debug.csv"),
            ("top_holdings_by_rebalance", "top_holdings_by_rebalance.csv"),
        ] + files_to_save

    for key, filename in files_to_save:
            frame = result.get(key)
            if isinstance(frame, pd.DataFrame):
                path = run_path / filename
                _frame_to_csv(frame, path)
                paths[key] = str(path)
    weights_path = run_path / "weights.csv"
    _frame_to_csv(weights, weights_path)
    paths["weights"] = str(weights_path)
    paths.update(save_performance_outputs(result, run_path))
    if full_output:
        paths.update(save_factor_outputs(result, run_path))
        paths.update(save_position_outputs(result, run_path))
        paths.update(save_risk_gate_outputs(result, run_path))

    nav = result.get("nav", pd.DataFrame())
    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "final_nav": float(nav["nav"].iloc[-1]) if isinstance(nav, pd.DataFrame) and not nav.empty else None,
        "nav_rows": int(len(nav)) if isinstance(nav, pd.DataFrame) else 0,
        "trade_rows": int(len(result.get("trades", pd.DataFrame()))),
        "holding_rows": int(len(result.get("holdings", pd.DataFrame()))),
        "rebalance_debug_rows": int(len(result.get("rebalance_debug", pd.DataFrame()))),
        "weight_rows": int(len(weights)),
        "output_mode": output_mode,
        "required_range": required,
        "cache_status": cache_status,
        "performance_summary": performance_summary.to_dict(orient="records") if isinstance(performance_summary, pd.DataFrame) else [],
        "risk_gate_summary": pd.read_csv(paths["risk_gate_summary"]).to_dict(orient="records") if "risk_gate_summary" in paths else [],
        "regime_summary_rows": int(len(result.get("regime_summary", pd.DataFrame()))),
        "regime_weight_resolution_rows": int(len(result.get("regime_weight_resolution", pd.DataFrame()))),
        "allocation_gate_summary_rows": int(len(result.get("allocation_gate_summary", pd.DataFrame()))),
    }
    (run_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (run_path / "config_used.json").write_text(json.dumps(config_used, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    paths["summary"] = str(run_path / "summary.json")
    paths["config_used"] = str(run_path / "config_used.json")
    return paths


def run_local_backtest(
    *,
    start: str,
    end: str,
    api_file: str | Path = "api.txt",
    config_file: str | Path | None = None,
    data_dir: str | Path = "data_cache",
    runs_dir: str | Path = "backtest_runs",
    frequency: str = "monthly",
    force_rebuild_cache: bool = False,
) -> Path:
    config = _load_config(config_file)
    strategy_cfg = dict(config.get("strategy", {}))
    strategy_mode = str(strategy_cfg.get("mode", "legacy_composite")).lower()
    run_dir = create_run_dir(runs_dir)
    log_path = run_dir / "logs.txt"
    run_started_at = pd.Timestamp.now()
    with run_status_context(run_dir), tee_stdout_to_file(log_path):
        with TeeLogger() as logger:
            logger.log(f"result_dir={run_dir}")
            logger.log(f"user_requested_range={start} to {end}")
            if strategy_mode != "legacy_composite":
                raise RuntimeError(f"Unsupported strategy mode in this repository: {strategy_mode}")
            credentials = load_api_credentials(api_file)
            logger.log(f"api_credentials_loaded file={api_file} keys={sorted(credentials)} token_redacted=True")

            bt_config = build_backtest_config(config)
            history_overrides = dict(config.get("history_overrides", {}))
            history_overrides.setdefault("enabled_factors", bt_config.enabled_factors)
            history_overrides.setdefault("use_risk_gating", bt_config.use_risk_gating)
            required = infer_required_history_range(start, end, history_overrides=history_overrides)
            logger.log(
                "required_range "
                f"effective_backtest_range={required['backtest_start'].date()}..{required['backtest_end'].date()} "
                f"request_range={required['request_start'].date()}..{required['request_end'].date()} "
                f"history_days={required['required_history_days']} "
                f"bulk_months={required['bulk_months']}"
            )

            cache_cfg = dict(config.get("cache", {}))
            if force_rebuild_cache:
                cache_cfg["force_rebuild_cache"] = True
            data_path = Path(data_dir or cache_cfg.get("data_dir", "data_cache"))
            save_format = str(cache_cfg.get("save_format", "parquet"))
            api_config = JQuantsApiConfig(
                **credentials,
                api_version=str(cache_cfg.get("api_version", "v2")),
                base_url=str(cache_cfg.get("base_url", "https://api.jquants.com/v2")),
                include_margin=bool(cache_cfg.get("include_margin", True)),
                margin_dataset=str(cache_cfg.get("margin_dataset", "weekly_margin_interest")),
                margin_optional=bool(cache_cfg.get("margin_optional", True)),
                allow_missing_market_cap=bool(cache_cfg.get("allow_missing_market_cap", True)),
                market_cap_fallback=str(cache_cfg.get("market_cap_fallback", "liquidity_proxy")),
                force_rebuild_cache=bool(cache_cfg.get("force_rebuild_cache", False)),
                include_financials=bool(cache_cfg.get("include_financials", True)),
                include_topix_index=bool(cache_cfg.get("include_topix_index", True)),
                include_nikkei_index=bool(cache_cfg.get("include_nikkei_index", True)),
                additional_index_codes=tuple(str(code).upper() for code in cache_cfg.get("additional_index_codes", [])),
                save_format=save_format,
                request_sleep_sec=float(cache_cfg.get("request_sleep_sec", 1.0)),
                rate_limit_sleep_sec=float(cache_cfg.get("rate_limit_sleep_sec", 300.0)),
                rate_limit_max_cooldowns=int(cache_cfg.get("rate_limit_max_cooldowns", 3)),
                extra_history_days=int(required["required_history_days"]),
            )

            strict_cache_validation = bool(cache_cfg.get("strict_cache_validation", False))
            debug_dump_cache = bool(cache_cfg.get("debug_dump_cache_summary", False))
            cache_status = check_local_cache(
                data_path,
                required,
                save_format=api_config.save_format,
                strict_validation=strict_cache_validation,
                include_financials=api_config.include_financials,
                include_margin=api_config.include_margin,
                include_topix_index=api_config.include_topix_index,
                include_nikkei_index=api_config.include_nikkei_index,
                additional_index_codes=api_config.additional_index_codes,
            )
            if cache_status.get("invalid_tables") and not api_config.force_rebuild_cache:
                for dataset in cache_status.get("invalid_datasets", []):
                    reason = cache_status.get("invalid_reasons", {}).get(dataset, "")
                    logger.log(f"cache invalid dataset={dataset} reason={reason}")
                if _cache_invalid_requires_force_rebuild(cache_status):
                    logger.log(f"invalid cache values detected; enabling force_rebuild_cache invalid_tables={cache_status.get('invalid_tables')}")
                    api_config.force_rebuild_cache = True
                else:
                    logger.log(
                        "cache coverage gap detected; keeping existing cache and "
                        f"incrementally downloading missing data invalid_tables={cache_status.get('invalid_tables')}"
                    )
            logger.log(
                f"cache_status hit={cache_status['cache_hit']} "
                f"missing_tables={cache_status['missing_tables']} "
                f"missing_price_shards={cache_status['missing_price_shards']}"
            )
            logger.log(f"cache_table_ranges={cache_status.get('table_ranges')}")
            if not cache_status["cache_hit"]:
                download_missing_data(
                    start=required["backtest_start"],
                    end=required["request_end"],
                    data_dir=data_path,
                    api_config=api_config,
                    logger=logger,
                )
                cache_status = check_local_cache(
                    data_path,
                    required,
                    save_format=api_config.save_format,
                    strict_validation=strict_cache_validation,
                    include_financials=api_config.include_financials,
                    include_margin=api_config.include_margin,
                    include_topix_index=api_config.include_topix_index,
                    include_nikkei_index=api_config.include_nikkei_index,
                    additional_index_codes=api_config.additional_index_codes,
                )
                logger.log(
                    f"cache_status_after_download hit={cache_status['cache_hit']} "
                    f"missing_tables={cache_status['missing_tables']} "
                    f"missing_price_shards={cache_status['missing_price_shards']}"
                )
                logger.log(f"cache_table_ranges_after_download={cache_status.get('table_ranges')}")
                if cache_status.get("invalid_datasets"):
                    for dataset in cache_status.get("invalid_datasets", []):
                        reason = cache_status.get("invalid_reasons", {}).get(dataset, "")
                        logger.log(f"cache still invalid dataset={dataset} reason={reason}")
                else:
                    logger.log("cache rebuild validation passed")
            else:
                logger.log("cache hit: using local cache without J-Quants network calls")

            manifest_path = update_manifest(data_path, required, cache_status, config)
            logger.log(f"manifest_updated path={manifest_path}")
            if debug_dump_cache:
                debug_dump_cache_summary(
                    data_path,
                    required_tables=("universe", "sector", "market_cap", "prices"),
                    asof_date=required["backtest_start"],
                )
            else:
                logger.log("cache_summary_skipped debug_dump_cache_summary=False")

            provider = load_provider_from_local_cache(data_path)
            logger.log("provider_loaded_from_local_cache")
            rules = build_rules(config)
            initial_capital = float(config.get("initial_capital", 100_000_000.0))
            if frequency == "weekly":
                rebalance_dates = weekly_rebalance_dates(required["backtest_start"], required["backtest_end"], provider)
            elif frequency == "biweekly":
                rebalance_dates = biweekly_rebalance_dates(required["backtest_start"], required["backtest_end"], provider)
            else:
                rebalance_dates = monthly_rebalance_dates(required["backtest_start"], required["backtest_end"], provider)
            logger.log(f"rebalance_dates count={len(rebalance_dates)} frequency={frequency}")
            if not rebalance_dates:
                raise RuntimeError("No rebalance dates found in local price cache for the effective backtest range.")

            result = run_backtest(rebalance_dates, provider=provider, rules=rules, config=bt_config, initial_capital=initial_capital)
            config_used = {
                "cli": {"start": start, "end": end, "api_file": str(api_file), "data_dir": str(data_path), "runs_dir": str(runs_dir), "frequency": frequency},
                "config_file": str(config_file) if config_file else None,
                "config": config,
            }
            output_paths = save_backtest_outputs(
                result=result,
                run_dir=run_dir,
                config_used=config_used,
                required=required,
                cache_status=cache_status,
            )
            performance_summary = result.get("_performance_summary", pd.DataFrame())
            if isinstance(performance_summary, pd.DataFrame):
                print(performance_log_summary(performance_summary, run_started_at, run_dir), flush=True)
            logger.log(f"outputs_saved run_dir={run_dir} files={output_paths}")
            return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local JP equity historical backtest with J-Quants local cache.")
    default_root = Path(__file__).resolve().parent
    parser.add_argument("--start", required=True, help="User requested backtest start date, e.g. 2025-04-01")
    parser.add_argument("--end", required=True, help="User requested backtest end date, e.g. 2026-03-31")
    parser.add_argument("--api-file", default=str(default_root / "api.txt"), help="Path to api.txt. Default: backtest/api.txt.")
    parser.add_argument("--config", default=None, help="Optional JSON/YAML config file.")
    parser.add_argument("--data-dir", default=str(default_root / "data_cache"), help="Local cache directory.")
    parser.add_argument("--runs-dir", default=str(default_root / "backtest_runs"), help="Backtest result root directory.")
    parser.add_argument("--frequency", choices=["monthly", "biweekly", "weekly"], default="monthly", help="Rebalance frequency.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Delete provider market-data cache files/shards before downloading.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = run_local_backtest(
        start=args.start,
        end=args.end,
        api_file=args.api_file,
        config_file=args.config,
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        frequency=args.frequency,
        force_rebuild_cache=bool(args.force_rebuild_cache),
    )
    print(f"BACKTEST_RUN_DIR={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
