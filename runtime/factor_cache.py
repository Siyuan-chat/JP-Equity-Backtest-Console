from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


CACHE_VERSION = "single_factor_cache_v2_overlap_reuse"


@dataclass(frozen=True)
class FactorCacheLookup:
    factor_name: str
    anchor_date: pd.Timestamp
    cache_key: str
    factor_dir: Path
    data_path: Path
    detail_path: Path
    summary_path: Path
    metadata_path: Path

    @property
    def exists(self) -> bool:
        return self.data_path.exists() and self.summary_path.exists() and self.metadata_path.exists()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0])) if _json_safe(v) is not None}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return None
    if callable(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _stable_hash(payload: Any) -> str:
    text = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise_universe_for_hash(universe: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(universe, pd.DataFrame) or universe.empty:
        return pd.DataFrame(columns=["code"])
    out = universe.copy()
    if "code" in out.columns:
        out["code"] = out["code"].astype("string")
        out = out.sort_values("code", kind="mergesort")
    cols = sorted(str(c) for c in out.columns)
    out = out[cols].reset_index(drop=True)
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.fillna("<NA>").astype(str)


def universe_fingerprint(universe: pd.DataFrame) -> str:
    """Stable fingerprint for the exact candidate universe handed to a factor."""

    work = _normalise_universe_for_hash(universe)
    if work.empty:
        return _stable_hash({"rows": 0, "columns": list(work.columns)})
    row_hashes = pd.util.hash_pandas_object(work, index=False).astype(str).tolist()
    return _stable_hash({"rows": len(work), "columns": list(work.columns), "row_hashes": row_hashes})


def provider_data_fingerprint(provider: Any) -> dict[str, Any]:
    """Return a stable local-data namespace fingerprint.

    Factor cache is intentionally reusable across overlapping backtest ranges.
    Appending missing local J-Quants shards changes parquet file mtimes, but it
    does not normally revise already-computed point-in-time factor inputs.  The
    cache key therefore avoids whole-file size/mtime and relies on factor date,
    universe fingerprint, factor parameters, and cache version for validity.
    """

    data_dir = getattr(provider, "data_dir", "")
    out: dict[str, Any] = {"provider_name": getattr(provider, "provider_name", ""), "data_dir": str(data_dir)}
    return out


def build_factor_cache_lookup(
    cache_dir: str | Path,
    *,
    factor_name: str,
    anchor_date: pd.Timestamp,
    universe: pd.DataFrame,
    factor_config: Mapping[str, Any],
    provider: Any,
) -> FactorCacheLookup:
    anchor = pd.Timestamp(anchor_date).normalize()
    payload = {
        "cache_version": CACHE_VERSION,
        "factor_name": factor_name,
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "universe_hash": universe_fingerprint(universe),
        "factor_config": _json_safe(factor_config),
        "input_data": provider_data_fingerprint(provider),
    }
    cache_key = _stable_hash(payload)[:20]
    factor_dir = Path(cache_dir) / factor_name
    stem = f"{anchor.strftime('%Y-%m-%d')}_{cache_key}"
    return FactorCacheLookup(
        factor_name=factor_name,
        anchor_date=anchor,
        cache_key=cache_key,
        factor_dir=factor_dir,
        data_path=factor_dir / f"{stem}_minimal.parquet",
        detail_path=factor_dir / f"{stem}_detail.parquet",
        summary_path=factor_dir / f"{stem}_summary.json",
        metadata_path=factor_dir / f"{stem}_metadata.json",
    )


def load_factor_cache(lookup: FactorCacheLookup) -> dict[str, Any] | None:
    if not lookup.exists:
        return None
    try:
        minimal = pd.read_parquet(lookup.data_path)
        detail = pd.read_parquet(lookup.detail_path) if lookup.detail_path.exists() else pd.DataFrame()
        summary = json.loads(lookup.summary_path.read_text(encoding="utf-8"))
        summary = dict(summary)
        summary["cache_status"] = "hit"
        summary["cache_key"] = lookup.cache_key
        return {"minimal": minimal, "detail": detail, "summary": summary}
    except Exception as exc:
        print(
            f"[factor-cache] invalid factor={lookup.factor_name} date={lookup.anchor_date.date()} "
            f"key={lookup.cache_key} reason={type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def is_cacheable_factor_result(result: Mapping[str, Any]) -> bool:
    summary = result.get("summary", {}) if isinstance(result, Mapping) else {}
    if isinstance(summary, Mapping) and summary.get("error"):
        return False
    minimal = result.get("minimal") if isinstance(result, Mapping) else None
    return isinstance(minimal, pd.DataFrame) and not minimal.empty


def save_factor_cache(lookup: FactorCacheLookup, result: Mapping[str, Any]) -> None:
    if not is_cacheable_factor_result(result):
        return
    lookup.factor_dir.mkdir(parents=True, exist_ok=True)
    minimal = result.get("minimal")
    detail = result.get("detail")
    summary = dict(result.get("summary", {}) or {})
    if not isinstance(minimal, pd.DataFrame):
        return
    minimal.to_parquet(lookup.data_path, index=False)
    (detail if isinstance(detail, pd.DataFrame) else pd.DataFrame()).to_parquet(lookup.detail_path, index=False)
    summary.update({"cache_status": "saved", "cache_key": lookup.cache_key})
    lookup.summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "cache_version": CACHE_VERSION,
        "factor_name": lookup.factor_name,
        "anchor_date": lookup.anchor_date.strftime("%Y-%m-%d"),
        "cache_key": lookup.cache_key,
        "minimal_rows": int(len(minimal)),
        "detail_rows": int(len(detail)) if isinstance(detail, pd.DataFrame) else 0,
        "created_at": pd.Timestamp.now().isoformat(),
        "minimal_path": str(lookup.data_path),
        "detail_path": str(lookup.detail_path),
    }
    lookup.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    append_factor_cache_manifest(lookup.factor_dir.parent, metadata)


def append_factor_cache_manifest(cache_dir: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(cache_dir) / "manifest.csv"
    frame = pd.DataFrame([dict(row)])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
