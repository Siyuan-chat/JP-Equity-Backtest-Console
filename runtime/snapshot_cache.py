from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from topix_pool_runtime import RunReport, ScreenRunResult, ScreenRules


CACHE_VERSION = "stock_pool_snapshot_cache_v4_corporate_action_continuity"


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
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


def _provider_stock_pool_fingerprint(provider: Any) -> dict[str, Any]:
    """Return a stable local-data namespace for stock-pool cache reuse.

    Snapshot cache is keyed by anchor date and stock-pool rules.  It should be
    reusable across overlapping runs after new local shards are appended.  Using
    whole-file size/mtime here would invalidate old snapshots whenever the cache
    directory is incrementally extended, even though the point-in-time inputs for
    an already-computed anchor date did not change.
    """

    data_dir = getattr(provider, "data_dir", "")
    return {"provider_name": getattr(provider, "provider_name", ""), "data_dir": str(data_dir)}


def build_snapshot_cache_paths(
    cache_dir: str | Path,
    *,
    anchor_date: pd.Timestamp,
    rules: ScreenRules,
    provider: Any,
) -> dict[str, Any]:
    anchor = pd.Timestamp(anchor_date).normalize()
    rules_payload = rules.to_dict() if hasattr(rules, "to_dict") else _json_safe(rules)
    payload = {
        "cache_version": CACHE_VERSION,
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "rules": rules_payload,
        "stock_pool_input_data": _provider_stock_pool_fingerprint(provider),
    }
    cache_key = _stable_hash(payload)[:20]
    root = Path(cache_dir)
    date_dir = root / anchor.strftime("%Y-%m-%d")
    stem = f"{anchor.strftime('%Y-%m-%d')}_{cache_key}"
    return {
        "cache_key": cache_key,
        "date_dir": date_dir,
        "result_path": date_dir / f"{stem}_selected.parquet",
        "snapshot_path": date_dir / f"{stem}_snapshot.parquet",
        "metadata_path": date_dir / f"{stem}_metadata.json",
    }


def load_snapshot_cache(paths: Mapping[str, Any], anchor_date: pd.Timestamp) -> ScreenRunResult | None:
    result_path = Path(paths["result_path"])
    snapshot_path = Path(paths["snapshot_path"])
    metadata_path = Path(paths["metadata_path"])
    if not (result_path.exists() and snapshot_path.exists() and metadata_path.exists()):
        return None
    try:
        result = pd.read_parquet(result_path)
        snapshot = pd.read_parquet(snapshot_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        report = RunReport(mode="historical_stock_pool_cache", anchor_date=str(pd.Timestamp(anchor_date).date()))
        report.bump("snapshot_rows", int(len(snapshot)))
        report.bump("selected_rows", int(len(result)))
        report.metadata.update(
            {
                "snapshot_cache_status": "hit",
                "snapshot_cache_key": paths["cache_key"],
                "snapshot_cache_metadata": metadata,
            }
        )
        return ScreenRunResult(result=result, snapshot=snapshot, report=report)
    except Exception as exc:
        print(
            f"[snapshot-cache] invalid date={pd.Timestamp(anchor_date).date()} "
            f"key={paths.get('cache_key')} reason={type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def save_snapshot_cache(paths: Mapping[str, Any], run: ScreenRunResult, anchor_date: pd.Timestamp, rules: ScreenRules) -> None:
    result = run.result if isinstance(run.result, pd.DataFrame) else pd.DataFrame()
    snapshot = run.snapshot if isinstance(run.snapshot, pd.DataFrame) else pd.DataFrame()
    if result.empty or snapshot.empty:
        return
    date_dir = Path(paths["date_dir"])
    date_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(Path(paths["result_path"]), index=False)
    snapshot.to_parquet(Path(paths["snapshot_path"]), index=False)
    metadata = {
        "cache_version": CACHE_VERSION,
        "anchor_date": pd.Timestamp(anchor_date).strftime("%Y-%m-%d"),
        "cache_key": paths["cache_key"],
        "selected_rows": int(len(result)),
        "snapshot_rows": int(len(snapshot)),
        "rules": rules.to_dict() if hasattr(rules, "to_dict") else _json_safe(rules),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    Path(paths["metadata_path"]).write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    append_snapshot_cache_manifest(date_dir.parent, metadata)


def append_snapshot_cache_manifest(cache_dir: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(cache_dir) / "manifest.csv"
    frame = pd.DataFrame([dict(row)])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
