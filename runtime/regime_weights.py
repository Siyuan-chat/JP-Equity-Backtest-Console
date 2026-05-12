from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from factor_composite import WeightConfig


DEFAULT_WEIGHT_SOURCE_MODE = "manual"

_WEIGHT_FIELD_NAMES = tuple(asdict(WeightConfig()).keys())


def default_weight_dict() -> dict[str, float]:
    return {key: float(value) for key, value in asdict(WeightConfig()).items()}


def weight_dict_from_config(weights: WeightConfig | dict[str, Any] | None) -> dict[str, float]:
    if isinstance(weights, WeightConfig):
        return {key: float(value) for key, value in asdict(weights).items()}
    if isinstance(weights, dict):
        normalized = normalize_weight_keys(weights)
        return {key: float(value) for key, value in normalized.items()}
    return default_weight_dict()


def normalize_weight_keys(weight_dict: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(weight_dict, dict):
        return {}
    alias_map = {
        "wq": "w_q",
        "wv": "w_v",
        "wm": "w_m",
        "wts": "w_ts",
        "wb": "w_b",
        "wt": "w_t",
        "wr": "w_r",
        "wrev": "w_r",
    }
    out: dict[str, float] = {}
    for raw_key, raw_value in weight_dict.items():
        key = "".join(ch for ch in str(raw_key).strip().lower() if ch.isalnum())
        canonical = alias_map.get(key)
        if canonical is None:
            continue
        try:
            value = float(raw_value)
        except Exception:
            continue
        out[canonical] = value
    return out


def merge_weights(base: dict[str, float], overlay: dict[str, float]) -> dict[str, float]:
    out = dict(base)
    for key in _WEIGHT_FIELD_NAMES:
        if key in overlay:
            out[key] = float(overlay[key])
    return out


def load_weights_from_json(json_path: str | Path) -> dict[str, Any]:
    path = Path(json_path)
    if not path.is_absolute() and not path.exists():
        candidate = Path(__file__).resolve().parent / path
        if candidate.exists():
            path = candidate
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("weight json must contain an object")
    normalized = normalize_weight_keys(payload)
    if not normalized:
        raise ValueError("weight json contains no recognized weight fields")
    return {
        "path": str(path),
        "raw": payload,
        "weights": normalized,
        "recognized_keys": sorted(normalized),
        "unknown_keys": sorted(str(key) for key in payload.keys() if str(key) not in normalized.keys()),
    }


def resolve_weights(
    *,
    regime: str | None,
    mode: str | None,
    gui_weights: WeightConfig | dict[str, Any] | None,
    regime_json_map: dict[str, str] | None,
    default_weights: WeightConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_mode = str(mode or DEFAULT_WEIGHT_SOURCE_MODE).strip().lower()
    default_dict = weight_dict_from_config(default_weights)
    gui_dict = merge_weights(default_dict, weight_dict_from_config(gui_weights))
    result = {
        "resolved_weights": dict(gui_dict),
        "source_mode": source_mode,
        "source": "gui_manual",
        "regime": regime or "",
        "json_path": "",
        "used_fallback": False,
        "warnings": [],
        "recognized_json_keys": [],
    }
    if source_mode != "json":
        return result

    mapping = dict(regime_json_map or {})
    json_path = mapping.get(str(regime or ""))
    if not json_path:
        result["source"] = "fallback_gui_manual"
        result["used_fallback"] = True
        result["warnings"].append(f"no json mapping configured for regime={regime}")
        return result

    result["json_path"] = str(json_path)
    try:
        payload = load_weights_from_json(json_path)
    except Exception as exc:
        result["source"] = "fallback_gui_manual"
        result["used_fallback"] = True
        result["warnings"].append(f"failed to load json path={json_path}: {type(exc).__name__}: {exc}")
        return result

    merged = merge_weights(gui_dict, payload["weights"])
    result["resolved_weights"] = merged
    result["source"] = "json"
    result["recognized_json_keys"] = payload["recognized_keys"]
    return result


def to_weight_config(weight_dict: dict[str, Any]) -> WeightConfig:
    normalized = merge_weights(default_weight_dict(), normalize_weight_keys(weight_dict))
    return WeightConfig(**normalized)
