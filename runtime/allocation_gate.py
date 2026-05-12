from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from factor_composite import WeightConfig
from regime_weights import resolve_weights


DEFAULT_ALLOCATION_GATE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "mode": "matrix",
    "risk_override_threshold": 3,
    "severe_risk_score_threshold": 4,
    "defensive_regime": "riskoff_highvol",
}

DEFAULT_REGIME_POSITION_PROFILES: dict[str, dict[str, Any]] = {
    "trend_lowvol": {
        "gross_multiplier": 1.00,
        "equity_multiplier": 1.00,
        "cash_floor": 0.00,
        "short_gross_multiplier": 0.60,
        "turnover_multiplier": 1.00,
        "allow_new_longs": True,
        "allow_new_shorts": True,
    },
    "range_lowvol": {
        "gross_multiplier": 0.70,
        "equity_multiplier": 0.75,
        "cash_floor": 0.10,
        "short_gross_multiplier": 0.80,
        "turnover_multiplier": 0.85,
        "allow_new_longs": True,
        "allow_new_shorts": True,
    },
    "riskoff_highvol": {
        "gross_multiplier": 0.35,
        "equity_multiplier": 0.40,
        "cash_floor": 0.35,
        "short_gross_multiplier": 1.00,
        "turnover_multiplier": 0.55,
        "allow_new_longs": False,
        "allow_new_shorts": True,
    },
    "hot_riskon": {
        "gross_multiplier": 1.00,
        "equity_multiplier": 1.00,
        "cash_floor": 0.00,
        "short_gross_multiplier": 0.35,
        "turnover_multiplier": 1.10,
        "allow_new_longs": True,
        "allow_new_shorts": True,
    },
    "macro_reflation": {
        "gross_multiplier": 0.90,
        "equity_multiplier": 0.95,
        "cash_floor": 0.05,
        "short_gross_multiplier": 0.45,
        "turnover_multiplier": 0.95,
        "allow_new_longs": True,
        "allow_new_shorts": True,
    },
}


@dataclass
class AllocationGateDecision:
    signal_date: pd.Timestamp
    gate_enabled: bool = True
    gate_mode: str = "matrix"
    gate_state: str = ""
    raw_regime: str = ""
    active_regime: str = ""
    weight_regime: str = ""
    position_regime: str = ""
    position_profile: str = ""
    defensive_regime: str = "riskoff_highvol"
    risk_score_0_5: int | None = None
    risk_score_effective: int | None = None
    risk_on: bool | None = None
    trend_on: bool | None = None
    risk_override_applied: bool = False
    severe_risk_active: bool = False
    gross_multiplier: float = 1.0
    equity_multiplier: float = 1.0
    cash_floor: float = 0.0
    short_gross_multiplier: float = 1.0
    allow_new_longs: bool = True
    allow_new_shorts: bool = True
    turnover_multiplier: float = 1.0
    notes: list[str] = field(default_factory=list)
    weight_resolution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_date": self.signal_date,
            "gate_enabled": self.gate_enabled,
            "gate_mode": self.gate_mode,
            "gate_state": self.gate_state,
            "raw_regime": self.raw_regime,
            "active_regime": self.active_regime,
            "weight_regime": self.weight_regime,
            "position_regime": self.position_regime,
            "position_profile": self.position_profile,
            "defensive_regime": self.defensive_regime,
            "risk_score_0_5": self.risk_score_0_5,
            "risk_score_effective": self.risk_score_effective,
            "risk_on": self.risk_on,
            "trend_on": self.trend_on,
            "risk_override_applied": self.risk_override_applied,
            "severe_risk_active": self.severe_risk_active,
            "gross_multiplier": self.gross_multiplier,
            "equity_multiplier": self.equity_multiplier,
            "cash_floor": self.cash_floor,
            "short_gross_multiplier": self.short_gross_multiplier,
            "allow_new_longs": self.allow_new_longs,
            "allow_new_shorts": self.allow_new_shorts,
            "turnover_multiplier": self.turnover_multiplier,
            "notes": " | ".join(self.notes),
        }


def _merged_gate_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_ALLOCATION_GATE_CONFIG)
    if isinstance(config, dict):
        merged.update(config)
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["mode"] = str(merged.get("mode", "matrix") or "matrix").strip().lower()
    merged["risk_override_threshold"] = int(merged.get("risk_override_threshold", 4))
    merged["severe_risk_score_threshold"] = int(merged.get("severe_risk_score_threshold", 5))
    merged["defensive_regime"] = str(merged.get("defensive_regime", "riskoff_highvol") or "riskoff_highvol")
    merged["regime_position_profiles"] = _merge_position_profiles(merged.get("regime_position_profiles"))
    return merged


def _merge_position_profiles(overrides: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    merged = {name: dict(profile) for name, profile in DEFAULT_REGIME_POSITION_PROFILES.items()}
    if not isinstance(overrides, dict):
        return merged
    for regime_name, raw_profile in overrides.items():
        name = str(regime_name or "").strip()
        if not name or not isinstance(raw_profile, dict):
            continue
        base = dict(merged.get(name, {}))
        for key, value in raw_profile.items():
            if key in {"allow_new_longs", "allow_new_shorts"}:
                base[key] = bool(value)
            else:
                try:
                    base[key] = float(value)
                except Exception:
                    continue
        merged[name] = base
    return merged


def _resolve_position_profile(regime: str, gate_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profiles = dict(gate_cfg.get("regime_position_profiles", {}))
    profile_name = str(regime or "").strip()
    profile = dict(profiles.get(profile_name, {}))
    if profile:
        return profile_name, profile
    fallback_name = "range_lowvol"
    return fallback_name, dict(profiles.get(fallback_name, DEFAULT_REGIME_POSITION_PROFILES[fallback_name]))


def _extract_risk_snapshot(risk: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(risk, dict):
        return {
            "risk_score_0_5": None,
            "risk_score_effective": None,
            "risk_on": None,
            "trend_on": None,
            "gross_multiplier": 1.0,
            "equity_multiplier": 1.0,
            "cash_floor": 0.0,
        }
    minimal = risk.get("minimal")
    if not isinstance(minimal, pd.DataFrame) or minimal.empty:
        return {
            "risk_score_0_5": None,
            "risk_score_effective": None,
            "risk_on": None,
            "trend_on": None,
            "gross_multiplier": 1.0,
            "equity_multiplier": 1.0,
            "cash_floor": 0.0,
        }
    row = minimal.iloc[0]
    return {
        "risk_score_0_5": int(row["risk_score_0_5"]) if pd.notna(row.get("risk_score_0_5")) else None,
        "risk_score_effective": int(row["risk_score_effective"]) if pd.notna(row.get("risk_score_effective")) else None,
        "risk_on": bool(row["risk_on"]) if pd.notna(row.get("risk_on")) else None,
        "trend_on": bool(row["trend_on"]) if pd.notna(row.get("trend_on")) else None,
        "gross_multiplier": float(row["gross_exposure_multiplier"]) if pd.notna(row.get("gross_exposure_multiplier")) else 1.0,
        "equity_multiplier": float(row["equity_exposure_multiplier"]) if pd.notna(row.get("equity_exposure_multiplier")) else 1.0,
        "cash_floor": float(row["cash_weight_floor"]) if pd.notna(row.get("cash_weight_floor")) else 0.0,
    }


def resolve_allocation_gate(
    *,
    signal_date: pd.Timestamp,
    regime_result: dict[str, Any] | None,
    risk: dict[str, Any] | None,
    weight_source_mode: str | None,
    gui_weights: WeightConfig | dict[str, Any] | None,
    regime_weight_map: dict[str, str] | None,
    default_weights: WeightConfig | dict[str, Any] | None,
    allocation_gate_config: dict[str, Any] | None = None,
) -> AllocationGateDecision:
    gate_cfg = _merged_gate_config(allocation_gate_config)
    regime_result = dict(regime_result or {})
    risk_snapshot = _extract_risk_snapshot(risk)
    active_regime = str(regime_result.get("active_regime") or "")
    raw_regime = str(regime_result.get("raw_regime") or "")
    risk_score = risk_snapshot["risk_score_effective"]
    defensive_regime = str(gate_cfg["defensive_regime"])
    gate_mode = str(gate_cfg["mode"])
    gate_enabled = bool(gate_cfg["enabled"])
    weight_regime = active_regime
    position_regime = active_regime
    risk_override_applied = False
    severe_risk_active = False
    notes: list[str] = []

    if gate_enabled and gate_mode == "matrix" and risk_score is not None:
        if int(risk_score) >= int(gate_cfg["risk_override_threshold"]):
            weight_regime = defensive_regime
            position_regime = defensive_regime
            risk_override_applied = bool(defensive_regime and defensive_regime != active_regime)
            if risk_override_applied:
                notes.append(f"risk override applied: {active_regime} -> {defensive_regime}")
        if int(risk_score) >= int(gate_cfg["severe_risk_score_threshold"]):
            severe_risk_active = True
            notes.append("severe risk state active")

    gate_state = f"{weight_regime or active_regime or 'manual'}__risk{risk_score if risk_score is not None else 'na'}"
    if severe_risk_active:
        gate_state = "capital_preserve"
    elif risk_override_applied:
        gate_state = f"{weight_regime}__defensive"

    weight_resolution = resolve_weights(
        regime=weight_regime or active_regime,
        mode=weight_source_mode,
        gui_weights=gui_weights,
        regime_json_map=regime_weight_map,
        default_weights=default_weights,
    )
    weight_resolution = dict(weight_resolution)
    position_profile_name, position_profile = _resolve_position_profile(position_regime or weight_regime or active_regime, gate_cfg)
    weight_resolution.update(
        {
            "requested_regime": active_regime,
            "weight_regime": weight_regime or active_regime,
            "position_regime": position_regime or weight_regime or active_regime,
            "position_profile": position_profile_name,
            "gate_mode": gate_mode,
            "gate_state": gate_state,
            "risk_score_effective": risk_score,
            "risk_override_applied": risk_override_applied,
            "severe_risk_active": severe_risk_active,
        }
    )

    regime_gross = float(position_profile.get("gross_multiplier", 1.0))
    regime_equity = float(position_profile.get("equity_multiplier", 1.0))
    regime_cash = float(position_profile.get("cash_floor", 0.0))
    regime_short_gross = float(position_profile.get("short_gross_multiplier", 1.0))
    regime_turnover = float(position_profile.get("turnover_multiplier", 1.0))
    allow_new_longs = bool(position_profile.get("allow_new_longs", True)) and not severe_risk_active
    allow_new_shorts = bool(position_profile.get("allow_new_shorts", True))
    risk_turnover = 0.5 if severe_risk_active else (0.75 if risk_override_applied else 1.0)
    severe_gross_cap = 0.60 if severe_risk_active else 1.0
    severe_equity_cap = 0.60 if severe_risk_active else 1.0
    severe_short_cap = 0.75 if severe_risk_active else 1.0
    severe_cash_floor = 0.50 if severe_risk_active else 0.0
    gross_multiplier = float(risk_snapshot["gross_multiplier"]) * regime_gross * severe_gross_cap
    equity_multiplier = float(risk_snapshot["equity_multiplier"]) * regime_equity * severe_equity_cap
    cash_floor = max(float(risk_snapshot["cash_floor"]), regime_cash, severe_cash_floor)
    short_gross_multiplier = regime_short_gross * severe_short_cap
    turnover_multiplier = regime_turnover * risk_turnover
    notes.append(
        "position profile="
        f"{position_profile_name} gross={gross_multiplier:.2f} "
        f"equity={equity_multiplier:.2f} short={short_gross_multiplier:.2f} cash_floor={cash_floor:.2f}"
    )
    return AllocationGateDecision(
        signal_date=pd.Timestamp(signal_date).normalize(),
        gate_enabled=gate_enabled,
        gate_mode=gate_mode,
        gate_state=gate_state,
        raw_regime=raw_regime,
        active_regime=active_regime,
        weight_regime=weight_regime or active_regime,
        position_regime=position_regime or weight_regime or active_regime,
        position_profile=position_profile_name,
        defensive_regime=defensive_regime,
        risk_score_0_5=risk_snapshot["risk_score_0_5"],
        risk_score_effective=risk_snapshot["risk_score_effective"],
        risk_on=risk_snapshot["risk_on"],
        trend_on=risk_snapshot["trend_on"],
        risk_override_applied=risk_override_applied,
        severe_risk_active=severe_risk_active,
        gross_multiplier=gross_multiplier,
        equity_multiplier=equity_multiplier,
        cash_floor=cash_floor,
        short_gross_multiplier=short_gross_multiplier,
        allow_new_longs=allow_new_longs,
        allow_new_shorts=allow_new_shorts,
        turnover_multiplier=turnover_multiplier,
        notes=notes,
        weight_resolution=weight_resolution,
    )
