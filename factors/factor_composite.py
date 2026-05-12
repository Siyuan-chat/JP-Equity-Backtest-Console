"""
Linear factor compositor for JP small-cap research in Google Colab.

What this script does
1. Shows a small Colab UI for weight inputs and formula guidance.
2. Uploads factor CSVs produced by upstream notebooks.
3. Merges factor files on ticker/code.
4. Runs linear scoring:
   Score = w_q * Q + w_v * V + w_m * M + w_ts * TS + w_b * B + w_t * T + w_r * REV
5. Runs QA / freshness / missing-data checks.
6. Exports ranking, QA, and flagged rows CSV files.

Designed for Colab, but most core functions also work in a normal Python session.
"""

from __future__ import annotations

import ast
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import Markdown, HTML, display
except Exception:  # pragma: no cover
    widgets = None
    Markdown = None
    HTML = None
    display = print

try:
    from google.colab import files
except Exception:  # pragma: no cover
    files = None


DEFAULT_CONFIG = {
    "key_candidates": ["code", "Code", "ticker", "Ticker", "stock_code", "symbol"],
    "date_candidates": ["anchor_date", "asof_date", "date", "Date", "signal_month", "signal_date"],
    "industry_candidates": ["industry", "Industry", "canonical_industry"],
    "log_mcap_candidates": ["log_mktcap", "log_market_cap", "log_market_cap_billion_jpy"],
    "winsor_lower_q": 0.02,
    "winsor_upper_q": 0.98,
    "min_required_modules": 4,
    "strict_freshness_mode": False,
    "strict_missing_mode": False,
    "price_factor_max_lag_days": 7,
    "fundamental_factor_max_lag_days": 180,
    "macro_factor_max_lag_days": 31,
    "top_n_preview": 20,
    "bottom_n_preview": 10,
    "adaptive_missing_threshold": 100,
    "adaptive_factor_coverage_threshold": 0.60,
    "suppress_reversal_when_negative_mgate": False,
    "f_penalty_when_nonpositive_mgate": 1.0,
    "score_formula": None,
}


FACTOR_SPECS = {
    "quality": {
        "label": "Quality (Q)",
        "group": "fundamental",
        "column_candidates": ["QualityScore", "quality_score", "Q", "q_score"],
        "required": True,
    },
    "value": {
        "label": "Value (V)",
        "group": "fundamental",
        "column_candidates": ["ValueScore", "value_score", "V", "v_score"],
        "required": True,
    },
    "momentum": {
        "label": "12-1 Momentum (M)",
        "group": "price",
        "column_candidates": [
            "resid_ir_neutralized_score",
            "resid_tstat_neutralized_score",
            "resid_cumret_neutralized_score",
            "M",
            "momentum_score",
        ],
        "required": True,
    },
    "dual_ma": {
        "label": "Dual-MA Gate (TS)",
        "group": "price",
        "column_candidates": ["dual_ma_gate", "ts_gate", "TS", "trend_gate", "ma_gate"],
        "required": False,
    },
    "reversal": {
        "label": "Reversal (REV)",
        "group": "price",
        "column_candidates": ["reversal_factor_zscore", "reversal_zscore", "REV", "reversal_score"],
        "required": True,
    },
    "behaviour": {
        "label": "Behaviour Crowding (B)",
        "group": "price",
        "column_candidates": ["credit_residual_factor", "behaviour_score", "B", "behavior_score"],
        "required": True,
    },
    "attention": {
        "label": "Attention Crowding (T)",
        "group": "price",
        "column_candidates": ["attention_raw", "attention_factor", "attention_score", "T"],
        "required": True,
    },
}


@dataclass
class WeightConfig:
    w_q: float = 0.20
    w_v: float = 0.20
    w_m: float = 0.20
    w_ts: float = 0.15
    w_b: float = 0.10
    w_t: float = 0.10
    w_r: float = 0.05


def validate_weights(weights: WeightConfig) -> None:
    for name in ["w_q", "w_v", "w_m", "w_ts", "w_b", "w_t", "w_r"]:
        value = float(getattr(weights, name))
        if not np.isfinite(value):
            raise ValueError(f"Weight '{name}' must be finite. Got {value!r}.")
        if value < 0:
            raise ValueError(f"Weight '{name}' must be non-negative. Got {value!r}.")


def normalize_main_weights(weights: WeightConfig) -> WeightConfig:
    total = float(weights.w_q + weights.w_v + weights.w_m + weights.w_ts + weights.w_b + weights.w_t + weights.w_r)
    if not np.isfinite(total) or abs(total) < 1e-12:
        return WeightConfig()
    return WeightConfig(
        w_q=float(weights.w_q) / total,
        w_v=float(weights.w_v) / total,
        w_m=float(weights.w_m) / total,
        w_ts=float(weights.w_ts) / total,
        w_b=float(weights.w_b) / total,
        w_t=float(weights.w_t) / total,
        w_r=float(weights.w_r) / total,
    )


def _safe_display(obj) -> None:
    if display is print:
        print(obj)
    else:
        display(obj)


def show_formula_help() -> None:
    text = """
## Factor Composite Formula

Default linear structure:

```text
Score = w_q * Q + w_v * V + w_m * M + w_ts * TS + w_b * B + w_t * T + w_r * REV
```

Meaning of each symbol:

- `Q`: quality factor score
- `V`: value factor score
- `M`: standard 12-1 momentum score
- `TS`: dual moving-average trend-strength score or gate
- `B`: behaviour crowding, usually credit residual / margin-driven crowding
- `T`: attention crowding, usually abnormal volume + turnover style heat
- `REV`: short-term reversal, used as a light timing optimizer

Recommended interpretation:

- Higher `Q`, `V`, `M`, `TS`, `B`, `T`, and `REV` contribute more when their weights are positive
- If you want a penalty-style component, express that directly in the formula with a minus sign
- Higher `REV` is better

Practical note:

- `SizeScore` is not used directly in the total score in this version.
- `dual_ma` is treated as its own additive input in the default formula.
- Rows with missing `TS` are still called out in QA outputs.
"""
    _safe_display(Markdown(text) if Markdown else text)


def normalize_code(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if re.fullmatch(r"\d{4}", text):
        return text
    if re.fullmatch(r"\d{4}\.T", upper):
        return text[:4]
    if re.fullmatch(r"\d{5}", text):
        return text[:4]
    if re.search(r"\d{5,}", text):
        return None
    return None


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def winsorize_series(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series.copy()
    lower = valid.quantile(lower_q)
    upper = valid.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def zscore_series(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - valid.mean()) / std


def neutralize_series(
    values: pd.Series,
    industry: Optional[pd.Series] = None,
    log_mcap: Optional[pd.Series] = None,
) -> pd.Series:
    work = pd.DataFrame({"y": values})
    features: List[np.ndarray] = [np.ones(len(work))]

    if log_mcap is not None:
        work["log_mcap"] = pd.to_numeric(log_mcap, errors="coerce")
        log_mcap_mean = work["log_mcap"].mean()
        if pd.notna(log_mcap_mean):
            features.append(work["log_mcap"].fillna(log_mcap_mean).to_numpy())

    if industry is not None:
        ind = industry.astype("string").fillna("__MISSING__")
        dummies = pd.get_dummies(ind, prefix="ind", dummy_na=False, drop_first=True)
        if not dummies.empty:
            features.append(dummies.to_numpy())

    if len(features) == 1:
        return values.copy()

    valid_mask = work["y"].notna()
    if valid_mask.sum() < 5:
        return values.copy()

    x = np.column_stack([feat[valid_mask.to_numpy()] for feat in features])
    y = work.loc[valid_mask, "y"].to_numpy(dtype=float)

    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ coef
        resid = y - fitted
        out = pd.Series(np.nan, index=values.index, dtype=float)
        out.loc[valid_mask] = resid
        return out
    except Exception:
        return values.copy()


def preprocess_factor(
    series: pd.Series,
    lower_q: float,
    upper_q: float,
    industry: Optional[pd.Series],
    log_mcap: Optional[pd.Series],
    do_winsorize: bool = True,
    do_neutralize: bool = True,
    do_zscore: bool = True,
) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if do_winsorize:
        out = winsorize_series(out, lower_q, upper_q)
    if do_neutralize:
        out = neutralize_series(out, industry=industry, log_mcap=log_mcap)
    if do_zscore:
        out = zscore_series(out)
    return out


def parse_any_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def infer_factor_date_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df.columns, DEFAULT_CONFIG["date_candidates"])


def infer_composite_factor_date_col(df: pd.DataFrame, factor_name: str) -> Optional[str]:
    group = FACTOR_SPECS[factor_name]["group"]
    if group == "price":
        price_candidates = ["freshness_date", "data_end_date", "asof_date", "anchor_date", "date", "Date", "signal_date", "signal_month"]
        return first_existing(df.columns, price_candidates)
    if group == "fundamental":
        fundamental_candidates = ["freshness_date", "data_end_date", "asof_date", "anchor_date", "date", "Date", "signal_date", "signal_month"]
        return first_existing(df.columns, fundamental_candidates)
    return infer_factor_date_col(df)


def infer_key_col(df: pd.DataFrame) -> str:
    key_col = first_existing(df.columns, DEFAULT_CONFIG["key_candidates"])
    if key_col is None:
        raise ValueError(f"Could not infer merge key. Candidate columns: {DEFAULT_CONFIG['key_candidates']}")
    return key_col


def infer_factor_col(df: pd.DataFrame, factor_name: str) -> Optional[str]:
    return first_existing(df.columns, FACTOR_SPECS[factor_name]["column_candidates"])


def standardize_key_col(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    out = df.copy()
    out["code"] = out[key_col].map(normalize_code)
    out = out.loc[out["code"].notna()].copy()
    out["code"] = out["code"].astype(str)
    return out


def deduplicate_latest_by_code(df: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    out = df.copy()
    date_col = f"{factor_name}_date"
    if date_col in out.columns and out[date_col].notna().any():
        out = out.sort_values(["code", date_col], ascending=[True, True], na_position="first")
        return out.drop_duplicates(subset=["code"], keep="last").copy()
    # Fallback when no usable factor date is available: retain prior file-order behavior explicitly.
    return out.drop_duplicates(subset=["code"], keep="last").copy()


def coalesce_metadata_columns(df: pd.DataFrame, field: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c == field or c.startswith(f"{field}_")]
    if not cols:
        return df
    out = df.copy()
    if field == "industry":
        merged = pd.Series(pd.NA, index=out.index, dtype="string")
        for col in cols:
            merged = merged.fillna(out[col].astype("string"))
        out[field] = merged
    else:
        merged = pd.Series(np.nan, index=out.index, dtype=float)
        for col in cols:
            merged = merged.fillna(pd.to_numeric(out[col], errors="coerce"))
        out[field] = merged
    drop_cols = [c for c in cols if c != field]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def read_uploaded_csv(prompt_text: str) -> Tuple[str, pd.DataFrame]:
    if files is None:
        raise RuntimeError("Native upload is only available in Google Colab.")
    print(prompt_text)
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file was uploaded.")
    filename = next(iter(uploaded))
    content = uploaded[filename]
    frame = pd.read_csv(io.BytesIO(content))
    return filename, frame


def upload_factor_bundle() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    factor_frames: Dict[str, pd.DataFrame] = {}
    upload_log_rows: List[Dict[str, object]] = []

    for factor_name, spec in FACTOR_SPECS.items():
        prompt = f"Upload CSV for {spec['label']} ({factor_name}). Skip only if optional."
        try:
            filename, df = read_uploaded_csv(prompt)
        except Exception as exc:
            if spec["required"]:
                raise
            print(f"Optional upload skipped for {factor_name}: {exc}")
            continue

        key_col = infer_key_col(df)
        date_col = infer_composite_factor_date_col(df, factor_name)
        factor_col = infer_factor_col(df, factor_name)
        if factor_col is None:
            raise ValueError(
                f"Could not infer factor column for '{factor_name}'. "
                f"Expected one of: {FACTOR_SPECS[factor_name]['column_candidates']}"
            )

        clean = standardize_key_col(df, key_col)
        clean = clean.rename(columns={factor_col: f"{factor_name}_raw"})

        if date_col is not None and date_col in clean.columns:
            clean[f"{factor_name}_date"] = parse_any_date(clean[date_col])
        else:
            clean[f"{factor_name}_date"] = pd.NaT
        clean = deduplicate_latest_by_code(clean, factor_name)

        for extra_name, candidates in [
            ("industry", DEFAULT_CONFIG["industry_candidates"]),
            ("log_mcap", DEFAULT_CONFIG["log_mcap_candidates"]),
        ]:
            col = first_existing(clean.columns, candidates)
            if col is not None and extra_name not in clean.columns:
                clean[extra_name] = clean[col]

        keep_cols = ["code", f"{factor_name}_raw", f"{factor_name}_date"]
        for extra in ["industry", "log_mcap"]:
            if extra in clean.columns:
                keep_cols.append(extra)
        factor_frames[factor_name] = clean[keep_cols].copy()

        upload_log_rows.append(
            {
                "factor_name": factor_name,
                "file_name": filename,
                "rows": len(df),
                "usable_rows": len(clean),
                "key_col": key_col,
                "factor_col": factor_col,
                "date_col": date_col or "",
            }
        )

    return factor_frames, pd.DataFrame(upload_log_rows)


def merge_factor_frames(factor_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for factor_name, df in factor_frames.items():
        tmp = df.copy()
        if merged is None:
            merged = tmp
            continue
        merged = merged.merge(tmp, on="code", how="outer")

    if merged is None:
        raise ValueError("No factor files were uploaded.")

    for shared in ["industry", "log_mcap"]:
        merged = coalesce_metadata_columns(merged, shared)

    if "industry" not in merged.columns:
        merged["industry"] = pd.NA
    if "log_mcap" not in merged.columns:
        merged["log_mcap"] = np.nan

    return merged


def normalize_factor_name_for_composite(name: str) -> str:
    aliases = {
        "residual_momentum": "momentum",
        "resid_momentum": "momentum",
        "momentum": "momentum",
        "quality": "quality",
        "value": "value",
        "dual_ma": "dual_ma",
        "trend": "dual_ma",
        "reversal": "reversal",
        "behaviour": "behaviour",
        "behavior": "behaviour",
        "attention": "attention",
    }
    key = str(name).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unknown factor name for composite: {name}")
    return aliases[key]


def unwrap_factor_frame(frame_or_result: object) -> pd.DataFrame:
    if isinstance(frame_or_result, dict) and "minimal" in frame_or_result:
        return frame_or_result["minimal"].copy()
    if isinstance(frame_or_result, pd.DataFrame):
        return frame_or_result.copy()
    raise TypeError("Each factor input must be a DataFrame or a result dict with a 'minimal' DataFrame.")


def standardize_factor_frame_for_composite(
    factor_name: str,
    frame_or_result: object,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    composite_name = normalize_factor_name_for_composite(factor_name)
    df = unwrap_factor_frame(frame_or_result)
    key_col = infer_key_col(df)
    clean = standardize_key_col(df, key_col)

    factor_col = infer_factor_col(clean, composite_name)
    if factor_col is None and "factor_value" in clean.columns:
        factor_col = "factor_value"
    if factor_col is None:
        raise ValueError(
            f"Could not infer factor column for '{composite_name}'. "
            f"Expected one of {FACTOR_SPECS[composite_name]['column_candidates']} or factor_value."
        )

    date_col = infer_composite_factor_date_col(clean, composite_name)
    if date_col is None:
        if "data_end_date" in clean.columns:
            date_col = "data_end_date"
        elif "rebalance_date" in clean.columns:
            date_col = "rebalance_date"

    out = clean.rename(columns={factor_col: f"{composite_name}_raw"})
    if date_col is not None and date_col in out.columns:
        out[f"{composite_name}_date"] = parse_any_date(out[date_col])
    else:
        out[f"{composite_name}_date"] = pd.NaT
    out = deduplicate_latest_by_code(out, composite_name)

    for extra_name, candidates in [
        ("industry", DEFAULT_CONFIG["industry_candidates"]),
        ("log_mcap", DEFAULT_CONFIG["log_mcap_candidates"]),
    ]:
        col = first_existing(out.columns, candidates)
        if col is not None and extra_name not in out.columns:
            out[extra_name] = out[col]

    keep_cols = ["code", f"{composite_name}_raw", f"{composite_name}_date"]
    for extra in ["industry", "log_mcap"]:
        if extra in out.columns:
            keep_cols.append(extra)

    log_row = {
        "factor_name": composite_name,
        "file_name": "in_memory_frame",
        "rows": len(df),
        "usable_rows": len(out),
        "key_col": key_col,
        "factor_col": factor_col,
        "date_col": date_col or "",
    }
    return out[keep_cols].copy(), log_row


def compute_freshness_days(rebalance_date: pd.Timestamp, factor_date: pd.Series) -> pd.Series:
    return (rebalance_date - factor_date).dt.days


def classify_freshness_limit(factor_name: str, config: Dict[str, object]) -> int:
    group = FACTOR_SPECS[factor_name]["group"]
    if group == "fundamental":
        return int(config["fundamental_factor_max_lag_days"])
    if group == "macro":
        return int(config["macro_factor_max_lag_days"])
    return int(config["price_factor_max_lag_days"])


def build_warning_table(merged: pd.DataFrame, rebalance_date: pd.Timestamp, config: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for factor_name in FACTOR_SPECS:
        raw_col = f"{factor_name}_raw"
        date_col = f"{factor_name}_date"
        if raw_col not in merged.columns:
            rows.append(
                {
                    "factor_name": factor_name,
                    "status": "missing_upload",
                    "non_null_count": 0,
                    "missing_rate": 1.0,
                    "latest_date": pd.NaT,
                    "max_lag_days": np.nan,
                    "allowed_lag_days": classify_freshness_limit(factor_name, config),
                    "stale_count": np.nan,
                }
            )
            continue

        series = pd.to_numeric(merged[raw_col], errors="coerce")
        date_series = merged[date_col] if date_col in merged.columns else pd.Series(pd.NaT, index=merged.index)
        lag = compute_freshness_days(rebalance_date, date_series)
        limit = classify_freshness_limit(factor_name, config)
        missing_date_mask = date_series.isna()
        future_mask = lag.notna() & lag.lt(0)
        stale_mask = lag.notna() & lag.gt(limit)

        rows.append(
            {
                "factor_name": factor_name,
                "status": "ok",
                "non_null_count": int(series.notna().sum()),
                "missing_rate": float(series.isna().mean()) if len(series) else np.nan,
                "latest_date": date_series.max(),
                "max_lag_days": float(lag.max()) if lag.notna().any() else np.nan,
                "allowed_lag_days": limit,
                "missing_date_count": int(missing_date_mask.sum()),
                "future_date_count": int(future_mask.sum()),
                "stale_count": int(stale_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_runtime_log_messages(df: pd.DataFrame) -> List[str]:
    messages: List[str] = []
    adaptive_dropped = df.attrs.get("adaptive_dropped_factors") or []
    if adaptive_dropped:
        messages.append(f"adaptive_missing_rule dropped_factors={','.join(sorted(set(adaptive_dropped)))}")
    if "dropped_due_to_missing_ts_flag" in df.columns:
        dropped = int(df["dropped_due_to_missing_ts_flag"].sum())
        if dropped:
            col_missing = int(df.get("dual_ma_missing_flag", pd.Series(dtype=int)).sum())
            row_missing = int(df.get("dual_ma_row_missing_flag", pd.Series(dtype=int)).sum())
            messages.append(
                f"TS gate missing: dropped_rows={dropped}, missing_ts_column_rows={col_missing}, missing_ts_value_rows={row_missing}"
            )
    if "strict_missing_score_null_flag" in df.columns:
        strict_missing = int(df["strict_missing_score_null_flag"].sum())
        if strict_missing:
            messages.append(f"strict_missing_mode candidates flagged={strict_missing}")
    if "strict_freshness_score_null_flag" in df.columns:
        strict_freshness = int(df["strict_freshness_score_null_flag"].sum())
        if strict_freshness:
            messages.append(f"strict_freshness_mode candidates flagged={strict_freshness}")
    return messages


def _compute_factor_coverage(df: pd.DataFrame) -> Dict[str, float]:
    coverage: Dict[str, float] = {}
    total = len(df)
    if total <= 0:
        return {name: 0.0 for name in FACTOR_SPECS}
    for factor_name in FACTOR_SPECS:
        raw_col = f"{factor_name}_raw"
        if raw_col in df.columns:
            coverage[factor_name] = float(pd.to_numeric(df[raw_col], errors="coerce").notna().mean())
        else:
            coverage[factor_name] = 0.0
    return coverage


def _apply_adaptive_missing_rule(
    df: pd.DataFrame,
    weights: WeightConfig,
    config: Dict[str, object],
) -> tuple[WeightConfig, Dict[str, float], List[str], bool]:
    candidate_count = int(len(df))
    threshold = int(config.get("adaptive_missing_threshold", 100))
    coverage_threshold = float(config.get("adaptive_factor_coverage_threshold", 0.60))
    coverage = _compute_factor_coverage(df)
    if candidate_count >= threshold:
        return weights, coverage, [], False

    dropped: List[str] = [
        factor_name
        for factor_name, rate in coverage.items()
        if rate < coverage_threshold
    ]
    if not dropped:
        return weights, coverage, [], True

    updated = WeightConfig(**weights.__dict__)
    if "quality" in dropped:
        updated.w_q = 0.0
    if "value" in dropped:
        updated.w_v = 0.0
    if "behaviour" in dropped:
        updated.w_b = 0.0
    if "attention" in dropped:
        updated.w_t = 0.0
    if "momentum" in dropped:
        updated.w_m = 0.0
    if "dual_ma" in dropped:
        updated.w_ts = 0.0
    if "reversal" in dropped:
        updated.w_r = 0.0
    total_main = float(updated.w_q + updated.w_v + updated.w_m + updated.w_ts + updated.w_b + updated.w_t + updated.w_r)
    if total_main > 1e-12:
        updated = normalize_main_weights(updated)
    return updated, coverage, dropped, True


def build_runtime_config(
    *,
    strict_freshness_mode: bool = False,
    strict_missing_mode: bool = False,
    runtime_overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    runtime_config = DEFAULT_CONFIG.copy()
    runtime_config["strict_freshness_mode"] = bool(strict_freshness_mode)
    runtime_config["strict_missing_mode"] = bool(strict_missing_mode)
    if runtime_overrides:
        runtime_config.update(runtime_overrides)
    return runtime_config


class CompositeFormulaEvaluator(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Name,
        ast.Constant,
        ast.Num,
    )

    def __init__(self, variables: Dict[str, object]) -> None:
        self.variables = variables

    def generic_visit(self, node: ast.AST) -> object:
        if not isinstance(node, self.allowed_nodes):
            raise ValueError(f"Unsupported formula syntax: {type(node).__name__}")
        return super().generic_visit(node)

    def visit_Expression(self, node: ast.Expression) -> object:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> object:
        if node.id not in self.variables:
            raise ValueError(f"Unknown formula variable: {node.id}")
        return self.variables[node.id]

    def visit_Constant(self, node: ast.Constant) -> object:
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed in formulas.")
        return float(node.value)

    def visit_Num(self, node: ast.Num) -> object:  # pragma: no cover - legacy AST
        return float(node.n)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> object:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> object:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")


def evaluate_score_formula(
    formula: str | None,
    *,
    df: pd.DataFrame,
    weights: WeightConfig,
) -> pd.Series:
    text = str(formula or "").strip()
    if not text:
        text = "w_q*Q + w_v*V + w_m*M + w_ts*TS + w_b*B + w_t*T + w_r*REV"
    variables: Dict[str, object] = {
        "Q": df["Q"],
        "V": df["V"],
        "M": df["M"],
        "TS": df["TS"],
        "REV": df["REV_effective"] if "REV_effective" in df.columns else df["REV"],
        "B": df["B"],
        "T": df["T"],
        "w_q": float(weights.w_q),
        "w_v": float(weights.w_v),
        "w_m": float(weights.w_m),
        "w_ts": float(weights.w_ts),
        "w_b": float(weights.w_b),
        "w_t": float(weights.w_t),
        "w_r": float(weights.w_r),
    }
    try:
        tree = ast.parse(text, mode="eval")
        evaluated = CompositeFormulaEvaluator(variables).visit(tree)
    except Exception as exc:
        raise ValueError(f"Invalid composite formula: {exc}") from exc
    if isinstance(evaluated, pd.Series):
        return pd.to_numeric(evaluated, errors="coerce")
    if np.isscalar(evaluated):
        return pd.Series(float(evaluated), index=df.index, dtype=float)
    return pd.Series(evaluated, index=df.index, dtype=float)


def prepare_composite_ready_frame(
    merged: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    config: Dict[str, object],
    do_preprocess: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = merged.copy()
    df["rebalance_date"] = rebalance_date
    df["industry"] = df["industry"].astype("string")
    df["log_mcap"] = pd.to_numeric(df["log_mcap"], errors="coerce")
    candidate_count = int(len(df))
    adaptive_threshold = int(config.get("adaptive_missing_threshold", 100))
    adaptive_coverage = _compute_factor_coverage(df)
    adaptive_dropped = [
        factor_name
        for factor_name, rate in adaptive_coverage.items()
        if candidate_count < adaptive_threshold and rate < float(config.get("adaptive_factor_coverage_threshold", 0.60))
    ]
    adaptive_active = bool(candidate_count < adaptive_threshold and adaptive_dropped)
    bypass_ts_gate = "dual_ma" in adaptive_dropped
    ts_source_missing = ("dual_ma_raw" not in df.columns) and not bypass_ts_gate
    df["_bypass_ts_gate_flag"] = int(bypass_ts_gate)
    df["_ts_source_missing_flag"] = int(ts_source_missing)

    factor_alias = {
        "Q": "quality",
        "V": "value",
        "M": "momentum",
        "TS": "dual_ma",
        "REV": "reversal",
        "B": "behaviour",
        "T": "attention",
    }

    for short_name, factor_name in factor_alias.items():
        raw_col = f"{factor_name}_raw"
        out_col = short_name
        if raw_col not in df.columns:
            if factor_name == "dual_ma" and bypass_ts_gate:
                df[out_col] = 1.0
                continue
            df[out_col] = np.nan
            continue
        if factor_name == "dual_ma":
            if bypass_ts_gate:
                df[out_col] = 1.0
                continue
            gate = pd.to_numeric(df[raw_col], errors="coerce")
            df[out_col] = gate.clip(lower=0.0, upper=1.0)
            continue
        do_neutralize = do_preprocess
        do_zscore = do_preprocess
        if factor_name == "reversal":
            do_neutralize = False
            do_zscore = False
        df[out_col] = preprocess_factor(
            df[raw_col],
            lower_q=float(config["winsor_lower_q"]),
            upper_q=float(config["winsor_upper_q"]),
            industry=df["industry"],
            log_mcap=df["log_mcap"],
            do_winsorize=do_preprocess if factor_name != "reversal" else False,
            do_neutralize=do_neutralize,
            do_zscore=do_zscore,
        )

    warning_table = build_warning_table(df, rebalance_date, config)

    for factor_name in FACTOR_SPECS:
        date_col = f"{factor_name}_date"
        lag_col = f"{factor_name}_lag_days"
        missing_date_col = f"{factor_name}_missing_date_flag"
        future_date_col = f"{factor_name}_future_date_flag"
        stale_col = f"{factor_name}_stale_flag"
        freshness_fail_col = f"{factor_name}_freshness_fail_flag"
        if date_col in df.columns:
            df[lag_col] = compute_freshness_days(rebalance_date, df[date_col])
            limit = classify_freshness_limit(factor_name, config)
            df[missing_date_col] = df[date_col].isna().astype(int)
            df[future_date_col] = df[lag_col].lt(0).fillna(False).astype(int)
            df[stale_col] = df[lag_col].gt(limit).fillna(False).astype(int)
            df[freshness_fail_col] = (
                df[[missing_date_col, future_date_col, stale_col]].max(axis=1)
            ).astype(int)
        else:
            df[lag_col] = np.nan
            df[missing_date_col] = 1
            df[future_date_col] = 0
            df[stale_col] = 0
            df[freshness_fail_col] = 1

    freshness_fail_cols = [f"{factor_name}_freshness_fail_flag" for factor_name in FACTOR_SPECS if f"{factor_name}_freshness_fail_flag" in df.columns]
    df["freshness_failure_count"] = df[freshness_fail_cols].sum(axis=1) if freshness_fail_cols else 0
    df["freshness_failure_list"] = [
        ",".join([factor_name for factor_name in FACTOR_SPECS if f"{factor_name}_freshness_fail_flag" in df.columns and bool(df.at[idx, f"{factor_name}_freshness_fail_flag"])]) or ""
        for idx in df.index
    ]
    df["overall_stale_flag"] = df[freshness_fail_cols].max(axis=1) if freshness_fail_cols else 0
    df["strict_freshness_score_null_flag"] = df["overall_stale_flag"].astype(int)
    freshness_failure_text = df["freshness_failure_list"].fillna("").astype(str)
    df["strict_freshness_reason"] = np.where(
        df["strict_freshness_score_null_flag"].eq(1),
        "freshness_failure:" + freshness_failure_text,
        "",
    )

    df.attrs["adaptive_dropped_factors"] = adaptive_dropped
    df.attrs["adaptive_factor_coverage"] = adaptive_coverage
    df.attrs["adaptive_rule_active"] = adaptive_active
    return df, warning_table


def score_prepared_composite_frame(
    prepared: pd.DataFrame,
    weights: WeightConfig,
    rebalance_date: pd.Timestamp,
    config: Dict[str, object],
) -> pd.DataFrame:
    df = prepared.copy()
    adaptive_weights, adaptive_coverage, adaptive_dropped, adaptive_active = _apply_adaptive_missing_rule(df, weights, config)
    weights = adaptive_weights
    zero_tol = 1e-12
    bypass_ts_gate = bool(pd.to_numeric(df.get("_bypass_ts_gate_flag", pd.Series([0])), errors="coerce").fillna(0).astype(int).max())
    ts_source_missing = bool(pd.to_numeric(df.get("_ts_source_missing_flag", pd.Series([0])), errors="coerce").fillna(0).astype(int).max())
    ts_gate_required = abs(float(weights.w_ts)) > zero_tol

    if not ts_gate_required:
        ts_missing_mask = pd.Series(False, index=df.index)
        df["dual_ma_missing_flag"] = 0
        df["dual_ma_row_missing_flag"] = 0
    elif ts_source_missing:
        df["TS"] = np.nan
        ts_missing_mask = pd.Series(True, index=df.index)
        df["dual_ma_missing_flag"] = 1
        df["dual_ma_row_missing_flag"] = 0
    else:
        ts_missing_mask = df["TS"].isna()
        df["dual_ma_missing_flag"] = 0
        df["dual_ma_row_missing_flag"] = ts_missing_mask.astype(int)
    if bypass_ts_gate:
        ts_missing_mask = pd.Series(False, index=df.index)
        df["dual_ma_missing_flag"] = 0
        df["dual_ma_row_missing_flag"] = 0
    df["dropped_due_to_missing_ts_flag"] = ts_missing_mask.astype(int)

    q = df["Q"].fillna(0.0) if abs(float(weights.w_q)) < 1e-12 else df["Q"]
    v = df["V"].fillna(0.0) if abs(float(weights.w_v)) < 1e-12 else df["V"]
    m = df["M"].fillna(0.0) if abs(float(weights.w_m)) < 1e-12 else df["M"]
    ts = df["TS"].fillna(0.0) if abs(float(weights.w_ts)) < 1e-12 else df["TS"]
    b = df["B"].fillna(0.0) if abs(float(weights.w_b)) < 1e-12 else df["B"]
    t = df["T"].fillna(0.0) if abs(float(weights.w_t)) < 1e-12 else df["T"]
    df["M_gate"] = df["M"] * df["TS"]
    df["REV_effective"] = df["REV"]
    df["linear_quality"] = q
    df["linear_value"] = v
    df["linear_momentum"] = m
    df["linear_trend_strength"] = ts
    df["linear_behaviour"] = b
    df["linear_attention"] = t

    score_df = df.copy()
    if abs(float(weights.w_q)) < zero_tol:
        score_df["Q"] = pd.to_numeric(score_df["Q"], errors="coerce").fillna(0.0)
    if abs(float(weights.w_v)) < zero_tol:
        score_df["V"] = pd.to_numeric(score_df["V"], errors="coerce").fillna(0.0)
    if abs(float(weights.w_m)) < zero_tol:
        score_df["M"] = pd.to_numeric(score_df["M"], errors="coerce").fillna(0.0)
    if abs(float(weights.w_ts)) < zero_tol:
        score_df["TS"] = pd.to_numeric(score_df["TS"], errors="coerce").fillna(0.0)
    if abs(float(weights.w_b)) < zero_tol:
        score_df["B"] = pd.to_numeric(score_df["B"], errors="coerce").fillna(0.0)
    if abs(float(weights.w_t)) < zero_tol:
        score_df["T"] = pd.to_numeric(score_df["T"], errors="coerce").fillna(0.0)
    if abs(float(weights.w_r)) < zero_tol:
        score_df["REV_effective"] = pd.to_numeric(score_df["REV_effective"], errors="coerce").fillna(0.0)
        score_df["REV"] = pd.to_numeric(score_df["REV"], errors="coerce").fillna(0.0)

    df["Score"] = evaluate_score_formula(
        config.get("score_formula"),
        df=score_df,
        weights=weights,
    )
    df["score_formula"] = str(config.get("score_formula") or "").strip() or "w_q*Q + w_v*V + w_m*M + w_ts*TS + w_b*B + w_t*T + w_r*REV"

    component_cols = ["Q", "V", "M", "TS", "B", "T", "REV"]
    df["available_module_count"] = df[component_cols].notna().sum(axis=1)
    df["available_factor_count"] = df[component_cols].notna().sum(axis=1)
    df["insufficient_module_flag"] = df["available_module_count"].lt(int(config["min_required_modules"])).astype(int)
    component_weight_map = {
        "Q": float(weights.w_q),
        "V": float(weights.w_v),
        "M": float(weights.w_m),
        "TS": float(weights.w_ts),
        "B": float(weights.w_b),
        "T": float(weights.w_t),
        "REV": float(weights.w_r),
    }
    component_missing_cols: List[str] = []
    for component in component_cols:
        missing_col = f"{component}_missing_component_flag"
        df[missing_col] = df[component].isna().astype(int)
        component_missing_cols.append(missing_col)
    df["missing_component_count"] = df[component_missing_cols].sum(axis=1)
    df["missing_component_list"] = [
        ",".join([component for component in component_cols if bool(df.at[idx, f"{component}_missing_component_flag"])]) or ""
        for idx in df.index
    ]
    strict_required_components = [component for component, weight in component_weight_map.items() if abs(weight) > 1e-12]
    if strict_required_components:
        strict_mask = df[[f"{component}_missing_component_flag" for component in strict_required_components]].any(axis=1)
    else:
        strict_mask = pd.Series(False, index=df.index)
    df["strict_missing_score_null_flag"] = strict_mask.astype(int)
    missing_component_text = df["missing_component_list"].fillna("").astype(str)
    df["strict_missing_reason"] = np.where(strict_mask, "missing_weighted_component:" + missing_component_text, "")

    if bool(config["strict_missing_mode"]):
        df.loc[df["strict_missing_score_null_flag"].eq(1), "Score"] = np.nan

    df.loc[df["dropped_due_to_missing_ts_flag"].eq(1), "Score"] = np.nan
    if "strict_missing_reason" in df.columns:
        df.loc[df["dropped_due_to_missing_ts_flag"].eq(1), "strict_missing_reason"] = "missing_required_gate:TS"

    if bool(config["strict_freshness_mode"]):
        df.loc[df["strict_freshness_score_null_flag"].eq(1), "Score"] = np.nan

    df["TotalRank"] = df["Score"].rank(ascending=False, method="dense")
    df = df.sort_values(["Score", "code"], ascending=[False, True], na_position="last").reset_index(drop=True)
    df.attrs["adaptive_dropped_factors"] = adaptive_dropped
    df.attrs["adaptive_factor_coverage"] = adaptive_coverage
    df.attrs["adaptive_rule_active"] = adaptive_active
    return df


def compute_composite_scores(
    merged: pd.DataFrame,
    weights: WeightConfig,
    rebalance_date: pd.Timestamp,
    config: Dict[str, object],
    do_preprocess: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prepared, warning_table = prepare_composite_ready_frame(
        merged=merged,
        rebalance_date=rebalance_date,
        config=config,
        do_preprocess=do_preprocess,
    )
    scored = score_prepared_composite_frame(
        prepared=prepared,
        weights=weights,
        rebalance_date=rebalance_date,
        config=config,
    )
    return scored, warning_table


def build_flagged_rows(df: pd.DataFrame) -> pd.DataFrame:
    flagged: List[pd.DataFrame] = []
    base_cols = [c for c in ["code", "Score", "TotalRank"] if c in df.columns]

    def add_flag(mask: pd.Series, reason: str) -> None:
        if not mask.any():
            return
        tmp = df.loc[mask, base_cols].copy()
        tmp["flag_reason"] = reason
        flagged.append(tmp)

    add_flag(df["insufficient_module_flag"].eq(1), "insufficient_module_count")
    if "overall_stale_flag" in df.columns:
        add_flag(df["overall_stale_flag"].eq(1), "stale_factor_input")
    if "dual_ma_missing_flag" in df.columns:
        add_flag(df["dual_ma_missing_flag"].eq(1), "dual_ma_column_missing_rows_dropped")
    if "dual_ma_row_missing_flag" in df.columns:
        add_flag(df["dual_ma_row_missing_flag"].eq(1), "dual_ma_row_missing_rows_dropped")
    if "dropped_due_to_missing_ts_flag" in df.columns:
        add_flag(df["dropped_due_to_missing_ts_flag"].eq(1), "missing_ts_gate")
    if "strict_missing_score_null_flag" in df.columns:
        add_flag(df["strict_missing_score_null_flag"].eq(1), "strict_missing_mode_score_nulled")
    if "strict_freshness_score_null_flag" in df.columns:
        add_flag(df["strict_freshness_score_null_flag"].eq(1), "strict_freshness_mode_score_nulled")
    add_flag(df["Score"].isna(), "missing_total_score")

    if not flagged:
        return pd.DataFrame(columns=base_cols + ["flag_reason"])
    return pd.concat(flagged, ignore_index=True).drop_duplicates()


def build_qa_summary(
    df: pd.DataFrame,
    upload_log: pd.DataFrame,
    warning_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"check": "uploaded_factor_files", "value": len(upload_log)},
        {"check": "merged_rows", "value": len(df)},
        {"check": "score_non_null", "value": int(df["Score"].notna().sum())},
        {"check": "score_null", "value": int(df["Score"].isna().sum())},
        {"check": "insufficient_module_rows", "value": int(df["insufficient_module_flag"].sum())},
        {"check": "overall_stale_rows", "value": int(df.get("overall_stale_flag", pd.Series(dtype=int)).sum())},
        {"check": "dual_ma_missing_rows", "value": int(df.get("dual_ma_missing_flag", pd.Series(dtype=int)).sum())},
        {"check": "dual_ma_row_missing_rows", "value": int(df.get("dual_ma_row_missing_flag", pd.Series(dtype=int)).sum())},
        {"check": "dropped_due_to_missing_ts_rows", "value": int(df.get("dropped_due_to_missing_ts_flag", pd.Series(dtype=int)).sum())},
    ]
    for _, row in warning_table.iterrows():
        rows.append({"check": f"{row['factor_name']}_missing_rate", "value": row["missing_rate"]})
        rows.append({"check": f"{row['factor_name']}_missing_date_count", "value": row["missing_date_count"]})
        rows.append({"check": f"{row['factor_name']}_future_date_count", "value": row["future_date_count"]})
        rows.append({"check": f"{row['factor_name']}_stale_count", "value": row["stale_count"]})
    return pd.DataFrame(rows)


def export_outputs(
    scored_df: pd.DataFrame,
    qa_summary: pd.DataFrame,
    warning_table: pd.DataFrame,
    flagged_rows: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> Dict[str, str]:
    anchor = rebalance_date.strftime("%Y%m%d")
    out_dir = Path("/content")
    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = out_dir / f"factor_composite_ranking_{anchor}.csv"
    qa_path = out_dir / f"factor_composite_qa_{anchor}.csv"
    warning_path = out_dir / f"factor_composite_freshness_{anchor}.csv"
    flagged_path = out_dir / f"factor_composite_flagged_{anchor}.csv"

    final_cols = [
        "TotalRank", "code", "Score", "Q", "V", "M", "TS", "B", "T", "REV", "M_gate",
        "quality_raw", "value_raw", "momentum_raw", "dual_ma_raw",
        "behaviour_raw", "attention_raw", "reversal_raw",
        "industry", "log_mcap", "available_module_count", "available_factor_count",
        "insufficient_module_flag", "overall_stale_flag", "missing_component_count",
        "missing_component_list", "strict_missing_score_null_flag", "strict_missing_reason",
        "strict_freshness_score_null_flag", "strict_freshness_reason",
        "dual_ma_missing_flag", "dual_ma_row_missing_flag", "dropped_due_to_missing_ts_flag",
    ]
    final_cols += [c for c in scored_df.columns if c.endswith("_date") or c.endswith("_lag_days") or c.endswith("_stale_flag") or c.endswith("_missing_date_flag") or c.endswith("_future_date_flag") or c.endswith("_freshness_fail_flag") or c.endswith("_missing_component_flag")]
    final_cols = [c for c in final_cols if c in scored_df.columns]

    scored_df[final_cols].to_csv(final_path, index=False, encoding="utf-8-sig")
    qa_summary.to_csv(qa_path, index=False, encoding="utf-8-sig")
    warning_table.to_csv(warning_path, index=False, encoding="utf-8-sig")
    flagged_rows.to_csv(flagged_path, index=False, encoding="utf-8-sig")

    if files is not None:
        files.download(str(final_path))
        files.download(str(qa_path))
        files.download(str(warning_path))
        files.download(str(flagged_path))

    return {
        "final_ranking": str(final_path),
        "qa_summary": str(qa_path),
        "freshness_report": str(warning_path),
        "flagged_rows": str(flagged_path),
    }


def show_result_preview(scored_df: pd.DataFrame, qa_summary: pd.DataFrame, warning_table: pd.DataFrame, config: Dict[str, object]) -> None:
    top_n = int(config["top_n_preview"])
    bottom_n = int(config["bottom_n_preview"])

    _safe_display(Markdown("## QA Summary"))
    _safe_display(qa_summary)

    _safe_display(Markdown("## Freshness / Missing Summary"))
    _safe_display(warning_table)

    runtime_messages = build_runtime_log_messages(scored_df)
    if runtime_messages:
        _safe_display(Markdown("## Runtime Notes"))
        for message in runtime_messages:
            print(message)

    preview_cols = [c for c in ["TotalRank", "code", "Score", "Q", "V", "M", "TS", "B", "T", "REV", "M_gate"] if c in scored_df.columns]

    _safe_display(Markdown(f"## Top {top_n}"))
    _safe_display(scored_df[preview_cols].head(top_n))

    _safe_display(Markdown(f"## Bottom {bottom_n}"))
    _safe_display(scored_df[preview_cols].dropna(subset=["Score"]).tail(bottom_n))


def run_factor_compositor(
    weights: WeightConfig,
    rebalance_date: str,
    strict_freshness_mode: bool = False,
    strict_missing_mode: bool = False,
    do_preprocess: bool = True,
) -> Dict[str, object]:
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    validate_weights(weights)
    weights = normalize_main_weights(weights)

    runtime_config = DEFAULT_CONFIG.copy()
    runtime_config["strict_freshness_mode"] = bool(strict_freshness_mode)
    runtime_config["strict_missing_mode"] = bool(strict_missing_mode)

    show_formula_help()
    factor_frames, upload_log = upload_factor_bundle()
    merged = merge_factor_frames(factor_frames)
    scored_df, warning_table = compute_composite_scores(
        merged=merged,
        weights=weights,
        rebalance_date=rebalance_ts,
        config=runtime_config,
        do_preprocess=do_preprocess,
    )
    qa_summary = build_qa_summary(scored_df, upload_log, warning_table)
    flagged_rows = build_flagged_rows(scored_df)
    show_result_preview(scored_df, qa_summary, warning_table, runtime_config)
    output_paths = export_outputs(scored_df, qa_summary, warning_table, flagged_rows, rebalance_ts)

    return {
        "upload_log": upload_log,
        "merged": merged,
        "scored_df": scored_df,
        "qa_summary": qa_summary,
        "warning_table": warning_table,
        "flagged_rows": flagged_rows,
        "output_paths": output_paths,
    }


def run_factor_composite_from_frames(
    factor_frames: Dict[str, object],
    rebalance_date: str,
    weights: Optional[WeightConfig] = None,
    strict_freshness_mode: bool = False,
    strict_missing_mode: bool = False,
    do_preprocess: bool = True,
    export: bool = False,
    show_preview: bool = False,
    runtime_overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Run the factor compositor from in-memory factor outputs.

    Values in factor_frames may be either DataFrames or factor result dicts
    returned by run_*_factor functions.  This entry point is intended for the
    main Colab control notebook so it can avoid CSV upload handoffs.
    """
    if not factor_frames:
        raise ValueError("factor_frames is empty.")

    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    weights = weights or WeightConfig()
    validate_weights(weights)
    weights = normalize_main_weights(weights)

    standardized_frames: Dict[str, pd.DataFrame] = {}
    upload_log_rows: List[Dict[str, object]] = []
    for factor_name, frame_or_result in factor_frames.items():
        composite_name = normalize_factor_name_for_composite(factor_name)
        clean, log_row = standardize_factor_frame_for_composite(composite_name, frame_or_result)
        standardized_frames[composite_name] = clean
        upload_log_rows.append(log_row)

    runtime_config = build_runtime_config(
        strict_freshness_mode=strict_freshness_mode,
        strict_missing_mode=strict_missing_mode,
        runtime_overrides=runtime_overrides,
    )

    upload_log = pd.DataFrame(upload_log_rows)
    merged = merge_factor_frames(standardized_frames)
    scored_df, warning_table = compute_composite_scores(
        merged=merged,
        weights=weights,
        rebalance_date=rebalance_ts,
        config=runtime_config,
        do_preprocess=do_preprocess,
    )
    qa_summary = build_qa_summary(scored_df, upload_log, warning_table)
    flagged_rows = build_flagged_rows(scored_df)

    output_paths: Dict[str, str] = {}
    if show_preview:
        show_result_preview(scored_df, qa_summary, warning_table, runtime_config)
    if export:
        output_paths = export_outputs(scored_df, qa_summary, warning_table, flagged_rows, rebalance_ts)

    return {
        "upload_log": upload_log,
        "merged": merged,
        "scored_df": scored_df,
        "qa_summary": qa_summary,
        "warning_table": warning_table,
        "flagged_rows": flagged_rows,
        "output_paths": output_paths,
    }


def run_factor_composite_from_prepared_frame(
    prepared_df: pd.DataFrame,
    rebalance_date: str,
    weights: Optional[WeightConfig] = None,
    strict_freshness_mode: bool = False,
    strict_missing_mode: bool = False,
    runtime_overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    weights = weights or WeightConfig()
    validate_weights(weights)
    weights = normalize_main_weights(weights)
    runtime_config = build_runtime_config(
        strict_freshness_mode=strict_freshness_mode,
        strict_missing_mode=strict_missing_mode,
        runtime_overrides=runtime_overrides,
    )
    warning_table = build_warning_table(prepared_df, rebalance_ts, runtime_config)
    scored_df = score_prepared_composite_frame(
        prepared=prepared_df,
        weights=weights,
        rebalance_date=rebalance_ts,
        config=runtime_config,
    )
    qa_summary = build_qa_summary(scored_df, pd.DataFrame(), warning_table)
    flagged_rows = build_flagged_rows(scored_df)
    return {
        "upload_log": pd.DataFrame(),
        "merged": prepared_df.copy(),
        "scored_df": scored_df,
        "qa_summary": qa_summary,
        "warning_table": warning_table,
        "flagged_rows": flagged_rows,
        "output_paths": {},
    }


def launch_factor_compositor_ui() -> None:
    if widgets is None:
        raise RuntimeError("ipywidgets is required for the Colab UI.")

    show_formula_help()

    style = {"description_width": "170px"}
    layout = widgets.Layout(width="320px")

    controls = {
        "w_q": widgets.FloatText(value=0.20, description="w_q", style=style, layout=layout),
        "w_v": widgets.FloatText(value=0.20, description="w_v", style=style, layout=layout),
        "w_m": widgets.FloatText(value=0.20, description="w_m", style=style, layout=layout),
        "w_ts": widgets.FloatText(value=0.15, description="w_ts", style=style, layout=layout),
        "w_b": widgets.FloatText(value=0.10, description="w_b", style=style, layout=layout),
        "w_t": widgets.FloatText(value=0.10, description="w_t", style=style, layout=layout),
        "w_r": widgets.FloatText(value=0.05, description="w_r", style=style, layout=layout),
        "rebalance_date": widgets.Text(value=pd.Timestamp.today().strftime("%Y-%m-%d"), description="rebalance_date", style=style, layout=layout),
        "strict_freshness_mode": widgets.Checkbox(value=False, description="strict_freshness_mode"),
        "strict_missing_mode": widgets.Checkbox(value=False, description="strict_missing_mode"),
        "do_preprocess": widgets.Checkbox(value=True, description="do_preprocess"),
    }

    run_button = widgets.Button(description="Run Composite", button_style="primary", icon="play")
    output = widgets.Output()

    formula_card = widgets.HTML(
        value="""
        <div style="border:1px solid #d9d9d9;padding:12px;border-radius:8px;background:#fbfbfb;">
          <div style="font-size:16px;font-weight:700;margin-bottom:8px;">Formula</div>
          <div style="font-family:monospace;white-space:pre-wrap;line-height:1.6;">
Score  = w_q * Q + w_v * V + w_m * M + w_ts * TS + w_b * B + w_t * T + w_r * REV
          </div>
          <div style="margin-top:10px;font-size:13px;line-height:1.6;">
            Q: Quality, V: Value, M: 12-1 Momentum, TS: Dual-MA Trend Strength,
            B: Behaviour Crowding, T: Attention Crowding, REV: Reversal.
            The default model is a simple linear weighting of factor inputs.
            Rows with missing TS are still reported in QA outputs.
          </div>
        </div>
        """
    )

    left_box = widgets.VBox([
        controls["w_q"], controls["w_v"], controls["w_m"], controls["w_ts"],
        controls["w_b"], controls["w_t"], controls["w_r"],
        controls["rebalance_date"], controls["strict_freshness_mode"],
        controls["strict_missing_mode"], controls["do_preprocess"], run_button,
    ])
    _safe_display(widgets.HBox([left_box, formula_card]))
    _safe_display(output)

    def on_run(_):
        output.clear_output()
        with output:
            try:
                result = run_factor_compositor(
                    weights=WeightConfig(
                        w_q=float(controls["w_q"].value),
                        w_v=float(controls["w_v"].value),
                        w_m=float(controls["w_m"].value),
                        w_ts=float(controls["w_ts"].value),
                        w_b=float(controls["w_b"].value),
                        w_t=float(controls["w_t"].value),
                        w_r=float(controls["w_r"].value),
                    ),
                    rebalance_date=str(controls["rebalance_date"].value),
                    strict_freshness_mode=bool(controls["strict_freshness_mode"].value),
                    strict_missing_mode=bool(controls["strict_missing_mode"].value),
                    do_preprocess=bool(controls["do_preprocess"].value),
                )
                print("Composite run completed.")
                print(result["output_paths"])
            except Exception as exc:
                print(f"Run failed: {exc}")

    run_button.on_click(on_run)


if __name__ == "__main__":
    print("Use this script in Colab with:")
    print("from factor_composite import launch_factor_compositor_ui")
    print("launch_factor_compositor_ui()")
