from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config.defaults import FORMULA_VARIABLES, VARIABLE_TO_FACTOR


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(True, [])

    @classmethod
    def failure(cls, errors: list[str]) -> "ValidationResult":
        return cls(False, errors)


class FormulaValidator(ast.NodeVisitor):
    """Small arithmetic expression validator for composite formula preview."""

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

    def __init__(self) -> None:
        self.names: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self.allowed_nodes):
            raise ValueError(f"不支持的公式语法: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)
        if node.id not in FORMULA_VARIABLES:
            raise ValueError(f"公式引用了未知变量: {node.id}")


def validate_formula(formula: str, enabled_factors: set[str]) -> list[str]:
    errors: list[str] = []
    if not formula.strip():
        return ["合成公式不能为空"]
    try:
        tree = ast.parse(formula, mode="eval")
        visitor = FormulaValidator()
        visitor.visit(tree)
    except Exception as exc:
        return [f"合成公式不合法: {exc}"]
    for name in visitor.names:
        factor = VARIABLE_TO_FACTOR.get(name)
        if factor and factor not in enabled_factors:
            errors.append(f"公式变量 {name} 引用了未启用的因子: {factor}")
    return errors


def validate_parameters(params: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    start = params["start_date"]
    end = params["end_date"]
    if start > end:
        errors.append("开始日期不能晚于结束日期")
    api_key = str(params.get("api_key", "")).strip()
    if not api_key:
        errors.append("请在 GUI 中输入 API key")
    if params["initial_capital"] <= 0:
        errors.append("初始资金必须大于 0")
    if params["target_count"] <= 0:
        errors.append("目标持仓数量必须大于 0")
    if not (0 < params["max_gross_exposure"] <= 2.0):
        errors.append("最大总仓位必须在 0 到 2 之间")
    if params.get("enable_short_leg"):
        if not (0 < float(params.get("short_gross_exposure", 0.0)) <= 1.5):
            errors.append("Short gross exposure 必须在 0 到 1.5 之间")
        if int(params.get("short_target_count", 0)) <= 0:
            errors.append("启用 short hedge 时，Short target count 必须大于 0")
        if int(params.get("short_target_count", 0)) > int(params["selection_top_n"]):
            errors.append("Short target count 不能大于候选池上限")
        if params.get("short_construction_mode") not in {"simple_bottom_n", "advanced_bottom_decile"}:
            errors.append("short_construction_mode 必须是 simple_bottom_n 或 advanced_bottom_decile")
        if params.get("short_weighting_mode") not in {"equal_weight", "rank_weighted", "score_proportional"}:
            errors.append("short_weighting_mode 必须是 equal_weight、rank_weighted 或 score_proportional")
        if params.get("short_expansion_mode") not in {"bottom_decile_only", "expand_to_bottom20", "expand_to_bottom30"}:
            errors.append("short_expansion_mode 必须是 bottom_decile_only、expand_to_bottom20 或 expand_to_bottom30")
        if int(params.get("short_min_names", 0)) <= 0:
            errors.append("启用 short hedge 时，Short min names 必须大于 0")
        if not (0 < float(params.get("short_max_single_weight", 0.0)) <= 1.0):
            errors.append("Short max single weight 必须在 0 到 1 之间")
        if float(params.get("short_sector_cap", 0.0) or 0.0) < 0:
            errors.append("Short sector cap 不能为负")
        if float(params.get("short_adv_threshold", 0.0) or 0.0) < 0:
            errors.append("Short ADV threshold 不能为负")
    if params.get("use_market_cap_filter") and params["market_cap_mode"] == "rank" and params["rank_min"] > params["rank_max"]:
        errors.append("市值排名下限不能大于上限")
    if params.get("use_market_cap_filter") and params["market_cap_mode"] == "range_billion_jpy" and params["min_market_cap"] > params["max_market_cap"]:
        errors.append("最小市值不能大于最大市值")
    if params.get("use_liquidity_filter") and params["min_avg_daily_volume"] < 0:
        errors.append("最低日均成交量不能为负")
    if params.get("use_volatility_filter") and params["max_annualized_volatility"] <= 0:
        errors.append("最大年化波动率必须大于 0")
    if params.get("use_max_lot_cost_filter") and float(params.get("max_lot_cost_jpy", 0.0)) <= 0:
        errors.append("启用 max lot cost 时，阈值必须大于 0")
    if params.get("use_sector_constraints"):
        if int(params.get("max_names_per_sector", 0)) <= 0:
            errors.append("启用行业约束时，Max names per sector 必须大于 0")
        if params.get("sector_cap_mode") not in {"hard", "soft_penalty", "exception"}:
            errors.append("sector_cap_mode 必须是 hard、soft_penalty 或 exception")
    if params["risk_trend_short_ma"] >= params["risk_trend_long_ma"]:
        errors.append("Risk gate 的短期均线窗口必须小于长期均线窗口")
    if params["risk_rv_window"] >= params["risk_rv_q_window"]:
        errors.append("Risk gate 的波动率窗口应小于波动率分位数窗口")
    enabled = {key for key, state in params["enabled_factors"].items() if state}
    if not enabled:
        errors.append("至少需要启用一个因子")
    if params.get("weight_source_mode") not in {"manual"}:
        errors.append("公开版只支持手动公式和权重输入")
    for name, value in params["composite_weights"].items():
        if not (0.0 <= float(value) <= 2.0):
            errors.append(f"权重 {name} 必须在 0 到 2 之间")
    if params.get("allocation_gate_mode") not in {"matrix", "legacy"}:
        errors.append("allocation_gate_mode 必须是 matrix 或 legacy")
    if int(params.get("allocation_gate_risk_override_threshold", 0)) < 1 or int(params.get("allocation_gate_risk_override_threshold", 0)) > 5:
        errors.append("统一闸门的 risk override threshold 必须在 1 到 5 之间")
    if int(params.get("allocation_gate_severe_risk_score_threshold", 0)) < 1 or int(params.get("allocation_gate_severe_risk_score_threshold", 0)) > 5:
        errors.append("统一闸门的 severe risk score threshold 必须在 1 到 5 之间")
    if int(params.get("allocation_gate_risk_override_threshold", 0)) > int(params.get("allocation_gate_severe_risk_score_threshold", 0)):
        errors.append("统一闸门的 risk override threshold 不能大于 severe risk score threshold")
    post_score_adjustments = dict(params.get("post_score_adjustments", {}))
    if post_score_adjustments:
        if post_score_adjustments.get("regime_sector_bias_regime_source") not in {"active_regime", "weight_regime", "position_regime", "raw_regime"}:
            errors.append("regime_sector_bias_regime_source 必须是 active_regime、weight_regime、position_regime 或 raw_regime")
    errors.extend(validate_formula(params["formula"], enabled))
    return ValidationResult.failure(errors) if errors else ValidationResult.success()
