from __future__ import annotations

from gui_app.validators import validate_formula


def test_validate_formula_accepts_simple_linear_expression() -> None:
    errors = validate_formula("w_q*Q + w_v*V", {"quality", "value"})
    assert errors == []


def test_validate_formula_rejects_unsafe_expression() -> None:
    errors = validate_formula('__import__("os").system("echo bad")', {"quality"})
    assert errors
    assert "不合法" in errors[0]


def test_validate_formula_rejects_unknown_variable() -> None:
    errors = validate_formula("w_q*Q + BAD", {"quality"})
    assert errors
    assert "BAD" in errors[0]


def test_validate_formula_rejects_disabled_factor_variable() -> None:
    errors = validate_formula("w_q*Q + w_m*M", {"quality"})
    assert errors
    assert any("M" in message for message in errors)
