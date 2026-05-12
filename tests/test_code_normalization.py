from __future__ import annotations

from runtime.historical_data import normalize_code


def test_normalize_code_handles_plain_code() -> None:
    assert normalize_code("7203") == "7203"


def test_normalize_code_handles_ticker_suffix() -> None:
    assert normalize_code("7203.T") == "7203"


def test_normalize_code_handles_five_digit_variant() -> None:
    assert normalize_code("72030") == "7203"


def test_normalize_code_returns_none_for_invalid_input() -> None:
    assert normalize_code("ABC") is None
    assert normalize_code(None) is None
