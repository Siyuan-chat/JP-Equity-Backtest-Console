"""
Backward-compatible wrapper for the renamed market risk gating runtime.
"""

from market_risk_gating_runtime import *  # noqa: F401,F403
from market_risk_gating_runtime import download_risk_index_close_yfinance as download_nikkei_close_yfinance
from market_risk_gating_runtime import load_risk_index_close as load_nikkei_close
