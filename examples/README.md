# Synthetic schema demo

This directory contains a tiny synthetic example that shows the expected local
cache shapes without using any real JPX or J-Quants data.

## What this is for

- understanding the expected CSV schema,
- seeing how the project organizes local cache tables,
- reviewing a minimal configuration without needing API access first.

## What this is not

- not a validated strategy demo,
- not a realistic market sample,
- not a substitute for the J-Quants-backed cache flow used by the main project.

All symbols, prices, dates, sectors, and market-cap values in
`examples/synthetic_data/` are fictional.

## Files

- `synthetic_data/prices.csv`
- `synthetic_data/universe.csv`
- `synthetic_data/sector.csv`
- `synthetic_data/market_cap.csv`
- `synthetic_data/index_prices.csv`
- `synthetic_config.json`

`synthetic_config.json` is a schema-oriented companion config. If you want to
run the real backtest workflow, start with `configs/minimal_long_only.example.json`
and prepare a proper local cache with your own licensed data access.
