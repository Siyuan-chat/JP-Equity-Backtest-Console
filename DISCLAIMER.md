# Disclaimer and Data-Usage Notice

This repository is provided as a software and research utility only.

It is not legal advice, compliance advice, investment advice, a solicitation, a recommendation to buy or sell securities, a portfolio management service, or a brokerage service.

## 1. No investment advice

The software in this repository is intended for technical analysis, factor research, historical simulation, and data workflow experimentation.

It does not:

- recommend specific securities,
- recommend specific portfolio allocations,
- guarantee profitability,
- guarantee fitness for live trading,
- provide individualized investment advice,
- act as an investment research report prepared for regulated distribution.

Any conclusions, rankings, signals, scores, or backtest results produced with this repository are the user’s own responsibility.

The developer and contributors are not responsible for:

- trading decisions,
- investment losses,
- opportunity costs,
- model misuse,
- data misuse,
- misinterpretation of outputs,
- regulatory consequences arising from deployment or redistribution.

## 2. Research-only positioning

This project should be described and used as a data research tool.

It should not be marketed or represented as:

- a stock-picking service,
- an investment recommendation engine,
- a discretionary or automated investment advisor,
- a guaranteed alpha engine,
- a resale package for JPX or J-Quants data.

## 3. JPX / J-Quants data rights and usage boundaries

This repository may be used together with data obtained from JPX-related services, including J-Quants.

Users are responsible for ensuring that their own acquisition, storage, processing, publication, and redistribution of data complies with the applicable JPX / J-Quants terms, licenses, and policies.

Based on the official JPX and J-Quants materials available as of May 12, 2026:

- rights in JPX website content and related information belong to JPX and related group entities,
- modification or adaptation of JPX website content without prior permission is restricted,
- collection of website data or secondary commercial use of website information is restricted unless separately permitted,
- JPX states that its website information is not intended as a solicitation for investment,
- JPX disclaims responsibility for actions taken based on such information,
- J-Quants states that users may publish their own analysis results and methodologies, but distributing raw data or data in directly viewable form is prohibited,
- J-Quants states that continuously providing investment analysis results to third parties does not fall within personal-use scope,
- JPX market-data pages indicate that redistribution to third parties may require prior permission and, in some cases, a separate contract or license.

Users should review the official source documents directly before publishing any derivative product, hosted tool, API, dashboard, report service, newsletter, or commercial offering.

Official references:

- JPX Disclaimer / Terms of Use
  - https://www.jpx.co.jp/english/term-of-use/index.html
- JPX J-Quants API overview
  - https://www.jpx.co.jp/english/markets/other-data-services/j-quants-api/index.html
- J-Quants public site / dashboard
  - https://jpx-jquants.com/dashboard/menu/
- JPX paid market-data distribution guidance
  - https://www.jpx.co.jp/english/markets/paid-info-equities/realtime/02.html

## 4. No warranty

This repository is provided “as is” and “as available,” without warranties of any kind, whether express or implied.

This includes, without limitation, no warranty regarding:

- correctness,
- completeness,
- merchantability,
- fitness for a particular purpose,
- non-infringement,
- data freshness,
- continuity of external data access,
- reproducibility of backtest results across environments.

Backtest outputs can be wrong, stale, biased, incomplete, or operationally misleading.

## 5. Backtest-specific warning

Historical simulation is not evidence of future results.

Backtests are especially sensitive to:

- survivorship bias,
- look-ahead bias,
- data revisions,
- missing corporate-action handling,
- liquidity assumptions,
- slippage assumptions,
- transaction cost assumptions,
- universe definition errors,
- benchmark choice,
- parameter overfitting.

Even when a run completes successfully, its results should not be treated as validated for production trading.

## 6. Redistribution and publication

If you publish anything built with this repository, you should ensure that:

- you are not redistributing restricted JPX / J-Quants source data,
- you are not exposing data in a way that violates your plan, contract, or license,
- you are not implying endorsement by JPX, J-Quants, or the developer,
- you clearly distinguish your own derived analysis from the underlying licensed data,
- you obtain legal or compliance review when necessary.

## 7. Developer responsibility boundary

The developer’s role is limited to providing software code in this repository.

The developer does not assume responsibility for:

- user-generated models,
- user-defined formulas,
- user-created factors,
- uploaded credentials,
- local cache contents,
- exported reports,
- public statements made by downstream users,
- any financial, legal, or regulatory outcome connected to use of this software.

## 8. Recommended wording when reusing this project

If you redistribute or fork this project, a safe summary is:

“Software for historical market-data research and backtesting only. Not investment advice. Users are responsible for complying with JPX / J-Quants data-license and redistribution rules.”
