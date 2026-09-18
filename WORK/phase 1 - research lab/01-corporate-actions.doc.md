# Corporate actions and price adjustment

§12.1 stage 5, and the prerequisite for everything else in Phase 1. Splits and dividends come in
from Yahoo, become multiplicative factors, and the corpus is audited against them.

## Why this came before the feature library

§12.1 times the quality gate as "Phase 1, ongoing", but one part of it cannot wait. A
triple-barrier label spans days. Across an unadjusted 4:1 split the price path shows -75%, which
trips the stop barrier on a position that never lost a cent — and the same phantom move poisons
ATR, realised volatility and every return-based feature, on exactly the dates with the most price
action. Labelling an unadjusted corpus does not produce a noisy answer. It produces a confident
wrong one.

## What exists

| Module | Responsibility |
|---|---|
| `core/actions.py` | `CorporateAction`, `AdjustmentSeries`, `adjust_bars`, `unexplained_gaps`, `PriceGap`, `tidy_decimal` |
| `core/ports.py` | `CorporateActionsPort` — `async fetch_actions(symbol, start, end)` |
| `adapters/feeds/yfinance_actions.py` | `YFinanceActions` + `YahooActionsDownloader`, cached per symbol |
| `adapters/storage/actions_parquet.py` | `ActionStore` — one file under `derived/actions/yfinance/` |
| `ingest/actions.py` | `fetch_actions` (collect) and `scan_gaps` (audit). Ports only |
| `cli.py` | `neurotrade actions fetch` / `actions check`; `make actions` / `make actions-check` |

## The decisions

**Two factors, not one.** `price_factor` removes splits only and reconstructs the price path a
trader saw — barrier touches are decided on it, because a stop is hit when the *price* trades
there and a dividend fills nothing. `total_return_factor` removes splits and dividends, and is
what a return or momentum feature wants so an ex-dividend date does not read as a loss. Labelling
uses the first, return features the second. One "adjusted price" would be wrong in one direction
or the other.

**An action applies to bars strictly before its effective date.** The bar stamped on the effective
date already trades on the new basis. Getting this inclusive shifts the whole series by one bar.

**`as_of` is required, never "today".** A 2025 split must not touch a 2022 decision, or the
backtest is computed from a price nobody could have had and will not reproduce live.

**A dividend with no prior close is skipped, not guessed.** The factor is `1 - dividend/close`, so
without the denominator there is no factor; inventing one drifts every downstream return.

**A symbol with no actions is stored as a row with null values.** Otherwise "Yahoo says it never
split" and "we never asked" are indistinguishable, and a failed fetch reads as a clean answer. A
symbol the feed *refuses* is recorded as a failure and not stored at all.

**Actions live under `derived/`, not `raw/`.** They are downloaded, so `raw/` looks right — but
feeds *revise* corporate actions, and re-fetching would mean overwriting `raw/`, which the
immutability invariant forbids. Treating the set as recomputable (delete, re-fetch) keeps the
invariant and costs one free HTTP call.

**The audit is separate from the fetch, and reads prices rather than the response.** A feed cannot
report an omission it does not know it made. `scan_gaps` applies every action we hold and looks
for what is left over — the only evidence a missing split leaves.

## What is in the corpus

Measured 2026-09-17: `neurotrade actions fetch --start 2015-01-01` collected **43 symbols, 18
splits, 1,588 dividends, 0 failures** into `data/derived/actions/yfinance/actions.parquet`.

`neurotrade actions check --start 2021-09-15 --end 2026-09-11` scans 53,879 daily bars and reports
**4 gaps**, all genuine news moves rather than missing actions: AMD 2025-10-06 +37.5%, INTC
2025-09-18 +27.6%, NVDA 2023-05-25 +26.2%, NFLX 2022-04-20 -29.7%. The command exits 1 whenever
any gap stands, so read it as a lead generator, not a pass/fail on data quality — `implied_split`
is what separates the two, since 0.73:1 is not a ratio any company declares.

## Not done yet, and the seam left for it

- **Adjustment is not yet applied on the read path.** `adjust_bars` exists and is tested; nothing
  wires it into a feature or label pass, because neither exists. The seam is `ActionStore.series`.
- **Only Yahoo.** `CorporateActionsPort` is the seam for a second source. An IBKR
  `ADJUSTED_LAST` cross-check was considered and deferred: daily-only, pacing-expensive, and it
  catches the same class the gap scan already does.
- **`prior_close` is not populated by any caller**, so `total_return_factor` is currently
  reachable only from tests. It wants the daily corpus keyed by ex-date.
- **The rest of stage 5** — duplicate prints, halt marking, session-gap detection against the
  venue calendar, survivorship audit — is the next commit and gates nothing.
