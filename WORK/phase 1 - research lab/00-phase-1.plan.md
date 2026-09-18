# Phase 1 — Research Lab: plan

**Spec:** §13 row 1 (deliverable + exit gate), §8 (validation methodology), §12.1 stage 5
(corpus quality gate, "Phase 1, ongoing"), §20.4 (opening move and its ordering).

**Deliverable:** feature library (PIT-enforced), triple-barrier labelling, CPCV harness,
calibrated cost model, DSR/PBO reporting, trial ledger.
**Exit gate:** the lab correctly rejects a deliberately overfit control strategy.

## What already exists

- `features/registry.py` — `FeatureSpec` declares a lookback; `evaluate(history, as_of)` raises
  `LookaheadError` on any bar stamped after `as_of`, and returns `None` until the window is full.
  **PIT enforcement from §20.4 is already built.** No features are defined.
- `strategies/base.py` — contract only.
- `lab/replay.py` — G1.
- Corpus: 1.02M seed minute bars, 53.9k daily bars, 47.6k IBKR minute bars, 1,202 universe
  sessions. **All price data is unadjusted; corporate actions are not stored.**

## Why stage 5 comes first, not "ongoing"

§12.1 times the quality gate as "Phase 1, ongoing", but one part of it is a hard prerequisite.
Every price series in the corpus is unadjusted and no split/dividend dataset exists. A
triple-barrier label spans days: across a 4:1 split an unadjusted series shows −75%, which trips
the stop barrier on a position that never lost anything. Labels, ATR-scaled barriers, realized
vol and every return-based feature are all poisoned at exactly the dates that matter most.
Gap/duplicate/halt detection can stay ongoing; **adjustment cannot**.

## Commits, in dependency order

| # | Scope | Contents |
|---|---|---|
| 1 | Corporate actions + adjustment | Splits/dividends into `raw/actions/` (own dataset, not bar columns); adjustment-factor series; applied at read, never mutating `raw/`. PIT: factors as of `t`, not as of today. |
| 2 | Corpus quality report | Gap detection vs the venue calendar, duplicate prints, halt marking, survivorship audit. a corpus-check make target. Report-only — never edits the corpus. |
| 3 | Feature library v1 | First real features against the existing registry: returns, ATR, realized vol, relative volume, VWAP distance, opening range. Plus fractional differentiation (§8 stationarity). |
| 4 | Triple-barrier labelling | a labelling module in lab/ — ATR-scaled profit/stop barriers + vertical time barrier; label, touch time, realized return. Uniqueness / concurrency weights for overlapping windows (§8 sample weighting). |
| 5 | Cost model | a costs module in lab/, plus a fee schedule in `config/`. Spread, commission, slippage scaling with size and vol, maker/taker. One implementation, shared research/live. Applied inside simulation. |
| 6 | CPCV harness | a cross-validation module in lab/ — N groups choose k test groups, C(N,k) paths; purging of overlapping labels and embargo. Walk-forward as the secondary sanity check. |
| 7 | Significance + trial ledger | a significance module (DSR, PBO via CSCV) and a trial-ledger module, both in lab/ — persistent append-only hypothesis record feeding the deflation. |
| 8 | Exit gate | A deliberately overfit control strategy the lab must reject, plus a verify-lab make target. |

Ordering follows §20.4 (features → labelling → costs → CPCV last, validated by the control),
with stage-5 adjustment inserted ahead of it for the reason above.

## Decisions

**Spread — a `SpreadSource` port, not a hardcoded estimator.** The corpus has no quotes, so
commit 5 cannot measure a spread today. Rather than pick an estimator and bake it in, the cost
model takes spread from a port with three implementations over the project's life: the real
quote in live; an estimate from daily bars in research now; real historical `BID_ASK` bars once
the crawler can fetch them. The estimator is a versioned, config-selected plugin so it can be
replaced once there is something to calibrate against.

The research default is an estimate **floored**, never used raw:
`spread = max(estimate, one tick, per-liquidity-bucket floor)`, with negative estimates — which
high/low estimators produce often — clamped to the floor rather than discarded. The bias is
deliberately toward *over*-stating cost: §17 names the optimistic backtest as the project's
primary risk, so a strategy wrongly rejected is the cheaper error.

*Unverified:* whether IBKR serves historical `BID_ASK` bars to this account. `whatToShow` is
hardcoded to `"TRADES"` at `adapters/ibkr/market_data.py:268`; `ib_async` 2.1.0 passes the
string through, so nothing in our code blocks it. A probe on 2026-09-17 22:38 ET connected
(server 178) but hung before any historical response — inside IBKR's nightly data-farm restart
window, the same symptom `08-gotchas.doc.md` records. **Re-probe during RTH.** If BID_ASK
works, real spreads for the crawled window replace the estimator and this question closes.
Note a BID_ASK bar's OHLC does not carry normal bar semantics, so it needs its own record type
rather than passing through `Bar` — verify the field meanings against a live response before
modelling them.

**Adjustment source — yfinance actions, self-validated by unexplained gaps.** `get_actions()`
on yfinance 1.7.0 returns `Dividends` and `Stock Splits` with tz-aware dates (checked
2026-09-17: AAPL, 97 action rows, splits 2005 2:1, 2014 7:1, 2020 8-31 4:1). Splits are stamped
at 09:30 ET, i.e. effective from that session's open, which is the PIT rule for the factor
series. Free, covers `.TO`, and already the daily feed's source.

Its weakness is silent omission, so commit 1 carries its own check: any overnight gap in the
daily corpus past a threshold that no recorded action explains is flagged. A missing split is
the failure mode that matters, and this catches it for nothing. IBKR `ADJUSTED_LAST` as a
cross-check is **deferred** — daily-only, pacing-expensive, and it catches the same class the
gap check already does.

**Commit 2 does not block.** The unexplained-gap validation moves into commit 1, where it
belongs: it is the proof the adjustment is correct, not a separate report. What remains in
commit 2 — duplicate prints, halt marking, session-gap detection against the venue calendar,
survivorship audit — is report-only and gates nothing. It lands next for sequencing, but
commits 3–8 do not wait on it.

**Crawl timing — not late evening.** One request per instrument-session; the pacer allows 55 per
600s, so ~330/hour. At 43 names × ~252 sessions, one year of depth is ~10.8k requests ≈ **33
hours of crawling**. IBKR charges nothing per historical request and the entitlements are
already proven, so the cost is wall-clock, not money. Start it in RTH or weekday morning — the
22:38 ET probe hang above is what a nightly restart window looks like.

**Model — sonnet for 1–3, opus for 4, 6, 7.** Commits 1–3 are execution against a settled
design. Labelling, CPCV and DSR/PBO are the methodology where being wrong is silent.

## Open risks

- **Survivorship.** The universe artifact is flagged biased (yfinance lists survivors only).
  The audit in commit 2 can measure it but cannot fix it without a delisted-name source.
- **Spread remains estimated** until BID_ASK is confirmed or quotes are captured live. Every
  cost-adjusted number before then carries that caveat, and commit 5 should say so in its output
  rather than only in a doc.
