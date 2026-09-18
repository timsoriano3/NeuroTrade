# Daily bars from Yahoo — `neurotrade daily backfill`

§12.1 stage 3, first half: "daily bars and universe history via yfinance, including `.TO`
tickers". **The bars are done and in the corpus; universe history is not started** — see
*Not done* below.

## What exists

`adapters/feeds/yfinance_daily.py`:

- `YFinanceDailyFeed` — a `MarketDataPort` serving unadjusted daily bars. Constructed with a
  `CalendarPort`, a `Clock` and the `start`/`end` range it will be asked about.
- `YahooDownloader` — the network half, one `yfinance.Ticker.history` call per symbol. Injected,
  so no test reaches Yahoo.
- `yahoo_ticker(symbol)` — `SHOP.TSX` → `SHOP.TO`, `WELL.TSXV` → `WELL.V`, US listings bare,
  and a dot inside a ticker becomes a hyphen (`BRK.B` → `BRK-B`).
- `feed.unmatched()` — dates Yahoo returned that the venue calendar denies.

One command, with `make daily`:

```bash
make daily START=2021-09-15                 # to yesterday, UTC
make daily START=2024-01-02 END=2024-06-28 LIMIT=200
```

`core/calendar.py` changed with it: `TradingSession.expected_bars` now answers **1** for any
interval at least as long as the session (see the gotcha).

## Decisions worth knowing

**It is the same crawler, at `1d` instead of `1m`.** `crawl` supplies the calendar trim,
resumability and the outcome report; the feed supplies bars. No second ingestion path (§3.6).

**One HTTP request per symbol, not per cell.** Yahoo answers a multi-year history in one call
while the crawler offers work one instrument-session at a time, so the feed downloads the whole
range on the first cell for a symbol and serves the rest from memory. The `requests` count in
`daily_backfill_complete` is therefore *cells*, not HTTP calls — 43,129 cells came from 43
downloads.

**Bars are stamped at the venue's session close, from the calendar.** Yahoo indexes a daily row
at local midnight — the *start* of the day — and keeping that stamp would claim the whole day's
range was known before the open. Early closes need no special case.

**Prices are DIVIDEND-unadjusted (`auto_adjust=False`), and `Adj Close` is discarded — but they
are split-adjusted, which this doc previously got wrong.** `auto_adjust=False` suppresses only
the dividend adjustment. Yahoo's OHLC is *always* split-adjusted, and there is no option to turn
that off. Measured 2026-09-17: AMZN's stored closes run 121-125 straight through its 20:1 split
on 2022-06-06, a week when it traded near $2,400.

The consequence is not cosmetic. Applying a split factor to this corpus divides by the ratio a
second time, and the first run of `neurotrade actions check` duly reported a phantom 20x gap at
AMZN's split and twelve more like it. `scan_gaps` now takes `already_split_adjusted`, which the
CLI sets for this source. Anything else reading `derived/daily/yfinance/` must know the same
thing: **these bars are already on one split basis.** The IBKR minute corpus and the vendor
samples are not.

**Bars land in `derived/daily/yfinance/`, never beside the minute bars.** Same reason the seed
feeds get their own root: the catalog counts bars without looking at provenance, so a daily bar
in the IBKR root would mark that session held and hide a real gap.

**Yahoo's floats are rounded to 4 decimals before becoming `Price`** — the finest tick a North
American equity quotes in. See the gotcha; this one failed a live run.

**A row on a date the calendar denies is dropped and reported**, not stamped. Either Yahoo
invented a row or our holidays are wrong, and both want a human.

## Verified this session (2026-09-15)

`make check` PASS — 1173 tests, 3 deselected (ibkr), 7 import contracts, `docs-check` clean.

Live `neurotrade daily backfill --start 2021-09-15 --end 2026-09-12` over the committed universe:
**53,879 bars, 43 instruments, 1,280 distinct session dates**, 0 failed, 0 empty, 0 unmatched,
in about 85 seconds. AAPL and SHOP.TSX each hold 1,253 sessions, 2021-09-15 → 2026-09-11
(the 12th was a Saturday). No instrument-session holds more than one bar.

**The FirstRateData sample is unadjusted** — the question `12-seed-feeds.doc.md` left open.
AAPL, 2023-09-29, our corpus against our seed corpus:

| | yfinance daily (unadjusted) | FRD minutes, aggregated |
|---|---|---|
| open | 172.02 | 172.02 |
| high | 173.07 | 173.07 |
| close | 171.21 | 171.22 |

Yahoo's *adjusted* close for that day is 168.93, about 1.8% below both. FRD matches the
unadjusted line exactly on the open and high; the one-cent close difference is the official
consolidated close against the last minute bar's print. **Caveat: this tests dividends, not
splits** — AAPL's last split was 2020-08, before FRD's window opens.

## Not done

- **Universe history** — the other half of stage 3. Nothing is built. The honest shape from
  daily bars is a point-in-time liquidity screen (trailing dollar volume and price floor) over
  this corpus, with thresholds in `config/`; note that yfinance lists only *surviving* names, so
  any membership history built from it is survivorship-biased and must say so where §17 can see
  it.
- **Corporate actions are not stored.** `actions=False` on the fetch. Splits and dividends are
  what stage 5 needs to adjust prices, and they want their own dataset rather than columns on a
  bar.
- **One file per instrument-session**, so the daily corpus is 53,879 tiny Parquet files. Correct
  and consistent with the minute corpus, but at a 2,000-name universe it is millions of files;
  the Phase 1 quality gate is the place to revisit partition grain, not here.
- **The Kibot-vs-IBKR overlap check is still blocked** on a successful weekday IBKR crawl.
