# Seed data — the FirstRateData and Kibot samples, fetched and ingested

§12.1 stage 2: "seed with FirstRateData and Kibot free samples so the lab has something to work
with immediately". **Done and in the corpus.** `11-seed-data.plan.md` holds the vendor facts and
the decisions behind the shape.

## What exists

`adapters/feeds/` — `seed_sources.py` (`config/seed_sources.yaml` → vendor file ↔ `Symbol`),
`vendor_download.py` (fetch, manifest, and snapshot read-back), `firstrate.py` and `kibot.py`
(the feeds), `_bars.py` (the shared time conversion), `errors.py` (`FeedError`).

Two commands in `cli.py`, with `make seed` / `seed-fetch` / `seed-ingest`:

```bash
make seed                                  # fetch then ingest, every vendor
make seed-fetch SOURCE=kibot               # one vendor
make seed-ingest SNAPSHOT=2026-09-16       # an older dated snapshot
```

`seed fetch` downloads into `raw/vendor/<source>/<UTC date>/` beside a `manifest.json`.
`seed ingest` hands one snapshot's files to the feed and crawls them into
`derived/seed/<source>/`.

**Both vendors enter through `MarketDataPort`, and ingest is the existing `crawl`** — the same
calendar trim, resumability and `CrawlReport` the IBKR backfill gets, not a second ingestion path
(§3.6, the project's primary failure mode). The wiring lives in `cli.py` because `ingest/` may not
import a concrete adapter.

## Decisions worth knowing

**The crawl range comes from the files.** `FirstRateFeed.coverage()` / `KibotFeed.coverage()`
return the first and last **ET session date** the configured files hold, and that is what
`seed ingest` crawls. A `--start`/`--end` pair would be wrong as soon as it moved: FRD's window is
fixed but written nowhere in the file, and Kibot's rolls. The date is the *open's* ET date,
recovered by undoing the open→close shift — see the gotcha below.

**Bars land in `derived/seed/<source>/`, one root per vendor, never in `raw/bars/`.** The catalog
counts bars without looking at provenance and the store keeps the first row for a
`(interval, ts_event, seq)`, so a sample bar in the IBKR root would both mark that session complete
for the backfill and outrank IBKR's own bar for the same minute. Tested.

**Provenance is explicit.** `_SEED_PROVENANCE` maps `SeedSource → storage `Source`` rather than
relying on the two enums sharing their string values, and a test asserts every vendor has both an
entry and a feed. `seed_ingest_complete` logs the sha256 of every file in the snapshot: a vendor
sample is not reproducible, so "which rows were these" can only be answered by the digest of the
bytes they came from.

**A missing manifest fails the ingest.** Rows whose bytes cannot be identified afterwards do not
belong in the corpus. A *partial* snapshot is fine — absent files are reported and the rest is
taken.

**Re-running is safe.** The crawler plans from the corpus, so a second ingest of the same snapshot
writes nothing. A second *fetch* on the same UTC day fails on purpose (raw is immutable).

## Not done

- ~~FRD's adjustment basis is unverified~~ — **settled 2026-09-15: the sample is unadjusted.**
  AAPL 2023-09-29 matches yfinance's unadjusted open and high exactly (172.02 / 173.07) and sits
  ~1.8% above its adjusted close of 168.93. Dividends only — AAPL's window holds no split.
  `manifest.json` still records `unknown`, which is what was known when those bytes were fetched.
  Working in `13-daily-bars.doc.md`.
- **The Kibot-vs-IBKR overlap check is blocked** on a successful weekday IBKR crawl; `raw/bars/`
  is still empty.
- Splits, dividends, gap and halt marking are stage 5 (Phase 1).
- VXX and OIH venues in `config/seed_sources.yaml` are guesses until an IBKR qualify resolves
  them; the file says so.

## Verified this session (2026-09-15)

`make check` PASS — 1128 tests, 3 deselected (ibkr), 7 import contracts, `docs-check` clean.

Live `make seed-fetch` → 12 files (10 FRD zips, 2 Kibot `_unadjusted`), 21 MB + 2.2 MB, in
`data/raw/vendor/*/2026-09-16/` (the folder is UTC-dated). Live `make seed-ingest` →
**1,019,421 bars**: firstrate 974,635 over 2,510 instrument-sessions, 2022-09-30 → 2023-09-29;
kibot 44,786 over 124, 2026-06-17 → 2026-09-15. Both spans came from the files, not from a flag.

Per-symbol, checked against `VenueCalendar.expected_bars`:

| | Sessions | Bars | Complete sessions |
|---|---|---|---|
| AAPL AMZN MSFT META TSLA QQQ (NASDAQ) | 251 each | 97,530 each | 251/251 |
| SPY | 251 | 97,526 | 250 |
| EEM | 251 | 97,490 | 225 |
| DIA | 251 | 97,332 | 141 |
| VXX | 251 | 97,107 | 124 |
| IBM (kibot) | 62 | 24,167 | 52 |
| OIH (kibot) | 62 | 20,619 | 2 |

**No session anywhere holds more than its expected bars**, which is what proves the extended-hours
trim (FRD carries 04:00–20:00 ET) and the half-day closes came out right.

AAPL's last FRD session, 2023-09-29: 390 bars, first open `172.02`, last close `171.22`. Those
are the numbers the stage-3 yfinance check compared against, which is how the sample is now known
to be unadjusted.

Traps found building it are in `08-gotchas.doc.md`.
