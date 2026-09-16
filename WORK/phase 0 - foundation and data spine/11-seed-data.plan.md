# Plan — §12.1 stage 2: free sample seed data

**Status: plan, not built.** Spec (§12.1 table) times stage 2 in **Phase 0**: "seed with
FirstRateData and Kibot free samples so the lab has something to work with immediately".
Detail docs get written per commit; this is the map.

## What exists upstream (researched 2026-09-12)

| | FirstRateData | Kibot |
|---|---|---|
| Free intraday | AAPL AMZN MSFT META TSLA; SPY QQQ DIA EEM VXX; indices SPX DJI VIX NDX RUT | IBM, OIH 1-min (adjusted **and** `_unadjusted`); IVE, WDC tick |
| Window | fixed 1 yr, 2022-09-30 → 2023-09-29 (AAPL, verified) | rolling ~3 months; IBM file ran 2026-06-15 → 2026-09-11 (verified) |
| Fetch | direct: `frd001.s3-us-east-2.amazonaws.com/{T}_1min_sample_firstratedata.zip` (verified) | direct, opaque codes: `api.kibot.com/?get=<code>`, codes scraped from `free-historical-intraday-data.html`, file named by `Content-Disposition` (verified) |
| Format | zip → CSV, header `timestamp,open,high,low,close,volume`, `YYYY-MM-DD HH:MM:SS` | CSV, no header, `MM/DD/YYYY,HH:MM,O,H,L,C,V` |
| Timezone | ET (vendor claim) | ET (vendor docs) |
| Bar stamp | **open** — inferred: 04:00–19:59, never 20:00 | **open** — 09:30–15:59, never 16:00 |
| Hours | extended, 04:00–20:00 | regular only |
| Adjusted | vendor says split+dividend; **unverified** for sample | default adjusted; `_unadjusted` file exists |
| No-trade minutes | absent, not zero rows | absent (unverified) |
| Canada | none | none |
| Licence | private use only (secondhand) | internal use only, no redistribution (verified page) |

## Decisions

1. **The adapter downloads the files itself.** Both vendors serve plain HTTPS with no
   login, so a manual step would just be a way to get it wrong. `neurotrade seed fetch` pulls
   them with stdlib `urllib` (no new dependency). A file dropped in by hand works too: ingest
   only reads the directory, however the files got there.
2. **Vendor files are raw and never touched:** `data/raw/vendor/<source>/<fetched-date>/<original
   name>`, plus a `manifest.json` in each folder (url, sha256, bytes, fetched_at). The folder is
   dated because Kibot's window rolls, so every fetch is a new snapshot and never overwrites
   an old one.
3. **Normalised bars are derived:** `data/derived/seed/<source>/`, one `ParquetStore` root per
   source, rebuilt from the vendor files. **Never inside `data/raw/bars/` (IBKR's folder).**
   The catalog counts bars without looking at source, and the store keeps the *first* row per
   `(interval, ts_event, seq)`. Sample bars there would make the crawler skip those sessions,
   and would also win over IBKR's bars for the same minutes.
4. **Nothing is committed.** Both licences forbid redistribution; `data/` is already
   gitignored. Tests use a few hand-written rows in each vendor's format, never real vendor
   rows.
5. **Vendor files enter through `MarketDataPort`,** and ingest is the existing `crawl`, not new
   code. A file-backed feed answers `fetch_bars` from the parsed file, so the same calendar
   trim, resumability and outcome report come for free, and research gets no second
   ingestion path.
6. **Normalise to the corpus shape:** ET → UTC via `zoneinfo`; bar-open stamp → close stamp (+1
   interval, the IBKR rule); drop extended hours via `TradingSession.holds_bar`; prices through
   `Price.from_float`.
7. **Kibot: take the `_unadjusted` files only.** Adjusting prices is stage 5's job, and raw
   must stay raw. FRD has no unadjusted variant, so its manifest says `adjusted: unknown`
   until checked (see below).
8. **Scope is 1-minute bars only.** Skip the FRD index files (no volume, not tradable
   listings) and Kibot's IVE/WDC tick files (Phase 4 depth territory).
9. **Ticker → `Symbol` lives in `config/seed_sources.yaml`** (source, file, ticker, venue),
   so each vendor file maps to an explicit venue. Venue guesses (e.g. VXX → BATS, OIH) are
   **unverified** until an IBKR qualify resolves them.

## Commits

Grouped by scope — two, not five.

1. **Seed feeds** (adapters, one scope) — **DONE, see `12-seed-feeds.doc.md`**: *feeds: vendor_download* (FRD URL pattern, Kibot page
   scrape, manifest, dated folders), *feeds: firstrate* and *feeds: kibot* — file-backed
   `MarketDataPort` feeds: parse, ET→UTC, open→close, RTH trim — plus
   `config/seed_sources.yaml` and the feeds README. Tests on hand-written vendor rows.
2. **Seed commands and verification** — **DONE, see `12-seed-feeds.doc.md`**:
   `neurotrade seed fetch` / `seed ingest` + *make seed* / *seed-fetch* / *seed-ingest*, each
   source crawled into `derived/seed/<source>/` with the manifest sha256 logged, plus a live
   fetch and ingest (1,019,421 bars). **Both cross-checks are still open and are blocked, not
   skipped:** the Kibot-vs-IBKR overlap needs a successful weekday IBKR crawl (`raw/bars/` is
   empty), and FRD AAPL vs yfinance needs yfinance, which arrives in stage 3. What the checks
   will compare is recorded in `12-seed-feeds.doc.md`.

## Not in this stage

Gap/halt marking, split/dividend adjustment, survivorship → stage 5 (Phase 1). Unioning seed and
IBKR bars into one view for the lab → first Phase 1 consumer decides. yfinance → stage 3.

## Open

- FRD adjustment basis — **still open**, deferred to stage 3 when yfinance lands; ingested
  as-is, with `manifest.json` recording `unknown`.
- Kibot `get=` codes may rotate; if the scrape finds no match, fetch fails loudly.
