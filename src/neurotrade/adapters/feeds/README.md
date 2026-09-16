# adapters/feeds

Free intraday samples — FirstRateData and Kibot — used to seed the lab before
the IBKR backfill has run long enough to matter (TRADER_PLAN §12.1 stage 2).

## Files

| File | What it does |
|---|---|
| `seed_sources.py` | `SeedSourcesFile` — reads `config/seed_sources.yaml` into vendor file → `Symbol` entries |
| `vendor_download.py` | Downloads the FRD and Kibot free samples over plain HTTPS, writes them under `<raw_dir>/vendor/<source>/<date>/` with a `manifest.json` |
| `firstrate.py` | `FirstRateFeed` — `MarketDataPort` reading a FirstRateData sample zip already on disk |
| `kibot.py` | `KibotFeed` — `MarketDataPort` reading a Kibot `_unadjusted` sample file already on disk |
| `_bars.py` | The ET-open to UTC-close conversion shared by both feeds |
| `errors.py` | `FeedError` — raised by the feeds and the downloader |

## Both vendors enter through `MarketDataPort`

`firstrate.py` and `kibot.py` are feeds, the same shape as `ibkr/market_data.py`:
the backfill crawler (`ingest/crawler.py`) drives them the same way, so the
same calendar trim, resumability and outcome reporting apply with no second
ingestion path. This layer only reads a file already on disk; deciding what to
crawl, and where the file came from, is the caller's job.

## The download step is optional

`vendor_download.py` is a convenience, not a requirement: both vendors serve
plain HTTPS with no login, so a `seed fetch` command (Phase 0 commit 2) is
just automation on top of it. A file dropped into
`<raw_dir>/vendor/<source>/<date>/` by hand works identically — the feeds only
read the directory, however the file got there.

## Vendor files are raw and immutable

Never written twice: `vendor_download.py` refuses to overwrite an existing
file, and every file gets a `manifest.json` entry — url, sha256, bytes,
`fetched_at`, and adjustment basis — written beside it. Kibot's ~3-month
window rolls, so every fetch is dated and none is assumed to be the same
snapshot as the last.

## Bar timestamps: open in the file, close in the domain

Both vendors stamp a bar at its **open**, in US/Eastern, naive. `Bar.ts_event`
is the **close**, in UTC — the same shift `ibkr/market_data.py` applies live,
done here without a socket. Getting this backwards is the systematic
lookahead bug described in `08-gotchas.doc.md`.

## Scope: 1-minute bars only, no calendar trim here

Both feeds serve only `BarInterval.MIN_1` — the only size either sample format
holds — and raise `FeedError` for anything else, or for a symbol not in the
mapping they were built with. Neither trims to regular trading hours itself:
FRD's extended-hours rows and Kibot's regular-hours-only rows both pass
through unfiltered, because the crawler's session trim
(`TradingSession.holds_bar`) already does that on the way into the corpus.

## Nothing here is committed

Both vendors' licences forbid redistribution, and `data/` is already
gitignored. Tests use a few hand-written rows in each vendor's own format,
never real vendor rows.
