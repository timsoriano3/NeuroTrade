# adapters/feeds

Free intraday samples — FirstRateData and Kibot — used to seed the lab before
the IBKR backfill has run long enough to matter (TRADER_PLAN §12.1 stage 2).

## Files

| File | What it does |
|---|---|
| `seed_sources.py` | `SeedSourcesFile` — reads `config/seed_sources.yaml` into vendor file → `Symbol` entries |
| `vendor_download.py` | Downloads the FRD and Kibot free samples over plain HTTPS, writes them under `<raw_dir>/vendor/<source>/<date>/` with a `manifest.json`, and reads those snapshots back (`snapshots`, `latest_snapshot`, `read_manifest`) |
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

## Who drives this

`neurotrade seed fetch` and `neurotrade seed ingest` (in `cli.py`, with
`make seed`) are the only callers: fetch writes a dated snapshot, ingest hands
the files in it to the feed below and crawls it into
`<derived_dir>/seed/<source>/`. The composition lives in the CLI because
`ingest/` may not import a concrete adapter.

The download step is a convenience, not a requirement: both vendors serve
plain HTTPS with no login. A file dropped into
`<raw_dir>/vendor/<source>/<date>/` by hand works identically — the feeds only
read the directory, however the file got there — as long as the folder's
`manifest.json` describes it, which is what `seed ingest` reads the provenance
from.

## The crawl range comes from the files

`FirstRateFeed.coverage()` and `KibotFeed.coverage()` report the first and
last ET session date the configured files hold, which is what `seed ingest`
crawls over. A range typed in by hand would be wrong as soon as it moved:
FRD's window is fixed but undocumented in the file itself, and Kibot's rolls.
The date is the **open's** date in Eastern, recovered by undoing the
open-to-close shift below — a 19:59 ET bar closes at 00:59 UTC the next day,
and dating it by the close would stretch the span past what the file holds.

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
