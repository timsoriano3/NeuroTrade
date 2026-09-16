# adapters/feeds

Market data from outside IBKR. Free intraday samples — FirstRateData and Kibot
— seed the lab before the IBKR backfill has run long enough to matter
(TRADER_PLAN §12.1 stage 2), and Yahoo Finance supplies daily bars across the
whole universe, Canadian `.TO` lines included (§12.1 stage 3).

## Files

| File | What it does |
|---|---|
| `seed_sources.py` | `SeedSourcesFile` — reads `config/seed_sources.yaml` into vendor file → `Symbol` entries |
| `vendor_download.py` | Downloads the FRD and Kibot free samples over plain HTTPS, writes them under `<raw_dir>/vendor/<source>/<date>/` with a `manifest.json`, and reads those snapshots back (`snapshots`, `latest_snapshot`, `read_manifest`) |
| `firstrate.py` | `FirstRateFeed` — `MarketDataPort` reading a FirstRateData sample zip already on disk |
| `kibot.py` | `KibotFeed` — `MarketDataPort` reading a Kibot `_unadjusted` sample file already on disk |
| `yfinance_daily.py` | `YFinanceDailyFeed` — `MarketDataPort` serving unadjusted daily bars from Yahoo, one download per symbol; `YahooDownloader` is the network half and `yahoo_ticker` the name mapping |
| `_bars.py` | The ET-open to UTC-close conversion shared by the two sample feeds |
| `errors.py` | `FeedError` — raised by the feeds and the downloader |

## Every vendor enters through `MarketDataPort`

`firstrate.py`, `kibot.py` and `yfinance_daily.py` are feeds, the same shape as
`ibkr/market_data.py`: the backfill crawler (`ingest/crawler.py`) drives all of
them the same way, so the same calendar trim, resumability and outcome
reporting apply with no second ingestion path. Deciding *what* to crawl is the
caller's job — the sample feeds read a file someone already put on disk, and
the Yahoo feed fetches the range it was constructed with.

## Who drives this

`neurotrade daily backfill` (with `make daily`) drives `yfinance_daily.py`
straight into `<derived_dir>/daily/yfinance/` — there is no fetch half, because
Yahoo answers a whole range in one call and keeps no file on disk.

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

## Yahoo's daily bars: one download, stamped at the close, unadjusted

Three things differ from the sample feeds and are deliberate.

**One request per symbol.** Yahoo answers a multi-year daily history in a
single call while the crawler offers work one instrument-session at a time, so
the whole configured range is downloaded on the first cell for a symbol and
cached for the rest.

**Stamped at the venue's session close**, taken from `CalendarPort`. Yahoo
indexes a daily row at local midnight — the *start* of the day — and keeping
that would claim the whole day's range was known before the open. Early closes
need no special case, because the calendar already holds them.

**Unadjusted** (`auto_adjust=False`): the OHLC written is what traded on the
day, and Yahoo's `Adj Close` is discarded. Adjustment is recomputable from
corporate actions (§12.1 stage 5); a back-adjusted price would bake today's
split history into a bar that is never re-downloaded.

A Yahoo row on a date the venue calendar denies is dropped, not stamped, and
reported by `unmatched()` — either Yahoo invented a row or our holidays are
wrong, and both want a human.

## Scope: one interval per feed, no calendar trim here

The sample feeds serve only `BarInterval.MIN_1` — the only size either format
holds — and the Yahoo feed only `BarInterval.DAY_1`; each raises `FeedError`
for anything else, or for a symbol it was not built with. Yahoo serves intraday
too, over a short trailing window and with partial volume, and this feed refuses
it rather than seeding the corpus with bars that disagree with IBKR's.

No feed here trims to regular trading hours itself: FRD's extended-hours rows
and Kibot's regular-hours-only rows both pass through unfiltered, because the
crawler's session trim (`TradingSession.holds_bar`) already does that on the way
into the corpus.

## Nothing here is committed, and no test reaches a network

Both sample vendors' licences forbid redistribution, and `data/` is already
gitignored. Tests use a few hand-written rows in each vendor's own format,
never real vendor rows; the Yahoo tests inject a fake downloader, and the one
test of the real `YahooDownloader` replaces `yfinance.Ticker`.
