# The seed feeds — FirstRateData and Kibot

§12.1 stage 2: "seed with FirstRateData and Kibot free samples so the lab has something to work
with immediately". The reading half exists — `adapters/feeds/`. The commands that run it
(`seed fetch`, `seed ingest`) are commit 2; see `11-seed-data.plan.md` for the plan and the
vendor facts behind it.

## What the package holds

`adapters/feeds/` — `seed_sources.py` (`config/seed_sources.yaml` → vendor file ↔ `Symbol`),
`vendor_download.py` (fetch + manifest), `firstrate.py` and `kibot.py` (the feeds), `_bars.py`
(the shared time conversion), `errors.py` (`FeedError`).

**Both vendors enter through `MarketDataPort`,** exactly as `ibkr/market_data.py` does, so
`ingest/crawler.py` drives them unchanged: same calendar trim, same resumability, same
`CrawlReport`. That is the point — a second ingestion path is how research and live drift apart,
and the spec's one-implementation rule (§3.6) is the project's primary failure mode.

A feed reads a file already on disk and nothing else. It holds a `Symbol → Path` mapping and a
`Clock`, parses each file once and caches it, answers only `MIN_1`, and rejects a symbol it was
not configured with. Downloading is a separate concern in `vendor_download.py`.

## The conversion, in `_bars.py`

Both vendors stamp a bar at its **open**, in **US/Eastern**, as naive local time.
`close_ts_ns(open_et, interval)` attaches `ZoneInfo("America/New_York")`, converts through
`to_nanos`, and adds one interval — `Bar.ts_event` is the close. Getting this backwards is the
same silent lookahead the IBKR adapter guards against, on every bar, forever.

It rejects an already-aware datetime rather than converting it: a caller passing UTC here has
misread the file's timezone, and the result would be off by four or five hours without failing.

**Why the feed needs no calendar.** FRD files carry 04:00–20:00 ET, the corpus is regular hours
only, and the crawler already trims each fetch to `session.holds_bar`. Adding a calendar here
would duplicate that and give it a second chance to disagree.

## Provenance and immutability

`vendor_download.py` writes to `<raw_dir>/vendor/<source>/<YYYY-MM-DD>/<original name>`, via a
temp file and `os.replace`, and **refuses to overwrite an existing file**. Each folder carries a
`manifest.json` — url, sha256, bytes, `fetched_at`, and adjustment basis (`unadjusted` for Kibot,
`unknown` for FRD, which has no unadjusted variant).

The folder is dated because **Kibot's window rolls**: its file is the last ~3 months, so two
fetches a week apart are different data, not a retry. The dated folder makes that visible rather
than silently replacing history.

Where the bars land is **`derived/seed/<source>/`, never `raw/bars/`** (the IBKR root). The
catalog counts bars without looking at source and the store keeps the first row written for a
`(interval, ts_event, seq)`, so sample bars in the IBKR root would both mark those sessions
complete and win over IBKR's own bars for the same minutes.

## Not done

- **`seed fetch` / `seed ingest` commands and the make target** — commit 2. Nothing has been
  ingested into the real corpus yet; the live check below used a scratch root.
- **FRD's adjustment basis** is still unverified — `manifest.json` records `unknown`. Commit 2
  settles it against yfinance around an ex-dividend date.
- **VXX and OIH venues** in `config/seed_sources.yaml` are guesses until an IBKR qualify
  resolves them; the file says so.
- Splits, dividends, gap and halt marking are stage 5 (Phase 1), not here.

## Verified

`make check` PASS, 1076 tests (62 of them in `tests/adapters/feeds/`), 7 import contracts,
`docs-check` clean. Live, against the real vendors this session:

- FRD AAPL, March 2023 — 23 sessions, 8,970 bars, exactly 23 × 390.
- Kibot IBM unadjusted, July 2026 — 22 sessions; first bar 13:31 UTC, last 20:00 UTC.
- A second pass re-offered one session that is one bar short, which is the crawler's known
  permanent-shortfall case, not a feed bug.
- Kibot's adjusted and unadjusted files disagree (first open 270.06 vs 272.00 on 2026-06-15),
  which is what confirms the `_unadjusted` files are the right ones to take.

Traps found building it are in `08-gotchas.doc.md`.
