# Universe history — `neurotrade universe build`

§12.1 stage 3, second half: "daily bars and universe history via yfinance". Stage 3's first half
filled `derived/daily/` (`13-daily-bars.doc.md`); this turns that corpus into a record of **who
was tradable when**, so a backtest at date `t` sees the universe as it was at `t`.

Stage 3 is now complete. The corpus quality gate remains Phase 1.

## What exists

`core/universe.py` gains two types beside `Universe`:

- `UniverseMembership(session_date, symbols)` — who was eligible on one session. Symbols rather
  than a `Universe` because a screen may legitimately admit nobody and `Universe` rejects empty.
- `UniverseHistory(rows, survivorship_biased=)` — the rows, sorted, with `as_of(day)` (latest
  membership at or before `day`), `dates`, and a BLAKE2b `digest` that **includes the bias flag**.

`ingest/universe_history.py` — `screen_universe(universe, calendar, store, *, start, end, rules,
survivorship_biased, interval=DAY_1)`. `ScreenRules` holds the window length and one
`LiquidityFloor` per currency. Reads through `StoragePort`; ingest still depends on core alone.

`adapters/universe/universe_history_parquet.py` — `UniverseHistoryStore`, one Parquet file per
source under `derived/universe/`, stamped with the config hash, the candidate universe's digest
and the history digest.

`config/base.yaml` gains `universe_screen`, and it is **hashed** rather than environmental: a
floor decides which instruments a strategy was allowed to see.

```bash
make universe START=2022-01-03 END=2026-09-11
```

## Decisions worth knowing

**The trailing window ends at `t-1`.** A bar is stamped at its session close, so the bar for
session `t` is not observable when membership for `t` is decided before the open. This is the
whole point of the module, and the first test in its file.

**A short window excludes rather than screens.** Fewer than `lookback_sessions` of history is not
a smaller sample, it is a different screen, so warm-up admits nobody.

**Median, not mean.** One frantic session must not carry a name that is otherwise untradeable.

**Per-currency floors, no conversion.** `Money` refuses a cross-currency comparison, and that is
load-bearing here: a single "$10M" applied to both markets would silently mean two things.
Defaults are USD $20M / CAD $5M median dollar volume, close ≥ 5 in both, over 20 sessions.

**An instrument is only eligible on days its own venue traded.** A date is a decision point if
*any* venue traded; Canada Day drops the TSX lines and keeps the US ones. This is why membership
on the real corpus ranges 14–43 rather than sitting at 43.

**The bias flag travels in the artifact**, not in a README. Yahoo lists names that still trade, so
anything delisted inside the range is absent from the corpus entirely and no screen can recover
it. `digest` changes with the flag, so a biased history can never be mistaken for a clean one.

**A session that admitted nobody still gets a row**, with a null ticker. Drop it and an empty
session is indistinguishable from one never evaluated, and `as_of` would carry the previous
membership across it.

**The digest is recomputed on read.** A truncated artifact would quietly change which names a
backtest could trade, and nothing downstream would notice.

## Verified this session (2026-09-15)

- `make universe START=2022-01-03 END=2026-09-11` → **1,202 sessions, 14–43 members of 43**,
  history digest `ed83640bfe2ef898`, universe digest `2ea7e3533356185d`, config hash
  `cfg_8a9b0bd899669161`, written to `data/derived/universe/yfinance/history.parquet`.
- `make check` → 1,233 passed, 3 deselected (ibkr), 7 import contracts, `docs-check` clean.
- `make verify-replay` → `4f58fe2c99cd26dc7cbb8faf033a39d1`, unchanged: the config hash moved
  when `universe_screen` was added, and G1's digest does not depend on it.

## Not done

- **The screen is close to a no-op on the current 43 names.** They are hand-picked large caps;
  every one clears every floor on every session its venue was open. The machinery is what stage 3
  owed, and it does real work once the universe reaches the ~2,000 names §12.1 targets. Expanding
  `config/universe.yaml` is Phase 2's Universe Selector work.
- **Nothing consumes the artifact yet.** `UniversePort` was deliberately left alone — the crawler
  still reads the flat `config/universe.yaml`, which is right: a crawl fills history, it does not
  replay it. The consumer arrives with the Phase 1 backtest.
- **No corporate-action handling**, so the daily closes behind the screen are unadjusted. A split
  inside the window distorts that name's dollar volume until stage 5 lands.
