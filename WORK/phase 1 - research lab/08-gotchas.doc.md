# Gotchas — Phase 1

Each of these cost real time. They are recorded so they cost it once. Phase 0's list is in
`WORK/phase 0 - foundation and data spine/08-gotchas.doc.md` and still applies.

## Data

**Yahoo's OHLC is ALWAYS split-adjusted. `auto_adjust=False` only turns off DIVIDEND adjustment.**
`13-daily-bars.doc.md` asserted the daily corpus was unadjusted; it is not, and there is no option
to make it so. Measured: AMZN's stored closes run 121-125 straight through its 20:1 split on
2022-06-06, a week it traded near $2,400. The first real run of `actions check` reported thirteen
gaps, of which nine were this double-adjustment and not bad data. Hence `already_split_adjusted`
on `scan_gaps`, set by the CLI for this source. **The IBKR minute corpus and the vendor samples
are NOT pre-adjusted** — the basis differs per source and nothing in a `Bar` records it, so any
new reader has to be told.

**`Ticker.get_actions()` returns only the columns a name actually has.** A name that has only paid
dividends comes back with no `Stock Splits` column; one that has only split comes back with no
`Dividends`. Indexing blindly raises `KeyError` on exactly the cleanest histories — SPY and TSLA
were the first two to fail. A missing column means zero, not an error.

**`Ticker.get_actions()` takes `period` and nothing else.** Passing `timeout=` raises `TypeError`
and every symbol in the universe fails at once. `history()` accepts one only because it forwards
`**kwargs`. Checked with `inspect.signature` against the installed version, which is the habit
that would have avoided writing it.

**Yahoo reports "no split on this row" as `0.0`, not `1.0`.** Passed through, it hits
`CorporateAction`'s positive-ratio check and raises. The identity is substituted at the adapter
boundary, which is where the convention is known.

## Decimal

**`Decimal.normalize()` pushes whole numbers into an exponent.** `Decimal("10.00").normalize()` is
`1E+1`, and `Decimal(1000) / Decimal("0.5")` is already `2E+3` before anything normalizes. Every
value stays numerically correct and every comparison still holds — what breaks is `str()` in a
report, the value written to Parquet, and any doctest. `tidy_decimal` in `core/actions.py`
requantizes when the exponent goes positive. Found by a doctest, not by review.

**`decimal128(18, 8)` refuses a value with more places than it holds**, rather than rounding it:
"Rescaling Decimal value would cause data loss". A dividend factor is `1 - 0.26/169.34`, which
runs to the context precision, so adjusted prices are quantized at the boundary.

## Process

**`pytest.raises(match=...)` is a REGEX — escape the dots in a symbol.** `AAPL.NASDAQ` in a match
pattern fails ruff RUF043. Phase 0 recorded this four times; this commit made it five.

**`str.replace()` without a count edits docstrings too.** Adding an import by replacing the
existing import line also rewrote the identical line inside two `Example:` blocks, and `ruff
format` then re-indented both docstrings trying to parse the result. Doctest collection caught it.
Anchor on something unique, or pass a count.

## Corpus queries

**`DuckDBCatalog.gaps` is one query per instrument-session, not per symbol.** Calling it inside a
per-session loop over five years of 43 names is ~52,000 queries against 53,879 tiny Parquet files;
the first daily audit ran past ten minutes and was killed. It is also vacuous for daily bars — one
bar per session has no inside — so `audit_corpus` skips it for `DAY_1` entirely. Check the arity
of a catalog method before looping over it.

**A dataclass in a report needs `__str__` or the terminal gets its `repr`.** The first real
`corpus check` printed `(Symbol(ticker='JPM', venue=<Venue.NYSE: 'NYSE'>), datetime.date(2026, 9,
11))` per line. Anything a command prints gets a `__str__`; anything it only counts does not.

## Features

**A smoothed indicator that carries state across windows breaks reproducibility.** Wilder's ATR
and a continuously-maintained EMA both depend on where the series started, so the same bar gets a
different value depending on how much history the slice included — backtest and live disagree,
silently. Both are computed from the declared window alone here, and tests pin it by prepending 50
unrelated bars and asserting no change.

**`Decimal` division runs to context precision, and a `Price` is an order price.** `session_vwap`
over a real session returned `330.2556630065255193527176013`, which `decimal128(18, 8)` refuses
outright. Any computed `Price` gets quantized at the boundary. Second time this class of bug
appeared; see the Decimal section above.

**`Price` refuses a non-positive value at construction.** A zero-division guard against a zero
price is unreachable code pretending to be a safeguard — a test asserted the guard's message and
got "Price must be positive" instead, which is how it was found.

## Labelling and costs

**`FlooredSpread` floors at one tick, so there is no such thing as a free cost model.** A test
fixture named `FREE` with every fee zeroed still charged a tick crossed twice, and a 2% gross win
came back as 1.99%. The floor is correct — nothing trades tighter than a tick — but a test that
expects exact gross arithmetic will fail, and the fixture is the thing that is wrong.

**Costs are bigger than small barriers.** A 10bp profit target on a $100 name does not survive a
5c spread crossed twice plus commission; break-even was around 6bp at 1,000 shares. Worth knowing
before wondering why a scalping strategy labels as a loss.

**`lab/` cannot be imported by `execution/`.** Anything both research and live need — the cost
model, the feature implementations, the labelling definitions that live code must agree with —
belongs in `core/`. The plan put costs in `lab/` and it had to move.

## Validation harness

**Purging against the hull of all test groups destroys the training set.** A CPCV split holds out
`k` groups that are usually not adjacent — 0 and 5 of 6, say. Taking `(min start, max end)` across
all of them and purging anything that overlaps purges the entire middle of the sample, which *is*
the training data, and reports it as a purge count nobody reads. `purge_and_embargo` therefore
takes `test_blocks` — one sequence per contiguous run — not a flat test set. Nothing fails when
this is wrong: the splits still run, training is just tiny and the model is noise.

**A constant return series has no Sharpe ratio, and that silently inverts PBO.** A CSCV fixture of
two strategies with flat returns (`0.02` every period versus `-0.01` every period) makes both
columns zero-variance, so both score 0.0, every split ties, and PBO comes back 1.0 for a strategy
that was strictly better in every block. Any fixture feeding `probability_of_backtest_overfitting`
needs real variance. `_column_sharpe` scores a degenerate column 0.0 rather than raising, because
a strategy that did not trade in a subsample is a real outcome to rank.

**`pytest.raises(match=...)` is a REGEX — an alternation needs a raw string.**
`match="must (be at least 1|not be negative)"` fails ruff RUF043. Sixth occurrence across the
project.

**A trailing field comment that pushes a line past 100 chars gets the value wrapped in parens.**
`ruff format` turned a `StrEnum` member into `DISCOVERY = (\n "discovery"  # ...\n)` rather than
moving the comment. Field comments are required by the documentation standard, so keep the whole
line inside the limit instead of relying on the formatter.
