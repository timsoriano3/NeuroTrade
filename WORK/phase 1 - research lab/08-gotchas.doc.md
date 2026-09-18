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
