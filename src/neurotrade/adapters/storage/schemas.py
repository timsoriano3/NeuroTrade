"""The on-disk shape of the corpus, and the mapping to and from domain objects.

**The mapping is explicit, field by field.** Reflecting over dataclass fields
would be shorter and would silently change the file format the moment someone
adds a field to `Bar`. A format that changes without anyone deciding to change
it is unreadable history: every file written before the change parses
differently from every file written after, and nothing announces it.

**Decimals stay decimals.** Parquet has a native `DECIMAL(p, s)` type, so the
exactness that `neurotrade.core.types` insists on in memory survives to disk and
back. Storing prices as `float64` would quantise on every write, and a bar
reloaded from the corpus would differ from the bar that was computed — which
makes a replay digest differ from a live run for no reason anyone could find.

**Timestamps are `int64` nanoseconds, not a Parquet timestamp.** Parquet's
`timestamp('ns')` is stored as int64 anyway, but reading it back through Arrow
produces a Python `datetime`, which holds only microseconds. That would silently
truncate every timestamp on a round trip. Raw int64 keeps `Nanos` exactly what
it is in memory; `DuckDBCatalog` provides human-readable views for querying.

**Partitioning is `venue / ticker / session_date`.** Venue comes first and is
not optional: `TD` is Toronto-Dominion on TSX and Tandem Diabetes on NASDAQ, and
a layout keyed on ticker alone would merge two unrelated companies into one
directory.

**`session_date` is supplied, not derived.** It is tempting to take the UTC date
from `ts_event`, and it is wrong: a US post-market bar at 19:30 ET is 00:30 UTC
the *next* day, so deriving would scatter one session across two partitions
every evening. The feed knows the venue calendar; this layer does not, so the
feed supplies it.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import pyarrow as pa

from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue

__all__ = [
    "BAR_SCHEMA",
    "SCHEMA_VERSION",
    "Source",
    "bar_to_row",
    "partition_path",
    "row_to_bar",
]

SCHEMA_VERSION = "1"
"""Bumped whenever the column set or a type changes. Written into the file
metadata so a reader can tell which shape it is looking at instead of guessing
from the columns present."""

PRICE_TYPE = pa.decimal128(18, 8)
"""Room for 10 digits before the point and 8 after. US equities quote to 4
decimals; 8 leaves headroom for VWAP and derived levels without ever rounding."""

QUANTITY_TYPE = pa.decimal128(28, 8)
"""Wider than price because share counts get large — a heavily traded name can
print hundreds of millions of shares in a session — and fractional shares are
supported, so the scale is needed too."""


class Source(StrEnum):
    """Where a row came from.

    Recorded on every row because §12.1 seeds the corpus from several places
    before the IBKR backfill completes, and those sources disagree. IEX-only
    data has partial volume, free samples have gaps, and a bar's provenance is
    the only way to know whether a volume feature computed from it means
    anything.

    Example:
        >>> Source.IBKR.value
        'ibkr'
    """

    IBKR = "ibkr"  # primary; full consolidated volume
    YFINANCE = "yfinance"  # daily bars and corporate actions
    FIRSTRATE = "firstrate"  # free intraday samples, bootstrap only
    KIBOT = "kibot"  # free 1-minute samples, bootstrap only
    ALPACA_IEX = "alpaca_iex"  # IEX only — partial volume, research use only
    SYNTHETIC = "synthetic"  # generated; never for alpha discovery (§9.3-H)


BAR_SCHEMA = pa.schema(
    [
        # Instrument. Split into ticker and venue rather than one "AAPL.NASDAQ"
        # string so both can be partition keys without escaping a separator.
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        # Time. int64 nanoseconds since the Unix epoch, UTC — see module docstring.
        pa.field("ts_event", pa.int64(), nullable=False),
        pa.field("ts_init", pa.int64(), nullable=False),
        pa.field("seq", pa.int32(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
        # Prices and size, exact.
        pa.field("open", PRICE_TYPE, nullable=False),
        pa.field("high", PRICE_TYPE, nullable=False),
        pa.field("low", PRICE_TYPE, nullable=False),
        pa.field("close", PRICE_TYPE, nullable=False),
        pa.field("volume", QUANTITY_TYPE, nullable=False),
        # Optional: not every feed supplies these.
        pa.field("vwap", PRICE_TYPE, nullable=True),
        pa.field("trade_count", pa.int64(), nullable=True),
        # Partition key and provenance.
        pa.field("session_date", pa.date32(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.int64(), nullable=False),
    ],
    metadata={"schema_version": SCHEMA_VERSION, "dataset": "bars"},
)
"""Canonical column layout for bar data.

Column order is fixed and meaningful: instrument, then time, then values, then
provenance. Readers should select by name, but a stable order keeps diffs of
`parquet-tools schema` output readable across versions."""

PARTITION_KEYS = ("venue", "ticker", "session_date")
"""Directory levels, outermost first. Venue leads because it disambiguates the
ticker; date is innermost because almost every query is a date range over a few
instruments, and that layout lets the reader prune whole directories."""


def bar_to_row(
    bar: Bar,
    *,
    source: Source,
    session_date: date,
    ingested_at: Nanos,
) -> dict[str, object]:
    """Flatten a `Bar` into one Parquet row.

    Args:
        bar: The bar to write.
        source: Which feed produced it.
        session_date: The trading day it belongs to, in the venue's terms. Not
            derived from `ts_event` — see the module docstring.
        ingested_at: When we wrote it, for provenance.

    Returns:
        A mapping matching `BAR_SCHEMA`, with `Decimal` values preserved exactly.

    Example:
        >>> from datetime import date
        >>> row = bar_to_row(
        ...     Bar(symbol=AAPL, ts_event=1_000, ts_init=1_000,
        ...         interval=BarInterval.MIN_1, open=Price("100"), high=Price("101"),
        ...         low=Price("99"), close=Price("100.5"), volume=Quantity(1_000)),
        ...     source=Source.IBKR, session_date=date(2026, 3, 14), ingested_at=2_000,
        ... )
        >>> (row["ticker"], row["venue"], row["close"])
        ('AAPL', 'NASDAQ', Decimal('100.5'))
    """
    return {
        "ticker": bar.symbol.ticker,
        "venue": bar.symbol.venue.value,
        "ts_event": bar.ts_event,
        "ts_init": bar.ts_init,
        "seq": bar.seq,
        "interval": bar.interval.value,
        "open": bar.open.value,
        "high": bar.high.value,
        "low": bar.low.value,
        "close": bar.close.value,
        "volume": bar.volume.value,
        "vwap": None if bar.vwap is None else bar.vwap.value,
        "trade_count": bar.trade_count,
        "session_date": session_date,
        "source": source.value,
        "ingested_at": ingested_at,
    }


def row_to_bar(row: dict[str, object]) -> Bar:
    """Rebuild a `Bar` from a Parquet row.

    The inverse of `bar_to_row`. Provenance columns are dropped: they describe
    the row, not the bar, and a `Bar` that carried them would compare unequal to
    an identical bar from a different feed.

    Args:
        row: A mapping with the columns of `BAR_SCHEMA`.

    Returns:
        The reconstructed bar, equal to the one that was written.

    Raises:
        ValueError: If the row violates a domain rule — an inverted high and
            low, a negative volume. Reaching this means the file is corrupt or
            was written by something that bypassed the domain model.

    Example:
        >>> from datetime import date
        >>> original = Bar(
        ...     symbol=AAPL, ts_event=1_000, ts_init=1_000, interval=BarInterval.MIN_1,
        ...     open=Price("100"), high=Price("101"), low=Price("99"),
        ...     close=Price("100.5"), volume=Quantity(1_000),
        ... )
        >>> row = bar_to_row(original, source=Source.IBKR,
        ...                  session_date=date(2026, 3, 14), ingested_at=2_000)
        >>> row_to_bar(row) == original
        True
    """
    vwap = row["vwap"]
    trade_count = row["trade_count"]
    return Bar(
        symbol=Symbol(str(row["ticker"]), Venue(str(row["venue"]))),
        ts_event=_int(row["ts_event"]),
        ts_init=_int(row["ts_init"]),
        seq=_int(row["seq"]),
        interval=BarInterval(str(row["interval"])),
        open=Price(_decimal(row["open"])),
        high=Price(_decimal(row["high"])),
        low=Price(_decimal(row["low"])),
        close=Price(_decimal(row["close"])),
        volume=Quantity(_decimal(row["volume"])),
        vwap=None if vwap is None else Price(_decimal(vwap)),
        trade_count=None if trade_count is None else _int(trade_count),
    )


def _int(value: object) -> int:
    """Coerce a stored integer column back to `int`.

    Columns arrive typed as `object` from a row mapping, and `int()` on an
    arbitrary object is not something a type checker can accept. Narrowing here
    keeps the call sites clean and rejects anything unexpected loudly.
    """
    if isinstance(value, int):
        return value
    raise TypeError(f"expected an integer column, got {type(value).__name__}: {value!r}")


def _decimal(value: object) -> Decimal:
    """Coerce a stored value back to `Decimal` without going via `float`.

    Arrow returns `Decimal` for decimal columns, so this is normally a no-op.
    It exists for the case where a row arrives from somewhere looser — a hand-
    written fixture, a DuckDB result — and routing through `float` there would
    reintroduce exactly the imprecision the decimal column exists to avoid.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        raise TypeError(f"refusing to build a Decimal from float {value!r}; the column is exact")
    return Decimal(str(value))


def partition_path(root: Path, symbol: Symbol, session_date: date) -> Path:
    """Directory a bar belongs in, under Hive-style partitioning.

    Args:
        root: Dataset root, e.g. `data/raw/bars_1m`.
        symbol: The instrument. Its venue is the outermost key.
        session_date: The trading day.

    Returns:
        The partition directory. Values are written `key=value` so DuckDB and
        Arrow both recognise the layout and can prune on it without a manifest.

    Example:
        >>> from datetime import date
        >>> partition_path(Path("data/raw/bars_1m"), AAPL, date(2026, 3, 14)).as_posix()
        'data/raw/bars_1m/venue=NASDAQ/ticker=AAPL/session_date=2026-03-14'
    """
    return (
        root
        / f"venue={symbol.venue.value}"
        / f"ticker={symbol.ticker}"
        / f"session_date={session_date.isoformat()}"
    )
