"""Tests for the on-disk bar schema.

The round-trip tests go through a real Parquet file rather than a dict, because
a dict round trip proves nothing about the file format. What matters is whether
a bar written to disk and read back is the same bar — if it is not, backtests
computed from the corpus disagree with live runs and nothing says why.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neurotrade.adapters.storage.schemas import (
    BAR_SCHEMA,
    SCHEMA_VERSION,
    Source,
    bar_to_row,
    partition_path,
    row_to_bar,
)
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
NOW = 1_773_495_000_000_000_000
SESSION = date(2026, 3, 14)


def make_bar(**overrides: object) -> Bar:
    defaults: dict[str, object] = {
        "symbol": AAPL,
        "ts_event": NOW,
        "ts_init": NOW + 2_000_000,
        "seq": 3,
        "interval": BarInterval.MIN_1,
        "open": Price("100.12345678"),
        "high": Price("101.87654321"),
        "low": Price("99.00000001"),
        "close": Price("100.5"),
        "volume": Quantity("1234567.89"),
    }
    return Bar(**{**defaults, **overrides})  # type: ignore[arg-type]


def write_and_read(bars: list[Bar], tmp_path: Path) -> list[dict[str, object]]:
    """Round-trip bars through an actual Parquet file."""
    written = [
        bar_to_row(bar, source=Source.IBKR, session_date=SESSION, ingested_at=NOW) for bar in bars
    ]
    path = tmp_path / "bars.parquet"
    pq.write_table(pa.Table.from_pylist(written, schema=BAR_SCHEMA), path)
    read_back: list[dict[str, object]] = pq.read_table(path).to_pylist()
    return read_back


# ── The round trip must be exact ─────────────────────────────


def test_a_bar_survives_a_parquet_round_trip(tmp_path: Path) -> None:
    original = make_bar()
    assert row_to_bar(write_and_read([original], tmp_path)[0]) == original


def test_decimal_precision_survives_to_disk(tmp_path: Path) -> None:
    """The reason the columns are DECIMAL and not DOUBLE.

    float64 cannot represent 100.12345678 exactly, so a float column would
    return a different number than was written — and every backtest computed
    from the corpus would disagree with a live run for no findable reason.
    """
    original = make_bar(open=Price("100.12345678"), low=Price("99.00000001"))
    restored = row_to_bar(write_and_read([original], tmp_path)[0])
    assert restored.open.value == Decimal("100.12345678")
    assert restored.low.value == Decimal("99.00000001")


def test_nanosecond_timestamps_survive_to_disk(tmp_path: Path) -> None:
    """A Parquet timestamp column would read back as datetime, which is only
    microsecond-precise — silently truncating three digits on every row."""
    original = make_bar(ts_event=NOW + 123, ts_init=NOW + 456)
    restored = row_to_bar(write_and_read([original], tmp_path)[0])
    assert restored.ts_event == NOW + 123
    assert restored.ts_init == NOW + 456


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"vwap": Price("100.4"), "trade_count": 812},
        {"volume": Quantity(0)},  # illiquid minute
        {"symbol": SHOP},  # Canadian listing
        {"interval": BarInterval.DAY_1},
        {"seq": 0},
        {"open": Price("100"), "high": Price("100"), "low": Price("100"), "close": Price("100")},
    ],
)
def test_round_trip_across_shapes(overrides: dict[str, object], tmp_path: Path) -> None:
    original = make_bar(**overrides)
    assert row_to_bar(write_and_read([original], tmp_path)[0]) == original


def test_many_bars_round_trip_in_order(tmp_path: Path) -> None:
    originals = [make_bar(ts_event=NOW + i * 60_000_000_000, seq=i) for i in range(50)]
    restored = [row_to_bar(row) for row in write_and_read(originals, tmp_path)]
    assert restored == originals


def test_optional_columns_stay_absent(tmp_path: Path) -> None:
    """None must round-trip as None, not as zero — a zero trade_count is a claim."""
    restored = row_to_bar(write_and_read([make_bar()], tmp_path)[0])
    assert restored.vwap is None
    assert restored.trade_count is None


# ── Schema shape ─────────────────────────────────────────────


def test_row_keys_match_the_schema_exactly() -> None:
    """A row with a stray key silently fails at write time; a missing one nulls."""
    row = bar_to_row(make_bar(), source=Source.IBKR, session_date=SESSION, ingested_at=NOW)
    assert set(row) == set(BAR_SCHEMA.names)


def test_schema_version_is_recorded_in_the_file(tmp_path: Path) -> None:
    """A reader can tell which shape it is looking at instead of inferring."""
    rows = [bar_to_row(make_bar(), source=Source.IBKR, session_date=SESSION, ingested_at=NOW)]
    path = tmp_path / "bars.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=BAR_SCHEMA), path)
    metadata = pq.read_schema(path).metadata
    assert metadata[b"schema_version"] == SCHEMA_VERSION.encode()


def test_price_and_volume_columns_are_decimal_not_float() -> None:
    for column in ("open", "high", "low", "close", "vwap"):
        assert pa.types.is_decimal(BAR_SCHEMA.field(column).type)
    assert pa.types.is_decimal(BAR_SCHEMA.field("volume").type)


def test_timestamp_columns_are_int64_not_timestamp() -> None:
    for column in ("ts_event", "ts_init", "ingested_at"):
        assert BAR_SCHEMA.field(column).type == pa.int64()


def test_only_the_optional_columns_are_nullable() -> None:
    nullable = {field.name for field in BAR_SCHEMA if field.nullable}
    assert nullable == {"vwap", "trade_count"}


# ── Provenance ───────────────────────────────────────────────


def test_provenance_is_written() -> None:
    row = bar_to_row(make_bar(), source=Source.KIBOT, session_date=SESSION, ingested_at=NOW + 5)
    assert row["source"] == "kibot"
    assert row["ingested_at"] == NOW + 5


def test_provenance_is_dropped_on_read(tmp_path: Path) -> None:
    """A Bar carrying its source would compare unequal to an identical bar from
    a different feed, which would break deduplication across sources."""
    from_ibkr = bar_to_row(make_bar(), source=Source.IBKR, session_date=SESSION, ingested_at=1)
    from_kibot = bar_to_row(make_bar(), source=Source.KIBOT, session_date=SESSION, ingested_at=2)
    assert row_to_bar(from_ibkr) == row_to_bar(from_kibot)


def test_session_date_is_supplied_not_derived() -> None:
    """A US post-market bar at 19:30 ET is 00:30 UTC the next day.

    Deriving the partition from ts_event would scatter one session across two
    directories every evening.
    """
    post_market = NOW + 11 * 3_600_000_000_000  # ~00:30 UTC the following day
    row = bar_to_row(
        make_bar(ts_event=post_market),
        source=Source.IBKR,
        session_date=SESSION,  # still the 14th, per the venue calendar
        ingested_at=NOW,
    )
    assert row["session_date"] == SESSION


# ── Exactness guards ─────────────────────────────────────────


def test_a_float_in_a_price_column_is_refused() -> None:
    """Routing through float would reintroduce the imprecision decimals prevent."""
    row = bar_to_row(make_bar(), source=Source.IBKR, session_date=SESSION, ingested_at=NOW)
    row["close"] = 100.5
    with pytest.raises(TypeError, match="refusing to build a Decimal from float"):
        row_to_bar(row)


def test_a_corrupt_row_is_rejected_by_the_domain_model() -> None:
    """Reading does not bypass validation: a file cannot smuggle in an
    impossible bar that then propagates through every feature."""
    row = bar_to_row(make_bar(), source=Source.IBKR, session_date=SESSION, ingested_at=NOW)
    row["high"] = Decimal("1")  # below the low
    with pytest.raises(ValueError, match="below low"):
        row_to_bar(row)


# ── Partitioning ─────────────────────────────────────────────


def test_partition_layout_is_venue_then_ticker_then_date() -> None:
    path = partition_path(Path("data/raw/bars_1m"), AAPL, SESSION)
    assert path.as_posix() == "data/raw/bars_1m/venue=NASDAQ/ticker=AAPL/session_date=2026-03-14"


def test_the_same_ticker_on_two_venues_lands_in_different_partitions() -> None:
    """TD is Toronto-Dominion on TSX and Tandem Diabetes on NASDAQ."""
    root = Path("data/raw/bars_1m")
    canadian = partition_path(root, Symbol("TD", Venue.TSX), SESSION)
    american = partition_path(root, Symbol("TD", Venue.NASDAQ), SESSION)
    assert canadian != american


def test_partition_values_are_hive_encoded() -> None:
    """DuckDB and Arrow both recognise key=value directories and prune on them."""
    parts = partition_path(Path("root"), AAPL, SESSION).parts[1:]
    assert all("=" in part for part in parts)
