"""Tests for corpus queries.

These are the questions the crawler asks to decide what to fetch and the quality
gate asks to decide whether the corpus can be trusted, so the cases that matter
are the ones about absence: missing sessions, holes inside a session, and an
empty corpus during the first weeks of a backfill.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from neurotrade.adapters.storage.duckdb_catalog import Coverage, DuckDBCatalog, Gap
from neurotrade.adapters.storage.parquet_store import ParquetStore
from neurotrade.core.clock import SimClock
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
TD_TSX = Symbol("TD", Venue.TSX)
TD_US = Symbol("TD", Venue.NASDAQ)

MINUTE = 60_000_000_000
OPEN_NS = 1_773_495_000_000_000_000
MON = date(2026, 3, 16)
TUE = date(2026, 3, 17)
WED = date(2026, 3, 18)


def bar(index: int, *, symbol: Symbol = AAPL, interval: BarInterval = BarInterval.MIN_1) -> Bar:
    ts = OPEN_NS + index * MINUTE
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=interval,
        open=Price("100"),
        high=Price("100"),
        low=Price("100"),
        close=Price("100"),
        volume=Quantity(1_000),
    )


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[ParquetStore, DuckDBCatalog]:
    root = tmp_path / "bars"
    return ParquetStore(root, SimClock(OPEN_NS)), DuckDBCatalog(root)


# ── An empty corpus is normal, not an error ──────────────────


def test_every_query_works_on_an_empty_corpus(tmp_path: Path) -> None:
    """For weeks of a backfill this is the state of most instruments."""
    catalog = DuckDBCatalog(tmp_path / "nothing-here")
    assert catalog.coverage() == ()
    assert catalog.sessions_held(AAPL) == ()
    assert catalog.gaps(AAPL, MON) == ()
    assert catalog.duplicate_timestamps() == ()
    assert catalog.summary() == {"bars": 0, "instruments": 0, "sessions": 0}
    assert catalog.missing_sessions(AAPL, [MON, TUE]) == (MON, TUE)


# ── Coverage ─────────────────────────────────────────────────


def test_coverage_reports_one_row_per_instrument_session(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    store.write_bars([bar(i) for i in range(10)], source="ibkr", session_date=MON)
    store.write_bars([bar(i) for i in range(5)], source="ibkr", session_date=TUE)

    entries = catalog.coverage(AAPL)
    assert [(entry.session_date, entry.bar_count) for entry in entries] == [(MON, 10), (TUE, 5)]


def test_coverage_records_the_time_span(corpus: tuple[ParquetStore, DuckDBCatalog]) -> None:
    store, catalog = corpus
    store.write_bars([bar(i) for i in range(10)], source="ibkr", session_date=MON)
    entry = catalog.coverage(AAPL)[0]
    assert entry.first_ts == OPEN_NS
    assert entry.last_ts == OPEN_NS + 9 * MINUTE


def test_coverage_spans_instruments_when_unfiltered(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    store.write_bars([bar(0, symbol=AAPL), bar(0, symbol=SHOP)], source="ibkr", session_date=MON)
    assert {entry.symbol for entry in catalog.coverage()} == {AAPL, SHOP}


def test_coverage_separates_the_same_ticker_on_two_venues(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    """TD trades on TSE in CAD and on NYSE in USD — different prices."""
    store, catalog = corpus
    store.write_bars([bar(0, symbol=TD_TSX), bar(0, symbol=TD_US)], source="ibkr", session_date=MON)
    assert {entry.symbol for entry in catalog.coverage()} == {TD_TSX, TD_US}


def test_coverage_is_per_interval(corpus: tuple[ParquetStore, DuckDBCatalog]) -> None:
    store, catalog = corpus
    store.write_bars(
        [bar(0, interval=BarInterval.MIN_1), bar(0, interval=BarInterval.MIN_5)],
        source="ibkr",
        session_date=MON,
    )
    assert catalog.coverage(AAPL, BarInterval.MIN_1)[0].bar_count == 1
    assert catalog.coverage(AAPL, BarInterval.MIN_5)[0].bar_count == 1


def test_completeness_allows_extended_hours(corpus: tuple[ParquetStore, DuckDBCatalog]) -> None:
    """Pre- and post-market bars push a session past the regular-hours count."""
    store, catalog = corpus
    store.write_bars([bar(i) for i in range(400)], source="ibkr", session_date=MON)
    entry = catalog.coverage(AAPL)[0]
    assert entry.is_complete(390)
    assert not entry.is_complete(500)


# ── Provenance ───────────────────────────────────────────────


def test_sources_are_recorded_per_session(corpus: tuple[ParquetStore, DuckDBCatalog]) -> None:
    store, catalog = corpus
    store.write_bars([bar(0)], source="ibkr", session_date=MON)
    assert catalog.coverage(AAPL)[0].sources == ("ibkr",)


def test_a_session_stitched_from_two_feeds_is_flagged(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    """IEX-only volume is partial, so a mixed session has a discontinuity that
    no price column reveals."""
    store, catalog = corpus
    store.write_bars([bar(0)], source="ibkr", session_date=MON)
    store.write_bars([bar(1)], source="alpaca_iex", session_date=MON)
    entry = catalog.coverage(AAPL)[0]
    assert entry.sources == ("alpaca_iex", "ibkr")
    assert entry.is_mixed_source


def test_a_single_source_session_is_not_flagged(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    store.write_bars([bar(0), bar(1)], source="ibkr", session_date=MON)
    assert not catalog.coverage(AAPL)[0].is_mixed_source


# ── Missing sessions: the crawler's work queue ───────────────


def test_missing_sessions_is_the_calendar_minus_what_is_held(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    store.write_bars([bar(0)], source="ibkr", session_date=TUE)
    assert catalog.missing_sessions(AAPL, [MON, TUE, WED]) == (MON, WED)


def test_nothing_is_missing_when_everything_is_held(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    for session in (MON, TUE):
        store.write_bars([bar(0)], source="ibkr", session_date=session)
    assert catalog.missing_sessions(AAPL, [MON, TUE]) == ()


def test_missing_sessions_is_per_instrument(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    store.write_bars([bar(0, symbol=AAPL)], source="ibkr", session_date=MON)
    assert catalog.missing_sessions(AAPL, [MON]) == ()
    assert catalog.missing_sessions(SHOP, [MON]) == (MON,)


def test_extra_held_sessions_are_not_reported(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    """Holding a day the calendar did not list is not a gap to fill."""
    store, catalog = corpus
    for session in (MON, TUE):
        store.write_bars([bar(0)], source="ibkr", session_date=session)
    assert catalog.missing_sessions(AAPL, [MON]) == ()


# ── Bar counts: what plan_backfill asks per symbol ───────────


def test_bar_counts_is_empty_for_an_empty_corpus(tmp_path: Path) -> None:
    catalog = DuckDBCatalog(tmp_path / "nothing-here")
    assert catalog.bar_counts(AAPL) == {}


def test_bar_counts_is_per_session_ascending(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    """Written out of order; the mapping still comes back oldest first."""
    store, catalog = corpus
    store.write_bars([bar(i) for i in range(3)], source="ibkr", session_date=WED)
    store.write_bars([bar(i) for i in range(5)], source="ibkr", session_date=MON)
    store.write_bars([bar(i) for i in range(2)], source="ibkr", session_date=TUE)

    counts = catalog.bar_counts(AAPL)
    assert counts == {MON: 5, TUE: 2, WED: 3}
    assert list(counts) == [MON, TUE, WED]


def test_sessions_with_nothing_held_are_absent_not_zero(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    store.write_bars([bar(0)], source="ibkr", session_date=MON)
    counts = catalog.bar_counts(AAPL)
    assert counts == {MON: 1}
    assert TUE not in counts


def test_bar_counts_is_per_interval(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    """A session's 1-minute bars are not counted as 5-minute bars."""
    store, catalog = corpus
    store.write_bars(
        [bar(i, interval=BarInterval.MIN_1) for i in range(3)], source="ibkr", session_date=MON
    )
    store.write_bars([bar(0, interval=BarInterval.MIN_5)], source="ibkr", session_date=MON)

    assert catalog.bar_counts(AAPL, BarInterval.MIN_1) == {MON: 3}
    assert catalog.bar_counts(AAPL, BarInterval.MIN_5) == {MON: 1}


def test_bar_counts_agrees_with_coverage(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    store.write_bars([bar(i) for i in range(10)], source="ibkr", session_date=MON)
    store.write_bars([bar(i) for i in range(5)], source="ibkr", session_date=TUE)

    from_bar_counts = catalog.bar_counts(AAPL)
    from_coverage = {entry.session_date: entry.bar_count for entry in catalog.coverage(AAPL)}
    assert from_bar_counts == from_coverage


# ── Gaps inside a session ────────────────────────────────────


def test_a_contiguous_session_has_no_gaps(corpus: tuple[ParquetStore, DuckDBCatalog]) -> None:
    store, catalog = corpus
    store.write_bars([bar(i) for i in range(30)], source="ibkr", session_date=MON)
    assert catalog.gaps(AAPL, MON) == ()


def test_a_hole_is_found_and_measured(corpus: tuple[ParquetStore, DuckDBCatalog]) -> None:
    """Minutes 10-14 absent: a halt, or a fetch that failed."""
    present = [bar(i) for i in list(range(10)) + list(range(15, 20))]
    store, catalog = corpus
    store.write_bars(present, source="ibkr", session_date=MON)

    found = catalog.gaps(AAPL, MON)
    assert len(found) == 1
    assert found[0].after_ts == OPEN_NS + 9 * MINUTE
    assert found[0].before_ts == OPEN_NS + 15 * MINUTE
    assert found[0].missing_bars == 5


def test_several_holes_are_reported_in_order(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    indices = [0, 1, 5, 6, 10]
    store, catalog = corpus
    store.write_bars([bar(i) for i in indices], source="ibkr", session_date=MON)
    found = catalog.gaps(AAPL, MON)
    assert [gap.missing_bars for gap in found] == [3, 3]
    assert found[0].after_ts < found[1].after_ts


def test_gaps_are_scoped_to_one_session(corpus: tuple[ParquetStore, DuckDBCatalog]) -> None:
    """The overnight break between sessions is not a gap."""
    store, catalog = corpus
    store.write_bars([bar(0)], source="ibkr", session_date=MON)
    store.write_bars([bar(5_000)], source="ibkr", session_date=TUE)
    assert catalog.gaps(AAPL, MON) == ()
    assert catalog.gaps(AAPL, TUE) == ()


def test_gaps_respect_the_interval(corpus: tuple[ParquetStore, DuckDBCatalog]) -> None:
    """Five-minute bars one minute apart are not contiguous; five apart are."""
    store, catalog = corpus
    store.write_bars(
        [bar(0, interval=BarInterval.MIN_5), bar(5, interval=BarInterval.MIN_5)],
        source="ibkr",
        session_date=MON,
    )
    assert catalog.gaps(AAPL, MON, BarInterval.MIN_5) == ()


# ── Duplicates should never exist ────────────────────────────


def test_no_duplicates_after_repeated_writes(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    """Proves the store's deduplication rather than assuming it — a duplicate
    reads as double the volume and nothing downstream would flag it."""
    store, catalog = corpus
    bars = [bar(i) for i in range(20)]
    for _ in range(3):
        store.write_bars(bars, source="ibkr", session_date=MON)
    assert catalog.duplicate_timestamps() == ()
    assert catalog.coverage(AAPL)[0].bar_count == 20


# ── Totals ───────────────────────────────────────────────────


def test_summary_counts_bars_instruments_and_sessions(
    corpus: tuple[ParquetStore, DuckDBCatalog],
) -> None:
    store, catalog = corpus
    store.write_bars(
        [bar(0, symbol=AAPL), bar(1, symbol=AAPL), bar(0, symbol=SHOP)],
        source="ibkr",
        session_date=MON,
    )
    store.write_bars([bar(0, symbol=AAPL)], source="ibkr", session_date=TUE)
    assert catalog.summary() == {"bars": 4, "instruments": 2, "sessions": 2}


# ── Value objects ────────────────────────────────────────────


def test_gap_missing_bars_arithmetic() -> None:
    gap = Gap(
        symbol=AAPL,
        session_date=MON,
        interval=BarInterval.MIN_1,
        after_ts=OPEN_NS,
        before_ts=OPEN_NS + 5 * MINUTE,
    )
    assert gap.missing_bars == 4


def test_coverage_completeness_boundary() -> None:
    entry = Coverage(
        symbol=AAPL,
        session_date=MON,
        interval=BarInterval.MIN_1,
        bar_count=389,
        first_ts=0,
        last_ts=1,
        sources=("ibkr",),
    )
    assert not entry.is_complete(390)
