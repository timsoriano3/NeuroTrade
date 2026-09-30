"""The universe as of one closed tick.

The lag is the point, so the first tests are about it: a strategy acting at
10:31 must never see a 10:31 bar from another instrument, whatever order the
feed delivered them in. After that, the arithmetic — a trailing return measured
on a shared clock, and a beta that is `None` rather than one when it cannot be
estimated.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from neurotrade.core.types import Price, Symbol, Venue
from neurotrade.features.cross_section import (
    BETA_MEMORY,
    CrossSection,
    CrossSectionTracker,
    InstrumentSnapshot,
)
from neurotrade.features.levels import SessionLevels


class Fixed:
    """A `BenchmarkSource` that ignores the date.

    The real one is `core.sectors.SectorMap`, which is point-in-time; these
    tests are about the tracker's arithmetic and not about reclassification, so
    a flat answer keeps them reading as one idea. `test_sectors.py` is where the
    dated behaviour is pinned.
    """

    def __init__(self, pairs: dict[Symbol, Symbol]) -> None:
        self._pairs = pairs

    def benchmark_of(self, symbol: Symbol, *, on: date) -> Symbol | None:
        del on
        return self._pairs.get(symbol)


AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
SPY = Symbol("SPY", Venue.ARCA)

MINUTE = 60_000_000_000
CLOSE_NS = 390 * MINUTE
JULY_8 = date(2024, 7, 8)


def levels(
    *,
    close: str,
    open_: str = "100",
    minute: int = 0,
    day: date = JULY_8,
    volume: str = "1000",
    rvol: float | None = None,
) -> SessionLevels:
    price = Price(close)
    return SessionLevels(
        session_date=day,
        open_ns=0,
        close_ns=CLOSE_NS,
        session_open=Price(open_),
        high=price,
        low=price,
        close=price,
        vwap=price,
        bar_count=minute + 1,
        prior_close=Price(open_),
        session_volume=Decimal(volume),
        relative_volume_from_open=rvol,
    )


# ── The lag, which is the whole design ──────────────────────────────────


def test_nothing_is_visible_until_a_tick_has_closed() -> None:
    tracker = CrossSectionTracker()
    tracker.observe(levels(close="101"), AAPL, 0)
    assert tracker.snapshot() is None


def test_a_tick_closes_only_when_the_clock_moves_past_it() -> None:
    """Both instruments' bars at one tick land in the same snapshot, and only after."""
    tracker = CrossSectionTracker()
    tracker.observe(levels(close="101"), AAPL, MINUTE)
    tracker.observe(levels(close="102"), MSFT, MINUTE)
    assert tracker.snapshot() is None
    tracker.observe(levels(close="103", minute=2), AAPL, 2 * MINUTE)
    section = tracker.snapshot()
    assert section is not None
    assert section.as_of == MINUTE
    assert section.symbols == (AAPL, MSFT)


def test_the_snapshot_never_carries_the_tick_being_assembled() -> None:
    """The no-lookahead guarantee, stated as an inequality.

    A same-tick view would hand the first instrument dispatched the other
    names' bars before they were published — every bar, for every symbol but
    the last.
    """
    tracker = CrossSectionTracker()
    for minute in range(1, 5):
        tracker.observe(levels(close="101", minute=minute), AAPL, minute * MINUTE)
        section = tracker.snapshot()
        if section is not None:
            assert section.as_of < minute * MINUTE


def test_an_instrument_that_did_not_print_is_stale_not_absent() -> None:
    """Dropping it would make a thin name vanish on the minutes it is quietest.

    That is a selection effect on exactly the names a stocks-in-play filter is
    supposed to find.
    """
    tracker = CrossSectionTracker()
    tracker.observe(levels(close="101"), AAPL, MINUTE)
    tracker.observe(levels(close="102"), MSFT, MINUTE)
    tracker.observe(levels(close="103", minute=2), AAPL, 2 * MINUTE)
    tracker.observe(levels(close="104", minute=3), AAPL, 3 * MINUTE)
    section = tracker.snapshot()
    assert section is not None
    assert MSFT in section.rows  # carried forward from the tick it last printed on


def test_time_going_backwards_is_a_wiring_bug_not_a_recoverable_state() -> None:
    tracker = CrossSectionTracker()
    tracker.observe(levels(close="101", minute=2), AAPL, 2 * MINUTE)
    with pytest.raises(ValueError, match="cross-section went backwards"):
        tracker.observe(levels(close="101"), AAPL, MINUTE)


# ── The measures ────────────────────────────────────────────────────────


def test_the_session_return_is_the_close_over_the_open() -> None:
    tracker = CrossSectionTracker()
    tracker.observe(levels(close="102", open_="100"), AAPL, MINUTE)
    tracker.observe(levels(close="102", open_="100", minute=2), AAPL, 2 * MINUTE)
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].session_return == pytest.approx(0.02)


def test_a_trailing_return_is_measured_on_the_session_clock() -> None:
    """Between two points of the session-return profile, keyed by minute."""
    tracker = CrossSectionTracker(trailing_minutes=2)
    for minute, close in enumerate(("100", "101", "102", "103")):
        tracker.observe(levels(close=close, minute=minute), AAPL, minute * MINUTE)
    tracker.observe(levels(close="103", minute=4), AAPL, 4 * MINUTE)
    section = tracker.snapshot()
    assert section is not None
    # Minute 3 against minute 1: 103/101 - 1.
    assert section.rows[AAPL].trailing_return == pytest.approx(103 / 101 - 1)


def test_a_missing_print_leaves_the_trailing_return_none_not_longer() -> None:
    """An instrument with a gap must not be ranked over a longer look than its neighbours."""
    tracker = CrossSectionTracker(trailing_minutes=2)
    tracker.observe(levels(close="100", minute=0), AAPL, 0)
    # Minute 1 never prints, so minute 3's two-minute look has no base.
    tracker.observe(levels(close="102", minute=3), AAPL, 3 * MINUTE)
    tracker.observe(levels(close="102", minute=4), AAPL, 4 * MINUTE)
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].trailing_return is None


def test_the_window_cannot_reach_before_the_session_open() -> None:
    tracker = CrossSectionTracker(trailing_minutes=30)
    tracker.observe(levels(close="101", minute=1), AAPL, MINUTE)
    tracker.observe(levels(close="101", minute=2), AAPL, 2 * MINUTE)
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].trailing_return is None


def test_relative_volume_is_taken_from_the_levels_not_recomputed() -> None:
    """One implementation of a session's volume profile, in `features/levels.py`."""
    tracker = CrossSectionTracker()
    tracker.observe(levels(close="101", rvol=2.5), AAPL, MINUTE)
    tracker.observe(levels(close="101", minute=2, rvol=2.5), AAPL, 2 * MINUTE)
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].relative_volume == 2.5


def test_a_positive_trailing_minutes_is_required() -> None:
    with pytest.raises(ValueError, match="trailing_minutes 0 must be at least 1"):
        CrossSectionTracker(trailing_minutes=0)


# ── Beta ────────────────────────────────────────────────────────────────


def run_sessions(tracker: CrossSectionTracker, moves: list[tuple[float, float]]) -> None:
    """Drive one open-to-close observation per instrument per session."""
    for index, (own, market) in enumerate(moves):
        day = date(2024, 1, 1) + timedelta(days=index)
        for symbol, move in ((AAPL, own), (SPY, market)):
            tracker.observe(
                levels(close=f"{100 * (1 + move):.8f}", day=day, minute=1),
                symbol,
                index * 1_000 * MINUTE + MINUTE,
            )


def test_no_benchmark_means_no_beta_rather_than_a_default_of_one() -> None:
    """Assuming one is assuming the answer for the names whose answer is missing."""
    tracker = CrossSectionTracker()
    run_sessions(tracker, [(0.01, 0.01)] * (BETA_MEMORY + 2))
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].beta is None


def test_a_short_history_has_no_beta() -> None:
    tracker = CrossSectionTracker(benchmarks=Fixed({AAPL: SPY}))
    run_sessions(tracker, [(0.01, 0.01), (-0.02, -0.01)] * 5)
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].beta is None


def test_a_beta_of_two_is_recovered_from_sessions_that_move_twice_as_far() -> None:
    tracker = CrossSectionTracker(benchmarks=Fixed({AAPL: SPY}))
    moves = [(0.02 * step, 0.01 * step) for step in (1, -1, 2, -2, 1, -3)]
    run_sessions(tracker, moves * (BETA_MEMORY // len(moves) + 2))
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].beta == pytest.approx(2.0, abs=0.05)


def test_a_benchmark_that_never_moved_has_no_slope() -> None:
    tracker = CrossSectionTracker(benchmarks=Fixed({AAPL: SPY}))
    moves = [(0.02 * step, 0.0) for step in (1, -1, 2, -2)]
    run_sessions(tracker, moves * (BETA_MEMORY // len(moves) + 2))
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].beta is None


# ── The snapshot's own arithmetic ───────────────────────────────────────


def test_a_residual_strips_what_the_beta_explains() -> None:
    row = InstrumentSnapshot(
        AAPL, session_return=0.02, trailing_return=None, relative_volume=None, beta=1.5
    )
    assert row.residual_against(0.01) == pytest.approx(0.005)


def test_a_row_with_no_beta_has_no_residual() -> None:
    """A raw return in a column of residuals is at an extreme on any day the market moved."""
    row = InstrumentSnapshot(AAPL, 0.02, None, None, None)
    assert row.residual_against(0.01) is None
    assert not row.is_residualised


def test_every_row_is_residualised_against_the_same_benchmark_tick() -> None:
    """Filled at tick close, not per bar.

    Per bar, the names dispatched before the benchmark would use its previous
    tick and the rest this one — a ranking whose column is measured two ways.
    """
    tracker = CrossSectionTracker(benchmarks=Fixed({AAPL: SPY, MSFT: SPY}))
    moves = [(0.02 * step, 0.01 * step, 0.03 * step) for step in (1, -1, 2, -2)]
    for index, (own, market, other) in enumerate(moves * (BETA_MEMORY // 4 + 2)):
        day = date(2024, 1, 1) + timedelta(days=index)
        # SPY dispatched between the two names, so a per-bar fill would split them.
        for symbol, move in ((AAPL, own), (SPY, market), (MSFT, other)):
            tracker.observe(
                levels(close=f"{100 * (1 + move):.8f}", day=day, minute=1),
                symbol,
                index * 1_000 * MINUTE + MINUTE,
            )
    section = tracker.snapshot()
    assert section is not None
    market_return = section.rows[SPY].session_return
    for symbol in (AAPL, MSFT):
        row = section.rows[symbol]
        assert row.residual == pytest.approx(row.residual_against(market_return))


def test_an_instrument_with_no_benchmark_gets_no_residual() -> None:
    tracker = CrossSectionTracker()
    tracker.observe(levels(close="101"), AAPL, MINUTE)
    tracker.observe(levels(close="101", minute=2), AAPL, 2 * MINUTE)
    section = tracker.snapshot()
    assert section is not None
    assert section.rows[AAPL].residual is None


def test_a_ranking_skips_rows_the_measure_is_none_for() -> None:
    section = CrossSection(
        as_of=0,
        rows={
            AAPL: InstrumentSnapshot(AAPL, 0.01, None, None, 1.0),
            MSFT: InstrumentSnapshot(MSFT, 0.03, None, None, None),
        },
    )
    assert section.rank_by(lambda row: row.beta) == (AAPL,)


def test_ties_break_on_the_symbol_so_two_runs_agree() -> None:
    """Exact ties in a rounded measure are common at one-minute bars."""
    section = CrossSection(
        as_of=0,
        rows={
            MSFT: InstrumentSnapshot(MSFT, 0.01, None, None, None),
            AAPL: InstrumentSnapshot(AAPL, 0.01, None, None, None),
        },
    )
    assert section.rank_by(lambda row: row.session_return) == (AAPL, MSFT)


def test_a_rank_carries_the_total_it_is_out_of() -> None:
    """Third of four is weak and third of fifty is strong."""
    section = CrossSection(
        as_of=0,
        rows={
            AAPL: InstrumentSnapshot(AAPL, 0.01, None, None, None),
            MSFT: InstrumentSnapshot(MSFT, 0.03, None, None, None),
        },
    )
    assert section.rank_of(MSFT, lambda row: row.session_return) == (0, 2)
    assert section.rank_of(SPY, lambda row: row.session_return) is None


def test_ranking_ascending_reverses_the_order() -> None:
    section = CrossSection(
        as_of=0,
        rows={
            AAPL: InstrumentSnapshot(AAPL, 0.01, None, None, None),
            MSFT: InstrumentSnapshot(MSFT, 0.03, None, None, None),
        },
    )
    assert section.rank_by(lambda r: r.session_return, descending=False) == (AAPL, MSFT)


def test_is_rankable_requires_every_cold_capable_field() -> None:
    assert InstrumentSnapshot(AAPL, 0.01, 0.002, 2.0, 1.0).is_rankable
    assert not InstrumentSnapshot(AAPL, 0.01, None, 2.0, 1.0).is_rankable
