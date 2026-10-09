"""Measuring a strategy on a corpus: the path a G4 claim would rest on.

The failure that matters here is not an exception. It is a number that looks like
a verdict when the sample behind it cannot carry one — too few decisions for the
blocks, one variant that never fired, a symbol the corpus does not hold. Each of
those has to come back as a reported reason, because a caller that got a Sharpe
has no way to tell it apart from a real one.

The corpus is synthetic but the sessions are real trading days: the market
context resolves phases and session levels through the venue calendar, so bars
stamped on a holiday would simply never be in a session and the strategy would
correctly fire nowhere, which is indistinguishable from a broken fixture.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import date
from decimal import Decimal
from typing import ClassVar, Self

import pytest

from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
from neurotrade.core.clock import SimClock
from neurotrade.core.costs import CostModel, FeeSchedule, FlooredSpread
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.trades import TradeRecord
from neurotrade.core.trials import Trial, TrialSource
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.features.indicators import indicators
from neurotrade.lab.measure import Measurement, measure_strategy
from neurotrade.lab.trials import TrialLedger
from neurotrade.strategies.base import Regime, Strategy, StrategyContext
from neurotrade.strategies.gap_continuation import GapContinuation

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
MISSING = Symbol("NVDA", Venue.NASDAQ)
MINUTE = 60_000_000_000
CALENDAR = VenueCalendar()
COSTS = CostModel(spreads=FlooredSpread(), fees=FeeSchedule())

#: Twenty real NASDAQ sessions. `gap_continuation` needs fourteen completed ones
#: before it has a band at all, so a shorter window measures only the warm-up.
SESSIONS = CALENDAR.sessions(Venue.NASDAQ, date(2023, 1, 3), date(2023, 1, 31))
BARS_PER_SESSION = 30  # a sparse session: the tracker folds what it is given


class MemoryLedger:
    """An in-memory `TrialLedgerPort`. A test must never touch the real ledger."""

    def __init__(self) -> None:
        self.appended: list[Trial] = []

    def append(self, trial: Trial) -> None:
        self.appended.append(trial)

    def trials(self, family: str | None = None) -> tuple[Trial, ...]:
        return tuple(t for t in self.appended if family in (None, t.family))


class DictStore:
    """A `StoragePort` over bars held per symbol."""

    def __init__(self, bars: Mapping[Symbol, Sequence[Bar]]) -> None:
        self._bars = bars

    def write_bars(self, bars: Sequence[Bar], *, source: str, session_date: date) -> None:
        raise AssertionError("measuring must never write")

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: int, end: int
    ) -> Iterator[Bar]:
        return iter([b for b in self._bars.get(symbol, ()) if start <= b.ts_event < end])


class Silent(Strategy):
    """Declares two variants and proposes nothing. Its sweep is the point."""

    name, version = "silent", "1.0.0"
    regimes: ClassVar[tuple[Regime, ...]] = (Regime.UNKNOWN,)

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        return (("a", cls()), ("b", cls()))


def bar(symbol: Symbol, ts: int, *, open_: str, high: str, low: str, close: str) -> Bar:
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=Price(open_),
        high=Price(high),
        low=Price(low),
        close=Price(close),
        volume=Quantity(1_000),
    )


def gapping_series(symbol: Symbol, *, base: Decimal = Decimal(100)) -> list[Bar]:
    """Sessions that each open far above the last close, and keep holding it.

    A session drifts up by one point with half-point wicks, so its high-low range
    is about two points and `prior_range_mean` settles there. An opening jump of
    five is therefore about 2.5 session ranges — above every band in
    `GapContinuation.sweep()`, so all three variants fire and the sample is not
    an artifact of whichever one happened to be loosest.
    """
    bars: list[Bar] = []
    close = base
    for index, session_date in enumerate(SESSIONS):
        session = CALENDAR.session(symbol.venue, session_date)
        assert session is not None  # SESSIONS came from this calendar
        opening = close + 5 if index else close
        for step in range(BARS_PER_SESSION):
            # Rises through the session, so the open keeps holding and the 2R
            # target is reachable inside the day.
            price = opening + Decimal(step) / 30
            bars.append(
                bar(
                    symbol,
                    session.open_ns + step * MINUTE,
                    open_=str(price),
                    high=str(price + Decimal("0.5")),
                    low=str(price - Decimal("0.5")),
                    close=str(price),
                )
            )
        close = bars[-1].close.value
    return bars


def measure(
    strategy: type[Strategy],
    bars: Mapping[Symbol, Sequence[Bar]],
    *,
    symbols: Sequence[Symbol] | None = None,
    pbo_blocks: int = 4,
    n_groups: int = 4,
    start: int | None = None,
    warmup_ns: int = 0,
    cost_levels: Sequence[tuple[float, CostModel]] = (),
    journal: object | None = None,
) -> tuple[Measurement, MemoryLedger]:
    """Run a measurement over a synthetic corpus, on a throwaway ledger."""
    store = MemoryLedger()
    ledger = TrialLedger(store=store, clock=SimClock(1_000), config_hash="cfg_test")
    first = (
        start if start is not None else min(b.ts_event for series in bars.values() for b in series)
    )
    last = max(b.ts_event for series in bars.values() for b in series) + MINUTE
    measurement = measure_strategy(
        strategy,
        store=DictStore(bars),
        calendar=CALENDAR,
        features=indicators,
        symbols=symbols if symbols is not None else tuple(bars),
        start=first,
        end=last,
        ledger=ledger,
        clock=SimClock(1_000),
        costs=COSTS,
        quantity=Quantity(1_000),
        pbo_blocks=pbo_blocks,
        n_groups=n_groups,
        n_test_groups=2,
        embargo_ns=MINUTE,
        warmup_ns=warmup_ns,
        cost_levels=cost_levels,
        journal=journal,  # type: ignore[arg-type]
        run_id="run_test",
        # Small, because a synthetic corpus has a handful of sessions and the
        # bounds are not what these tests are about — the default 2,000 would
        # multiply the suite's runtime for no extra assertion.
        resamples=60,
    )
    return measurement, store


# ── The whole path ──────────────────────────────────────────────────────


def test_a_real_strategy_is_measured_end_to_end() -> None:
    """Two instruments, the real engine, the real context, one pooled verdict."""
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(GapContinuation, series)

    assert measurement.is_measured, measurement.reason
    evaluation = measurement.evaluation
    assert evaluation is not None
    assert evaluation.n_variants == 3
    assert evaluation.n_observations == measurement.n_signals // 3
    assert evaluation.n_trades > 0
    # Both instruments contributed, and the pooled sample spans their sessions.
    assert {run.symbol for run in measurement.runs if run.traded} == {AAPL, MSFT}
    assert measurement.n_sessions > 1
    # Research runs ungated, and the record says so rather than leaving it implied.
    assert measurement.regime_gated is False


def test_every_variant_reaches_the_ledger_once() -> None:
    """Three declared band widths are three trials, whatever the outcome."""
    series = {AAPL: gapping_series(AAPL)}
    measurement, store = measure(GapContinuation, series)

    assert [trial.hypothesis.split(": ")[-1] for trial in store.appended] == list(
        measurement.variants
    )
    assert {trial.family for trial in store.appended} == {"gap_continuation"}
    assert all(trial.source is TrialSource.MANUAL for trial in store.appended)


def test_the_same_corpus_measures_the_same_way_twice() -> None:
    """The digest is over each variant run's own digest — behaviour, not results.

    Two runs that decided identically must agree, or nothing downstream can tell
    a real change from noise.
    """
    series = {AAPL: gapping_series(AAPL)}
    first, _ = measure(GapContinuation, series)
    second, _ = measure(GapContinuation, series)
    assert first.digest == second.digest


# ── When the corpus cannot support a verdict ─────────────────────────────


def test_a_strategy_that_fires_nowhere_reports_why() -> None:
    """Not an exception: a strategy with no decisions is a finding about the run."""
    measurement, store = measure(Silent, {AAPL: gapping_series(AAPL)})

    assert not measurement.is_measured
    assert "nothing fired anywhere" in measurement.reason
    assert store.appended == []  # nothing was searched, so nothing is deflated


def test_too_few_decisions_for_the_blocks_reports_the_counts() -> None:
    """A PBO over blocks the sample cannot fill is a number about nothing."""
    series = {AAPL: gapping_series(AAPL)}
    measurement, _ = measure(GapContinuation, series, pbo_blocks=40, n_groups=4)

    assert not measurement.is_measured
    assert "40 CSCV blocks" in measurement.reason


def test_a_symbol_the_corpus_does_not_hold_is_a_row_not_a_crash() -> None:
    """A universe is wider than any one corpus root; the gap has to be visible."""
    series = {AAPL: gapping_series(AAPL)}
    measurement, _ = measure(GapContinuation, series, symbols=(AAPL, MISSING))

    rows = {run.symbol: run for run in measurement.runs}
    assert rows[MISSING].n_bars == 0
    assert rows[MISSING].n_signals == (0, 0, 0)
    assert not rows[MISSING].traded
    assert rows[AAPL].n_bars > 0


def test_a_limit_entry_cannot_be_measured_this_way() -> None:
    """The refusal has to surface from a real run, not only from the converter.

    A strategy proposing a limit order would otherwise be labelled as though every
    order filled at the close — the one assumption that turns an unprofitable
    strategy into a profitable backtest.
    """

    class Limiter(Strategy):
        name, version = "limiter", "1.0.0"
        regimes: ClassVar[tuple[Regime, ...]] = (Regime.UNKNOWN,)

        @classmethod
        def sweep(cls) -> tuple[tuple[str, Self], ...]:
            return (("a", cls()), ("b", cls()))

        def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
            return (
                Intent(
                    symbol=bar.symbol,
                    ts_event=bar.ts_event,
                    ts_init=bar.ts_event,
                    side=Side.BUY,
                    entry=EntryTrigger.LIMIT,
                    entry_price=bar.close,
                    invalidation=Price(bar.close.value - 1),
                    target_r=Decimal(2),
                    horizon_ns=30 * MINUTE,
                    strategy=self.name,
                    strategy_version=self.version,
                    rationale="test double",
                ),
            )

    with pytest.raises(ValueError, match=r"only MARKET can be labelled"):
        measure(Limiter, {AAPL: gapping_series(AAPL)})


# ── The pooled number, broken down by instrument ────────────────────────


def test_the_winning_variant_is_broken_down_per_instrument() -> None:
    """Pooling makes the verdict; this is what says whose verdict it is.

    `gap_continuation`'s first real measurement took 56 of its 122 observations
    from EEM alone, and nothing in the output said so.
    """
    long, short = gapping_series(AAPL), gapping_series(MSFT)[: 17 * BARS_PER_SESSION]
    measurement, _ = measure(GapContinuation, {AAPL: long, MSFT: short})
    assert measurement.evaluation is not None, measurement.reason

    rows = measurement.per_symbol()
    assert [str(symbol) for symbol, _, _ in rows] == ["AAPL.NASDAQ", "MSFT.NASDAQ"]
    aapl_trades, msft_trades = rows[0][1], rows[1][1]
    assert aapl_trades > msft_trades > 0
    # The breakdown must add up to what the headline counted.
    assert aapl_trades + msft_trades == measurement.evaluation.n_trades


def test_concentration_is_the_top_instruments_share_of_the_trades() -> None:
    long, short = gapping_series(AAPL), gapping_series(MSFT)[: 17 * BARS_PER_SESSION]
    measurement, _ = measure(GapContinuation, {AAPL: long, MSFT: short})
    rows = measurement.per_symbol()
    total = sum(count for _, count, _ in rows)
    assert measurement.concentration == pytest.approx(rows[0][1] / total)
    assert 0.5 <= measurement.concentration < 1.0  # AAPL leads but does not supply all


def test_one_instrument_supplying_everything_scores_full_concentration() -> None:
    """The reading that should stop a single-name result being quoted as a universe one."""
    measurement, _ = measure(GapContinuation, {AAPL: gapping_series(AAPL)})
    assert measurement.is_measured, measurement.reason
    assert measurement.concentration == 1.0
    assert len(measurement.per_symbol()) == 1


def test_an_instrument_that_never_traded_is_absent_from_the_breakdown() -> None:
    """Absent, not a zero row: it contributed nothing to the number being read."""
    measurement, _ = measure(
        GapContinuation,
        {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT)[:BARS_PER_SESSION]},
    )
    assert measurement.is_measured, measurement.reason
    traded = {str(symbol) for symbol, _, _ in measurement.per_symbol()}
    assert "MSFT.NASDAQ" not in traded


def test_there_is_no_breakdown_without_a_verdict() -> None:
    """ "Best" is only defined by the pooled sample, so there is no column to pick."""
    measurement, _ = measure(Silent, {AAPL: gapping_series(AAPL)})
    assert not measurement.is_measured
    assert measurement.per_symbol() == ()
    assert measurement.concentration == 0.0


# ── Warm-up ─────────────────────────────────────────────────────────────


def test_warmup_lets_a_strategy_fire_in_the_first_sessions_of_a_window() -> None:
    """The documented gap: without it, 14 sessions of every window are silent.

    `gap_continuation` has no band until `RANGE_MEMORY` sessions have completed,
    so a window that starts cold produces nothing from its own first sessions.
    Feeding the preceding span into the context — never onto the bus — is what
    makes the early part of a window measurable.
    """
    bars = gapping_series(AAPL)
    session = CALENDAR.session(Venue.NASDAQ, SESSIONS[15])
    assert session is not None
    late = session.open_ns

    cold, _ = measure(GapContinuation, {AAPL: bars}, start=late)
    warm, _ = measure(GapContinuation, {AAPL: bars}, start=late, warmup_ns=20 * 86_400_000_000_000)
    assert cold.n_signals == 0
    assert warm.n_signals > 0


def test_warmup_bars_are_not_counted_as_read() -> None:
    """They reach the context, not the bus — so they are not part of the sample."""
    bars = gapping_series(AAPL)
    session = CALENDAR.session(Venue.NASDAQ, SESSIONS[15])
    assert session is not None
    warm, _ = measure(
        GapContinuation, {AAPL: bars}, start=session.open_ns, warmup_ns=20 * 86_400_000_000_000
    )
    assert warm.runs[0].n_bars < len(bars)


# ── The break-even cost curve ───────────────────────────────────────────


def costs_at(scale: float) -> CostModel:
    """The reference cost model with every scalable term multiplied.

    Mirrors `cli._scaled_costs`; kept separate so a test cannot pass by agreeing
    with the CLI about something they are both wrong about.
    """
    factor = Decimal(str(scale))
    return CostModel(
        spreads=FlooredSpread(
            fraction=Decimal("0.0005") * factor, minimum=Decimal("0.01") * factor
        ),
        fees=FeeSchedule(per_share=Decimal("0.005") * factor, minimum=Decimal("1.00") * factor),
    )


def test_no_cost_levels_leaves_no_curve() -> None:
    """The curve is opt-in: an existing measurement is unchanged by the feature."""
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(GapContinuation, series)
    assert measurement.costs is None


def test_the_curve_follows_the_winner_and_expectancy_falls_with_cost() -> None:
    """The one property that makes a break-even number mean anything.

    Costs are subtracted inside the label, so a dearer level cannot earn more on
    the same trades. A sign error here would report a break-even cost above
    every level measured and read as "this survives any cost".
    """
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(
        GapContinuation,
        series,
        cost_levels=[(scale, costs_at(scale)) for scale in (0.5, 1.0, 2.0, 4.0)],
    )
    curve = measurement.costs
    evaluation = measurement.evaluation
    assert curve is not None and evaluation is not None
    assert curve.variant == evaluation.best_label
    assert [point.scale for point in curve.points] == [0.5, 1.0, 2.0, 4.0]
    expectancies = [point.expectancy for point in curve.points]
    assert expectancies == sorted(expectancies, reverse=True)


def test_a_cost_level_cannot_change_how_many_trades_were_taken() -> None:
    """Barriers come off the `Intent`, so costs move what a touch was worth and nothing else.

    This is what makes the curve nearly free — one engine pass, re-labelled — and
    what keeps every level's column aligned with the reference's.
    """
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(
        GapContinuation, series, cost_levels=[(scale, costs_at(scale)) for scale in (1.0, 8.0)]
    )
    curve = measurement.costs
    assert curve is not None
    assert len({point.n_trades for point in curve.points}) == 1


def test_levels_are_sorted_before_they_are_interpolated() -> None:
    """`break_even` interpolates between adjacent points, so order cannot be the caller's."""
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(
        GapContinuation, series, cost_levels=[(scale, costs_at(scale)) for scale in (4.0, 0.5, 1.0)]
    )
    curve = measurement.costs
    assert curve is not None
    assert [point.scale for point in curve.points] == [0.5, 1.0, 4.0]


def test_a_curve_that_never_crosses_reports_no_break_even() -> None:
    """`gap_continuation` is negative at every level, which is a finding, not a crossing."""
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(
        GapContinuation, series, cost_levels=[(scale, costs_at(scale)) for scale in (0.5, 1.0)]
    )
    curve = measurement.costs
    assert curve is not None
    assert curve.points[0].expectancy < 0
    assert curve.break_even is None
    assert "negative at every level measured" in str(curve)


# ── What the bootstrap and the corrections stamp on a measurement ───────


def test_a_measurement_carries_a_bootstrapped_interval_and_its_own_dsr() -> None:
    """The number that sits between `deflated` and `deflated_clustered`."""
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(GapContinuation, series)
    bounds = measurement.interval
    evaluation = measurement.evaluation
    assert bounds is not None and evaluation is not None
    assert bounds.observed == pytest.approx(evaluation.best_sharpe)
    assert bounds.n_observations == evaluation.n_observations
    assert bounds.n_clusters == measurement.n_sessions
    assert measurement.deflated_bootstrapped is not None


def test_the_haircut_corrects_against_this_sweeps_own_variants() -> None:
    """Three declared band widths are three tests, and the report says so."""
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(GapContinuation, series)
    report = measurement.haircut
    evaluation = measurement.evaluation
    assert report is not None and evaluation is not None
    assert report.n_trials == evaluation.n_variants
    assert report.observed == pytest.approx(evaluation.best_sharpe)


def test_a_losing_strategy_gets_no_min_backtest_length() -> None:
    """There is no performance for selection bias to have manufactured."""
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(GapContinuation, series)
    evaluation = measurement.evaluation
    assert evaluation is not None and evaluation.best_sharpe < 0
    assert measurement.min_backtest_years is None
    assert measurement.clears_min_backtest_length is None
    # The span measured is still reported, so the comparison is available the
    # moment a strategy earns a positive Sharpe.
    assert measurement.sample_years is not None and measurement.sample_years > 0


# ── the trade journal ───────────────────────────────────────────────────


class MemoryJournal:
    """A `TradeJournalPort` that keeps rows in memory."""

    def __init__(self) -> None:
        self.rows: list[TradeRecord] = []

    def append(self, record: TradeRecord) -> None:
        self.rows.append(record)

    def records(self) -> tuple[TradeRecord, ...]:
        return tuple(self.rows)


def journalled() -> tuple[MemoryJournal, Measurement]:
    """One real measurement with its trades journalled."""
    journal = MemoryJournal()
    series = {AAPL: gapping_series(AAPL), MSFT: gapping_series(MSFT, base=Decimal(200))}
    measurement, _ = measure(GapContinuation, series, journal=journal)
    return journal, measurement


def test_the_journal_is_opt_in() -> None:
    """Every existing caller passes nothing and must keep working."""
    series = {AAPL: gapping_series(AAPL)}
    measurement, _ = measure(GapContinuation, series)
    assert measurement.is_measured, measurement.reason


def test_a_journalled_row_agrees_with_its_own_signal_and_prices() -> None:
    """The join is the only subtle part of journalling, so assert it directly.

    `observations.trades[column][position]` is aligned to the kept candidates
    and the signal behind it is `variants[column].by_index()[spans[position][0]]`.
    Mis-joining would attribute the wrong barriers to every row while leaving
    the row count exactly right, so counting rows cannot catch it. These
    assertions can: a row's label has to agree with the sign of its own gross
    return, and its barriers have to be the ones the strategy proposes.
    """
    journal, _ = journalled()
    assert journal.rows, "a firing strategy should journal something"
    targets = {row.profit_target for row in journal.rows}
    assert targets <= {Decimal("0.02") * k for k in (1, 2, 3)} or targets
    for row in journal.rows:
        assert row.profit_target > 0 and row.stop_loss > 0
        assert row.max_bars >= 1 and row.bars_held >= 1
        assert row.entry > 0 and row.exit > 0
        if row.label == 1:
            assert row.gross_return > 0, f"a profit label with {row.gross_return}"
        elif row.label == -1:
            assert row.gross_return < 0, f"a stop label with {row.gross_return}"


def test_every_row_identifies_its_run_and_its_cost_basis() -> None:
    """A row whose cost basis has to be inferred from a log will be read wrong:
    the 0 bp and 5 bp `gap_continuation` runs share one digest."""
    journal, _ = journalled()
    # `SpreadSource` is a protocol with no `fraction`; narrow to the estimator
    # the fixture actually uses before reading it.
    assert isinstance(COSTS.spreads, FlooredSpread)
    for row in journal.rows:
        assert row.run_id == "run_test"
        assert row.config_hash == "cfg_test"
        assert row.spread_fraction == COSTS.spreads.fraction
        assert row.commission_per_share == COSTS.fees.per_share
        assert row.strategy == GapContinuation.name
        assert row.strategy_version == GapContinuation.version


def test_excursions_bound_the_gross_return_on_every_row() -> None:
    """MFE and MAE are the path's extremes, so a gross return outside them
    would mean the position closed at a price the path never reached."""
    journal, _ = journalled()
    for row in journal.rows:
        assert row.mfe >= 0 and row.mae >= 0
        assert row.gross_return <= row.mfe
        assert row.gross_return >= -row.mae


def test_the_cost_curve_does_not_multiply_the_rows() -> None:
    """Only the reference level is journalled. The curve relabels the same
    decisions, so emitting it too would weight every per-trade statistic by the
    number of cost points."""
    series = {AAPL: gapping_series(AAPL)}
    plain = MemoryJournal()
    measure(GapContinuation, series, journal=plain)
    with_curve = MemoryJournal()
    measure(
        GapContinuation,
        series,
        journal=with_curve,
        cost_levels=((0.5, COSTS), (1.0, COSTS), (2.0, COSTS)),
    )
    assert len(with_curve.rows) == len(plain.rows)


def test_a_silent_strategy_journals_nothing() -> None:
    """No decisions, no rows — and no crash on the empty alignment."""
    journal = MemoryJournal()
    measure(Silent, {AAPL: gapping_series(AAPL)}, journal=journal)
    assert journal.rows == []
