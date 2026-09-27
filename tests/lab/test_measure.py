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
) -> tuple[Measurement, MemoryLedger]:
    """Run a measurement over a synthetic corpus, on a throwaway ledger."""
    store = MemoryLedger()
    ledger = TrialLedger(store=store, clock=SimClock(1_000), config_hash="cfg_test")
    first = min(b.ts_event for series in bars.values() for b in series)
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
