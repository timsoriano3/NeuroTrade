"""The backtest engine: strategies over the corpus, under a simulated clock.

The properties that matter here are the ones a profitable-looking backtest can
quietly violate: that the run is reproducible, that a strategy's output is part
of what gets hashed, and that the regime gate is enforced by the host rather
than trusted to each strategy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date
from decimal import Decimal
from typing import ClassVar

import pytest

from neurotrade.bus import HandlerFailed
from neurotrade.core.clock import SimClock
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.lab.engine import BacktestEngine, BacktestResult, default_context
from neurotrade.lab.feed import CorpusFeed
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)


def bar(symbol: Symbol, ts: int) -> Bar:
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=Price("100"),
        high=Price("101"),
        low=Price("99"),
        close=Price("100.5"),
        volume=Quantity(1_000),
    )


class ListStore:
    """A `StoragePort` over a flat list of bars, filtered per symbol."""

    def __init__(self, bars: Sequence[Bar]) -> None:
        self._bars = bars

    def write_bars(self, bars: Sequence[Bar], *, source: str, session_date: date) -> None:
        raise AssertionError("the engine must never write")

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: int, end: int
    ) -> Iterator[Bar]:
        return iter([b for b in self._bars if b.symbol == symbol and start <= b.ts_event < end])


class Proposer(Strategy):
    """Fires on every bar. Declares UNKNOWN so it runs under `default_context`."""

    name, version = "proposer", "1.0.0"
    regimes: ClassVar[tuple[Regime, ...]] = (Regime.UNKNOWN,)
    rationale = "test double"

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        return (
            Intent(
                symbol=bar.symbol,
                ts_event=bar.ts_event,
                ts_init=bar.ts_event,
                side=Side.BUY,
                entry=EntryTrigger.LIMIT,
                entry_price=Price("100"),
                invalidation=Price("99"),
                target_r=Decimal(2),
                horizon_ns=3_600_000_000_000,
                strategy=self.name,
                strategy_version=self.version,
                rationale=self.rationale,
            ),
        )


class Other(Proposer):
    name, version = "other", "1.0.0"


class Quiet(Strategy):
    name, version = "quiet", "1.0.0"
    regimes: ClassVar[tuple[Regime, ...]] = (Regime.UNKNOWN,)

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        return ()


class Undeclared(Proposer):
    """Declares no regime — the safe default. Must never fire."""

    name, version = "undeclared", "1.0.0"
    regimes = ()


def engine_over(bars: Sequence[Bar], symbols: tuple[Symbol, ...]) -> BacktestEngine:
    return BacktestEngine(CorpusFeed(ListStore(bars), symbols, BarInterval.MIN_1))


# ── The default context ──────────────────────────────────────


def test_default_context_grants_no_regime() -> None:
    """§5.7: UNKNOWN grants nothing, so an undeclared strategy cannot fire."""
    assert default_context(bar(AAPL, 10)).regime is Regime.UNKNOWN


def test_default_context_is_stamped_at_the_bar_close() -> None:
    """`as_of` at the open would be the classic lookahead bug."""
    assert default_context(bar(AAPL, 4_200)).as_of == 4_200


def test_default_context_exposes_no_features() -> None:
    assert default_context(bar(AAPL, 10)).values == {}


# ── Intent collection ────────────────────────────────────────


def test_intents_are_collected_in_bus_order() -> None:
    engine = engine_over([bar(AAPL, 10), bar(AAPL, 20)], (AAPL,))
    engine.add_strategy(Proposer())
    result = engine.run(0, 100)
    assert [i.ts_event for i in result.intents] == [10, 20]


def test_a_quiet_strategy_proposes_nothing() -> None:
    engine = engine_over([bar(AAPL, 10)], (AAPL,))
    engine.add_strategy(Quiet())
    assert engine.run(0, 100).intents == ()


def test_by_strategy_filters_by_name() -> None:
    engine = engine_over([bar(AAPL, 10)], (AAPL,))
    engine.add_strategy(Proposer())
    engine.add_strategy(Other())
    result = engine.run(0, 100)
    assert len(result.intents) == 2
    assert [i.strategy for i in result.by_strategy("other")] == ["other"]


def test_by_strategy_of_an_unregistered_name_is_empty() -> None:
    engine = engine_over([bar(AAPL, 10)], (AAPL,))
    engine.add_strategy(Proposer())
    assert engine.run(0, 100).by_strategy("nope") == ()


# ── The regime gate ──────────────────────────────────────────


def test_an_undeclared_strategy_never_fires() -> None:
    """`regimes = ()` is the safe default and the host must enforce it."""
    engine = engine_over([bar(AAPL, 10)], (AAPL,))
    engine.add_strategy(Undeclared())
    assert engine.run(0, 100).intents == ()


def test_the_gate_is_enforced_by_the_host_not_the_strategy() -> None:
    """An ineligible strategy must never see the bar at all."""
    seen: list[Bar] = []

    class Nosy(Undeclared):
        name = "nosy"

        def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
            seen.append(bar)
            return ()

    engine = engine_over([bar(AAPL, 10)], (AAPL,))
    engine.add_strategy(Nosy())
    engine.run(0, 100)
    assert seen == []


# ── Registration order ───────────────────────────────────────


def test_strategies_report_in_registration_order() -> None:
    engine = engine_over([], (AAPL,))
    engine.add_strategy(Other())
    engine.add_strategy(Proposer())
    assert engine.strategies == ("other", "proposer")


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (Proposer, Other, ["proposer", "other"]),
        (Other, Proposer, ["other", "proposer"]),
    ],
    ids=["proposer-first", "other-first"],
)
def test_registration_order_drives_intent_order(
    first: type[Strategy], second: type[Strategy], expected: list[str]
) -> None:
    """Two strategies on one bar: the order must be configuration, not chance."""
    engine = engine_over([bar(AAPL, 10)], (AAPL,))
    engine.add_strategy(first())
    engine.add_strategy(second())
    assert [i.strategy for i in engine.run(0, 100).intents] == expected


# ── The digest ───────────────────────────────────────────────


def test_two_identical_runs_agree() -> None:
    """The same property G1 proves for replay, for the backtest engine."""
    bars = [bar(AAPL, 10), bar(MSFT, 10), bar(AAPL, 20)]

    def once() -> str:
        engine = engine_over(bars, (AAPL, MSFT))
        engine.add_strategy(Proposer())
        return engine.run(0, 100).digest

    assert once() == once()


def test_the_digest_covers_intents_not_only_bars() -> None:
    """Same input bars, different decisions — the digest must move."""
    bars = [bar(AAPL, 10)]

    quiet = engine_over(bars, (AAPL,))
    quiet.add_strategy(Quiet())

    loud = engine_over(bars, (AAPL,))
    loud.add_strategy(Proposer())

    assert quiet.run(0, 100).digest != loud.run(0, 100).digest


def test_a_changed_rationale_changes_the_digest() -> None:
    """The hash reaches the whole intent, not just the fact that one occurred."""
    bars = [bar(AAPL, 10)]

    class Reworded(Proposer):
        rationale = "a different reason"

    plain = engine_over(bars, (AAPL,))
    plain.add_strategy(Proposer())

    reworded = engine_over(bars, (AAPL,))
    reworded.add_strategy(Reworded())

    assert plain.run(0, 100).digest != reworded.run(0, 100).digest


def test_dispatched_counts_intents_as_well_as_bars() -> None:
    engine = engine_over([bar(AAPL, 10), bar(AAPL, 20)], (AAPL,))
    engine.add_strategy(Proposer())
    result = engine.run(0, 100)
    assert (result.run.events_read, result.run.events_dispatched) == (2, 4)


# ── The clock ────────────────────────────────────────────────


def test_the_clock_leads_each_bar() -> None:
    """A strategy reading the clock sees the modelled moment, not the wall."""
    clock = SimClock(0)
    seen: list[int] = []

    class ClockWatcher(Quiet):
        name = "watcher"

        def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
            seen.append(clock.now_ns())
            return ()

    engine = BacktestEngine(
        CorpusFeed(ListStore([bar(AAPL, 10), bar(AAPL, 20)]), (AAPL,), BarInterval.MIN_1),
        clock=clock,
    )
    engine.add_strategy(ClockWatcher())
    engine.run(0, 100)
    assert seen == [10, 20]


def test_a_backwards_corpus_fails_loudly() -> None:
    """SimClock will not move backwards; a mis-ordered source must not pass."""

    class BackwardsStore(ListStore):
        def read_bars(
            self, symbol: Symbol, interval: BarInterval, start: int, end: int
        ) -> Iterator[Bar]:
            return iter([bar(AAPL, 20), bar(AAPL, 10)])

    engine = BacktestEngine(CorpusFeed(BackwardsStore([]), (AAPL,), BarInterval.MIN_1))
    with pytest.raises(ValueError, match="backwards"):
        engine.run(0, 100)


# ── Failure ──────────────────────────────────────────────────


def test_a_raising_strategy_stops_the_run() -> None:
    """A partial run must not report a digest for a session that never finished."""

    class Broken(Quiet):
        name = "broken"

        def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
            raise RuntimeError("bad feature")

    engine = engine_over([bar(AAPL, 10)], (AAPL,))
    engine.add_strategy(Broken())
    with pytest.raises(HandlerFailed):
        engine.run(0, 100)


# ── Empty runs ───────────────────────────────────────────────


def test_an_empty_corpus_runs_clean() -> None:
    engine = engine_over([], (AAPL,))
    engine.add_strategy(Proposer())
    result = engine.run(0, 100)
    assert result.is_empty
    assert result.intents == ()


def test_result_holds_no_timing() -> None:
    """Wall-clock duration varies for reasons unrelated to behaviour."""
    names = set(BacktestResult.__dataclass_fields__)
    assert names == {"run", "intents"}
