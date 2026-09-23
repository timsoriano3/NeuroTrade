"""Running strategies over the corpus.

The backtest half of the lab. It is deliberately the *same* machinery as a
replay — same bus, same `SimClock`, same digest, same drive loop — differing
only in where events come from: the corpus rather than a recorded event log.
That is §3.6's one-implementation rule applied to the runner itself. A strategy
cannot tell which of the two is driving it, which is precisely why a backtest
result means something about live behaviour.

**Intents are published, not returned.** A strategy's proposals go back onto the
bus, so they land in the digest alongside the bars that caused them. Two
consequences fall out for free: a strategy that changes its mind changes the run
digest even though the input data is identical, and the collected intents are
guaranteed to be exactly what the bus saw rather than a parallel bookkeeping
that could drift from it.

**No fills happen here.** An intent is a proposal (§5.1), and what becomes of it
is `lab/labelling.py`'s triple barrier, not a fill simulator. Keeping the engine
out of the execution business means there is only one place that decides how a
position resolves, and it is the place the labels already come from.

**The view a strategy gets is assembled by a `ContextSource`.** Venue phase,
regime and resolved features come from `strategies/context.py`, per strategy,
out of the venue calendar and the feature library. The engine's default is
`NullContext`, which grants nothing and needs neither — what a replay or a
wiring test runs under.

The consequence is worth stating plainly, because it looks like a bug the first
time a run comes back empty: `Strategy.regimes` defaults to `()`, and
`Regime.UNKNOWN` "grants nothing" (§5.7), so **under `NullContext` a real
strategy does not fire at all**. That is the safe direction to fail.

**`ungated` is how Phase 2 measures a strategy before the classifier exists.**
§5.7's HMM lands in Phase 5. Until it does, every regime a real classifier would
name is `UNKNOWN`, and nothing that declares a real one would ever fire. Setting
`ungated` treats `UNKNOWN` — and only `UNKNOWN` — as permissive, so a strategy
can be measured now. A regime that *was* classified still gates normally, which
is what keeps §5.7's midday liquidity lull a no-trade window even in a research
run. The result carries `regime_gated` so that an ungated number cannot be read
six weeks later as though the gate had been on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from neurotrade.bus import EventBus
from neurotrade.core.clock import Nanos, SimClock
from neurotrade.core.events import Bar, Event, MarketSession
from neurotrade.core.intent import Intent
from neurotrade.lab.drive import RunDigest, RunResult, drive
from neurotrade.lab.feed import CorpusFeed
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

__all__ = ["BacktestEngine", "BacktestResult", "ContextSource", "NullContext"]


class ContextSource(Protocol):
    """Supplies each strategy's view of a bar.

    Three calls, in this order over a run: `declare` once per strategy before
    the first bar, `observe` once per bar, and `__call__` once per strategy per
    bar. Separating observation from projection is what lets the feature work be
    done once per bar while each strategy still sees only what it declared.
    """

    def declare(self, strategy: Strategy) -> None:
        """Register a strategy's declared dependencies, before the run starts."""
        ...

    def observe(self, bar: Bar) -> None:
        """Advance whatever history this source keeps. Once per bar."""
        ...

    def __call__(self, bar: Bar, strategy: Strategy) -> StrategyContext:
        """Build one strategy's view of the bar just observed."""
        ...


class NullContext:
    """A context source that grants nothing, for runs with nothing wired in.

    `Regime.UNKNOWN` permits no strategy family (§5.7) and no feature is
    resolved, so a strategy running under this one cannot act on anything it did
    not already have. `MarketSession.REGULAR` is the single claim it does make,
    and it is safe only because the corpus is fetched with `useRTH=1` — every
    bar in it really is a regular-session bar.

    Example:
        >>> class Quiet(Strategy):
        ...     name, version = "quiet", "1.0.0"
        >>> view = NullContext()(a_bar, Quiet())
        >>> (view.symbol.ticker, view.as_of, view.regime.value)
        ('AAPL', 1000, 'UNKNOWN')
    """

    __slots__ = ()

    def declare(self, strategy: Strategy) -> None:
        """Accept any declaration. Nothing is resolved, so nothing is needed."""

    def observe(self, bar: Bar) -> None:
        """Keep no history."""

    def __call__(self, bar: Bar, strategy: Strategy) -> StrategyContext:
        """The bar's instrument and close time, and no permissions.

        Args:
            bar: The bar being evaluated.
            strategy: Ignored; every strategy gets the same empty view.

        Returns:
            A context scoped to that bar's instrument and close time.
        """
        del strategy
        return StrategyContext(
            symbol=bar.symbol,
            # The bar's close, never its open: features may only read what had
            # finished happening. Stamping `as_of` earlier is the lookahead bug.
            as_of=bar.ts_event,
            session=MarketSession.REGULAR,
            regime=Regime.UNKNOWN,
            values={},
        )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """What a backtest produced: the run, plus the proposals it made.

    `run` is the same record a replay returns, so the two are directly
    comparable — which is the check that a backtest and a replay of the same
    session really did behave identically.
    """

    run: RunResult  # digest, counts and span, exactly as a replay reports them
    intents: tuple[Intent, ...]  # every proposal, in the order the bus saw them
    regime_gated: bool = True  # False when UNKNOWN was treated as permissive

    @property
    def digest(self) -> str:
        """The run digest — covers the intents as well as the bars."""
        return self.run.digest

    @property
    def is_empty(self) -> bool:
        """Whether the backtest found any bars to run over."""
        return self.run.is_empty

    def by_strategy(self, name: str) -> tuple[Intent, ...]:
        """Proposals from one strategy, by name.

        Args:
            name: The strategy's declared `name`.

        Returns:
            Its intents, in bus order. Empty if it proposed nothing.
        """
        return tuple(intent for intent in self.intents if intent.strategy == name)


@dataclass(eq=False)
class BacktestEngine:
    """Drives a universe's corpus bars through strategies under a sim clock.

    Strategies are invoked in **registration order**, which is what makes a
    multi-strategy run reproducible: two strategies reacting to the same bar
    both publish intents, and the order those enter the digest has to be a
    property of the configuration rather than of dict iteration.

    Example:
        >>> from neurotrade.core.events import BarInterval
        >>> from neurotrade.lab.engine import BacktestEngine
        >>> from neurotrade.lab.feed import CorpusFeed
        >>> class OneBarStore:
        ...     def write_bars(self, bars, *, source, session_date): ...
        ...     def read_bars(self, symbol, interval, start, end):
        ...         return iter([a_bar])
        >>> feed = CorpusFeed(OneBarStore(), (AAPL,), BarInterval.MIN_1)
        >>> result = BacktestEngine(feed).run(0, 10_000)
        >>> (result.run.events_read, result.intents)
        (1, ())
    """

    feed: CorpusFeed  # where bars come from
    clock: SimClock = field(default_factory=lambda: SimClock(0))
    context: ContextSource = field(default_factory=NullContext)
    bus: EventBus = field(default_factory=EventBus)
    ungated: bool = False  # research only: let UNKNOWN permit what it otherwise blocks

    _digest: RunDigest = field(default_factory=RunDigest, init=False, repr=False)
    _strategies: list[Strategy] = field(default_factory=list, init=False, repr=False)
    _intents: list[Intent] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # Digest first, so an input is folded in before whatever it causes; then
        # the intent collector, which must see every intent any strategy emits
        # regardless of which strategy emitted it.
        self.bus.subscribe(Event, self._digest)
        self.bus.subscribe(Intent, self._collect)
        # Ahead of every strategy, and independent of them: the feature history
        # is a property of the data the run saw, not of who was eligible to act
        # on it. A bar skipped here would leave a hole in a window nothing
        # downstream could detect.
        self.bus.subscribe(Bar, self._observe)

    def add_strategy(self, strategy: Strategy) -> None:
        """Register a strategy and subscribe it to bars.

        Args:
            strategy: An instance, already configured. Registration order is
                the order it will be invoked in, and therefore part of the run
                digest.

        Example:
            >>> from neurotrade.core.events import BarInterval
            >>> from neurotrade.lab.engine import BacktestEngine
            >>> from neurotrade.lab.feed import CorpusFeed
            >>> class Quiet(Strategy):
            ...     name, version = "quiet", "1.0.0"
            ...     def on_bar(self, bar, context): return ()
            >>> class EmptyStore:
            ...     def write_bars(self, bars, *, source, session_date): ...
            ...     def read_bars(self, symbol, interval, start, end):
            ...         return iter(())
            >>> engine = BacktestEngine(
            ...     CorpusFeed(EmptyStore(), (AAPL,), BarInterval.MIN_1)
            ... )
            >>> engine.add_strategy(Quiet())
            >>> engine.strategies
            ('quiet',)
        """
        self.context.declare(strategy)
        self._strategies.append(strategy)
        self.bus.subscribe(Bar, self._handler_for(strategy))

    @property
    def strategies(self) -> tuple[str, ...]:
        """Registered strategy names, in registration order."""
        return tuple(strategy.name for strategy in self._strategies)

    def run(self, start: Nanos, end: Nanos, *, warmup_ns: Nanos = 0) -> BacktestResult:
        """Run the backtest over a half-open range.

        Args:
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.
            warmup_ns: Span before `start` to read into the context's feature
                history without dispatching it. Zero leaves every feature cold
                for its first `lookback` bars, which silences exactly the
                open-of-session strategies of §5.2; `MarketContext.lookback`
                says how many bars are needed, and the span is nanoseconds
                because bars are not evenly spaced across a weekend.

        Returns:
            The run record plus every intent proposed, in bus order.

        Raises:
            ValueError: If the corpus yields bars out of order — `SimClock`
                will not move backwards.
            HandlerFailed: If a strategy raises. The run stops rather than
                reporting a digest for a session that never finished.
        """
        if warmup_ns:
            # Straight into the context, never onto the bus: a warm-up bar is
            # history, not an event the run saw. Keeping it out of the digest is
            # what lets two runs over the same range stay comparable while one
            # of them primed further back.
            for bar in self.feed.events(max(0, start - warmup_ns), start):
                self.context.observe(bar)
        result = drive(
            self.feed.events(start, end),
            clock=self.clock,
            bus=self.bus,
            digest=self._digest,
        )
        return BacktestResult(
            run=result, intents=tuple(self._intents), regime_gated=not self.ungated
        )

    def _observe(self, event: Event) -> None:
        """Advance the context's history. Subscribed to `Bar`, so this holds."""
        assert isinstance(event, Bar)
        self.context.observe(event)

    def _may_fire(self, strategy: Strategy, regime: Regime) -> bool:
        """Whether the host lets this strategy act in this regime.

        The un-gate relaxes `UNKNOWN` alone. A regime that was actually
        classified — the midday lull, and from Phase 5 the rest — still gates
        normally, so a research run cannot quietly trade a window the spec calls
        a default no-trade window.
        """
        return strategy.is_eligible(regime) or (self.ungated and regime is Regime.UNKNOWN)

    def _collect(self, event: Event) -> None:
        """Record one intent. Subscribed to `Intent`, so the narrowing holds."""
        assert isinstance(event, Intent)
        self._intents.append(event)

    def _handler_for(self, strategy: Strategy) -> Callable[[Event], None]:
        """Bind one strategy into a bus handler.

        Closes over the strategy rather than looking it up per event, so the
        dispatch cost does not grow with the number registered.
        """

        def handle(event: Event) -> None:
            assert isinstance(event, Bar)
            view = self.context(event, strategy)
            if not self._may_fire(strategy, view.regime):
                return
            self._publish(strategy.on_bar(event, view))

        return handle

    def _publish(self, intents: Sequence[Intent]) -> None:
        """Put a strategy's proposals on the bus.

        Published rather than appended directly: the bus is what the digest
        observes, so an intent that skipped it would be invisible to the proof
        that this run behaved the way the last one did.
        """
        for intent in intents:
            self.bus.publish(intent)
