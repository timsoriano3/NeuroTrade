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

**Two seams are stubbed, on purpose.** Market session and regime arrive with the
`MarketContext` work, and feature resolution with the feature-library work. Both
are represented here by `default_context`, which grants nothing: `Regime.UNKNOWN`
and no feature values.

The consequence is worth stating plainly, because it looks like a bug the first
time a run comes back empty: `Strategy.regimes` defaults to `()`, and
`Regime.UNKNOWN` "grants nothing" (§5.7), so **under `default_context` a real
strategy does not fire at all**. That is the safe direction to fail — a strategy
running against an unclassified tape is precisely what the regime gate exists to
prevent — and it is why the only things that fire here today are test doubles
that opt in by declaring `regimes = (Regime.UNKNOWN,)`. Wiring the classifier is
what makes the arsenal live.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from neurotrade.bus import EventBus
from neurotrade.core.clock import Nanos, SimClock
from neurotrade.core.events import Bar, Event, MarketSession
from neurotrade.core.intent import Intent
from neurotrade.lab.drive import RunDigest, RunResult, drive
from neurotrade.lab.feed import CorpusFeed
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

__all__ = ["BacktestEngine", "BacktestResult", "ContextFactory", "default_context"]

#: Builds the view of the world a strategy is given for one bar. The seam the
#: `MarketContext` and feature-library work replace; see the module docstring.
ContextFactory = Callable[[Bar], StrategyContext]


def default_context(bar: Bar) -> StrategyContext:
    """The context used until session, regime and features are wired in.

    Grants nothing: no features, and `Regime.UNKNOWN`, which §5.7 defines as
    permitting no strategy family. `MarketSession.REGULAR` is the one claim it
    does make, and it is safe only because the corpus is fetched with
    `useRTH=1` — every bar in it really is a regular-session bar.

    Args:
        bar: The bar being evaluated.

    Returns:
        A context scoped to that bar's instrument and close time.

    Example:
        >>> from neurotrade.lab.engine import default_context
        >>> ctx = default_context(a_bar)
        >>> (ctx.symbol.ticker, ctx.as_of, ctx.regime.value)
        ('AAPL', 1000, 'UNKNOWN')
    """
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
    context_for: ContextFactory = default_context
    bus: EventBus = field(default_factory=EventBus)

    _digest: RunDigest = field(default_factory=RunDigest, init=False, repr=False)
    _strategies: list[Strategy] = field(default_factory=list, init=False, repr=False)
    _intents: list[Intent] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # Digest first, so an input is folded in before whatever it causes; then
        # the intent collector, which must see every intent any strategy emits
        # regardless of which strategy emitted it.
        self.bus.subscribe(Event, self._digest)
        self.bus.subscribe(Intent, self._collect)

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
        self._strategies.append(strategy)
        self.bus.subscribe(Bar, self._handler_for(strategy))

    @property
    def strategies(self) -> tuple[str, ...]:
        """Registered strategy names, in registration order."""
        return tuple(strategy.name for strategy in self._strategies)

    def run(self, start: Nanos, end: Nanos) -> BacktestResult:
        """Run the backtest over a half-open range.

        Args:
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            The run record plus every intent proposed, in bus order.

        Raises:
            ValueError: If the corpus yields bars out of order — `SimClock`
                will not move backwards.
            HandlerFailed: If a strategy raises. The run stops rather than
                reporting a digest for a session that never finished.
        """
        result = drive(
            self.feed.events(start, end),
            clock=self.clock,
            bus=self.bus,
            digest=self._digest,
        )
        return BacktestResult(run=result, intents=tuple(self._intents))

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
            context = self.context_for(event)
            if not strategy.is_eligible(context.regime):
                return
            self._publish(strategy.on_bar(event, context))

        return handle

    def _publish(self, intents: Sequence[Intent]) -> None:
        """Put a strategy's proposals on the bus.

        Published rather than appended directly: the bus is what the digest
        observes, so an intent that skipped it would be invisible to the proof
        that this run behaved the way the last one did.
        """
        for intent in intents:
            self.bus.publish(intent)
