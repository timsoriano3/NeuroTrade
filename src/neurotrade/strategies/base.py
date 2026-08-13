"""The contract every strategy plugin implements.

A strategy answers one question: *given what is knowable right now, is there a
trade here, which way, and where would I be wrong?* It does not answer *how
much* — that belongs to the risk engine, working from the model's calibrated
probability (§7.3). Keeping those apart is what keeps direction explainable
while confining ML to the problem where it demonstrably helps.

**A strategy never sees raw history.** It receives a `StrategyContext` holding
feature values already resolved and point-in-time checked by the engine. This is
deliberate: hand a strategy the bar series and sooner or later one of them
indexes past the current bar, and the resulting backtest is worthless in a way
nobody notices for months. Withholding the series makes that impossible rather
than merely discouraged.

**Declared feature dependencies are load-bearing.** `features` is not
documentation — the engine resolves exactly those, and `StrategyContext.feature`
refuses any name a strategy did not declare. A strategy therefore cannot quietly
start depending on something the engine is not loading history for.

**Regime gating is a permission, not a preference.** §5.7 has the regime
classifier decide which strategy families may fire at all. An ORB strategy and
an ORB-fade strategy are opposites and must never be live together; declaring
`regimes` lets the host enforce that centrally rather than each strategy
remembering to check.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, MarketSession, Quote, TickTrade
from neurotrade.core.intent import Intent
from neurotrade.core.registry import Registry
from neurotrade.core.types import Symbol

__all__ = [
    "FeatureRef",
    "Regime",
    "Strategy",
    "StrategyContext",
    "StrategyRegistry",
    "UndeclaredFeature",
]


class Regime(StrEnum):
    """Market state, as classified by §5.7's regime model.

    Determines which strategy families may fire. Trend days, chop days and
    reversal days reward different tools, and running a breakout strategy
    through a chop day is not a small inefficiency — it is the mechanism by
    which a strategy with a real edge still produces a losing month.

    Phase 5 fits the classifier that produces these. Until then everything is
    `UNKNOWN`, which grants nothing.

    Example:
        >>> Regime.TREND_UP.value
        'TREND_UP'
    """

    TREND_UP = "TREND_UP"  # sustained directional move higher
    TREND_DOWN = "TREND_DOWN"  # sustained directional move lower
    CHOP = "CHOP"  # range-bound; breakouts fail
    REVERSAL = "REVERSAL"  # direction turning; continuation fails
    HIGH_VOLATILITY = "HIGH_VOLATILITY"  # wide ranges, gappy fills
    LIQUIDITY_LULL = "LIQUIDITY_LULL"  # the 12:00-14:00 ET lull; default no-trade
    UNKNOWN = "UNKNOWN"  # not yet classified; grants nothing


class UndeclaredFeature(KeyError):
    """Raised when a strategy reads a feature it did not declare.

    Always a bug in the strategy. The engine loads history only for declared
    features, so an undeclared one would be missing, stale, or computed over the
    wrong window — none of which is detectable from the value itself.
    """


@dataclass(frozen=True, slots=True)
class FeatureRef:
    """A strategy's declared dependency on one feature.

    Not orderable by field. The generated comparison would compare
    `(name, version)` tuples, and an unpinned ref holds `None` where a pinned
    one holds a string — so sorting a mixed set raises `TypeError`. That mix is
    the normal case, not an edge case: §10.2 runs a champion pinned to an exact
    feature version beside a challenger tracking latest. Sort by `str` instead.

    Example:
        >>> sorted([FeatureRef("rvol"), FeatureRef("atr", "1.2.0")], key=str)
        [FeatureRef(name='atr', version='1.2.0'), FeatureRef(name='rvol', version=None)]
    """

    name: str  # registered feature name
    version: str | None = None  # exact version; None means latest at load time

    def __str__(self) -> str:
        return self.name if self.version is None else f"{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy is allowed to know at one moment.

    Deliberately narrow. There is no clock, no bar history and no broker here.
    A strategy able to reach any of those could look ahead, read wall-clock time
    or place an order directly, and each of those breaks a guarantee the rest of
    the system depends on.

    Example:
        >>> ctx = StrategyContext(
        ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
        ...     regime=Regime.TREND_UP, values={"atr": 1.25},
        ... )
        >>> ctx.feature("atr")
        1.25
    """

    symbol: Symbol  # the instrument being evaluated
    as_of: Nanos  # the moment being modelled; features are valid up to here
    session: MarketSession  # venue phase — pre, regular, post, closed
    regime: Regime  # current classification, gating which strategies may fire
    values: Mapping[str, float | None] = field(default_factory=dict)
    """Resolved feature values by name. `None` means still warming up — not
    zero, and not "no signal"."""

    def feature(self, name: str) -> float | None:
        """Read a declared feature.

        Args:
            name: Feature name, as declared in the strategy's `features`.

        Returns:
            The value, or `None` if the feature has not warmed up. A `None` must
            not be treated as zero; a strategy needing a value should decline to
            fire instead.

        Raises:
            UndeclaredFeature: If the strategy did not declare this feature.

        Example:
            >>> ctx = StrategyContext(
            ...     symbol=AAPL, as_of=0, session=MarketSession.REGULAR,
            ...     regime=Regime.CHOP, values={"rvol": None},
            ... )
            >>> ctx.feature("rvol") is None        # declared, still warming up
            True
        """
        try:
            return self.values[name]
        except KeyError:
            raise UndeclaredFeature(
                f"{name!r} was not declared by this strategy; available: {sorted(self.values)}"
            ) from None

    def requires(self, *names: str) -> bool:
        """True when every named feature has a value.

        The usual guard at the top of a handler: a strategy needing three
        features cannot act until all three are warm.

        Example:
            >>> ctx = StrategyContext(
            ...     symbol=AAPL, as_of=0, session=MarketSession.REGULAR,
            ...     regime=Regime.CHOP, values={"a": 1.0, "b": None},
            ... )
            >>> (ctx.requires("a"), ctx.requires("a", "b"))
            (True, False)
        """
        return all(self.feature(name) is not None for name in names)


class Strategy(ABC):
    """Base class for strategy plugins (§5.1).

    Subclasses declare metadata as class attributes and override whichever
    handlers they need. Every handler defaults to producing nothing, so a
    bar-driven strategy need not stub quote and trade handlers it will never use.

    Example:
        >>> class Breakout(Strategy):
        ...     name, version = "breakout", "1.0.0"
        ...     regimes = (Regime.TREND_UP,)
        ...     features = (FeatureRef("atr"),)
        >>> Breakout().is_eligible(Regime.CHOP)
        False
    """

    name: ClassVar[str]
    """Registered name. Appears on every Intent and every trade record."""

    version: ClassVar[str]
    """`MAJOR.MINOR.PATCH`. Bump on any change to what the strategy decides — a
    retune that keeps the old version makes the journal's history a lie."""

    regimes: ClassVar[tuple[Regime, ...]] = ()
    """Regimes in which this strategy may fire. Empty means never, the safe
    default for a subclass that forgets to declare."""

    features: ClassVar[tuple[FeatureRef, ...]] = ()
    """Declared feature dependencies. The engine resolves exactly these."""

    cost_sensitivity: ClassVar[float] = 1.0
    """Multiple of modelled cost this strategy's edge must clear to be worth
    trading (§5.9). A scalper needs a higher bar than a multi-hour hold, because
    it pays the spread far more often for the same gross move."""

    def is_eligible(self, regime: Regime) -> bool:
        """Whether this strategy may fire in the given regime.

        Checked by the host before handlers are called, so an ineligible
        strategy never sees the event — stronger than each strategy remembering
        to check, and what keeps an ORB strategy and an ORB-fade strategy from
        ever being live at the same time.

        Example:
            >>> class Fade(Strategy):
            ...     name, version = "fade", "1.0.0"
            ...     regimes = (Regime.CHOP, Regime.REVERSAL)
            >>> (Fade().is_eligible(Regime.CHOP), Fade().is_eligible(Regime.TREND_UP))
            (True, False)
        """
        return regime in self.regimes

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Handle a completed bar.

        Args:
            bar: The bar that just closed. Its `ts_event` is the close time, so
                acting on it is not lookahead.
            context: Resolved features and market state as of that instant.

        Returns:
            Zero or more proposals. Returning nothing is the common case and
            carries no penalty — a strategy that fires on every bar is not a
            strategy.
        """
        return ()

    def on_quote(self, quote: Quote, context: StrategyContext) -> Sequence[Intent]:
        """Handle a top-of-book update.

        §5.1 calls this `on_book`; it is named for the event type it actually
        receives. Full depth arrives with the microstructure layer in Phase 4.
        """
        return ()

    def on_trade(self, trade: TickTrade, context: StrategyContext) -> Sequence[Intent]:
        """Handle a trade print from the tape."""
        return ()

    @property
    def qualified_name(self) -> str:
        """`name@version`, as recorded on every Intent this strategy produces.

        Example:
            >>> class Orb(Strategy):
            ...     name, version = "orb", "1.0.0"
            >>> Orb().qualified_name
            'orb@1.0.0'
        """
        return f"{self.name}@{self.version}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.qualified_name}>"


class StrategyRegistry:
    """The strategy arsenal.

    Holds classes rather than instances, so the host controls construction and
    can build a fresh strategy per symbol — shared instances would share state
    across instruments, which is a subtle and expensive bug.

    Example:
        >>> registry = StrategyRegistry()
        >>> @registry.strategy
        ... class Orb(Strategy):
        ...     name, version = "orb", "1.0.0"
        ...     regimes = (Regime.TREND_UP,)
        >>> registry.get("orb") is Orb
        True
    """

    __slots__ = ("_registry",)

    def __init__(self) -> None:
        self._registry: Registry[type[Strategy]] = Registry("strategy")

    def strategy(self, cls: type[Strategy]) -> type[Strategy]:
        """Register a strategy class, taking name and version from the class.

        Used bare as `@registry.strategy`. Reading identity off the class rather
        than from decorator arguments means a strategy cannot be registered
        under a name different from the one it stamps onto its own Intents.

        Args:
            cls: The strategy class.

        Returns:
            The class unchanged, so it stays importable and testable directly.

        Raises:
            ValueError: If the class does not declare `name` and `version`.
            DuplicateRegistration: If that name and version are already taken.

        Example:
            >>> registry = StrategyRegistry()
            >>> @registry.strategy
            ... class Vwap(Strategy):
            ...     name, version = "vwap_reclaim", "2.0.0"
            >>> registry.snapshot()
            ('vwap_reclaim@2.0.0',)
        """
        for attribute in ("name", "version"):
            if not getattr(cls, attribute, None):
                raise ValueError(f"{cls.__name__} must declare a {attribute}")
        self._registry.add(cls.name, cls.version, cls)
        return cls

    def get(self, name: str, version: str | None = None) -> type[Strategy]:
        """Look up a strategy class, latest version when unspecified.

        Raises:
            KeyError: If the strategy, or that version of it, is not registered.
        """
        return self._registry.get(name, version)

    def eligible(self, regime: Regime) -> tuple[type[Strategy], ...]:
        """Strategy classes permitted to fire in a regime.

        Example:
            >>> registry = StrategyRegistry()
            >>> @registry.strategy
            ... class Breakout(Strategy):
            ...     name, version = "breakout", "1.0.0"
            ...     regimes = (Regime.TREND_UP,)
            >>> @registry.strategy
            ... class Fade(Strategy):
            ...     name, version = "fade", "1.0.0"
            ...     regimes = (Regime.CHOP,)
            >>> [cls.name for cls in registry.eligible(Regime.CHOP)]
            ['fade']
        """
        return tuple(cls for _, _, cls in self._registry if regime in cls.regimes)

    def required_features(self) -> tuple[FeatureRef, ...]:
        """Every feature any registered strategy depends on, deduplicated.

        The engine uses this to decide what to compute. Computing the whole
        library when four strategies need three features is wasted work on every
        bar of every backtest, and backtests are run in the millions.

        Example:
            >>> registry = StrategyRegistry()
            >>> @registry.strategy
            ... class A(Strategy):
            ...     name, version = "a", "1.0.0"
            ...     features = (FeatureRef("atr"), FeatureRef("rvol"))
            >>> @registry.strategy
            ... class B(Strategy):
            ...     name, version = "b", "1.0.0"
            ...     features = (FeatureRef("atr"),)
            >>> [str(ref) for ref in registry.required_features()]
            ['atr', 'rvol']
        """
        return tuple(sorted({ref for _, _, cls in self._registry for ref in cls.features}, key=str))

    def snapshot(self) -> tuple[str, ...]:
        """Deterministic record of the arsenal, for the config hash."""
        return self._registry.snapshot()

    def freeze(self) -> None:
        """Close registration once discovery is done."""
        self._registry.freeze()

    def names(self) -> tuple[str, ...]:
        """Every registered strategy name, sorted."""
        return self._registry.names()

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: object) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"StrategyRegistry({len(self._registry)} registered)"
