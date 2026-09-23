"""Deliberately constructed controls for the Phase 1 exit gate (§13).

Two strategies and two synthetic price series, built so their verdicts are known
before the lab is asked. The gate in `gate.py` runs both and fails if either
comes back wrong.

**The overfit control** searches a grid of moving-average crossovers over a
zero-drift random walk. There is no edge in that data — by construction, not by
assumption — so every variant is worthless, but the best of thirty-five posts a
respectable in-sample Sharpe ratio anyway. That number is the thing the lab has
to refuse.

**The honest control** runs a small, pre-declared set of the *same* strategy over
a series with genuine serial correlation in its returns. Same code path, same
statistics, same thresholds: only the data and the size of the search differ.
That symmetry is the point. A gate with only the overfit control is passed by a
lab that rejects everything, and `return REJECT` is not a validation harness.

Both series are generated from an explicit `random.Random(seed)`, never the
module-level RNG, so a gate run is reproducible on any machine.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.registry import Version
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.features.registry import FeatureSpec
from neurotrade.strategies.base import FeatureRef, Regime, Strategy, StrategyContext

__all__ = [
    "CONTROL_SYMBOL",
    "HONEST_GRID",
    "SNOOPED_GRID",
    "Crossover",
    "CrossoverParams",
    "intent_bars",
    "momentum_bars",
    "random_walk_bars",
    "sma_spec",
]

CONTROL_SYMBOL: Final = Symbol("CTRL", Venue.NASDAQ)
"""Not a real listing. A control must never be confusable with a corpus symbol."""

_START_PRICE: Final = 100.0  # round number; nothing in the controls depends on the level
_BAR_VOLUME: Final = Quantity(10_000)  # constant, so relative-volume effects cannot leak in
_WICK_FRACTION: Final = 0.4  # how far high/low extend past the body, as a share of the move


_PENNY: Final = Decimal("0.01")  # US equities quote in pennies; synthetic prices do too


def _price(value: float) -> Price:
    """Quantise a generated level to a tick.

    A generated series is a feed boundary, which is the one place `from_float`
    is allowed (see the precision invariant in CLAUDE.md). Rounding to the tick
    there keeps every downstream `Decimal` tidy — an unquantised float carries
    seventeen places into the cost model and the barrier arithmetic.
    """
    return Price(Price.from_float(value).value.quantize(_PENNY))


def _bar(*, index: int, open_: float, high: float, low: float, close: float) -> Bar:
    """Assemble one synthetic bar at 1-minute spacing."""
    ts = index * BarInterval.MIN_1.nanos
    return Bar(
        symbol=CONTROL_SYMBOL,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=_price(open_),
        high=_price(high),
        low=_price(low),
        close=_price(close),
        volume=_BAR_VOLUME,
    )


def _series_to_bars(closes: Sequence[float], rng: random.Random) -> tuple[Bar, ...]:
    """Wrap a close series in OHLC bars whose highs and lows bracket the body.

    The wicks are noise around the body rather than data: the controls trade on
    closes, and the barriers in `lab.labelling` read highs and lows. A series
    with no wick at all would let a stop sit exactly on a close and never trip.
    """
    bars: list[Bar] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        body = abs(close - previous)
        wick = body * _WICK_FRACTION + previous * 0.0005  # never zero, even on a flat bar
        high = max(previous, close) + wick * rng.random()
        low = min(previous, close) - wick * rng.random()
        bars.append(_bar(index=index, open_=previous, high=high, low=low, close=close))
        previous = close
    return tuple(bars)


def random_walk_bars(*, seed: int, n_bars: int, volatility: float = 0.004) -> tuple[Bar, ...]:
    """A zero-drift geometric random walk: the series with no edge in it.

    Independent, identically distributed log returns. Every pattern a search
    finds here is a coincidence, which is exactly what makes it the right data
    for an overfit control — the naive Sharpe ratio it produces is known to be
    worthless without having to argue about it.

    The log returns are **demeaned**, so the drift is zero in this sample rather
    than only in expectation. That matters more than it sounds. Any one
    realisation of a walk finishes somewhere, and a directional rule aligned
    with where it finished wins in every subsample of it — which is a stable
    ranking, so PBO reports a low number and is right to. Removing the realised
    drift leaves only the idiosyncratic luck a search can fit, which is the null
    the overfit control is supposed to pose.

    Args:
        seed: Passed to a private `random.Random`; the module-level RNG is never
            touched, so two runs at one seed agree bar for bar.
        n_bars: Length of the series.
        volatility: Standard deviation of the per-bar log return.

    Returns:
        Bars at 1-minute spacing, oldest first.

    Raises:
        ValueError: If `n_bars` is below 2 or `volatility` is not positive.

    Example:
        >>> bars = random_walk_bars(seed=7, n_bars=4)
        >>> len(bars), bars[0].symbol.ticker
        (4, 'CTRL')
    """
    if n_bars < 2:
        raise ValueError(f"n_bars must be at least 2, got {n_bars}")
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, got {volatility}")

    rng = random.Random(seed)
    returns = [rng.gauss(0.0, volatility) for _ in range(n_bars)]
    drift = math.fsum(returns) / n_bars
    level = _START_PRICE
    closes: list[float] = []
    for value in returns:
        level *= math.exp(value - drift)
        closes.append(level)
    return _series_to_bars(closes, random.Random(seed + 1))


def momentum_bars(
    *,
    seed: int,
    n_bars: int,
    volatility: float = 0.004,
    persistence: float = 0.98,
    drift_scale: float = 0.0013,
) -> tuple[Bar, ...]:
    """A series with a real, stated edge: a drift that persists for tens of bars.

    `mu[t] = persistence * mu[t-1] + eta`, and `r[t] = mu[t] + noise`. A trend
    follower genuinely makes money here, so a lab that rejects it is broken in
    the direction nobody notices — a false negative throws away a real strategy
    in silence, and files no complaint about it.

    **The persistence has to match the horizon of the rule being tested**, which
    is the trap this generator was rewritten to avoid. Autocorrelating the
    *returns* at lag 1 (`r[t] = 0.35 * r[t-1] + noise`) looks like strong
    momentum and is invisible to a moving-average crossover: correlation at lag
    `k` decays as `0.35**k`, so by the twenty-bar window the rule averages over,
    there is nothing left. Persisting the *drift* at 0.98 gives a half-life
    around thirty-four bars, which is the scale a crossover can see.

    `persistence` and `drift_scale` together are far beyond anything an equity
    market offers. That is deliberate: the positive control exists to prove the
    harness can say yes, not to be a realistic alpha.

    Args:
        seed: Private RNG seed, as `random_walk_bars`.
        n_bars: Length of the series.
        volatility: Standard deviation of the per-bar noise.
        persistence: AR(1) coefficient on the latent drift. Must sit in
            `(-1, 1)` or the drift diverges instead of mean-reverting.
        drift_scale: Stationary standard deviation of the latent drift, per bar.

    Returns:
        Bars at 1-minute spacing, oldest first.

    Raises:
        ValueError: If `n_bars` is below 2, `volatility` or `drift_scale` is not
            positive, or `persistence` is outside `(-1, 1)`.

    Example:
        >>> bars = momentum_bars(seed=7, n_bars=4)
        >>> len(bars)
        4
    """
    if n_bars < 2:
        raise ValueError(f"n_bars must be at least 2, got {n_bars}")
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, got {volatility}")
    if drift_scale <= 0:
        raise ValueError(f"drift_scale must be positive, got {drift_scale}")
    if not -1.0 < persistence < 1.0:
        raise ValueError(f"persistence must lie in (-1, 1), got {persistence}")

    rng = random.Random(seed)
    # Innovation sized so the drift's stationary standard deviation is
    # `drift_scale`: var(mu) = eta**2 / (1 - persistence**2).
    innovation = drift_scale * math.sqrt(1.0 - persistence**2)
    level = _START_PRICE
    drift = 0.0
    closes: list[float] = []
    for _ in range(n_bars):
        drift = persistence * drift + rng.gauss(0.0, innovation)
        level *= math.exp(drift + rng.gauss(0.0, volatility))
        closes.append(level)
    return _series_to_bars(closes, random.Random(seed + 1))


def sma_spec(window: int) -> FeatureSpec:
    """A simple moving average of closes over exactly `window` bars.

    Computed from the declared window alone — no state carried between calls —
    so the same bar produces the same value regardless of how much history the
    caller happened to slice. That is the reproducibility trap recorded in the
    Phase 1 gotchas, and it is why this is not an EMA.

    Args:
        window: Bars averaged. Becomes the feature's declared lookback, which is
            what makes `FeatureSpec.evaluate` return `None` while warming up.

    Returns:
        A spec, unregistered. The gate binds it into a local `FeatureRegistry`;
        a control must never reach the registry live code reads.

    Example:
        >>> sma_spec(20).lookback
        20
    """
    return FeatureSpec(
        name=f"sma_{window}",
        version=Version.parse("1.0.0"),
        lookback=window,
        interval=BarInterval.MIN_1,
        compute=lambda bars: float(sum(bar.close.value for bar in bars[-window:]) / window),
        description=f"mean close over the last {window} bars",
    )


@dataclass(frozen=True, slots=True, order=True)
class CrossoverParams:
    """One point in the crossover search space.

    Example:
        >>> CrossoverParams(fast=5, slow=40).label
        'sma 5/40'
        >>> CrossoverParams(fast=5, slow=40, fade=True).label
        'sma 5/40 fade'
    """

    fast: int  # bars in the fast average; the responsive one
    slow: int  # bars in the slow average; must exceed `fast` or the rule is inverted
    fade: bool = False  # trade against the crossover instead of with it

    def __post_init__(self) -> None:
        """Validate the pair.

        Raises:
            ValueError: If either window is below 1, or `fast` is not strictly
                shorter than `slow` — equal windows never cross, and a reversed
                pair is the same strategy with its sign flipped, which would put
                two labels on one hypothesis in the ledger.
        """
        if self.fast < 1 or self.slow < 1:
            raise ValueError(f"windows must be at least 1, got fast={self.fast} slow={self.slow}")
        if self.fast >= self.slow:
            raise ValueError(f"fast must be shorter than slow, got {self.fast} >= {self.slow}")

    @property
    def label(self) -> str:
        """How this variant is named in the trial ledger."""
        return f"sma {self.fast}/{self.slow}{' fade' if self.fade else ''}"


SNOOPED_GRID: Final = tuple(
    CrossoverParams(fast=fast, slow=slow, fade=fade)
    for slow in (20, 30, 40, 50, 60, 80, 100)
    for fast in (2, 3, 5, 8, 13)
    for fade in (False, True)
)
"""The overfit control's search space — 70 variants, all tried, all recorded.

Both directions, which is what makes this a search rather than a sweep. Thirty-
five window pairs alone barely differ from one another on a single realisation:
they all trade the same path, so their returns are highly correlated and the
best of them is scarcely luckier than the median. Adding the fade flag pairs
every variant with its mirror, whose returns are its negative, and the search
can then pick a *sign* by luck. That is the textbook data-snooping setup and it
is the thing PBO exists to catch.
"""

HONEST_GRID: Final = (
    CrossoverParams(fast=5, slow=20),
    CrossoverParams(fast=5, slow=60),
    CrossoverParams(fast=13, slow=40),
    CrossoverParams(fast=13, slow=100),
)
"""The honest control's search space — four variants, declared before the run.

Four rather than one because PBO ranks strategies against each other and needs
at least two columns to rank. Four is still a small enough search that the
family's hurdle stays low, which is the honest researcher's position. None of
them fades: the hypothesis was declared as *momentum* before the run, and
trying both directions would be the start of the snoop.
"""


@dataclass(frozen=True, slots=True)
class Crossover(Strategy):
    """Long while the fast average is above the slow one, short while it is below.

    The oldest trend-following rule there is, chosen because it is transparently
    parameterised: the grid in `SNOOPED_GRID` is a search space a reader can see
    the size of.

    It takes a side on **every** bar, not only on the bar the averages cross.
    The gate evaluates a fixed grid of candidate decisions shared by every
    variant, so "which way was this variant leaning here?" has to be answerable
    at any bar — and one pair of label sets then serves the whole search, which
    is what keeps the variants comparable column by column.

    Long **and** short, deliberately. A long-only control cannot fit noise: on
    any one realisation of a random walk it inherits that path's realised drift
    and every variant agrees about the sign. Giving the search both directions
    is what lets thirty-five worthless rules produce an impressive best — which
    is the thing the lab is being asked to see through.

    Example:
        >>> strategy = Crossover(params=CrossoverParams(fast=5, slow=20))
        >>> strategy.qualified_name
        'control-crossover@1.0.0'
        >>> strategy.is_eligible(Regime.TREND_UP)
        True
    """

    name = "control-crossover"
    version = "1.0.0"
    regimes = (Regime.TREND_UP, Regime.TREND_DOWN)
    features = (FeatureRef("sma_fast"), FeatureRef("sma_slow"))

    params: CrossoverParams  # the windows this instance trades

    stop_fraction: Decimal = Decimal("0.02")  # stop distance as a fraction of the entry price
    horizon_bars: int = 30  # vertical barrier, in bars

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Take the side the two averages point to.

        Both features come back `None` until their windows fill, and warming up
        is not a signal: a missing value produces no intent rather than a zero.
        Exactly equal averages produce nothing either — a tie is not a view.

        Args:
            bar: The completed bar being decided on.
            context: Resolved features as of that bar; nothing else is readable.

        Returns:
            One intent, or nothing.

        Example:
            >>> strategy = Crossover(params=CrossoverParams(fast=2, slow=4))
            >>> bar = random_walk_bars(seed=1, n_bars=2)[0]
            >>> def context(fast, slow):
            ...     return StrategyContext(
            ...         symbol=CONTROL_SYMBOL, as_of=bar.ts_event,
            ...         session=MarketSession.REGULAR, regime=Regime.TREND_UP,
            ...         values={"sma_fast": fast, "sma_slow": slow},
            ...     )
            >>> strategy.on_bar(bar, context(None, 1.0))
            ()
            >>> strategy.on_bar(bar, context(2.0, 1.0))[0].side
            <Side.BUY: 'BUY'>
            >>> strategy.on_bar(bar, context(1.0, 2.0))[0].side
            <Side.SELL: 'SELL'>
            >>> fader = Crossover(params=CrossoverParams(fast=2, slow=4, fade=True))
            >>> fader.on_bar(bar, context(2.0, 1.0))[0].side
            <Side.SELL: 'SELL'>
        """
        fast = context.feature("sma_fast")
        slow = context.feature("sma_slow")
        if fast is None or slow is None or fast == slow:
            return ()
        rising = fast > slow
        side = Side.BUY if rising is not self.params.fade else Side.SELL
        return (intent_bars(bar, self, side, MarketSession.REGULAR),)


def intent_bars(bar: Bar, strategy: Crossover, side: Side, session: MarketSession) -> Intent:
    """Build the intent a control emits on `bar`.

    Every variant emits the same barrier geometry — the grid varies the entry
    rule and nothing else. That is what lets one pair of triple-barrier label
    sets serve all thirty-five variants, and it is a deliberate simplification
    of the gate rather than a property of strategies in general.

    Args:
        bar: The bar being decided on; its close is the reference level.
        strategy: The emitting instance, for the audit fields and the barriers.
        side: Direction the averages point to.
        session: Venue phase to stamp on the proposal.

    Returns:
        A market entry with a symmetric stop and target.

    Example:
        >>> strategy = Crossover(params=CrossoverParams(fast=2, slow=4))
        >>> bar = random_walk_bars(seed=1, n_bars=2)[0]
        >>> intent_bars(bar, strategy, Side.SELL, MarketSession.REGULAR).side
        <Side.SELL: 'SELL'>
    """
    offset = strategy.stop_fraction if side is Side.BUY else -strategy.stop_fraction
    stop = Price((bar.close.value * (Decimal(1) - offset)).quantize(Decimal("0.01")))
    return Intent(
        ts_event=bar.ts_event,
        ts_init=bar.ts_event,
        symbol=bar.symbol,
        side=side,
        entry=EntryTrigger.MARKET,
        entry_price=None,
        invalidation=stop,
        target_r=Decimal(1),  # symmetric barriers; the label set is shared across variants
        horizon_ns=strategy.horizon_bars * BarInterval.MIN_1.nanos,
        strategy=strategy.name,
        strategy_version=strategy.version,
        rationale=f"control: {strategy.params.label} crossover held",
    )
