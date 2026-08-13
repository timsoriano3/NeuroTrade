"""Feature specifications and the registry that holds them.

A feature is a pure function of recent market data with a **declared lookback**.
Both halves matter, and both are enforced here rather than trusted.

**Why lookback is declared rather than inferred.** Three things need it:

1. *Warm-up.* A 20-bar moving average computed from 8 bars is not a 20-bar
   moving average — it is a different number that looks plausible. Emitting it
   means the first minutes of every session use a computation that differs from
   the rest, and the effect is invisible in aggregate statistics. `evaluate`
   returns ``None`` until the window is full.
2. *Loading.* The engine has to know how much history to fetch before a session
   starts. `FeatureRegistry.max_lookback` answers that for the whole library.
3. *Cost.* Lookback bounds what a feature can see, which bounds what it costs.

**Point-in-time correctness is checked, not documented.** `evaluate` takes an
``as_of`` timestamp and refuses any window containing a bar stamped after it.
Lookahead is the single most productive way to build a backtest that works
beautifully and loses money live, and it is nearly undetectable by inspection:
the code looks right, the numbers look right, and the edge evaporates in
production. Making it raise is cheap; finding it later is not.

**Features return ``float``, not ``Decimal``.** Prices and money are exact
(`neurotrade.core.types`), but a moving average is an estimate — speed matters
and a rounding error in the twelfth decimal cannot cost anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.registry import Registry, Version

__all__ = [
    "FeatureFn",
    "FeatureRegistry",
    "FeatureSpec",
    "LookaheadError",
]

type FeatureFn = Callable[[Sequence[Bar]], float]
"""A feature's computation. Receives exactly `lookback` bars, oldest first, and
returns a single value. Must be pure: same bars in, same number out, no reading
of the clock, the network or anything else."""


class LookaheadError(ValueError):
    """Raised when a feature is given data from after the moment being modelled.

    Always a bug in the caller, never in the data. It means some part of the
    pipeline handed a feature a bar the system could not have seen yet, and any
    result computed from that window is worthless.
    """


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One registered feature.

    Example:
        >>> spec = FeatureSpec(
        ...     name="close", version=Version.parse("1.0.0"), lookback=1,
        ...     interval=BarInterval.MIN_1, description="last close",
        ...     compute=lambda bars: float(bars[-1].close.value),
        ... )
        >>> spec.lookback
        1
    """

    name: str  # registered name, e.g. "atr" or "rvol"
    version: Version  # semantic version; a retune is a new version
    lookback: int  # bars of history required before a value can be produced
    interval: BarInterval  # bar size this feature is defined against
    compute: FeatureFn  # the pure function
    description: str  # what it measures, in one line, for the dashboard

    def __post_init__(self) -> None:
        """Validate the declaration.

        Raises:
            ValueError: If the name or description is blank, or lookback is not
                at least 1. A zero lookback would mean a feature that sees no
                data.
        """
        if not self.name.strip():
            raise ValueError("feature name must not be empty")
        if self.lookback < 1:
            raise ValueError(f"{self.name} lookback must be at least 1, got {self.lookback}")
        if not self.description.strip():
            raise ValueError(f"{self.name} must describe what it measures")

    def evaluate(self, history: Sequence[Bar], as_of: Nanos) -> float | None:
        """Compute the feature as of a moment, or `None` if still warming up.

        Args:
            history: Bars in ascending `ts_event` order, ending at or before
                `as_of`. May be longer than `lookback`; only the most recent
                `lookback` bars are passed to `compute`.
            as_of: The moment being modelled. In live trading this is the
                current bar's close; in a backtest it is the simulated instant.

        Returns:
            The feature's value, or `None` while fewer than `lookback` bars are
            available. `None` means "not yet computable" and must not be
            substituted with a default — a zero here becomes a real number in a
            model input.

        Raises:
            LookaheadError: If the window contains a bar stamped after `as_of`,
                or if the bars are not in ascending order. Both mean the caller
                handed over data the system could not have had.

        Example:
            >>> spec = FeatureSpec(
            ...     name="mean_close", version=Version.parse("1.0.0"), lookback=2,
            ...     interval=BarInterval.MIN_1, description="mean of last 2 closes",
            ...     compute=lambda bars: sum(float(b.close.value) for b in bars) / len(bars),
            ... )
            >>> spec.evaluate([], as_of=0) is None          # warming up
            True
        """
        if len(history) < self.lookback:
            return None

        window = history[-self.lookback :]

        # Bounded by lookback, so this stays cheap even on hot paths, and it
        # catches an out-of-order feed at the point of use rather than as a
        # wrong number weeks later.
        previous = window[0].ts_event
        for bar in window[1:]:
            if bar.ts_event < previous:
                raise LookaheadError(
                    f"{self.name} received bars out of order: {bar.ts_event} after {previous}"
                )
            previous = bar.ts_event

        if window[-1].ts_event > as_of:
            raise LookaheadError(
                f"{self.name} was given a bar stamped {window[-1].ts_event}, "
                f"after the moment being modelled ({as_of})"
            )

        return self.compute(window)

    @property
    def qualified_name(self) -> str:
        """`name@version`, as it appears in a snapshot or a trade record.

        Example:
            >>> FeatureSpec(
            ...     name="atr", version=Version.parse("1.2.0"), lookback=14,
            ...     interval=BarInterval.MIN_1, description="average true range",
            ...     compute=lambda bars: 0.0,
            ... ).qualified_name
            'atr@1.2.0'
        """
        return f"{self.name}@{self.version}"


class FeatureRegistry:
    """The feature library.

    Wraps a generic `Registry` to add the one thing features need beyond
    name-and-version lookup: knowing how much history to load before a session
    can start.

    Example:
        >>> registry = FeatureRegistry()
        >>> @registry.feature("last_close", "1.0.0", lookback=1,
        ...                   description="most recent close")
        ... def last_close(bars):
        ...     return float(bars[-1].close.value)
        >>> registry.get("last_close").lookback
        1
    """

    __slots__ = ("_registry",)

    def __init__(self) -> None:
        self._registry: Registry[FeatureSpec] = Registry("feature")

    def feature(
        self,
        name: str,
        version: str,
        *,
        lookback: int,
        description: str,
        interval: BarInterval = BarInterval.MIN_1,
    ) -> Callable[[FeatureFn], FeatureFn]:
        """Register a function as a feature.

        Args:
            name: Registered name.
            version: `MAJOR.MINOR.PATCH`. Bump it when the computation changes —
                a feature that quietly changes meaning invalidates every model
                trained against it.
            lookback: Bars of history required. At least 1.
            description: One line on what it measures.
            interval: Bar size it is defined against. Defaults to 1-minute, the
                corpus standard.

        Returns:
            A decorator that registers the function and returns it unchanged, so
            it stays directly callable and testable.

        Example:
            >>> registry = FeatureRegistry()
            >>> @registry.feature("range", "1.0.0", lookback=1, description="bar range")
            ... def bar_range(bars):
            ...     return float(bars[-1].range)
            >>> registry.get("range").qualified_name
            'range@1.0.0'
        """

        def decorate(function: FeatureFn) -> FeatureFn:
            self._registry.add(
                name,
                version,
                FeatureSpec(
                    name=name,
                    version=Version.parse(version),
                    lookback=lookback,
                    interval=interval,
                    compute=function,
                    description=description,
                ),
            )
            return function

        return decorate

    def get(self, name: str, version: str | None = None) -> FeatureSpec:
        """Look up a feature, latest version when unspecified.

        Raises:
            KeyError: If the feature, or that version of it, is not registered.
        """
        return self._registry.get(name, version)

    def max_lookback(self, interval: BarInterval = BarInterval.MIN_1) -> int:
        """Most history any registered feature needs, for one bar size.

        The engine loads this much before a session so that every feature is
        warm at the open. Loading less means the first bars silently produce
        `None`, and a strategy that treats `None` as "no signal" would simply
        not trade the open — which is where several of §5.2's strategies live.

        Args:
            interval: Bar size to consider.

        Returns:
            The largest declared lookback, or 0 if nothing is registered.

        Example:
            >>> registry = FeatureRegistry()
            >>> @registry.feature("a", "1.0.0", lookback=20, description="a")
            ... def a(bars): return 0.0
            >>> @registry.feature("b", "1.0.0", lookback=200, description="b")
            ... def b(bars): return 0.0
            >>> registry.max_lookback()
            200
        """
        return max(
            (spec.lookback for _, _, spec in self._registry if spec.interval is interval),
            default=0,
        )

    def snapshot(self) -> tuple[str, ...]:
        """Deterministic record of the registered library, for the config hash."""
        return self._registry.snapshot()

    def freeze(self) -> None:
        """Close registration once discovery is done."""
        self._registry.freeze()

    def names(self) -> tuple[str, ...]:
        """Every registered feature name, sorted."""
        return self._registry.names()

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: object) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self._registry)} registered)"
