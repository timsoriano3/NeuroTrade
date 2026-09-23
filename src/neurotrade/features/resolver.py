"""Per-symbol bar history, and the registered features evaluated over it.

A `FeatureSpec` knows how to turn *n* bars into a number, and refuses data from
after the moment being modelled. What it does not do is remember anything: it is
handed a window each time. Something has to keep that window, per instrument,
as bars arrive. This is that something.

**Why the history lives here and not in the strategy.** `StrategyContext` hands
a strategy resolved values and no bar series at all, precisely so that no
strategy can index past the current bar (§3.6). That guarantee only holds if
exactly one component owns the series — this one — and every reader goes through
`FeatureSpec.evaluate`, which raises rather than returning a lookahead value.

**History is continuous across sessions, deliberately.** `FeatureRegistry.
max_lookback` exists so the engine can load that much history *before* a session
starts, "so that every feature is warm at the open"; several of §5.2's
strategies trade the first minutes and would never fire against a buffer that
reset at midnight. The consequence is that a window spanning the overnight gap
contains it — which is what ATR is supposed to see, since a gap is a real part
of the range, and is stated here because it is not what a reader assumes.
"""

from __future__ import annotations

from collections.abc import Mapping

from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Symbol
from neurotrade.features.registry import FeatureSpec

__all__ = ["FeatureResolver"]


class FeatureResolver:
    """Keeps the rolling window each registered feature needs, per symbol.

    Keys are labels chosen by the caller, not feature names, because two
    strategies may depend on two *versions* of one feature — §10.2 runs a
    champion pinned to an exact version beside a challenger tracking latest.
    Keying on the name alone would silently give both the same number.

    Example:
        >>> from neurotrade.features.indicators import indicators
        >>> resolver = FeatureResolver(
        ...     {"atr@1.0.0": indicators.get("atr")}, interval=BarInterval.MIN_1
        ... )
        >>> resolver.lookback
        15
    """

    __slots__ = ("_history", "_interval", "_lookback", "_specs")

    def __init__(self, specs: Mapping[str, FeatureSpec], *, interval: BarInterval) -> None:
        """Bind a fixed set of features to one bar size.

        Args:
            specs: Features to evaluate, by the label each is reported under.
            interval: Bar size this resolver is fed. Every spec must declare it.

        Raises:
            ValueError: If any spec declares a different interval. A daily
                feature fed minute bars returns a number rather than an error,
                and that number is wrong in a way nothing downstream can see.
        """
        mismatched = sorted(
            f"{label} ({spec.interval.value})"
            for label, spec in specs.items()
            if spec.interval is not interval
        )
        if mismatched:
            raise ValueError(
                f"features declared for {interval.value} bars: {', '.join(mismatched)}"
            )
        # Sorted, so the resolved mapping's iteration order is a property of the
        # declaration rather than of the order strategies happened to register.
        self._specs: dict[str, FeatureSpec] = dict(sorted(specs.items()))
        self._interval = interval
        self._lookback = max((spec.lookback for spec in specs.values()), default=0)
        self._history: dict[Symbol, list[Bar]] = {}

    @property
    def lookback(self) -> int:
        """Bars of history the widest declared feature needs.

        What a caller must supply before the first decision if every feature is
        to be warm at it.
        """
        return self._lookback

    def observe(self, bar: Bar) -> None:
        """Add a bar to its symbol's window.

        Args:
            bar: A completed bar. Must be at this resolver's interval.

        Raises:
            ValueError: If the bar is at a different interval.

        Example:
            >>> from neurotrade.features.indicators import indicators
            >>> resolver = FeatureResolver(
            ...     {"logret": indicators.get("log_return")}, interval=BarInterval.MIN_1
            ... )
            >>> resolver.observe(a_bar)
            >>> resolver.resolve(a_bar)["logret"] is None      # one bar, needs two
            True
        """
        if bar.interval is not self._interval:
            raise ValueError(
                f"resolver is fed {self._interval.value} bars, got {bar.interval.value}"
            )
        if self._lookback == 0:
            return
        history = self._history.setdefault(bar.symbol, [])
        history.append(bar)
        # Trimmed in batches rather than on every append: `del history[0]` is a
        # memmove of the whole list, and this runs once per bar per symbol over
        # a corpus of millions. Only the tail is ever read, so carrying up to
        # twice the lookback between trims costs nothing.
        if len(history) > 2 * self._lookback:
            del history[: len(history) - self._lookback]

    def resolve(self, bar: Bar) -> dict[str, float | None]:
        """Evaluate every declared feature for a bar's symbol, as of its close.

        Args:
            bar: The bar being modelled. Its `ts_event` is the close, so it is
                the latest instant a feature may read — pass the bar *after*
                observing it.

        Returns:
            Value by label, in sorted label order. `None` means still warming
            up, never zero.

        Raises:
            LookaheadError: If the window holds a bar stamped after the close,
                or the bars are out of order. Both mean the feed is wrong.

        Example:
            >>> from dataclasses import replace
            >>> from neurotrade.features.indicators import indicators
            >>> resolver = FeatureResolver(
            ...     {"logret": indicators.get("log_return")}, interval=BarInterval.MIN_1
            ... )
            >>> later = replace(a_bar, ts_event=61_000, close=Price("101"))
            >>> for one in (a_bar, later):
            ...     resolver.observe(one)
            >>> round(resolver.resolve(later)["logret"], 6)
            0.004963
        """
        history = self._history.get(bar.symbol, ())
        return {label: spec.evaluate(history, bar.ts_event) for label, spec in self._specs.items()}

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"FeatureResolver({len(self._specs)} features, lookback={self._lookback})"
