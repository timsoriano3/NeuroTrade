"""The corpus as one ordered event stream.

A backtest over a universe needs every symbol's bars interleaved into a single
ascending sequence, because that is what the live system sees: one tape, many
instruments, in the order the venues produced them. Reading symbol-by-symbol
instead would let a strategy see all of AAPL's day before any of MSFT's, which
is not a backtest of anything that could happen.

**Ordering is total, not just by time.** Many symbols share a `ts_event` — at
one-minute bars, every instrument in the universe closes its bar on the same
tick. A merge keyed on time alone would leave their relative order up to
whichever iterator the heap happened to pop first, and the determinism invariant
would be violated in a way no test on a single symbol could catch. The key is
therefore `(ts_event, seq, symbol)`, which is total: no two bars in the corpus
can tie on all three.

**Symbols are sorted on the way in** for the same reason. A caller passing a set
or a dict's keys would otherwise hand the merge a different input order per
process, and `Universe` already sorts — this just means nothing else has to.

Reading stays lazy: `StoragePort.read_bars` returns an iterator per symbol and
`heapq.merge` pulls from them one bar at a time, so a five-year backtest holds
one bar per symbol in memory rather than five years of them.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.ports import StoragePort
from neurotrade.core.types import Symbol

__all__ = ["CorpusFeed", "merge_bars"]


def _order_key(bar: Bar) -> tuple[Nanos, int, Symbol]:
    """Total order over bars: time, then tiebreaker, then instrument."""
    return (bar.ts_event, bar.seq, bar.symbol)


def merge_bars(
    store: StoragePort,
    symbols: Iterable[Symbol],
    interval: BarInterval,
    start: Nanos,
    end: Nanos,
) -> Iterator[Bar]:
    """Interleave many symbols' bars into one ascending stream.

    Args:
        store: The corpus to read from.
        symbols: Instruments to merge. Sorted internally, so the caller's
            iteration order cannot affect the result.
        interval: Bar size. §12.1 targets one-minute bars.
        start: Inclusive lower bound on `ts_event`.
        end: Exclusive upper bound on `ts_event`.

    Yields:
        Bars in ascending `(ts_event, seq, symbol)` order.

    Example:
        >>> from neurotrade.lab.feed import merge_bars
        >>> class OneBarStore:
        ...     def write_bars(self, bars, *, source, session_date): ...
        ...     def read_bars(self, symbol, interval, start, end):
        ...         return iter([a_bar])
        >>> merged = merge_bars(
        ...     OneBarStore(), [AAPL], BarInterval.MIN_1, 0, 10_000
        ... )
        >>> [bar.symbol.ticker for bar in merged]
        ['AAPL']
    """
    streams = [store.read_bars(symbol, interval, start, end) for symbol in sorted(set(symbols))]
    return heapq.merge(*streams, key=_order_key)


@dataclass(frozen=True, slots=True)
class CorpusFeed:
    """A universe's bars, ready to drive a backtest.

    Binds the three things that do not change across a run — the store, the
    instruments and the bar size — so that `events` takes only a time range.
    That is the shape `drive` wants, and it keeps the range the one thing a
    caller varies between folds of a cross-validation.

    Example:
        >>> from neurotrade.lab.feed import CorpusFeed
        >>> class OneBarStore:
        ...     def write_bars(self, bars, *, source, session_date): ...
        ...     def read_bars(self, symbol, interval, start, end):
        ...         return iter([a_bar])
        >>> feed = CorpusFeed(OneBarStore(), (AAPL,), BarInterval.MIN_1)
        >>> len(list(feed.events(0, 10_000)))
        1
    """

    store: StoragePort  # the corpus
    symbols: tuple[Symbol, ...]  # instruments in the run's universe
    interval: BarInterval = BarInterval.MIN_1  # bar size; §12.1's standard

    def events(self, start: Nanos, end: Nanos) -> Iterator[Bar]:
        """Bars over a half-open range, merged across the universe.

        Args:
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            An iterator of bars in ascending `(ts_event, seq, symbol)` order.
        """
        return merge_bars(self.store, self.symbols, self.interval, start, end)
