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

**Split adjustment happens here, on the way out of the corpus.** `raw/` is
immutable (§12.1) and `lab/labelling.py` says in its own docstring that prices
must be split-adjusted before they reach it and that *nothing there can detect
that it was not done*. `AdjustingStore` is what does it: a `StoragePort` that
wraps another one and rescales every bar into one basis as it is read. Nothing
downstream changes — `merge_bars`, `CorpusFeed`, `measure_strategy` and the gate
all take a port and cannot tell which they were given.

Adjusting on read rather than materialising a `derived/bars/` corpus is
deliberate. An adjusted corpus on disk freezes the basis it was built in, so the
next split silently makes every file stale, and it doubles 38k Parquet files for
a factor that is `1` on all but a handful of sessions. A basis chosen per run is
also the honest one: `as_of` is a parameter of the question being asked.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from neurotrade.core.actions import AdjustmentSeries, CorporateAction, adjust_stream
from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.ports import StoragePort
from neurotrade.core.types import Symbol

__all__ = ["AdjustingStore", "CorpusFeed", "merge_bars"]


def _order_key(bar: Bar) -> tuple[Nanos, int, Symbol]:
    """Total order over bars: time, then tiebreaker, then instrument."""
    return (bar.ts_event, bar.seq, bar.symbol)


@dataclass(frozen=True, slots=True)
class AdjustingStore:
    """A corpus read through a corporate-action adjustment.

    Implements `StoragePort` by delegating to another one and rescaling every
    bar it yields into the basis of `as_of`. That makes adjustment a property of
    the *corpus handed to a run* rather than something each caller has to
    remember, which is the only arrangement that survives: the labeller cannot
    detect an unadjusted series, so a caller who forgets gets a plausible number
    instead of an error.

    **What goes wrong without it.** An unadjusted 10:1 split is a 90% overnight
    fall. `prior_close` reads it as the largest gap in the corpus, the noise
    band built from it explodes, and every stop between the two sessions is
    trodden through. Four splits fall inside the crawled IBKR range, two of them
    10:1, so this is not a hypothetical.

    Args:
        inner: The corpus underneath — normally the Parquet store.
        series: One `AdjustmentSeries` per symbol. A symbol absent from the
            mapping is read straight through, which is also what an empty series
            does; both are recorded as "no action known", and `neurotrade actions
            check` is what tells that apart from "no action happened".
        as_of: The basis every bar is expressed in. Actions after it are
            ignored, so a run's prices are on the footing they were on at the
            end of the window it measures.
        total_return: Adjust for dividends too. **Left false**, because these
            prices decide barrier touches: a dividend-adjusted series moves a
            stop that the market never moved.

    Example:
        A symbol with nothing recorded against it reads through untouched.

        >>> from datetime import date
        >>> from neurotrade.core.actions import AdjustmentSeries
        >>> class OneBarStore:
        ...     def write_bars(self, bars, *, source, session_date): ...
        ...     def read_bars(self, symbol, interval, start, end):
        ...         return iter([a_bar])
        >>> store = AdjustingStore(
        ...     OneBarStore(), {AAPL: AdjustmentSeries(AAPL, ())}, as_of=date(2024, 1, 1)
        ... )
        >>> [bar is a_bar for bar in store.read_bars(AAPL, BarInterval.MIN_1, 0, 10_000)]
        [True]
    """

    inner: StoragePort  # the corpus underneath
    series: Mapping[Symbol, AdjustmentSeries]  # actions per instrument
    as_of: date  # basis every price is expressed in; later actions are ignored
    total_return: bool = False  # adjust dividends as well as splits; see the class docstring

    def write_bars(self, bars: Sequence[Bar], *, source: str, session_date: date) -> None:
        """Refuse the write.

        Raises:
            NotImplementedError: Always. These bars have been rescaled into one
                basis, and `raw/` is immutable — writing them back would put
                derived prices in the corpus everything else is recomputed from,
                with no record of the basis they were adjusted to.
        """
        raise NotImplementedError(
            "an AdjustingStore is a read-side view; adjusted bars are never written back"
        )

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: Nanos, end: Nanos
    ) -> Iterator[Bar]:
        """Read one symbol's bars, rescaled into the `as_of` basis.

        Args:
            symbol: Instrument to read.
            interval: Bar size.
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Bars in ascending `ts_event` order, lazily. A symbol with no actions
            known is returned as **the inner iterator itself**, so a corpus
            where nothing splits is read exactly as it would be unwrapped —
            which is what lets an existing measurement's digest be reproduced
            through this class and prove the wrapping changed nothing else.
        """
        raw = self.inner.read_bars(symbol, interval, start, end)
        series = self.series.get(symbol)
        if series is None or not series.actions:
            return raw
        return adjust_stream(raw, series, as_of=self.as_of, total_return=self.total_return)

    def splits_in_force(self, *, since: date) -> tuple[tuple[Symbol, CorporateAction], ...]:
        """Every split that actually rescales a bar in a window, for the report.

        Scoped to the window rather than to the whole series, because a split
        *before* the window changes nothing inside it: a bar on session `d` takes
        the product of the actions in `(d, as_of]`, so an action older than the
        first session has a factor of exactly 1 everywhere. Listing those would
        report adjustments that did not happen, on the same principle that keeps
        dividends out of this list — `total_return` is false, so a dividend is
        not applied at all.

        Args:
            since: The window's first session. Splits effective on or before it
                are already in basis and excluded.

        Returns:
            `(symbol, action)` pairs, ordered by effective date then symbol.
            Empty means every price came out of the corpus as it went in — a
            finding worth printing, not an absence worth assuming.
        """
        found = [
            (symbol, action)
            for symbol, series in self.series.items()
            for action in series.actions
            if action.is_split and since < action.effective_date <= self.as_of
        ]
        return tuple(sorted(found, key=lambda pair: (pair[1].effective_date, str(pair[0]))))


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
