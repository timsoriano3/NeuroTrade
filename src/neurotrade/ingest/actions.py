"""Fetching corporate actions for a universe, and auditing the corpus against them.

Two halves of one job (§12.1 stage 5). The fetch collects what the feed knows;
the audit asks whether the prices agree. Both work through ports only — nothing
here imports a concrete adapter, so the same code serves Yahoo today and any
other source later.

**Why the audit exists at all.** A corporate-action feed cannot report an error
it does not know it made. If Yahoo omits a split, the fetch succeeds, the file
looks complete, and the corpus quietly contains a -75% overnight move that will
be labelled as a stop-out. The only evidence is in the prices, so the audit
reads bars, applies every action that *was* reported, and looks for what is
left over. A clean scan is weak evidence that the adjustment is right; a dirty
one is strong evidence that it is wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal

from neurotrade.core.actions import AdjustmentSeries, CorporateAction, PriceGap, unexplained_gaps
from neurotrade.core.clock import to_nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.ports import CorporateActionsPort, StoragePort
from neurotrade.core.types import Symbol
from neurotrade.core.universe import Universe
from neurotrade.logs import get_logger

__all__ = ["ActionFetchReport", "GapScanReport", "fetch_actions", "scan_gaps"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ActionFetchReport:
    """What one fetch collected.

    Example:
        >>> ActionFetchReport(symbols=3, splits=1, dividends=8, failures=()).describe()
        '3 symbols, 1 splits, 8 dividends'
    """

    symbols: int  # instruments fetched, including those with no actions
    splits: int  # split rows collected across all symbols
    dividends: int  # dividend rows collected across all symbols
    failures: tuple[tuple[Symbol, str], ...]  # symbols the feed refused, with the reason

    def describe(self) -> str:
        """One line for the terminal."""
        line = f"{self.symbols} symbols, {self.splits} splits, {self.dividends} dividends"
        if self.failures:
            line += f", {len(self.failures)} failed"
        return line


@dataclass(frozen=True, slots=True)
class GapScanReport:
    """What one audit found.

    Example:
        >>> GapScanReport(symbols=40, bars=1000, gaps=()).clean
        True
    """

    symbols: int  # instruments scanned
    bars: int  # daily bars examined
    gaps: tuple[PriceGap, ...]  # moves no recorded action explains

    @property
    def clean(self) -> bool:
        """Whether every overnight move was explained."""
        return not self.gaps

    def describe(self) -> str:
        """One line for the terminal."""
        verdict = "clean" if self.clean else f"{len(self.gaps)} unexplained"
        return f"{self.symbols} symbols, {self.bars} bars, {verdict}"


async def fetch_actions(
    universe: Universe,
    feed: CorporateActionsPort,
    *,
    start: date,
    end: date,
    on_symbol: Callable[[Symbol, int], None] | None = None,
) -> tuple[dict[Symbol, tuple[CorporateAction, ...]], ActionFetchReport]:
    """Fetch every instrument's actions in range.

    Sequential rather than concurrent. The universe is a few dozen names today
    and a few thousand later, and Yahoo rate-limits an enthusiastic client into
    empty responses — which, being indistinguishable from "no actions", would
    enter the corpus as a silent wrong answer rather than an error.

    A symbol the feed refuses is recorded as a failure and skipped, **not**
    stored as empty. Storing it empty would assert that the name never split.

    Args:
        universe: Instruments to fetch.
        feed: The source, behind its port.
        start: First effective date of interest.
        end: Last effective date of interest.
        on_symbol: Called with each symbol and its action count as it lands,
            for progress reporting.

    Returns:
        The actions by symbol, and a report. Only symbols that were fetched
        successfully appear in the mapping.

    Raises:
        ValueError: If `end` precedes `start`.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")

    collected: dict[Symbol, tuple[CorporateAction, ...]] = {}
    failures: list[tuple[Symbol, str]] = []
    splits = dividends = 0

    for symbol in universe:
        try:
            actions = tuple(await feed.fetch_actions(symbol, start, end))
        except Exception as error:
            log.warning("actions_fetch_failed", symbol=str(symbol), error=str(error))
            failures.append((symbol, str(error)))
            continue
        collected[symbol] = actions
        splits += sum(1 for action in actions if action.is_split)
        dividends += sum(1 for action in actions if action.is_dividend)
        if on_symbol is not None:
            on_symbol(symbol, len(actions))

    report = ActionFetchReport(
        symbols=len(collected),
        splits=splits,
        dividends=dividends,
        failures=tuple(failures),
    )
    log.info(
        "actions_fetched",
        symbols=report.symbols,
        splits=report.splits,
        dividends=report.dividends,
        failures=len(report.failures),
    )
    return collected, report


def scan_gaps(
    universe: Universe,
    storage: StoragePort,
    actions: Mapping[Symbol, Sequence[CorporateAction]],
    *,
    start: date,
    end: date,
    threshold: Decimal | None = None,
    already_split_adjusted: bool = False,
) -> GapScanReport:
    """Audit the daily corpus for moves the recorded actions do not explain.

    Reads daily bars for each instrument, adjusts them with that instrument's
    actions, and reports any remaining overnight jump past the threshold.

    **Some sources adjust before we see the data.** Yahoo's `auto_adjust=False`
    suppresses only the *dividend* adjustment; its OHLC is always split-adjusted.
    Verified 2026-09-17: AMZN's stored closes run 121-125 straight through its
    20:1 split on 2022-06-06, when it traded near $2,400 the week before.
    Applying our own split factor on top of that divides by twenty a second
    time. `already_split_adjusted` is how a caller states the basis it has,
    because nothing in a bar records it.

    `as_of` for the adjustment is `end` — the whole range is being examined at
    once from the vantage point of its last session, which is the correct basis
    for an audit even though it would be lookahead in a backtest. The audit is
    not making a trading decision; it is asking whether the data is internally
    consistent, and it needs every action to do that.

    Args:
        universe: Instruments to scan.
        storage: The corpus, behind its port.
        actions: Actions by symbol, as `fetch_actions` returned them.
        start: First session to examine.
        end: Last session to examine, inclusive.
        threshold: Fractional move past which a gap is reported. Defaults to
            `unexplained_gaps`'s own default.
        already_split_adjusted: Whether the stored bars are *already* on one
            split basis. Set it for any source that adjusts before we see the
            data, or every split is counted twice and the audit reports a
            phantom 20x gap at exactly the date it was meant to vindicate.
            Yahoo is such a source — see the note below.

    Returns:
        A report. `clean` is the result to want.

    Raises:
        ValueError: If `end` precedes `start`.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")

    lower = to_nanos(datetime.combine(start, time.min, tzinfo=UTC))
    # Exclusive upper bound on ts_event, so push past the last session's close.
    upper = to_nanos(datetime.combine(end, time.max, tzinfo=UTC))

    gaps: list[PriceGap] = []
    scanned = bars_seen = 0
    for symbol in universe:
        bars = tuple(storage.read_bars(symbol, BarInterval.DAY_1, lower, upper))
        if not bars:
            continue
        scanned += 1
        bars_seen += len(bars)
        # An already-adjusted source needs no factor applied, only the scan.
        series = AdjustmentSeries(symbol, () if already_split_adjusted else actions.get(symbol, ()))
        kwargs = {} if threshold is None else {"threshold": threshold}
        gaps.extend(unexplained_gaps(bars, series, as_of=end, **kwargs))

    report = GapScanReport(symbols=scanned, bars=bars_seen, gaps=tuple(gaps))
    log.info(
        "actions_gap_scan",
        symbols=report.symbols,
        bars=report.bars,
        gaps=len(report.gaps),
        clean=report.clean,
    )
    return report
