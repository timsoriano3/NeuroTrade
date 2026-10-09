"""One labelled trade, as it is written down.

**Why the record is in `core`.** `TradeJournalPort` in `core.ports` has to name
what it appends, and `core` depends on nothing — so the record cannot live in
`lab/` beside the code that produces it, for the same reason `Trial` cannot.

**Why this exists at all.** `lab.labelling.BarrierTouch` is computed for every
decision a strategy takes and was, until this record, aggregated into counts and
thrown away. That made three questions unanswerable: which trades the
`ambiguous` flag charged against us, how much of the available move each
strategy kept, and whether a position that reached `+jR` tends to carry on to
`+kR` — the last being the gate on any add-to-winners logic. All three need the
individual trades, not their mean.

**The cost basis travels with the record.** A measurement's digest covers the
decision stream, so the same digest can carry opposite signs under different
spread assumptions — `gap_continuation` scored `-0.00015` and `+0.00020` under
one digest. `spread_fraction` and `commission_per_share` are therefore stamped
on every record: a row whose cost basis has to be inferred from a run log is a
row that will eventually be read wrong.

**`label` is an int, not an enum.** `lab.labelling.Label` is an `IntEnum` and
`core` may not import `lab`, so the value travels as `+1` profit, `0` timeout,
`-1` stop. The mapping is stated here because it is the one thing a reader
cannot recover from the field's type.

**What is deliberately absent: the point-in-time feature snapshot.** The
Invariants require one on every trade record, and nothing in the pipeline
produces it today — `Intent` carries no feature values, so the engine's
`StrategyContext.feature` readings are gone by the time a decision is labelled.
Attaching them is a change to an event type and therefore to replay
determinism, which belongs in its own commit. Until then this record describes
an outcome fully and its cause not at all, and no model should be trained on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from neurotrade.core.clock import Nanos
from neurotrade.core.types import Side

__all__ = ["TradeRecord"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TradeRecord:
    """One decision, its barriers, and what became of it.

    Example:
        >>> record = TradeRecord(
        ...     run_id="run_a", config_hash="cfg_a", recorded_ns=5,
        ...     symbol="AAPL.NASDAQ", strategy="gap_continuation",
        ...     strategy_version="1.0.0", variant="gap>=1.5 ranges",
        ...     side=Side.BUY, entry_ns=1, exit_ns=4, bars_held=3,
        ...     entry=Decimal("100"), exit=Decimal("102"),
        ...     profit_target=Decimal("0.02"), stop_loss=Decimal("0.01"),
        ...     max_bars=30, label=1, ambiguous=False,
        ...     realised_return=Decimal("0.0194"), gross_return=Decimal("0.02"),
        ...     mfe=Decimal("0.025"), mae=Decimal("0.004"),
        ...     spread_fraction=Decimal("0.0005"),
        ...     commission_per_share=Decimal("0.005"),
        ... )
        >>> record.is_win, record.reached_target
        (True, True)
    """

    run_id: str  # the run that produced it, so a row ties back to a log
    config_hash: str  # configuration in force (Invariants); from `TrialLedger.config_hash`
    recorded_ns: Nanos  # when written, from the run's Clock — never wall clock
    symbol: str  # `str(Symbol)`, e.g. "AAPL.NASDAQ"; a string so the row is self-describing
    strategy: str  # registered plugin name
    strategy_version: str  # its semantic version, so a retune is distinguishable
    variant: str  # which configuration of it — the hypothesis the trial ledger counts
    side: Side  # BUY for long, SELL for short
    entry_ns: Nanos  # ts_event of the bar whose close is the entry reference
    exit_ns: Nanos  # ts_event of the bar the position closed on
    bars_held: int  # bars from entry to exit, inclusive of the exit bar
    entry: Decimal  # entry reference price
    exit: Decimal  # price the position closed at
    profit_target: Decimal  # profit barrier, positive fraction of entry
    stop_loss: Decimal  # stop barrier, positive fraction of entry
    max_bars: int  # time barrier, in bars after the entry bar
    label: int  # +1 profit, 0 timeout, -1 stop — see the module docstring
    ambiguous: bool  # both barriers fell in the exit bar; the stop was assumed
    realised_return: Decimal  # signed fractional return NET of modelled costs
    gross_return: Decimal  # signed fractional return BEFORE costs
    mfe: Decimal  # max favourable excursion while held, fraction of entry, gross, >= 0
    mae: Decimal  # max adverse excursion while held, fraction of entry, gross, >= 0
    spread_fraction: Decimal  # the spread estimate in force, as a fraction of price
    commission_per_share: Decimal  # the commission in force, per share, in the fee currency

    @property
    def is_win(self) -> bool:
        """Whether the trade made money after costs.

        Not the same as `reached_target`: a position can touch its profit
        barrier and still lose once the spread and commission are paid, which
        is the case a cost-blind label hides.
        """
        return self.realised_return > 0

    @property
    def reached_target(self) -> bool:
        """Whether the favourable excursion ever got as far as the profit barrier.

        True on an assumed-stop trade whose price did reach the target inside
        the exit bar — which is the whole reason `mfe` is recorded. Counting
        these is how the cost of the `ambiguous` tie-break becomes visible
        rather than being a known-unknown in the methodology.

        Example:
            >>> TradeRecord(
            ...     run_id="r", config_hash="c", recorded_ns=1, symbol="AAPL.NASDAQ",
            ...     strategy="s", strategy_version="1.0.0", variant="v",
            ...     side=Side.BUY, entry_ns=1, exit_ns=2, bars_held=1,
            ...     entry=Decimal("100"), exit=Decimal("98"),
            ...     profit_target=Decimal("0.02"), stop_loss=Decimal("0.02"),
            ...     max_bars=5, label=-1, ambiguous=True,
            ...     realised_return=Decimal("-0.02"), gross_return=Decimal("-0.02"),
            ...     mfe=Decimal("0.02"), mae=Decimal("0.02"),
            ...     spread_fraction=Decimal("0"), commission_per_share=Decimal("0"),
            ... ).reached_target
            True
        """
        return self.mfe >= self.profit_target
