"""Shared shape for the three cross-sectional strategies — §5.4, §5.5.

`intraday_reversal`, `relative_strength` and `residual_reversion` differ in two
things only: which number they rank the universe on, and which end of the
ranking they buy. Everything else — the extremity filter, one trade per
instrument per session, the stop, flat at the bell — is identical, and writing
it three times is how three strategies drift into being three different rules
nobody meant.

**Private, because it is not a plugin.** Nothing registers here and nothing
imports it but the three modules named above. A strategy is still a file plus
one line in `plugins.py`; this is the part of those files that was the same.

**The stop is a fraction of the move being traded, not a fixed distance.**
Each of the three acts on a move that has already happened — a trailing return,
or a residual. The size of that move is the natural scale for the risk: a
reversal of a 2% run is wrong when the run extends to 3%, and a reversal of a
0.2% drift is wrong far sooner. A fixed stop would make the same rule aggressive
on quiet names and timid on volatile ones, and would put the strategy's real
risk in whichever instruments happened to be moving.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from decimal import Decimal
from typing import Final

from neurotrade.core.events import Bar, MarketSession
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.types import CORPUS_PLACES, Price, Side, Symbol, tidy_decimal
from neurotrade.features.cross_section import CrossSection, InstrumentSnapshot
from neurotrade.strategies.base import StrategyContext

__all__ = [
    "DEFAULT_EXTREME_NAMES",
    "DEFAULT_STOP_FRACTION",
    "MIN_RANKED",
    "SWEEP_EXTREME_NAMES",
    "cross_sectional_intent",
]

DEFAULT_EXTREME_NAMES: Final = 5
"""Names taken from each end of the ranking.

Five of 58 is roughly a decile per side, which is the portfolio construction
every cross-sectional result in the literature is reported on. It is the one
axis these rules have published evidence for varying, so it is what `sweep`
moves."""

SWEEP_EXTREME_NAMES: Final = (3, 5, 8)
"""The extremities measured, and the whole declared search for each of the three.

Three trials per strategy on the decile-versus-quintile axis. The **trailing
window is deliberately not swept**: it is set once on `CrossSectionTracker`, so
every strategy in a run shares it, and varying it would mean running several
trackers and quietly changing what every other cross-sectional strategy saw."""

DEFAULT_STOP_FRACTION: Final = 0.5
"""How much further the move may go before the idea is wrong, as a fraction of it.

Half. A reversal of a 2% run is wrong by the time the run is 3%; a continuation
of a 2% run is wrong once half of it has been given back. Not swept — it is the
same number on both sides of the trade and no published result separates its
values."""

MIN_RANKED: Final = 10
"""Instruments that must be rankable before any end of the ranking is traded.

Below ten, "top three" and "bottom three" overlap or nearly do, and the rule
stops being cross-sectional — it becomes "trade whatever printed". Ten is twice
the largest extremity swept, so the two ends can never intersect."""

_NO_TRADE: Final[tuple[Intent, ...]] = ()


def cross_sectional_intent(
    bar: Bar,
    context: StrategyContext,
    *,
    measure: Callable[[InstrumentSnapshot], float | None],
    long_end_is_top: bool,
    extreme_names: int,
    stop_fraction: float,
    target_r: Decimal,
    traded: dict[Symbol, date],
    strategy: str,
    version: str,
    measure_name: str,
) -> Sequence[Intent]:
    """Rank the universe on one measure and trade this instrument if it is extreme.

    Args:
        bar: The bar that just closed.
        context: Its view. `cross_section` carries the universe as of the
            previous tick, and `levels` the bell this trade is flat by.
        measure: The number the ranking is on. `None` excludes a row, which is
            how a cold beta or a cold trailing return keeps a name out of the
            ranking instead of placing it at one end.
        long_end_is_top: `True` for a continuation rule — buy the top of the
            ranking — and `False` for a reversion rule, which buys the bottom.
            The single flag that makes two opposite strategies one function.
        extreme_names: How many names at each end are traded.
        stop_fraction: Further move, as a fraction of the measured one, at
            which the idea is wrong.
        target_r: Profit barrier in multiples of that risk.
        traded: The caller's per-symbol "already traded today", mutated here.
            Passed in rather than kept here because the state belongs to the
            strategy instance the host subscribed, not to this helper.
        strategy: Registered plugin name, stamped on the intent.
        version: Its semantic version.
        measure_name: What the ranking is on, for the rationale.

    Returns:
        At most one proposal, and nothing at all unless this instrument sits in
        the top or bottom `extreme_names` of a ranking of at least `MIN_RANKED`.
    """
    levels = context.levels
    section = context.cross_section
    if levels is None or section is None:
        return _NO_TRADE
    if context.session is not MarketSession.REGULAR or bar.ts_event >= levels.close_ns:
        return _NO_TRADE
    if traded.get(bar.symbol) == levels.session_date:
        return _NO_TRADE

    placing = _placing(section, bar.symbol, measure, extreme_names)
    if placing is None:
        return _NO_TRADE
    at_top, value = placing

    # A continuation buys the top and a reversion buys the bottom. Everything
    # else about the two rules is identical, which is why they share this file.
    side = Side.BUY if at_top is long_end_is_top else Side.SELL
    # Risk scales with the move that produced the signal, so a signal from a
    # 0.2% drift buys a small position and one from a 2% run buys a large stop
    # and therefore a small position too (§6.1).
    risk = abs(value) * stop_fraction
    if risk <= 0:
        return _NO_TRADE
    invalidation = _stop_price(bar.close, side=side, risk_fraction=risk)
    if invalidation == bar.close:
        # The move was too small to move the stop off the entry at corpus
        # precision. R would be zero and `signals_from_intents` would drop it
        # silently; declining here keeps the decision visible as a non-decision.
        return _NO_TRADE

    traded[bar.symbol] = levels.session_date
    return (
        Intent(
            symbol=bar.symbol,
            ts_event=bar.ts_event,
            ts_init=bar.ts_event,
            side=side,
            entry=EntryTrigger.MARKET,
            entry_price=None,
            invalidation=invalidation,
            target_r=target_r,
            horizon_ns=levels.close_ns - bar.ts_event,
            strategy=strategy,
            strategy_version=version,
            rationale=(
                f"{measure_name} {value:+.4f} ranks "
                f"{'top' if at_top else 'bottom'} {extreme_names} of "
                f"{len(section)} at {bar.ts_event}"
            ),
        ),
    )


def _placing(
    section: CrossSection,
    symbol: Symbol,
    measure: Callable[[InstrumentSnapshot], float | None],
    extreme_names: int,
) -> tuple[bool, float] | None:
    """Whether this instrument is at an extreme of the ranking, and its measure.

    Returns:
        `(at_top, value)`, or `None` when the instrument is not ranked, the
        ranking is shorter than `MIN_RANKED`, or the instrument sits in the
        middle. `None` for a short ranking rather than a wider extremity: the
        rule is a statement about a cross-section, and a cross-section of six is
        not one.
    """
    ordering = section.rank_by(measure)
    if len(ordering) < MIN_RANKED or symbol not in ordering:
        return None
    row = section.rows[symbol]
    value = measure(row)
    if value is None:  # unreachable while it is in `ordering`; kept for the type
        return None
    position = ordering.index(symbol)
    if position < extreme_names:
        return (True, value)
    if position >= len(ordering) - extreme_names:
        return (False, value)
    return None


def _stop_price(entry: Price, *, side: Side, risk_fraction: float) -> Price:
    """The invalidation level, `risk_fraction` away from the entry, on the legal side.

    Via `str`, so the binary artefact of a float fraction never reaches a price
    a stop order would be placed at.
    """
    move = Decimal(str(risk_fraction))
    factor = (1 - move) if side is Side.BUY else (1 + move)
    return Price(tidy_decimal(entry.value * factor, CORPUS_PLACES))
