"""Prior-close pressure reversal: fade the closing push, overnight — §5.3, Family B.

**The idea in one line.** A price pushed away from the day's volume-weighted
average in the last minutes of trading was pushed by someone who had to be done
by the bell, and that pressure is not information — it reverses once the next
session opens.

**The evidence, and the proxy.** The literature's finding is about the *closing
auction*: deviations from fair value in the closing print reverse almost fully
overnight, a third to a half of it within the first 30 minutes. We have no
auction data — the corpus is one-minute RTH bars — so the deviation is measured
as distance from session VWAP in units of the session's own volume-weighted
dispersion, exactly as `vwap_band_reversion` measures it intraday. That is a
proxy for auction pressure, not a measurement of it, and the strategy claims no
more than that.

**This is the only strategy in the arsenal that is not flat at the bell**, which
is why Phase 2 plan decision 2 exists. It declares `holds_overnight`, and
`BacktestEngine` refuses to host it alongside a day strategy: Reg-T overnight
margin is 2:1 against 4:1 intraday, and a stop cannot fill through a gap, so an
overnight position's risk model is not the day engine's. Mixing them corrupts
the sizing model for both.

**The exit is 30 minutes into the next session, and that is why `next_open_ns`
exists.**
`Intent.horizon_ns` is a span, and a fixed 17.5 hours works on a weeknight and
lands on a Saturday after a Friday close — where the labeller finds no bars and
the barrier collapses back to Friday's bell, measuring the last five minutes
instead of the overnight. `SessionLevels.next_open_ns` comes out of the trading
calendar, which is published years ahead and is not market data, so reading it
is not lookahead. Without it the strategy declines.

**Index ETFs only, per plan decision 2** — "a separate overnight family, index
ETFs only, unlevered, own limits". Enforced here as a declared instrument set
rather than left to a host: a single name can gap 20% on an earnings release
after the close, which is a different risk from an index's overnight drift and
is not one this rule has any evidence about.

**What this cannot honestly measure, and says so.** The label walks bar to bar,
so the first bar of the next session is where the barriers are first tested — and
if price gapped through the stop overnight, the labeller records a touch at that
bar's low as though a stop had filled there. It did not; it would have filled at
the open. That overstates the strategy, in the direction that flatters it. The
barriers are therefore placed far enough out that a normal overnight cannot reach
them and the position resolves on the **time** barrier at the next open, which is
the quantity the literature actually reports. `TARGET_R` and `STOP_SIGMAS` exist
to define R, not to be hit. Honest fills are Phase 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import ClassVar, Final, Self

from neurotrade.core.events import Bar, MarketSession
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.types import CORPUS_PLACES, Price, Side, Symbol, tidy_decimal
from neurotrade.features.levels import vwap_distance
from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.base import Regime, Strategy, StrategyContext

__all__ = ["EXIT_MINUTES_AFTER_OPEN", "OVERNIGHT_INSTRUMENTS", "PriorCloseReversal"]

_NO_TRADE: Final[tuple[Intent, ...]] = ()

_NANOS_PER_MINUTE: Final = 60_000_000_000

OVERNIGHT_INSTRUMENTS: Final = frozenset({"SPY", "QQQ", "DIA", "IWM", "XIU"})
"""Tickers the overnight family may hold, by ticker rather than by `Symbol`.

Plan decision 2: index ETFs only. By ticker because the same index proxy may be
listed on more than one venue and the restriction is about *what* the instrument
is, not where it trades. A single name is excluded because it can gap 20% on an
earnings release after the close — a different risk from an index's overnight
drift, and one this rule has no evidence about. `EEM` is left out deliberately:
it is an emerging-markets basket whose constituents trade while the US is shut,
so its overnight move is other markets' regular session, not a reversal of
anything that happened here."""

DEFAULT_ENTRY_MINUTES_BEFORE_CLOSE: Final = 10
"""How close to the bell an entry is considered, in minutes.

Ten because the pressure the rule is about builds into the closing auction, and
a deviation measured at 15:00 has most of an hour to resolve inside the session —
which is `vwap_band_reversion`'s trade, not this one. Every value is a separate
trial, which is what `sweep` declares."""

SWEEP_ENTRY_MINUTES: Final = (5, 10, 20)
"""The entry windows measured, and the whole declared search.

Three trials on the one axis that distinguishes this rule from the intraday fade:
how much of the session is left. The band width is **not** swept — it is shared
with `vwap_band_reversion`, which already spends three trials on it, and
searching it twice would deflate the same axis against two families."""

DEFAULT_BAND_SIGMAS: Final = 1.5
"""Deviation from VWAP, in session dispersions, at which the push is faded.

Lower than `vwap_band_reversion`'s 2.0, and deliberately: this rule requires the
deviation to survive to within minutes of the bell, which is itself a filter that
a mid-session reading does not pass. Fixed, not swept — see `SWEEP_ENTRY_MINUTES`."""

STOP_SIGMAS: Final = 6.0
"""Where the stop sits, in session dispersions beyond the entry.

Six, which a normal overnight cannot reach — and that is the point. The labeller
walks bar to bar, so a stop the overnight gaps through would be recorded as
filling at the next session's first bar's low, which it would not have. Placing
it out of reach makes the position resolve on the time barrier at the next open,
which is the quantity the literature reports. It exists to define R, not to be
hit."""

TARGET_R: Final = Decimal(3)
"""Profit barrier in multiples of risk, also deliberately out of reach.

With the stop at six sigmas, three R is eighteen sigmas of session dispersion.
Like `intraday_momentum`'s `NO_TARGET_R`, it exists because `Intent.target_r`
must be positive, and is not a tuned parameter."""

EXIT_MINUTES_AFTER_OPEN: Final = 30
"""Minutes past the next session's open that the position is closed.

**Thirty for two reasons, and the second is not optional.** The literature puts a
third to a half of the reversal inside the first 30 minutes, so that is where the
edge is claimed to be. And `signals_from_intents` turns the horizon into a bar
count with `bisect_right(timestamps, deadline)`: `next_open_ns` is the *open
instant*, while the next session's first bar is stamped at its **close**, one
minute later. A deadline exactly at the open therefore finds no bar of the next
session at all and the time barrier collapses back to tonight's bell — measuring
the last five minutes of today instead of the overnight, which is the failure this
whole strategy exists to avoid. Anything past one minute fixes it; thirty is where
the published number is."""

MIN_DISPERSION_BARS: Final = 30
"""Bars a session needs before its dispersion is treated as one.

Same threshold and same reason as `vwap_band_reversion`: a volume-weighted
standard deviation over four prints is not a measure of where the day's volume
sits. Not binding in practice here — the entry window is minutes from the bell —
and kept so a half day with a truncated corpus cannot produce a signal."""


@arsenal.strategy
class PriorCloseReversal(Strategy):
    """Fade a close pushed away from session VWAP, exiting at the next open.

    The one strategy in the arsenal that is not flat at the bell. Declares no
    features: session VWAP and its dispersion are §5.3 shared infrastructure.

    Example:
        >>> PriorCloseReversal().qualified_name
        'prior_close_reversal@1.0.0'
        >>> PriorCloseReversal.holds_overnight
        True
    """

    name, version = "prior_close_reversal", "1.0.0"

    holds_overnight: ClassVar[bool] = True
    """The quarantine. `BacktestEngine` refuses to host this beside a day
    strategy, so the overnight family's different margin and different exit
    mechanics cannot leak into the day engine's sizing model."""

    regimes: ClassVar[tuple[Regime, ...]] = (Regime.CHOP, Regime.REVERSAL)
    """Reversion regimes. Never a trend day: a close pushed to the high of a
    trend day is continuation, and fading it overnight is the opposite trade.
    Never the lull either, though the entry window makes that moot."""

    cost_sensitivity: ClassVar[float] = 1.0
    """One round trip per signal over a hold of seventeen hours, which is the
    lowest turnover in the arsenal. §5.9's multiple is about how often the spread
    is paid against the gross move, and here it is paid once."""

    entry_minutes_before_close: int = DEFAULT_ENTRY_MINUTES_BEFORE_CLOSE
    """Minutes before the bell an entry is considered. Each value is a trial."""

    band_sigmas: float = DEFAULT_BAND_SIGMAS
    """Deviation required, in session dispersions. Shared with the intraday fade
    and not swept here."""

    target_r: ClassVar[Decimal] = TARGET_R

    def __init__(
        self,
        *,
        entry_minutes_before_close: int = DEFAULT_ENTRY_MINUTES_BEFORE_CLOSE,
        band_sigmas: float = DEFAULT_BAND_SIGMAS,
    ) -> None:
        """Start with nothing proposed on any instrument.

        Example:
            >>> PriorCloseReversal(entry_minutes_before_close=5).entry_minutes_before_close
            5
        """
        self.entry_minutes_before_close = entry_minutes_before_close
        self.band_sigmas = band_sigmas
        self._proposed: dict[Symbol, date] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per entry window in `SWEEP_ENTRY_MINUTES`.

        Example:
            >>> [label for label, _ in PriorCloseReversal.sweep()]
            ['within 5m of close', 'within 10m of close', 'within 20m of close']
        """
        return tuple(
            (f"within {value}m of close", cls(entry_minutes_before_close=value))
            for value in SWEEP_ENTRY_MINUTES
        )

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Propose a fade of the closing push, held to the next session's open.

        Returns:
            At most one proposal per instrument per session, and nothing at all
            outside the entry window, on an instrument outside
            `OVERNIGHT_INSTRUMENTS`, or when the calendar holds no next session.

        Example:
            >>> PriorCloseReversal().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.CHOP,
            ... ))
            ()
        """
        levels = context.levels
        if levels is None or context.session is not MarketSession.REGULAR:
            return _NO_TRADE
        if bar.symbol.ticker not in OVERNIGHT_INSTRUMENTS:
            return _NO_TRADE
        # The exit is a calendar fact, not an estimate. Without it the horizon
        # would have to be guessed, and a guess lands on a Saturday one night in
        # five — see the module docstring.
        if levels.next_open_ns is None or levels.next_open_ns <= bar.ts_event:
            return _NO_TRADE
        if levels.vwap is None or levels.vwap_sigma is None:
            return _NO_TRADE
        if levels.bar_count < MIN_DISPERSION_BARS or levels.vwap_sigma <= 0:
            return _NO_TRADE
        if self._proposed.get(bar.symbol) == levels.session_date:
            return _NO_TRADE

        # Inside the closing window, and strictly before the bell: a bar stamped
        # at the close is the last print and there is nothing left to enter on.
        remaining = (levels.close_ns - bar.ts_event) // _NANOS_PER_MINUTE
        if not 0 < remaining <= self.entry_minutes_before_close:
            return _NO_TRADE

        stretch = vwap_distance(bar.close, levels.vwap) / levels.vwap_sigma
        if stretch >= self.band_sigmas:
            side = Side.SELL  # pushed up into the bell; fade it
        elif stretch <= -self.band_sigmas:
            side = Side.BUY
        else:
            return _NO_TRADE

        exit_ns = levels.next_open_ns + EXIT_MINUTES_AFTER_OPEN * _NANOS_PER_MINUTE
        invalidation = _beyond(bar.close, levels.vwap_sigma, side=side)
        if invalidation == bar.close:
            # A dispersion small enough that six of them round to nothing at
            # corpus precision. R would be zero and `Intent` refuses it.
            return _NO_TRADE

        self._proposed[bar.symbol] = levels.session_date
        return (
            Intent(
                symbol=bar.symbol,
                ts_event=bar.ts_event,
                ts_init=bar.ts_event,
                side=side,
                entry=EntryTrigger.MARKET,
                entry_price=None,
                invalidation=invalidation,
                target_r=self.target_r,
                # The next open plus `EXIT_MINUTES_AFTER_OPEN`, from the
                # calendar. This is the barrier that actually resolves the
                # trade; the price barriers are out of reach on purpose. The
                # offset past the open is load-bearing, not a preference — see
                # the constant.
                horizon_ns=exit_ns - bar.ts_event,
                strategy=self.name,
                strategy_version=self.version,
                rationale=(
                    f"{remaining}m to the bell, {stretch:+.2f} sigma from vwap "
                    f"{levels.vwap}, exit {EXIT_MINUTES_AFTER_OPEN}m after the next open"
                ),
            ),
        )


def _beyond(entry: Price, sigma: float, *, side: Side) -> Price:
    """The stop, `STOP_SIGMAS` dispersions past the entry on the legal side.

    Via `str`, so the binary artefact of a float dispersion never reaches a price
    an order would be placed at.
    """
    move = Decimal(str(STOP_SIGMAS * sigma))
    factor = (1 - move) if side is Side.BUY else (1 + move)
    return Price(tidy_decimal(entry.value * factor, CORPUS_PLACES))
