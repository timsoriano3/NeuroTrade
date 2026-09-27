"""Fade a stretch away from VWAP that volume does not support — §5.3, Family B.

**The idea in one line.** Session VWAP is the price the day's volume actually
paid; a print far above it is a few buyers ahead of the crowd rather than a new
consensus, so sell it back toward the average.

**"Far" is measured in the day's own dispersion.** `SessionLevels.vwap_sigma` is
the volume-weighted standard deviation of price around VWAP, as a fraction of
it, so a two-sigma stretch means the same thing on a quiet index ETF and on a
name that has moved 4% — which an absolute distance in cents does not. §5.3
states the strategy as "fade +/-2 sigma deviations", and 2.0 is the default here
for that reason.

**Where the reversion is claimed to live.** §5.3: "strongest in the first 90 and
final 60 minutes when institutional flow concentrates". That window is the
filter, not a preference — a fade held through the midday lull is the trade this
module is least entitled to make, and the lull is a no-trade regime besides.

**The counterpart to `intraday_momentum`, structurally.** That strategy declares
TREND_UP / TREND_DOWN / HIGH_VOLATILITY; this one declares CHOP and REVERSAL, so
the host can never have both live at once. They are the same observation read
two ways — a stretched price is a breakout on a trend day and an overshoot on a
choppy one — and the regime classifier, not the strategies, is what decides
which reading applies. Until Phase 5 that decision does not exist, which is
exactly why a Phase 2 number from either has to be stamped `ungated`.

**What is missing against the plan.** Phase 2's plan names this strategy
"VWAP band reversion, VIX-gated" and the sweep found no academic source for it —
it is the weakest of the six Tier 1 candidates, kept because §5.3 names it and
because it exercises the levels module. The VIX gate is **not implemented**: the
corpus holds equities and ETFs, and VIX is an index the data plan does not
ingest, so gating on it would mean either inventing a proxy or reading a series
that is not there. VXX is in the universe but it is a futures-roll ETN, not the
index. Stated rather than approximated, because a volatility gate built out of a
proxy is a modelling claim wearing a published rule's clothes.
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

__all__ = ["VwapBandReversion"]

_NO_TRADE: Final[tuple[Intent, ...]] = ()

_NANOS_PER_MINUTE: Final = 60_000_000_000

DEFAULT_BAND_SIGMAS: Final = 2.0
"""Deviation from VWAP, in session dispersions, at which a fade is taken.

Two because that is the number §5.3 states the strategy in. It is also where the
trade's arithmetic works out: the stop sits one dispersion beyond the band, so
the distance back to VWAP is exactly `band_sigmas` times the risk."""

SWEEP_BAND_SIGMAS: Final = (1.5, 2.0, 2.5)
"""The band widths measured, and the whole declared search.

Three trials on the only axis the spec states: 2.0 is its number, and 1.5 and
2.5 bracket it. The session windows are not swept — they come from §5.3 as
written, and a fourth and fifth value tried beside them would raise the
deflation hurdle for every strategy in the family (§17)."""

MIN_DISPERSION_BARS: Final = 30
"""Bars a session needs before its dispersion is treated as one.

A volume-weighted standard deviation over four prints is not a measure of where
the day's volume sits, and the first bars of a session are its widest — quoting
a band off them would fire the strategy hardest exactly where the number means
least. Thirty is the first opening-range window in `OPENING_WINDOWS`, so the
earliest entry is half an hour in."""

OPENING_WINDOW_MINUTES: Final = 90
"""Minutes after the open during which a fade is taken (§5.3's "first 90")."""

CLOSING_WINDOW_MINUTES: Final = 60
"""Minutes before the close during which a fade is taken (§5.3's "final 60")."""


@arsenal.strategy
class VwapBandReversion(Strategy):
    """Fade a price stretched away from session VWAP, back toward it.

    Reads only session levels, so it declares no features: VWAP and its
    dispersion are §5.3 shared infrastructure and arrive on every context.

    Example:
        >>> VwapBandReversion().qualified_name
        'vwap_band_reversion@1.0.0'
    """

    name, version = "vwap_band_reversion", "1.0.0"

    regimes: ClassVar[tuple[Regime, ...]] = (Regime.CHOP, Regime.REVERSAL)
    """The days a stretch comes back. Never TREND_UP, TREND_DOWN or
    HIGH_VOLATILITY, where the same observation is `intraday_momentum`'s entry
    and fading it is standing in front of the day's move; never the lull."""

    cost_sensitivity: ClassVar[float] = 1.5
    """Higher than a trend trade's. The gross move being collected is one band
    width — a fraction of a percent on an index ETF — and it is collected in
    under an hour, so the spread is a larger share of it (§5.9)."""

    band_sigmas: float = DEFAULT_BAND_SIGMAS
    """Deviation from VWAP, in session dispersions, at which a fade is taken.
    Each distinct value is a separate trial, which is what `sweep` declares. Not
    a `ClassVar`: an instance carries its own, so one run can measure several."""

    def __init__(self, *, band_sigmas: float = DEFAULT_BAND_SIGMAS) -> None:
        """Start with nothing proposed on any instrument.

        Args:
            band_sigmas: Deviation from VWAP, in session dispersions, at which a
                fade is taken. Each value is its own trial; see `sweep`.

        Example:
            >>> VwapBandReversion(band_sigmas=2.5).band_sigmas
            2.5
        """
        self.band_sigmas = band_sigmas
        # Per symbol, because the host subscribes one instance to the whole
        # universe; the date is carried so a new session starts clean.
        self._proposed: dict[Symbol, tuple[date, Side]] = {}

    @classmethod
    def sweep(cls) -> tuple[tuple[str, Self], ...]:
        """One variant per band width in `SWEEP_BAND_SIGMAS`.

        Example:
            >>> [label for label, _ in VwapBandReversion.sweep()]
            ['1.5 sigma', '2 sigma', '2.5 sigma']
        """
        return tuple((f"{value:g} sigma", cls(band_sigmas=value)) for value in SWEEP_BAND_SIGMAS)

    def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
        """Propose a fade back toward VWAP, or nothing.

        Args:
            bar: The bar that just closed.
            context: Its view of the world; `levels` carries everything read.

        Returns:
            At most one proposal, and nothing outside the two session windows
            §5.3 concentrates the effect in.

        Example:
            >>> VwapBandReversion().on_bar(a_bar, StrategyContext(
            ...     symbol=AAPL, as_of=1_000, session=MarketSession.REGULAR,
            ...     regime=Regime.CHOP,
            ... ))
            ()
        """
        levels = context.levels
        if levels is None or context.session is not MarketSession.REGULAR:
            return _NO_TRADE
        if bar.ts_event >= levels.close_ns:
            return _NO_TRADE
        vwap, sigma = levels.vwap, levels.vwap_sigma
        if vwap is None or sigma is None or sigma <= 0.0:
            return _NO_TRADE
        if levels.bar_count < MIN_DISPERSION_BARS:
            return _NO_TRADE

        minute = (bar.ts_event - levels.open_ns) // _NANOS_PER_MINUTE
        minutes_left = (levels.close_ns - bar.ts_event) // _NANOS_PER_MINUTE
        if minute > OPENING_WINDOW_MINUTES and minutes_left > CLOSING_WINDOW_MINUTES:
            return _NO_TRADE

        band = self.band_sigmas * sigma
        deviation = vwap_distance(bar.close, vwap)
        # The stop is one dispersion beyond the band: the band's own unit is the
        # natural distance at which "stretched" becomes "wrong", and it makes R
        # the unit the signal is measured in, so the reward-to-risk ratio is the
        # band width rather than a number chosen to look good.
        stop_fraction = band + sigma
        if not 0.0 < stop_fraction < 1.0:
            return _NO_TRADE
        if deviation >= band:
            side = Side.SELL
            invalidation = _scaled(vwap, 1 + stop_fraction)
        elif deviation <= -band:
            side = Side.BUY
            invalidation = _scaled(vwap, 1 - stop_fraction)
        else:
            return _NO_TRADE

        # One fade per direction per session: a second stretch the same way is
        # the same hypothesis, and would let one session dominate a backtest.
        if self._proposed.get(bar.symbol) == (levels.session_date, side):
            return _NO_TRADE
        self._proposed[bar.symbol] = (levels.session_date, side)

        return (
            Intent(
                symbol=bar.symbol,
                ts_event=bar.ts_event,
                ts_init=bar.ts_event,
                side=side,
                entry=EntryTrigger.MARKET,
                entry_price=None,
                invalidation=invalidation,
                # VWAP is `band_sigmas` dispersions away and the stop is one, so
                # the target is the mean itself when the fill is at the band. A
                # fill past it shrinks R and pulls the target inside VWAP, which
                # is the conservative direction.
                target_r=Decimal(str(self.band_sigmas)),
                horizon_ns=levels.close_ns - bar.ts_event,
                strategy=self.name,
                strategy_version=self.version,
                rationale=(
                    f"{deviation:+.4f} from vwap {vwap}, {deviation / sigma:+.2f} session sigmas"
                ),
            ),
        )


def _scaled(price: Price, factor: float) -> Price:
    """`price * factor`, at the corpus scale.

    Via `str`: the factor is built from a float feature, and `Decimal(float)`
    would carry the binary artefact of it into a level a stop is read from.
    """
    return Price(tidy_decimal(price.value * Decimal(str(factor)), CORPUS_PLACES))
