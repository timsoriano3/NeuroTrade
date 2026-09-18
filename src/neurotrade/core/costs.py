"""What a trade costs: spread, commission and modelled slippage (§8, §3.3).

**Why this is in `core` and not in `lab`.** Costs belong to the backtest — §3.3
is explicit that spread, fees and slippage are applied *inside* the simulation
and never subtracted afterwards. But they belong to the live engine too: §3.3's
"every signal must beat its own cost" is a decision made before sending an
order, and §8 recalibrates the model nightly from real fills. `lab/` cannot be
imported by `execution/` under the layering contract, so a cost model there
would have to be duplicated — and a cost model that differs between research
and live makes every backtest a claim about software that is not the software
trading. One implementation, in the one layer both sides can reach.

## The three components, and why they are separate

- **Spread** — the gap between what a buyer pays and a seller receives. Cross
  it and you have lost half of it immediately, before the price has moved.
- **Commission** — the broker's fee. Known exactly, and the only one of the
  three that is not an estimate.
- **Slippage** — the price moves against you while you fill. Scales with how
  much of the day's volume you are trying to take and with how volatile the
  name is.

They are reported separately rather than as one number because they behave
differently as size grows: commission is nearly linear, spread is linear, and
slippage is super-linear. A strategy that looks profitable at 100 shares and
fails at 10,000 fails through the third term, and a single total would hide
which one moved.

## Spread is a port, because the corpus has no quotes

Minute bars carry no bid or ask, so a backtest over this corpus cannot measure
a spread — it has to estimate one. That estimate will be replaced (by real
`BID_ASK` history if IBKR serves it, by captured quotes once the live engine
runs), so it sits behind `SpreadSource` rather than being wired in. Live passes
the real quote; research passes an estimator.

**Estimates are floored, never used raw.** `FlooredSpread` takes the greater of
the estimate, one tick, and a per-liquidity-bucket minimum. The bias is
deliberately toward over-stating cost: §17 names the optimistic backtest as the
project's primary risk, so a strategy wrongly rejected is the cheaper error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, Protocol, runtime_checkable

from neurotrade.core.types import Currency, Money, Price, Quantity, Side

__all__ = [
    "CostModel",
    "FeeSchedule",
    "FlooredSpread",
    "SpreadSource",
    "TradeCost",
]

_CENT: Final = Decimal("0.01")
"""The tick above $1.00 on every North American equity venue."""

_SUB_DOLLAR_TICK: Final = Decimal("0.0001")
"""The tick below $1.00. A sub-dollar name quotes in hundredths of a cent, so
flooring its spread at a penny would overstate its cost by two orders of
magnitude."""

_MONEY_PLACES: Final = Decimal("0.00000001")
"""Scale every money amount is quantized to — the corpus decimal scale."""


def tick_size(price: Price) -> Decimal:
    """The minimum price increment at a given price level.

    Args:
        price: The price to size a tick at.

    Returns:
        `0.0001` below $1.00, `0.01` at or above it — SEC Rule 612's
        sub-penny threshold, which the Canadian venues match.

    Example:
        >>> (tick_size(Price("0.50")), tick_size(Price("100")))
        (Decimal('0.0001'), Decimal('0.01'))
    """
    return _SUB_DOLLAR_TICK if price.value < 1 else _CENT


@runtime_checkable
class SpreadSource(Protocol):
    """Where the bid-ask spread for an instrument comes from.

    Three implementations over this project's life: the real quote in live, an
    estimate in research now, and real historical `BID_ASK` bars once the
    crawler can fetch them. Keeping it a port is what lets the third replace the
    second without touching the cost model or anything that calls it.

    Example:
        Conformance is structural — no import from this module is needed:

        >>> class PennyWide:
        ...     def spread(self, price, volatility=None):
        ...         return Decimal("0.01")
        >>> isinstance(PennyWide(), SpreadSource)
        True
    """

    def spread(self, price: Price, volatility: float | None = None) -> Decimal:
        """The full bid-ask spread in price units.

        Args:
            price: The instrument's price level.
            volatility: Recent realised volatility as a fraction, when the
                caller has it. Spreads widen with volatility, so an estimator
                that has this can use it.

        Returns:
            The full spread, not the half. Crossing costs half of it per side,
            and `CostModel` halves it once so no caller has to remember.
        """
        ...


@dataclass(frozen=True, slots=True)
class FlooredSpread:
    """A spread estimate that is never allowed below a floor.

    The research default. `fraction` is applied to the price to give a
    proportional estimate, and the result is floored at one tick and at
    `minimum` — so a thinly traded name never prices as though it were SPY.

    **The floors matter more than the estimate.** High/low spread estimators
    (Corwin-Schultz, Abdi-Ranaldo) produce negative values a meaningful share
    of the time on intraday data, and a negative spread is a *subsidy* for
    trading. Whatever estimator eventually supplies `fraction`, the floor is
    what keeps the result honest.

    Example:
        >>> wide = FlooredSpread(fraction=Decimal("0.0005"), minimum=Decimal("0.01"))
        >>> wide.spread(Price("100"))
        Decimal('0.05')
        >>> wide.spread(Price("2"))
        Decimal('0.01')
    """

    fraction: Decimal = Decimal("0.0005")  # proportional estimate: 5 bps of price
    minimum: Decimal = Decimal("0.01")  # per-liquidity-bucket floor, in price units
    volatility_multiplier: Decimal = Decimal("0")  # extra spread per unit of realised vol

    def __post_init__(self) -> None:
        """Validate the estimator.

        Raises:
            ValueError: If any term is negative. A negative spread would pay
                the strategy to trade, which is the single most effective way
                to manufacture a backtest edge that does not exist.
        """
        for name, value in (
            ("fraction", self.fraction),
            ("minimum", self.minimum),
            ("volatility_multiplier", self.volatility_multiplier),
        ):
            if value < 0:
                raise ValueError(f"{name} {value} must not be negative")

    def spread(self, price: Price, volatility: float | None = None) -> Decimal:
        """The estimated spread, floored.

        Args:
            price: The instrument's price level.
            volatility: Recent realised volatility as a fraction. Ignored when
                `volatility_multiplier` is zero, which is the default.

        Returns:
            The greater of the proportional estimate, one tick, and `minimum`.
        """
        estimate = price.value * self.fraction
        if volatility is not None and self.volatility_multiplier > 0:
            estimate += price.value * self.volatility_multiplier * Decimal(repr(volatility))
        # Quantized because a proportional estimate carries the trailing zeros
        # of the multiplication (100 * 0.0005 is `0.0500`, not `0.05`) and this
        # value is reported as well as multiplied.
        return max(estimate, tick_size(price), self.minimum).quantize(_MONEY_PLACES).normalize()


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """A broker's commission terms.

    Defaults are IBKR's US fixed tier as of 2026: half a cent per share, a
    dollar minimum per order, capped at 1% of trade value. The cap is what
    stops the minimum from dominating a small order in a cheap stock — 100
    shares of a $2 name is $200 of value, and a $1 minimum would be 0.5% of it
    before the spread.

    Example:
        >>> FeeSchedule().commission(Quantity(100), Price("50")).amount
        Decimal('1.00000000')
    """

    per_share: Decimal = Decimal("0.005")  # commission per share
    minimum: Decimal = Decimal("1.00")  # floor per order
    maximum_fraction: Decimal = Decimal("0.01")  # cap as a fraction of trade value
    currency: Currency = Currency.USD  # the currency these amounts are in

    def __post_init__(self) -> None:
        """Validate the schedule.

        Raises:
            ValueError: If any term is negative.
        """
        for name, value in (
            ("per_share", self.per_share),
            ("minimum", self.minimum),
            ("maximum_fraction", self.maximum_fraction),
        ):
            if value < 0:
                raise ValueError(f"{name} {value} must not be negative")

    def commission(self, quantity: Quantity, price: Price) -> Money:
        """Commission on one fill.

        Args:
            quantity: Shares filled.
            price: Fill price.

        Returns:
            The fee, floored at `minimum` and capped at `maximum_fraction` of
            trade value. A zero-share fill costs nothing — the minimum applies
            to an order that traded, not to one that did not.
        """
        if quantity.is_zero:
            return Money.zero(self.currency)
        value = price.value * quantity.value
        raw = max(self.per_share * quantity.value, self.minimum)
        capped = min(raw, value * self.maximum_fraction)
        return Money(capped.quantize(_MONEY_PLACES), self.currency)


@dataclass(frozen=True, slots=True)
class TradeCost:
    """What one fill cost, broken into its three parts.

    Example:
        >>> cost = TradeCost(
        ...     spread=Money(Decimal("2.50"), Currency.USD),
        ...     commission=Money(Decimal("1.00"), Currency.USD),
        ...     slippage=Money(Decimal("0.50"), Currency.USD),
        ... )
        >>> str(cost.total)
        '4.00 USD'
    """

    spread: Money  # half the quoted spread, times size — paid on crossing
    commission: Money  # the broker's fee
    slippage: Money  # modelled adverse move while filling

    @property
    def total(self) -> Money:
        """The three parts summed."""
        return self.spread + self.commission + self.slippage

    def per_share(self, quantity: Quantity) -> Decimal:
        """Total cost divided by shares, for comparing against an edge.

        Raises:
            ValueError: If `quantity` is zero — per-share cost is undefined
                with no shares, and returning zero would read as "free".
        """
        if quantity.is_zero:
            raise ValueError("per-share cost is undefined for a zero-quantity fill")
        return (self.total.amount / quantity.value).quantize(_MONEY_PLACES)


@dataclass(frozen=True, slots=True)
class CostModel:
    """Spread, commission and slippage for a fill.

    Example:
        >>> model = CostModel(spreads=FlooredSpread(), fees=FeeSchedule())
        >>> cost = model.cost(Side.BUY, Quantity(100), Price("50"))
        >>> (str(cost.spread), str(cost.commission))
        ('1.25000000 USD', '1.00000000 USD')
    """

    spreads: SpreadSource  # where the spread comes from
    fees: FeeSchedule  # the broker's terms
    impact_coefficient: Decimal = Decimal("0.1")  # slippage scaling; see `cost`

    def cost(
        self,
        side: Side,
        quantity: Quantity,
        price: Price,
        *,
        volatility: float | None = None,
        average_volume: Quantity | None = None,
        is_maker: bool = False,
    ) -> TradeCost:
        """Model the cost of one fill.

        **Spread.** Half the quoted spread per side. A resting order that is
        filled by someone else crossing — `is_maker` — *earns* the half spread
        rather than paying it, so its spread cost is zero here rather than
        negative: modelling a rebate as a cost reduction invites a strategy
        whose entire edge is the modelled rebate.

        **Slippage.** `impact_coefficient * volatility * sqrt(participation)`,
        where participation is the order's size as a fraction of average
        volume. The square root is the standard market-impact shape — impact
        grows with size but less than proportionally. With no `average_volume`
        there is no participation to compute and slippage is zero, which is
        optimistic and stated rather than hidden.

        Args:
            side: Buy or sell. Does not change the magnitude, only the sign of
                the price effect, which the caller applies.
            quantity: Shares filled.
            price: Reference price, taken as the midpoint.
            volatility: Recent realised volatility as a fraction.
            average_volume: Typical volume over the relevant window, for
                participation.
            is_maker: Whether this fill added liquidity rather than taking it.

        Returns:
            The three components. Every one is a positive cost.

        Raises:
            ValueError: If `average_volume` is zero — participation against no
                volume is undefined, and treating it as zero slippage would
                make the most illiquid names look the cheapest to trade.
        """
        currency = self.fees.currency
        if quantity.is_zero:
            zero = Money.zero(currency)
            return TradeCost(spread=zero, commission=zero, slippage=zero)

        full_spread = self.spreads.spread(price, volatility)
        half_spread = Decimal(0) if is_maker else full_spread / 2
        spread_cost = Money((half_spread * quantity.value).quantize(_MONEY_PLACES), currency)

        slippage = Money.zero(currency)
        if average_volume is not None:
            if average_volume.is_zero:
                raise ValueError("participation is undefined against zero average volume")
            if volatility is not None and volatility > 0:
                participation = float(quantity.value / average_volume.value)
                move = (
                    self.impact_coefficient
                    * Decimal(repr(volatility))
                    * Decimal(repr(math.sqrt(participation)))
                )
                slippage = Money(
                    (move * price.value * quantity.value).quantize(_MONEY_PLACES), currency
                )

        return TradeCost(
            spread=spread_cost,
            commission=self.fees.commission(quantity, price),
            slippage=slippage,
        )
