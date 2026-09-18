"""Tests for the registered indicator features.

Heavier on the properties that make a feature trustworthy than on arithmetic.
An ATR that is off by a rounding error costs nothing; an ATR that depends on
history outside its declared window makes the backtest and the live engine
disagree about the same bar, and nothing downstream would notice.
"""

from __future__ import annotations

import math

import pytest

from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.features.indicators import (
    FRACDIFF_D,
    atr,
    ema,
    fracdiff_close,
    fracdiff_weights,
    indicators,
    log_return,
    realised_volatility,
    relative_volume,
    true_range,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MINUTE = 60_000_000_000


def bar(
    index: int,
    close: str,
    *,
    high: str | None = None,
    low: str | None = None,
    volume: str = "1000",
) -> Bar:
    """A bar whose high/low default to its close, so ranges are zero unless set."""
    price = Price(close)
    ts = index * MINUTE
    return Bar(
        symbol=AAPL,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=price,
        high=Price(high) if high else price,
        low=Price(low) if low else price,
        close=price,
        volume=Quantity(volume),
    )


def flat(count: int, close: str = "100", volume: str = "1000") -> list[Bar]:
    return [bar(i, close, volume=volume) for i in range(count)]


# ── Registration ─────────────────────────────────────────────────────────────


def test_every_feature_is_registered_with_a_version() -> None:
    names = set(indicators.names())
    assert {"log_return", "atr", "realised_vol", "rvol", "ema", "fracdiff_close"} <= names


@pytest.mark.parametrize(
    "name", ["log_return", "atr", "realised_vol", "rvol", "ema", "fracdiff_close"]
)
def test_a_feature_refuses_a_window_that_is_still_warming_up(name: str) -> None:
    """A 20-bar average from 8 bars is a different number that looks plausible."""
    spec = indicators.get(name)
    short = flat(spec.lookback - 1)
    assert spec.evaluate(short, as_of=short[-1].ts_event) is None


@pytest.mark.parametrize(
    "name", ["log_return", "atr", "realised_vol", "rvol", "ema", "fracdiff_close"]
)
def test_a_feature_refuses_a_bar_stamped_after_the_moment(name: str) -> None:
    """The lookahead guard is the whole reason `evaluate` takes `as_of`."""
    from neurotrade.features.registry import LookaheadError

    spec = indicators.get(name)
    window = flat(spec.lookback)
    with pytest.raises(LookaheadError):
        spec.evaluate(window, as_of=window[-1].ts_event - 1)


@pytest.mark.parametrize(
    "name", ["log_return", "atr", "realised_vol", "rvol", "ema", "fracdiff_close"]
)
def test_every_feature_returns_a_plain_float(name: str) -> None:
    """Model inputs are floats; a Decimal leaking through changes behaviour."""
    spec = indicators.get(name)
    window = flat(spec.lookback)
    value = spec.evaluate(window, as_of=window[-1].ts_event)
    assert type(value) is float


# ── true_range ───────────────────────────────────────────────────────────────


def test_true_range_uses_the_bar_range_when_there_is_no_gap() -> None:
    assert true_range(bar(0, "100"), bar(1, "100", high="102", low="99")) == 3.0


def test_true_range_uses_the_gap_when_the_bar_opens_away() -> None:
    """A name that gaps 5% then trades tight had a violent day."""
    assert true_range(bar(0, "100"), bar(1, "105", high="106", low="105")) == 6.0


# ── atr and ema: reproducibility from the declared window ───────────────────


def test_atr_depends_only_on_the_window_it_is_given() -> None:
    """Wilder's smoothing would make this fail — and make backtests unreproducible."""
    spec = indicators.get("atr")
    window = [bar(i, "100", high="101", low="99") for i in range(spec.lookback)]
    with_prefix = [*flat(50, "42"), *window]
    assert atr(window) == atr(with_prefix[-spec.lookback :])


def test_ema_depends_only_on_the_window_it_is_given() -> None:
    spec = indicators.get("ema")
    window = [bar(i, str(100 + i)) for i in range(spec.lookback)]
    with_prefix = [*flat(50, "42"), *window]
    assert ema(window) == ema(with_prefix[-spec.lookback :])


def test_atr_of_flat_bars_is_zero() -> None:
    assert atr(flat(15)) == 0.0


def test_ema_of_a_constant_series_is_that_constant() -> None:
    assert ema(flat(20, "100")) == pytest.approx(100.0)


def test_ema_lags_a_rising_series() -> None:
    """An average of the window must sit below its own latest value."""
    rising = [bar(i, str(100 + i)) for i in range(20)]
    assert ema(rising) < float(rising[-1].close.value)


# ── log_return ───────────────────────────────────────────────────────────────


def test_log_return_is_zero_when_the_price_does_not_move() -> None:
    assert log_return(flat(2)) == 0.0


def test_log_returns_add_across_time() -> None:
    """The property logs are chosen for: two one-bar returns sum to the two-bar one."""
    first = log_return([bar(0, "100"), bar(1, "110")])
    second = log_return([bar(1, "110"), bar(2, "121")])
    whole = log_return([bar(0, "100"), bar(2, "121")])
    assert first + second == pytest.approx(whole)


# ── realised volatility ──────────────────────────────────────────────────────


def test_realised_volatility_of_a_flat_series_is_zero() -> None:
    assert realised_volatility(flat(21)) == 0.0


def test_realised_volatility_rises_with_larger_moves() -> None:
    def alternating(step: int) -> list[Bar]:
        return [bar(i, str(100 + (step if i % 2 else 0))) for i in range(21)]

    assert realised_volatility(alternating(1)) < realised_volatility(alternating(5))


# ── relative volume ──────────────────────────────────────────────────────────


def test_relative_volume_is_one_for_an_ordinary_bar() -> None:
    assert relative_volume(flat(21)) == pytest.approx(1.0)


def test_relative_volume_reports_a_volume_spike() -> None:
    window = [*flat(20), bar(20, "100", volume="3000")]
    assert relative_volume(window) == pytest.approx(3.0)


def test_relative_volume_is_zero_when_nothing_traded_in_the_baseline() -> None:
    """A real state for an illiquid name; the alternative is a division by zero."""
    window = [*flat(20, volume="0"), bar(20, "100", volume="500")]
    assert relative_volume(window) == 0.0


# ── fractional differencing ──────────────────────────────────────────────────


def test_fracdiff_weights_of_order_one_are_a_plain_first_difference() -> None:
    assert fracdiff_weights(1.0, 3) == (1.0, -1.0, 0.0)


def test_fracdiff_weights_of_order_zero_leave_the_series_untouched() -> None:
    weights = fracdiff_weights(0.0, 4)
    assert weights[0] == 1.0
    assert all(weight == 0.0 for weight in weights[1:])


def test_fracdiff_weights_alternate_and_decay() -> None:
    """Decay is what lets a fixed window approximate an infinite series."""
    weights = fracdiff_weights(FRACDIFF_D, 32)
    assert weights[0] == 1.0
    assert all(weight < 0 for weight in weights[1:6])
    assert abs(weights[-1]) < abs(weights[1])


def test_fracdiff_weights_reject_a_zero_width() -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        fracdiff_weights(0.4, 0)


def test_fracdiff_retains_the_level_unlike_a_plain_return() -> None:
    """The point of a fractional order (§8): memory of the level survives.

    A plain first difference gives the same answer for a $400 series and a $40
    one that move identically in proportion. Fractional differencing does not.
    """
    high = fracdiff_close([bar(i, str(400 + i)) for i in range(32)])
    low = fracdiff_close([bar(i, str(40 + i * 0.1)) for i in range(32)])
    assert not math.isclose(high, low)
