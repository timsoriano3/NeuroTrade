"""The controls are fixtures with known answers — these tests pin the answers.

A control that quietly stopped being what it claims would turn the exit gate
into a rubber stamp, so the properties the gate relies on are asserted here
directly: the walk has no drift, the planted series does, and both are
reproducible from their seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import pairwise
from statistics import fmean

import pytest

from neurotrade.core.events import Bar, MarketSession
from neurotrade.core.types import Side
from neurotrade.features.registry import LookaheadError
from neurotrade.lab.controls import (
    CONTROL_SYMBOL,
    HONEST_GRID,
    SNOOPED_GRID,
    Crossover,
    CrossoverParams,
    momentum_bars,
    random_walk_bars,
    sma_spec,
)
from neurotrade.strategies.base import Regime, StrategyContext, StrategyRegistry

_Generator = Callable[..., tuple[Bar, ...]]


def _log_returns(bars: Sequence[Bar]) -> list[float]:
    return [
        math.log(float(later.close.value) / float(earlier.close.value))
        for earlier, later in pairwise(bars)
    ]


def _context(bar: Bar, fast: float | None, slow: float | None) -> StrategyContext:
    return StrategyContext(
        symbol=CONTROL_SYMBOL,
        as_of=bar.ts_event,
        session=MarketSession.REGULAR,
        regime=Regime.TREND_UP,
        values={"sma_fast": fast, "sma_slow": slow},
    )


# ── Synthetic series ─────────────────────────────────────────────────────


@pytest.mark.parametrize("generator", [random_walk_bars, momentum_bars])
def test_series_are_reproducible_from_the_seed(generator: _Generator) -> None:
    first = generator(seed=11, n_bars=200)
    second = generator(seed=11, n_bars=200)
    assert [bar.close for bar in first] == [bar.close for bar in second]


@pytest.mark.parametrize("generator", [random_walk_bars, momentum_bars])
def test_different_seeds_give_different_series(generator: _Generator) -> None:
    assert [b.close for b in generator(seed=11, n_bars=200)] != [
        b.close for b in generator(seed=12, n_bars=200)
    ]


@pytest.mark.parametrize("generator", [random_walk_bars, momentum_bars])
def test_bars_satisfy_the_ohlc_invariants(generator: _Generator) -> None:
    """A high inside the body would let a barrier never trip, or trip early."""
    for bar in generator(seed=3, n_bars=500):
        assert bar.high >= max(bar.open, bar.close)
        assert bar.low <= min(bar.open, bar.close)
        assert bar.high > bar.low  # a zero-range bar has no inside to touch


def test_the_walk_has_no_drift_in_sample() -> None:
    """Not merely zero in expectation — zero in *this* realisation.

    A residual drift is a real feature of the sample that a directional rule
    picks up in every subsample, which is what made PBO report a stable
    ranking on data that is supposed to have nothing in it.

    Not exactly zero: the closes are quantised to a penny, so the returns read
    back off the bars carry the rounding. The bound is a hundredth of a
    per-bar standard deviation — three orders of magnitude below anything a
    thirty-bar rule could trade.
    """
    volatility = 0.004  # the generator's default
    for seed in (1, 2, 3, 20260922):
        returns = _log_returns(random_walk_bars(seed=seed, n_bars=2000))
        assert abs(fmean(returns)) < volatility / 100


def test_the_planted_series_actually_trends() -> None:
    """The positive control is only a positive control if the edge is there.

    Autocorrelation is measured at lag 20 — the scale a moving-average
    crossover reads. A series correlated only at lag 1 looks like momentum and
    is invisible to the rule being tested, which is exactly the trap this
    generator was rewritten to avoid.
    """
    returns = _log_returns(momentum_bars(seed=5, n_bars=4000))
    mean = fmean(returns)
    centred = [value - mean for value in returns]
    lag = 20
    numerator = math.fsum(a * b for a, b in zip(centred, centred[lag:], strict=False))
    denominator = math.fsum(value * value for value in centred)
    # Measured 0.047 at this seed. The bar is set below that rather than at it:
    # the assertion is "the edge is there at the scale the rule reads", and
    # pinning the exact figure would fail on the next tweak to the generator
    # without telling anyone anything.
    assert numerator / denominator > 0.02


@pytest.mark.parametrize("generator", [random_walk_bars, momentum_bars])
def test_a_series_needs_at_least_two_bars(generator: _Generator) -> None:
    with pytest.raises(ValueError, match="n_bars must be at least 2"):
        generator(seed=1, n_bars=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"volatility": 0.0}, "volatility must be positive"),
        ({"persistence": 1.0}, r"persistence must lie in \(-1, 1\)"),
        ({"drift_scale": -0.1}, "drift_scale must be positive"),
    ],
)
def test_the_planted_series_rejects_incoherent_parameters(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        momentum_bars(seed=1, n_bars=10, **kwargs)


# ── The moving average ───────────────────────────────────────────────────


def test_the_average_is_none_until_its_window_fills() -> None:
    bars = random_walk_bars(seed=1, n_bars=10)
    spec = sma_spec(5)
    assert spec.evaluate(bars[:4], as_of=bars[3].ts_event) is None
    assert spec.evaluate(bars[:5], as_of=bars[4].ts_event) is not None


def test_the_average_refuses_a_bar_from_the_future() -> None:
    """The PIT guard §20.4 asks for, exercised through the control's own feature."""
    bars = random_walk_bars(seed=1, n_bars=10)
    with pytest.raises(LookaheadError, match="after the moment being modelled"):
        sma_spec(5).evaluate(bars[:5], as_of=bars[0].ts_event)


def test_the_average_does_not_depend_on_how_much_history_was_sliced() -> None:
    """A smoothed indicator carrying state across windows breaks reproducibility."""
    bars = random_walk_bars(seed=1, n_bars=200)
    spec = sma_spec(20)
    tight = spec.evaluate(bars[80:100], as_of=bars[99].ts_event)
    generous = spec.evaluate(bars[:100], as_of=bars[99].ts_event)
    assert tight == generous


# ── The strategy ─────────────────────────────────────────────────────────


def test_warming_up_produces_no_intent() -> None:
    """`None` means not yet computable, and must never be read as a signal."""
    bar = random_walk_bars(seed=1, n_bars=2)[0]
    strategy = Crossover(params=CrossoverParams(fast=2, slow=4))
    assert strategy.on_bar(bar, _context(bar, None, 1.0)) == ()
    assert strategy.on_bar(bar, _context(bar, 1.0, None)) == ()


def test_a_tie_is_not_a_view() -> None:
    bar = random_walk_bars(seed=1, n_bars=2)[0]
    strategy = Crossover(params=CrossoverParams(fast=2, slow=4))
    assert strategy.on_bar(bar, _context(bar, 1.0, 1.0)) == ()


@pytest.mark.parametrize(
    ("fade", "fast", "slow", "expected"),
    [
        (False, 2.0, 1.0, Side.BUY),
        (False, 1.0, 2.0, Side.SELL),
        (True, 2.0, 1.0, Side.SELL),
        (True, 1.0, 2.0, Side.BUY),
    ],
)
def test_the_fade_flag_mirrors_the_side(
    fade: bool, fast: float, slow: float, expected: Side
) -> None:
    """The dimension that gives the snooped search a sign to pick by luck."""
    bar = random_walk_bars(seed=1, n_bars=2)[0]
    strategy = Crossover(params=CrossoverParams(fast=2, slow=4, fade=fade))
    assert strategy.on_bar(bar, _context(bar, fast, slow))[0].side is expected


def test_the_stop_sits_on_the_losing_side_for_both_directions() -> None:
    bar = random_walk_bars(seed=1, n_bars=2)[0]
    strategy = Crossover(params=CrossoverParams(fast=2, slow=4))
    assert strategy.on_bar(bar, _context(bar, 2.0, 1.0))[0].invalidation < bar.close
    assert strategy.on_bar(bar, _context(bar, 1.0, 2.0))[0].invalidation > bar.close


def test_the_control_registers_as_a_plugin() -> None:
    """It honours the plugin contract — into a local registry, never a shared one."""
    registry = StrategyRegistry()
    registry.strategy(Crossover)
    assert registry.get("control-crossover") is Crossover


# ── The grids ────────────────────────────────────────────────────────────


def test_the_snooped_grid_pairs_every_variant_with_its_mirror() -> None:
    assert len(SNOOPED_GRID) == 70
    assert len({params.label for params in SNOOPED_GRID}) == 70
    faded = {(p.fast, p.slow) for p in SNOOPED_GRID if p.fade}
    plain = {(p.fast, p.slow) for p in SNOOPED_GRID if not p.fade}
    assert faded == plain


def test_the_honest_grid_is_small_and_declares_one_direction() -> None:
    """Trying both directions would be the start of the snoop, not an a-priori view."""
    assert len(HONEST_GRID) == 4
    assert not any(params.fade for params in HONEST_GRID)


@pytest.mark.parametrize(
    ("fast", "slow", "message"),
    [
        (0, 10, "windows must be at least 1"),
        (10, 0, "windows must be at least 1"),
        (10, 10, "fast must be shorter than slow"),
        (20, 10, "fast must be shorter than slow"),
    ],
)
def test_incoherent_windows_are_refused(fast: int, slow: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CrossoverParams(fast=fast, slow=slow)
