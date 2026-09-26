"""The measurement harness: what a strategy proposed, scored honestly.

Three defect classes get the weight here, because each one produces a *better*
number rather than an error. A conversion that misreads an `Intent`'s barriers
labels a trade nobody took. A candidate set that keeps a decision for one
variant and drops it for another makes the columns incomparable, and every
statistic downstream silently assumes they are not. And expectancy measured per
*observation* rather than per trade divides a sparse strategy's edge by the bars
it sat out, which flatters or buries it depending on the direction of the zeros.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from neurotrade.core.clock import SimClock
from neurotrade.core.costs import CostModel, FeeSchedule, FlooredSpread
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.trials import Trial, TrialSource
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.lab.controls import momentum_bars
from neurotrade.lab.cv import CombinatorialPurgedCV
from neurotrade.lab.evaluation import (
    Observations,
    Signal,
    Variant,
    assess,
    evaluate,
    label_signals,
    signals_from_intents,
)
from neurotrade.lab.significance import sharpe_ratio
from neurotrade.lab.trials import TrialLedger

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
MINUTE = 60_000_000_000
COSTS = CostModel(spreads=FlooredSpread(), fees=FeeSchedule())
POSITION = Quantity(1_000)
BARRIER = Decimal("0.02")


class MemoryLedger:
    """An in-memory `TrialLedgerPort`. The harness must never touch storage."""

    def __init__(self) -> None:
        self.appended: list[Trial] = []

    def append(self, trial: Trial) -> None:
        self.appended.append(trial)

    def trials(self, family: str | None = None) -> tuple[Trial, ...]:
        return tuple(t for t in self.appended if family in (None, t.family))


def ledger_and_clock() -> tuple[TrialLedger, MemoryLedger, SimClock]:
    """A ledger over an in-memory store, with a clock the test controls."""
    store = MemoryLedger()
    clock = SimClock(1_000)
    return TrialLedger(store=store, clock=clock, config_hash="cfg_test"), store, clock


def bar(ts: int, close: str, *, high: str | None = None, low: str | None = None) -> Bar:
    """A one-minute AAPL bar. `high`/`low` default to the close, so a series
    built from closes alone touches no barrier it was not asked to."""
    return Bar(
        symbol=AAPL,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=Price(close),
        high=Price(high or close),
        low=Price(low or close),
        close=Price(close),
        volume=Quantity(1_000),
    )


def gap_intent(**overrides: object) -> Intent:
    """A gap-continuation-shaped proposal: market long, stop at the session open
    (99), 2R target, an hour of time barrier."""
    base = {
        "symbol": AAPL,
        "ts_event": 0,
        "ts_init": 0,
        "side": Side.BUY,
        "entry": EntryTrigger.MARKET,
        "entry_price": None,
        "invalidation": Price("99"),
        "target_r": Decimal(2),
        "horizon_ns": 60 * MINUTE,
        "strategy": "gap_continuation",
        "strategy_version": "1.0.0",
        "rationale": "gap +1.40 session ranges, open holding",
    }
    return Intent(**{**base, **overrides})  # type: ignore[arg-type]


def uniform(label: str, indices: range | tuple[int, ...], side: Side = Side.BUY) -> Variant:
    """A variant proposing the same fixed barriers at every index given."""
    return Variant(
        label=label,
        signals=tuple(
            Signal(index=index, side=side, profit_target=BARRIER, stop_loss=BARRIER, max_bars=30)
            for index in indices
        ),
    )


# ── Intents into decisions ───────────────────────────────────────────────


def test_the_conversion_reads_the_barriers_the_intent_itself_defines() -> None:
    """The fractions must reproduce `Intent.target_price`, not approximate it.

    `Intent` states risk as a price and reward as a multiple of it; the labeller
    takes fractions of the entry. Two ways of expressing one trade, and a drift
    between them labels a position the strategy never proposed.
    """
    bars = [bar(0, "100"), bar(MINUTE, "100")]
    signal = signals_from_intents(bars, [gap_intent()])[0]

    entry = bars[0].close
    assert signal.stop_loss * entry.value == gap_intent().risk_per_share(entry)
    assert entry.value * (1 + signal.profit_target) == gap_intent().target_price(entry).value


def test_the_walk_is_counted_off_the_real_bars_not_the_clock() -> None:
    """A halt inside the horizon shortens the walk; it does not invent bars.

    Dividing the horizon by the bar interval would hand the labeller more bars
    than the corpus printed, and a timeout would then be reported at a price
    that never traded.
    """
    dense = [bar(index * MINUTE, "100") for index in range(5)]
    assert signals_from_intents(dense, [gap_intent()])[0].max_bars == 4

    halted = [dense[0], dense[1], bar(120 * MINUTE, "100")]  # the hour ends after bar 1
    assert signals_from_intents(halted, [gap_intent()])[0].max_bars == 1


def test_a_decision_with_no_bar_left_to_walk_is_dropped_not_timed_out() -> None:
    """Zero forward bars inside the horizon is an unlabelled decision.

    Calling it a timeout would put a fabricated outcome in the sample at exactly
    the most recent dates — the point `triple_barrier` returns `None` for.
    """
    assert signals_from_intents([bar(0, "100"), bar(120 * MINUTE, "100")], [gap_intent()]) == ()


def test_a_stop_sitting_on_the_entry_close_is_dropped() -> None:
    """R is undefined there, and `Intent.risk_per_share` raises on it."""
    bars = [bar(0, "99"), bar(MINUTE, "99")]
    assert signals_from_intents(bars, [gap_intent()]) == ()


def test_a_limit_entry_is_refused_rather_than_assumed_to_have_filled() -> None:
    """Labelling a limit order at the close asserts a fill nobody proved."""
    limit = gap_intent(entry=EntryTrigger.LIMIT, entry_price=Price("100"))
    with pytest.raises(ValueError, match=r"only MARKET can be labelled"):
        signals_from_intents([bar(0, "100"), bar(MINUTE, "100")], [limit])


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"symbol": MSFT, "invalidation": Price("99")}, r"evaluate one instrument at a time"),
        ({"ts_event": 7, "ts_init": 7}, r"no bar at ts_event 7"),
    ],
)
def test_an_intent_that_does_not_belong_to_the_series_raises(
    overrides: dict[str, object], expected: str
) -> None:
    """Both would otherwise be scored against another instrument's prices."""
    with pytest.raises(ValueError, match=expected):
        signals_from_intents([bar(0, "100"), bar(MINUTE, "100")], [gap_intent(**overrides)])


def test_the_decisions_come_back_in_bar_order() -> None:
    """Ordering is the caller's alignment guarantee, not the bus's."""
    bars = [bar(index * MINUTE, "100") for index in range(5)]
    late = gap_intent(ts_event=2 * MINUTE, ts_init=2 * MINUTE)
    signals = signals_from_intents(bars, [late, gap_intent()])
    assert [signal.index for signal in signals] == [0, 2]


# ── A variant is what the ledger counts ─────────────────────────────────


def test_two_decisions_on_one_bar_is_refused() -> None:
    """Indexing for scoring would keep one and discard the other in silence."""
    twice = (
        Signal(index=4, side=Side.BUY, profit_target=BARRIER, stop_loss=BARRIER, max_bars=30),
        Signal(index=4, side=Side.SELL, profit_target=BARRIER, stop_loss=BARRIER, max_bars=30),
    )
    with pytest.raises(ValueError, match=r"two signals on one bar"):
        Variant(label="both ways", signals=twice)


def test_a_variant_needs_a_label() -> None:
    """The ledger records it; a blank one is an unreconstructable trial."""
    with pytest.raises(ValueError, match=r"needs a label"):
        Variant(label="  ", signals=())


# ── Labelling and alignment ─────────────────────────────────────────────


def test_variants_that_agree_at_a_candidate_share_one_label() -> None:
    """Same side, same barriers, same bar: one walk, not two.

    Object identity is the check — an equal-but-separate label would mean the
    grid pays for the labelling once per variant, which is what makes a wide
    sweep unaffordable.
    """
    bars = momentum_bars(seed=3, n_bars=200)
    candidates = range(0, 160, 20)
    scored = label_signals(
        bars,
        [uniform("a", candidates), uniform("b", candidates)],
        candidates=candidates,
        horizon_bars=30,
        costs=COSTS,
        quantity=POSITION,
    )
    assert scored.trades[0][0] is scored.trades[1][0]
    assert scored.returns[0] == scored.returns[1]


def test_a_signal_off_the_candidate_grid_raises() -> None:
    """Ignoring it would report part of a strategy as though it were all of it."""
    bars = momentum_bars(seed=3, n_bars=200)
    with pytest.raises(ValueError, match=r"outside the candidate set, first at bar 25"):
        label_signals(
            bars,
            [uniform("a", (0, 25))],
            candidates=range(0, 160, 20),
            horizon_bars=30,
            costs=COSTS,
            quantity=POSITION,
        )


def test_an_unlabellable_candidate_leaves_the_sample_for_every_variant() -> None:
    """Keeping it for the variants that could be labelled makes the columns
    incomparable, and CSCV's ranking then compares different samples."""
    bars = momentum_bars(seed=3, n_bars=50)
    at_the_end = 49  # the last bar: there is nothing forward to walk at all
    scored = label_signals(
        bars,
        [uniform("fires late", (10, at_the_end)), uniform("fires early", (10,))],
        candidates=(10, at_the_end),
        horizon_bars=30,
        costs=COSTS,
        quantity=POSITION,
    )
    assert scored.dropped == (at_the_end,)
    assert scored.candidates == (10,)
    assert [len(series) for series in scored.returns] == [1, 1]


def test_a_candidate_nobody_traded_still_reaches_the_purging_floor() -> None:
    """Its span is what stops the next block training on its tail. Zero reach
    there would leak the label overlap the embargo exists to cut."""
    bars = momentum_bars(seed=3, n_bars=200)
    scored = label_signals(
        bars,
        [uniform("a", (20,)), uniform("b", (20,), side=Side.SELL)],
        candidates=(20, 40),
        horizon_bars=30,
        costs=COSTS,
        quantity=POSITION,
    )
    assert scored.spans == ((20, 50), (40, 70))


def test_the_span_is_the_furthest_reach_any_variant_proposed() -> None:
    """One span per observation, and the conservative one is the long horizon."""
    bars = momentum_bars(seed=3, n_bars=200)
    near = Variant(
        label="near",
        signals=(
            Signal(index=20, side=Side.BUY, profit_target=BARRIER, stop_loss=BARRIER, max_bars=5),
        ),
    )
    far = Variant(
        label="far",
        signals=(
            Signal(index=20, side=Side.BUY, profit_target=BARRIER, stop_loss=BARRIER, max_bars=60),
        ),
    )
    scored = label_signals(
        bars, [near, far], candidates=(20,), horizon_bars=10, costs=COSTS, quantity=POSITION
    )
    assert scored.spans == ((20, 80),)


# ── The verdict ─────────────────────────────────────────────────────────


def scored_pair(n_bars: int = 400) -> tuple[list[Variant], Observations]:
    """Two variants over one series, both trading, with flat observations."""
    bars = momentum_bars(seed=4, n_bars=n_bars)
    candidates = tuple(range(0, n_bars - 40, 10))
    variants = [
        uniform("every candidate", candidates),
        uniform("every other candidate", candidates[::2]),
    ]
    scored = label_signals(
        bars,
        variants,
        candidates=candidates,
        horizon_bars=30,
        costs=COSTS,
        quantity=POSITION,
    )
    return variants, scored


def test_every_variant_is_recorded_before_anything_is_deflated() -> None:
    """Recording only the winner is the bias the ledger exists to remove."""
    variants, scored = scored_pair()
    ledger, store, clock = ledger_and_clock()
    evaluation = assess(
        scored,
        variants,
        family="harness-test",
        hypothesis_prefix="momentum series",
        ledger=ledger,
        clock=clock,
        cv=CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=30),
        pbo_blocks=8,
        source=TrialSource.DISCOVERY,
    )
    assert len(store.appended) == len(variants) == evaluation.n_variants
    assert [trial.hypothesis for trial in store.appended] == [
        f"momentum series: {variant.label}" for variant in variants
    ]
    assert {trial.source for trial in store.appended} == {TrialSource.DISCOVERY}
    assert evaluation.hurdle > 0  # a search of two has a hurdle above zero


def test_expectancy_is_per_trade_not_per_observation() -> None:
    """A sparse strategy's edge divided by the bars it sat out is not its edge.

    Every flat observation in the return vector is a zero, so the two means only
    coincide when the strategy traded everything — which is exactly the case
    that hides the bug.
    """
    variants, scored = scored_pair()
    ledger, _, clock = ledger_and_clock()
    evaluation = assess(
        scored,
        variants,
        family="harness-test",
        hypothesis_prefix="momentum series",
        ledger=ledger,
        clock=clock,
        cv=CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=30),
        pbo_blocks=8,
    )
    best = evaluation.best_index
    trades = scored.trades[best]
    per_trade = sum(float(touch.realised_return) for touch in trades) / len(trades)
    column = scored.returns[best]
    per_observation = sum(column) / len(column)

    assert evaluation.expectancy == pytest.approx(per_trade)
    assert evaluation.n_trades == len(trades)
    if evaluation.n_trades < evaluation.n_observations:
        assert evaluation.expectancy != pytest.approx(per_observation)


def test_the_headline_variant_is_the_one_a_naive_search_would_have_reported() -> None:
    """The number under scrutiny is the one a human would have been tempted by."""
    variants, scored = scored_pair()
    ledger, _, clock = ledger_and_clock()
    evaluation = assess(
        scored,
        variants,
        family="harness-test",
        hypothesis_prefix="momentum series",
        ledger=ledger,
        clock=clock,
        cv=CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=30),
        pbo_blocks=8,
    )
    assert evaluation.best_label == variants[evaluation.best_index].label
    assert evaluation.best_sharpe == max(sharpe_ratio(series) for series in scored.returns)


def test_a_search_of_one_cannot_be_assessed() -> None:
    """CSCV and the CPCV selection both need something to choose between."""
    variants, scored = scored_pair()
    ledger, _, clock = ledger_and_clock()
    with pytest.raises(ValueError, match=r"cannot be assessed"):
        assess(
            replace(scored, returns=scored.returns[:1], trades=scored.trades[:1]),
            variants[:1],
            family="harness-test",
            hypothesis_prefix="momentum series",
            ledger=ledger,
            clock=clock,
            cv=CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=30),
            pbo_blocks=8,
        )


def test_a_variant_that_never_traded_stops_the_run_rather_than_scoring_zero() -> None:
    """An all-flat column has no Sharpe. Scoring it 0.0 enters a variant that
    never fired into the ranking as though it had been tried and found average.
    """
    bars = momentum_bars(seed=4, n_bars=400)
    candidates = tuple(range(0, 340, 10))
    variants = [uniform("trades", candidates), Variant(label="never fires", signals=())]
    scored = label_signals(
        bars, variants, candidates=candidates, horizon_bars=30, costs=COSTS, quantity=POSITION
    )
    ledger, _, clock = ledger_and_clock()
    with pytest.raises(ValueError, match=r"no labelled trade: never fires"):
        assess(
            scored,
            variants,
            family="harness-test",
            hypothesis_prefix="momentum series",
            ledger=ledger,
            clock=clock,
            cv=CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=30),
            pbo_blocks=8,
        )


def test_too_few_observations_to_fill_the_cscv_blocks_raises() -> None:
    """A PBO over blocks that cannot be filled is a number about nothing."""
    bars = momentum_bars(seed=4, n_bars=200)
    candidates = (0, 40, 80)
    variants = [uniform("a", candidates), uniform("b", candidates[:2])]
    scored = label_signals(
        bars, variants, candidates=candidates, horizon_bars=30, costs=COSTS, quantity=POSITION
    )
    ledger, _, clock = ledger_and_clock()
    with pytest.raises(ValueError, match=r"cannot fill 8 CSCV blocks"):
        assess(
            scored,
            variants,
            family="harness-test",
            hypothesis_prefix="momentum series",
            ledger=ledger,
            clock=clock,
            cv=CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=30),
            pbo_blocks=8,
        )


# ── Sparse strategies, which is what the arsenal is ─────────────────────


def test_a_block_the_selected_variant_sat_out_scores_no_edge_rather_than_failing() -> None:
    """One decision per session leaves whole CPCV blocks untraded.

    `sharpe_ratio` refuses a zero-variance series, which is right for a headline
    number and fatal for a subsample: the run would die on a variant that simply
    had no view in one block. A flat block demonstrates no edge, and that is
    what it scores.
    """
    bars = momentum_bars(seed=6, n_bars=600)
    candidates = tuple(range(0, 540, 45))
    early = uniform("first half only", candidates[: len(candidates) // 2])
    late = uniform("second half only", candidates[len(candidates) // 2 :])
    ledger, _, clock = ledger_and_clock()

    evaluation = evaluate(
        bars,
        [early, late],
        candidates=candidates,
        horizon_bars=30,
        family="sparse",
        hypothesis_prefix="half-sample variants",
        ledger=ledger,
        clock=clock,
        costs=COSTS,
        quantity=POSITION,
        cv=CombinatorialPurgedCV(n_groups=4, n_test_groups=2, embargo=30),
        pbo_blocks=4,
    )
    assert len(evaluation.path_sharpes) == 3
    assert all(sharpe == sharpe for sharpe in evaluation.path_sharpes)  # no NaN
    assert evaluation.n_trades == len(candidates) // 2
    assert evaluation.n_observations == len(candidates)
