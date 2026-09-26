"""Measuring a strategy: decisions in, a deflated verdict out.

This is the path from what a strategy proposed to whether the number it posted
survives the search that found it. `gate.py` ran it first, privately, against
two synthetic controls; it lives here because **the arsenal has to be scored by
the same code the gate is trusted on.** A second measurement path would be the
one-implementation invariant (§3.6) broken in the worst possible place: a gate
that proves the lab rejects a snooped control proves nothing about a real
strategy if the real strategy is scored by different arithmetic.

```
Signals (one per variant, per candidate bar)
      -> triple_barrier labels, costs inside, barriers each signal's own
      -> one return vector per variant, aligned on a shared candidate set
      -> TrialLedger.record (every variant, before anything is deflated)
      -> hurdle -> deflated Sharpe        the verdict
      -> CombinatorialPurgedCV            out-of-sample path Sharpes
      -> CSCV                             PBO (reported; it does not vote)
```

**Three seams, on purpose.** `signals_from_intents` turns what a strategy emitted
into decisions with barriers; `label_signals` turns those into an observation
matrix; `assess` turns the matrix into a verdict. Only the first is specific to
one bar series, which is where pooling a universe will have to be answered
(§9.3 is silent on it) — the statistics take a matrix and spans and do not care
how they were assembled.

**A candidate set is a methodological choice, not a detail.** Passing every
fifth bar scores a variant against the counterfactual "what if it had a view
here", which is what makes a grid of variants comparable column by column.
Passing only the bars some variant fired on scores the decisions that were
actually taken. Both are legitimate; they answer different questions, and the
caller has to say which. What is never legitimate is a candidate set that
depends on the outcome.

**Costs are inside every label** (§3.3) because `triple_barrier` charges them
there. Expectancy reported here is therefore already cost-adjusted, which is
what gate G4 asks about.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from neurotrade.core.clock import SimClock
from neurotrade.core.costs import CostModel
from neurotrade.core.events import Bar
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.trials import TrialSource
from neurotrade.core.types import Quantity, Side
from neurotrade.lab.cv import CombinatorialPurgedCV
from neurotrade.lab.labelling import BarrierTouch, Label, triple_barrier
from neurotrade.lab.significance import (
    OverfittingReport,
    moments,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from neurotrade.lab.trials import TrialLedger

__all__ = [
    "Evaluation",
    "Observations",
    "Signal",
    "Variant",
    "assess",
    "evaluate",
    "label_signals",
    "signals_from_intents",
]


@dataclass(frozen=True, slots=True)
class Signal:
    """One variant's decision on one bar, with the barriers it proposed.

    Barrier distances are fractions of the entry rather than prices, because
    that is what `triple_barrier` takes and because a fraction is what survives
    a split adjustment unchanged.

    Example:
        >>> Signal(index=7, side=Side.BUY, profit_target=Decimal("0.02"),
        ...        stop_loss=Decimal("0.01"), max_bars=30).reach
        37
    """

    index: int  # position in the bar series whose close is the entry reference
    side: Side  # direction proposed
    profit_target: Decimal  # profit barrier, as a positive fraction of the entry price
    stop_loss: Decimal  # stop barrier, as a positive fraction of the entry price
    max_bars: int  # vertical barrier, in bars after the entry bar

    @property
    def reach(self) -> int:
        """The furthest bar index this decision can still be open at."""
        return self.index + self.max_bars


@dataclass(frozen=True, slots=True)
class Variant:
    """One configuration of one strategy, and every decision it took.

    A variant is what the trial ledger counts. `min_gap_ranges` at 1.0 and at
    1.2 are two hypotheses, not one strategy with a knob, and deflation is only
    honest if both are recorded.

    Example:
        >>> Variant(label="gap>=1.2", signals=()).label
        'gap>=1.2'
    """

    label: str  # what distinguishes this configuration inside its family; goes in the ledger
    signals: tuple[Signal, ...]  # its decisions; at most one per bar

    def __post_init__(self) -> None:
        """Validate the label and reject two decisions on one bar.

        Raises:
            ValueError: If the label is blank, or two signals share an index —
                which would silently discard one of them when the signals are
                indexed for scoring.
        """
        if not self.label.strip():
            raise ValueError("a variant needs a label — the ledger records it")
        indices = [signal.index for signal in self.signals]
        if len(set(indices)) != len(indices):
            raise ValueError(f"variant {self.label!r} has two signals on one bar")

    def by_index(self) -> dict[int, Signal]:
        """Its decisions, keyed on the bar they were taken on."""
        return {signal.index: signal for signal in self.signals}


@dataclass(frozen=True, slots=True)
class Observations:
    """The scored matrix: what each variant earned at each shared candidate.

    Example:
        >>> Observations(candidates=(0, 5), spans=((0, 30), (5, 35)),
        ...              returns=((0.01, 0.0),), trades=((),), dropped=()).n_observations
        2
    """

    candidates: tuple[int, ...]  # bar indices scored, ascending
    spans: tuple[tuple[int, int], ...]  # (opened, furthest reach) per candidate, in bar indices
    returns: tuple[tuple[float, ...], ...]  # per variant, aligned on candidates; 0.0 where flat
    trades: tuple[tuple[BarrierTouch, ...], ...]  # per variant, only the decisions it took
    dropped: tuple[int, ...]  # candidates discarded: a label some variant needed could not form

    @property
    def n_observations(self) -> int:
        """How many candidates survived labelling."""
        return len(self.candidates)

    def performance(self) -> tuple[tuple[float, ...], ...]:
        """The matrix transposed: one row per observation, one column per variant.

        The shape CSCV wants. Built on demand rather than stored, because the
        column-major form is what everything else reads.
        """
        return tuple(
            tuple(series[row] for series in self.returns) for row in range(self.n_observations)
        )


@dataclass(frozen=True, slots=True)
class Evaluation:
    """What the lab concluded about one search.

    `best_*` describes the variant a naive in-sample search would have reported.
    That is deliberate: the number under scrutiny is the one a human would have
    been tempted by, and `deflated` is the answer to whether it means anything.

    Example:
        >>> Evaluation(family="gap", n_variants=4, n_observations=200, best_index=1,
        ...            best_label="gap>=1.2", best_sharpe=0.08, hurdle=0.05, deflated=0.97,
        ...            pbo=0.2, path_sharpes=(0.06,), n_trades=180,
        ...            expectancy=0.0012, hit_rate=0.44, n_timeouts=20,
        ...            n_ambiguous=3).positive_expectancy
        True
    """

    family: str  # trial-ledger family the variants were recorded under
    n_variants: int  # size of the search, i.e. how many hypotheses it burned
    n_observations: int  # candidates every variant was scored over
    best_index: int  # position in the variants given, so the caller can recover its own object
    best_label: str  # that variant's label
    best_sharpe: float  # its per-observation Sharpe, net of costs — the headline number
    hurdle: float  # expected best Sharpe from a worthless search of this size
    deflated: float  # DSR confidence that `best_sharpe` beats that hurdle
    pbo: float  # probability of backtest overfitting, via CSCV
    path_sharpes: tuple[float, ...]  # out-of-sample Sharpe per recombined CPCV path
    n_trades: int  # decisions the best variant actually took and could label
    expectancy: float  # its mean return per trade, net of modelled costs — gate G4's criterion
    hit_rate: float  # share of those trades that made money after costs
    n_timeouts: int  # trades closed by the time barrier rather than a price barrier
    n_ambiguous: int  # trades where both barriers fell in one bar; the stop was assumed

    @property
    def positive_expectancy(self) -> bool:
        """Whether the best variant made money per trade after costs.

        G4's wording. On its own it is not evidence of an edge — `deflated` is
        what says whether the search that found it was honest.
        """
        return self.expectancy > 0

    @property
    def is_overfit(self) -> bool:
        """Whether PBO says selecting the in-sample winner beats a coin flip."""
        return self.pbo >= 0.5

    def __str__(self) -> str:
        return (
            f"{self.best_label} sharpe={self.best_sharpe:+.4f} hurdle={self.hurdle:.4f} "
            f"dsr={self.deflated:.3f} pbo={self.pbo:.3f} "
            f"exp={self.expectancy:+.5f}/trade hit={self.hit_rate:.2f} "
            f"trials={self.n_variants} obs={self.n_observations} trades={self.n_trades}"
        )


def signals_from_intents(bars: Sequence[Bar], intents: Sequence[Intent]) -> tuple[Signal, ...]:
    """Turn one instrument's proposals into labellable decisions.

    The conversion is the interesting part, because an `Intent` states its risk
    as a *price* (the invalidation level) and its reward as a multiple of it,
    while the labeller takes fractions of the entry. The entry reference is the
    close of the bar the proposal was made on — the labeller's convention, and
    the only honest one for a market order, since a decision taken on a bar's
    close cannot be filled inside that bar.

    Args:
        bars: One instrument's bars, ascending by `ts_event`, split-adjusted.
        intents: Proposals for that same instrument, in any order.

    Returns:
        One signal per labellable intent, ascending by bar index. Two intents
        are dropped rather than labelled: one whose stop sits exactly on the
        entry close (R would be zero, and `Intent.risk_per_share` raises on it),
        and one whose time barrier expires before the next bar (there is no bar
        to walk to). Both are reported by the count difference, never silently
        turned into a trade.

    Raises:
        ValueError: If an intent is for another instrument, carries a timestamp
            no bar in the series has, or proposes a non-`MARKET` entry. The last
            is not a limitation to work around: labelling a limit order as
            though it filled at the close asserts a fill nobody proved, which
            is exactly the flattering fiction §17 warns about. It becomes
            answerable when fill simulation lands.

    Example:
        `demo_intent` is a long stopped at 99 with a 2R target and a one-hour
        time barrier, proposed on a bar closing at 100 — so 1R is 1% of the
        entry, the target is 2%, and one bar falls inside the hour.

        >>> from dataclasses import replace
        >>> def bar(ts, close):
        ...     return Bar(symbol=AAPL, ts_event=ts, ts_init=ts, interval=BarInterval.MIN_1,
        ...                open=Price(close), high=Price(close), low=Price(close),
        ...                close=Price(close), volume=Quantity(100))
        >>> market = replace(demo_intent, entry=EntryTrigger.MARKET, entry_price=None)
        >>> signal = signals_from_intents([bar(1_000, "100"), bar(61_000_000_000, "101")],
        ...                               [market])[0]
        >>> (signal.index, signal.stop_loss, signal.profit_target, signal.max_bars)
        (0, Decimal('0.01'), Decimal('0.02'), 1)
    """
    if not bars:
        raise ValueError("cannot label intents against an empty series")
    symbol = bars[0].symbol
    timestamps = [bar.ts_event for bar in bars]
    index_of = {ts: index for index, ts in enumerate(timestamps)}

    signals: list[Signal] = []
    for intent in intents:
        if intent.symbol != symbol:
            raise ValueError(
                f"intent for {intent.symbol} against a {symbol} series — "
                f"evaluate one instrument at a time"
            )
        if intent.entry is not EntryTrigger.MARKET:
            raise ValueError(
                f"{intent.strategy} proposed a {intent.entry.value} entry; only MARKET can be "
                f"labelled honestly until fills are simulated"
            )
        index = index_of.get(intent.ts_event)
        if index is None:
            raise ValueError(f"no bar at ts_event {intent.ts_event} in the series given")

        entry = bars[index].close
        risk = abs(entry.value - intent.invalidation.value)
        if risk == 0:
            continue  # R undefined: the stop is the entry. Never labelled as a trade.

        # Counted off the real series rather than divided by the bar interval, so
        # a halt or a missing bar shortens the walk instead of inventing bars
        # that were never printed.
        deadline = intent.ts_event + intent.horizon_ns
        max_bars = bisect_right(timestamps, deadline) - index - 1
        if max_bars < 1:
            continue  # the horizon closes before the next bar; nothing to walk.

        stop_loss = risk / entry.value
        signals.append(
            Signal(
                index=index,
                side=intent.side,
                profit_target=intent.target_r * stop_loss,
                stop_loss=stop_loss,
                max_bars=max_bars,
            )
        )
    return tuple(sorted(signals, key=lambda signal: signal.index))


def label_signals(
    bars: Sequence[Bar],
    variants: Sequence[Variant],
    *,
    candidates: Sequence[int],
    horizon_bars: int,
    costs: CostModel,
    quantity: Quantity,
) -> Observations:
    """Label every variant's decisions and align them on one candidate set.

    Labels are cached on `(index, side, barriers, horizon)`, so variants that
    agree at a candidate share one label instead of paying for it twice. When
    every variant proposes the same barriers — a grid that varies the entry rule
    only — that collapses to one label per side per candidate, which is the
    counterfactual framing of §9.3: a variant's return at a candidate is the
    outcome its side selected, or zero when it had no view.

    Args:
        bars: One instrument's bars, ascending, split-adjusted.
        variants: The search. Every signal must fall on a candidate.
        candidates: Bar indices to score. Sorted and de-duplicated on the way
            in, so a caller passing a set cannot move the output.
        horizon_bars: A floor on how far a candidate is assumed to reach, used
            for purging. Applied even where a variant's own barrier is nearer:
            purging too much only costs training data, while purging too little
            leaks the label's tail into the training set.
        costs: Charged at entry and exit, inside each label.
        quantity: Position size; costs are not linear in it.

    Returns:
        The observation matrix. A candidate is dropped for **every** variant
        when any label needed there could not form — at the end of the corpus,
        typically. Dropping the column-mismatched candidate rather than the one
        variant's signal is what keeps the columns comparable, which CSCV and
        the CPCV selection both require.

    Raises:
        ValueError: If a variant has a signal on a bar that is not a candidate.
            Silently ignoring it would discard a real decision and report the
            remainder as if it were the whole strategy.

    Example:
        >>> from neurotrade.core.costs import CostModel, FeeSchedule, FlooredSpread
        >>> from neurotrade.lab.controls import momentum_bars
        >>> bars = momentum_bars(seed=3, n_bars=200)
        >>> variant = Variant(label="long every 20th", signals=tuple(
        ...     Signal(index=index, side=Side.BUY, profit_target=Decimal("0.02"),
        ...            stop_loss=Decimal("0.02"), max_bars=30)
        ...     for index in range(0, 160, 20)))
        >>> scored = label_signals(bars, [variant], candidates=range(0, 160, 20),
        ...                        horizon_bars=30,
        ...                        costs=CostModel(spreads=FlooredSpread(), fees=FeeSchedule()),
        ...                        quantity=Quantity(1_000))
        >>> (scored.n_observations, len(scored.trades[0]), scored.spans[1])
        (8, 8, (20, 50))
    """
    grid = tuple(sorted(set(candidates)))
    if not grid:
        raise ValueError("no candidates to score")
    indexed = [variant.by_index() for variant in variants]

    stray = sorted({index for decisions in indexed for index in decisions} - set(grid))
    if stray:
        raise ValueError(
            f"{len(stray)} signal(s) fall outside the candidate set, first at bar {stray[0]}"
        )

    cache: dict[tuple[int, Side, Decimal, Decimal, int], BarrierTouch | None] = {}

    def touch_for(signal: Signal) -> BarrierTouch | None:
        key = (signal.index, signal.side, signal.profit_target, signal.stop_loss, signal.max_bars)
        if key not in cache:
            cache[key] = triple_barrier(
                bars,
                signal.index,
                side=signal.side,
                profit_target=signal.profit_target,
                stop_loss=signal.stop_loss,
                max_bars=signal.max_bars,
                costs=costs,
                quantity=quantity,
            )
        return cache[key]

    kept: list[int] = []
    dropped: list[int] = []
    for index in grid:
        needed = [decisions[index] for decisions in indexed if index in decisions]
        if any(touch_for(signal) is None for signal in needed):
            dropped.append(index)
        else:
            kept.append(index)

    spans = tuple(
        (
            index,
            max(
                [decisions[index].reach for decisions in indexed if index in decisions]
                + [index + horizon_bars]
            ),
        )
        for index in kept
    )

    returns: list[tuple[float, ...]] = []
    trades: list[tuple[BarrierTouch, ...]] = []
    for decisions in indexed:
        earned: dict[int, float] = {}
        taken: list[BarrierTouch] = []
        for index in kept:
            signal = decisions.get(index)
            if signal is None:
                continue
            touch = touch_for(signal)
            assert touch is not None  # a candidate whose label failed was dropped above
            earned[index] = float(touch.realised_return)
            taken.append(touch)
        returns.append(tuple(earned.get(index, 0.0) for index in kept))
        trades.append(tuple(taken))

    return Observations(
        candidates=tuple(kept),
        spans=spans,
        returns=tuple(returns),
        trades=tuple(trades),
        dropped=tuple(dropped),
    )


def assess(
    observations: Observations,
    variants: Sequence[Variant],
    *,
    family: str,
    hypothesis_prefix: str,
    ledger: TrialLedger,
    clock: SimClock,
    cv: CombinatorialPurgedCV,
    pbo_blocks: int,
    source: TrialSource = TrialSource.MANUAL,
) -> Evaluation:
    """Record the search, then deflate its winner against it.

    **Every variant is recorded before anything is deflated.** Recording only
    the ones that looked good is the exact bias the ledger exists to remove, and
    it is the caller's `family` that decides what the winner is deflated
    against. The trials are written with `n_paths=0` because they are single
    backtests at the point they are recorded; the CPCV paths below describe the
    winner, not each variant.

    Args:
        observations: The scored matrix from `label_signals`.
        variants: The same variants, in the same order.
        family: Trial-ledger family. Deflation happens within it, so a family
            that lumps unrelated searches together deflates against noise and
            one that splits a single search across families hides its size.
        hypothesis_prefix: Prepended to each variant's label in the ledger, so a
            reader six months later can reconstruct what was searched.
        ledger: Where the trials go. A measurement run must use the project's
            real ledger; only the controls in `gate.py` may use a scratch one.
        clock: Advanced one nanosecond per trial, so the records are distinct
            and ordered without depending on wall time.
        cv: The CPCV design used for the out-of-sample paths.
        pbo_blocks: CSCV blocks, even and at least 4.
        source: What ran the search; `DISCOVERY` for an automated sweep, which
            §17 counts exactly like a manual one.

    Returns:
        The verdict.

    Raises:
        ValueError: If fewer than two variants are given — a search of one has
            nothing for the CPCV selection or CSCV to choose between — or if
            some variant took no labelled trade at all. The second is not a
            degenerate case to paper over with a zero: an all-flat column has no
            Sharpe, and scoring it as 0.0 would quietly enter a variant that
            never traded into the ranking as though it had been tried and found
            average.

    Example:
        >>> evaluation = assess(scored, variants, family="gap-continuation",  # doctest: +SKIP
        ...                     hypothesis_prefix="gap continuation on SPY 1m",
        ...                     ledger=ledger, clock=clock, cv=cv, pbo_blocks=8)
        >>> evaluation.positive_expectancy, evaluation.deflated >= 0.95  # doctest: +SKIP
        (True, False)
    """
    if len(variants) < 2:
        raise ValueError(f"a search of {len(variants)} variant(s) cannot be assessed; need 2+")
    if len(variants) != len(observations.returns):
        raise ValueError(
            f"{len(variants)} variants against {len(observations.returns)} scored columns"
        )
    if observations.n_observations < pbo_blocks:
        raise ValueError(
            f"{observations.n_observations} observations cannot fill {pbo_blocks} CSCV blocks"
        )
    empty = [
        variant.label
        for variant, taken in zip(variants, observations.trades, strict=True)
        if not taken
    ]
    if empty:
        raise ValueError(f"variant(s) took no labelled trade: {', '.join(empty)}")

    returns = observations.returns
    for variant, series in zip(variants, returns, strict=True):
        clock.advance_ns(1)
        ledger.record(
            hypothesis=f"{hypothesis_prefix}: {variant.label}",
            family=family,
            sharpe=sharpe_ratio(series),
            n_observations=len(series),
            source=source,
            n_paths=0,
        )

    best_index = max(range(len(variants)), key=lambda i: sharpe_ratio(returns[i]))
    best_series = returns[best_index]
    best_sharpe = sharpe_ratio(best_series)
    shape = moments(best_series)

    report: OverfittingReport = probability_of_backtest_overfitting(
        observations.performance(), n_blocks=pbo_blocks
    )
    taken = observations.trades[best_index]

    return Evaluation(
        family=family,
        n_variants=len(variants),
        n_observations=observations.n_observations,
        best_index=best_index,
        best_label=variants[best_index].label,
        best_sharpe=best_sharpe,
        hurdle=ledger.hurdle(family),
        deflated=ledger.deflate(
            best_sharpe,
            family=family,
            n_observations=len(best_series),
            skew=shape.skew,
            kurtosis=shape.kurtosis,
        ),
        pbo=report.pbo,
        path_sharpes=_path_sharpes(cv, observations.spans, returns),
        n_trades=len(taken),
        expectancy=sum(float(touch.realised_return) for touch in taken) / len(taken),
        hit_rate=sum(touch.is_win for touch in taken) / len(taken),
        n_timeouts=sum(touch.label is Label.TIMEOUT for touch in taken),
        n_ambiguous=sum(touch.ambiguous for touch in taken),
    )


def evaluate(
    bars: Sequence[Bar],
    variants: Sequence[Variant],
    *,
    candidates: Sequence[int],
    horizon_bars: int,
    family: str,
    hypothesis_prefix: str,
    ledger: TrialLedger,
    clock: SimClock,
    costs: CostModel,
    quantity: Quantity,
    cv: CombinatorialPurgedCV,
    pbo_blocks: int,
    source: TrialSource = TrialSource.MANUAL,
) -> Evaluation:
    """Label a search over one series and assess it, in one call.

    The convenience both callers use. Split the two halves apart when the
    observations themselves are wanted — pooling several instruments, or
    inspecting the trades behind a verdict.

    Args:
        bars: One instrument's bars, ascending, split-adjusted.
        variants: The search.
        candidates: The observation grid; see the module docstring on why this
            is a methodological choice.
        horizon_bars: Purging floor, in bars.
        family: Trial-ledger family.
        hypothesis_prefix: Ledger prefix for each variant.
        ledger: Where the trials go.
        clock: Advanced once per trial recorded.
        costs: Charged inside every label.
        quantity: Position size.
        cv: The CPCV design.
        pbo_blocks: CSCV blocks.
        source: What ran the search.

    Returns:
        The verdict.

    Example:
        >>> evaluate(bars, variants, candidates=candidates, horizon_bars=30,  # doctest: +SKIP
        ...          family="control-planted", hypothesis_prefix="crossover",
        ...          ledger=ledger, clock=clock, costs=costs, quantity=Quantity(1_000),
        ...          cv=cv, pbo_blocks=8).best_label
        'fast=5 slow=20'
    """
    observations = label_signals(
        bars,
        variants,
        candidates=candidates,
        horizon_bars=horizon_bars,
        costs=costs,
        quantity=quantity,
    )
    return assess(
        observations,
        variants,
        family=family,
        hypothesis_prefix=hypothesis_prefix,
        ledger=ledger,
        clock=clock,
        cv=cv,
        pbo_blocks=pbo_blocks,
        source=source,
    )


def _sharpe_or_flat(series: Sequence[float]) -> float:
    """Sharpe over a slice, treating a flat slice as no edge rather than an error.

    `sharpe_ratio` refuses a zero-variance series — it raises, via `moments`,
    rather than returning an infinity nobody could beat. That is right for a
    headline number and wrong for a subsample: a CPCV split in which the
    selected variant never traded is a real thing to score, and "no edge
    demonstrated here" is what it demonstrates. Dense searches never reach this
    — it exists because a sparse strategy, one decision per session, can easily
    leave a whole block untraded.

    The flat test is on the values themselves, not on `moments`, because
    `moments` is what raises.
    """
    if len(series) < 2 or len(set(series)) < 2:
        return 0.0
    return sharpe_ratio(series)


def _path_sharpes(
    cv: CombinatorialPurgedCV,
    spans: Sequence[tuple[int, int]],
    returns: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Out-of-sample Sharpe for each recombined CPCV path.

    Selecting the parameters **is** the fitting here, so each split picks its
    winner on the purged training observations alone and is scored on the test
    groups it never saw. Re-dealing those test segments gives one complete
    walk-through of the sample per path, and the spread across paths is the
    distribution §8 asks for rather than a single lucky number.

    **Expect the paths to agree when the selection is stable, often exactly.** A
    path covers every group once, so when every split picks the same variant all
    the paths are the same series in a different order and post an identical
    Sharpe. The spread only opens up when the splits disagree about what to fit.
    That is a property of a small grid on one series, not of CPCV: a model
    refitted per split varies whether or not the choice of variant does.
    """
    blocks = cv.groups(len(spans))
    chosen: list[int] = []
    for split in cv.split(spans):
        chosen.append(
            max(
                range(len(returns)),
                key=lambda i: _sharpe_or_flat([returns[i][index] for index in split.train]),
            )
        )

    sharpes: list[float] = []
    for path in cv.paths():
        series: list[float] = []
        for split_index, group in path:
            variant = chosen[split_index]
            series.extend(returns[variant][index] for index in blocks[group])
        sharpes.append(_sharpe_or_flat(series))
    return tuple(sharpes)
