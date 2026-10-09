"""Running a registered strategy over the corpus and scoring what it did.

This is the end-to-end path: bars out of the corpus, through the real engine and
the real context, into the same measurement `gate.py`'s controls go through
(`lab/evaluation.py`). Until this existed a strategy could be written, tested and
committed without anybody knowing whether it fires on real data at all.

**One engine pass per variant, over the whole universe at once.** That is how a
live host runs — one instance subscribed to every instrument, keeping its state
per symbol — so measuring it any other way would measure something else (§3.6).
Labelling then happens per instrument, because a triple barrier walks forward
through *one* series, and the per-instrument samples are pooled at the end.

**Pooling is not optional here.** A gap strategy fires at most once per session
per symbol, so a single instrument over the seed window offers a few dozen
decisions — far too few to deflate anything. Pooling buys the sample; what it
does not buy is independence, and the report says so: ten symbols gapping on one
morning is one market event, not ten bets. `Measurement.n_sessions` is there to
be read next to `n_observations`, and a deflated Sharpe computed as though every
observation were independent is optimistic by exactly that ratio.

**Nothing here decides anything.** The verdict is `Evaluation.deflated` and the
G4 criterion is `Evaluation.positive_expectancy`; this module produces them and
prints the counts behind them. A corpus too thin to support a verdict yields
`evaluation=None` and a reason, rather than a number nobody should trust.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from statistics import fmean
from typing import Final

from neurotrade.core.clock import Nanos, SimClock, to_datetime
from neurotrade.core.costs import CostModel
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.intent import Intent
from neurotrade.core.ports import CalendarPort, StoragePort, TradeJournalPort
from neurotrade.core.trades import TradeRecord
from neurotrade.core.trials import TrialSource
from neurotrade.core.types import Quantity, Symbol
from neurotrade.features.cross_section import BenchmarkSource
from neurotrade.features.registry import FeatureRegistry
from neurotrade.lab.bootstrap import DEFAULT_RESAMPLES, SharpeInterval, bootstrap_sharpe
from neurotrade.lab.cv import CombinatorialPurgedCV
from neurotrade.lab.engine import BacktestEngine
from neurotrade.lab.evaluation import (
    Evaluation,
    Observations,
    Part,
    Variant,
    assess,
    label_signals,
    pool,
    signals_from_intents,
)
from neurotrade.lab.feed import CorpusFeed
from neurotrade.lab.labelling import BarrierTouch
from neurotrade.lab.regimes import (
    RegimeCoverage,
    VolatilityBucket,
    bucket_sessions,
    coverage_of,
    trailing_volatility,
)
from neurotrade.lab.significance import (
    HaircutReport,
    annualized,
    haircut_sharpe,
    minimum_backtest_length,
    moments,
    sharpe_ratio,
)
from neurotrade.lab.trials import TrialLedger
from neurotrade.strategies.base import Strategy
from neurotrade.strategies.context import MarketContext

__all__ = [
    "DEFAULT_EMBARGO_NS",
    "CostCurve",
    "CostPoint",
    "Measurement",
    "SymbolRun",
    "measure_strategy",
]

DEFAULT_EMBARGO_NS: Final = 86_400_000_000_000
"""One calendar day of embargo, in nanoseconds.

Spans are nanoseconds once a universe is pooled, so the embargo has to be too. A
day is the shortest defensible choice for a day strategy: a label that opens the
morning after a test block still shares that block's overnight news, and the
features behind it are built from the same sessions.
"""

_NOT_ENOUGH = "the corpus supports no verdict"

_NANOS_PER_YEAR: Final = 31_557_600_000_000_000
"""Julian year, in nanoseconds — 365.25 days.

Used only to turn a per-observation Sharpe into an annualised one for
`minimum_backtest_length`, which is stated in years. Calendar years rather than
252 trading days on purpose: the span being converted is a wall-clock window,
weekends included, and the observation count already carries how often the
strategy actually decided inside it."""


@dataclass(frozen=True, slots=True)
class CostPoint:
    """What the winning variant earned at one cost level.

    Example:
        >>> CostPoint(scale=2.0, n_trades=100, expectancy=-0.0004, sharpe=-0.02).label
        '2x'
    """

    scale: float  # multiple of the reference cost model this level charges
    n_trades: int  # labelled trades at this level; costs cannot change the count
    expectancy: float  # mean return per trade, net of this level's costs
    sharpe: float  # per-observation Sharpe at this level, over the same candidates

    @property
    def label(self) -> str:
        """The level, as it appears in a printed curve."""
        return f"{self.scale:g}x"

    def __str__(self) -> str:
        return f"{self.label:<6} exp={self.expectancy:+.5f}/trade sharpe={self.sharpe:+.4f}"


@dataclass(frozen=True, slots=True)
class CostCurve:
    """The winning variant's expectancy as modelled costs rise.

    **Nearly free, and that is why it exists.** Costs enter only inside
    `label_signals`, after the engine has already turned bars into intents, so one
    engine pass can be re-labelled at every level on the curve. The expensive half
    of a measurement is not repeated.

    **The variant is fixed at the reference level, deliberately.** Re-choosing a
    winner at each cost level would be a fresh search at every point and would
    need its own trials; what a break-even cost answers is "how much cost does
    *this* result survive", which is a question about one variant.

    Example:
        >>> curve = CostCurve(variant="VM 1", points=(
        ...     CostPoint(scale=1.0, n_trades=100, expectancy=0.0002, sharpe=0.01),
        ...     CostPoint(scale=2.0, n_trades=100, expectancy=-0.0002, sharpe=-0.01),
        ... ))
        >>> curve.break_even
        1.5
    """

    variant: str  # the variant the curve follows, chosen at the reference level
    points: tuple[CostPoint, ...]  # ascending by scale

    @property
    def break_even(self) -> float | None:
        """The cost multiple at which expectancy crosses zero.

        Linearly interpolated between the two points that straddle the crossing.
        Linear because expectancy is a mean of per-trade returns and every term
        in the cost model is either proportional to size or a per-order floor, so
        the relationship is close to affine over a handful of levels — not
        because the curve was measured densely enough to fit anything better.

        Returns:
            The multiple, or `None` when the curve never crosses: either already
            negative at the cheapest level measured, or still positive at the
            dearest. Both are findings, and `__str__` says which.

        Example:
            >>> CostCurve(variant="a", points=(
            ...     CostPoint(scale=1.0, n_trades=9, expectancy=-0.001, sharpe=-0.1),
            ... )).break_even is None
            True
        """
        for earlier, later in zip(self.points, self.points[1:], strict=False):
            if earlier.expectancy > 0 >= later.expectancy:
                span = earlier.expectancy - later.expectancy
                return earlier.scale + (later.scale - earlier.scale) * earlier.expectancy / span
        return None

    def __str__(self) -> str:
        levels = "  ".join(f"{point.label}:{point.expectancy:+.5f}" for point in self.points)
        crossing = self.break_even
        if crossing is not None:
            verdict = f"break-even at {crossing:.2f}x"
        elif self.points and self.points[0].expectancy <= 0:
            verdict = "negative at every level measured"
        else:
            verdict = f"still positive at {self.points[-1].label}" if self.points else "no levels"
        return f"{self.variant} {levels}  -> {verdict}"


@dataclass(frozen=True, slots=True)
class SymbolRun:
    """What one instrument contributed.

    Example:
        >>> run = SymbolRun(symbol=AAPL, n_bars=0, n_signals=(0,), n_observations=0, dropped=0,
        ...                 n_trades=(0,), expectancy=(0.0,))
        >>> run.traded
        False
    """

    symbol: Symbol  # the instrument
    n_bars: int  # bars the corpus held for it in the window
    n_signals: tuple[int, ...]  # decisions taken, per variant, in sweep order
    n_observations: int  # candidate bars scored — the union of the variants' decisions
    dropped: int  # candidates discarded because a label could not form (end of corpus)
    n_trades: tuple[int, ...]  # labelled trades, per variant, in sweep order
    expectancy: tuple[float, ...]  # mean return per trade, per variant; 0.0 where it never traded

    @property
    def traded(self) -> bool:
        """Whether any variant proposed anything on this instrument."""
        return self.n_observations > 0

    def __str__(self) -> str:
        signals = "/".join(str(count) for count in self.n_signals)
        return (
            f"{self.symbol!s:<14} bars={self.n_bars:>7} signals={signals:<12} "
            f"obs={self.n_observations:<5} dropped={self.dropped}"
        )


@dataclass(frozen=True, slots=True)
class Measurement:
    """One strategy, its declared sweep, and what the corpus said about it.

    Example:
        >>> Measurement(strategy="gap_continuation", version="1.0.0", family="gap_continuation",
        ...             variants=("a", "b"), runs=(), n_sessions=0, regime_gated=False,
        ...             digest="deadbeef", evaluation=None, reason="no signals").is_measured
        False
    """

    strategy: str  # registered plugin name
    version: str  # its semantic version, so a retune is distinguishable
    family: str  # trial-ledger family the variants were recorded under
    variants: tuple[str, ...]  # the sweep's labels, in order
    runs: tuple[SymbolRun, ...]  # per instrument, in the order they were read
    n_sessions: int  # distinct sessions the pooled observations opened on
    regime_gated: bool  # False when UNKNOWN was treated as permissive (research only)
    digest: str  # over every variant run's own digest: behaviour, not results
    evaluation: Evaluation | None  # None when the sample could not support one
    reason: str  # why there is no evaluation; empty when there is one
    regimes: RegimeCoverage | None = None  # per-volatility-bucket expectancy of the winner
    interval: SharpeInterval | None = None  # session-bootstrapped bounds on the winner's Sharpe
    deflated_bootstrapped: float | None = None  # DSR charged at the bootstrapped effective count
    haircut: HaircutReport | None = None  # Harvey-Liu multiple-testing discount on the winner
    min_backtest_years: float | None = None  # MinBTL: sample this search would need to be credible
    sample_years: float | None = None  # wall-clock span measured, for comparison against MinBTL
    costs: CostCurve | None = None  # expectancy against modelled cost; None when no levels swept

    @property
    def clears_min_backtest_length(self) -> bool | None:
        """Whether the window measured is as long as MinBTL wants.

        Returns:
            `None` when either number is missing — a non-positive Sharpe has no
            MinBTL, since there is no performance for selection bias to have
            manufactured.
        """
        if self.min_backtest_years is None or self.sample_years is None:
            return None
        return self.sample_years >= self.min_backtest_years

    @property
    def is_measured(self) -> bool:
        """Whether a verdict came back at all."""
        return self.evaluation is not None

    @property
    def n_signals(self) -> int:
        """Decisions taken across every instrument and variant."""
        return sum(sum(run.n_signals) for run in self.runs)

    def per_symbol(self) -> tuple[tuple[Symbol, int, float], ...]:
        """The winning variant broken down by instrument, heaviest contributor first.

        Pooling is mandatory — a sparse strategy fires too rarely for any single
        instrument to deflate anything — but it hides the case where the pooled
        number is substantially one name's. This is how that becomes visible.

        Returns:
            `(symbol, n_trades, expectancy)` per instrument that traded, sorted
            by trade count descending then by symbol, for the variant the
            pooled assessment picked. Empty when there is no verdict to break
            down, since "best" is only defined by the pooled sample.

        Example:
            >>> Measurement(strategy="s", version="1.0.0", family="s", variants=("a",),
            ...             runs=(), n_sessions=0, regime_gated=False, digest="d",
            ...             evaluation=None, reason="no signals").per_symbol()
            ()
        """
        if self.evaluation is None:
            return ()
        index = self.evaluation.best_index
        found = [
            (run.symbol, run.n_trades[index], run.expectancy[index])
            for run in self.runs
            if run.n_trades[index] > 0
        ]
        return tuple(sorted(found, key=lambda row: (-row[1], str(row[0]))))

    @property
    def concentration(self) -> float:
        """Share of the winning variant's trades taken on its heaviest instrument.

        The one number that says whether a pooled result is a market finding or
        one name's. `gap_continuation`'s first measurement scored 0.46 — 56 of
        122 observations were EEM — which is the reading that should have
        stopped it being quoted as a universe result.

        Returns:
            A fraction in `(0, 1]`, or `0.0` when there is no verdict. One
            instrument supplying everything returns `1.0`.
        """
        rows = self.per_symbol()
        if not rows:
            return 0.0
        total = sum(count for _, count, _ in rows)
        return rows[0][1] / total if total else 0.0

    def __str__(self) -> str:
        head = f"{self.strategy}@{self.version} [{', '.join(self.variants)}]"
        if self.evaluation is None:
            return f"{head}  NOT MEASURED — {self.reason}"
        gated = "gated" if self.regime_gated else "ungated"
        lines = [f"{head}  {self.evaluation}  sessions={self.n_sessions} {gated}"]
        if self.regimes is not None:
            lines.append(f"           {self.regimes}")
        if self.interval is not None:
            bootstrapped = (
                ""
                if self.deflated_bootstrapped is None
                else f" dsr_boot={self.deflated_bootstrapped:.3f}"
            )
            lines.append(f"           bootstrap: {self.interval}{bootstrapped}")
        if self.haircut is not None:
            lines.append(f"           haircut: {self.haircut}")
        if self.min_backtest_years is not None and self.sample_years is not None:
            verdict = "clears" if self.clears_min_backtest_length else "SHORT"
            lines.append(
                f"           minBTL: wants {self.min_backtest_years:.2f} yr, "
                f"have {self.sample_years:.2f} yr — {verdict}"
            )
        if self.costs is not None:
            lines.append(f"           costs: {self.costs}")
        return "\n".join(lines)


def measure_strategy(
    strategy: type[Strategy],
    *,
    store: StoragePort,
    calendar: CalendarPort,
    features: FeatureRegistry,
    symbols: Sequence[Symbol],
    start: Nanos,
    end: Nanos,
    ledger: TrialLedger,
    clock: SimClock,
    costs: CostModel,
    quantity: Quantity,
    interval: BarInterval = BarInterval.MIN_1,
    warmup_ns: Nanos = 0,
    embargo_ns: Nanos = DEFAULT_EMBARGO_NS,
    n_groups: int = 6,
    n_test_groups: int = 2,
    pbo_blocks: int = 8,
    family: str | None = None,
    ungated: bool = True,
    journal: TradeJournalPort | None = None,
    run_id: str = "",
    source: TrialSource = TrialSource.MANUAL,
    cost_levels: Sequence[tuple[float, CostModel]] = (),
    resamples: int = DEFAULT_RESAMPLES,
    benchmarks: BenchmarkSource | None = None,
) -> Measurement:
    """Run every variant a strategy declares over the corpus and assess them.

    Args:
        strategy: The registered plugin class. Its `sweep()` decides what is
            measured, and every entry in it is a trial the ledger counts.
        store: The corpus, behind the storage port — never a concrete adapter,
            which is what keeps `lab/` off the Parquet layer.
        calendar: Session bounds, for the market context.
        features: Library the strategy's declared features are resolved from.
        symbols: The universe. Sorted and de-duplicated, so the caller's
            container type cannot move the result.
        start: First moment to dispatch, inclusive.
        end: Last moment, exclusive.
        ledger: Where the trials go — the project's real one for a real run.
        clock: Advanced one nanosecond per trial recorded.
        costs: Charged inside every label (§3.3).
        quantity: Position size the costs are modelled at.
        interval: Bar size. Must match what the features expect.
        warmup_ns: Span before `start` read into the context without dispatching
            it. A strategy reading session history needs this or its first
            sessions are silent — `gap_continuation` has no band at all until 14
            sessions have completed.
        embargo_ns: Embargo for CPCV, in nanoseconds because pooled spans are.
        n_groups: CPCV blocks.
        n_test_groups: Blocks held out per split.
        pbo_blocks: CSCV blocks; even, at least 4.
        family: Trial-ledger family. Defaults to the strategy's name, which is
            the right grain: one strategy's parameter sweep is one search.
        ungated: Treat an unclassified regime as permissive. **True by default
            here**, because until the Phase 5 classifier lands every regime is
            unclassified and a gated run fires nowhere at all; the result records
            which way it ran.
        source: What ran the search. `DISCOVERY` for an automated sweep, which
            §17 counts exactly like a manual one.
        cost_levels: `(scale, model)` pairs to re-label the winning variant at,
            for a break-even cost curve. Free of a second engine pass — costs
            enter inside `label_signals` and nowhere earlier — so the only cost
            is one extra labelling pass per level. Empty leaves `costs` `None`.
            Not a search: the variant is chosen once at `costs`, so no level here
            spends a trial.
        resamples: Resamples for the session bootstrap on the winner's Sharpe.
        benchmarks: What each instrument's return is regressed on, per session —
            a `core.sectors.SectorMap`. Only read by a strategy that declares
            `needs_cross_section`; without it every beta and every residual is
            `None`, so a residual-ranking strategy fires nowhere. Silent rather
            than an error, because most strategies neither need nor read it.

    Returns:
        The measurement. `evaluation` is `None`, with a reason, when the pooled
        sample cannot carry the statistics — too few observations for the CSCV
        blocks, or a variant that never traded. That is a finding about the
        corpus, not an error, and it is the expected answer on a one-regime
        window.

    Example:
        >>> measurement = measure_strategy(GapContinuation, ...)  # doctest: +SKIP
        >>> measurement.evaluation.positive_expectancy  # doctest: +SKIP
        True
    """
    sweep = strategy.sweep()
    labels = tuple(label for label, _ in sweep)
    ordered = tuple(sorted(set(symbols), key=str))

    fingerprint = hashlib.blake2b(digest_size=8)
    by_variant: list[dict[Symbol, list[Intent]]] = []
    regime_gated = True
    for label, instance in sweep:
        engine = BacktestEngine(
            feed=CorpusFeed(store=store, symbols=ordered, interval=interval),
            clock=SimClock(0),
            context=MarketContext(
                features=features,
                calendar=calendar,
                interval=interval,
                benchmarks=benchmarks,
            ),
            ungated=ungated,
            # Taken from the strategy rather than from a parameter: the family
            # is a property of the rule, and a caller who had to remember to
            # pass it is a caller who will forget.
            overnight=strategy.holds_overnight,
        )
        engine.add_strategy(instance)
        result = engine.run(start, end, warmup_ns=warmup_ns)
        regime_gated = result.regime_gated
        fingerprint.update(f"{label}={result.digest}\n".encode())

        grouped: dict[Symbol, list[Intent]] = {}
        for intent in result.intents:
            grouped.setdefault(intent.symbol, []).append(intent)
        by_variant.append(grouped)

    parts: list[Part] = []
    runs: list[SymbolRun] = []
    session_ranges: dict[date, list[float]] = {}
    # One accumulator per cost level per variant, filled inside the symbol loop
    # while that symbol's bars are still in hand. Holding the bars to re-label
    # afterwards is the trap `pool` already avoids: 58 instruments of minute bars
    # is tens of millions of `Bar` objects, and all a cost curve needs from them
    # is two floats per observation.
    curve: list[list[_Level]] = [[_Level() for _ in labels] for _ in cost_levels]
    for symbol in ordered:
        bars = list(store.read_bars(symbol, interval, start, end))
        # A session's high-low range over its close: the volatility proxy the
        # regime buckets are cut from. Accumulated market-wide, because a
        # volatility regime is a property of the market and pooling discards
        # the symbol anyway. Derived statistic, so float is correct here.
        extremes: dict[date, tuple[Decimal, Decimal, Decimal]] = {}
        for one in bars:
            session = to_datetime(one.ts_event).date()
            held = extremes.get(session)
            if held is None:
                extremes[session] = (one.low.value, one.high.value, one.close.value)
            else:
                extremes[session] = (
                    min(held[0], one.low.value),
                    max(held[1], one.high.value),
                    one.close.value,
                )
        for session, (low, high, close) in extremes.items():
            if close > 0:
                session_ranges.setdefault(session, []).append(float((high - low) / close))
        if not bars:
            runs.append(
                SymbolRun(
                    symbol=symbol,
                    n_bars=0,
                    n_signals=tuple(0 for _ in labels),
                    n_observations=0,
                    dropped=0,
                    n_trades=tuple(0 for _ in labels),
                    expectancy=tuple(0.0 for _ in labels),
                )
            )
            continue

        variants = tuple(
            Variant(label=label, signals=signals_from_intents(bars, grouped.get(symbol, ())))
            for label, grouped in zip(labels, by_variant, strict=True)
        )
        candidates = sorted({signal.index for variant in variants for signal in variant.signals})
        counts = tuple(len(variant.signals) for variant in variants)
        if not candidates:
            runs.append(
                SymbolRun(
                    symbol=symbol,
                    n_bars=len(bars),
                    n_signals=counts,
                    n_observations=0,
                    dropped=0,
                    n_trades=tuple(0 for _ in labels),
                    expectancy=tuple(0.0 for _ in labels),
                )
            )
            continue

        observations = label_signals(
            bars,
            variants,
            candidates=candidates,
            # Every candidate is a bar something fired on, so each carries its own
            # reach and the floor never binds. A strategy's own time barrier is a
            # better purging span than any constant this module could pick.
            horizon_bars=0,
            costs=costs,
            quantity=quantity,
        )
        if journal is not None:
            _journal_trades(
                journal,
                symbol=symbol,
                bars=bars,
                variants=variants,
                observations=observations,
                strategy=strategy,
                costs=costs,
                config_hash=ledger.config_hash,
                run_id=run_id,
                recorded_ns=clock.now_ns(),
            )
        parts.append(
            Part(
                key=str(symbol),
                timestamps=[bar.ts_event for bar in bars],
                observations=observations,
            )
        )
        for level, (_, model) in enumerate(cost_levels):
            # Same bars, same candidates, same barriers — only the charge differs.
            # `Signal.profit_target` and `stop_loss` come off the `Intent`, so a
            # cost level cannot move which barrier was touched, only what the
            # touch was worth. That is why the trade counts match across levels
            # and a level's column stays aligned with the reference's.
            scored = label_signals(
                bars,
                variants,
                candidates=candidates,
                horizon_bars=0,
                costs=model,
                quantity=quantity,
            )
            for column, (returns, touches) in enumerate(
                zip(scored.returns, scored.trades, strict=True)
            ):
                curve[level][column].absorb(returns, touches)
        # Per instrument, per variant: what it actually earned. Pooling is what
        # makes the verdict, but a pooled number that is really one instrument's
        # is the failure mode pooling hides — `gap_continuation`'s first result
        # took 56 of its 122 observations from EEM alone.
        per_variant = tuple(_expectancy(column) for column in observations.trades)
        runs.append(
            SymbolRun(
                symbol=symbol,
                n_bars=len(bars),
                n_signals=counts,
                n_observations=observations.n_observations,
                dropped=len(observations.dropped),
                n_trades=tuple(count for count, _ in per_variant),
                expectancy=tuple(value for _, value in per_variant),
            )
        )

    pooled = pool(parts)
    sessions = len({to_datetime(opened).date() for opened, _ in pooled.spans})
    # Trailing volatility reads only prior sessions, so the measure itself is
    # live-safe; the tercile boundaries are full-sample, which is right for
    # describing a finished run and wrong for a live gate. See lab/regimes.py.
    readings = trailing_volatility({s: fmean(v) for s, v in session_ranges.items()})
    by_session = bucket_sessions(readings) if readings else {}
    observation_buckets: tuple[VolatilityBucket | None, ...] = tuple(
        by_session.get(to_datetime(opened).date()) for opened, _ in pooled.spans
    )
    shell = Measurement(
        strategy=strategy.name,
        version=strategy.version,
        family=family or strategy.name,
        variants=labels,
        runs=tuple(runs),
        n_sessions=sessions,
        regime_gated=regime_gated,
        digest=fingerprint.hexdigest(),
        evaluation=None,
        reason="",
    )

    refusal = _why_not(pooled, labels, pbo_blocks=pbo_blocks, n_groups=n_groups)
    if refusal:
        return replace(shell, reason=refusal)

    evaluation = assess(
        pooled,
        labels,
        family=shell.family,
        hypothesis_prefix=(
            f"{strategy.name}@{strategy.version} on {len(parts)} instruments, "
            f"{to_datetime(start).date()}..{to_datetime(end).date()} {interval.value}"
        ),
        ledger=ledger,
        clock=clock,
        cv=CombinatorialPurgedCV(
            n_groups=n_groups, n_test_groups=n_test_groups, embargo=embargo_ns
        ),
        pbo_blocks=pbo_blocks,
        source=source,
        # Ten instruments gapping on one morning is one market event, and a
        # strategy that decides several times in a session reads that session
        # once. Sessions are the coarsest honest grouping available here, so the
        # verdict carries a second deflation charged at that count.
        n_clusters=sessions if sessions >= 2 else None,
    )
    coverage = coverage_of(
        observation_buckets,
        tuple(
            None if touch is None else float(touch.realised_return)
            for touch in pooled.trades[evaluation.best_index]
        ),
    )
    winner = pooled.returns[evaluation.best_index]
    # The session each observation opened in — the bootstrap's resampling unit,
    # and the same grouping `n_clusters` above charges the clustered DSR at.
    sessions_of = tuple(to_datetime(opened).date() for opened, _ in pooled.spans)
    # Named `bounds`, not `interval`: `interval` is this function's bar size.
    bounds = bootstrap_sharpe(winner, sessions_of, n_resamples=resamples)
    bootstrapped: float | None = None
    if bounds is not None:
        shape = moments(winner)
        # The whole point of the interval: DSR charged at the sample size the
        # dependence implies, rather than at either end of the bracket already
        # reported. `deflate` only reads the family's history, so calling it a
        # second time records nothing and cannot move the ledger.
        bootstrapped = ledger.deflate(
            evaluation.best_sharpe,
            family=shell.family,
            n_observations=max(2, round(bounds.effective_n)),
            skew=shape.skew,
            kurtosis=shape.kurtosis,
        )
    years = (end - start) / _NANOS_PER_YEAR
    wanted: float | None = None
    if evaluation.best_sharpe > 0 and years > 0 and evaluation.n_observations > 0:
        # MinBTL is stated in years, so the Sharpe going into it has to be
        # annualised — and the right frequency is how often this strategy decided,
        # not how many bars the corpus held.
        per_year = evaluation.n_observations / years
        wanted = minimum_backtest_length(
            n_trials=max(2, evaluation.n_variants),
            target_sharpe=annualized(evaluation.best_sharpe, periods_per_year=per_year),
        )
    return replace(
        shell,
        evaluation=evaluation,
        regimes=coverage,
        interval=bounds,
        deflated_bootstrapped=bootstrapped,
        # Corrected against this sweep's own variants, not the family's history:
        # BHY reads the shape of the p-value distribution and the ledger stores
        # only Sharpe ratios computed over other samples, whose `n_observations`
        # differ. `deflated` is what prices the family's whole history.
        haircut=haircut_sharpe(
            evaluation.best_sharpe,
            trial_sharpes=[_pooled_sharpe(column) for column in pooled.returns],
            n_observations=evaluation.n_observations,
        ),
        min_backtest_years=wanted,
        sample_years=years,
        costs=_curve_of(curve, cost_levels, evaluation.best_index, evaluation.best_label),
    )


@dataclass(slots=True)
class _Level:
    """One cost level's running totals for one variant, across every instrument.

    Mutable and private: it exists only to keep the cost curve out of the trap
    of holding every instrument's bars until the winner is known.
    """

    returns: list[float] = field(default_factory=list)
    n_trades: int = 0
    total: float = 0.0

    def absorb(self, returns: Sequence[float], touches: Sequence[BarrierTouch | None]) -> None:
        """Fold one instrument's column in."""
        self.returns.extend(returns)
        for touch in touches:
            if touch is not None:
                self.n_trades += 1
                self.total += float(touch.realised_return)

    @property
    def expectancy(self) -> float:
        """Mean return per trade. Zero when nothing traded, never `nan`."""
        return self.total / self.n_trades if self.n_trades else 0.0


def _journal_trades(
    journal: TradeJournalPort,
    *,
    symbol: Symbol,
    bars: Sequence[Bar],
    variants: Sequence[Variant],
    observations: Observations,
    strategy: type[Strategy],
    costs: CostModel,
    config_hash: str,
    run_id: str,
    recorded_ns: Nanos,
) -> None:
    """Write one row per labelled trade for this instrument.

    **The join, which is the only subtle part.** `observations.trades[v]` is a
    tuple per variant, aligned to the candidate bars that were kept, and
    `observations.spans[i][0]` is the bar index of position `i` in that
    alignment. So the signal behind `trades[v][i]` is
    `variants[v].by_index()[spans[i][0]]`, and `None` means that variant did
    not fire on that bar. Getting this wrong would mis-attribute every barrier
    in the journal while leaving the counts plausible, which is why
    `test_journalled_rows_carry_the_barriers_of_their_own_signal` asserts the
    emitted barriers against the signal's rather than only counting rows.

    Only the reference cost level is journalled. The curve's levels relabel the
    same decisions at different charges, so emitting them too would multiply
    identical rows by the number of cost points and make every per-trade
    statistic silently weighted.
    """
    # `FlooredSpread` is the research default and the only estimator with a
    # `fraction`; a future measured estimator may have none, so the basis is
    # recorded as zero rather than guessed at.
    spread_fraction = Decimal(getattr(costs.spreads, "fraction", Decimal(0)))
    commission = Decimal(getattr(costs.fees, "per_share", Decimal(0)))
    indexed = [variant.by_index() for variant in variants]
    for position, (index, _) in enumerate(observations.spans):
        entry_ns = bars[index].ts_event
        for column, (variant, decisions) in enumerate(zip(variants, indexed, strict=True)):
            # By position, never `variants.index(variant)`: two variants with the
            # same label and signals compare equal, and `.index` would then
            # attribute both columns' trades to the first of them.
            touch = observations.trades[column][position]
            signal = decisions.get(index)
            if touch is None or signal is None:
                continue
            journal.append(
                TradeRecord(
                    run_id=run_id,
                    config_hash=config_hash,
                    recorded_ns=recorded_ns,
                    symbol=str(symbol),
                    strategy=strategy.name,
                    strategy_version=strategy.version,
                    variant=variant.label,
                    side=signal.side,
                    entry_ns=entry_ns,
                    exit_ns=touch.touched_at,
                    bars_held=touch.bars_held,
                    entry=touch.entry.value,
                    exit=touch.exit.value,
                    profit_target=signal.profit_target,
                    stop_loss=signal.stop_loss,
                    max_bars=signal.max_bars,
                    label=int(touch.label),
                    ambiguous=touch.ambiguous,
                    realised_return=touch.realised_return,
                    gross_return=touch.gross_return,
                    mfe=touch.mfe,
                    mae=touch.mae,
                    spread_fraction=spread_fraction,
                    commission_per_share=commission,
                )
            )


def _curve_of(
    curve: Sequence[Sequence[_Level]],
    cost_levels: Sequence[tuple[float, CostModel]],
    best_index: int,
    best_label: str,
) -> CostCurve | None:
    """Assemble the break-even curve for the winning variant, or `None`.

    Sorted by scale rather than trusting the caller's order, because
    `CostCurve.break_even` interpolates between adjacent points and an unsorted
    curve would interpolate between unrelated levels.
    """
    if not cost_levels:
        return None
    points = sorted(
        (
            CostPoint(
                scale=scale,
                n_trades=curve[level][best_index].n_trades,
                expectancy=curve[level][best_index].expectancy,
                sharpe=_pooled_sharpe(curve[level][best_index].returns),
            )
            for level, (scale, _) in enumerate(cost_levels)
        ),
        key=lambda point: point.scale,
    )
    return CostCurve(variant=best_label, points=tuple(points))


def _pooled_sharpe(returns: Sequence[float]) -> float:
    """Per-observation Sharpe, or 0.0 where variance is undefined.

    `moments` raises on zero variance rather than reporting a zero stdev, and the
    values have to be tested rather than the result read — the same trap
    `lab/evaluation.py` documents: a sparse strategy leaves whole columns flat,
    so this path is hit on every real measurement.
    """
    if len(returns) < 2 or len(set(returns)) < 2:
        return 0.0
    return sharpe_ratio(returns)


def _expectancy(column: Sequence[BarrierTouch | None]) -> tuple[int, float]:
    """Trade count and mean net return per trade, for one variant on one instrument.

    Args:
        column: That variant's labels, aligned on candidates, `None` where it
            had no view.

    Returns:
        `(n_trades, expectancy)`. A variant that never traded here returns
        `(0, 0.0)` — not a mean of an empty sample, and not `nan`, which would
        propagate silently into the printed table.

    Example:
        >>> _expectancy([None, None])
        (0, 0.0)
    """
    touches = [touch for touch in column if touch is not None]
    if not touches:
        return (0, 0.0)
    # float, matching `Evaluation.expectancy`: this is a reported statistic, not
    # a price or a P&L that the Decimal invariant governs.
    return (len(touches), float(sum(touch.realised_return for touch in touches) / len(touches)))


def _why_not(
    pooled: Observations,
    labels: Sequence[str],
    *,
    pbo_blocks: int,
    n_groups: int,
) -> str:
    """The reason this sample cannot be assessed, or an empty string.

    Checked here rather than caught from `assess`, because "the corpus is too
    thin" is a finding to report and an exception is a failure to fix. The
    conditions mirror the ones `assess` refuses on, deliberately: two places
    stating one rule is how they drift, so this one names them and lets `assess`
    remain the enforcement.
    """
    observations = pooled.n_observations
    if len(labels) < 2:
        return f"{_NOT_ENOUGH}: a sweep of {len(labels)} variant cannot be ranked"
    if observations == 0:
        return f"{_NOT_ENOUGH}: nothing fired anywhere in the window"
    if observations < max(pbo_blocks, n_groups):
        return (
            f"{_NOT_ENOUGH}: {observations} observations against "
            f"{pbo_blocks} CSCV blocks and {n_groups} CPCV groups"
        )
    silent = [
        label
        for label, column in zip(labels, pooled.trades, strict=True)
        if not any(touch is not None for touch in column)
    ]
    if silent:
        return f"{_NOT_ENOUGH}: no trade at all from {', '.join(silent)}"
    return ""
