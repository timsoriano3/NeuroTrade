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
from dataclasses import dataclass, replace
from typing import Final

from neurotrade.core.clock import Nanos, SimClock, to_datetime
from neurotrade.core.costs import CostModel
from neurotrade.core.events import BarInterval
from neurotrade.core.intent import Intent
from neurotrade.core.ports import CalendarPort, StoragePort
from neurotrade.core.trials import TrialSource
from neurotrade.core.types import Quantity, Symbol
from neurotrade.features.registry import FeatureRegistry
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
from neurotrade.lab.trials import TrialLedger
from neurotrade.strategies.base import Strategy
from neurotrade.strategies.context import MarketContext

__all__ = ["DEFAULT_EMBARGO_NS", "Measurement", "SymbolRun", "measure_strategy"]

DEFAULT_EMBARGO_NS: Final = 86_400_000_000_000
"""One calendar day of embargo, in nanoseconds.

Spans are nanoseconds once a universe is pooled, so the embargo has to be too. A
day is the shortest defensible choice for a day strategy: a label that opens the
morning after a test block still shares that block's overnight news, and the
features behind it are built from the same sessions.
"""

_NOT_ENOUGH = "the corpus supports no verdict"


@dataclass(frozen=True, slots=True)
class SymbolRun:
    """What one instrument contributed.

    Example:
        >>> SymbolRun(symbol=AAPL, n_bars=0, n_signals=(0,), n_observations=0, dropped=0).traded
        False
    """

    symbol: Symbol  # the instrument
    n_bars: int  # bars the corpus held for it in the window
    n_signals: tuple[int, ...]  # decisions taken, per variant, in sweep order
    n_observations: int  # candidate bars scored — the union of the variants' decisions
    dropped: int  # candidates discarded because a label could not form (end of corpus)

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

    @property
    def is_measured(self) -> bool:
        """Whether a verdict came back at all."""
        return self.evaluation is not None

    @property
    def n_signals(self) -> int:
        """Decisions taken across every instrument and variant."""
        return sum(sum(run.n_signals) for run in self.runs)

    def __str__(self) -> str:
        head = f"{self.strategy}@{self.version} [{', '.join(self.variants)}]"
        if self.evaluation is None:
            return f"{head}  NOT MEASURED — {self.reason}"
        gated = "gated" if self.regime_gated else "ungated"
        return f"{head}  {self.evaluation}  sessions={self.n_sessions} {gated}"


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
    source: TrialSource = TrialSource.MANUAL,
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
            context=MarketContext(features=features, calendar=calendar, interval=interval),
            ungated=ungated,
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
    for symbol in ordered:
        bars = list(store.read_bars(symbol, interval, start, end))
        if not bars:
            runs.append(
                SymbolRun(
                    symbol=symbol,
                    n_bars=0,
                    n_signals=tuple(0 for _ in labels),
                    n_observations=0,
                    dropped=0,
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
        parts.append(
            Part(
                key=str(symbol),
                timestamps=[bar.ts_event for bar in bars],
                observations=observations,
            )
        )
        runs.append(
            SymbolRun(
                symbol=symbol,
                n_bars=len(bars),
                n_signals=counts,
                n_observations=observations.n_observations,
                dropped=len(observations.dropped),
            )
        )

    pooled = pool(parts)
    sessions = len({to_datetime(opened).date() for opened, _ in pooled.spans})
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
    )
    return replace(shell, evaluation=evaluation)


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
