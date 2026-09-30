"""What the rest of the universe was doing, one tick ago — §5.4, §5.5.

Every strategy written so far reads one instrument. A cross-sectional strategy
does not: a stocks-in-play ranking, a relative-strength sort and a residual
reversion all need to know where *this* name sits among the others right now.
This module is the only place that view is assembled, so the point-in-time
question is answered once rather than in every strategy that wants it.

**The snapshot is always one tick stale, and that is not a limitation to fix.**
The engine merges instruments into one stream keyed on `(ts_event, seq,
symbol)`, so at 10:31 the bars arrive one after another in symbol order. Serving
"the cross-section at 10:31" to the first symbol dispatched would hand it the
other instruments' 10:31 bars before they had been published — lookahead, on
every bar, for every symbol but the last. Waiting for all of them is not
available either: a halted or thinly traded name may print nothing at 10:31, and
nothing distinguishes "not yet" from "never". So a tick is closed only when the
clock moves past it, and a strategy acting at 10:31 sees the universe as of
10:30. One minute of staleness changes no ranking that a one-minute strategy
should be making.

**Everything here is measured on one clock.** That is the reason this is not a
set of per-symbol features. `FeatureResolver` gives each instrument its own last
`N` bars, which for a name with a gap in the corpus is a different window from
its neighbour's — fine for an ATR, fatal for a ranking, because the instrument
with the missing bars would be ranked on a longer or shorter look than everyone
else. The tracker here indexes by *minute of session*, so every row in a
`CrossSection` covers the same wall-clock span or is absent.

**Betas are estimated on completed sessions, never intraday.** A beta fitted on
one-minute returns is dominated by microstructure — bid-ask bounce and
non-synchronous trading pull it toward zero (Scholes-Williams is the classical
correction) — and the quantity §5.4 and §5.5 want is the ordinary daily beta.
The tracker therefore folds one observation per instrument per completed
session and regresses over `BETA_MEMORY` of them. A name with fewer than that
has no beta and is *excluded* from a residual ranking rather than given a beta
of one, because assuming one is assuming the answer for exactly the names whose
answer we do not have.

**Nothing here decides anything.** A `CrossSection` is a set of numbers about
the universe; which of them is a trade is `strategies/`' business, and the
ranking helpers below return orderings rather than positions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Final, Protocol, runtime_checkable

from neurotrade.core.clock import Nanos
from neurotrade.core.types import Price, Symbol
from neurotrade.features.levels import SessionLevels

__all__ = [
    "BETA_MEMORY",
    "DEFAULT_TRAILING_MINUTES",
    "BenchmarkSource",
    "CrossSection",
    "CrossSectionTracker",
    "InstrumentSnapshot",
]


@runtime_checkable
class BenchmarkSource(Protocol):
    """Says what an instrument's return should be regressed on, on a session.

    A Protocol rather than a `Mapping`, because the answer is **point in time**:
    GICS moved the payment networks between sectors part way through the seed
    window, and a flat dictionary would have to be resolved as of some single
    date — as of the run's end, which is lookahead, or as of its start, which is
    stale by years. `core.sectors.SectorMap` satisfies this structurally.
    """

    def benchmark_of(self, symbol: Symbol, *, on: date) -> Symbol | None:
        """The benchmark, or `None` for an instrument that has none."""
        ...


BETA_MEMORY: Final = 60
"""Completed sessions a beta is estimated over.

Sixty is a quarter of trading days: long enough that a single outlier session
cannot dominate the covariance, short enough that a name whose beta really moved
— a company that changed what it does — is not held to what it was two years
ago. It is also the point below which the standard error of a daily beta is
wide enough that ranking on it is ranking on noise."""

DEFAULT_TRAILING_MINUTES: Final = 30
"""Window a trailing return is measured over, in minutes of session.

Thirty because §5.5's short-horizon reversal is stated at 30 to 60 minutes and
this is the shorter end. Minutes of session rather than bars, so a name missing
a print is measured over the same span as its neighbours rather than over a
longer one."""

_NANOS_PER_MINUTE: Final = 60_000_000_000

_RETURN_MEMORY: Final = 2 * BETA_MEMORY
"""Sessions of return history kept per instrument.

Twice the estimation window, so two instruments whose trading days do not line
up exactly still share `BETA_MEMORY` of them. Bounded at all, because the
tracker lives for a whole backtest and an unbounded dict per symbol is the
memory profile `lab/evaluation.pool` already refuses."""

_MIN_BETA_VARIANCE: Final = 1e-12
"""Below this the benchmark did not move over the estimation window, so a beta
is a division by noise. Reported as `None`, never as a large number."""


@dataclass(frozen=True, slots=True)
class InstrumentSnapshot:
    """One instrument's place in the cross-section, at one closed tick.

    Every field is a `float` or `None`: these are derived statistics used for
    ranking and residualising, never order levels, so the `Decimal` invariant
    does not reach them. A strategy that wants a price reads `context.levels`.

    Example:
        >>> row = InstrumentSnapshot(
        ...     symbol=AAPL, session_return=0.012, trailing_return=0.004,
        ...     relative_volume=2.5, beta=1.1,
        ... )
        >>> row.is_rankable
        True
    """

    symbol: Symbol  # the instrument
    session_return: float  # close over session open, minus one
    trailing_return: float | None  # return over the trailing window; None while cold
    relative_volume: float | None  # session volume against its own profile; None while cold
    beta: float | None  # daily beta to this instrument's benchmark; None while cold
    residual: float | None = None
    """Session return net of what this instrument's beta explains, against its
    benchmark's return over the same session. `None` without a beta or without a
    benchmark row, which is what keeps an unresidualised name out of a residual
    ranking. Filled by `CrossSectionTracker` at tick close, so every row in one
    `CrossSection` is residualised against the *same* benchmark tick."""

    @property
    def is_residualised(self) -> bool:
        """Whether this row can enter a residual ranking."""
        return self.residual is not None

    @property
    def is_rankable(self) -> bool:
        """Whether every field a cross-sectional rule normally reads is warm.

        A convenience for the common guard, not a rule: a ranking on session
        return alone is perfectly valid and should test `session_return` only.
        """
        return (
            self.trailing_return is not None
            and self.relative_volume is not None
            and self.beta is not None
        )

    def residual_against(self, benchmark_return: float) -> float | None:
        """This instrument's session return net of what its beta explains.

        Args:
            benchmark_return: The benchmark's session return over the same span.

        Returns:
            `session_return - beta * benchmark_return`, or `None` without a
            beta. `None` rather than the raw return: an unresidualised return in
            a column of residuals sits at an extreme of the ranking on any day
            the market moved, which is every day.

        Example:
            >>> row = InstrumentSnapshot(symbol=AAPL, session_return=0.02,
            ...                          trailing_return=None, relative_volume=None, beta=1.5)
            >>> round(row.residual_against(0.01), 6)
            0.005
        """
        if self.beta is None:
            return None
        return self.session_return - self.beta * benchmark_return


@dataclass(frozen=True, slots=True)
class CrossSection:
    """The universe as of one closed tick.

    Example:
        >>> section = CrossSection(as_of=1_000, rows={AAPL: InstrumentSnapshot(
        ...     symbol=AAPL, session_return=0.01, trailing_return=0.002,
        ...     relative_volume=3.0, beta=1.0)})
        >>> [symbol.ticker for symbol in section.symbols]
        ['AAPL']
    """

    as_of: Nanos  # the tick this describes; strictly before the bar being decided on
    rows: Mapping[Symbol, InstrumentSnapshot] = field(default_factory=dict)

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """Every instrument with a row, sorted.

        Sorted rather than in insertion order: a ranking built by iterating this
        would otherwise depend on which symbol the feed happened to deliver
        first, which the determinism invariant forbids.
        """
        return tuple(sorted(self.rows))

    def get(self, symbol: Symbol) -> InstrumentSnapshot | None:
        """One instrument's row, or `None` if it had not printed by this tick."""
        return self.rows.get(symbol)

    def rank_by(
        self,
        measure: Callable[[InstrumentSnapshot], float | None],
        *,
        descending: bool = True,
    ) -> tuple[Symbol, ...]:
        """Instruments ordered by a measure, skipping those it is `None` for.

        Args:
            measure: Reads one number off a row. `None` excludes the row, which
                is how a cold beta or a cold relative volume keeps a name out of
                a ranking rather than placing it at one end of it.
            descending: Largest first, the usual direction for a strength sort.

        Returns:
            The ordering. **Ties break on the symbol**, not on iteration order:
            at one-minute bars exact ties in a rounded measure are common, and
            an unstable ranking would make two runs of one backtest disagree.

        Example:
            >>> MSFT = Symbol("MSFT", AAPL.venue)
            >>> section = CrossSection(as_of=0, rows={
            ...     AAPL: InstrumentSnapshot(AAPL, 0.01, None, None, None),
            ...     MSFT: InstrumentSnapshot(MSFT, 0.03, None, None, None),
            ... })
            >>> [s.ticker for s in section.rank_by(lambda row: row.session_return)]
            ['MSFT', 'AAPL']
        """
        scored = [
            (value, symbol)
            for symbol in self.symbols
            if (value := measure(self.rows[symbol])) is not None
        ]
        scored.sort(key=lambda pair: (-pair[0] if descending else pair[0], str(pair[1])))
        return tuple(symbol for _, symbol in scored)

    def rank_of(
        self,
        symbol: Symbol,
        measure: Callable[[InstrumentSnapshot], float | None],
        *,
        descending: bool = True,
    ) -> tuple[int, int] | None:
        """Where one instrument sits in a ranking, and how many were ranked.

        Returns:
            `(position, total)` with position zero-based, or `None` when the
            instrument is not in the ranking. The total travels with the
            position because "third" means nothing without it — third of four is
            weak and third of fifty is strong, and a strategy comparing against
            a fixed count would change meaning every time the universe grew.
        """
        ordering = self.rank_by(measure, descending=descending)
        if symbol not in ordering:
            return None
        return (ordering.index(symbol), len(ordering))

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"CrossSection(as_of={self.as_of}, {len(self.rows)} instruments)"


class CrossSectionTracker:
    """Builds a `CrossSection` from the session levels the engine already keeps.

    Fed after `SessionLevelTracker`, from the same bars, so the session anchors
    have exactly one implementation. What this adds is the parts that are only
    definable across instruments: a trailing return on a shared clock, and a
    beta against a benchmark.

    Example:
        >>> tracker = CrossSectionTracker()
        >>> tracker.snapshot() is None          # nothing has closed yet
        True
    """

    __slots__ = (
        "_benchmark",
        "_benchmarks",
        "_closed",
        "_pending",
        "_returns",
        "_sessions",
        "_tick",
        "_trailing",
        "_trailing_minutes",
    )

    def __init__(
        self,
        *,
        benchmarks: BenchmarkSource | None = None,
        trailing_minutes: int = DEFAULT_TRAILING_MINUTES,
    ) -> None:
        """Wire the tracker to whatever says what an instrument regresses on.

        Args:
            benchmarks: Resolves an instrument's benchmark for a session — a
                `core.sectors.SectorMap`, normally. `None` means nothing has a
                benchmark, so nothing has a beta or a residual, which is the
                state a run of single-instrument strategies is in.
            trailing_minutes: Window for `trailing_return`, in minutes of
                session.

        Raises:
            ValueError: If `trailing_minutes` is not positive.
        """
        if trailing_minutes < 1:
            raise ValueError(f"trailing_minutes {trailing_minutes} must be at least 1")
        self._benchmarks = benchmarks
        self._trailing_minutes = trailing_minutes
        self._tick: Nanos | None = None
        self._pending: dict[Symbol, InstrumentSnapshot] = {}
        self._closed: CrossSection | None = None
        # Per symbol: the session it is in, its return profile by minute of
        # session, and one observation per completed session for the beta.
        self._sessions: dict[Symbol, date] = {}
        self._trailing: dict[Symbol, dict[int, float]] = {}
        # The benchmark each instrument resolved to, at the session it last
        # printed in. Cached rather than re-resolved in `_beta` and
        # `_residual_of`, so both read the same answer for one tick.
        self._benchmark: dict[Symbol, Symbol | None] = {}
        # Keyed by session **date**, not appended to a list. Within one tick an
        # instrument may have rolled into the new session while its benchmark
        # has not — they are dispatched in symbol order — so two lists of
        # returns can differ in length by one, and pairing them from the tail
        # then regresses each session on the one before it. That produced a beta
        # of -1.62 where the fixture said 2.0, and is pinned in
        # `test_a_beta_of_two_is_recovered_from_sessions_that_move_twice_as_far`.
        self._returns: dict[Symbol, dict[date, float]] = {}

    def observe(self, levels: SessionLevels, symbol: Symbol, ts: Nanos) -> None:
        """Fold one instrument's current levels into the tick being assembled.

        Args:
            levels: That instrument's session anchors as of `ts`.
            symbol: The instrument. Taken separately because `SessionLevels`
                does not carry one — it is a snapshot of a session, not of an
                instrument.
            ts: The bar's close. A `ts` later than the tick being assembled
                closes that tick and starts a new one.

        Raises:
            ValueError: If `ts` moves backwards. The feed is ts-ordered by
                construction, so this means the engine was wired wrong, and a
                cross-section assembled out of order is the kind of defect that
                surfaces months later as an unreproducible backtest.
        """
        if self._tick is not None and ts < self._tick:
            raise ValueError(f"cross-section went backwards: {ts} after {self._tick}")
        if self._tick is not None and ts > self._tick:
            self._close_tick()
        self._tick = ts
        self._roll_session(symbol, levels)

        minute = int((ts - levels.open_ns) // _NANOS_PER_MINUTE)
        session_return = _fraction(levels.close, levels.session_open)
        self._trailing.setdefault(symbol, {})[minute] = session_return
        # Resolved for *this* session, so a reclassification is respected. The
        # beta is then this instrument's beta to the sector it is in **now**,
        # estimated on the sessions the two share — which is the right quantity:
        # what V's return will do against XLF, not what it did against XLK.
        self._benchmark[symbol] = (
            None
            if self._benchmarks is None
            else self._benchmarks.benchmark_of(symbol, on=levels.session_date)
        )
        self._pending[symbol] = InstrumentSnapshot(
            symbol=symbol,
            session_return=session_return,
            trailing_return=self._trailing_return(symbol, minute, session_return),
            relative_volume=levels.relative_volume_from_open,
            beta=self._beta(symbol),
        )

    def snapshot(self) -> CrossSection | None:
        """The most recently *closed* tick, or `None` before one has closed.

        Never the tick being assembled. See the module docstring: serving the
        open tick would hand the first instrument dispatched the other
        instruments' bars before they were published.
        """
        return self._closed

    def _close_tick(self) -> None:
        """Freeze the tick just finished and start collecting the next.

        The rows are carried forward rather than cleared: an instrument that did
        not print this minute is stale, not absent, and dropping it would make a
        thinly traded name vanish from every ranking on the minutes it is
        quietest — which is a selection effect on exactly the names a
        stocks-in-play filter is about.
        """
        assert self._tick is not None  # only called once a tick has been seen
        # Residuals are filled here rather than in `observe`, so every row is
        # residualised against the benchmark's reading at the *same* tick.
        # Filling them per bar would residualise the names dispatched before the
        # benchmark against its previous tick and the rest against this one —
        # not lookahead, but a ranking whose column is measured two ways.
        rows = {
            symbol: replace(row, residual=self._residual_of(symbol, row))
            for symbol, row in self._pending.items()
        }
        self._closed = CrossSection(as_of=self._tick, rows=rows)

    def _residual_of(self, symbol: Symbol, row: InstrumentSnapshot) -> float | None:
        """The row's return net of its beta times its benchmark's, or `None`.

        `None` when the instrument has no benchmark, no beta, or a benchmark
        that has not printed yet this run — never the raw return, which in a
        column of residuals sits at an extreme on any day the market moved.
        """
        benchmark = self._benchmark.get(symbol)
        if benchmark is None:
            return None
        reference = self._pending.get(benchmark)
        if reference is None:
            return None
        return row.residual_against(reference.session_return)

    def _roll_session(self, symbol: Symbol, levels: SessionLevels) -> None:
        """Record a completed session's return when the date changes."""
        held = self._sessions.get(symbol)
        if held == levels.session_date:
            return
        if held is not None:
            profile = self._trailing.get(symbol) or {}
            if profile:
                # The last observation of the session that just ended is its
                # open-to-close return. Open-to-close rather than close-to-close
                # because the strategies reading this beta are flat overnight, so
                # the overnight gap is not a risk they take and should not be a
                # risk the beta prices.
                history = self._returns.setdefault(symbol, {})
                history[held] = profile[max(profile)]
                if len(history) > _RETURN_MEMORY:
                    # Trimmed on the oldest dates rather than on insertion
                    # order: a name that starts trading late inserts its first
                    # session after its benchmark's hundredth.
                    for stale in sorted(history)[: len(history) - _RETURN_MEMORY]:
                        del history[stale]
        self._sessions[symbol] = levels.session_date
        self._trailing[symbol] = {}

    def _trailing_return(self, symbol: Symbol, minute: int, session_return: float) -> float | None:
        """Return over `trailing_minutes` of session, or `None` when the window is short.

        Measured between two points of the *session-return* profile, so it is
        exact rather than compounded: `(1 + r_now) / (1 + r_then) - 1`. Keyed by
        minute, so an instrument that missed a print is `None` here rather than
        being handed the return over a longer window than its neighbours got.
        """
        earlier = minute - self._trailing_minutes
        if earlier < 0:
            return None
        profile = self._trailing.get(symbol)
        if profile is None or earlier not in profile:
            return None
        base = 1.0 + profile[earlier]
        if base <= 0:
            return None
        return (1.0 + session_return) / base - 1.0

    def _beta(self, symbol: Symbol) -> float | None:
        """Ordinary least-squares beta of this instrument on its benchmark.

        `cov(own, benchmark) / var(benchmark)` over the sessions both have, which
        is the OLS slope. Paired on the sessions the two **share**, by date: a
        name listed later than its benchmark has a shorter history, and one that
        was halted for a day has a hole in the middle of its own. Either would
        silently shift every observation by one if the two were lined up by
        position — which is not hypothetical, because within a tick an
        instrument may have rolled into the new session while its benchmark has
        not.

        Returns:
            The slope, or `None` when the instrument has no benchmark, when
            fewer than `BETA_MEMORY` sessions are shared, or when the benchmark
            did not move over them. `None` excludes the name from a residual
            ranking, which is the safe direction — a default of one would assume
            the answer for exactly the names whose answer is missing.
        """
        benchmark = self._benchmark.get(symbol)
        if benchmark is None:
            return None
        own = self._returns.get(symbol, {})
        market = self._returns.get(benchmark, {})
        shared = sorted(own.keys() & market.keys())[-BETA_MEMORY:]
        if len(shared) < BETA_MEMORY:
            return None
        return _slope([own[day] for day in shared], [market[day] for day in shared])

    def __repr__(self) -> str:
        return f"CrossSectionTracker({len(self._pending)} pending, tick={self._tick})"


def _fraction(close: Price, base: Price) -> float:
    """`close / base - 1`, as a float.

    The division happens in `Decimal` and only the result is converted, so no
    binary artefact enters before the rounding does. A return is a derived
    statistic and therefore a float — the boundary the precision invariant
    draws — while the prices it came from stay `Price`.
    """
    return float(close.value / base.value) - 1.0


def _slope(own: Sequence[float], market: Sequence[float]) -> float | None:
    """OLS slope of `own` on `market`, or `None` when the market did not move."""
    n = len(market)
    mean_own = sum(own) / n
    mean_market = sum(market) / n
    covariance = sum((a - mean_own) * (b - mean_market) for a, b in zip(own, market, strict=True))
    variance = sum((b - mean_market) ** 2 for b in market)
    if variance < _MIN_BETA_VARIANCE:
        return None
    return covariance / variance
