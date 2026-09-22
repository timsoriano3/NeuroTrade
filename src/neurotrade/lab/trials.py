"""The trial ledger — how many times did we look? (§8)

`lab/significance.py` can deflate a Sharpe ratio for the size of the search
that produced it, but only if something knows that size. This is that
something: a thin layer over `TrialLedgerPort` that stamps a trial with the
run's clock and configuration, appends it, and hands the accumulated history
back to the deflation.

**Record before you know whether you care.** The intended order is: run the
test, `record` it, then `deflate`. Recording afterwards, once a result has
proved interesting, reproduces exactly the bias the ledger exists to remove —
the trials that get written down become the trials that worked, and the
expected maximum computed from them is the expected maximum of a filtered
sample, which is far too low.

**Families, not one global count.** Trials deflate within a family. The
expected best of 500 opening-range variants says nothing about a mean-reversion
idea tried once, and pooling them would punish the second for the first's
search. A family is a search space: name it for the space, not for the run.

**The clock is a port here too.** `record` takes its timestamp from the
injected `Clock`, so a research script driven by a `SimClock` produces the same
trial ids every time it runs. That is the determinism invariant applied to
research: a ledger that gained a different id on every replay could not be
rebuilt or compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from neurotrade.core.clock import Clock
from neurotrade.core.ids import TrialId
from neurotrade.core.ports import TrialLedgerPort
from neurotrade.core.trials import Trial, TrialSource
from neurotrade.lab.significance import deflated_sharpe_ratio, expected_max_sharpe

__all__ = [
    "TrialLedger",
]


@dataclass(frozen=True, slots=True)
class TrialLedger:
    """Records hypotheses and feeds their count into the deflation.

    Example:
        >>> from neurotrade.core.clock import SimClock
        >>> class MemoryLedger:
        ...     def __init__(self): self._trials = []
        ...     def append(self, trial): self._trials.append(trial)
        ...     def trials(self, family=None):
        ...         return tuple(t for t in self._trials if family in (None, t.family))
        >>> clock = SimClock(1_000)
        >>> ledger = TrialLedger(store=MemoryLedger(), clock=clock, config_hash="cfg_a")
        >>> for i in range(3):
        ...     clock.advance_ns(1)
        ...     _ = ledger.record(
        ...         hypothesis=f"orb {i}m", family="opening-range", sharpe=0.01 * i,
        ...         n_observations=1000,
        ...     )
        >>> ledger.count("opening-range")
        3
    """

    store: TrialLedgerPort  # where trials are appended and read back from
    clock: Clock  # supplies `recorded_ns`; never `datetime.now()`
    config_hash: str  # configuration in force, stamped on every trial recorded here

    def record(
        self,
        *,
        hypothesis: str,
        family: str,
        sharpe: float,
        n_observations: int,
        source: TrialSource = TrialSource.MANUAL,
        n_paths: int = 0,
    ) -> Trial:
        """Append one tested hypothesis and return the record written.

        Args:
            hypothesis: What was tested, in plain English. This is what a
                reader six months later has to reconstruct the search from.
            family: The search space. Deflation happens within it.
            sharpe: Per-period Sharpe ratio observed, **net of costs**. A gross
                Sharpe recorded here would deflate a candidate against a
                benchmark built from numbers nobody could have earned.
            n_observations: Returns the Sharpe was computed from.
            source: What ran the test.
            n_paths: CPCV paths behind the number; 0 for a single backtest.

        Returns:
            The `Trial` as stored, including its derived id.

        Raises:
            ValueError: If the hypothesis or family is blank, or the counts are
                negative — see `Trial.__post_init__`.

        Example:
            >>> from neurotrade.core.clock import SimClock
            >>> class MemoryLedger:
            ...     def __init__(self): self._trials = []
            ...     def append(self, trial): self._trials.append(trial)
            ...     def trials(self, family=None): return tuple(self._trials)
            >>> ledger = TrialLedger(
            ...     store=MemoryLedger(), clock=SimClock(1_000), config_hash="cfg_a"
            ... )
            >>> ledger.record(
            ...     hypothesis="orb 15m", family="opening-range",
            ...     sharpe=0.04, n_observations=1200,
            ... ).trial_id
            TrialId(value='trl_2baa928f60e9d0a4')
        """
        recorded_ns = self.clock.now_ns()
        trial = Trial(
            trial_id=TrialId.derive(
                hypothesis=hypothesis, config_hash=self.config_hash, recorded_ns=recorded_ns
            ),
            recorded_ns=recorded_ns,
            hypothesis=hypothesis,
            family=family,
            source=source,
            config_hash=self.config_hash,
            sharpe=sharpe,
            n_observations=n_observations,
            n_paths=n_paths,
        )
        self.store.append(trial)
        return trial

    def count(self, family: str | None = None) -> int:
        """How many trials the ledger holds.

        Args:
            family: Restrict to one search space; all of them when omitted.

        Returns:
            The trial count — the `N` the deflation is computed against.
        """
        return len(self.store.trials(family))

    def sharpes(self, family: str | None = None) -> tuple[float, ...]:
        """The per-period Sharpe ratio of every trial, in record order.

        Args:
            family: Restrict to one search space.

        Returns:
            One value per trial. Both the count and the spread matter: a wide
            search over wildly different ideas produces a higher expected
            maximum than a narrow one with the same trial count.
        """
        return tuple(trial.sharpe for trial in self.store.trials(family))

    def hurdle(self, family: str) -> float:
        """The Sharpe ratio a new candidate in this family has to clear.

        The expected maximum across the family's recorded trials — what the
        best of them would have scored if none of them worked. Useful on its
        own: it is the number to quote when explaining why a backtest showing
        a Sharpe of 0.08 is not a finding.

        Args:
            family: The search space.

        Returns:
            The hurdle in per-period Sharpe units. Zero when the family holds
            one trial or fewer: one look is no search.

        Raises:
            ValueError: If the family holds no trials.

        Example:
            >>> from neurotrade.core.clock import SimClock
            >>> class MemoryLedger:
            ...     def __init__(self): self._trials = []
            ...     def append(self, trial): self._trials.append(trial)
            ...     def trials(self, family=None):
            ...         return tuple(t for t in self._trials if family in (None, t.family))
            >>> clock = SimClock(1_000)
            >>> ledger = TrialLedger(store=MemoryLedger(), clock=clock, config_hash="cfg_a")
            >>> for i in range(200):
            ...     clock.advance_ns(1)
            ...     _ = ledger.record(
            ...         hypothesis=f"orb {i}", family="opening-range",
            ...         sharpe=0.001 * i - 0.1, n_observations=1000,
            ...     )
            >>> round(ledger.hurdle("opening-range"), 4)
            0.1601
        """
        values = self.sharpes(family)
        if not values:
            raise ValueError(f"family {family!r} holds no trials; record before you deflate")
        n_trials = len(values)
        if n_trials == 1:
            return 0.0
        mean = math.fsum(values) / n_trials
        variance = math.fsum((value - mean) ** 2 for value in values) / (n_trials - 1)
        return expected_max_sharpe(n_trials=n_trials, trial_variance=variance)

    def deflate(
        self,
        observed: float,
        *,
        family: str,
        n_observations: int,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Deflated Sharpe ratio for a candidate, against its family's history.

        Call this **after** recording the candidate, so that it counts itself.
        A search of 200 that deflates against 199 is not wrong by much, but the
        habit of excluding the candidate is how the count drifts downward.

        Args:
            observed: Per-period Sharpe ratio of the candidate, net of costs.
            family: The search space to deflate against.
            n_observations: Returns the candidate's Sharpe was computed from.
            skew: Third standardized moment of the candidate's returns.
            kurtosis: Fourth standardized moment, not excess.

        Returns:
            Confidence in `(0, 1)` that the candidate beats the best its own
            search would have produced from noise.

        Raises:
            ValueError: If the family holds no trials.
        """
        values = self.sharpes(family)
        if not values:
            raise ValueError(f"family {family!r} holds no trials; record before you deflate")
        return deflated_sharpe_ratio(
            observed,
            trial_sharpes=values,
            n_observations=n_observations,
            skew=skew,
            kurtosis=kurtosis,
        )
