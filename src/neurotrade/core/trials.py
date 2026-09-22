"""One hypothesis, tested — the record the deflation counts (§8).

The deflated Sharpe ratio in `lab/significance.py` needs to know how many
things were tried. That number is not knowable from the results that were kept:
a search of two thousand parameter sets that reports its best three looks, from
the outside, exactly like three good ideas. So every test writes a record here,
before its result is known to be interesting, and the count comes from the
ledger rather than from memory.

**Why the record is in `core`.** `TrialLedgerPort` in `core.ports` has to name
the type it stores, and `core` may import nothing above it. The ledger's
*behaviour* — recording, counting, feeding the deflation — is research logic
and lives in `lab/trials.py`; only the record itself is here.

**The honesty this buys is procedural, not technical.** Nothing forces a
process to append before it tests. What the ledger gives is a single place
where an under-count is visible: a discovery run that reports a Sharpe of 2.0
with eleven trials in the ledger is making a claim about its own search that
can be checked against `discovery/`'s own logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from neurotrade.core.clock import Nanos
from neurotrade.core.ids import TrialId

__all__ = [
    "Trial",
    "TrialSource",
]


class TrialSource(StrEnum):
    """What ran the test.

    Kept as a field rather than inferred, because the deflation question is
    "how large was the search space", and a sweep of 5,000 configurations and a
    human trying one idea contribute to it very differently. `StrEnum` so the
    value serializes as itself.

    Example:
        >>> TrialSource.DISCOVERY.value
        'discovery'
    """

    MANUAL = "manual"  # a person tried one idea
    SWEEP = "sweep"  # a parameter grid, every point of which is a trial
    DISCOVERY = "discovery"  # automated alpha search (§9.2); easiest source to under-count


@dataclass(frozen=True, slots=True)
class Trial:
    """One hypothesis tested against the corpus.

    Frozen, and appended to an append-only store: a trial that turned out badly
    is not deleted, because deleting it is precisely how the search space
    shrinks to fit the answer.

    Example:
        >>> trial = Trial(
        ...     trial_id=TrialId.derive(
        ...         hypothesis="orb 15m", config_hash="cfg_a", recorded_ns=1_000
        ...     ),
        ...     recorded_ns=1_000,
        ...     hypothesis="orb 15m",
        ...     family="opening-range",
        ...     source=TrialSource.MANUAL,
        ...     config_hash="cfg_a",
        ...     sharpe=0.04,
        ...     n_observations=1200,
        ...     n_paths=5,
        ... )
        >>> trial.family
        'opening-range'
    """

    trial_id: TrialId  # derived from hypothesis, config hash and time; stable across a replay
    recorded_ns: Nanos  # when the trial was recorded, from the run's Clock — never wall clock
    hypothesis: str  # plain-English statement of what was tested
    family: str  # search space this belongs to; trials deflate against their own family
    source: TrialSource  # what ran it
    config_hash: str  # configuration in force, from `neurotrade.config.config_hash`
    sharpe: float  # per-period Sharpe ratio observed, net of costs
    n_observations: int  # returns the Sharpe was computed from
    n_paths: int  # CPCV paths behind the number; 0 when it came from a single backtest

    def __post_init__(self) -> None:
        """Reject records that cannot mean anything.

        Raises:
            ValueError: If the hypothesis or family is blank, or if the counts
                are negative. A blank family would silently pool one search
                space into another and understate both.
        """
        if not self.hypothesis.strip():
            raise ValueError("a trial needs a hypothesis; an unlabelled trial cannot be audited")
        if not self.family.strip():
            raise ValueError("a trial needs a family; deflation is computed within one")
        if self.n_observations < 0:
            raise ValueError(f"n_observations {self.n_observations} must not be negative")
        if self.n_paths < 0:
            raise ValueError(f"n_paths {self.n_paths} must not be negative")
