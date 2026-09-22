"""Tests for the trial ledger.

The ledger's whole value is that it cannot quietly shrink. So the tests here
are mostly about counting: that a recorded trial stays recorded, that the count
the deflation sees is the count that was appended, and that a family's hurdle
rises as its search widens.
"""

from __future__ import annotations

import pytest

from neurotrade.core.clock import SimClock
from neurotrade.core.ports import TrialLedgerPort
from neurotrade.core.trials import Trial, TrialSource
from neurotrade.lab.significance import deflated_sharpe_ratio
from neurotrade.lab.trials import TrialLedger


class MemoryLedger:
    """An in-memory `TrialLedgerPort`, for tests that are not about storage."""

    def __init__(self) -> None:
        self.appended: list[Trial] = []

    def append(self, trial: Trial) -> None:
        self.appended.append(trial)

    def trials(self, family: str | None = None) -> tuple[Trial, ...]:
        return tuple(t for t in self.appended if family in (None, t.family))


def build(clock: SimClock | None = None) -> tuple[TrialLedger, MemoryLedger, SimClock]:
    """A ledger over an in-memory store, with a clock the test controls."""
    sim = clock or SimClock(1_000)
    store = MemoryLedger()
    return TrialLedger(store=store, clock=sim, config_hash="cfg_a"), store, sim


def fill(ledger: TrialLedger, clock: SimClock, count: int, *, family: str, spread: float) -> None:
    """Record `count` trials whose Sharpe ratios fan out around zero."""
    for index in range(count):
        clock.advance_ns(1)
        ledger.record(
            hypothesis=f"{family} variant {index}",
            family=family,
            sharpe=spread * (index / max(count - 1, 1) - 0.5),
            n_observations=1_000,
            source=TrialSource.SWEEP,
        )


# ── The record ───────────────────────────────────────────────


def test_the_in_memory_store_satisfies_the_port() -> None:
    assert isinstance(MemoryLedger(), TrialLedgerPort)


def test_recording_stamps_the_clock_and_the_config_hash() -> None:
    ledger, store, clock = build(SimClock(7_000))
    trial = ledger.record(
        hypothesis="orb 15m", family="opening-range", sharpe=0.04, n_observations=1_200
    )
    assert (trial.recorded_ns, trial.config_hash) == (7_000, "cfg_a")
    assert store.appended == [trial]
    assert clock.now_ns() == 7_000


def test_the_same_trial_under_a_sim_clock_derives_the_same_id() -> None:
    # Determinism applied to research: a rerun must not inflate the ledger
    # with records that are the same trial wearing new ids.
    first, _, _ = build(SimClock(1_000))
    second, _, _ = build(SimClock(1_000))
    left = first.record(
        hypothesis="orb 15m", family="opening-range", sharpe=0.04, n_observations=1_200
    )
    right = second.record(
        hypothesis="orb 15m", family="opening-range", sharpe=0.04, n_observations=1_200
    )
    assert left.trial_id == right.trial_id


def test_trials_differing_only_in_time_are_distinct_records() -> None:
    ledger, _, clock = build()
    first = ledger.record(hypothesis="orb", family="f", sharpe=0.0, n_observations=10)
    clock.advance_ns(1)
    second = ledger.record(hypothesis="orb", family="f", sharpe=0.0, n_observations=10)
    assert first.trial_id != second.trial_id
    assert ledger.count() == 2


def test_the_default_source_is_manual() -> None:
    ledger, _, _ = build()
    trial = ledger.record(hypothesis="orb", family="f", sharpe=0.0, n_observations=10)
    assert trial.source is TrialSource.MANUAL


@pytest.mark.parametrize(
    ("hypothesis", "family", "message"),
    [
        ("", "f", "needs a hypothesis"),
        ("   ", "f", "needs a hypothesis"),
        ("orb", "", "needs a family"),
        ("orb", "  ", "needs a family"),
    ],
)
def test_an_unlabelled_trial_is_rejected(hypothesis: str, family: str, message: str) -> None:
    ledger, store, _ = build()
    with pytest.raises(ValueError, match=message):
        ledger.record(hypothesis=hypothesis, family=family, sharpe=0.0, n_observations=10)
    assert store.appended == []


@pytest.mark.parametrize(("observations", "paths"), [(-1, 0), (10, -2)])
def test_negative_counts_are_rejected(observations: int, paths: int) -> None:
    ledger, _, _ = build()
    with pytest.raises(ValueError, match="must not be negative"):
        ledger.record(
            hypothesis="orb",
            family="f",
            sharpe=0.0,
            n_observations=observations,
            n_paths=paths,
        )


# ── Counting and families ────────────────────────────────────


def test_counting_is_per_family_and_overall() -> None:
    ledger, _, clock = build()
    fill(ledger, clock, 12, family="opening-range", spread=0.1)
    fill(ledger, clock, 5, family="mean-reversion", spread=0.1)
    assert (ledger.count("opening-range"), ledger.count("mean-reversion")) == (12, 5)
    assert ledger.count() == 17


def test_sharpes_come_back_in_record_order() -> None:
    ledger, _, clock = build()
    for index in range(4):
        clock.advance_ns(1)
        ledger.record(hypothesis=f"h{index}", family="f", sharpe=float(index), n_observations=10)
    assert ledger.sharpes("f") == (0.0, 1.0, 2.0, 3.0)


# ── The hurdle ───────────────────────────────────────────────


def test_a_single_trial_sets_no_hurdle() -> None:
    ledger, _, _ = build()
    ledger.record(hypothesis="orb", family="f", sharpe=0.3, n_observations=10)
    assert ledger.hurdle("f") == 0.0


def test_the_hurdle_rises_as_the_search_widens() -> None:
    narrow, _, narrow_clock = build()
    fill(narrow, narrow_clock, 30, family="f", spread=0.1)
    wide, _, wide_clock = build()
    fill(wide, wide_clock, 600, family="f", spread=0.1)
    assert narrow.hurdle("f") < wide.hurdle("f")


def test_an_unsearched_family_has_no_hurdle_to_quote() -> None:
    ledger, _, clock = build()
    fill(ledger, clock, 5, family="opening-range", spread=0.1)
    with pytest.raises(ValueError, match="holds no trials"):
        ledger.hurdle("mean-reversion")


# ── Deflation ────────────────────────────────────────────────


def test_deflate_uses_the_family_history_and_nothing_else() -> None:
    ledger, _, clock = build()
    fill(ledger, clock, 40, family="opening-range", spread=0.2)
    fill(ledger, clock, 900, family="mean-reversion", spread=2.0)
    expected = deflated_sharpe_ratio(
        0.1, trial_sharpes=ledger.sharpes("opening-range"), n_observations=500
    )
    assert ledger.deflate(0.1, family="opening-range", n_observations=500) == expected


def test_a_wider_search_lowers_the_deflated_sharpe_for_the_same_result() -> None:
    narrow, _, narrow_clock = build()
    fill(narrow, narrow_clock, 20, family="f", spread=0.1)
    wide, _, wide_clock = build()
    fill(wide, wide_clock, 2_000, family="f", spread=0.1)
    assert wide.deflate(0.1, family="f", n_observations=500) < narrow.deflate(
        0.1, family="f", n_observations=500
    )


def test_deflating_against_an_empty_family_is_refused() -> None:
    # Silently treating "no recorded trials" as "one trial" would report an
    # undeflated number under a deflated name.
    ledger, _, _ = build()
    with pytest.raises(ValueError, match="record before you deflate"):
        ledger.deflate(0.1, family="opening-range", n_observations=500)
