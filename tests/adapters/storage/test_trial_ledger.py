"""Tests for the on-disk trial ledger.

Two properties matter more than the rest: an append never loses what was
already there, and a line that cannot be read back is an error rather than a
skipped record. Both failures would lower the trial count, which lowers the
deflation hurdle, which approves strategies that should have been rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurotrade.adapters.storage.trial_ledger import CorruptTrialLedger, TrialLedgerStore
from neurotrade.core.ids import TrialId
from neurotrade.core.ports import TrialLedgerPort
from neurotrade.core.trials import Trial, TrialSource


def make(index: int, *, family: str = "opening-range", sharpe: float = 0.04) -> Trial:
    """One trial, distinguished by `index`."""
    return Trial(
        trial_id=TrialId.derive(
            hypothesis=f"variant {index}", config_hash="cfg_a", recorded_ns=index
        ),
        recorded_ns=index,
        hypothesis=f"variant {index}",
        family=family,
        source=TrialSource.SWEEP,
        config_hash="cfg_a",
        sharpe=sharpe,
        n_observations=1_200,
        n_paths=5,
    )


# ── Round trip ───────────────────────────────────────────────


def test_the_store_satisfies_the_port(tmp_path: Path) -> None:
    assert isinstance(TrialLedgerStore(tmp_path / "ledger.jsonl"), TrialLedgerPort)


def test_a_trial_survives_the_round_trip_unchanged(tmp_path: Path) -> None:
    store = TrialLedgerStore(tmp_path / "ledger.jsonl")
    trial = make(1)
    store.append(trial)
    assert store.trials() == (trial,)


def test_records_come_back_in_append_order(tmp_path: Path) -> None:
    store = TrialLedgerStore(tmp_path / "ledger.jsonl")
    for index in range(5):
        store.append(make(index))
    assert [trial.recorded_ns for trial in store.trials()] == [0, 1, 2, 3, 4]


def test_an_empty_ledger_is_the_ordinary_starting_state(tmp_path: Path) -> None:
    assert TrialLedgerStore(tmp_path / "never-written.jsonl").trials() == ()


def test_the_parent_directory_is_created_on_first_append(tmp_path: Path) -> None:
    store = TrialLedgerStore(tmp_path / "nested" / "deeper" / "ledger.jsonl")
    store.append(make(1))
    assert store.path.exists()


def test_filtering_by_family_leaves_the_others_on_disk(tmp_path: Path) -> None:
    store = TrialLedgerStore(tmp_path / "ledger.jsonl")
    store.append(make(1, family="opening-range"))
    store.append(make(2, family="mean-reversion"))
    assert len(store.trials("opening-range")) == 1
    assert len(store.trials()) == 2


# ── Append-only ──────────────────────────────────────────────


def test_reopening_a_ledger_continues_it_rather_than_truncating(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    TrialLedgerStore(path).append(make(1))
    TrialLedgerStore(path).append(make(2))
    assert len(TrialLedgerStore(path).trials()) == 2


def test_every_append_adds_exactly_one_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = TrialLedgerStore(path)
    for index in range(3):
        store.append(make(index))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_identical_trials_serialize_byte_identically(tmp_path: Path) -> None:
    # Keys are sorted on write, so a rebuilt ledger can be diffed against the
    # old one instead of re-read record by record.
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    TrialLedgerStore(first).append(make(1))
    TrialLedgerStore(second).append(make(1))
    assert first.read_bytes() == second.read_bytes()


# ── Damage ───────────────────────────────────────────────────


def test_blank_lines_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = TrialLedgerStore(path)
    store.append(make(1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n   \n")
    store.append(make(2))
    assert len(store.trials()) == 2


@pytest.mark.parametrize(
    "line",
    [
        "not json at all",
        '{"trial_id": "trl_0000000000000000"}',
        '{"trial_id": "bad_prefix", "recorded_ns": 1, "hypothesis": "h", "family": "f",'
        ' "source": "sweep", "config_hash": "c", "sharpe": 0.1, "n_observations": 1,'
        ' "n_paths": 0}',
        '{"trial_id": "trl_0000000000000000", "recorded_ns": 1, "hypothesis": "h",'
        ' "family": "f", "source": "not-a-source", "config_hash": "c", "sharpe": 0.1,'
        ' "n_observations": 1, "n_paths": 0}',
        '{"trial_id": "trl_0000000000000000", "recorded_ns": 1, "hypothesis": "",'
        ' "family": "f", "source": "sweep", "config_hash": "c", "sharpe": 0.1,'
        ' "n_observations": 1, "n_paths": 0}',
    ],
)
def test_an_unreadable_record_is_fatal_not_skipped(tmp_path: Path, line: str) -> None:
    # Skipping would silently lower the trial count, which is the one
    # direction of error this subsystem exists to prevent.
    path = tmp_path / "ledger.jsonl"
    store = TrialLedgerStore(path)
    store.append(make(1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with pytest.raises(CorruptTrialLedger, match="cannot be read back"):
        store.trials()


def test_the_error_names_the_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = TrialLedgerStore(path)
    store.append(make(1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{broken\n")
    with pytest.raises(CorruptTrialLedger, match=r"ledger\.jsonl:2"):
        store.trials()
