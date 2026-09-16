"""Tests for the universe-history artifact.

The round-trip of an empty session is the one with teeth: drop that row and
`as_of` silently carries the previous membership across a date on which the
screen admitted nobody.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neurotrade.adapters.universe.universe_history_parquet import (
    HISTORY_SCHEMA,
    UniverseHistoryStore,
)
from neurotrade.core.clock import SimClock
from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import UniverseHistory, UniverseMembership

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
JULY_1 = date(2024, 7, 1)
JULY_2 = date(2024, 7, 2)


def a_history(*, survivorship_biased: bool = True) -> UniverseHistory:
    return UniverseHistory(
        [
            UniverseMembership(JULY_1, [AAPL, SHOP]),
            UniverseMembership(JULY_2, []),
        ],
        survivorship_biased=survivorship_biased,
    )


def a_store(tmp_path: Path) -> UniverseHistoryStore:
    return UniverseHistoryStore(tmp_path / "universe")


def write(store: UniverseHistoryStore, history: UniverseHistory) -> Path:
    return store.write(history, universe_digest="abc123", config_hash="cfg_1", clock=SimClock(42))


# ── round trip ───────────────────────────────────────────────


def test_a_history_survives_the_round_trip(tmp_path: Path) -> None:
    store = a_store(tmp_path)
    write(store, a_history())
    assert store.read().digest == a_history().digest


def test_a_session_that_admitted_nobody_comes_back_empty_not_missing(
    tmp_path: Path,
) -> None:
    store = a_store(tmp_path)
    write(store, a_history())
    read_back = store.read()
    assert read_back.dates == (JULY_1, JULY_2)
    assert read_back.as_of(JULY_2) == ()


def test_the_bias_flag_survives(tmp_path: Path) -> None:
    store = a_store(tmp_path)
    write(store, a_history(survivorship_biased=True))
    assert store.read().survivorship_biased


def test_provenance_is_stamped_on_the_file(tmp_path: Path) -> None:
    """Which candidates and which thresholds produced this membership."""
    store = a_store(tmp_path)
    path = write(store, a_history())
    metadata = pq.read_table(path).schema.metadata
    assert metadata[b"universe_digest"] == b"abc123"
    assert metadata[b"config_hash"] == b"cfg_1"
    assert metadata[b"built_at_ns"] == b"42"


def test_writing_creates_the_root(tmp_path: Path) -> None:
    store = a_store(tmp_path)
    assert not (tmp_path / "universe").exists()  # constructing a reader leaves nothing
    write(store, a_history())
    assert store.path.exists()


# ── refusals ─────────────────────────────────────────────────


def test_reading_nothing_is_an_error_not_an_empty_history(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no universe history"):
        a_store(tmp_path).read()


def test_an_edited_artifact_is_refused(tmp_path: Path) -> None:
    """A truncated artifact is worse than a missing one: it changes which names
    a backtest could trade and nothing downstream would notice."""
    store = a_store(tmp_path)
    path = write(store, a_history())
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 1), path)  # metadata kept, a row removed

    with pytest.raises(ValueError, match="but was written as"):
        store.read()


def test_an_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    store = a_store(tmp_path)
    store._root.mkdir(parents=True)
    schema = HISTORY_SCHEMA.with_metadata({"schema_version": "99"})
    pq.write_table(pa.Table.from_pylist([], schema=schema), store.path)

    with pytest.raises(ValueError, match="schema version 99"):
        store.read()
