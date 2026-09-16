"""Tests for `SeedSourcesFile`, the YAML-backed vendor file to `Symbol` map.

Style mirrors `tests/adapters/universe/test_universe_file.py`: heavier on
rejection than on happy paths, since the module's job is refusing a malformed
file loudly at startup.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from neurotrade.adapters.feeds.seed_sources import (
    InvalidSeedSourcesFile,
    SeedEntry,
    SeedSource,
    SeedSourcesFile,
)
from neurotrade.core.types import Symbol, Venue

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "seed_sources.yaml"
    path.write_text(text)
    return path


def _entry_yaml(*, source: str = "firstrate", ticker: str = "AAPL", venue: str = "NASDAQ") -> str:
    return (
        "version: 1\nentries:\n"
        f"  - source: {source}\n    ticker: {ticker}\n    venue: {venue}\n"
        f"    file: {ticker}_1min_sample_firstratedata.zip\n"
    )


# ── Well-formed files ─────────────────────────────────────────


def test_parses_entries_across_more_than_one_vendor(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "version: 1\nentries:\n"
        "  - source: firstrate\n    ticker: AAPL\n    venue: NASDAQ\n"
        "    file: AAPL_1min_sample_firstratedata.zip\n"
        "  - source: kibot\n    ticker: IBM\n    venue: NYSE\n"
        "    file: IBM_unadjusted.txt\n",
    )
    entries = SeedSourcesFile(path).entries()
    assert entries == (
        SeedEntry(
            source=SeedSource.FIRSTRATE,
            file="AAPL_1min_sample_firstratedata.zip",
            symbol=Symbol("AAPL", Venue.NASDAQ),
        ),
        SeedEntry(
            source=SeedSource.KIBOT, file="IBM_unadjusted.txt", symbol=Symbol("IBM", Venue.NYSE)
        ),
    )


def test_entries_filters_to_one_source(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "version: 1\nentries:\n"
        "  - source: firstrate\n    ticker: AAPL\n    venue: NASDAQ\n"
        "    file: AAPL_1min_sample_firstratedata.zip\n"
        "  - source: kibot\n    ticker: IBM\n    venue: NYSE\n"
        "    file: IBM_unadjusted.txt\n",
    )
    source = SeedSourcesFile(path)
    assert [e.symbol.ticker for e in source.entries(SeedSource.KIBOT)] == ["IBM"]
    assert [e.symbol.ticker for e in source.entries(SeedSource.FIRSTRATE)] == ["AAPL"]


def test_path_property_returns_the_constructor_argument(tmp_path: Path) -> None:
    path = _write(tmp_path, _entry_yaml())
    assert SeedSourcesFile(path).path == path


def test_repr_shows_path_and_entry_count(tmp_path: Path) -> None:
    path = _write(tmp_path, _entry_yaml())
    assert repr(SeedSourcesFile(path)) == f"SeedSourcesFile(path={path}, entries=1)"


# ── The committed seed_sources.yaml ───────────────────────────


def test_committed_seed_sources_file_loads_and_is_non_empty() -> None:
    entries = SeedSourcesFile(_REPO_ROOT / "config" / "seed_sources.yaml").entries()
    assert len(entries) > 0


def test_committed_seed_sources_file_has_both_vendors() -> None:
    source = SeedSourcesFile(_REPO_ROOT / "config" / "seed_sources.yaml")
    assert len(source.entries(SeedSource.FIRSTRATE)) > 0
    assert len(source.entries(SeedSource.KIBOT)) > 0


def test_committed_seed_sources_file_every_symbol_resolves_to_a_listing_venue() -> None:
    entries = SeedSourcesFile(_REPO_ROOT / "config" / "seed_sources.yaml").entries()
    for entry in entries:
        assert entry.symbol.venue is not Venue.SMART


# ── Missing or malformed file ─────────────────────────────────


def test_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SeedSourcesFile(tmp_path / "does-not-exist.yaml")


def test_top_level_not_a_mapping_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidSeedSourcesFile, match="must contain a mapping"):
        SeedSourcesFile(_write(tmp_path, "- firstrate\n- kibot\n"))


def test_unsupported_version_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 2\nentries:\n  - source: firstrate\n")
    with pytest.raises(InvalidSeedSourcesFile, match="unsupported version"):
        SeedSourcesFile(path)


def test_entries_not_a_list_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nentries: not-a-list\n")
    with pytest.raises(InvalidSeedSourcesFile, match="'entries' must be a list"):
        SeedSourcesFile(path)


def test_no_entries_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nentries: []\n")
    with pytest.raises(InvalidSeedSourcesFile, match="describes no entries"):
        SeedSourcesFile(path)


def test_entry_not_a_mapping_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nentries:\n  - firstrate\n")
    with pytest.raises(InvalidSeedSourcesFile, match="entry must be a mapping"):
        SeedSourcesFile(path)


def test_unknown_source_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, _entry_yaml(source="yahoo"))
    with pytest.raises(InvalidSeedSourcesFile, match="unknown source 'yahoo'"):
        SeedSourcesFile(path)


def test_unknown_venue_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, _entry_yaml(venue="LSE"))
    with pytest.raises(InvalidSeedSourcesFile, match="unknown venue 'LSE'"):
        SeedSourcesFile(path)


def test_smart_venue_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, _entry_yaml(venue="SMART"))
    with pytest.raises(InvalidSeedSourcesFile, match="order route"):
        SeedSourcesFile(path)


def test_duplicate_symbol_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "version: 1\nentries:\n"
        "  - source: firstrate\n    ticker: AAPL\n    venue: NASDAQ\n"
        "    file: AAPL_1min_sample_firstratedata.zip\n"
        "  - source: kibot\n    ticker: AAPL\n    venue: NASDAQ\n"
        "    file: AAPL_unadjusted.txt\n",
    )
    with pytest.raises(InvalidSeedSourcesFile, match=r"AAPL\.NASDAQ listed twice"):
        SeedSourcesFile(path)


def test_missing_file_name_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "version: 1\nentries:\n  - source: firstrate\n    ticker: AAPL\n    venue: NASDAQ\n",
    )
    with pytest.raises(InvalidSeedSourcesFile, match="no valid 'file' name"):
        SeedSourcesFile(path)


def test_non_string_ticker_raises() -> None:
    """`ON` is a real NASDAQ listing (ON Semiconductor) but parses as the
    boolean `True` unquoted — see `08-gotchas.doc.md`."""
    text = (
        "version: 1\nentries:\n  - source: firstrate\n    ticker: ON\n    venue: NASDAQ\n"
        "    file: ON_1min_sample_firstratedata.zip\n"
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "seed_sources.yaml"
        path.write_text(text)
        with pytest.raises(InvalidSeedSourcesFile, match="not a string"):
            SeedSourcesFile(path)
