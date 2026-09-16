"""Tests for the vendor download helpers.

Every test injects `FakeOpener` (`conftest.py`); none reach a network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.adapters.feeds.vendor_download import (
    fetch_firstrate_samples,
    fetch_kibot_samples,
    latest_snapshot,
    read_manifest,
    snapshots,
)
from neurotrade.core.clock import SimClock
from tests.adapters.feeds.conftest import FakeOpener, FakeResponse

_JAN_1_2023_NS = 1_672_531_200_000_000_000  # 2023-01-01T00:00:00Z
_KIBOT_PAGE_URL = "https://www.kibot.com/free-historical-intraday-data.html"
_FRD_AAPL_URL = "https://frd001.s3-us-east-2.amazonaws.com/AAPL_1min_sample_firstratedata.zip"


# ── FirstRateData ──────────────────────────────────────────────


def test_firstrate_download_writes_file_and_manifest(tmp_path: Path) -> None:
    data = b"pretend-zip-bytes"
    opener = FakeOpener({_FRD_AAPL_URL: FakeResponse(data)})
    clock = SimClock(_JAN_1_2023_NS)

    paths = fetch_firstrate_samples(["AAPL"], raw_dir=tmp_path, clock=clock, opener=opener)

    destination = (
        tmp_path / "vendor" / "firstrate" / "2023-01-01" / "AAPL_1min_sample_firstratedata.zip"
    )
    assert paths == (destination,)
    assert destination.read_bytes() == data

    manifest = json.loads((destination.parent / "manifest.json").read_text())
    entry = manifest["AAPL_1min_sample_firstratedata.zip"]
    assert entry["url"] == _FRD_AAPL_URL
    assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["bytes"] == len(data)
    assert entry["fetched_at"] == _JAN_1_2023_NS
    assert entry["adjusted"] == "unknown"


def test_firstrate_fetches_each_ticker_in_order(tmp_path: Path) -> None:
    opener = FakeOpener(
        {
            "https://frd001.s3-us-east-2.amazonaws.com/AAPL_1min_sample_firstratedata.zip": (
                FakeResponse(b"aapl")
            ),
            "https://frd001.s3-us-east-2.amazonaws.com/SPY_1min_sample_firstratedata.zip": (
                FakeResponse(b"spy")
            ),
        }
    )
    paths = fetch_firstrate_samples(
        ["AAPL", "SPY"], raw_dir=tmp_path, clock=SimClock(_JAN_1_2023_NS), opener=opener
    )
    assert [path.name for path in paths] == [
        "AAPL_1min_sample_firstratedata.zip",
        "SPY_1min_sample_firstratedata.zip",
    ]


def test_firstrate_refuses_to_overwrite_an_existing_file(tmp_path: Path) -> None:
    clock = SimClock(_JAN_1_2023_NS)
    fetch_firstrate_samples(
        ["AAPL"],
        raw_dir=tmp_path,
        clock=clock,
        opener=FakeOpener({_FRD_AAPL_URL: FakeResponse(b"first")}),
    )

    with pytest.raises(FeedError, match="refusing to overwrite"):
        fetch_firstrate_samples(
            ["AAPL"],
            raw_dir=tmp_path,
            clock=clock,
            opener=FakeOpener({_FRD_AAPL_URL: FakeResponse(b"second")}),
        )


# ── Kibot ───────────────────────────────────────────────────────


def test_kibot_download_resolves_filename_from_content_disposition(tmp_path: Path) -> None:
    page = b'<a href="https://api.kibot.com/?get=abc123">IBM 1-min unadjusted</a>'
    data = b"01/03/2023,09:30,1,2,3,4,5\n"
    opener = FakeOpener(
        {
            _KIBOT_PAGE_URL: FakeResponse(page),
            "https://api.kibot.com/?get=abc123": FakeResponse(
                data, content_disposition='attachment; filename="IBM_unadjusted.txt"'
            ),
        }
    )
    clock = SimClock(_JAN_1_2023_NS)

    paths = fetch_kibot_samples(
        ["IBM_unadjusted.txt"], raw_dir=tmp_path, clock=clock, opener=opener
    )

    destination = tmp_path / "vendor" / "kibot" / "2023-01-01" / "IBM_unadjusted.txt"
    assert paths == (destination,)
    assert destination.read_bytes() == data
    manifest = json.loads((destination.parent / "manifest.json").read_text())
    assert manifest["IBM_unadjusted.txt"]["adjusted"] == "unadjusted"
    assert manifest["IBM_unadjusted.txt"]["sha256"] == hashlib.sha256(data).hexdigest()


def test_kibot_skips_links_that_do_not_match_a_wanted_name(tmp_path: Path) -> None:
    page = (
        b'<a href="https://api.kibot.com/?get=other">x</a>'
        b'<a href="https://api.kibot.com/?get=abc123">y</a>'
    )
    opener = FakeOpener(
        {
            _KIBOT_PAGE_URL: FakeResponse(page),
            "https://api.kibot.com/?get=other": FakeResponse(
                b"nope", content_disposition='attachment; filename="OTHER_unadjusted.txt"'
            ),
            "https://api.kibot.com/?get=abc123": FakeResponse(
                b"wanted", content_disposition='attachment; filename="IBM_unadjusted.txt"'
            ),
        }
    )
    paths = fetch_kibot_samples(
        ["IBM_unadjusted.txt"], raw_dir=tmp_path, clock=SimClock(_JAN_1_2023_NS), opener=opener
    )
    assert [path.name for path in paths] == ["IBM_unadjusted.txt"]


def test_kibot_raises_loudly_when_a_wanted_file_matches_no_link(tmp_path: Path) -> None:
    page = b'<a href="https://api.kibot.com/?get=abc123">something else</a>'
    opener = FakeOpener(
        {
            _KIBOT_PAGE_URL: FakeResponse(page),
            "https://api.kibot.com/?get=abc123": FakeResponse(
                b"data", content_disposition='attachment; filename="OTHER_unadjusted.txt"'
            ),
        }
    )
    with pytest.raises(FeedError, match=r"IBM_unadjusted\.txt"):
        fetch_kibot_samples(
            ["IBM_unadjusted.txt"],
            raw_dir=tmp_path,
            clock=SimClock(_JAN_1_2023_NS),
            opener=opener,
        )


# ── Reading a snapshot back ─────────────────────────────────────


def _snapshot(tmp_path: Path, source: str, day: str) -> Path:
    folder = tmp_path / "vendor" / source / day
    folder.mkdir(parents=True)
    return folder


def test_snapshots_are_returned_oldest_first(tmp_path: Path) -> None:
    for day in ("2026-09-13", "2026-08-02", "2026-09-02"):
        _snapshot(tmp_path, "kibot", day)
    assert [path.name for path in snapshots(tmp_path, "kibot")] == [
        "2026-08-02",
        "2026-09-02",
        "2026-09-13",
    ]


def test_snapshots_ignore_anything_not_named_as_a_date(tmp_path: Path) -> None:
    """A stray file or a scratch folder is not a reason to refuse to ingest."""
    _snapshot(tmp_path, "kibot", "2026-09-13")
    _snapshot(tmp_path, "kibot", "scratch")
    (tmp_path / "vendor" / "kibot" / "notes.txt").touch()
    assert [path.name for path in snapshots(tmp_path, "kibot")] == ["2026-09-13"]


def test_snapshots_of_a_vendor_never_fetched_is_empty(tmp_path: Path) -> None:
    assert snapshots(tmp_path, "firstrate") == ()


def test_latest_snapshot_is_the_newest_day(tmp_path: Path) -> None:
    for day in ("2026-09-02", "2026-09-13"):
        _snapshot(tmp_path, "kibot", day)
    folder = latest_snapshot(tmp_path, "kibot")
    assert folder is not None
    assert folder.name == "2026-09-13"


def test_latest_snapshot_is_none_when_nothing_was_fetched(tmp_path: Path) -> None:
    assert latest_snapshot(tmp_path, "kibot") is None


def test_a_fetch_writes_a_manifest_that_reads_back(tmp_path: Path) -> None:
    """The round trip that matters: what `seed ingest` reads is what
    `seed fetch` wrote, field for field."""
    opener = FakeOpener({_FRD_AAPL_URL: FakeResponse(b"zip-bytes")})
    written = fetch_firstrate_samples(
        ["AAPL"], raw_dir=tmp_path, clock=SimClock(_JAN_1_2023_NS), opener=opener
    )
    records = read_manifest(written[0].parent)
    assert len(records) == 1
    record = records[0]
    assert record.name == "AAPL_1min_sample_firstratedata.zip"
    assert record.url == _FRD_AAPL_URL
    assert record.sha256 == hashlib.sha256(b"zip-bytes").hexdigest()
    assert record.size_bytes == len(b"zip-bytes")
    assert record.fetched_at == _JAN_1_2023_NS
    assert record.adjusted == "unknown"


def test_read_manifest_sorts_records_by_name(tmp_path: Path) -> None:
    opener = FakeOpener(
        {
            _FRD_AAPL_URL: FakeResponse(b"aapl"),
            _FRD_AAPL_URL.replace("AAPL", "MSFT"): FakeResponse(b"msft"),
        }
    )
    fetch_firstrate_samples(
        ["MSFT", "AAPL"], raw_dir=tmp_path, clock=SimClock(_JAN_1_2023_NS), opener=opener
    )
    records = read_manifest(latest_snapshot(tmp_path, "firstrate") or tmp_path)
    assert [record.name for record in records] == [
        "AAPL_1min_sample_firstratedata.zip",
        "MSFT_1min_sample_firstratedata.zip",
    ]


def test_read_manifest_rejects_a_snapshot_with_no_manifest(tmp_path: Path) -> None:
    """A snapshot with no manifest has no provenance, and rows from it could
    not be explained afterwards."""
    folder = _snapshot(tmp_path, "kibot", "2026-09-13")
    with pytest.raises(FeedError, match=r"no manifest\.json"):
        read_manifest(folder)


def test_read_manifest_rejects_an_unparseable_manifest(tmp_path: Path) -> None:
    folder = _snapshot(tmp_path, "kibot", "2026-09-13")
    (folder / "manifest.json").write_text("{not json")
    with pytest.raises(FeedError, match="does not parse"):
        read_manifest(folder)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("[]", id="a list, not an object"),
        pytest.param('"text"', id="a bare string"),
    ],
)
def test_read_manifest_rejects_a_manifest_that_is_not_keyed_by_name(
    tmp_path: Path, payload: str
) -> None:
    folder = _snapshot(tmp_path, "kibot", "2026-09-13")
    (folder / "manifest.json").write_text(payload)
    with pytest.raises(FeedError, match="keyed by file name"):
        read_manifest(folder)


def test_read_manifest_rejects_an_entry_that_is_not_an_object(tmp_path: Path) -> None:
    folder = _snapshot(tmp_path, "kibot", "2026-09-13")
    (folder / "manifest.json").write_text(json.dumps({"IBM_unadjusted.txt": "nope"}))
    with pytest.raises(FeedError, match="is not an object"):
        read_manifest(folder)


@pytest.mark.parametrize("dropped", ["name", "url", "sha256", "bytes", "fetched_at", "adjusted"])
def test_read_manifest_rejects_an_entry_missing_any_field(tmp_path: Path, dropped: str) -> None:
    folder = _snapshot(tmp_path, "kibot", "2026-09-13")
    record = {
        "name": "IBM_unadjusted.txt",
        "url": "https://api.kibot.com/?get=x",
        "sha256": "ab",
        "bytes": 2,
        "fetched_at": 0,
        "adjusted": "unadjusted",
    }
    del record[dropped]
    (folder / "manifest.json").write_text(json.dumps({"IBM_unadjusted.txt": record}))
    with pytest.raises(FeedError, match="is incomplete"):
        read_manifest(folder)
