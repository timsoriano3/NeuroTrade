"""Tests for the vendor download helpers.

Every test injects `FakeOpener` (`conftest.py`); none reach a network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.adapters.feeds.vendor_download import fetch_firstrate_samples, fetch_kibot_samples
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
