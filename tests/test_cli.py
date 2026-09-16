"""Tests for the command-line entry point.

The stdout/stderr split is the one worth guarding: `neurotrade config hash` is
meant to be pipeable, and a stray log line on stdout would silently corrupt
whatever consumes it.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pyarrow.dataset as ds
import pytest
import structlog
from typer.testing import CliRunner

from neurotrade import __version__
from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
from neurotrade.adapters.feeds import yfinance_daily
from neurotrade.adapters.feeds.seed_sources import SeedSource
from neurotrade.adapters.feeds.vendor_download import latest_snapshot, read_manifest
from neurotrade.adapters.feeds.yfinance_daily import DailyRow
from neurotrade.adapters.ibkr.connection import IbkrConnectionError
from neurotrade.adapters.storage.duckdb_catalog import DuckDBCatalog
from neurotrade.cli import _SEED_PROVENANCE, _seed_feed, _SeedJob, app
from neurotrade.config import Profile, config_hash, load_settings
from neurotrade.core.clock import LiveClock, SimClock, to_nanos
from neurotrade.core.events import Bar, BarInterval, Event
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.logs import clear_context
from tests.adapters.feeds.conftest import FakeOpener, FakeResponse

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """The CLI configures logging globally; do not leak it between tests."""
    yield
    clear_context()
    structlog.reset_defaults()


# ── Basics ───────────────────────────────────────────────────


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_bare_invocation_shows_help() -> None:
    """A tool that does nothing useful with no arguments should say so."""
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout


def test_unknown_command_fails() -> None:
    assert runner.invoke(app, ["definitely-not-a-command"]).exit_code != 0


# ── Profile resolution ───────────────────────────────────────


def test_profile_flag_selects_the_profile() -> None:
    result = runner.invoke(app, ["--profile", "live", "config", "hash"])
    assert result.exit_code == 0
    assert result.stdout.strip() == config_hash(load_settings(Profile.LIVE))


def test_profile_defaults_to_research(monkeypatch: pytest.MonkeyPatch) -> None:
    """The only default that cannot spend money."""
    monkeypatch.delenv("NEUROTRADE_PROFILE", raising=False)
    result = runner.invoke(app, ["config", "hash"])
    assert result.stdout.strip() == config_hash(load_settings(Profile.RESEARCH))


def test_profile_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROTRADE_PROFILE", "paper")
    result = runner.invoke(app, ["config", "hash"])
    assert result.stdout.strip() == config_hash(load_settings(Profile.PAPER))


def test_profile_flag_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROTRADE_PROFILE", "paper")
    result = runner.invoke(app, ["--profile", "live", "config", "hash"])
    assert result.stdout.strip() == config_hash(load_settings(Profile.LIVE))


def test_unknown_profile_is_rejected() -> None:
    assert runner.invoke(app, ["--profile", "backtest", "config", "hash"]).exit_code != 0


# ── Output discipline ────────────────────────────────────────


def test_hash_output_is_exactly_one_pipeable_line() -> None:
    """Anything extra on stdout breaks `$(neurotrade config hash)`."""
    result = runner.invoke(app, ["--profile", "paper", "config", "hash"])
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("cfg_")


def test_config_show_reports_the_resolved_values() -> None:
    result = runner.invoke(app, ["--profile", "paper", "config", "show"])
    assert "profile" in result.stdout
    assert "storage.data_root" in result.stdout
    assert "config_hash" in result.stdout
    assert "run_id" in result.stdout


def test_config_show_reflects_an_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command exists to answer 'what am I actually running with'."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", "/Volumes/nvme/neurotrade")
    result = runner.invoke(app, ["--profile", "paper", "config", "show"])
    assert "/Volumes/nvme/neurotrade" in result.stdout


# ── Startup wiring ───────────────────────────────────────────


def test_every_command_gets_settings_and_a_run_id() -> None:
    """The root callback resolves both, so no command builds its own."""
    result = runner.invoke(app, ["--profile", "paper", "config", "show"])
    assert result.exit_code == 0
    assert "run_" in result.stdout


def test_the_reported_hash_matches_the_reported_settings() -> None:
    """A command must not report one configuration and run under another."""
    result = runner.invoke(app, ["--profile", "live", "config", "show"])
    assert "allow_live_orders    True" in result.stdout
    assert config_hash(load_settings(Profile.LIVE)) in result.stdout


# ── Replay ───────────────────────────────────────────────────


def _record_session(root: Path, session: str) -> list[Event]:
    """Write a small session to the log location the CLI will look in."""
    from neurotrade.adapters.storage.event_store import EventStore

    events: list[Event] = [
        Bar(
            symbol=Symbol("AAPL", Venue.NASDAQ),
            ts_event=1_000 + index,
            ts_init=1_000 + index,
            interval=BarInterval.MIN_1,
            open=Price(f"{100 + index}"),
            high=Price(f"{101 + index}"),
            low=Price(f"{99 + index}"),
            close=Price(f"{100 + index}.5"),
            volume=Quantity(1_000),
        )
        for index in range(5)
    ]
    with EventStore(root / "events" / f"{session}.jsonl") as store:
        for event in events:
            store.append(event)
    return events


def test_replay_prints_only_the_digest_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """So `$(neurotrade replay …)` captures one clean line."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _record_session(tmp_path, "2026-03-16")

    result = runner.invoke(app, ["replay", "--session", "2026-03-16"])
    assert result.exit_code == 0
    assert len(result.stdout.strip().splitlines()) == 1


def test_replaying_twice_gives_the_same_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate G1, through the interface an operator actually uses."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _record_session(tmp_path, "2026-03-16")

    first = runner.invoke(app, ["replay", "-s", "2026-03-16"]).stdout.strip()
    second = runner.invoke(app, ["replay", "-s", "2026-03-16"]).stdout.strip()
    assert first == second
    assert len(first) == 32


def test_different_sessions_give_different_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _record_session(tmp_path, "2026-03-16")

    from neurotrade.adapters.storage.event_store import EventStore

    with EventStore(tmp_path / "events" / "2026-03-17.jsonl") as store:
        store.append(
            Bar(
                symbol=Symbol("AAPL", Venue.NASDAQ),
                ts_event=9_999,
                ts_init=9_999,
                interval=BarInterval.MIN_1,
                open=Price("500"),
                high=Price("500"),
                low=Price("500"),
                close=Price("500"),
                volume=Quantity(1),
            )
        )

    monday = runner.invoke(app, ["replay", "-s", "2026-03-16"]).stdout.strip()
    tuesday = runner.invoke(app, ["replay", "-s", "2026-03-17"]).stdout.strip()
    assert monday != tuesday


def test_a_missing_session_exits_non_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Silently printing a digest for nothing would be worse than failing."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["replay", "-s", "1999-01-01"])
    assert result.exit_code == 1


def test_session_log_path_follows_the_configured_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the data root moves the logs with it — no path is hardcoded."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    settings = load_settings(Profile.RESEARCH)
    assert settings.storage.session_log("2026-03-16") == tmp_path / "events" / "2026-03-16.jsonl"


def test_replay_accepts_a_log_path_directly(tmp_path: Path) -> None:
    """A log worth replaying does not always live where this machine keeps its own."""
    _record_session(tmp_path, "2026-03-16")
    result = runner.invoke(app, ["replay", "--log", str(tmp_path / "events" / "2026-03-16.jsonl")])
    assert result.exit_code == 0
    assert len(result.stdout.strip()) == 32


def test_replay_requires_exactly_one_source() -> None:
    """Neither, or both, is a mistake worth naming rather than guessing at."""
    assert runner.invoke(app, ["replay"]).exit_code == 2
    assert (
        runner.invoke(app, ["replay", "-s", "2026-03-16", "-l", "somewhere.jsonl"]).exit_code == 2
    )


def test_the_committed_fixture_is_replayable_from_the_cli() -> None:
    """`make replay` on a fresh clone has to work; nothing writes logs yet."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "session.jsonl"
    result = runner.invoke(app, ["replay", "--log", str(fixture)])
    assert result.exit_code == 0
    assert result.stdout.strip() == "4f58fe2c99cd26dc7cbb8faf033a39d1"


def test_a_missing_session_suggests_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error that only says 'not found' leaves the reader stuck."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["replay", "-s", "2026-03-16"])
    assert result.exit_code == 1
    assert "tests/fixtures/session.jsonl" in result.output


def test_a_missing_session_lists_the_ones_that_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _record_session(tmp_path, "2026-03-16")
    result = runner.invoke(app, ["replay", "-s", "1999-01-01"])
    assert "2026-03-16" in result.output


# ── IBKR backfill ────────────────────────────────────────────

# A known NASDAQ week, verified against the real calendar rather than assumed:
# 07-01, 07-02, 07-05, 07-08, 07-09 and 07-10 are ordinary 390-bar sessions;
# 07-03 is the pre-July-4th half day, 210 bars; the 4th and the weekend are
# closed. Picking real dates means a fixture disagreeing with the calendar
# fails the test instead of silently drifting.
_HALF_DAY = date(2024, 7, 3)


def _write_universe(tmp_path: Path, venues: dict[str, list[str]]) -> Path:
    """A tiny universe file in the format `config/universe.yaml` uses."""
    lines = ["version: 1", "venues:"]
    for venue, tickers in venues.items():
        lines.append(f"  {venue}:")
        lines.extend(f"    - {ticker}" for ticker in tickers)
    path = tmp_path / "universe.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _session_bars(symbol: Symbol, session_date: date) -> tuple[Bar, ...]:
    """Every bar a complete session holds, valid and closing inside it.

    Built from the real `VenueCalendar` so a fixture cannot silently disagree
    with the calendar the crawler itself plans against.
    """
    session = VenueCalendar().session(symbol.venue, session_date)
    assert session is not None, f"{session_date} is not a session on {symbol.venue}"
    count = session.expected_bars(BarInterval.MIN_1)
    return tuple(
        Bar(
            symbol=symbol,
            ts_event=session.open_ns + (index + 1) * BarInterval.MIN_1.nanos,
            ts_init=session.open_ns + (index + 1) * BarInterval.MIN_1.nanos,
            interval=BarInterval.MIN_1,
            open=Price("100"),
            high=Price("101"),
            low=Price("99"),
            close=Price("100.5"),
            volume=Quantity(10),
        )
        for index in range(count)
    )


@dataclass
class _ConnectionLog:
    """What happened to the fake `IbkrConnection` during one invocation."""

    constructed: int = 0
    connected: bool = False
    disconnected: bool = False
    raise_on_connect: Exception | None = None


@dataclass
class _FeedLog:
    """What happened to the fake `IbkrMarketData` during one invocation."""

    calls: list[tuple[Symbol, BarInterval, int, int]] = field(default_factory=list)
    bars: dict[Symbol, tuple[Bar, ...]] = field(default_factory=dict)
    raise_on_fetch: Exception | None = None


def _patch_ibkr(
    monkeypatch: pytest.MonkeyPatch, connection_log: _ConnectionLog, feed_log: _FeedLog
) -> None:
    """Replace `cli.IbkrConnection` and `cli.IbkrMarketData` with fakes that
    never touch a socket, wired to logs a test can inspect afterwards."""

    class FakeConnection:
        def __init__(self, settings: object) -> None:
            del settings
            connection_log.constructed += 1

        async def connect(self) -> None:
            if connection_log.raise_on_connect is not None:
                raise connection_log.raise_on_connect
            connection_log.connected = True

        def disconnect(self) -> None:
            connection_log.disconnected = True

    class FakeMarketData:
        def __init__(self, connection: object, clock: object) -> None:
            del connection, clock

        async def fetch_bars(
            self, symbol: Symbol, interval: BarInterval, start: int, end: int
        ) -> tuple[Bar, ...]:
            feed_log.calls.append((symbol, interval, start, end))
            if feed_log.raise_on_fetch is not None:
                raise feed_log.raise_on_fetch
            return tuple(
                bar for bar in feed_log.bars.get(symbol, ()) if start <= bar.ts_event < end
            )

        async def is_connected(self) -> bool:
            return True

    monkeypatch.setattr("neurotrade.cli.IbkrConnection", FakeConnection)
    monkeypatch.setattr("neurotrade.cli.IbkrMarketData", FakeMarketData)


# ── backfill: argument validation ─────────────────────────────


def test_backfill_requires_start() -> None:
    """No default is deliberate — see the command's docstring — so a missing
    `--start` must be a usage error, not a silent guess at how much history to
    fill."""
    result = runner.invoke(app, ["ibkr", "backfill"])
    assert result.exit_code == 2
    assert "Usage" in result.stderr


def test_backfill_rejects_an_end_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    connection_log = _ConnectionLog()
    _patch_ibkr(monkeypatch, connection_log, _FeedLog())

    result = runner.invoke(
        app, ["ibkr", "backfill", "--start", "2024-07-08", "--end", "2024-07-03"]
    )

    assert result.exit_code == 2
    assert "before" in result.stderr
    assert connection_log.constructed == 0


def test_backfill_rejects_a_missing_universe_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    connection_log = _ConnectionLog()
    _patch_ibkr(monkeypatch, connection_log, _FeedLog())

    result = runner.invoke(
        app,
        [
            "ibkr",
            "backfill",
            "--start",
            "2024-07-03",
            "--end",
            "2024-07-03",
            "--universe",
            str(tmp_path / "does-not-exist.yaml"),
        ],
    )

    assert result.exit_code == 2
    assert connection_log.constructed == 0


def test_backfill_rejects_a_malformed_universe_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A universe silently short by a venue would produce a corpus silently
    short by a venue (see `UniverseFile`) — the command must refuse outright."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text("version: 1\nvenues:\n  NASDAQ: not-a-list\n")
    connection_log = _ConnectionLog()
    _patch_ibkr(monkeypatch, connection_log, _FeedLog())

    result = runner.invoke(
        app,
        [
            "ibkr",
            "backfill",
            "--start",
            "2024-07-03",
            "--end",
            "2024-07-03",
            "--universe",
            str(universe_path),
        ],
    )

    assert result.exit_code == 2
    assert connection_log.constructed == 0


# ── backfill: Gateway reachability ─────────────────────────────


def test_backfill_exits_nonzero_when_gateway_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL"]})
    connection_log = _ConnectionLog(raise_on_connect=IbkrConnectionError("gateway unreachable"))
    _patch_ibkr(monkeypatch, connection_log, _FeedLog())

    result = runner.invoke(
        app,
        [
            "ibkr",
            "backfill",
            "--start",
            "2024-07-03",
            "--end",
            "2024-07-03",
            "--universe",
            str(universe_path),
        ],
    )

    assert result.exit_code == 1
    assert "gateway unreachable" in result.stderr


# ── backfill: happy path ─────────────────────────────────────────


def test_backfill_writes_the_corpus_and_reports_bars_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL"]})
    symbol = Symbol("AAPL", Venue.NASDAQ)
    bars = _session_bars(symbol, _HALF_DAY)
    connection_log = _ConnectionLog()
    feed_log = _FeedLog(bars={symbol: bars})
    _patch_ibkr(monkeypatch, connection_log, feed_log)

    result = runner.invoke(
        app,
        [
            "ibkr",
            "backfill",
            "--start",
            "2024-07-03",
            "--end",
            "2024-07-03",
            "--universe",
            str(universe_path),
        ],
    )

    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    assert lines[0] == str(len(bars))
    assert "AAPL.NASDAQ" in result.stderr
    assert "2024-07-03" in result.stderr
    assert f"{len(bars)} bars" in result.stderr
    written = list((tmp_path / "raw" / "bars").rglob("*.parquet"))
    assert written
    assert connection_log.disconnected


# ── backfill: resumability ───────────────────────────────────────


def test_a_second_pass_over_a_complete_range_makes_no_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus is its own progress record (see `plan_backfill`), so
    re-running over ground already covered must not spend a single request."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL"]})
    symbol = Symbol("AAPL", Venue.NASDAQ)
    bars = _session_bars(symbol, _HALF_DAY)
    args = [
        "ibkr",
        "backfill",
        "--start",
        "2024-07-03",
        "--end",
        "2024-07-03",
        "--universe",
        str(universe_path),
    ]

    _patch_ibkr(monkeypatch, _ConnectionLog(), _FeedLog(bars={symbol: bars}))
    first = runner.invoke(app, args)
    assert first.exit_code == 0

    second_feed = _FeedLog(bars={symbol: bars})
    _patch_ibkr(monkeypatch, _ConnectionLog(), second_feed)
    second = runner.invoke(app, args)

    assert second.exit_code == 0
    assert second.stdout.strip() == "0"
    assert second_feed.calls == []


# ── backfill: passes ──────────────────────────────────────────────


def test_extra_passes_stop_once_the_next_plan_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passes exist to retry what an earlier one could not reach. The single
    cell here is filled on pass one; pass two finds nothing left to plan and
    stops there, one pass short of the `--passes 3` ceiling, rather than
    spending a third pass to confirm what the second already showed."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL"]})
    symbol = Symbol("AAPL", Venue.NASDAQ)
    bars = _session_bars(symbol, _HALF_DAY)
    feed_log = _FeedLog(bars={symbol: bars})
    _patch_ibkr(monkeypatch, _ConnectionLog(), feed_log)

    result = runner.invoke(
        app,
        [
            "ibkr",
            "backfill",
            "--start",
            "2024-07-03",
            "--end",
            "2024-07-03",
            "--universe",
            str(universe_path),
            "--passes",
            "3",
        ],
    )

    assert result.exit_code == 0
    # One request to fill the only cell; the second pass finds an empty plan
    # and makes none.
    assert len(feed_log.calls) == 1
    assert "passes    2, last planned 0 cells" in result.stderr


def test_a_pass_that_gives_up_exits_nonzero_and_runs_no_further_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five consecutive failures end a pass (`DEFAULT_MAX_CONSECUTIVE_FAILURES`
    in the crawler); the outer loop must treat that like a dead Gateway and
    stop, rather than spend two more passes re-learning the same failure."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL", "MSFT", "AMD", "INTC", "NVDA"]})
    feed_log = _FeedLog(raise_on_fetch=RuntimeError("boom"))
    _patch_ibkr(monkeypatch, _ConnectionLog(), feed_log)

    result = runner.invoke(
        app,
        [
            "ibkr",
            "backfill",
            "--start",
            "2024-07-03",
            "--end",
            "2024-07-03",
            "--universe",
            str(universe_path),
            "--passes",
            "3",
        ],
    )

    assert result.exit_code == 1
    assert len(feed_log.calls) == 5
    assert "passes    1, last planned 5 cells" in result.stderr
    assert "stopped" in result.stderr


# ── backfill: limit ───────────────────────────────────────────────


def test_limit_bounds_the_cells_offered_but_not_the_plan_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six cells exist here (2 symbols x 3 sessions). `--limit 1` must offer
    only one to the feed, while `planned` still reports the whole plan."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL", "MSFT"]})
    feed_log = _FeedLog()
    _patch_ibkr(monkeypatch, _ConnectionLog(), feed_log)

    result = runner.invoke(
        app,
        [
            "ibkr",
            "backfill",
            "--start",
            "2024-07-01",
            "--end",
            "2024-07-03",
            "--universe",
            str(universe_path),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "passes    1, last planned 6 cells" in result.stderr
    assert len(feed_log.calls) == 1


# ── backfill: default --end ────────────────────────────────────────


def test_default_end_is_yesterday_utc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--end` defaults to yesterday, UTC. A fixed stand-in for `LiveClock`
    makes this exact instead of depending on the date the suite happens to
    run on."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL"]})
    fixed_now = SimClock(to_nanos(datetime(2024, 7, 9, 12, 0, tzinfo=UTC)))
    monkeypatch.setattr("neurotrade.cli.LiveClock", lambda: fixed_now)
    feed_log = _FeedLog()
    _patch_ibkr(monkeypatch, _ConnectionLog(), feed_log)

    result = runner.invoke(
        app,
        [
            "ibkr",
            "backfill",
            "--start",
            "2024-07-05",
            "--universe",
            str(universe_path),
        ],
    )

    assert result.exit_code == 0
    # Yesterday UTC is 2024-07-08; the range must reach it and no further.
    assert "2024-07-08" in result.stderr
    assert "2024-07-09" not in result.stderr


# ── seed: fixtures ────────────────────────────────────────────

# Real NASDAQ sessions, as the backfill fixtures above use: 07-01 and 07-02
# are ordinary 390-bar days, 07-03 is the pre-July-4th half day (210 bars).
# Vendor rows below are hand-written in each vendor's own format — no real
# vendor bytes are committed (`11-seed-data.plan.md` decision 4).
_FRD_AAPL_URL = "https://frd001.s3-us-east-2.amazonaws.com/AAPL_1min_sample_firstratedata.zip"
_KIBOT_PAGE_URL = "https://www.kibot.com/free-historical-intraday-data.html"
_KIBOT_LINK = "https://api.kibot.com/?get=code1"
_AAPL_SAMPLE = "AAPL_1min_sample_firstratedata.zip"


def _seed_sources_file(tmp_path: Path, entries: list[tuple[str, str, str, str]]) -> Path:
    """A seed sources file in the format `config/seed_sources.yaml` uses."""
    lines = ["version: 1", "entries:"]
    for source, ticker, venue, name in entries:
        lines.append(f"  - source: {source}")
        lines.append(f"    ticker: {ticker}")
        lines.append(f"    venue: {venue}")
        lines.append(f"    file: {name}")
    path = tmp_path / "seed_sources.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _et_minutes(day: date, first: str, count: int) -> Iterator[datetime]:
    """`count` consecutive minute opens from `first`, naive ET, as the files are."""
    hour, minute = (int(part) for part in first.split(":"))
    # Naive on purpose: both vendors write local ET with no offset, and the
    # feed is what attaches the zone. noqa DTZ001 for exactly that reason.
    start = datetime(day.year, day.month, day.day, hour, minute)  # noqa: DTZ001
    for index in range(count):
        yield start + timedelta(minutes=index)


def _frd_rows(days: list[date], *, first: str = "09:30", count: int = 390) -> str:
    rows = ["timestamp,open,high,low,close,volume"]
    rows += [
        f"{stamp:%Y-%m-%d %H:%M:%S},100.0,101.0,99.0,100.5,10"
        for day in days
        for stamp in _et_minutes(day, first, count)
    ]
    return "\n".join(rows) + "\n"


def _kibot_rows(days: list[date], *, first: str = "09:30", count: int = 390) -> str:
    return (
        "\n".join(
            f"{stamp:%m/%d/%Y},{stamp:%H:%M},100.0,101.0,99.0,100.5,10"
            for day in days
            for stamp in _et_minutes(day, first, count)
        )
        + "\n"
    )


def _frd_zip_bytes(days: list[date], *, first: str = "09:30", count: int = 390) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("AAPL_1min.csv", _frd_rows(days, first=first, count=count))
    return buffer.getvalue()


def _snapshot(
    tmp_path: Path, source: str, day: str, files: dict[str, bytes], *, adjusted: str = "unknown"
) -> Path:
    """A fetched snapshot folder, files plus the manifest `seed fetch` writes."""
    folder = tmp_path / "raw" / "vendor" / source / day
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, data in files.items():
        (folder / name).write_bytes(data)
        manifest[name] = {
            "name": name,
            "url": f"https://example.invalid/{name}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "fetched_at": 0,
            "adjusted": adjusted,
        }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def _patch_opener(
    monkeypatch: pytest.MonkeyPatch, responses: dict[str, FakeResponse]
) -> FakeOpener:
    """Replace the real HTTP opener so `seed fetch` never reaches a network."""
    opener = FakeOpener(responses)
    monkeypatch.setattr("neurotrade.adapters.feeds.vendor_download.UrllibOpener", lambda: opener)
    return opener


# ── seed fetch ────────────────────────────────────────────────


def test_seed_fetch_writes_a_dated_snapshot_with_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _patch_opener(monkeypatch, {_FRD_AAPL_URL: FakeResponse(b"zip-bytes")})
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])

    result = runner.invoke(app, ["seed", "fetch", "--sources", str(sources)])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "1"  # stdout carries only the count
    folder = latest_snapshot(tmp_path / "raw", "firstrate")
    assert folder is not None
    assert (folder / _AAPL_SAMPLE).read_bytes() == b"zip-bytes"
    assert [record.adjusted for record in read_manifest(folder)] == ["unknown"]


def test_seed_fetch_takes_one_vendor_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    opener = _patch_opener(
        monkeypatch,
        {
            _KIBOT_PAGE_URL: FakeResponse(f'<a href="{_KIBOT_LINK}">IBM</a>'.encode()),
            _KIBOT_LINK: FakeResponse(
                b"rows", content_disposition='attachment; filename="IBM_unadjusted.txt"'
            ),
        },
    )
    sources = _seed_sources_file(
        tmp_path,
        [
            ("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE),
            ("kibot", "IBM", "NYSE", "IBM_unadjusted.txt"),
        ],
    )

    result = runner.invoke(app, ["seed", "fetch", "--source", "kibot", "--sources", str(sources)])

    assert result.exit_code == 0, result.stdout
    assert _FRD_AAPL_URL not in opener.opened
    assert latest_snapshot(tmp_path / "raw", "firstrate") is None


def test_seed_fetch_rejects_a_missing_sources_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["seed", "fetch", "--sources", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 2


def test_seed_fetch_refuses_to_overwrite_the_same_days_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kibot's window rolls, so a second fetch is new data rather than a
    retry; clobbering the first would destroy the only copy of those rows."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _patch_opener(monkeypatch, {_FRD_AAPL_URL: FakeResponse(b"zip-bytes")})
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])

    assert runner.invoke(app, ["seed", "fetch", "--sources", str(sources)]).exit_code == 0
    again = runner.invoke(app, ["seed", "fetch", "--sources", str(sources)])

    assert again.exit_code == 1
    assert "refusing to overwrite" in again.stderr


def test_seed_fetch_fails_when_the_config_names_a_file_the_vendor_does_not_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`seed ingest` finds a file by the configured name while the FRD
    downloader derives it from the ticker; a drifted `file:` must fail the
    fetch rather than produce an ingest that finds nothing."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _patch_opener(monkeypatch, {_FRD_AAPL_URL: FakeResponse(b"zip-bytes")})
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", "AAPL_wrong.zip")])

    result = runner.invoke(app, ["seed", "fetch", "--sources", str(sources)])

    assert result.exit_code == 1
    assert "AAPL_wrong.zip" in result.stderr


# ── seed ingest ───────────────────────────────────────────────


def _ingest_firstrate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    days: list[date],
    *,
    first: str = "09:30",
    count: int = 390,
    args: list[str] | None = None,
) -> tuple[int, str, str]:
    """Put a FRD snapshot on disk and ingest it. Returns exit code, out, err."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _snapshot(
        tmp_path,
        "firstrate",
        "2026-09-13",
        {_AAPL_SAMPLE: _frd_zip_bytes(days, first=first, count=count)},
    )
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])
    result = runner.invoke(app, ["seed", "ingest", "--sources", str(sources), *(args or [])])
    return result.exit_code, result.stdout, result.stderr


def test_seed_ingest_writes_the_sessions_the_file_covers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, _ = _ingest_firstrate(tmp_path, monkeypatch, [date(2024, 7, 1), date(2024, 7, 2)])

    assert code == 0, out
    assert out.strip() == "780"  # 2 x 390
    counts = DuckDBCatalog(tmp_path / "derived" / "seed" / "firstrate").bar_counts(
        Symbol("AAPL", Venue.NASDAQ), BarInterval.MIN_1
    )
    assert counts == {date(2024, 7, 1): 390, date(2024, 7, 2): 390}


def test_seed_ingest_keeps_seed_bars_out_of_the_ibkr_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sample bar under `raw/bars/` would mark that session complete for the
    backfill and outrank IBKR's own bar for the same minute."""
    code, out, _ = _ingest_firstrate(tmp_path, monkeypatch, [date(2024, 7, 1)])

    assert code == 0, out
    assert not (tmp_path / "raw" / "bars").exists()


def test_seed_ingest_trims_extended_hours_to_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FRD's file runs 04:00-20:00 ET; the corpus is regular hours only, and
    the trim comes from the crawler the IBKR backfill also uses."""
    code, out, _ = _ingest_firstrate(
        tmp_path, monkeypatch, [date(2024, 7, 1)], first="04:00", count=960
    )

    assert code == 0, out
    assert out.strip() == "390"


def test_seed_ingest_respects_a_half_day_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2024-07-03 closed at 13:00 ET. The file has a full day of rows; only the
    210 bars closing inside the session may be stored."""
    code, out, _ = _ingest_firstrate(tmp_path, monkeypatch, [_HALF_DAY])

    assert code == 0, out
    assert out.strip() == "210"


def test_seed_ingest_stamps_the_vendor_as_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance is the only way to know a bar's volume is a sample's, not
    IBKR's consolidated volume (see `Source`)."""
    code, out, _ = _ingest_firstrate(tmp_path, monkeypatch, [date(2024, 7, 1)])

    assert code == 0, out
    table = ds.dataset(tmp_path / "derived" / "seed" / "firstrate", format="parquet").to_table()
    assert set(table.column("source").to_pylist()) == {"firstrate"}


def test_seed_ingest_run_twice_writes_nothing_the_second_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crawler plans from the corpus, so re-running is cheap and safe."""
    first_code, _, _ = _ingest_firstrate(tmp_path, monkeypatch, [date(2024, 7, 1)])
    assert first_code == 0
    sources = tmp_path / "seed_sources.yaml"
    again = runner.invoke(app, ["seed", "ingest", "--sources", str(sources)])

    assert again.exit_code == 0, again.stdout
    assert again.stdout.strip() == "0"


def test_seed_ingest_reads_the_newest_snapshot_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _snapshot(
        tmp_path, "firstrate", "2026-09-13", {_AAPL_SAMPLE: _frd_zip_bytes([date(2024, 7, 1)])}
    )
    _snapshot(
        tmp_path,
        "firstrate",
        "2026-09-15",
        {_AAPL_SAMPLE: _frd_zip_bytes([date(2024, 7, 1), date(2024, 7, 2)])},
    )
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])

    result = runner.invoke(app, ["seed", "ingest", "--sources", str(sources)])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "780"


def test_seed_ingest_can_be_pointed_at_an_older_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kibot's window rolls, so an older snapshot holds sessions the newest no
    longer does."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _snapshot(
        tmp_path, "firstrate", "2026-09-13", {_AAPL_SAMPLE: _frd_zip_bytes([date(2024, 7, 1)])}
    )
    _snapshot(
        tmp_path,
        "firstrate",
        "2026-09-15",
        {_AAPL_SAMPLE: _frd_zip_bytes([date(2024, 7, 1), date(2024, 7, 2)])},
    )
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])

    result = runner.invoke(
        app, ["seed", "ingest", "--sources", str(sources), "--snapshot", "2026-09-13"]
    )

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "390"


def test_seed_ingest_rejects_a_snapshot_that_was_never_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _snapshot(
        tmp_path, "firstrate", "2026-09-13", {_AAPL_SAMPLE: _frd_zip_bytes([date(2024, 7, 1)])}
    )
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])

    result = runner.invoke(
        app, ["seed", "ingest", "--sources", str(sources), "--snapshot", "1999-01-01"]
    )

    assert result.exit_code == 2
    assert "1999-01-01" in result.stderr


def test_seed_ingest_says_what_to_run_when_nothing_was_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])

    result = runner.invoke(app, ["seed", "ingest", "--sources", str(sources)])

    assert result.exit_code == 2
    assert "seed fetch" in result.stderr


def test_seed_ingest_takes_the_files_a_partial_snapshot_does_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fetch that got some tickers is still worth ingesting — say which are
    absent and take the rest."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _snapshot(
        tmp_path, "firstrate", "2026-09-13", {_AAPL_SAMPLE: _frd_zip_bytes([date(2024, 7, 1)])}
    )
    sources = _seed_sources_file(
        tmp_path,
        [
            ("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE),
            ("firstrate", "MSFT", "NASDAQ", "MSFT_1min_sample_firstratedata.zip"),
        ],
    )

    result = runner.invoke(app, ["seed", "ingest", "--sources", str(sources)])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "390"
    assert "absent" in result.stderr
    assert "MSFT_1min_sample_firstratedata.zip" in result.stderr


def test_seed_ingest_fails_when_the_snapshot_holds_none_of_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _snapshot(tmp_path, "firstrate", "2026-09-13", {})
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])

    result = runner.invoke(app, ["seed", "ingest", "--sources", str(sources)])

    assert result.exit_code == 1
    assert "no firstrate files" in result.stderr


def test_seed_ingest_refuses_a_snapshot_with_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows whose bytes cannot be identified afterwards do not belong in the
    corpus."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    folder = _snapshot(
        tmp_path, "firstrate", "2026-09-13", {_AAPL_SAMPLE: _frd_zip_bytes([date(2024, 7, 1)])}
    )
    (folder / "manifest.json").unlink()
    sources = _seed_sources_file(tmp_path, [("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE)])

    result = runner.invoke(app, ["seed", "ingest", "--sources", str(sources)])

    assert result.exit_code == 1
    assert "manifest.json" in result.stderr


def test_seed_ingest_reads_kibots_headerless_format_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _snapshot(
        tmp_path,
        "kibot",
        "2026-09-13",
        {"IBM_unadjusted.txt": _kibot_rows([date(2024, 7, 1)]).encode()},
        adjusted="unadjusted",
    )
    sources = _seed_sources_file(tmp_path, [("kibot", "IBM", "NYSE", "IBM_unadjusted.txt")])

    result = runner.invoke(app, ["seed", "ingest", "--sources", str(sources)])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "390"
    table = ds.dataset(tmp_path / "derived" / "seed" / "kibot", format="parquet").to_table()
    assert set(table.column("source").to_pylist()) == {"kibot"}


def test_seed_ingest_keeps_each_vendor_in_its_own_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    _snapshot(
        tmp_path, "firstrate", "2026-09-13", {_AAPL_SAMPLE: _frd_zip_bytes([date(2024, 7, 1)])}
    )
    _snapshot(
        tmp_path,
        "kibot",
        "2026-09-13",
        {"IBM_unadjusted.txt": _kibot_rows([date(2024, 7, 1)]).encode()},
        adjusted="unadjusted",
    )
    sources = _seed_sources_file(
        tmp_path,
        [
            ("firstrate", "AAPL", "NASDAQ", _AAPL_SAMPLE),
            ("kibot", "IBM", "NYSE", "IBM_unadjusted.txt"),
        ],
    )

    result = runner.invoke(app, ["seed", "ingest", "--sources", str(sources)])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "780"
    seed_root = tmp_path / "derived" / "seed"
    assert {path.name for path in seed_root.iterdir()} == {"firstrate", "kibot"}


def test_every_seed_source_has_a_provenance_and_a_feed() -> None:
    """Adding a vendor without either would store rows under the wrong source,
    or parse them with another vendor's format."""
    for vendor in SeedSource:
        assert vendor in _SEED_PROVENANCE
        job = _SeedJob(source=vendor, folder=Path("."), files={}, manifest=())
        assert _seed_feed(job, LiveClock()) is not None


# ── daily backfill ────────────────────────────────────────────


class _FakeDownloader:
    """Stands in for `YahooDownloader` so no test reaches Yahoo."""

    rows: ClassVar[tuple[DailyRow, ...]] = ()
    calls: ClassVar[list[tuple[str, date, date]]] = []

    def __call__(self, ticker: str, *, start: date, end: date) -> Sequence[DailyRow]:
        _FakeDownloader.calls.append((ticker, start, end))
        return _FakeDownloader.rows


@pytest.fixture
def fake_yahoo(monkeypatch: pytest.MonkeyPatch) -> type[_FakeDownloader]:
    _FakeDownloader.calls = []
    _FakeDownloader.rows = ()
    monkeypatch.setattr(yfinance_daily, "YahooDownloader", _FakeDownloader)
    return _FakeDownloader


def _daily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    args: list[str] | None = None,
) -> tuple[int, str, str]:
    """Run `daily backfill` over a one-symbol universe. Returns code, out, err."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL"]})
    result = runner.invoke(
        app,
        [
            "daily",
            "backfill",
            "--start",
            "2024-07-01",
            "--end",
            "2024-07-05",
            "--universe",
            str(universe_path),
            *(args or []),
        ],
    )
    return result.exit_code, result.stdout, result.stderr


_DAILY_ROWS = (
    DailyRow(date(2024, 7, 1), 100.0, 102.0, 99.0, 101.0, 1_000.0),
    DailyRow(date(2024, 7, 2), 101.0, 103.0, 100.0, 102.0, 1_100.0),
)


def test_daily_backfill_writes_one_bar_per_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_yahoo: type[_FakeDownloader]
) -> None:
    fake_yahoo.rows = _DAILY_ROWS

    code, out, err = _daily(tmp_path, monkeypatch)

    assert code == 0, err
    assert out.strip() == "2"
    counts = DuckDBCatalog(tmp_path / "derived" / "daily" / "yfinance").bar_counts(
        Symbol("AAPL", Venue.NASDAQ), BarInterval.DAY_1
    )
    assert counts == {date(2024, 7, 1): 1, date(2024, 7, 2): 1}


def test_daily_backfill_asks_yahoo_once_for_the_whole_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_yahoo: type[_FakeDownloader]
) -> None:
    """One HTTP call per symbol, not one per session — the crawler offers three
    sessions here and Yahoo must still be asked exactly once."""
    fake_yahoo.rows = _DAILY_ROWS

    code, _, err = _daily(tmp_path, monkeypatch)

    assert code == 0, err
    assert fake_yahoo.calls == [("AAPL", date(2024, 7, 1), date(2024, 7, 5))]


def test_daily_backfill_keeps_daily_bars_out_of_the_minute_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_yahoo: type[_FakeDownloader]
) -> None:
    """A daily bar under `raw/bars/` would mark that session held for the
    minute backfill and hide a real gap."""
    fake_yahoo.rows = _DAILY_ROWS

    code, _, err = _daily(tmp_path, monkeypatch)

    assert code == 0, err
    assert not (tmp_path / "raw" / "bars").exists()
    assert not (tmp_path / "derived" / "seed").exists()


def test_daily_backfill_stamps_yfinance_as_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_yahoo: type[_FakeDownloader]
) -> None:
    fake_yahoo.rows = _DAILY_ROWS

    code, _, err = _daily(tmp_path, monkeypatch)

    assert code == 0, err
    coverage = DuckDBCatalog(tmp_path / "derived" / "daily" / "yfinance").coverage(
        Symbol("AAPL", Venue.NASDAQ), BarInterval.DAY_1
    )
    assert {held.sources for held in coverage} == {("yfinance",)}


def test_daily_backfill_run_twice_writes_nothing_the_second_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_yahoo: type[_FakeDownloader]
) -> None:
    """A daily session holds exactly one bar, so `expected_bars` must answer 1
    rather than flooring a 6.5-hour session to zero daily bars."""
    fake_yahoo.rows = _DAILY_ROWS

    first_code, _, _ = _daily(tmp_path, monkeypatch)
    second_code, out, err = _daily(tmp_path, monkeypatch)

    assert (first_code, second_code) == (0, 0), err
    assert out.strip() == "0"


def test_daily_backfill_reports_a_row_the_calendar_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_yahoo: type[_FakeDownloader]
) -> None:
    fake_yahoo.rows = (
        *_DAILY_ROWS,
        DailyRow(date(2024, 7, 4), 100.0, 102.0, 99.0, 101.0, 1_000.0),  # Independence Day
    )

    code, _, err = _daily(tmp_path, monkeypatch)

    assert code == 0, err
    assert "unmatched AAPL.NASDAQ 2024-07-04" in err


def test_daily_backfill_rejects_an_inverted_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_yahoo: type[_FakeDownloader]
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))
    universe_path = _write_universe(tmp_path, {"NASDAQ": ["AAPL"]})

    result = runner.invoke(
        app,
        [
            "daily",
            "backfill",
            "--start",
            "2024-07-05",
            "--end",
            "2024-07-01",
            "--universe",
            str(universe_path),
        ],
    )

    assert result.exit_code == 2
    assert "is before" in result.stderr


def test_daily_backfill_rejects_a_missing_universe_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_yahoo: type[_FakeDownloader]
) -> None:
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", str(tmp_path))

    result = runner.invoke(
        app,
        [
            "daily",
            "backfill",
            "--start",
            "2024-07-01",
            "--universe",
            str(tmp_path / "absent.yaml"),
        ],
    )

    assert result.exit_code == 2
    assert "not found" in result.stderr
