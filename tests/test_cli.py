"""Tests for the command-line entry point.

The stdout/stderr split is the one worth guarding: `neurotrade config hash` is
meant to be pipeable, and a stray log line on stdout would silently corrupt
whatever consumes it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import structlog
from typer.testing import CliRunner

from neurotrade import __version__
from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
from neurotrade.adapters.ibkr.connection import IbkrConnectionError
from neurotrade.cli import app
from neurotrade.config import Profile, config_hash, load_settings
from neurotrade.core.clock import SimClock, to_nanos
from neurotrade.core.events import Bar, BarInterval, Event
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.logs import clear_context

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
