"""Tests for the command-line entry point.

The stdout/stderr split is the one worth guarding: `neurotrade config hash` is
meant to be pipeable, and a stray log line on stdout would silently corrupt
whatever consumes it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from typer.testing import CliRunner

from neurotrade import __version__
from neurotrade.cli import app
from neurotrade.config import Profile, config_hash, load_settings
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
