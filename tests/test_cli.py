"""Tests for the command-line entry point.

The stdout/stderr split is the one worth guarding: `neurotrade config hash` is
meant to be pipeable, and a stray log line on stdout would silently corrupt
whatever consumes it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog
from typer.testing import CliRunner

from neurotrade import __version__
from neurotrade.cli import app
from neurotrade.config import Profile, config_hash, load_settings
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
