"""Tests for structured logging.

The load-bearing property is that a replay produces identical logs — which is
only true because the run id comes from the Clock rather than from randomness or
the wall clock.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator

import pytest
import structlog

from neurotrade.config import Profile, config_hash, load_settings
from neurotrade.core.clock import LiveClock, SimClock
from neurotrade.logs import bind, clear_context, configure, get_logger


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Logging is global state; leaking it between tests hides real bugs."""
    yield
    clear_context()
    structlog.reset_defaults()


def capture(profile: Profile = Profile.PAPER, clock: SimClock | None = None) -> io.StringIO:
    """Configure logging into a buffer and return it."""
    stream = io.StringIO()
    configure(load_settings(profile), clock or SimClock(1_000), stream=stream)
    return stream


def lines(stream: io.StringIO) -> list[dict[str, object]]:
    """Parse captured JSON log lines."""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ── Determinism: the point of deriving the run id from the Clock ──


def test_replay_under_a_simclock_produces_the_same_run_id() -> None:
    """Two replays of one session must be comparable line for line."""
    first = configure(load_settings(Profile.PAPER), SimClock(1_000), stream=io.StringIO())
    second = configure(load_settings(Profile.PAPER), SimClock(1_000), stream=io.StringIO())
    assert first == second


def test_a_different_start_time_is_a_different_run() -> None:
    a = configure(load_settings(Profile.PAPER), SimClock(1_000), stream=io.StringIO())
    b = configure(load_settings(Profile.PAPER), SimClock(2_000), stream=io.StringIO())
    assert a != b


def test_a_different_configuration_is_a_different_run() -> None:
    """Same instant, different settings — not the same run."""
    a = configure(load_settings(Profile.PAPER), SimClock(1_000), stream=io.StringIO())
    b = configure(load_settings(Profile.LIVE), SimClock(1_000), stream=io.StringIO())
    assert a != b


def test_live_runs_are_unique_without_randomness() -> None:
    """LiveClock advances, so successive runs differ with no uuid involved."""
    a = configure(load_settings(Profile.PAPER), LiveClock(), stream=io.StringIO())
    b = configure(load_settings(Profile.PAPER), LiveClock(), stream=io.StringIO())
    assert a != b


def test_two_replays_emit_identical_context() -> None:
    """The whole point: replayed logs are comparable, not merely similar."""
    outputs = []
    for _ in range(2):
        stream = capture(clock=SimClock(1_000))
        get_logger(__name__).info("bar_received", symbol="AAPL.NASDAQ")
        record = lines(stream)[0]
        del record["timestamp"]  # wall clock: describes the replay, not the session
        outputs.append(record)
    assert outputs[0] == outputs[1]


# ── Run context on every line ────────────────────────────────


def test_every_line_carries_the_run_context() -> None:
    """§6.3's operational half: a single line identifies its configuration."""
    stream = capture()
    get_logger(__name__).info("something_happened")
    record = lines(stream)[0]
    run_id = record["run_id"]
    assert isinstance(run_id, str)
    assert run_id.startswith("run_")
    assert record["profile"] == "paper"
    assert record["config_hash"] == config_hash(load_settings(Profile.PAPER))


def test_bound_fields_appear_on_later_lines() -> None:
    stream = capture()
    bind(symbol="AAPL.NASDAQ")
    get_logger(__name__).info("intent_emitted")
    assert lines(stream)[0]["symbol"] == "AAPL.NASDAQ"


def test_configure_clears_context_from_a_previous_run() -> None:
    """Otherwise one session's context leaks into the next one's log lines."""
    capture()
    bind(leftover="from the previous run")
    stream = capture()
    get_logger(__name__).info("fresh")
    assert "leftover" not in lines(stream)[0]


# ── Structure, not sentences ─────────────────────────────────


def test_fields_stay_queryable_rather_than_interpolated() -> None:
    """The reason for structured logging at all."""
    stream = capture()
    get_logger(__name__).info("order_submitted", quantity=100, symbol="AAPL.NASDAQ")
    record = lines(stream)[0]
    assert record["event"] == "order_submitted"
    assert record["quantity"] == 100  # a number, not text inside a message


def test_json_output_is_machine_parseable_in_paper_and_live() -> None:
    for profile in (Profile.PAPER, Profile.LIVE):
        stream = capture(profile)
        get_logger(__name__).info("check")
        assert json.loads(stream.getvalue().splitlines()[0])["event"] == "check"


def test_research_uses_human_readable_output() -> None:
    """Read by a person while working, so not JSON."""
    stream = io.StringIO()
    configure(load_settings(Profile.RESEARCH), SimClock(1_000), stream=stream)
    get_logger(__name__).info("scanning_universe")
    output = stream.getvalue()
    assert "scanning_universe" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.splitlines()[0])


# ── Levels ───────────────────────────────────────────────────


def test_level_comes_from_settings() -> None:
    """paper.yaml is INFO, so debug lines must be dropped."""
    stream = capture(Profile.PAPER)
    log = get_logger(__name__)
    log.debug("too_quiet")
    log.info("loud_enough")
    events = [record["event"] for record in lines(stream)]
    assert events == ["loud_enough"]


def test_exceptions_are_rendered_with_a_traceback() -> None:
    stream = capture()
    try:
        raise ValueError("broker rejected the order")
    except ValueError:
        get_logger(__name__).exception("submit_failed")
    record = lines(stream)[0]
    assert record["level"] == "error"
    assert "broker rejected the order" in str(record.get("exception", ""))
