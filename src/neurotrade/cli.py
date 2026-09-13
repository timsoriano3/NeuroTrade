"""Command-line entry point.

Every operational command routes through here, which is what makes startup have
exactly one shape: resolve the profile, configure logging, bind the run context,
then do the work. A command that built its own settings would be able to run
against a different configuration than the one its log lines claim.

**Output goes to stdout, logs go to stderr.** So ``neurotrade config hash`` can
be piped into another command without log noise, and redirecting logs never
swallows the answer you asked for.

``--profile`` is a global option, so it precedes the subcommand — the same shape
as ``git --no-pager log``::

    neurotrade --profile paper config show
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from neurotrade import __version__
from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
from neurotrade.adapters.ibkr.broker import IbkrBroker
from neurotrade.adapters.ibkr.connection import IbkrConnection, IbkrConnectionError
from neurotrade.adapters.ibkr.market_data import IbkrMarketData
from neurotrade.adapters.storage.duckdb_catalog import DuckDBCatalog
from neurotrade.adapters.storage.event_store import EventStore
from neurotrade.adapters.storage.parquet_store import ParquetStore
from neurotrade.adapters.storage.schemas import Source
from neurotrade.adapters.universe.universe_file import InvalidUniverseFile, UniverseFile
from neurotrade.config import (
    DEFAULT_CONFIG_DIR,
    Profile,
    Settings,
    config_hash,
    describe,
    load_settings,
)
from neurotrade.core.clock import LiveClock, SimClock, to_datetime
from neurotrade.core.events import BarInterval
from neurotrade.core.ids import IntentId, OrderId, RunId
from neurotrade.core.orders import Fill, Order, OrderType
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.ingest.crawler import CellOutcome, CellStatus, CrawlReport, crawl
from neurotrade.lab.replay import ReplayEngine
from neurotrade.logs import configure, get_logger

__all__ = ["app"]

log = get_logger(__name__)

_FOREVER_NS = 2**63 - 1
"""Upper bound for a whole-session replay. The log holds one session, so the
range is only there to satisfy the port; bounding it by date would mean the CLI
needing a venue calendar to know when the session ended."""

_DEFAULT_UNIVERSE = DEFAULT_CONFIG_DIR / "universe.yaml"
"""The committed seed universe. A flag rather than a setting because the
universe is data, not configuration — see `Universe.digest`."""


@dataclass(frozen=True, slots=True)
class AppContext:
    """What every command needs, resolved once by the root callback.

    Reached through ``ctx.obj``. Frozen because a command that mutated the
    settings mid-run would invalidate the config hash already stamped on this
    run's log lines.
    """

    settings: Settings  # resolved configuration for this invocation
    run_id: RunId  # identifies this run in the logs


app = typer.Typer(
    name="neurotrade",
    help="Autonomous day-trading system for US and Canadian equities.",
    no_args_is_help=True,
    add_completion=False,
)

config_app = typer.Typer(help="Inspect resolved configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

ibkr_app = typer.Typer(help="Talk to Interactive Brokers.", no_args_is_help=True)
app.add_typer(ibkr_app, name="ibkr")


@app.callback()
def main(
    ctx: typer.Context,
    profile: Annotated[
        Profile | None,
        typer.Option("--profile", "-p", help="Which environment to run as."),
    ] = None,
) -> None:
    """Resolve configuration and start logging before any command runs.

    Args:
        ctx: Typer context. Receives the resolved `AppContext` as `ctx.obj`.
        profile: Overrides `NEUROTRADE_PROFILE`. Defaults to `research`, the
            only profile that cannot spend money.
    """
    settings = load_settings(profile)
    # LiveClock: a command invocation happens in real time. Replay commands
    # will build their own SimClock for the session they drive.
    run_id = configure(settings, LiveClock())
    ctx.obj = AppContext(settings=settings, run_id=run_id)


@app.command()
def version() -> None:
    """Print the package version.

    Example:
        $ neurotrade version
        0.0.0
    """
    typer.echo(__version__)


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Print the resolved configuration and its hash.

    Answers "what is this process actually running with" without attaching a
    debugger — which matters most when a profile file, a `.env` and an
    environment variable disagree.

    Example:
        $ neurotrade --profile paper config show
    """
    app_context: AppContext = ctx.obj
    typer.echo(describe(app_context.settings))
    typer.echo(f"{'config_hash':<20} {config_hash(app_context.settings)}")
    typer.echo(f"{'run_id':<20} {app_context.run_id}")


@config_app.command("hash")
def config_hash_command(ctx: typer.Context) -> None:
    """Print only the config hash, for scripting.

    The hash covers settings that change trading decisions, not where the
    process happens to run — so it is stable across machines and comparable
    against the hash recorded on any order (§6.3).

    Example:
        $ neurotrade --profile live config hash
        cfg_0355e7b4b9bef4d8
    """
    app_context: AppContext = ctx.obj
    typer.echo(config_hash(app_context.settings))


@app.command()
def replay(
    ctx: typer.Context,
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Trading day to replay, as YYYY-MM-DD."),
    ] = None,
    log_path: Annotated[
        Path | None,
        typer.Option("--log", "-l", help="Replay a specific log file instead."),
    ] = None,
) -> None:
    """Replay a recorded session and print its digest.

    The digest is a hash of every event dispatched, in order. Running this twice
    on the same log must print the same value — that is gate G1, and it is what
    makes "did that change alter behaviour?" answerable without reading logs.

    Give either `--session`, which looks under the configured data root, or
    `--log`, which takes a path directly. The second exists because a log worth
    replaying does not always live where this machine keeps its own: a shadow
    run, a session copied off the trading host, or the test fixture.

    The digest goes to stdout alone so it can be compared directly; the summary
    goes to stderr, which is why piping this gives one clean line.

    Example:
        $ neurotrade replay --session 2026-03-16
        $ neurotrade replay --log tests/fixtures/session.jsonl
    """
    app_context: AppContext = ctx.obj
    storage = app_context.settings.storage

    if (session is None) == (log_path is None):
        typer.echo("give exactly one of --session or --log", err=True)
        raise typer.Exit(code=2)

    path = log_path if log_path is not None else storage.session_log(str(session))

    if not path.exists():
        typer.echo(f"no recorded session at {path}", err=True)
        available = sorted(p.stem for p in storage.events_dir.glob("*.jsonl"))
        if available:
            typer.echo(f"recorded sessions: {', '.join(available)}", err=True)
        else:
            typer.echo(
                f"nothing recorded under {storage.events_dir} yet — "
                f"sessions are written by the live engine, which is Phase 3. "
                f"To try this now: --log tests/fixtures/session.jsonl",
                err=True,
            )
        raise typer.Exit(code=1)

    # A fresh SimClock per replay: its start seeds nothing here, but sharing one
    # across runs would let the first run's end position offset the second.
    result = ReplayEngine(EventStore(path), SimClock(0)).run(0, _FOREVER_NS)

    log.info(
        "replay_complete",
        source=str(path),
        digest=result.digest,
        events_read=result.events_read,
        events_dispatched=result.events_dispatched,
    )

    typer.echo(f"source    {path}", err=True)
    typer.echo(
        f"events    {result.events_read} read, {result.events_dispatched} dispatched", err=True
    )
    typer.echo(f"span      {result.span_ns / 60_000_000_000:.0f} minutes", err=True)
    typer.echo(result.digest)


@ibkr_app.command("check")
def ibkr_check(ctx: typer.Context) -> None:
    """Check IB Gateway is reachable and is the account we expect.

    Connecting proves less than it appears: the socket answers whether Gateway
    is logged into paper or live. This reports *which* account answered and
    whether the port dialled is a paper port, so a misconfiguration is visible
    before an order is placed rather than after.

    Exits non-zero when the connection is unusable or the account does not match
    configuration.

    Example:
        $ neurotrade --profile paper ibkr check
    """
    app_context: AppContext = ctx.obj
    settings = app_context.settings.ibkr

    async def run() -> None:
        connection = IbkrConnection(settings)
        try:
            probe = await connection.probe()
        finally:
            connection.disconnect()

        typer.echo(probe.describe(), err=True)
        log.info(
            "ibkr_probe",
            host=settings.host,
            port=settings.port,
            accounts=list(probe.accounts),
            server_version=probe.server_version,
            healthy=probe.is_healthy,
        )
        if not probe.is_healthy:
            typer.echo("unhealthy — see above", err=True)
            raise typer.Exit(code=1)
        typer.echo(probe.account or ",".join(probe.accounts))

    try:
        asyncio.run(run())
    except IbkrConnectionError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@ibkr_app.command("paper-smoke")
def ibkr_paper_smoke(ctx: typer.Context) -> None:
    """Gate G2: submit a paper order, see it acknowledged, cancel it.

    Places a buy limit far below the market on a liquid name, so it cannot fill.
    The point is to prove the order path works end to end — submission,
    acknowledgement, cancellation — not to acquire a position.

    Every event is written to the session log, so the run is replayable and the
    order record carries the config hash that produced it (§6.3).

    Refuses to run outside a paper port, which is the same structural guard the
    broker applies.

    Example:
        $ neurotrade --profile paper ibkr paper-smoke
    """
    app_context: AppContext = ctx.obj
    settings = app_context.settings

    if not settings.ibkr.is_paper_port:
        typer.echo(
            f"refusing: port {settings.ibkr.port} is a live port, and this places "
            f"a real order there",
            err=True,
        )
        raise typer.Exit(code=2)

    async def run() -> None:
        connection = IbkrConnection(settings.ibkr)
        clock = LiveClock()
        log_path = settings.storage.session_log(to_datetime(clock.now_ns()).strftime("%Y-%m-%d"))
        recorded: list[str] = []

        with EventStore(log_path) as store:

            def record(fill: Fill) -> None:
                """A fill is unexpected here — the limit cannot trade — but if
                one arrives it belongs in the log like any other event."""
                store.append(fill)
                recorded.append(fill.id.value)

            broker = IbkrBroker(
                connection,
                clock,
                allow_live_orders=settings.allow_live_orders,
                on_fill=record,
            )
            order = _smoke_order(clock.now_ns(), config_hash(settings))
            try:
                await broker.submit(order)
                store.append(order)
                typer.echo(f"submitted  {order.id}", err=True)

                status = await _await_status(
                    broker, order.id, {"Submitted", "PreSubmitted", "Cancelled"}
                )
                typer.echo(f"status     {status}", err=True)
                if status not in {"Submitted", "PreSubmitted"}:
                    typer.echo("not acknowledged", err=True)
                    raise typer.Exit(code=1)

                await broker.cancel(order.id)
                status = await _await_status(broker, order.id, {"Cancelled", "ApiCancelled"})
                typer.echo(f"cancelled  {status}", err=True)
            finally:
                connection.disconnect()

        typer.echo(f"recorded   {log_path}", err=True)
        log.info(
            "paper_smoke_complete",
            order_id=order.id.value,
            status=status,
            fills=len(recorded),
            log=str(log_path),
        )
        typer.echo(order.id.value)

    try:
        asyncio.run(run())
    except IbkrConnectionError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@ibkr_app.command("backfill")
def ibkr_backfill(
    ctx: typer.Context,
    start: Annotated[
        datetime,
        typer.Option("--start", help="First session to fill, as YYYY-MM-DD.", formats=["%Y-%m-%d"]),
    ],
    end: Annotated[
        datetime | None,
        typer.Option(
            "--end",
            help="Last session to fill, as YYYY-MM-DD. Defaults to yesterday, UTC.",
            formats=["%Y-%m-%d"],
        ),
    ] = None,
    interval: Annotated[
        BarInterval, typer.Option("--interval", help="Bar size to fill.")
    ] = BarInterval.MIN_1,
    universe_path: Annotated[
        Path, typer.Option("--universe", help="Universe file naming the symbols to fill.")
    ] = _DEFAULT_UNIVERSE,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Most cells to offer per pass.")
    ] = None,
    passes: Annotated[
        int, typer.Option("--passes", min=1, help="Passes to run before exiting.")
    ] = 1,
) -> None:
    """Fill the bar corpus from IBKR history (§12.1 stage 1).

    Each pass plans from what is already on disk, fetches every missing or
    short instrument-session, and writes it under the raw data root. Killing
    the command loses nothing: the next run's plan starts from the corpus.

    **`--start` has no default** because a default would silently decide how
    much history the corpus holds. `--end` does: yesterday, because today's
    session may still be trading and would be stored short.

    Several passes retry what the previous one could not reach — a symbol
    skipped after a failure, a pass ended by a Gateway that went away. Passes
    stop early once the plan is empty or a pass gives up.

    One line per cell goes to stderr as it happens, so a crawl that takes a day
    can be watched. Stdout gets only the bars written, for scripting.

    Exits non-zero when Gateway is unreachable, or when the last pass gave up.

    Example:
        $ neurotrade --profile paper ibkr backfill --start 2026-08-01
        $ neurotrade ibkr backfill --start 2026-08-01 --end 2026-08-29 --limit 50
    """
    app_context: AppContext = ctx.obj
    settings = app_context.settings
    clock = LiveClock()

    first = start.date()
    # UTC yesterday is never later than venue-local yesterday, so the default
    # cannot reach a session that is still open anywhere in the universe.
    last = end.date() if end is not None else to_datetime(clock.now_ns()).date() - timedelta(days=1)
    if last < first:
        typer.echo(f"--end {last} is before --start {first}", err=True)
        raise typer.Exit(code=2)

    try:
        universe = UniverseFile(universe_path).universe()
    except (FileNotFoundError, InvalidUniverseFile) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error

    bars_root = settings.storage.raw_dir / "bars"

    def report_cell(outcome: CellOutcome) -> None:
        line = f"{outcome.status.value:<8} {outcome.symbol!s:<12} {outcome.session_date}"
        if outcome.status is CellStatus.FILLED:
            line += f"  {outcome.written} bars"
        elif outcome.error is not None:
            line += f"  {outcome.error}"
        typer.echo(line, err=True)

    async def run() -> list[CrawlReport]:
        connection = IbkrConnection(settings.ibkr)
        feed = IbkrMarketData(connection, clock)
        store = ParquetStore(bars_root, clock)
        catalog = DuckDBCatalog(bars_root)
        # The same calendar across passes: building one is the expensive part.
        calendar = VenueCalendar()
        reports: list[CrawlReport] = []
        try:
            # Connect up front. The feed would connect on first use, but then an
            # unreachable Gateway reads as five failed cells rather than as the
            # one clear error it is.
            await connection.connect()
            for number in range(1, passes + 1):
                report = await crawl(
                    universe,
                    calendar,
                    catalog,
                    feed,
                    store,
                    start=first,
                    end=last,
                    source=Source.IBKR.value,
                    interval=interval,
                    limit=limit,
                    on_outcome=report_cell,
                )
                reports.append(report)
                _log_pass(number, report, universe.digest, first, last, interval)
                if report.planned == 0 or not report.completed:
                    break
        finally:
            connection.disconnect()
        return reports

    try:
        reports = asyncio.run(run())
    except IbkrConnectionError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    last_report = reports[-1]
    typer.echo(f"passes    {len(reports)}, last planned {last_report.planned} cells", err=True)
    typer.echo(f"corpus    {bars_root}", err=True)
    if not last_report.completed:
        typer.echo(f"stopped   {last_report.stopped}", err=True)
    typer.echo(sum(report.bars_written for report in reports))
    if not last_report.completed:
        raise typer.Exit(code=1)


def _log_pass(
    number: int,
    report: CrawlReport,
    universe_digest: str,
    first: date,
    last: date,
    interval: BarInterval,
) -> None:
    """Record one pass. The universe digest goes on it because the universe is
    data, outside the config hash, and a corpus is only explainable if each
    pass says which list of symbols it was filling."""
    log.info(
        "backfill_pass_complete",
        number=number,
        universe_digest=universe_digest,
        start=str(first),
        end=str(last),
        interval=interval.value,
        planned=report.planned,
        requests=report.requests,
        bars_written=report.bars_written,
        filled=report.count(CellStatus.FILLED),
        empty=report.count(CellStatus.EMPTY),
        failed=report.count(CellStatus.FAILED),
        skipped=report.count(CellStatus.SKIPPED),
        stopped=report.stopped,
    )


def _smoke_order(now: int, fingerprint: str) -> Order:
    """One share of a liquid name, priced where it cannot fill."""
    symbol = Symbol("AAPL", Venue.NASDAQ)
    intent_id = IntentId.derive(
        strategy="paper_smoke",
        strategy_version="1.0.0",
        symbol=symbol,
        ts_event=now,
        seq=0,
    )
    return Order(
        id=OrderId.derive(intent_id=intent_id, ts_event=now),
        intent_id=intent_id,
        symbol=symbol,
        ts_event=now,
        ts_init=now,
        side=Side.BUY,
        quantity=Quantity(1),
        order_type=OrderType.LIMIT,
        limit_price=Price("1.00"),  # far below the market; cannot fill
        config_hash=fingerprint,
    )


async def _await_status(
    broker: IbkrBroker, order_id: OrderId, wanted: set[str], *, tries: int = 40
) -> str:
    """Poll until the order reaches one of `wanted`, or give up.

    Polling is right here and wrong in the trading loop: this is a one-shot
    probe with nothing else to do, where the live engine reacts to events.
    """
    trade = broker.working(order_id)
    if trade is None:
        return "unknown"
    for _ in range(tries):
        await asyncio.sleep(0.25)
        if trade.orderStatus.status in wanted:
            break
    return trade.orderStatus.status


if __name__ == "__main__":
    app()
