# IBKR adapter — how gate G2 is proven

Library: **`ib_async`** (the maintained successor to the archived `ib_insync` — do not follow
`ib_insync` docs or answers). Gateway ports: **4001 live, 4002 paper**; TWS 7496 / 7497.

Account in use: paper `DUT108414`, second username `timsorianopaper`.
Gateway API mode must be **IB API**, not FIX CTCI. Gateway has no "Enable ActiveX and Socket
Clients" checkbox — that is a TWS setting; the Gateway socket is always on.

Every IBKR test is marked `@pytest.mark.ibkr` and **excluded by default**
(`addopts = "... -m 'not ibkr'"`). Run them with `-m ibkr` and a logged-in Gateway.

## `connection.py`

`IbkrClient` is a `Protocol` describing the slice of `ib_async` actually used, so the rest of
the adapter is unit-testable against a fake with no network.

`IbkrConnection` — connect, disconnect, `is_connected`, `ib`, `settings`.
`ConnectionProbe` — `account`, `account_matches`, `is_healthy`, `describe`.
`make ibkr-check` runs the probe: reachable, and the account we expect.
Failures raise `IbkrConnectionError` rather than returning a falsy object.

## `market_data.py`

Historical bars → `Bar` events.

- `IBKR_EXCHANGE` maps `Venue` to IBKR's exchange string. **`Venue.TSX → "TSE"` and
  `Venue.TSXV → "VENTURE"`** — IBKR's names are not the venue's own.
- `_BAR_SIZE` maps `BarInterval` to IBKR's bar-size strings.
- **`_MAX_DAYS` caps the request duration per interval.** An oversized request does not error —
  it hangs until timeout. See `08-gotchas`.
- **IBKR timestamps a bar at its OPEN; our `Bar.ts_event` is its CLOSE.** The conversion adds
  the interval, and getting it wrong is a silent look-ahead bug:
  ```python
  close_ns = (opened_at * 1_000_000_000) + interval.nanos
  ```
- `MarketDataError` on a rejected or empty request.
- **Per-request timeout (`IbkrSettings.request_timeout_seconds`, default 60s).** A Gateway can
  be up and logged in while IBKR's data farms are unreachable — contract lookup then never
  answers, and `ib_async`'s historical request times out by quietly returning no bars,
  indistinguishable from a real halt. `_qualify` wraps `qualifyContractsAsync` in
  `asyncio.wait_for` (unpaced, no IBKR-side slot, free to abandon locally). `fetch_bars` instead
  passes `timeout=` into `reqHistoricalDataAsync` itself — `ib_async` cancels the request at
  IBKR on expiry, where `wait_for` would only abandon it client-side and leave it holding one of
  the few concurrent slots — then reclassifies an empty result as `MarketDataError` if the
  injected `Clock` shows elapsed time `>= timeout`. Both paths name the instrument and the
  timeout in the error.

## `pacing.py`

IBKR throttles historical data: roughly 60 requests per 10 minutes, and exceeding it earns a
soft ban rather than an error. `HistoricalPacer` holds a sliding window —
`DEFAULT_MAX_REQUESTS = 55` (deliberate headroom), `DEFAULT_WINDOW_SECONDS = 600`.
`record()`, `wait_seconds()`, `in_window`, `headroom`. Takes time from the `Clock` port, so it
is testable without sleeping.

## `broker.py`

`IbkrBroker` implements `BrokerPort`. Maps `Order` → `ib_async` order, and executions back to
`Fill` (`_SIDE_FROM_IBKR`, `_LIQUIDITY`). `working(order_id)`, `may_trade_live`.

The guard that matters:

```python
def _guard_live(self, order: Order) -> None:
    if self._connection.settings.is_paper_port or self._allow_live_orders:
        return
    raise LiveOrdersNotPermitted(...)
```

**Two independent conditions.** A live port with `allow_live_orders` false raises; only the
`live` profile sets that flag, and only `live.yaml` contains it. Sending a real order requires
both the right port and the right profile — neither alone is enough.

`OrderRejected` carries the broker's reason rather than a generic failure.

## Gate G2

```
make paper-smoke        # submit, acknowledge, cancel — against the real paper book
```
Verified: the order round-tripped, was written to the event log carrying its config hash, and
that log replayed deterministically. G2 therefore also re-proves G1 on live-sourced data rather
than on a synthetic fixture.
