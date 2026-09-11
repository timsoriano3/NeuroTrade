# `core/` — the domain model

Depends on nothing else in the repo (enforced: `.importlinter`, contract
`core-is-the-bottom`). No network, no files, no clock, no config.

## `types.py` — value objects

`Currency` `Side` `Venue` `Symbol` `Price` `Quantity` `Money`. All frozen, slotted, `order=True`.

- **`Price`** — `Decimal`, strictly positive and finite. `Price - Price` returns a bare
  `Decimal`, not a `Price`: a spread may be zero or negative, a price level may not.
- **`Quantity`** — `Decimal`, non-negative, fractional allowed. Subtracting below zero raises
  (closing more than is held is a bug, not a short).
- **`Money`** — `Decimal` + `Currency`. May be negative. **Mixing currencies raises** on every
  arithmetic and comparison operator; US and Canadian names are live simultaneously, and
  conversion needs an explicit rate that is not the domain layer's job.
- **`Symbol`** — `(ticker, venue)`. Ticker must be upper-case and unpadded. `Venue.SMART` is
  rejected: SMART is IBKR's order router, not a listing venue. Currency derives from venue.
  Same ticker on two venues is two instruments — `TD` is Toronto-Dominion on both TSX (CAD)
  and NYSE (USD), at different prices.
- **Constructing `Price`/`Quantity`/`Money` from a `float` raises `TypeError`.** At a feed
  boundary use `from_float`, which routes via `repr` — the marked precision boundary.

## `clock.py` — time

`Nanos` = int nanoseconds since the Unix epoch, UTC.

- `to_nanos` / `to_datetime` — **integer arithmetic only**; naive datetimes raise.
- `Clock` protocol → `LiveClock` (the only wall-clock in the system) and `SimClock`
  (`set_time_ns`, `set_time`, `advance_ns`; refuses to move backwards).
- Invariant: no `datetime.now()` in domain code. Ruff's `DTZ` rules back this up.

## `events.py` — the event hierarchy

`Event` → `MarketEvent` / `VenueEvent` → `Bar` `Quote` `TickTrade` `SessionBoundary`
`TradingHalt` `TradingResumed`. All frozen, slotted, `kw_only`.

- Every event carries `ts_event` (when the venue says it happened) and `ts_init` (when we
  received it). `latency_ns` is their difference; `sort_key` is `(ts_event, seq)`.
- **`Bar.ts_event` is the bar's CLOSE**, not its open. `ts_open` derives the other end.
  Getting this backwards is a look-ahead bug — see `08-gotchas`.
- `Quote` gives `spread`, `mid`, `microprice` (size-weighted mid — leans toward the side with
  less resting size, the better short-horizon fair-value estimate), `is_locked`, `is_crossed`.
- `BarInterval` is a `StrEnum` with a `nanos` property.
- Validation lives in `__post_init__` (e.g. `high >= max(open, close)`). Test fixtures that
  violate it are wrong three times out of three — see `08-gotchas`.

## `ids.py` — content-derived identifiers

`IntentId` `OrderId` `FillId` `RunId`, all `_DerivedId` subclasses: BLAKE2b over the
identifying tuple, 16 hex chars, `\x1f`-separated.

**Never UUID, never `hash()`** — `hash()` is process-salted, so it breaks replay across runs.
Deriving from content means the same input yields the same id in any process, which is what
makes an event log comparable to itself.

## `intent.py` — what a strategy emits

`Intent` (an `Event`): symbol, side, `EntryTrigger`, stop, target, expiry, rationale.
Carries `risk_per_share`, `target_price`, `reward_to_risk`. An intent is a *proposal* — risk
sizing and the broker come later. This is the seam a model may influence and a risk limit may
not be overridden at.

## `orders.py` — what the broker sees

`OrderType` `TimeInForce` `LiquidityFlag` (maker = added resting liquidity, taker = removed it;
fee schedules differ), `Order`, `Fill`.
`Fill` computes `notional`, `slippage_per_share`, `slippage_cost`, `total_cost` — all `Money`.

## `position.py` — accumulated fills

`Position` is immutable; `apply(fill)` returns a new one. Handles opening, adding, reducing,
closing and flipping through zero in a single fill. Provides `unrealised_pnl`, `net_realised_pnl`,
`total_pnl`, `realised_r` (R-multiple: P&L divided by the risk originally taken, so a trade that
made twice what it risked is `+2R` regardless of position size).

## `ports.py` — the hexagon's edges

`MarketDataPort` `BrokerPort` `StoragePort` `EventStorePort`, all `Protocol`s.
Adapters implement them; the core never imports an adapter.
