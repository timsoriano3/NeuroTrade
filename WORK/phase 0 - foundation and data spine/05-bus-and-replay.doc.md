# Event bus and replay — how gate G1 is proven

## `bus.py` — the event bus

Infrastructure, not a layer: it depends on nothing, because everything above it publishes to it.

`EventBus.subscribe(event_type, handler)` dispatches by type, **including subclasses** — a
handler on `MarketEvent` sees every `Bar`, `Quote` and `TickTrade`. Handlers fire in
registration order, never in set or dict order.

`publish` / `publish_all`. A raising handler becomes `HandlerFailed` naming the handler and the
event: a silently swallowed exception in a trading loop is the worst possible failure mode.
`published` and `subscriptions` exist so a test can assert on wiring.

## `lab/replay.py` — the replay engine

```python
for event in self._store.stream(start, end):
    self._clock.set_time_ns(event.ts_event)   # clock leads the event
    self._bus.publish(event)
```

**The clock is set before the event is published**, so any handler asking the clock for "now"
gets the event's own time, not the time of whatever was processed last. This is the single line
that makes replay reproduce live semantics.

- `RunDigest` — BLAKE2b over each dispatched event, **seeded with `CODEC_VERSION`** so a codec
  change necessarily changes the digest instead of silently producing a colliding one.
  `hexdigest`, `count`.
- `ReplayResult` — digest, count, span. `is_empty`, `span_ns`.
- `ReplayEngine` — owns a `SimClock` and an `EventBus`; `run(start, end)`.

## Gate G1

```
make verify-replay      # replays twice, compares digests
```
Current digest over the committed fixture: **`4f58fe2c99cd26dc7cbb8faf033a39d1`**.

Verified three ways:
1. Twice in one process.
2. Across three separate interpreter processes with **different `PYTHONHASHSEED`** — this is
   the run that would catch a dependence on dict or set iteration order.
3. Via `make verify-replay`, which is what CI runs.

The digest is pinned in the test suite, so any change to the codec, the event schema, the sort
order or the dispatch order fails the build rather than quietly producing different history.
