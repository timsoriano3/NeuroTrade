# strategies

The trading ideas. Each one answers a single question: *given what is knowable
right now, is there a trade here, which way, and where would I be wrong?*

A strategy does **not** decide how much to buy. It produces a proposal; the risk
engine decides size from the model's confidence. That split keeps direction
explainable and confines machine learning to the job it is actually good at —
filtering out the proposals that will not work.

## Files

| File | What it does |
|---|---|
| `base.py` | The contract every strategy implements, the registry that holds them, and the `holds_overnight` / `needs_cross_section` declarations |
| `context.py` | Fills in what a strategy may see: venue phase, regime, features, session levels, the cross-section |
| `arsenal.py` | The registry every plugin registers into |
| `plugins.py` | Imports every shipped plugin, for a caller that resolves a strategy by name |
| `selection.py` | The universe selector: which names are trading unusually heavily right now (§5) |
| `_cross_sectional.py` | The shape the three ranking strategies share — private, not a plugin |
| `gap_continuation.py` | Trades the direction of a gap too wide to expect a fill (§5.2) |
| `intraday_momentum.py` | Trades a half-hourly close outside the day's noise area (§5.4) |
| `vwap_band_reversion.py` | Fades a price stretched away from session VWAP (§5.3) |
| `momentum_ignition.py` | Trades a bar abnormal in volume **and** in range at once (§5.4) |
| `opening_range_breakout.py` | Trades an opening-range break, on the names in play (§5.2) |
| `orb_fade.py` | Fades an opening-range break that failed back inside (§5.2) |
| `intraday_reversal.py` | Fades the universe's biggest trailing movers (§5.5) |
| `relative_strength.py` | Buys the largest sector-adjusted residuals (§5.4) |
| `residual_reversion.py` | Sells the same residuals (§5.5) |
| `prior_close_reversal.py` | Fades a close pushed away from VWAP, **held overnight** (§5.3) |

## The overnight quarantine

`prior_close_reversal` is the only strategy whose positions survive the bell, and
it declares `holds_overnight`. `BacktestEngine(overnight=…)` hosts the day family
or the overnight family and **refuses a mismatch** — Phase 2 plan decision 2.
The reason is not tidiness: Reg-T overnight margin is 2:1 against 4:1 intraday,
and a stop cannot fill through a gap, so an overnight position's risk model is
not the day engine's and a mixed run's sizing is wrong for every position in it.

Its exit is the next session's open, read from `SessionLevels.next_open_ns` —
which comes out of the trading calendar, published years ahead, so it is not
market data and reading it is not lookahead. A fixed 17.5-hour span would land on
a Saturday after a Friday close, where the labeller finds no bars and the barrier
collapses back to Friday's bell.

## The three pairings, and why they exist

Six of the ten are three opposed pairs, and the opposition is enforced by
`Strategy.regimes` rather than by anyone remembering:

| Continuation (TREND_UP / TREND_DOWN / HIGH_VOLATILITY) | Reversion (CHOP / REVERSAL) | Same input |
|---|---|---|
| `opening_range_breakout` | `orb_fade` | the opening range |
| `relative_strength` | `residual_reversion` | the sector-adjusted residual |
| `intraday_momentum` | `vwap_band_reversion` | a price stretched from its anchor |

Which reading is right is a property of the day, which is exactly what §5.7's
classifier decides. It does not exist until Phase 5, so a Phase 2 run is
`ungated` and both halves of every pair fire — sometimes taking opposite sides
of one name on one bar. That is expected, it is stamped on
`BacktestResult.regime_gated`, and it is why no Phase 2 number is the number a
live gate would produce.

## Reading the universe

`opening_range_breakout`, `intraday_reversal`, `relative_strength` and
`residual_reversion` set `needs_cross_section`, which is what makes
`context.cross_section` non-`None` for them and leaves it `None` for everything
else. The view is **one tick stale by construction** — see
`features/cross_section.py` for why a same-tick view would be lookahead. The
residual's benchmark comes from `core/sectors.py`, point-in-time.

Three strategies exist: `gap_continuation`, `intraday_momentum` and
`vwap_band_reversion`. The rest of the twelve chosen for Phase 2 follow, and the
plan's Tier 2 — anything cross-sectional — waits on the crawl delivering breadth.

The last two are deliberately each other's opposite: one buys a stretch away
from the day's average and the other sells it. Which reading is right is a
property of the day, so the regime declarations make them mutually exclusive
rather than leaving the contradiction to a filter someone remembers to write.

**A strategy declares its own search.** `Strategy.sweep()` returns one
`(label, instance)` pair per parameter variant, and the trial ledger counts every
one of them: `min_gap_ranges` at 1.0 and at 1.2 are two hypotheses, not one
strategy with a knob (§17). Values are instance attributes rather than
`ClassVar`s so that one run can measure several, and `lab/measure.py` measures
exactly what the sweep declares — nothing else picks the grid.

**Importing `arsenal.py` does not populate the arsenal.** A plugin registers when
its own module is imported, so a run that wants one strategy imports that
strategy and gets only it. Which strategies exist in a run is configuration, not
a property of the code — §10.2 runs a champion beside a challenger.

## What a strategy can and cannot see

It receives a small bundle: the instrument, the moment, the market phase, the
current market conditions, and its declared feature values. That is all.

No price history, no clock, no broker. Each of those is withheld deliberately —
history would let it look ahead, a clock would break replay, and a broker would
let it skip the risk engine. Making those impossible is stronger than asking
everyone to remember.

A strategy also cannot read a feature it did not declare. The engine only
prepares what was asked for, so an undeclared one would be missing or computed
over the wrong window, and neither is visible in the number itself.

The session anchors are the one thing nobody declares: where the session opened,
how far it has travelled, the volume-weighted average price, the opening ranges
that have completed, and yesterday's close. §5.3 calls them shared
infrastructure feeding every other strategy, they cost one fold per bar, and
there is no history to load for them — so there is nothing for a declaration to
tell the host. They arrive whether or not anything asked, and they are `None`
before the session's first bar rather than half-filled.

## Market conditions gate what may run

Trend days, choppy days and reversal days reward different tools. Each strategy
declares which conditions it is allowed to fire in, and the host enforces it — so
a breakout strategy and a fade-the-breakout strategy can never be live at the
same time.

**Only one of those conditions is classified today.** The model that tells trend
from chop is an HMM and arrives in Phase 5. Fitting a stand-in now would put an
unvalidated model in the decision path of every result Phase 2 produces, and its
thresholds would spend trial budget that the deflated Sharpe counts against
every strategy. So `context.py` classifies the one part of §5.7 that is a clock
fact rather than a model — the 12:00–14:00 ET liquidity lull, a default no-trade
window — and reports everything else as unclassified, which grants nothing.

That leaves research needing a way to measure a strategy before the classifier
exists. The backtest engine has one: it can treat *unclassified* as permissive,
and it stamps the result to say it did. A condition that really was classified
still gates, so the lull stays closed even then, and a number produced that way
cannot later be mistaken for one produced under the real gate.
