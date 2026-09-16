"""Free sample vendor feeds: FirstRateData and Kibot.

Seeds the lab with something to work with before the IBKR backfill has run
long enough to matter (TRADER_PLAN §12.1 stage 2). Both vendors enter through
`MarketDataPort`, so ingest is the existing crawler, not a second code path —
see `firstrate.py` and `kibot.py`.
"""
