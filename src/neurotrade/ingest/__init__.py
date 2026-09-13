"""Building the corpus: deciding what to fetch.

§12.1 makes the training corpus a first-class asset with its own build plan,
separate from live data. This layer is that plan in code.

It depends on `core` alone — ports, never adapters. A crawler that imported the
IBKR adapter could not be tested without a broker, and the whole point of the
hexagon is that it can be.
"""
