"""Domain model: value objects, events, trading objects, ports, clock.

The bottom of the dependency graph. This package imports nothing else in the
repo — not configuration, not adapters, not storage. Everything above it may
depend on it; it depends on nobody. That is what makes the trading logic
testable without a broker, a database or a network.
"""
