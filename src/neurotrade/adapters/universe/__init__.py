"""Universe adapters: where the list of tradable instruments comes from.

A hand-written YAML file today, because §12.1 stage 1 needs a crawl queue
before stage 3 has built any universe history. Callers see `UniversePort` and
never learn which source answered.
"""
