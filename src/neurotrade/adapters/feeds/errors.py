"""Shared error type for the seed feed adapters.

`firstrate.py` and `kibot.py` both raise this, and `vendor_download.py` raises
it too. One class rather than several so a caller catching `FeedError` covers
every vendor without knowing which one served a given symbol.
"""

from __future__ import annotations

__all__ = ["FeedError"]


class FeedError(RuntimeError):
    """Raised when a seed feed or downloader cannot do what was asked.

    Distinct from "no data": an instrument with no bars in the requested range
    returns an empty sequence, which is ordinary — see `MarketDataPort`. This
    is raised only when the request itself cannot be answered: the symbol was
    never configured, the interval is not the 1-minute one the sample files
    hold, a vendor file's rows do not parse, or a download cannot be
    completed as asked (an existing file would be overwritten, or a wanted
    Kibot file matched no link on the page).
    """
