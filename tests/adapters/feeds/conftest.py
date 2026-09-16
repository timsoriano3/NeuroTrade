"""A fake HTTP opener, shared by the `vendor_download` tests.

Nothing here reaches a network. An unlisted URL raises rather than returning
something plausible, so a test that opens the wrong address fails loudly
instead of silently passing against the wrong fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeResponse:
    """One canned HTTP response."""

    data: bytes
    content_disposition: str | None = None

    def read(self) -> bytes:
        return self.data

    def getheader(self, name: str) -> str | None:
        if name == "Content-Disposition":
            return self.content_disposition
        return None

    def close(self) -> None:
        pass


@dataclass
class FakeOpener:
    """Returns a canned `FakeResponse` per URL. Records what was opened."""

    responses: dict[str, FakeResponse]
    opened: list[str] = field(default_factory=list)

    def open(self, url: str, *, timeout: float) -> FakeResponse:
        self.opened.append(url)
        if url not in self.responses:
            raise AssertionError(f"test fixture has no response for URL: {url}")
        return self.responses[url]
