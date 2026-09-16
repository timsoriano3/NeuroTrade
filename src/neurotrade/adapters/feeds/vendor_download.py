"""Downloads FirstRateData and Kibot free samples over plain HTTPS.

Both vendors serve files with no login (§12.1 stage 2, decision 1), so this
downloads them itself rather than depending on a manual step that would just
be a way to get it wrong. A file dropped in by hand works identically — the
feeds only read a directory, however the file got there.

**FirstRateData** is a direct, predictable URL per ticker. **Kibot** is not:
the file for a ticker comes from an opaque `?get=<code>` link scraped off its
free-sample page, and the real file name only appears in the response's
`Content-Disposition` header — the code itself carries no information a caller
could use up front.

Only `urllib` — no HTTP client dependency for a handful of one-shot downloads.
The opener is injected (`HttpOpener`) so every test here runs against a fake
and never touches the network.

**Vendor files are raw and never modified once written** (decision 2): each
lands in a dated folder, `<raw_dir>/vendor/<source>/<YYYY-MM-DD>/<name>` —
dated because Kibot's rolling window means every fetch is a new snapshot,
never an overwrite of an old one. A `manifest.json` beside the files records
how each one got there: url, sha256, size, fetch time, and adjustment basis.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import certifi

from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.core.clock import Clock

__all__ = [
    "HttpOpener",
    "HttpResponse",
    "UrllibOpener",
    "fetch_firstrate_samples",
    "fetch_kibot_samples",
]

_USER_AGENT = "NeuroTrade-seed-fetch/0.0 (+https://github.com/petersoriano/NeuroTrade)"
_TIMEOUT_SECONDS = 30.0

_FRD_URL = "https://frd001.s3-us-east-2.amazonaws.com/{ticker}_1min_sample_firstratedata.zip"
_KIBOT_PAGE_URL = "https://www.kibot.com/free-historical-intraday-data.html"
_KIBOT_LINK = re.compile(r"https://api\.kibot\.com/\?get=[\w-]+")
_CONTENT_DISPOSITION_FILENAME = re.compile(r'filename="?([^";]+)"?')


class HttpResponse(Protocol):
    """The part of an HTTP response this module reads.

    `http.client.HTTPResponse` — what `urllib.request.urlopen` returns —
    satisfies this structurally; nothing here imports it.
    """

    def read(self) -> bytes:
        """The response body, in full."""
        ...

    def getheader(self, name: str) -> str | None:
        """One header's value, or `None` if it was not sent."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class HttpOpener(Protocol):
    """Makes one GET request. Injected so tests never touch the network."""

    def open(self, url: str, *, timeout: float) -> HttpResponse:
        """Issue a GET and return the response, headers already available."""
        ...


class UrllibOpener:
    """The real opener: `urllib.request`, with a User-Agent and a timeout.

    Example:
        >>> callable(UrllibOpener().open)
        True
    """

    __slots__ = ()

    def open(self, url: str, *, timeout: float) -> HttpResponse:
        """Open `url`, identifying this fetch rather than sending no UA.

        Args:
            url: The address to GET.
            timeout: Seconds to wait before giving up.

        Returns:
            The live response. Structurally an `HttpResponse`.
        """
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        # certifi's CA bundle, not the interpreter's default store: python.org's
        # macOS build ships with no CA certificates, so the default context
        # rejects every HTTPS host until a post-install script is run by hand.
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(request, timeout=timeout, context=context)  # type: ignore[no-any-return]


def fetch_firstrate_samples(
    tickers: Sequence[str],
    *,
    raw_dir: Path,
    clock: Clock,
    opener: HttpOpener | None = None,
) -> tuple[Path, ...]:
    """Download each ticker's FirstRateData free sample.

    Args:
        tickers: Tickers to fetch, e.g. `("AAPL", "SPY")`. Each becomes
            `{ticker}_1min_sample_firstratedata.zip` at a fixed, documented
            URL.
        raw_dir: The corpus's raw root (`StorageSettings.raw_dir`). Files land
            under `<raw_dir>/vendor/firstrate/<YYYY-MM-DD>/`.
        clock: Names the dated folder and stamps the manifest. Never the wall
            clock directly — see the `LiveClock` invariant in `CLAUDE.md`.
        opener: HTTP opener. Defaults to `UrllibOpener()`; tests inject a fake.

    Returns:
        Paths written, one per ticker, in the order given.

    Raises:
        FeedError: If a destination file already exists. Vendor files are
            never overwritten (decision 2): a second run for the same day is
            either a no-op worth skipping deliberately or a mistake, and a
            mistake should not clobber a fetch someone might be relying on.

    Example:
        >>> class FakeResponse:
        ...     def __init__(self, data):
        ...         self._data = data
        ...     def read(self):
        ...         return self._data
        ...     def getheader(self, name):
        ...         return None
        ...     def close(self):
        ...         pass
        >>> class FakeOpener:
        ...     def open(self, url, *, timeout):
        ...         return FakeResponse(b"zip-bytes")
        >>> import tempfile
        >>> from neurotrade.core.clock import SimClock
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     paths = fetch_firstrate_samples(
        ...         ["AAPL"], raw_dir=Path(directory), clock=SimClock(0), opener=FakeOpener()
        ...     )
        ...     paths[0].name
        'AAPL_1min_sample_firstratedata.zip'
    """
    opener = opener if opener is not None else UrllibOpener()
    folder = _dated_folder(raw_dir, "firstrate", clock)
    written: list[Path] = []
    for ticker in tickers:
        name = f"{ticker}_1min_sample_firstratedata.zip"
        url = _FRD_URL.format(ticker=ticker)
        destination = _reserve(folder, name)
        response = opener.open(url, timeout=_TIMEOUT_SECONDS)
        try:
            data = response.read()
        finally:
            response.close()
        _write(destination, data)
        _update_manifest(folder, name=name, url=url, data=data, clock=clock, adjusted="unknown")
        written.append(destination)
    return tuple(written)


def fetch_kibot_samples(
    wanted_files: Sequence[str],
    *,
    raw_dir: Path,
    clock: Clock,
    opener: HttpOpener | None = None,
) -> tuple[Path, ...]:
    """Download the named files from Kibot's free-sample page.

    Kibot names files only in the response, not in the link: the page lists
    opaque `?get=<code>` links, and the real name arrives on the
    `Content-Disposition` header of the response each one produces. This reads
    that header off every link, keeping only the ones matching
    `wanted_files`.

    Args:
        wanted_files: Exact file names to keep, e.g. `("IBM_unadjusted.txt",)`.
        raw_dir: The corpus's raw root. Files land under
            `<raw_dir>/vendor/kibot/<YYYY-MM-DD>/`.
        clock: Names the dated folder and stamps the manifest.
        opener: HTTP opener. Defaults to `UrllibOpener()`; tests inject a fake.

    Returns:
        Paths written, one per matched name, in the order the page listed the
        links.

    Raises:
        FeedError: If any name in `wanted_files` matched no link on the page,
            or a destination file already exists (see
            `fetch_firstrate_samples`). Kibot's codes are undocumented and may
            rotate (open question in the plan); failing loudly here surfaces
            that rather than an ingest quietly running on last month's file.

    Example:
        >>> class FakeResponse:
        ...     def __init__(self, data, disposition=None):
        ...         self._data, self._disposition = data, disposition
        ...     def read(self):
        ...         return self._data
        ...     def getheader(self, name):
        ...         return self._disposition if name == "Content-Disposition" else None
        ...     def close(self):
        ...         pass
        >>> class FakeOpener:
        ...     def open(self, url, *, timeout):
        ...         if url == _KIBOT_PAGE_URL:
        ...             return FakeResponse(b'<a href="https://api.kibot.com/?get=abc123">x</a>')
        ...         return FakeResponse(b"csv-bytes", 'attachment; filename="IBM_unadjusted.txt"')
        >>> import tempfile
        >>> from neurotrade.core.clock import SimClock
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     paths = fetch_kibot_samples(
        ...         ["IBM_unadjusted.txt"], raw_dir=Path(directory),
        ...         clock=SimClock(0), opener=FakeOpener(),
        ...     )
        ...     paths[0].name
        'IBM_unadjusted.txt'
    """
    opener = opener if opener is not None else UrllibOpener()
    page = opener.open(_KIBOT_PAGE_URL, timeout=_TIMEOUT_SECONDS)
    try:
        html = page.read().decode("utf-8", errors="replace")
    finally:
        page.close()

    links = list(dict.fromkeys(_KIBOT_LINK.findall(html)))  # de-dup, keep first-seen order
    remaining = set(wanted_files)
    folder = _dated_folder(raw_dir, "kibot", clock)
    written: list[Path] = []

    for url in links:
        if not remaining:
            break
        response = opener.open(url, timeout=_TIMEOUT_SECONDS)
        try:
            name = _filename_from_content_disposition(response.getheader("Content-Disposition"))
            if name is None or name not in remaining:
                continue
            data = response.read()
        finally:
            response.close()
        destination = _reserve(folder, name)
        _write(destination, data)
        _update_manifest(folder, name=name, url=url, data=data, clock=clock, adjusted="unadjusted")
        written.append(destination)
        remaining.discard(name)

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise FeedError(f"kibot page had no link for: {missing}")
    return tuple(written)


def _filename_from_content_disposition(header: str | None) -> str | None:
    """Pull the file name out of a `Content-Disposition` header, if present."""
    if header is None:
        return None
    match = _CONTENT_DISPOSITION_FILENAME.search(header)
    return match.group(1) if match else None


def _dated_folder(raw_dir: Path, source: str, clock: Clock) -> Path:
    """`<raw_dir>/vendor/<source>/<YYYY-MM-DD>`, dated from the clock.

    Never the OS clock directly, so a research replay of a fetch step names
    the same folder every time it is re-run.
    """
    session_date = clock.now().date().isoformat()
    return raw_dir / "vendor" / source / session_date


def _reserve(folder: Path, name: str) -> Path:
    """Claim a destination file name, refusing to reuse one that exists.

    Raises:
        FeedError: If `folder / name` already exists.
    """
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / name
    if destination.exists():
        raise FeedError(f"refusing to overwrite existing vendor file: {destination}")
    return destination


def _write(destination: Path, data: bytes) -> None:
    """Write via a temp file plus atomic rename.

    A crash mid-write leaves only a stray temp file, never a truncated file
    sitting at `destination` looking complete.
    """
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, destination)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _update_manifest(
    folder: Path, *, name: str, url: str, data: bytes, clock: Clock, adjusted: str
) -> None:
    """Merge one file's record into `<folder>/manifest.json`.

    A folder can hold several files fetched over several calls — every FRD
    ticker in one run, or a fresh Kibot snapshot beside an older one fetched
    on the same day — so this reads the existing manifest and merges rather
    than replacing it.

    Args:
        folder: The dated vendor folder the file was written into.
        name: The file name, used as the manifest key.
        url: Where it was fetched from.
        data: Its bytes, hashed and sized here rather than passed in twice.
        clock: Stamps `fetched_at`.
        adjusted: `"unknown"` for FirstRateData (unverified for the sample —
            see the plan's open question) or `"unadjusted"` for Kibot
            (decision 7: only the `_unadjusted` files are ever fetched).
    """
    manifest_path = folder / "manifest.json"
    manifest: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest[name] = {
        "name": name,
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "fetched_at": clock.now_ns(),
        "adjusted": adjusted,
    }
    fd, tmp_name = tempfile.mkstemp(dir=folder, prefix=".tmp-manifest-")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(tmp_name, manifest_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
