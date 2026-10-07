"""Rate-limited, disk-cached HTTP fetcher for racing.hkjc.com."""
from __future__ import annotations

import gzip
import hashlib
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

# HKJC moved racing information from /racing/information/English/... (now a 302) to /en-us/local/information/...
BASE = "https://racing.hkjc.com/en-us/local/information"
USER_AGENT = "gc-hkg-racing personal research (contact via site owner)"


class Fetcher:
    def __init__(self, cache_dir: str | Path = "data/cache", delay: float = 3.0,
                 session: requests.Session | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last = 0.0
        self.live_requests = 0

    def cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / key[:2] / f"{key}.html.gz"

    def get(self, url: str, refresh: bool = False) -> str:
        path = self.cache_path(url)
        if path.exists() and not refresh:
            return gzip.decompress(path.read_bytes()).decode("utf-8")
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()  # delay is measured between request starts
        for attempt in range(4):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        self.live_requests += 1
        resp.encoding = resp.encoding or "utf-8"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(resp.text.encode("utf-8")))
        tmp.replace(path)
        return resp.text


def page_url(page: str, **params) -> str:
    """Build a racing-information URL, e.g. page_url("localresults", racedate="2025/09/07", RaceNo=1)."""
    query = urlencode({k: v for k, v in params.items() if v is not None})
    return f"{BASE}/{page}" + (f"?{query}" if query else "")
