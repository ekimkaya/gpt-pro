"""Australian case lookup via AustLII (URL-pattern probe)."""

from __future__ import annotations

import time
from typing import Any

from ..models import Citation, CitationKind


class AustLIIClient:
    BASE = "http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases"
    _PATH = {
        "HCA": "cth/HCA",
        "FCA": "cth/FCA",
        "FCAFC": "cth/FCAFC",
        "NSWCA": "nsw/NSWCA",
        "VSCA": "vic/VSCA",
    }

    def __init__(self, session: Any = None, timeout: float = 10.0, min_interval: float = 0.4):
        self.timeout = timeout
        self.min_interval = min_interval
        self._last = 0.0
        if session is not None:
            self._session = session
        else:
            try:
                import requests
            except ImportError as e:
                raise ImportError("requests required") from e
            self._session = requests.Session()

    def supports(self, c: Citation) -> bool:
        return c.kind == CitationKind.FOREIGN_CASE and c.jurisdiction == "AU"

    def _throttle(self) -> None:
        d = time.monotonic() - self._last
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self._last = time.monotonic()

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        if not (c.year and c.court and c.neutral):
            return None
        stem = self._PATH.get(c.court.upper())
        if not stem:
            return {"_error": f"unsupported AU court: {c.court}"}
        num = c.neutral.split()[-1]
        url = f"{self.BASE}/{stem}/{c.year}/{num}.html"
        self._throttle()
        try:
            r = self._session.head(url, timeout=self.timeout, allow_redirects=True)
        except Exception as e:  # noqa: BLE001
            return {"_error": f"network: {e}"}
        if r.status_code == 200:
            return {"canonical_title": c.neutral, "source": "austlii", "url": url}
        if r.status_code == 404:
            return None
        return {"_error": f"http {r.status_code}"}
