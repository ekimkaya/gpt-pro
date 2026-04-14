"""UK case lookup via BAILII.

BAILII does not publish a stable JSON API; the firm's production build
should prefer The National Archives' free case law API at
https://caselaw.nationalarchives.gov.uk which does. We support both:

  * If ``prefer_tna=True`` (default), we hit the TNA JSON endpoint first.
  * Otherwise, or on TNA miss, we resolve the BAILII URL directly. A 200
    response is treated as existence-proof; 404 means the citation is
    fabricated.
"""

from __future__ import annotations

import time
from typing import Any

from ..models import Citation, CitationKind


class BAILIIClient:
    BAILII_BASE = "https://www.bailii.org"
    TNA_BASE = "https://caselaw.nationalarchives.gov.uk"

    # BAILII path stems by court.
    _PATH = {
        "UKSC": "/uk/cases/UKSC",
        "UKHL": "/uk/cases/UKHL",
        "UKPC": "/uk/cases/UKPC",
        "EWCA": "/ew/cases/EWCA/Civ",  # default to Civil; Crim handled below
        "EWHC": "/ew/cases/EWHC",
    }

    def __init__(
        self,
        session: Any = None,
        prefer_tna: bool = True,
        timeout: float = 10.0,
        min_interval: float = 0.4,
    ):
        self.timeout = timeout
        self.min_interval = min_interval
        self.prefer_tna = prefer_tna
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
        return c.kind == CitationKind.FOREIGN_CASE and c.jurisdiction == "UK"

    def _throttle(self) -> None:
        d = time.monotonic() - self._last
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self._last = time.monotonic()

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        if not (c.year and c.court and c.neutral):
            return None

        if self.prefer_tna:
            tna = self._lookup_tna(c)
            if tna is not None:
                return tna

        # Fall back to a BAILII HEAD request.
        stem = self._PATH.get(c.court.split()[0].upper())
        if not stem:
            return {"_error": f"unsupported UK court: {c.court}"}
        # Extract the case number from the neutral citation.
        num = c.neutral.split()[-1]
        url = f"{self.BAILII_BASE}{stem}/{c.year}/{num}.html"
        self._throttle()
        try:
            r = self._session.head(url, timeout=self.timeout, allow_redirects=True)
        except Exception as e:  # noqa: BLE001
            return {"_error": f"network: {e}"}
        if r.status_code == 200:
            return {"canonical_title": c.neutral, "source": "bailii", "url": url}
        if r.status_code == 404:
            return None
        return {"_error": f"http {r.status_code}"}

    def _lookup_tna(self, c: Citation) -> dict[str, Any] | None:
        self._throttle()
        try:
            r = self._session.get(
                f"{self.TNA_BASE}/search",
                params={"query": c.neutral, "per_page": 1},
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
        except Exception:
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get("results") or data.get("data") or []
        if not results:
            return None
        first = results[0]
        return {
            "canonical_title": first.get("name") or c.neutral,
            "canonical_year": c.year,
            "source": "tna",
            "url": first.get("uri"),
        }
