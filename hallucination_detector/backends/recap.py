"""Federal docket lookup via CourtListener's RECAP archive."""

from __future__ import annotations

import os
import time
from typing import Any

from ..models import Citation, CitationKind


class RECAPClient:
    """Resolves federal docket numbers to dockets in the RECAP archive.

    RECAP mirrors a substantial (but not exhaustive) portion of PACER for
    free. For live/complete coverage, swap in a direct PACER client using
    the firm's PACER credentials.
    """

    BASE = "https://www.courtlistener.com/api/rest/v4"

    def __init__(
        self,
        token: str | None = None,
        session: Any = None,
        timeout: float = 10.0,
        min_interval: float = 0.3,
    ):
        self.token = token or os.environ.get("COURTLISTENER_TOKEN")
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
        return c.kind == CitationKind.DOCKET

    def _throttle(self) -> None:
        d = time.monotonic() - self._last
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self._last = time.monotonic()

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Token {self.token}"
        return h

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        if not c.docket_number:
            return None
        self._throttle()
        try:
            r = self._session.get(
                f"{self.BASE}/dockets/",
                params={"docket_number": c.docket_number},
                headers=self._headers(),
                timeout=self.timeout,
            )
        except Exception as e:  # noqa: BLE001
            return {"_error": f"network: {e}"}
        if r.status_code != 200:
            return None if r.status_code == 404 else {"_error": f"http {r.status_code}"}
        results = r.json().get("results") or []
        if not results:
            return None
        first = results[0]
        return {
            "canonical_title": first.get("case_name") or c.docket_number,
            "canonical_court": first.get("court"),
            "source": "recap",
            "url": first.get("absolute_url"),
        }
