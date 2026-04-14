"""Canadian case lookup via the CanLII API (requires free API key)."""

from __future__ import annotations

import os
import time
from typing import Any

from ..models import Citation, CitationKind


class CanLIIClient:
    BASE = "https://api.canlii.org/v1"

    def __init__(
        self,
        api_key: str | None = None,
        session: Any = None,
        timeout: float = 10.0,
        min_interval: float = 0.3,
    ):
        self.api_key = api_key or os.environ.get("CANLII_API_KEY")
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
        return c.kind == CitationKind.FOREIGN_CASE and c.jurisdiction == "CA"

    def _throttle(self) -> None:
        d = time.monotonic() - self._last
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self._last = time.monotonic()

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        if not (self.api_key and c.neutral):
            return {"_error": "no CANLII_API_KEY"} if c.neutral else None
        self._throttle()
        try:
            r = self._session.get(
                f"{self.BASE}/caseCitator/en/search",
                params={"citation": c.neutral, "api_key": self.api_key},
                timeout=self.timeout,
            )
        except Exception as e:  # noqa: BLE001
            return {"_error": f"network: {e}"}
        if r.status_code != 200:
            return None if r.status_code == 404 else {"_error": f"http {r.status_code}"}
        data = r.json()
        matches = data.get("cases") or data.get("results") or []
        if not matches:
            return None
        first = matches[0]
        return {
            "canonical_title": first.get("title") or c.neutral,
            "canonical_year": c.year,
            "source": "canlii",
            "url": first.get("url"),
        }
