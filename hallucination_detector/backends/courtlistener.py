"""CourtListener REST client with opinion-text fetch for quote checking."""

from __future__ import annotations

import os
import time
from typing import Any

from ..models import Citation, CitationKind


class CourtListenerClient:
    """Minimal CourtListener REST client.

    Supports the two operations we need:
      * :meth:`lookup` - resolve a citation to an opinion record.
      * :meth:`fetch_opinion_text` - pull the full opinion body for quote
        attribution.

    Pass an API token via ``token=`` or ``COURTLISTENER_TOKEN`` env var.
    """

    BASE = "https://www.courtlistener.com/api/rest/v4"

    def __init__(
        self,
        token: str | None = None,
        session: Any = None,
        timeout: float = 10.0,
        min_interval: float = 0.25,
    ):
        self.token = token or os.environ.get("COURTLISTENER_TOKEN")
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_call = 0.0
        if session is not None:
            self._session = session
        else:
            try:
                import requests
            except ImportError as e:
                raise ImportError(
                    "requests is required for CourtListenerClient. pip install requests"
                ) from e
            self._session = requests.Session()
        self._text_cache: dict[str, str] = {}

    def _throttle(self) -> None:
        delta = time.monotonic() - self._last_call
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last_call = time.monotonic()

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Token {self.token}"
        return h

    def supports(self, c: Citation) -> bool:
        return c.kind == CitationKind.CASE

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        self._throttle()
        try:
            r = self._session.get(
                f"{self.BASE}/search/",
                params={"q": c.normalized, "type": "o"},
                headers=self._headers(),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return {"_error": f"network: {exc}"}
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            return {"_error": f"http {r.status_code}: {r.text[:200]}"}
        results = r.json().get("results", [])
        return results[0] if results else None

    def fetch_opinion_text(self, opinion_id: str | int) -> str | None:
        """Fetch the full opinion body by CourtListener opinion ID. Cached."""
        key = str(opinion_id)
        if key in self._text_cache:
            return self._text_cache[key]
        self._throttle()
        try:
            r = self._session.get(
                f"{self.BASE}/opinions/{opinion_id}/",
                headers=self._headers(),
                timeout=self.timeout,
            )
        except Exception:
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        text = (
            data.get("plain_text")
            or data.get("html_with_citations")
            or data.get("html")
            or ""
        )
        self._text_cache[key] = text
        return text or None
