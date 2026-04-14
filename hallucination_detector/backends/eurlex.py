"""EU case lookup via EUR-Lex search API."""

from __future__ import annotations

import time
from typing import Any

from ..models import Citation, CitationKind


class EURLexClient:
    """Thin wrapper around EUR-Lex's free search endpoint.

    EUR-Lex supports citation search by ECLI or case number. We hit the
    public ``celex``/``ecli`` resolver and return the document record when
    found.
    """

    BASE = "https://eur-lex.europa.eu"

    def __init__(self, session: Any = None, timeout: float = 10.0, min_interval: float = 0.5):
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
        return c.kind == CitationKind.FOREIGN_CASE and c.jurisdiction == "EU"

    def _throttle(self) -> None:
        d = time.monotonic() - self._last
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self._last = time.monotonic()

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        if not c.neutral:
            return None
        q = c.neutral
        self._throttle()
        # EUR-Lex supports URL-based ECLI resolution: /legal-content/EN/TXT/?ecli=<ECLI>
        url = f"{self.BASE}/legal-content/EN/TXT/"
        params: dict[str, Any] = {"uri": f"ECLI:{q}"} if q.upper().startswith("ECLI:") else {"qid": q}
        try:
            r = self._session.get(
                url, params=params, headers={"Accept": "text/html"}, timeout=self.timeout
            )
        except Exception as e:  # noqa: BLE001
            return {"_error": f"network: {e}"}
        if r.status_code != 200 or "No documents matching" in r.text:
            return None
        return {"canonical_title": c.neutral, "source": "eurlex", "url": r.url}
