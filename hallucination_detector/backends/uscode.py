"""USC lookup via govinfo.gov (free, official)."""

from __future__ import annotations

import os
import time
from typing import Any

from ..models import Citation, CitationKind


class USCodeClient:
    """Looks up ``<title> U.S.C. § <section>`` via the govinfo collection API.

    govinfo exposes the US Code as structured documents; we query by
    package identifier (``USCODE-<year>-title<NN>``) or via granule search.
    For detection purposes we only need: (a) does the section exist,
    (b) optionally the section text so the judge can verify quoted language.
    """

    BASE = "https://api.govinfo.gov"

    def __init__(
        self,
        api_key: str | None = None,
        session: Any = None,
        timeout: float = 10.0,
        min_interval: float = 0.25,
    ):
        self.api_key = api_key or os.environ.get("GOVINFO_API_KEY", "DEMO_KEY")
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
        return c.kind == CitationKind.STATUTE and c.jurisdiction == "US"

    def _throttle(self) -> None:
        d = time.monotonic() - self._last
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self._last = time.monotonic()

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        if not (c.title and c.section):
            return None
        self._throttle()
        q = f'collection:USCODE AND title:"{c.title}" AND section:"{c.section}"'
        try:
            r = self._session.get(
                f"{self.BASE}/search",
                params={"query": q, "api_key": self.api_key, "pageSize": 1},
                timeout=self.timeout,
            )
        except Exception as e:  # noqa: BLE001
            return {"_error": f"network: {e}"}
        if r.status_code != 200:
            return {"_error": f"http {r.status_code}"}
        hits = r.json().get("results") or []
        if not hits:
            return None
        hit = hits[0]
        return {
            "canonical_title": f"{c.title} U.S.C. § {c.section}",
            "package_id": hit.get("packageId"),
            "granule_id": hit.get("granuleId"),
            "body_url": hit.get("download", {}).get("txtLink"),
        }

    def fetch_section_text(self, body_url: str) -> str | None:
        if not body_url:
            return None
        self._throttle()
        try:
            r = self._session.get(body_url, timeout=self.timeout)
        except Exception:
            return None
        return r.text if r.status_code == 200 else None
