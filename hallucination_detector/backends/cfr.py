"""CFR lookup via the free eCFR API (api.ecfr.gov)."""

from __future__ import annotations

import time
from typing import Any

from ..models import Citation, CitationKind


class CFRClient:
    """Validates ``<title> C.F.R. § <section>`` against eCFR.

    Uses the public versioner endpoint
    ``/versioner/v1/structure/{date}/title-{N}.json`` and walks the tree
    for the requested section. Structure is cached per title.
    """

    BASE = "https://www.ecfr.gov/api"

    def __init__(self, session: Any = None, timeout: float = 10.0, min_interval: float = 0.5):
        self.timeout = timeout
        self.min_interval = min_interval
        self._last = 0.0
        self._structure: dict[str, dict] = {}
        if session is not None:
            self._session = session
        else:
            try:
                import requests
            except ImportError as e:
                raise ImportError("requests required") from e
            self._session = requests.Session()

    def supports(self, c: Citation) -> bool:
        return c.kind == CitationKind.REGULATION

    def _throttle(self) -> None:
        d = time.monotonic() - self._last
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self._last = time.monotonic()

    def _get_structure(self, title: str) -> dict | None:
        if title in self._structure:
            return self._structure[title]
        self._throttle()
        url = f"{self.BASE}/versioner/v1/structure/current/title-{title}.json"
        try:
            r = self._session.get(url, timeout=self.timeout)
        except Exception:
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        self._structure[title] = data
        return data

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        if not (c.title and c.section):
            return None
        structure = self._get_structure(c.title)
        if structure is None:
            return {"_error": "ecfr unavailable"}
        if _section_exists(structure, c.section):
            return {
                "canonical_title": f"{c.title} C.F.R. § {c.section}",
                "source": "ecfr",
            }
        return None


def _section_exists(node: dict, section: str) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") == "section" and node.get("identifier") == section:
        return True
    for child in node.get("children") or []:
        if _section_exists(child, section):
            return True
    return False
