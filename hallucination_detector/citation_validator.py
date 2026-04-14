"""Layer 3 — Fact-checking legal citations via external APIs.

The public CourtListener API (https://www.courtlistener.com/api/) exposes
millions of US case opinions and lets us look up citations like
``410 U.S. 113`` or case names like ``Roe v. Wade``. If the citation
returns no hits, or the first hit's case name doesn't match what the model
claimed, we block the output - that's the hard "capture".

For firms using Westlaw or Lexis, swap :class:`CourtListenerClient` for a
wrapper around their respective APIs; the :class:`CitationValidator`
interface stays the same.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .models import Citation, CitationStatus, Finding, Severity


# --------------------------------------------------------------------------- #
# Citation extraction
# --------------------------------------------------------------------------- #


# Covers the common US reporter patterns: "410 U.S. 113", "123 F.3d 456",
# "567 F. Supp. 2d 890", "410 U. S. 113". The reporter group is kept loose
# and normalized below.
_RAW_CITE_RE = re.compile(
    r"""
    (?P<volume>\d{1,4})\s+
    (?P<reporter>
        (?:U\.?\s?S\.?)
      | (?:S\.?\s?Ct\.?)
      | (?:L\.?\s?Ed\.?(?:\s?2d)?)
      | (?:F\.?\s?(?:2d|3d|4th|Supp\.?(?:\s?2d|\s?3d)?)?)
      | (?:N\.?\s?[EW]\.?(?:\s?2d|\s?3d)?)
      | (?:A\.?(?:\s?2d|\s?3d)?)
      | (?:P\.?(?:\s?2d|\s?3d)?)
      | (?:S\.?\s?[EW]\.?(?:\s?2d|\s?3d)?)
      | (?:Cal\.?(?:\s?App\.?)?(?:\s?\d[a-z]{2})?)
    )\s+
    (?P<page>\d{1,5})
    (?:\s*\((?P<year>\d{4})\))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# "Smith v. Jones" style prefix - each party must be a sequence of Capitalized
# words (allowing &, ., etc.) so we don't accidentally capture lowercase
# sentence prefixes like "the controlling authority is ...".
_CAP_WORD = r"(?:[A-Z][A-Za-z0-9\.\-'’&]*|of|the|and|in|for|de)"
_CASENAME_RE = re.compile(
    rf"((?:[A-Z][A-Za-z0-9\.\-'’&]*)(?:\s+{_CAP_WORD}){{0,6}}?\s+v\.?\s+"
    rf"(?:[A-Z][A-Za-z0-9\.\-'’&]*)(?:\s+{_CAP_WORD}){{0,6}}?)[,\s]+"
)


def _normalize_reporter(rep: str) -> str:
    rep = re.sub(r"\s+", " ", rep.strip())
    # Canonicalize common forms.
    subs = [
        (r"^U\.?\s?S\.?$", "U.S."),
        (r"^S\.?\s?Ct\.?$", "S. Ct."),
        (r"^L\.?\s?Ed\.?(\s?2d)?$", lambda m: "L. Ed. 2d" if m.group(1) else "L. Ed."),
        (r"^F\.?\s?$", "F."),
        (r"^F\.?\s?2d$", "F.2d"),
        (r"^F\.?\s?3d$", "F.3d"),
        (r"^F\.?\s?4th$", "F.4th"),
        (r"^F\.?\s?Supp\.?$", "F. Supp."),
        (r"^F\.?\s?Supp\.?\s?2d$", "F. Supp. 2d"),
        (r"^F\.?\s?Supp\.?\s?3d$", "F. Supp. 3d"),
    ]
    for pat, repl in subs:
        if re.match(pat, rep, re.IGNORECASE):
            return repl(re.match(pat, rep, re.IGNORECASE)) if callable(repl) else repl
    return rep


def extract_legal_citations(text: str) -> list[Citation]:
    """Parse ``text`` and return every recognizable US reporter citation.

    Case names are attached when a ``Party v. Party`` prefix appears within
    120 characters before the citation.
    """

    cites: list[Citation] = []
    for m in _RAW_CITE_RE.finditer(text):
        start = m.start()
        pre = text[max(0, start - 120):start]
        name_match = list(_CASENAME_RE.finditer(pre))
        case_name = name_match[-1].group(1).strip(" ,") if name_match else None
        reporter = _normalize_reporter(m.group("reporter"))
        cites.append(
            Citation(
                raw=m.group(0),
                volume=m.group("volume"),
                reporter=reporter,
                page=m.group("page"),
                year=int(m.group("year")) if m.group("year") else None,
                case_name=case_name,
            )
        )
    return cites


# --------------------------------------------------------------------------- #
# External API client
# --------------------------------------------------------------------------- #


class CaseLookupClient(Protocol):
    """Pluggable citation lookup backend (CourtListener, Westlaw, Lexis, ...)."""

    def lookup(self, citation: Citation) -> dict[str, Any] | None: ...


class CourtListenerClient:
    """Minimal CourtListener REST client for citation lookup.

    Uses the public ``/api/rest/v4/search/`` endpoint. An API token is
    strongly recommended for production use (rate limits). Pass one via the
    ``token`` argument or the ``COURTLISTENER_TOKEN`` env var.
    """

    BASE_URL = "https://www.courtlistener.com/api/rest/v4/search/"

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

    def _throttle(self) -> None:
        delta = time.monotonic() - self._last_call
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last_call = time.monotonic()

    def lookup(self, citation: Citation) -> dict[str, Any] | None:
        self._throttle()
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Token {self.token}"
        q = citation.normalized
        params = {"q": q, "type": "o"}  # 'o' = opinions
        try:
            r = self._session.get(
                self.BASE_URL, params=params, headers=headers, timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001 - surface as API_ERROR
            return {"_error": f"network: {exc}"}
        if r.status_code >= 500:
            return {"_error": f"http {r.status_code}"}
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            return {"_error": f"http {r.status_code}: {r.text[:200]}"}
        data = r.json()
        results = data.get("results", [])
        return results[0] if results else None


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


@dataclass
class CitationValidator:
    """Extract citations from text and verify them against a lookup backend."""

    client: CaseLookupClient
    name_similarity_threshold: float = 0.6

    def validate(self, text: str) -> tuple[list[Citation], list[Finding]]:
        citations = extract_legal_citations(text)
        findings: list[Finding] = []
        for c in citations:
            result = self.client.lookup(c)
            if result is None:
                c.status = CitationStatus.NOT_FOUND
                findings.append(
                    Finding(
                        layer="api",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Citation {c.normalized!r} not found in case law database."
                            + (f" (claimed case: {c.case_name})" if c.case_name else "")
                        ),
                        evidence={"citation": c.normalized, "case_name": c.case_name},
                    )
                )
                continue
            if isinstance(result, dict) and result.get("_error"):
                c.status = CitationStatus.API_ERROR
                c.notes = result["_error"]
                findings.append(
                    Finding(
                        layer="api",
                        severity=Severity.MEDIUM,
                        message=f"Could not verify {c.normalized!r}: {result['_error']}",
                    )
                )
                continue

            canonical = (
                result.get("caseName")
                or result.get("case_name")
                or result.get("caseNameShort")
                or ""
            )
            c.canonical_title = canonical
            if c.case_name and canonical:
                similarity = _name_similarity(c.case_name, canonical)
                if similarity < self.name_similarity_threshold:
                    c.status = CitationStatus.MISMATCH
                    findings.append(
                        Finding(
                            layer="api",
                            severity=Severity.CRITICAL,
                            message=(
                                f"Citation {c.normalized!r} resolves to {canonical!r}, "
                                f"but the model referred to it as {c.case_name!r}."
                            ),
                            evidence={
                                "citation": c.normalized,
                                "claimed": c.case_name,
                                "actual": canonical,
                                "similarity": round(similarity, 3),
                            },
                        )
                    )
                    continue
            c.status = CitationStatus.VERIFIED
        return citations, findings


def _name_similarity(a: str, b: str) -> float:
    """Token-overlap similarity after stripping punctuation and common legal noise."""
    def norm(s: str) -> set[str]:
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        stop = {"v", "vs", "the", "of", "and", "inc", "llc", "corp", "co", "ltd"}
        return {w for w in s.split() if w and w not in stop}
    A, B = norm(a), norm(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
