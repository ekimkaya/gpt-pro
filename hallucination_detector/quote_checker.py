"""Verify that quoted language actually appears in the cited opinion.

This is the "real citation, fake quote" failure mode — arguably more
dangerous than fabricated citations because the citation passes basic
validation. After :class:`MultiJurisdictionValidator` populates
``opinion_id`` (and the backend caches opinion text), we:

  1. Extract quoted spans from the draft that sit within ``window_chars``
     of a citation.
  2. Attribute each quote to its nearest preceding verified citation.
  3. Fetch the opinion text (via a callable) and look for the quote as
     a substring. If absent, try a fuzzy (Levenshtein ratio) match.
  4. Miss => :class:`CitationStatus.QUOTE_MISMATCH` + CRITICAL finding.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Callable

from .models import Citation, CitationKind, CitationStatus, Finding, Quote, Severity


_QUOTE_RE = re.compile(r"[“\"]([^\"”]{12,600})[\"”]")


def attribute_quotes(text: str, citations: list[Citation], window_chars: int = 240) -> None:
    """Attach each discovered quote to its nearest preceding citation.

    Mutates ``citations`` in place, populating :attr:`Citation.quotes`.
    """
    for m in _QUOTE_RE.finditer(text):
        q = Quote(text=m.group(1), start=m.start(), end=m.end())
        best: Citation | None = None
        best_distance = window_chars + 1
        for c in citations:
            if not c.span:
                continue
            c_start, c_end = c.span
            # Distance between the two spans; 0 if they overlap.
            if q.end <= c_start:
                distance = c_start - q.end
            elif c_end <= q.start:
                distance = q.start - c_end
            else:
                distance = 0
            if distance <= window_chars and distance < best_distance:
                best, best_distance = c, distance
        if best is not None:
            q.attributed_citation = best.normalized
            best.quotes.append(q)


@dataclass
class QuoteAttributionChecker:
    """Compares quoted language against the source opinion text."""

    # Callable: opinion_id -> full text (None if unavailable). Usually
    # ``CourtListenerClient.fetch_opinion_text``.
    fetch_text: Callable[[str], str | None]
    fuzzy_threshold: float = 0.88

    def verify(self, citations: list[Citation]) -> list[Finding]:
        findings: list[Finding] = []
        for c in citations:
            if c.kind != CitationKind.CASE or not c.quotes:
                continue
            if c.status not in (CitationStatus.VERIFIED, CitationStatus.UNCHECKED):
                # Don't re-run on already-failed cites; upstream owns them.
                continue
            if c.opinion_text is None and c.opinion_id:
                c.opinion_text = self.fetch_text(c.opinion_id)
            if not c.opinion_text:
                # Can't verify, but don't block on infrastructure failure.
                findings.append(
                    Finding(
                        layer="quote",
                        severity=Severity.LOW,
                        message=(
                            f"Could not fetch opinion text for {c.normalized!r}; "
                            f"{len(c.quotes)} quote(s) not verified."
                        ),
                        span=c.span,
                    )
                )
                continue
            haystack = _normalize(c.opinion_text)
            for q in c.quotes:
                needle = _normalize(q.text)
                if needle in haystack:
                    continue
                # Fuzzy fallback for paraphrased quotes / whitespace noise.
                ratio = _best_ratio(needle, haystack)
                if ratio >= self.fuzzy_threshold:
                    continue
                c.status = CitationStatus.QUOTE_MISMATCH
                findings.append(
                    Finding(
                        layer="quote",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Quoted language attributed to {c.normalized!r} does "
                            "not appear in the opinion text."
                        ),
                        evidence={
                            "quote": q.text[:200],
                            "best_match_ratio": round(ratio, 3),
                        },
                        span=(q.start, q.end),
                    )
                )
        return findings


def _normalize(s: str) -> str:
    # Lowercase, collapse whitespace, strip punctuation that commonly differs
    # between the draft and the source opinion (trailing period vs. comma,
    # curly quotes, parenthetical nests).
    s = re.sub(r"\s+", " ", s.lower())
    s = re.sub(r"[“”‘’\"'`]", "", s)
    s = re.sub(r"[.,;:!?()\[\]]", "", s)
    return s.strip()


def _best_ratio(needle: str, haystack: str) -> float:
    """Approximate best matching substring ratio without quadratic scan.

    We split the haystack into overlapping windows of length ≈ len(needle)
    and take the max SequenceMatcher ratio. Cheap, good enough for typical
    opinion sizes.
    """
    if not needle or not haystack:
        return 0.0
    n = len(needle)
    stride = max(n // 2, 1)
    best = 0.0
    for i in range(0, max(len(haystack) - n + 1, 1), stride):
        window = haystack[i:i + n + 40]
        r = difflib.SequenceMatcher(a=needle, b=window, autojunk=False).ratio()
        if r > best:
            best = r
            if best >= 0.99:
                break
    return best
