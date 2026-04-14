"""Backwards-compatible facade over the new multi-kind citation system.

The original :class:`CitationValidator` was US-case-only. It now delegates
to :class:`MultiJurisdictionValidator` with a single CourtListener-style
backend, so existing callers keep working while the full multi-backend
pipeline is available via :mod:`hallucination_detector.backends`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .citation_extractors import extract_us_cases
from .models import Citation, CitationKind, Finding
from .backends.multi import MultiJurisdictionValidator


# --------------------------------------------------------------------------- #
# Re-exports so older code keeps importing from the same place.
# --------------------------------------------------------------------------- #

from .backends.courtlistener import CourtListenerClient  # noqa: F401


def extract_legal_citations(text: str) -> list[Citation]:
    """Back-compat wrapper - US cases only, matches the original API."""
    return extract_us_cases(text)


class CaseLookupClient(Protocol):
    def lookup(self, citation: Citation) -> dict[str, Any] | None: ...


@dataclass
class CitationValidator:
    """Legacy US-case-only validator. New code should use
    :class:`MultiJurisdictionValidator` from :mod:`backends`."""

    client: CaseLookupClient
    name_similarity_threshold: float = 0.6

    def validate(self, text: str) -> tuple[list[Citation], list[Finding]]:
        citations = extract_us_cases(text)

        # Wrap the legacy client in the ``supports``-style interface the new
        # validator expects.
        class _Adapter:
            def supports(_self, c: Citation) -> bool:
                return c.kind == CitationKind.CASE

            def lookup(_self, c: Citation):
                return self.client.lookup(c)

        mv = MultiJurisdictionValidator(
            backends=[_Adapter()],
            name_similarity_threshold=self.name_similarity_threshold,
        )
        findings = mv.validate(citations)
        return citations, findings
