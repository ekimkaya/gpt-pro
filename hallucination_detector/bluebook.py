"""Simple Bluebook / ALWD format linter.

This doesn't try to reimplement the full Bluebook; it catches the handful
of format errors that tend to cluster around AI-generated citations
(double spaces inside reporters, missing year parens, mixed styles).

The linter emits findings at LOW/MEDIUM severity - these don't block a
filing on their own, but a cluster of them alongside other signals
strongly suggests the citation was generated rather than looked up.
"""

from __future__ import annotations

import re

from .models import Citation, CitationKind, Finding, Severity


_DOUBLE_SPACE_RE = re.compile(r"  +")
_BAD_REPORTER_RE = re.compile(r"\b(F|A|P|N|S)\s+\d", re.IGNORECASE)  # "F 3d" without dot
_LOWERCASE_PARTY_RE = re.compile(r"\b[a-z]+ v\. [A-Z]")


def lint(text: str, citations: list[Citation]) -> list[Finding]:
    findings: list[Finding] = []

    for c in citations:
        if not c.span:
            continue
        raw = c.raw

        if _DOUBLE_SPACE_RE.search(raw):
            findings.append(
                Finding(
                    layer="bluebook",
                    severity=Severity.LOW,
                    message=f"Citation {raw!r} contains double spaces.",
                    span=c.span,
                )
            )

        if c.kind == CitationKind.CASE:
            if c.year is None and not c.resolves_to:
                findings.append(
                    Finding(
                        layer="bluebook",
                        severity=Severity.LOW,
                        message=f"Citation {raw!r} is missing a year in parentheses.",
                        span=c.span,
                    )
                )
            if c.reporter and "." not in c.reporter and c.reporter not in ("U.S.",):
                # Most reporters abbreviate with a period; bare "F 3d" is a red flag.
                if _BAD_REPORTER_RE.search(raw):
                    findings.append(
                        Finding(
                            layer="bluebook",
                            severity=Severity.MEDIUM,
                            message=(
                                f"Citation {raw!r} uses a non-standard reporter "
                                "abbreviation (missing periods)."
                            ),
                            span=c.span,
                        )
                    )
            if c.case_name and _LOWERCASE_PARTY_RE.search(c.case_name):
                findings.append(
                    Finding(
                        layer="bluebook",
                        severity=Severity.LOW,
                        message=f"Case name {c.case_name!r} has a lowercase first-party word.",
                        span=c.span,
                    )
                )

        if c.kind == CitationKind.STATUTE:
            if c.section and c.section.startswith("."):
                findings.append(
                    Finding(
                        layer="bluebook",
                        severity=Severity.LOW,
                        message=f"Statute section {c.section!r} starts with a period.",
                        span=c.span,
                    )
                )

    return findings
