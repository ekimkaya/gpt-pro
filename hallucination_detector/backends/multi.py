"""The one validator the orchestrator talks to.

It owns a list of backend clients, each of which advertises support via
``supports(citation) -> bool``. For every extracted citation we route to
the first backend that claims it, compare the backend's canonical record
against the model's claim (name, year, court, judge), and emit findings
for every disagreement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

from ..models import (
    Citation,
    CitationKind,
    CitationStatus,
    Finding,
    Severity,
)


class LookupBackend(Protocol):
    def supports(self, c: Citation) -> bool: ...
    def lookup(self, c: Citation) -> dict[str, Any] | None: ...


@dataclass
class MultiJurisdictionValidator:
    backends: list[LookupBackend]
    name_similarity_threshold: float = 0.6
    recent_days: int = 30  # citations newer than this that miss => UNCHECKED_RECENT
    overrides: set[str] = field(default_factory=set)  # attorney-signed normalized cites

    # ----- public -------------------------------------------------------- #

    def validate(self, citations: list[Citation]) -> list[Finding]:
        findings: list[Finding] = []
        for c in citations:
            if c.kind == CitationKind.SHORT_FORM:
                # Unresolved short forms were already flagged by the resolver.
                continue
            if c.normalized in self.overrides:
                c.status = CitationStatus.SEALED_OVERRIDE
                c.notes = "attorney override"
                continue
            backend = self._route(c)
            if backend is None:
                c.status = CitationStatus.UNCHECKED
                findings.append(
                    Finding(
                        layer="api",
                        severity=Severity.MEDIUM,
                        message=(
                            f"No lookup backend configured for {c.kind.value} citation "
                            f"{c.normalized!r} (jurisdiction={c.jurisdiction})."
                        ),
                        span=c.span,
                    )
                )
                continue
            result = backend.lookup(c)
            findings.extend(self._interpret(c, result))
        return findings

    def add_override(self, normalized_citation: str) -> None:
        self.overrides.add(normalized_citation)

    # ----- internals ----------------------------------------------------- #

    def _route(self, c: Citation) -> LookupBackend | None:
        for b in self.backends:
            try:
                if b.supports(c):
                    return b
            except Exception:  # noqa: BLE001
                continue
        return None

    def _interpret(self, c: Citation, result: dict | None) -> list[Finding]:
        if result is None:
            return [self._miss_finding(c)]
        if isinstance(result, dict) and result.get("_error"):
            c.status = CitationStatus.API_ERROR
            c.notes = result["_error"]
            return [
                Finding(
                    layer="api",
                    severity=Severity.MEDIUM,
                    message=f"Could not verify {c.normalized!r}: {result['_error']}",
                    span=c.span,
                )
            ]

        # Record canonical fields on the citation so downstream consumers
        # (quote attribution, field checks) can use them.
        canonical = (
            result.get("caseName")
            or result.get("case_name")
            or result.get("canonical_title")
            or ""
        )
        c.canonical_title = canonical or c.canonical_title
        c.opinion_id = (
            str(result.get("id")) if result.get("id") else c.opinion_id
        )
        if result.get("dateFiled"):
            try:
                c.canonical_year = int(str(result["dateFiled"])[:4])
            except ValueError:
                pass
        c.canonical_court = result.get("court") or result.get("canonical_court")
        judges_raw = result.get("judges") or result.get("canonical_judges")
        if judges_raw:
            c.canonical_judges = (
                judges_raw if isinstance(judges_raw, list) else _split_judges(str(judges_raw))
            )

        findings: list[Finding] = []

        # Case-name / canonical-title check.
        if c.case_name and c.canonical_title:
            sim = _name_similarity(c.case_name, c.canonical_title)
            if sim < self.name_similarity_threshold:
                c.status = CitationStatus.MISMATCH
                findings.append(
                    Finding(
                        layer="api",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Citation {c.normalized!r} resolves to "
                            f"{c.canonical_title!r}, but the model referred to it "
                            f"as {c.case_name!r}."
                        ),
                        evidence={"similarity": round(sim, 3)},
                        span=c.span,
                    )
                )

        # Year check (allow ±1 year for the filed-vs-decided gap).
        if c.year and c.canonical_year and abs(c.year - c.canonical_year) > 1:
            c.status = CitationStatus.YEAR_MISMATCH
            findings.append(
                Finding(
                    layer="api",
                    severity=Severity.HIGH,
                    message=(
                        f"Citation {c.normalized!r} has year {c.year} but the "
                        f"actual decision year is {c.canonical_year}."
                    ),
                    span=c.span,
                )
            )

        # Court check.
        if c.court and c.canonical_court:
            if c.court.lower() not in c.canonical_court.lower() and c.canonical_court.lower() not in c.court.lower():
                c.status = CitationStatus.COURT_MISMATCH
                findings.append(
                    Finding(
                        layer="api",
                        severity=Severity.HIGH,
                        message=(
                            f"Citation {c.normalized!r} attributes the case to "
                            f"{c.court!r}, but the record shows {c.canonical_court!r}."
                        ),
                        span=c.span,
                    )
                )

        # Judge check.
        if c.judge and c.canonical_judges:
            if not any(_name_similarity(c.judge, j) > 0.5 for j in c.canonical_judges):
                c.status = CitationStatus.JUDGE_MISMATCH
                findings.append(
                    Finding(
                        layer="api",
                        severity=Severity.HIGH,
                        message=(
                            f"Judge {c.judge!r} is not among the panel for "
                            f"{c.normalized!r} ({c.canonical_judges})."
                        ),
                        span=c.span,
                    )
                )

        if not findings:
            c.status = CitationStatus.VERIFIED
        return findings

    def _miss_finding(self, c: Citation) -> Finding:
        # If the claim references a recent date and no backend could find it,
        # soften to UNCHECKED_RECENT rather than declaring it fabricated.
        if c.year and _is_recent(c.year, self.recent_days):
            c.status = CitationStatus.UNCHECKED_RECENT
            return Finding(
                layer="api",
                severity=Severity.MEDIUM,
                message=(
                    f"Citation {c.normalized!r} not yet in database. Because the "
                    f"claim is recent ({c.year}), escalate to a human reviewer "
                    "rather than auto-blocking."
                ),
                span=c.span,
            )
        c.status = CitationStatus.NOT_FOUND
        extra = f" (claimed case: {c.case_name})" if c.case_name else ""
        return Finding(
            layer="api",
            severity=Severity.CRITICAL,
            message=f"Citation {c.normalized!r} not found in any configured database.{extra}",
            span=c.span,
        )


def _is_recent(year: int, days: int) -> bool:
    today = date.today()
    horizon = today - timedelta(days=days)
    # If the claimed year is >= horizon's year, treat it as "recent enough"
    # to warrant human review rather than outright rejection.
    return year >= horizon.year


def _split_judges(s: str) -> list[str]:
    return [p.strip() for p in re.split(r"[;,/]| and ", s) if p.strip()]


def _name_similarity(a: str, b: str) -> float:
    def norm(s: str) -> set[str]:
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        stop = {"v", "vs", "the", "of", "and", "inc", "llc", "corp", "co", "ltd", "plc"}
        return {w for w in s.split() if w and w not in stop}
    A, B = norm(a), norm(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
