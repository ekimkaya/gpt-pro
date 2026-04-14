"""Shared data models used across the four detection layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """How dangerous a finding is for downstream use."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CitationStatus(str, Enum):
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    MISMATCH = "mismatch"  # case name disagreement
    YEAR_MISMATCH = "year_mismatch"
    JUDGE_MISMATCH = "judge_mismatch"
    COURT_MISMATCH = "court_mismatch"
    QUOTE_MISMATCH = "quote_mismatch"
    UNCHECKED = "unchecked"
    UNCHECKED_RECENT = "unchecked_recent"  # missing, but claim is recent
    SEALED_OVERRIDE = "sealed_override"    # attorney-signed override
    API_ERROR = "api_error"


class CitationKind(str, Enum):
    CASE = "case"
    STATUTE = "statute"            # e.g. 42 U.S.C. § 1983
    REGULATION = "regulation"      # e.g. 29 C.F.R. § 1630.2
    STATE_STATUTE = "state_statute"
    FOREIGN_CASE = "foreign_case"  # UK, EU, CA, AU, etc.
    DOCKET = "docket"              # 1:23-cv-04456
    SECONDARY = "secondary"        # Restatement, treatise, law review
    SHORT_FORM = "short_form"      # id., supra - resolves to another citation


@dataclass
class Quote:
    """A quoted string found next to a citation in model output."""

    text: str
    start: int
    end: int
    attributed_citation: str | None = None  # normalized form of the cite


@dataclass
class Citation:
    """A legal citation extracted from model output.

    Fields are populated opportunistically by the extractor(s); different
    citation kinds use different subsets (e.g. statutes use ``title``,
    ``section``; foreign cases use ``jurisdiction``).
    """

    raw: str
    kind: CitationKind = CitationKind.CASE
    # Case-law fields
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    pincite: str | None = None
    case_name: str | None = None
    year: int | None = None
    court: str | None = None
    judge: str | None = None
    # Statute / regulation fields
    title: str | None = None
    section: str | None = None
    subsection: str | None = None
    # Foreign / docket fields
    jurisdiction: str | None = None
    neutral: str | None = None       # e.g. "2020 SCC 7"
    docket_number: str | None = None
    # Short-form linkage
    resolves_to: str | None = None   # normalized form of target citation
    # Verification results
    status: CitationStatus = CitationStatus.UNCHECKED
    canonical_title: str | None = None
    canonical_year: int | None = None
    canonical_judges: list[str] = field(default_factory=list)
    canonical_court: str | None = None
    opinion_id: str | None = None
    opinion_text: str | None = None  # cached full text, when available
    quotes: list[Quote] = field(default_factory=list)
    notes: str | None = None
    # Span in the source document the citation was extracted from.
    span: tuple[int, int] | None = None

    @property
    def normalized(self) -> str:
        if self.kind == CitationKind.CASE and self.volume and self.reporter and self.page:
            return f"{self.volume} {self.reporter} {self.page}"
        if self.kind == CitationKind.STATUTE and self.title and self.section:
            return f"{self.title} U.S.C. § {self.section}"
        if self.kind == CitationKind.REGULATION and self.title and self.section:
            return f"{self.title} C.F.R. § {self.section}"
        if self.kind == CitationKind.FOREIGN_CASE and self.neutral:
            return self.neutral
        if self.kind == CitationKind.DOCKET and self.docket_number:
            return self.docket_number
        return self.raw


@dataclass
class Finding:
    """A single suspicious element detected in the model's output."""

    layer: str  # "rag" | "judge" | "consensus" | "api" | "uncertainty"
    severity: Severity
    message: str
    span: tuple[int, int] | None = None  # char offsets into the output
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "severity": self.severity.value,
            "message": self.message,
            "span": list(self.span) if self.span else None,
            "evidence": self.evidence,
        }


@dataclass
class DetectionReport:
    """Aggregated output of the full detection pipeline."""

    output: str
    findings: list[Finding] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    confidence: float | None = None  # 0.0 - 1.0, None if not computed
    blocked: bool = False
    block_reason: str | None = None

    @property
    def max_severity(self) -> Severity:
        if not self.findings:
            return Severity.INFO
        order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return max(self.findings, key=lambda f: order.index(f.severity)).severity

    def summary(self) -> str:
        lines = [
            f"Hallucination report (blocked={self.blocked}, max_severity={self.max_severity.value}"
            + (f", confidence={self.confidence:.2f}" if self.confidence is not None else "")
            + ")"
        ]
        if self.block_reason:
            lines.append(f"  reason: {self.block_reason}")
        for f in self.findings:
            lines.append(f"  [{f.severity.value:>8}] {f.layer}: {f.message}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "confidence": self.confidence,
            "max_severity": self.max_severity.value,
            "findings": [f.as_dict() for f in self.findings],
            "citations": [
                {
                    "raw": c.raw,
                    "kind": c.kind.value,
                    "normalized": c.normalized,
                    "case_name": c.case_name,
                    "year": c.year,
                    "jurisdiction": c.jurisdiction,
                    "title": c.title,
                    "section": c.section,
                    "docket_number": c.docket_number,
                    "resolves_to": c.resolves_to,
                    "status": c.status.value,
                    "canonical_title": c.canonical_title,
                    "canonical_year": c.canonical_year,
                    "canonical_court": c.canonical_court,
                    "notes": c.notes,
                }
                for c in self.citations
            ],
        }
