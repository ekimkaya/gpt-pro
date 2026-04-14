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
    MISMATCH = "mismatch"
    UNCHECKED = "unchecked"
    API_ERROR = "api_error"


@dataclass
class Citation:
    """A legal (or other) citation extracted from model output."""

    raw: str
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    case_name: str | None = None
    year: int | None = None
    status: CitationStatus = CitationStatus.UNCHECKED
    canonical_title: str | None = None
    notes: str | None = None

    @property
    def normalized(self) -> str:
        if self.volume and self.reporter and self.page:
            return f"{self.volume} {self.reporter} {self.page}"
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
                    "normalized": c.normalized,
                    "case_name": c.case_name,
                    "year": c.year,
                    "status": c.status.value,
                    "canonical_title": c.canonical_title,
                    "notes": c.notes,
                }
                for c in self.citations
            ],
        }
