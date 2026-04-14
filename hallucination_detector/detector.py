"""The orchestrator that threads all four layers together.

Usage sketch (see ``examples/legal_brief_check.py`` for a runnable version):

    >>> detector = HallucinationDetector(
    ...     rag_index=index,
    ...     creator=AnthropicProvider(),
    ...     judge=OpenAIProvider(),
    ...     consensus_providers=[AnthropicProvider(), OpenAIProvider(), GeminiProvider()],
    ...     citation_validator=CitationValidator(CourtListenerClient()),
    ... )
    >>> report = detector.check("Summarize the holding in Roe v. Wade, 410 U.S. 113.")
    >>> if report.blocked:
    ...     escalate_to_human(report)

Every layer is optional - pass ``None`` to skip it. The detector always runs
the RAG + generation step (if ``rag_index`` is provided) or uses a plain
``creator.complete`` otherwise, and then applies whichever downstream layers
were configured.
"""

from __future__ import annotations

from dataclasses import dataclass

from .citation_validator import CitationValidator
from .models import DetectionReport, Finding, Severity
from .multi_agent import consensus_vote, judge_review
from .providers import LLMProvider
from .rag import RAGIndex, run_rag_with_citations
from .uncertainty import score_confidence


@dataclass
class HallucinationDetector:
    creator: LLMProvider
    rag_index: RAGIndex | None = None
    judge: LLMProvider | None = None
    consensus_providers: list[LLMProvider] | None = None
    citation_validator: CitationValidator | None = None
    score_uncertainty: bool = True
    # If the max finding severity reaches or exceeds this, block the output.
    block_at: Severity = Severity.CRITICAL
    # Minimum confidence below which we refuse to surface the draft.
    min_confidence: float = 0.6

    # ----- public API ----------------------------------------------------- #

    def check(self, question: str, *, top_k: int = 5) -> DetectionReport:
        report = DetectionReport(output="")

        # -- Layer 1: RAG with citations --------------------------------- #
        if self.rag_index is not None:
            rag = run_rag_with_citations(
                question, self.rag_index, self.creator, top_k=top_k
            )
            report.output = rag.answer
            report.findings.extend(rag.findings)
            sources_text = _format_sources(rag.retrieved)
        else:
            resp = self.creator.complete(question, temperature=0.0, max_tokens=1024)
            report.output = resp.text
            sources_text = None

        # -- Layer 3: citation API validation ---------------------------- #
        # Runs early so the judge can see which citations already failed.
        if self.citation_validator is not None and report.output:
            citations, findings = self.citation_validator.validate(report.output)
            report.citations = citations
            report.findings.extend(findings)

        # -- Layer 2a: judge review -------------------------------------- #
        if self.judge is not None and report.output:
            jr = judge_review(question, report.output, self.judge, sources=sources_text)
            report.findings.extend(jr.findings)
            if jr.verdict == "block":
                report.block_reason = f"Judge ({self.judge.model}) voted to block."

        # -- Layer 2b: consensus ----------------------------------------- #
        if self.consensus_providers and len(self.consensus_providers) >= 2:
            cr = consensus_vote(question, self.consensus_providers)
            report.findings.extend(cr.findings)

        # -- Layer 4: uncertainty / confidence --------------------------- #
        if self.score_uncertainty:
            cs = score_confidence(self.creator, question)
            report.confidence = cs.score
            report.findings.extend(cs.findings)
            if cs.score < self.min_confidence:
                report.block_reason = (
                    f"Model confidence {cs.score:.0%} below minimum "
                    f"{self.min_confidence:.0%}."
                )

        # -- Blocking decision ------------------------------------------- #
        self._apply_block_policy(report)
        return report

    # ----- internal ------------------------------------------------------- #

    def _apply_block_policy(self, report: DetectionReport) -> None:
        order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        threshold_idx = order.index(self.block_at)
        worst = report.max_severity
        if order.index(worst) >= threshold_idx:
            report.blocked = True
            if not report.block_reason:
                crits = [
                    f for f in report.findings if order.index(f.severity) >= threshold_idx
                ]
                report.block_reason = (
                    f"{len(crits)} finding(s) at or above {self.block_at.value} severity."
                )
        elif report.block_reason:
            # A soft block (e.g. low confidence) was requested by a layer.
            report.blocked = True


def _format_sources(docs) -> str:
    return "\n\n".join(
        f"[{d.doc_id}] {d.title}\n{d.text}" for d in docs
    )
