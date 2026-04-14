"""Top-level orchestrator.

Pipeline, top to bottom:

  1. Draft generation — either via RAG over a firm corpus (if ``rag_index``
     is provided) or a plain ``creator.complete``.
  2. Citation extraction — every kind (US cases, USC/CFR, foreign, dockets,
     short forms, secondary sources).
  3. Short-form resolution — ``id.`` / ``supra`` / short-volume refs are
     pointed at their full citation; unresolvable refs become findings.
  4. Quote attribution — quoted strings near each citation are associated
     with it for the quote checker.
  5. Multi-jurisdiction validation — route every citation to its backend
     (CourtListener, USC, CFR, BAILII, EUR-Lex, CanLII, AustLII, RECAP,
     Restatement), compare name/year/court/judge against the canonical
     record.
  6. Quote verification — fetch the opinion text and confirm quoted
     language actually appears there.
  7. Bluebook linting — flag format errors that cluster around AI output.
  8. Judge LLM review — independent audit of the draft.
  9. Consensus voting — cross-vendor agreement on citations.
 10. Uncertainty scoring — logprobs or semantic entropy.
 11. Policy / block decision — apply firm overrides, severity threshold,
     confidence floor, and write the audit record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .backends.multi import MultiJurisdictionValidator
from .bluebook import lint as bluebook_lint
from .citation_extractors import extract_all
from .citation_resolver import resolve_short_forms
from .models import DetectionReport, Severity
from .multi_agent import consensus_vote, judge_review
from .policy import FirmPolicy
from .providers import LLMProvider
from .quote_checker import QuoteAttributionChecker, attribute_quotes
from .rag import RAGIndex, run_rag_with_citations
from .uncertainty import score_confidence


@dataclass
class HallucinationDetector:
    creator: LLMProvider
    rag_index: RAGIndex | None = None
    judge: LLMProvider | None = None
    consensus_providers: list[LLMProvider] | None = None
    validator: MultiJurisdictionValidator | None = None
    # Fetches opinion text by opinion_id for quote attribution. Usually
    # ``CourtListenerClient.fetch_opinion_text``.
    opinion_fetcher: Callable[[str], str | None] | None = None
    score_uncertainty: bool = True
    bluebook_lint: bool = True
    policy: FirmPolicy = field(default_factory=FirmPolicy)
    # Block at or above this severity.
    block_at: Severity = Severity.CRITICAL
    min_confidence: float = 0.6

    def check_draft(
        self,
        draft: str,
        *,
        use_api: bool = True,
        use_quotes: bool = True,
    ) -> DetectionReport:
        """Validator-only mode: run deterministic layers on an attorney's
        draft with no LLM calls.

        This is the no-privilege-risk mode. It runs citation extraction,
        short-form resolution, quote attribution, API-based citation
        validation, quote verification, and Bluebook lint on a pre-written
        draft. No prompts are sent to any generative model.

        Set ``use_api=False`` to also skip outbound API calls — pure
        offline static analysis (regex + short-form chain + Bluebook).
        """
        report = DetectionReport(output=draft)
        if not draft.strip():
            self._finalize(report, "(validator-only run)", {})
            return report

        citations = extract_all(draft)
        report.findings.extend(resolve_short_forms(citations))
        attribute_quotes(draft, citations)

        if use_api and self.validator is not None:
            self.validator.overrides.update(self.policy.override_set())
            report.findings.extend(self.validator.validate(citations))

        if use_api and use_quotes and self.opinion_fetcher is not None:
            q_checker = QuoteAttributionChecker(fetch_text=self.opinion_fetcher)
            report.findings.extend(q_checker.verify(citations))

        if self.bluebook_lint:
            report.findings.extend(bluebook_lint(draft, citations))

        report.citations = citations
        self._finalize(report, "(validator-only run)", {"mode": "validator_only"})
        return report

    def check(self, question: str, *, top_k: int = 5) -> DetectionReport:
        report = DetectionReport(output="")
        extras: dict[str, Any] = {}

        # --- 1. Draft generation ---------------------------------------- #
        if self.rag_index is not None:
            rag = run_rag_with_citations(
                question, self.rag_index, self.creator, top_k=top_k
            )
            report.output = rag.answer
            report.findings.extend(rag.findings)
            sources_text = _format_sources(rag.retrieved)
            extras["retrieved"] = [d.doc_id for d in rag.retrieved]
        else:
            resp = self.creator.complete(question, temperature=0.0, max_tokens=1024)
            report.output = resp.text
            sources_text = None

        if not report.output:
            self._finalize(report, question, extras)
            return report

        # --- 2-3. Extract + resolve short forms ------------------------- #
        citations = extract_all(report.output)
        report.findings.extend(resolve_short_forms(citations))

        # --- 4. Attribute quotes to citations --------------------------- #
        attribute_quotes(report.output, citations)

        # --- 5. API validation across every kind ------------------------ #
        if self.validator is not None:
            self.validator.overrides.update(self.policy.override_set())
            report.findings.extend(self.validator.validate(citations))

        # --- 6. Quote verification -------------------------------------- #
        if self.opinion_fetcher is not None:
            q_checker = QuoteAttributionChecker(fetch_text=self.opinion_fetcher)
            report.findings.extend(q_checker.verify(citations))

        # --- 7. Bluebook lint ------------------------------------------- #
        if self.bluebook_lint:
            report.findings.extend(bluebook_lint(report.output, citations))

        report.citations = citations

        # --- 8. Judge review -------------------------------------------- #
        if self.judge is not None:
            jr = judge_review(question, report.output, self.judge, sources=sources_text)
            report.findings.extend(jr.findings)
            if jr.verdict == "block":
                report.block_reason = f"Judge ({self.judge.model}) voted to block."

        # --- 9. Consensus ----------------------------------------------- #
        if self.consensus_providers and len(self.consensus_providers) >= 2:
            cr = consensus_vote(question, self.consensus_providers)
            report.findings.extend(cr.findings)

        # --- 10. Uncertainty -------------------------------------------- #
        if self.score_uncertainty:
            cs = score_confidence(self.creator, question)
            report.confidence = cs.score
            report.findings.extend(cs.findings)
            if cs.score < self.min_confidence:
                report.block_reason = (
                    f"Model confidence {cs.score:.0%} below minimum "
                    f"{self.min_confidence:.0%}."
                )

        # --- 11. Block policy + audit ----------------------------------- #
        self._finalize(report, question, extras)
        return report

    # ----- internal ------------------------------------------------------ #

    def _finalize(self, report: DetectionReport, question: str, extras: dict) -> None:
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
                n = sum(
                    1 for f in report.findings if order.index(f.severity) >= threshold_idx
                )
                report.block_reason = (
                    f"{n} finding(s) at or above {self.block_at.value} severity."
                )
        elif report.block_reason:
            report.blocked = True

        self.policy.record_run(question, report, extras=extras)


def _format_sources(docs) -> str:
    return "\n\n".join(f"[{d.doc_id}] {d.title}\n{d.text}" for d in docs)
