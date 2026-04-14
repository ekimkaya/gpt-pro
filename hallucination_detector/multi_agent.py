"""Layer 2 — Multi-agent review and consensus voting.

Two complementary techniques:

* :func:`judge_review` - a second LLM instance (ideally from a different
  vendor) audits the first model's legal brief for fabricated cases,
  wrong dates, misstated parties, and unsupported holdings.

* :func:`consensus_vote` - run the same question across N providers and
  flag any factual disagreement. In legal practice this is particularly
  useful for case names, citations, and dates: if Claude, GPT-4, and
  Gemini all independently return the same citation, it is *probably*
  real; if they diverge, require human review.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .models import Finding, Severity
from .providers import LLMProvider


_JUDGE_SYSTEM = """You are a senior litigation partner performing fact-verification
on a draft produced by a junior associate (an AI).

You will receive:
  * the ORIGINAL_QUESTION the associate was asked,
  * the associate's DRAFT response,
  * optionally, authoritative SOURCES that were available to the associate.

Your job is to flag every factual claim that is unsupported or likely false.
Pay special attention to:
  * Case citations (reporter volume, page, year) that do not match a real case.
  * Party names, judges, and docket numbers.
  * Holdings or quoted language that are not in SOURCES.
  * Statutes or regulations cited by incorrect section/title.
  * Dates that conflict with the known history of the matter.

Output STRICT JSON with this schema and nothing else:
{
  "verdict": "pass" | "revise" | "block",
  "issues": [
     {"claim": "<exact quote from draft>", "problem": "<why it's suspect>", "severity": "low"|"medium"|"high"|"critical"}
  ],
  "notes": "<short overall assessment>"
}
"""


@dataclass
class JudgeReport:
    verdict: str  # pass | revise | block
    issues: list[dict]
    notes: str
    findings: list[Finding] = field(default_factory=list)
    raw: str = ""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from ``text``; return {} on failure."""
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def judge_review(
    original_question: str,
    draft: str,
    judge: LLMProvider,
    *,
    sources: str | None = None,
    max_tokens: int = 1500,
) -> JudgeReport:
    """Ask a "Judge" LLM (ideally from a different vendor) to fact-check ``draft``."""

    parts = [f"ORIGINAL_QUESTION:\n{original_question}\n", f"DRAFT:\n{draft}\n"]
    if sources:
        parts.append(f"SOURCES:\n{sources}\n")
    parts.append("Return the JSON verdict now.")
    prompt = "\n".join(parts)

    resp = judge.complete(prompt, system=_JUDGE_SYSTEM, temperature=0.0, max_tokens=max_tokens)
    data = _extract_json(resp.text)

    verdict = data.get("verdict", "revise")
    issues = data.get("issues", []) or []
    notes = data.get("notes", "")

    sev_map = {
        "low": Severity.LOW,
        "medium": Severity.MEDIUM,
        "high": Severity.HIGH,
        "critical": Severity.CRITICAL,
    }
    findings: list[Finding] = []
    for issue in issues:
        sev = sev_map.get(str(issue.get("severity", "medium")).lower(), Severity.MEDIUM)
        findings.append(
            Finding(
                layer="judge",
                severity=sev,
                message=f"{issue.get('problem', 'unspecified')} | claim: {issue.get('claim', '')!r}",
                evidence={"judge_issue": issue, "judge_model": judge.model},
            )
        )

    if verdict == "block" and not findings:
        findings.append(
            Finding(
                layer="judge",
                severity=Severity.HIGH,
                message=f"Judge ({judge.model}) voted to block: {notes}",
            )
        )

    return JudgeReport(
        verdict=verdict, issues=issues, notes=notes, findings=findings, raw=resp.text
    )


@dataclass
class ConsensusResult:
    answers: dict[str, str]  # provider_name -> answer
    citation_agreement: dict[str, int]  # normalized citation -> vote count
    disputed_citations: list[str]
    findings: list[Finding]


# Matches "Smith v. Jones, 410 U.S. 113 (1973)" style citations loosely.
_CITATION_EXTRACT_RE = re.compile(
    r"\b(\d+)\s+([A-Z][A-Za-z0-9\.\s]{1,20}?)\s+(\d+)\b"
)


def _extract_cites(text: str) -> set[str]:
    out = set()
    for m in _CITATION_EXTRACT_RE.finditer(text):
        vol, rep, page = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip(), m.group(3)
        out.add(f"{vol} {rep} {page}")
    return out


def consensus_vote(
    question: str,
    providers: list[LLMProvider],
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    min_votes: int = 2,
) -> ConsensusResult:
    """Run ``question`` across multiple providers and flag factual disagreement.

    A citation appearing in only one provider's output (below ``min_votes``)
    is flagged as a probable hallucination.
    """

    answers: dict[str, str] = {}
    cites_by_provider: dict[str, set[str]] = {}
    for p in providers:
        resp = p.complete(question, system=system, temperature=0.0, max_tokens=max_tokens)
        key = f"{p.name}:{p.model}"
        answers[key] = resp.text
        cites_by_provider[key] = _extract_cites(resp.text)

    vote_counts: dict[str, int] = {}
    for cites in cites_by_provider.values():
        for c in cites:
            vote_counts[c] = vote_counts.get(c, 0) + 1

    disputed = sorted([c for c, v in vote_counts.items() if v < min_votes])

    findings: list[Finding] = []
    for c in disputed:
        who = [p for p, s in cites_by_provider.items() if c in s]
        findings.append(
            Finding(
                layer="consensus",
                severity=Severity.HIGH,
                message=(
                    f"Citation {c!r} appeared in only {len(who)} of {len(providers)} "
                    f"models ({who}); possible hallucination."
                ),
                evidence={"citation": c, "providers": who, "votes": vote_counts[c]},
            )
        )

    return ConsensusResult(
        answers=answers,
        citation_agreement=vote_counts,
        disputed_citations=disputed,
        findings=findings,
    )
