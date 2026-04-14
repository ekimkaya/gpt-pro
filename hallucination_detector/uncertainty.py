"""Layer 4 — Uncertainty / confidence scoring.

Two signals:

* **Logprob-based confidence** — when the provider returns per-token
  logprobs (e.g. OpenAI), we compute the mean probability and flag
  low-confidence runs. Individual tokens with very low probability are
  surfaced as "suspicious spans" that a reviewer should highlight.

* **Semantic entropy** — sample the same prompt N times at moderate
  temperature. If the samples diverge heavily in their factual content
  (measured by citation / token overlap), the model is effectively
  guessing, which for a legal brief is the textbook hallucination regime.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .models import Finding, Severity
from .providers import LLMProvider


@dataclass
class ConfidenceScore:
    score: float  # 0.0 - 1.0
    method: str  # "logprobs" | "semantic_entropy" | "unknown"
    low_confidence_spans: list[tuple[str, float]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


def _mean_probability(logprobs: list[float]) -> float:
    if not logprobs:
        return 0.0
    return sum(math.exp(lp) for lp in logprobs) / len(logprobs)


def score_confidence(
    provider: LLMProvider,
    prompt: str,
    *,
    system: str | None = None,
    n_samples: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 512,
    low_token_threshold: float = 0.4,
    flag_threshold: float = 0.85,
) -> ConfidenceScore:
    """Return a single confidence score for ``prompt``.

    Strategy:
      1. If the provider returns logprobs, use mean token probability and
         flag individual tokens below ``low_token_threshold``.
      2. Otherwise, sample ``n_samples`` completions at ``temperature`` > 0
         and measure factual agreement between samples (semantic entropy
         proxy: 1 - avg Jaccard distance on content tokens and citations).
    """

    # Try logprobs first.
    try:
        resp = provider.complete(
            prompt,
            system=system,
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=True,
        )
    except TypeError:
        # Provider doesn't accept logprobs kwarg.
        resp = None

    if resp is not None and resp.token_logprobs:
        mean_p = _mean_probability(resp.token_logprobs)
        low = [
            (tok, math.exp(lp))
            for tok, lp in zip(resp.tokens, resp.token_logprobs)
            if math.exp(lp) < low_token_threshold
        ]
        findings: list[Finding] = []
        if mean_p < flag_threshold:
            findings.append(
                Finding(
                    layer="uncertainty",
                    severity=Severity.MEDIUM if mean_p >= 0.7 else Severity.HIGH,
                    message=(
                        f"Mean token confidence {mean_p:.2%} is below the "
                        f"{flag_threshold:.0%} threshold - model is guessing."
                    ),
                    evidence={"mean_probability": round(mean_p, 4), "low_token_count": len(low)},
                )
            )
        if low:
            findings.append(
                Finding(
                    layer="uncertainty",
                    severity=Severity.LOW,
                    message=(
                        f"{len(low)} token(s) below {low_token_threshold:.0%} probability "
                        "- highlight for reviewer."
                    ),
                    evidence={"tokens": low[:20]},
                )
            )
        return ConfidenceScore(
            score=mean_p,
            method="logprobs",
            low_confidence_spans=low,
            findings=findings,
        )

    # Fall back to semantic entropy sampling. Reuse the first response (if
    # any) to avoid wasting a provider call.
    samples: list[str] = []
    if resp is not None and resp.text:
        samples.append(resp.text)
    while len(samples) < max(2, n_samples):
        r = provider.complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        samples.append(r.text)

    agreement = _pairwise_agreement(samples)
    findings = []
    if agreement < flag_threshold:
        findings.append(
            Finding(
                layer="uncertainty",
                severity=Severity.MEDIUM if agreement >= 0.5 else Severity.HIGH,
                message=(
                    f"Samples agreed on only {agreement:.0%} of content - "
                    "model is unstable on this prompt."
                ),
                evidence={"n_samples": len(samples), "agreement": round(agreement, 3)},
            )
        )
    return ConfidenceScore(
        score=agreement,
        method="semantic_entropy",
        low_confidence_spans=[],
        findings=findings,
    )


_WORD_RE = re.compile(r"[A-Za-z0-9\.]+")
_CITE_RE = re.compile(r"\b\d+\s+[A-Z][A-Za-z\.\s]{1,20}?\s+\d+\b")


def _pairwise_agreement(samples: list[str]) -> float:
    """Average Jaccard similarity over content-word sets and extracted citations."""
    if len(samples) < 2:
        return 1.0
    feats = []
    for s in samples:
        words = {w.lower() for w in _WORD_RE.findall(s) if len(w) > 3}
        cites = set(_CITE_RE.findall(s))
        feats.append(words | {f"CITE::{c}" for c in cites})
    sims: list[float] = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            a, b = feats[i], feats[j]
            if not a or not b:
                continue
            sims.append(len(a & b) / len(a | b))
    return sum(sims) / len(sims) if sims else 0.0
