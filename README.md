# Hallucination Detector for Legal AI

A multi-layered verification tool for law firms that use LLMs to draft
filings, memos, or research. It's engineered to catch the failure mode that
cost the *Mata v. Avianca* attorneys their sanctions: fabricated case
citations slipping into a court filing.

The tool does not try to stop the LLM from hallucinating. It assumes it
will, and wraps the output in four independent verification layers so the
hallucination never reaches the court.

## The four layers

| Layer | Module | Role |
|---|---|---|
| 1. RAG | `hallucination_detector.rag` | Limits the model's answer to a retrieved set of authoritative documents and enforces `[doc_id]` citations. Any tag that doesn't resolve to a retrieved document is fabricated. |
| 2. Multi-agent | `hallucination_detector.multi_agent` | A "Judge" LLM (preferably from a different vendor) audits the "Creator" LLM's draft. Optional consensus voting across multiple models catches solo-citation hallucinations. |
| 3. API validation | `hallucination_detector.citation_validator` | Extracts every US reporter citation from the draft and verifies it against CourtListener (or your Westlaw/Lexis wrapper). 404s and case-name mismatches block the draft. |
| 4. Uncertainty | `hallucination_detector.uncertainty` | Uses per-token logprobs when available, or semantic-entropy sampling otherwise, to produce a confidence score. Low confidence = red-highlight for reviewer. |

## Quick start

```bash
pip install -r requirements.txt
python -m examples.legal_brief_check --offline   # no network; uses stubs
```

Online usage:

```python
from hallucination_detector import (
    HallucinationDetector, RAGIndex, Document,
    CitationValidator, CourtListenerClient,
)
from hallucination_detector.providers import AnthropicProvider, OpenAIProvider

index = RAGIndex([Document(doc_id="CASE-MIRANDA", title="...", text="...")])

detector = HallucinationDetector(
    creator=AnthropicProvider("claude-opus-4-6"),
    rag_index=index,
    judge=OpenAIProvider("gpt-4o"),
    consensus_providers=[AnthropicProvider(), OpenAIProvider()],
    citation_validator=CitationValidator(CourtListenerClient()),
)

report = detector.check("Draft one paragraph on Miranda warnings in custody.")
if report.blocked:
    escalate_to_human(report)
else:
    deliver_to_attorney(report.output, report.citations, report.confidence)
```

## Who this is for

Law firms and solo practitioners who are already using generative AI for
drafting but need a verification step before anything is filed, shared
with a client, or relied on in discovery. It is **not** a substitute for
a human attorney's review; it is a pre-filter that makes that review
tractable by flagging exactly which sentences need to be checked.

## Configuration

- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` — provider keys.
- `COURTLISTENER_TOKEN` — recommended for production (higher rate limits).
- `HallucinationDetector.block_at` — severity at which the output is
  automatically blocked. Defaults to `CRITICAL`.
- `HallucinationDetector.min_confidence` — confidence floor. Defaults to
  `0.6`.

## Tests

```bash
python -m unittest discover -s tests
```

All tests run offline using stub providers.
