"""End-to-end demo: run a question through all four hallucination layers.

This script is meant for law firm pilot deployments. It demonstrates the
"Mata v. Avianca" failure mode (fabricated case citations) and how the
detector would have caught it before filing.

Run with real providers:

    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    export COURTLISTENER_TOKEN=...   # strongly recommended
    python -m examples.legal_brief_check

Or run the offline demo (uses StubProvider, no network):

    python -m examples.legal_brief_check --offline
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the package importable when running the script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hallucination_detector import (  # noqa: E402
    CitationValidator,
    CourtListenerClient,
    Document,
    HallucinationDetector,
    RAGIndex,
)
from hallucination_detector.providers import (  # noqa: E402
    AnthropicProvider,
    OpenAIProvider,
    StubProvider,
)


LAW_FIRM_CORPUS = [
    Document(
        doc_id="CASE-ROE-1973",
        title="Roe v. Wade, 410 U.S. 113 (1973)",
        text=(
            "The Supreme Court held that the Due Process Clause of the Fourteenth "
            "Amendment protects a pregnant woman's liberty to choose to have an "
            "abortion without excessive government restriction."
        ),
        metadata={"jurisdiction": "US Supreme Court", "year": "1973"},
    ),
    Document(
        doc_id="CASE-MIRANDA-1966",
        title="Miranda v. Arizona, 384 U.S. 436 (1966)",
        text=(
            "Statements obtained from a defendant during custodial interrogation "
            "are inadmissible unless the prosecution demonstrates procedural "
            "safeguards effective to secure the Fifth Amendment privilege."
        ),
        metadata={"jurisdiction": "US Supreme Court", "year": "1966"},
    ),
    Document(
        doc_id="MEMO-2024-03",
        title="Internal memo: evidentiary standards for AI-generated filings",
        text=(
            "All court filings produced with AI assistance must be verified "
            "against CourtListener before submission. Any citation not resolvable "
            "by citation lookup is presumed fabricated."
        ),
        metadata={"client": "internal", "year": "2024"},
    ),
]


def run_online() -> None:
    index = RAGIndex(LAW_FIRM_CORPUS)
    creator = AnthropicProvider(model="claude-opus-4-6")
    judge = OpenAIProvider(model="gpt-4o")
    detector = HallucinationDetector(
        creator=creator,
        rag_index=index,
        judge=judge,
        consensus_providers=[creator, judge],
        citation_validator=CitationValidator(CourtListenerClient()),
    )
    question = (
        "In a brief for a motion to suppress, summarize the Miranda holding "
        "and cite any directly controlling Supreme Court authority."
    )
    report = detector.check(question)
    print(report.summary())
    print(json.dumps(report.as_dict(), indent=2))


def run_offline() -> None:
    """Simulates the 'Mata v. Avianca' failure and shows how each layer catches it."""

    index = RAGIndex(LAW_FIRM_CORPUS)

    # The "creator" model fabricates a case that isn't in the corpus.
    def creator_fn(prompt, system):
        if "Return the JSON verdict" in prompt:  # judge prompt - unused here
            return "{}"
        return (
            "The controlling authority is Varghese v. China Southern Airlines, "
            "925 F.3d 1339 (11th Cir. 2019), which held that equitable tolling "
            "applies to Montreal Convention claims [CASE-FAKE-2019]. "
            "See also Miranda v. Arizona, 384 U.S. 436 (1966) [CASE-MIRANDA-1966]."
        )

    # A "judge" model that notices the fabrication.
    def judge_fn(prompt, system):
        return json.dumps(
            {
                "verdict": "block",
                "issues": [
                    {
                        "claim": "Varghese v. China Southern Airlines, 925 F.3d 1339 (11th Cir. 2019)",
                        "problem": "No such case exists; this is a known AI hallucination from the Mata v. Avianca incident.",
                        "severity": "critical",
                    }
                ],
                "notes": "Block the draft. The non-Miranda citation is fabricated.",
            }
        )

    class FakeLookup:
        def lookup(self, citation):
            known = {
                "384 U.S. 436": {"caseName": "Miranda v. Arizona"},
                "410 U.S. 113": {"caseName": "Roe v. Wade"},
            }
            return known.get(citation.normalized)

    creator = StubProvider(creator_fn, model="creator-stub")
    judge = StubProvider(judge_fn, model="judge-stub")

    detector = HallucinationDetector(
        creator=creator,
        rag_index=index,
        judge=judge,
        consensus_providers=None,
        citation_validator=CitationValidator(FakeLookup()),
        score_uncertainty=False,  # stubs don't produce meaningful logprobs
    )
    question = (
        "Draft one paragraph arguing that equitable tolling applies to "
        "Montreal Convention claims; cite controlling authority."
    )
    report = detector.check(question)
    print(report.summary())
    print()
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run the stub demo with no network calls.")
    args = parser.parse_args()
    if args.offline:
        run_offline()
    else:
        run_online()
