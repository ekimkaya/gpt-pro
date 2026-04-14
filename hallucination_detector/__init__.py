"""Multi-layered hallucination detection and mitigation for high-stakes LLM output.

Layers:
    1. RAG with forced citations          (`rag`)
    2. Multi-agent judge + consensus      (`multi_agent`)
    3. Fact-checking via external APIs    (`citation_validator`)
    4. Uncertainty / logprob analysis     (`uncertainty`)

The :class:`HallucinationDetector` orchestrator in :mod:`detector` combines
all four into a single verification pipeline.
"""

from .models import (
    Citation,
    CitationStatus,
    DetectionReport,
    Finding,
    Severity,
)
from .rag import Document, RAGIndex, RAGResult, run_rag_with_citations
from .multi_agent import ConsensusResult, JudgeReport, consensus_vote, judge_review
from .citation_validator import (
    CitationValidator,
    CourtListenerClient,
    extract_legal_citations,
)
from .uncertainty import ConfidenceScore, score_confidence
from .detector import HallucinationDetector

__all__ = [
    "Citation",
    "CitationStatus",
    "ConfidenceScore",
    "ConsensusResult",
    "CourtListenerClient",
    "CitationValidator",
    "DetectionReport",
    "Document",
    "Finding",
    "HallucinationDetector",
    "JudgeReport",
    "RAGIndex",
    "RAGResult",
    "Severity",
    "consensus_vote",
    "extract_legal_citations",
    "judge_review",
    "run_rag_with_citations",
    "score_confidence",
]

__version__ = "0.1.0"
