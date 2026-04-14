"""Multi-layered hallucination detection and mitigation for legal AI.

Layers:
    1. RAG with forced citations          (``rag``)
    2. Multi-agent judge + consensus      (``multi_agent``)
    3. Fact-checking via external APIs    (``backends``, ``citation_validator``)
    4. Uncertainty / logprob analysis     (``uncertainty``)

Additional modules:
    * ``citation_extractors`` - extract every recognized citation kind
    * ``citation_resolver``   - resolve ``id.``/``supra``/short forms
    * ``quote_checker``       - verify quoted language against opinion text
    * ``bluebook``            - format lint
    * ``policy``              - overrides, audit log

The :class:`HallucinationDetector` orchestrator in :mod:`detector` wires
all of the above into one pipeline.
"""

from .backends import (
    AustLIIClient,
    BAILIIClient,
    CFRClient,
    CanLIIClient,
    CourtListenerClient,
    EURLexClient,
    MultiJurisdictionValidator,
    RECAPClient,
    RestatementCorpus,
    USCodeClient,
)
from .bluebook import lint as bluebook_lint
from .citation_extractors import (
    extract_all,
    extract_dockets,
    extract_foreign_cases,
    extract_short_forms,
    extract_statutes,
    extract_us_cases,
)
from .citation_resolver import resolve_short_forms
from .citation_validator import (
    CitationValidator,
    extract_legal_citations,
)
from .detector import HallucinationDetector
from .models import (
    Citation,
    CitationKind,
    CitationStatus,
    DetectionReport,
    Finding,
    Quote,
    Severity,
)
from .multi_agent import (
    ConsensusResult,
    JudgeReport,
    consensus_vote,
    judge_review,
)
from .policy import AuditVerification, FirmPolicy, Override, verify_audit_log
from .network_policy import NetworkEndpoint, format_report as format_network_report, report as network_report
from .quote_checker import QuoteAttributionChecker, attribute_quotes
from .redactor import Redactor, RedactionMap, RedactingProvider
from .rag import Document, RAGIndex, RAGResult, run_rag_with_citations
from .uncertainty import ConfidenceScore, score_confidence

__all__ = [
    "AuditVerification",
    "AustLIIClient",
    "BAILIIClient",
    "CFRClient",
    "CanLIIClient",
    "Citation",
    "CitationKind",
    "CitationStatus",
    "CitationValidator",
    "ConfidenceScore",
    "ConsensusResult",
    "CourtListenerClient",
    "DetectionReport",
    "Document",
    "EURLexClient",
    "Finding",
    "FirmPolicy",
    "HallucinationDetector",
    "JudgeReport",
    "MultiJurisdictionValidator",
    "NetworkEndpoint",
    "Override",
    "Quote",
    "QuoteAttributionChecker",
    "RAGIndex",
    "RAGResult",
    "RECAPClient",
    "RedactingProvider",
    "RedactionMap",
    "Redactor",
    "RestatementCorpus",
    "Severity",
    "USCodeClient",
    "attribute_quotes",
    "bluebook_lint",
    "consensus_vote",
    "extract_all",
    "extract_dockets",
    "extract_foreign_cases",
    "extract_legal_citations",
    "extract_short_forms",
    "extract_statutes",
    "extract_us_cases",
    "format_network_report",
    "judge_review",
    "network_report",
    "resolve_short_forms",
    "run_rag_with_citations",
    "score_confidence",
    "verify_audit_log",
]

__version__ = "0.4.0"
