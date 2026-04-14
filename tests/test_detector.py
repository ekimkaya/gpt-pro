"""Tests for the hallucination detector. Run with: python -m unittest discover -s tests"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hallucination_detector import (  # noqa: E402
    CitationStatus,
    CitationValidator,
    Document,
    HallucinationDetector,
    RAGIndex,
    Severity,
    extract_legal_citations,
)
from hallucination_detector.multi_agent import consensus_vote, judge_review  # noqa: E402
from hallucination_detector.providers import StubProvider  # noqa: E402
from hallucination_detector.rag import run_rag_with_citations  # noqa: E402
from hallucination_detector.uncertainty import score_confidence  # noqa: E402


CORPUS = [
    Document(
        doc_id="CASE-MIRANDA",
        title="Miranda v. Arizona, 384 U.S. 436 (1966)",
        text="Custodial interrogation requires warnings to preserve Fifth Amendment rights.",
    ),
    Document(
        doc_id="CASE-ROE",
        title="Roe v. Wade, 410 U.S. 113 (1973)",
        text="The Due Process Clause protects certain privacy interests.",
    ),
]


class TestCitationExtraction(unittest.TestCase):
    def test_extracts_us_reporter_with_case_name(self):
        txt = "See Miranda v. Arizona, 384 U.S. 436 (1966) for the rule."
        cites = extract_legal_citations(txt)
        self.assertEqual(len(cites), 1)
        c = cites[0]
        self.assertEqual(c.volume, "384")
        self.assertEqual(c.reporter, "U.S.")
        self.assertEqual(c.page, "436")
        self.assertEqual(c.year, 1966)
        self.assertIn("Miranda", c.case_name or "")

    def test_extracts_multiple_and_federal(self):
        txt = (
            "See 410 U.S. 113 and also Varghese v. China S. Airlines, "
            "925 F.3d 1339 (11th Cir. 2019)."
        )
        cites = extract_legal_citations(txt)
        self.assertGreaterEqual(len(cites), 2)
        norms = {c.normalized for c in cites}
        self.assertIn("410 U.S. 113", norms)
        self.assertIn("925 F.3d 1339", norms)


class FakeLookup:
    def __init__(self, db):
        self.db = db

    def lookup(self, citation):
        return self.db.get(citation.normalized)


class TestCitationValidator(unittest.TestCase):
    def test_flags_missing_citation(self):
        validator = CitationValidator(FakeLookup({"410 U.S. 113": {"caseName": "Roe v. Wade"}}))
        cites, findings = validator.validate(
            "See Varghese v. China Southern, 925 F.3d 1339 (11th Cir. 2019)."
        )
        self.assertTrue(any(f.severity == Severity.CRITICAL for f in findings))
        self.assertTrue(any(c.status == CitationStatus.NOT_FOUND for c in cites))

    def test_flags_name_mismatch(self):
        validator = CitationValidator(FakeLookup({"410 U.S. 113": {"caseName": "Roe v. Wade"}}))
        cites, findings = validator.validate(
            "Smith v. Jones, 410 U.S. 113 (1973), is controlling."
        )
        self.assertTrue(any(f.severity == Severity.CRITICAL for f in findings))
        self.assertTrue(any(c.status == CitationStatus.MISMATCH for c in cites))

    def test_passes_verified(self):
        validator = CitationValidator(FakeLookup({"384 U.S. 436": {"caseName": "Miranda v. Arizona"}}))
        cites, findings = validator.validate("Miranda v. Arizona, 384 U.S. 436 (1966).")
        self.assertFalse(findings)
        self.assertEqual(cites[0].status, CitationStatus.VERIFIED)


class TestRAG(unittest.TestCase):
    def test_flags_ungrounded_doc_id(self):
        index = RAGIndex(CORPUS)
        # Creator fabricates a doc id that wasn't retrieved.
        provider = StubProvider(
            lambda p, s: "Miranda requires warnings [CASE-MIRANDA]. Another rule [CASE-GHOST].",
            model="stub",
        )
        result = run_rag_with_citations("What did Miranda hold?", index, provider)
        self.assertIn("CASE-GHOST", result.ungrounded_citations)
        self.assertTrue(any(f.severity == Severity.CRITICAL for f in result.findings))

    def test_flags_no_citations(self):
        index = RAGIndex(CORPUS)
        provider = StubProvider(
            lambda p, s: "The rule is that warnings are required in custody.", model="stub"
        )
        result = run_rag_with_citations("What did Miranda hold?", index, provider)
        self.assertTrue(any(f.layer == "rag" for f in result.findings))


class TestJudge(unittest.TestCase):
    def test_judge_flags_issues(self):
        judge = StubProvider(
            lambda p, s: json.dumps(
                {
                    "verdict": "block",
                    "issues": [
                        {"claim": "Fake Case", "problem": "does not exist", "severity": "critical"}
                    ],
                    "notes": "blocked",
                }
            ),
            model="judge",
        )
        report = judge_review("Q?", "Draft citing Fake Case.", judge)
        self.assertEqual(report.verdict, "block")
        self.assertTrue(any(f.severity == Severity.CRITICAL for f in report.findings))

    def test_judge_handles_malformed_json(self):
        judge = StubProvider(lambda p, s: "not json at all", model="judge")
        report = judge_review("Q?", "Draft.", judge)
        self.assertEqual(report.verdict, "revise")  # default fallback
        self.assertEqual(report.issues, [])


class TestConsensus(unittest.TestCase):
    def test_flags_solo_citation(self):
        p1 = StubProvider(lambda p, s: "Miranda v. Arizona, 384 U.S. 436.", model="a")
        p2 = StubProvider(lambda p, s: "Miranda v. Arizona, 384 U.S. 436.", model="b")
        p3 = StubProvider(lambda p, s: "See also Fake v. Case, 999 F.3d 111.", model="c")
        result = consensus_vote("Cite authority.", [p1, p2, p3])
        self.assertIn("999 F.3d 111", result.disputed_citations)
        self.assertNotIn("384 U.S. 436", result.disputed_citations)


class TestUncertaintySemantic(unittest.TestCase):
    def test_semantic_entropy_flags_inconsistent(self):
        outputs = iter(
            [
                "The case is 100 F.3d 200 from 1995.",
                "The case is 500 U.S. 900 from 2001.",
                "The case is 12 A.2d 34 from 1988.",
            ]
        )
        provider = StubProvider(lambda p, s: next(outputs), model="stub")
        cs = score_confidence(provider, "Cite the authority.", n_samples=3)
        self.assertEqual(cs.method, "semantic_entropy")
        self.assertLess(cs.score, 0.85)
        self.assertTrue(cs.findings)

    def test_semantic_entropy_high_agreement(self):
        outputs = iter(
            [
                "The controlling case is Miranda v. Arizona, 384 U.S. 436.",
                "The controlling case is Miranda v. Arizona, 384 U.S. 436.",
                "The controlling case is Miranda v. Arizona, 384 U.S. 436.",
            ]
        )
        provider = StubProvider(lambda p, s: next(outputs), model="stub")
        cs = score_confidence(provider, "Cite the authority.", n_samples=3)
        self.assertGreaterEqual(cs.score, 0.9)


class TestOrchestrator(unittest.TestCase):
    def test_blocks_on_fabricated_citation(self):
        index = RAGIndex(CORPUS)

        creator = StubProvider(
            lambda p, s: (
                "The controlling case is Varghese v. China S. Airlines, "
                "925 F.3d 1339 (11th Cir. 2019) [CASE-GHOST]."
            ),
            model="creator",
        )
        judge = StubProvider(
            lambda p, s: json.dumps(
                {
                    "verdict": "block",
                    "issues": [
                        {"claim": "Varghese", "problem": "fabricated", "severity": "critical"}
                    ],
                    "notes": "block",
                }
            ),
            model="judge",
        )
        validator = CitationValidator(FakeLookup({}))  # empty DB -> all missing
        detector = HallucinationDetector(
            creator=creator,
            rag_index=index,
            judge=judge,
            citation_validator=validator,
            score_uncertainty=False,
        )
        # Query overlaps with corpus so RAG retrieves docs and the creator
        # gets a chance to produce a draft that later layers can evaluate.
        report = detector.check("What are the Miranda custodial interrogation rules?")
        self.assertTrue(report.blocked)
        self.assertEqual(report.max_severity, Severity.CRITICAL)
        # Report should mention the ghost citation and missing API lookup.
        layers = {f.layer for f in report.findings}
        self.assertIn("rag", layers)
        self.assertIn("api", layers)
        self.assertIn("judge", layers)


if __name__ == "__main__":
    unittest.main()
