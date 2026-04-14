"""Coverage for the extractors, resolver, quote checker, bluebook lint,
multi-jurisdiction validator, and policy module."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hallucination_detector import (  # noqa: E402
    Citation,
    CitationKind,
    CitationStatus,
    FirmPolicy,
    MultiJurisdictionValidator,
    QuoteAttributionChecker,
    RestatementCorpus,
    Severity,
    attribute_quotes,
    bluebook_lint,
    extract_all,
    extract_dockets,
    extract_foreign_cases,
    extract_statutes,
    extract_us_cases,
    resolve_short_forms,
)
from hallucination_detector.models import DetectionReport  # noqa: E402


class TestExtractors(unittest.TestCase):
    def test_usc_and_cfr(self):
        text = "See 42 U.S.C. § 1983 and also 29 C.F.R. § 1630.2(h)."
        cites = extract_statutes(text)
        kinds = [c.kind for c in cites]
        self.assertIn(CitationKind.STATUTE, kinds)
        self.assertIn(CitationKind.REGULATION, kinds)
        usc = next(c for c in cites if c.kind == CitationKind.STATUTE)
        self.assertEqual(usc.title, "42")
        self.assertEqual(usc.section, "1983")
        cfr = next(c for c in cites if c.kind == CitationKind.REGULATION)
        self.assertEqual(cfr.title, "29")
        self.assertEqual(cfr.section, "1630.2")
        self.assertEqual(cfr.subsection, "h")

    def test_foreign_uk(self):
        text = "See [2019] UKSC 4 and [2020] EWHC 1234 (Admin)."
        cites = extract_foreign_cases(text)
        self.assertEqual(len(cites), 2)
        uksc = cites[0]
        self.assertEqual(uksc.jurisdiction, "UK")
        self.assertEqual(uksc.year, 2019)
        self.assertEqual(uksc.court, "UKSC")

    def test_foreign_canada_and_australia(self):
        text = "2020 SCC 7 is persuasive; see also [2019] HCA 14."
        cites = extract_foreign_cases(text)
        juris = {c.jurisdiction for c in cites}
        self.assertIn("CA", juris)
        self.assertIn("AU", juris)

    def test_dockets(self):
        text = "This case, docketed as 1:23-cv-04456-XYZ, is ongoing."
        cites = extract_dockets(text)
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0].kind, CitationKind.DOCKET)
        self.assertIn("23-cv-04456", cites[0].docket_number)

    def test_extract_all_dedupes(self):
        text = "See Miranda v. Arizona, 384 U.S. 436 (1966) and 42 U.S.C. § 1983."
        cites = extract_all(text)
        kinds = [c.kind for c in cites]
        self.assertEqual(sorted(kinds), sorted([CitationKind.CASE, CitationKind.STATUTE]))

    def test_case_name_does_not_eat_prefix(self):
        text = "The controlling authority is Varghese v. China S. Airlines, 925 F.3d 1339 (11th Cir. 2019)."
        cites = extract_us_cases(text)
        self.assertEqual(len(cites), 1)
        # Case name should not include "The controlling authority is".
        self.assertNotIn("controlling authority", (cites[0].case_name or "").lower())
        self.assertIn("Varghese", cites[0].case_name or "")


class TestShortFormResolver(unittest.TestCase):
    def test_id_resolves_to_prior(self):
        text = "Miranda v. Arizona, 384 U.S. 436 (1966). Id. at 444."
        cites = extract_all(text)
        findings = resolve_short_forms(cites)
        self.assertFalse(findings)
        short = [c for c in cites if c.notes == "id"][0]
        self.assertEqual(short.resolves_to, "384 U.S. 436")

    def test_short_volume_catches_wrong_volume(self):
        text = (
            "Miranda v. Arizona, 384 U.S. 436 (1966). "
            "Miranda, 500 U.S. 600 at 602."
        )
        cites = extract_all(text)
        findings = resolve_short_forms(cites)
        # Volume/reporter mismatch -> CRITICAL.
        self.assertTrue(any(f.severity == Severity.CRITICAL for f in findings))

    def test_unresolvable_id(self):
        text = "Id. at 444."
        cites = extract_all(text)
        findings = resolve_short_forms(cites)
        self.assertTrue(any(f.layer == "resolver" for f in findings))


class TestQuoteAttribution(unittest.TestCase):
    def test_attribution_and_fake_quote(self):
        draft = (
            'The Court held, "Warnings must precede custodial interrogation." '
            "Miranda v. Arizona, 384 U.S. 436, 444 (1966)."
        )
        cites = extract_all(draft)
        # Mark as VERIFIED + give it an opinion body that does NOT contain the quote.
        cites[0].status = CitationStatus.VERIFIED
        cites[0].opinion_id = "123"
        attribute_quotes(draft, cites)
        self.assertEqual(len(cites[0].quotes), 1)

        def fetch(_id):
            return "The opinion body discusses custodial interrogation at length but never uses those exact words."

        checker = QuoteAttributionChecker(fetch_text=fetch, fuzzy_threshold=0.95)
        findings = checker.verify(cites)
        self.assertTrue(any(f.layer == "quote" and f.severity == Severity.CRITICAL for f in findings))
        self.assertEqual(cites[0].status, CitationStatus.QUOTE_MISMATCH)

    def test_quote_matches(self):
        draft = (
            'The Court held, "Warnings must precede custodial interrogation." '
            "Miranda v. Arizona, 384 U.S. 436, 444 (1966)."
        )
        cites = extract_all(draft)
        cites[0].status = CitationStatus.VERIFIED
        cites[0].opinion_id = "123"
        attribute_quotes(draft, cites)

        def fetch(_id):
            return "Before this court decides, warnings must precede custodial interrogation, the opinion continues..."

        checker = QuoteAttributionChecker(fetch_text=fetch)
        findings = checker.verify(cites)
        self.assertFalse([f for f in findings if f.severity == Severity.CRITICAL])


class TestBluebookLint(unittest.TestCase):
    def test_missing_year(self):
        text = "Miranda v. Arizona, 384 U.S. 436."  # no (YYYY)
        cites = extract_us_cases(text)
        findings = bluebook_lint(text, cites)
        self.assertTrue(any("year" in f.message.lower() for f in findings))


class TestMultiValidator(unittest.TestCase):
    def test_recent_citation_soft_fails(self):
        # Citation year = "now" and empty database -> UNCHECKED_RECENT, not NOT_FOUND.
        from datetime import date

        class EmptyBackend:
            def supports(self, c): return c.kind == CitationKind.CASE
            def lookup(self, c): return None

        c = Citation(
            raw="999 F.3d 1",
            kind=CitationKind.CASE,
            volume="999",
            reporter="F.3d",
            page="1",
            year=date.today().year,
        )
        v = MultiJurisdictionValidator(backends=[EmptyBackend()])
        findings = v.validate([c])
        self.assertEqual(c.status, CitationStatus.UNCHECKED_RECENT)
        # Medium severity, not critical.
        self.assertTrue(all(f.severity != Severity.CRITICAL for f in findings))

    def test_year_mismatch(self):
        class Backend:
            def supports(self, c): return c.kind == CitationKind.CASE
            def lookup(self, c):
                return {"caseName": "Miranda v. Arizona", "dateFiled": "1966-06-13"}

        c = Citation(
            raw="384 U.S. 436",
            kind=CitationKind.CASE,
            volume="384", reporter="U.S.", page="436",
            case_name="Miranda v. Arizona",
            year=1975,  # wrong
        )
        v = MultiJurisdictionValidator(backends=[Backend()])
        findings = v.validate([c])
        self.assertEqual(c.status, CitationStatus.YEAR_MISMATCH)
        self.assertTrue(any("year" in f.message.lower() for f in findings))

    def test_override_skips_check(self):
        class Backend:
            def supports(self, c): return c.kind == CitationKind.CASE
            def lookup(self, c): return None  # would otherwise fail

        c = Citation(
            raw="925 F.3d 1339",
            kind=CitationKind.CASE,
            volume="925", reporter="F.3d", page="1339",
            case_name="Sealed Case",
        )
        v = MultiJurisdictionValidator(backends=[Backend()])
        v.add_override("925 F.3d 1339")
        findings = v.validate([c])
        self.assertEqual(c.status, CitationStatus.SEALED_OVERRIDE)
        self.assertFalse(findings)


class TestRestatementCorpus(unittest.TestCase):
    def test_builtin_lookup(self):
        corpus = RestatementCorpus()
        c = Citation(
            raw="Restatement (Second) of Contracts § 90",
            kind=CitationKind.SECONDARY,
            title="Restatement (Second) of Contracts",
            section="90",
        )
        self.assertTrue(corpus.supports(c))
        self.assertIsNotNone(corpus.lookup(c))


class TestPolicy(unittest.TestCase):
    def test_override_and_audit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            policy = FirmPolicy(audit_path=path)
            policy.add_override("925 F.3d 1339", "J. Smith", "Sealed - JPM 2023")
            self.assertIn("925 F.3d 1339", policy.override_set())
            # Make sure the audit file has content.
            report = DetectionReport(output="draft", blocked=False)
            policy.record_run("q", report)
            with open(path) as f:
                lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
