"""Tests for the on-prem productization layer:
network policy reporter, OllamaProvider stub behavior, and training mode."""

from __future__ import annotations

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hallucination_detector import (  # noqa: E402
    HallucinationDetector,
    MultiJurisdictionValidator,
    RestatementCorpus,
    format_network_report,
    network_report,
)
from hallucination_detector.providers import OllamaProvider, StubProvider  # noqa: E402


class TestNetworkPolicy(unittest.TestCase):
    def test_offline_only_no_outbound(self):
        det = HallucinationDetector(
            creator=StubProvider(lambda p, s: "", model="x"),
            validator=MultiJurisdictionValidator(backends=[RestatementCorpus()]),
            score_uncertainty=False,
        )
        endpoints = network_report(det)
        # Nothing in the destinations list should send client data.
        self.assertFalse(any(e.sends_client_data for e in endpoints))
        # Restatement corpus is local-only.
        rc = [e for e in endpoints if "RestatementCorpus" in e.component]
        self.assertEqual(len(rc), 1)
        self.assertIn("local", rc[0].destination.lower())

    def test_format_report_signals_fully_offline(self):
        det = HallucinationDetector(
            creator=StubProvider(lambda p, s: "", model="x"),
            validator=MultiJurisdictionValidator(backends=[RestatementCorpus()]),
            score_uncertainty=False,
        )
        text = format_network_report(network_report(det))
        self.assertIn("validator-only", text.lower())

    def test_courtlistener_listed_when_present(self):
        from hallucination_detector import CourtListenerClient
        try:
            cl = CourtListenerClient()
        except ImportError:
            self.skipTest("requests not installed")
        det = HallucinationDetector(
            creator=StubProvider(lambda p, s: "", model="x"),
            validator=MultiJurisdictionValidator(backends=[cl]),
            opinion_fetcher=cl.fetch_opinion_text,
            score_uncertainty=False,
        )
        text = format_network_report(network_report(det))
        self.assertIn("courtlistener", text.lower())


class TestOllamaProviderStub(unittest.TestCase):
    """Doesn't talk to a real Ollama; just verifies construction + reachability."""

    def test_construction_and_unreachable_check(self):
        try:
            p = OllamaProvider(model="llama3.1:8b", base_url="http://localhost:1")
        except ImportError:
            self.skipTest("requests not installed")
        # Reachability returns False (port 1 is never reachable). Must not raise.
        self.assertFalse(OllamaProvider.is_available("http://localhost:1"))
        self.assertEqual(p.model, "llama3.1:8b")
        self.assertEqual(p.name, "ollama")


class TestTrainingMode(unittest.TestCase):
    def test_training_runs_offline_no_pause(self):
        from hallucination_detector.training import main as train_main

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            rc = train_main(["--offline", "--no-pause", "--draft", "1"])
        finally:
            sys.stdout = old
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("Training Mode", out)
        self.assertIn("DRAFT:", out)
        self.assertIn("DETECTOR REPORT:", out)


if __name__ == "__main__":
    unittest.main()
