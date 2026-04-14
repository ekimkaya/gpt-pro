"""Tests for the redactor, signed audit log, and validator-only CLI."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hallucination_detector import (  # noqa: E402
    FirmPolicy,
    Redactor,
    RedactingProvider,
    verify_audit_log,
)
from hallucination_detector.cli import main as cli_main  # noqa: E402
from hallucination_detector.models import DetectionReport  # noqa: E402
from hallucination_detector.providers import StubProvider  # noqa: E402


# --------------------------------------------------------------------------- #
# Redactor
# --------------------------------------------------------------------------- #


class TestRedactor(unittest.TestCase):
    def test_preserves_legal_citations(self):
        r = Redactor()
        text = (
            "On behalf of Mr. John Smith (SSN 123-45-6789), we cite "
            "Miranda v. Arizona, 384 U.S. 436 (1966)."
        )
        redacted, rmap = r.redact(text)
        self.assertIn("384 U.S. 436", redacted)
        self.assertIn("Miranda v. Arizona", redacted)
        self.assertNotIn("123-45-6789", redacted)
        self.assertNotIn("John Smith", redacted)

    def test_restore_roundtrip(self):
        r = Redactor(client_names={"Acme Corp."})
        text = "Our client Acme Corp. was sued at 555-123-4567."
        redacted, rmap = r.redact(text)
        self.assertNotIn("Acme", redacted)
        self.assertNotIn("555-123-4567", redacted)
        restored = r.restore(redacted, rmap)
        self.assertEqual(restored, text)

    def test_matter_number_custom_pattern(self):
        r = Redactor(matter_number_patterns=[r"MATTER-\d{6}"])
        text = "Re: MATTER-042195 and unrelated text."
        redacted, rmap = r.redact(text)
        self.assertNotIn("MATTER-042195", redacted)
        self.assertTrue(any("MATTER-042195" == v for v in rmap.forward.values()))

    def test_redacting_provider_roundtrip(self):
        r = Redactor(client_names={"Acme Corp."})
        # Inner provider echoes whatever it sees in the prompt after "echo:".
        def echo(prompt, system):
            start = prompt.find("echo:") + len("echo:")
            return prompt[start:].strip()
        logged = []
        wrapped = RedactingProvider(
            inner=StubProvider(echo, model="stub"),
            redactor=r,
            redacted_log=logged.append,
        )
        prompt = "echo: Acme Corp. filed suit on 2024-01-15 per Miranda v. Arizona, 384 U.S. 436 (1966)."
        resp = wrapped.complete(prompt)
        # The inner provider must not have seen "Acme Corp." or the date.
        seen = logged[0]["prompt"]
        self.assertNotIn("Acme Corp.", seen)
        self.assertNotIn("2024-01-15", seen)
        # But citation passed through unredacted.
        self.assertIn("384 U.S. 436", seen)
        # Restored response contains the originals back.
        self.assertIn("Acme Corp.", resp.text)
        self.assertIn("2024-01-15", resp.text)


# --------------------------------------------------------------------------- #
# Signed audit log
# --------------------------------------------------------------------------- #


class TestSignedAuditLog(unittest.TestCase):
    def _policy(self, path, key=b"secret-key"):
        return FirmPolicy(audit_path=path, audit_hmac_key=key)

    def test_chain_verifies_clean(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            policy = self._policy(path)
            policy.add_override("384 U.S. 436", "J. Smith", "sealed matter")
            report = DetectionReport(output="x")
            policy.record_run("q1", report)
            policy.record_run("q2", report)
            v = verify_audit_log(path, b"secret-key")
            self.assertTrue(v.ok, v.reason)
            self.assertEqual(v.lines_checked, 3)

    def test_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            policy = self._policy(path)
            policy.record_run("q1", DetectionReport(output="x"))
            policy.record_run("q2", DetectionReport(output="y"))
            # Tamper with the first line's record payload.
            with open(path) as f:
                lines = f.readlines()
            obj = json.loads(lines[0])
            obj["record"]["question"] = "q1-MODIFIED"
            lines[0] = json.dumps(obj) + "\n"
            with open(path, "w") as f:
                f.writelines(lines)
            v = verify_audit_log(path, b"secret-key")
            self.assertFalse(v.ok)
            self.assertEqual(v.first_bad_line, 1)

    def test_wrong_key_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            policy = self._policy(path)
            policy.record_run("q", DetectionReport(output="x"))
            v = verify_audit_log(path, b"wrong-key")
            self.assertFalse(v.ok)

    def test_chain_survives_restart(self):
        # If the process restarts, we load the last sig from disk and keep chaining.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            self._policy(path).record_run("q1", DetectionReport(output="x"))
            self._policy(path).record_run("q2", DetectionReport(output="y"))
            v = verify_audit_log(path, b"secret-key")
            self.assertTrue(v.ok)
            self.assertEqual(v.lines_checked, 2)


# --------------------------------------------------------------------------- #
# Validator-only CLI
# --------------------------------------------------------------------------- #


class TestCLI(unittest.TestCase):
    def test_offline_run_flags_ungrounded_short_form(self):
        draft = "The Court held as much. Id. at 115."
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "draft.txt")
            with open(path, "w") as f:
                f.write(draft)
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                rc = cli_main([path, "--offline"])
            finally:
                sys.stdout = old
            out = buf.getvalue()
            # Unresolvable Id. should produce a finding.
            self.assertIn("resolver", out.lower())
            # Exit code 1 or 2 means findings/blocked.
            self.assertIn(rc, (1, 2))

    def test_json_output_is_valid(self):
        draft = "Miranda v. Arizona, 384 U.S. 436 (1966). Id. at 444."
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "draft.txt")
            with open(path, "w") as f:
                f.write(draft)
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                cli_main([path, "--offline", "--json"])
            finally:
                sys.stdout = old
            data = json.loads(buf.getvalue())
            self.assertIn("citations", data)
            self.assertIn("findings", data)


if __name__ == "__main__":
    unittest.main()
