"""Command-line entry point for validator-only mode.

Run:

    python -m hallucination_detector path/to/draft.txt
    python -m hallucination_detector --offline path/to/draft.txt
    python -m hallucination_detector --json path/to/draft.txt > report.json
    cat draft.txt | python -m hallucination_detector -

Validator-only mode runs every deterministic layer (extraction,
short-form resolution, quote attribution, API validation, quote
verification, Bluebook lint) but makes **zero generative-model calls**,
which means **zero privilege risk from sending client matter to an LLM**.
This is the mode most firms should pilot first.

With ``--offline``, even API validation is skipped; the tool becomes pure
local static analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends.courtlistener import CourtListenerClient
from .backends.multi import MultiJurisdictionValidator
from .backends.restatement import RestatementCorpus
from .detector import HallucinationDetector
from .models import Severity
from .policy import FirmPolicy, verify_audit_log
from .providers import StubProvider


def _build_default_validator(offline: bool) -> MultiJurisdictionValidator:
    backends: list = [RestatementCorpus()]
    if not offline:
        try:
            backends.insert(0, CourtListenerClient())
        except ImportError:
            # requests not installed; offline-only anyway.
            pass
    return MultiJurisdictionValidator(backends=backends)


def _read_input(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text()


def _format_human(report, path: str) -> str:
    lines = [f"=== hallucination-detector ({path}) ==="]
    lines.append(report.summary())
    if report.citations:
        lines.append("")
        lines.append("Citations:")
        for c in report.citations:
            lines.append(f"  [{c.status.value:>16}] {c.kind.value:<13} {c.normalized}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hallucination-detector",
        description=(
            "Validator-only hallucination detection for legal drafts. Runs "
            "every deterministic layer (no LLM calls) against a draft file "
            "or stdin."
        ),
    )
    parser.add_argument(
        "draft",
        nargs="?",
        help="Path to a draft text file, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip all outbound API calls - pure local static analysis.",
    )
    parser.add_argument(
        "--no-quotes",
        action="store_true",
        help="Skip quote attribution / opinion-body fetch.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON instead of a human summary.",
    )
    parser.add_argument(
        "--audit",
        metavar="PATH",
        help="Append an HMAC-signed audit entry to this JSONL file.",
    )
    parser.add_argument(
        "--audit-key-env",
        default="FIRM_AUDIT_HMAC_KEY",
        help="Env var holding hex-encoded HMAC key (default: FIRM_AUDIT_HMAC_KEY).",
    )
    parser.add_argument(
        "--verify-audit",
        metavar="PATH",
        help=(
            "Instead of running detection, verify the HMAC chain of this "
            "audit log file and exit."
        ),
    )
    args = parser.parse_args(argv)

    # --- Verify-only branch -------------------------------------------- #
    if args.verify_audit:
        key = _load_hmac_key(args.audit_key_env)
        if key is None:
            print(f"error: {args.audit_key_env} is not set", file=sys.stderr)
            return 2
        result = verify_audit_log(args.verify_audit, key)
        if result.ok:
            print(f"audit OK ({result.lines_checked} signed records)")
            return 0
        print(
            f"audit FAILED: line {result.first_bad_line}: {result.reason} "
            f"({result.lines_checked} records checked)",
            file=sys.stderr,
        )
        return 1

    # --- Normal validation run ---------------------------------------- #
    if not args.draft:
        parser.error("draft path (or '-' for stdin) is required unless --verify-audit")
    draft = _read_input(args.draft)
    validator = _build_default_validator(args.offline)
    opinion_fetcher = None
    if not args.offline and not args.no_quotes:
        for b in validator.backends:
            if hasattr(b, "fetch_opinion_text"):
                opinion_fetcher = b.fetch_opinion_text
                break

    policy = FirmPolicy()
    if args.audit:
        key = _load_hmac_key(args.audit_key_env)
        if key is None:
            print(
                f"warning: {args.audit_key_env} not set; writing unsigned audit.",
                file=sys.stderr,
            )
        policy = FirmPolicy(audit_path=args.audit, audit_hmac_key=key)

    # StubProvider satisfies the orchestrator's creator slot but is never
    # called in check_draft(); validator-only mode makes zero LLM calls.
    detector = HallucinationDetector(
        creator=StubProvider(lambda p, s: "", model="unused"),
        validator=validator,
        opinion_fetcher=opinion_fetcher,
        policy=policy,
        score_uncertainty=False,
    )
    report = detector.check_draft(
        draft,
        use_api=not args.offline,
        use_quotes=not args.no_quotes,
    )

    if args.json:
        json.dump(report.as_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(_format_human(report, args.draft))

    # Exit code: 0 clean, 1 findings below CRITICAL, 2 blocked.
    if report.blocked:
        return 2
    if report.max_severity in (Severity.HIGH, Severity.CRITICAL):
        return 1
    return 0


def _load_hmac_key(env_var: str) -> bytes | None:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    # Accept hex or plain-text keys.
    try:
        return bytes.fromhex(raw)
    except ValueError:
        return raw.encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
