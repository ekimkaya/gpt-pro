"""Firm policy: overrides, audit logging, and recent-citation handling.

Use :class:`FirmPolicy` to wire human judgment into the detector. Key
capabilities:

  * ``add_override(citation, attorney, reason)`` - an attorney signs off
    on a citation the detector can't auto-verify (sealed case, very
    recent opinion not yet in CourtListener). The override is logged and
    consumed by :class:`MultiJurisdictionValidator` via its ``overrides``
    set.

  * ``audit_log`` - every detection run writes a JSONL record. When
    ``audit_hmac_key`` is provided, each record is signed using a
    chained HMAC (``sig_n = HMAC(key, sig_{n-1} || record)``) so any
    after-the-fact tampering is detectable. Use :func:`verify_audit_log`
    to validate the chain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DetectionReport


_GENESIS_SIG = "0" * 64  # first record chains off of this constant


@dataclass
class Override:
    citation: str       # normalized form
    attorney: str
    reason: str
    timestamp: str


@dataclass
class FirmPolicy:
    """Policy state the orchestrator consults while running."""

    audit_path: str | None = None
    audit_hmac_key: bytes | None = None
    overrides: dict[str, Override] = field(default_factory=dict)
    # Tracks the last chained signature; initialized from disk on first write.
    _last_sig: str | None = None

    def add_override(
        self, citation: str, attorney: str, reason: str
    ) -> Override:
        o = Override(
            citation=citation,
            attorney=attorney,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.overrides[citation] = o
        self._append_audit(
            {"type": "override_added", "override": o.__dict__}
        )
        return o

    def remove_override(self, citation: str) -> None:
        o = self.overrides.pop(citation, None)
        if o:
            self._append_audit({"type": "override_removed", "citation": citation})

    def override_set(self) -> set[str]:
        return set(self.overrides.keys())

    def record_run(
        self,
        question: str,
        report: DetectionReport,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self._append_audit(
            {
                "type": "run",
                "question": question,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "report": report.as_dict(),
                "overrides": {k: v.__dict__ for k, v in self.overrides.items()},
                "extras": extras or {},
            }
        )

    # ----- internals ----------------------------------------------------- #

    def _append_audit(self, record: dict[str, Any]) -> None:
        if not self.audit_path:
            return
        path = Path(self.audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.audit_hmac_key is not None:
            prev_sig = self._last_sig or _load_last_sig(path) or _GENESIS_SIG
            payload = json.dumps(record, default=str, sort_keys=True)
            sig = _chain_sig(self.audit_hmac_key, prev_sig, payload)
            signed = {"record": record, "prev_sig": prev_sig, "sig": sig}
            line = json.dumps(signed, default=str)
            self._last_sig = sig
        else:
            line = json.dumps(record, default=str)

        # Append-only JSONL for tamper-evident logging; rotate externally.
        with path.open("a") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def _chain_sig(key: bytes, prev_sig: str, payload: str) -> str:
    mac = hmac.new(key, digestmod=hashlib.sha256)
    mac.update(prev_sig.encode("utf-8"))
    mac.update(b"\x1e")  # record separator
    mac.update(payload.encode("utf-8"))
    return mac.hexdigest()


def _load_last_sig(path: Path) -> str | None:
    if not path.exists():
        return None
    last = None
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return None
            if isinstance(obj, dict) and "sig" in obj:
                last = obj["sig"]
    return last


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


@dataclass
class AuditVerification:
    ok: bool
    lines_checked: int
    first_bad_line: int | None = None
    reason: str | None = None


def verify_audit_log(path: str | Path, key: bytes) -> AuditVerification:
    """Walk the chain from genesis and confirm every signature.

    Returns the 1-indexed line number of the first tampered record, or
    ``ok=True`` if the whole file validates.
    """
    p = Path(path)
    prev = _GENESIS_SIG
    n = 0
    if not p.exists():
        return AuditVerification(ok=False, lines_checked=0, reason="file missing")
    with p.open("rb") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            n += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                return AuditVerification(
                    ok=False, lines_checked=n, first_bad_line=i,
                    reason="invalid json",
                )
            if not (isinstance(obj, dict) and "record" in obj and "sig" in obj):
                return AuditVerification(
                    ok=False, lines_checked=n, first_bad_line=i,
                    reason="unsigned record in signed log",
                )
            if obj.get("prev_sig") != prev:
                return AuditVerification(
                    ok=False, lines_checked=n, first_bad_line=i,
                    reason="chain break: prev_sig does not match last signature",
                )
            payload = json.dumps(obj["record"], default=str, sort_keys=True)
            expected = _chain_sig(key, prev, payload)
            if not hmac.compare_digest(expected, obj["sig"]):
                return AuditVerification(
                    ok=False, lines_checked=n, first_bad_line=i,
                    reason="hmac mismatch: record has been tampered",
                )
            prev = obj["sig"]
    return AuditVerification(ok=True, lines_checked=n)
