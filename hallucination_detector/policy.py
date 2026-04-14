"""Firm policy: overrides, audit logging, and recent-citation handling.

Use :class:`FirmPolicy` to wire human judgment into the detector. Key
capabilities:

  * ``add_override(citation, attorney, reason)`` - an attorney signs off
    on a citation the detector can't auto-verify (sealed case, very
    recent opinion not yet in CourtListener). The override is logged and
    consumed by :class:`MultiJurisdictionValidator` via its ``overrides``
    set.

  * ``audit_log`` - every detection run writes a JSONL record with the
    question, the verdict, the findings, and the override ledger. This
    is what you hand to bar counsel if an AI-assisted filing is ever
    challenged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DetectionReport


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
    overrides: dict[str, Override] = field(default_factory=dict)

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
        # Append-only JSONL for tamper-evident logging; rotate externally.
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
