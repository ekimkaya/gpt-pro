"""Resolve short-form references (``id.``, ``supra``, ``*Party*, 925 F.3d at 1342``)
to the full citation they point at.

Bluebook briefs rarely repeat full citations; short forms dominate. The
resolver walks the document in order, keeps a stack of recent full cites,
and rewrites each :class:`CitationKind.SHORT_FORM` in place to carry the
target's volume/reporter/page plus a ``resolves_to`` pointer. Unresolvable
short forms are flagged as findings — that's itself a hallucination signal
(or at minimum a drafting error).
"""

from __future__ import annotations

from .models import Citation, CitationKind, Finding, Severity


def resolve_short_forms(citations: list[Citation]) -> list[Finding]:
    """Mutates ``citations`` in place; returns any unresolvable findings.

    Assumes ``citations`` are sorted by their position in the source text.
    """
    findings: list[Finding] = []
    last_full: Citation | None = None
    party_index: dict[str, Citation] = {}

    for c in citations:
        if c.kind in (
            CitationKind.CASE,
            CitationKind.FOREIGN_CASE,
            CitationKind.STATUTE,
            CitationKind.REGULATION,
            CitationKind.DOCKET,
        ):
            last_full = c
            if c.case_name:
                first_party = c.case_name.split(" v")[0].strip()
                party_index[first_party.lower()] = c
            continue

        if c.kind != CitationKind.SHORT_FORM:
            continue

        target: Citation | None = None
        if c.notes == "id":
            target = last_full
        elif c.notes in ("supra", "short_volume") and c.case_name:
            target = party_index.get(c.case_name.split(" v")[0].strip().lower())
            # For short_volume, also verify the volume/reporter match.
            if target and c.notes == "short_volume":
                if (
                    c.volume
                    and c.reporter
                    and target.volume
                    and target.reporter
                    and (c.volume != target.volume or c.reporter != target.reporter)
                ):
                    findings.append(
                        Finding(
                            layer="resolver",
                            severity=Severity.CRITICAL,
                            message=(
                                f"Short-form reference {c.raw!r} cites "
                                f"{c.volume} {c.reporter} but the prior full "
                                f"citation for {c.case_name!r} was "
                                f"{target.volume} {target.reporter}."
                            ),
                            evidence={
                                "short_form": c.raw,
                                "claimed": f"{c.volume} {c.reporter}",
                                "actual": f"{target.volume} {target.reporter}",
                            },
                            span=c.span,
                        )
                    )
                    target = None  # treat as unresolved

        if target is None:
            findings.append(
                Finding(
                    layer="resolver",
                    severity=Severity.HIGH,
                    message=(
                        f"Unresolvable short-form reference {c.raw!r} - no "
                        "prior full citation matches."
                    ),
                    span=c.span,
                )
            )
            continue

        # Inherit target's identifying fields so downstream validators can
        # apply the same API check without re-parsing.
        c.resolves_to = target.normalized
        c.volume = c.volume or target.volume
        c.reporter = c.reporter or target.reporter
        c.page = c.page or target.page
        c.case_name = c.case_name or target.case_name
        c.year = c.year or target.year
        c.kind = target.kind  # promote so dispatch works
    return findings
