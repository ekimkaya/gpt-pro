"""Citation extractors for every citation kind we detect.

Each extractor returns a list of :class:`Citation` objects with ``span``
populated so downstream layers (quote attribution, short-form resolution)
can locate the hit inside the original text.

:func:`extract_all` runs every extractor in order and deduplicates by span.
"""

from __future__ import annotations

import re
from typing import Iterable

from .models import Citation, CitationKind


# --------------------------------------------------------------------------- #
# US case law (CourtListener-validated)
# --------------------------------------------------------------------------- #


_US_CASE_RE = re.compile(
    r"""
    (?P<volume>\d{1,4})\s+
    (?P<reporter>
        (?:U\.?\s?S\.?)
      | (?:S\.?\s?Ct\.?)
      | (?:L\.?\s?Ed\.?(?:\s?2d)?)
      | (?:F\.?\s?(?:2d|3d|4th|Supp\.?(?:\s?2d|\s?3d)?)?)
      | (?:N\.?\s?[EW]\.?(?:\s?2d|\s?3d)?)
      | (?:A\.?(?:\s?2d|\s?3d)?)
      | (?:P\.?(?:\s?2d|\s?3d)?)
      | (?:S\.?\s?[EW]\.?(?:\s?2d|\s?3d)?)
      | (?:Cal\.?(?:\s?App\.?)?(?:\s?\d[a-z]{2})?)
      | (?:N\.?\s?Y\.?(?:\s?2d|\s?3d)?)
      | (?:Tex\.?)
      | (?:Mass\.?)
    )\s+
    (?P<page>\d{1,5})
    (?:,\s*(?P<pincite>\d{1,5}))?
    (?:\s*\((?:[^)]*?)(?P<year>\d{4})\))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CAP = r"[A-Z][A-Za-z0-9\.\-'’&]*"
_CASE_NAME_RE = re.compile(
    rf"((?:{_CAP})(?:\s+(?:{_CAP}|of|the|and|in|for|de)){{0,6}}?\s+v\.?\s+"
    rf"(?:{_CAP})(?:\s+(?:{_CAP}|of|the|and|in|for|de)){{0,6}}?)[,\s]+",
)


def _normalize_reporter(rep: str) -> str:
    rep = re.sub(r"\s+", " ", rep.strip())
    subs = [
        (r"^U\.?\s?S\.?$", "U.S."),
        (r"^S\.?\s?Ct\.?$", "S. Ct."),
        (r"^L\.?\s?Ed\.?(\s?2d)?$", lambda m: "L. Ed. 2d" if m.group(1) else "L. Ed."),
        (r"^F\.?$", "F."),
        (r"^F\.?\s?2d$", "F.2d"),
        (r"^F\.?\s?3d$", "F.3d"),
        (r"^F\.?\s?4th$", "F.4th"),
        (r"^F\.?\s?Supp\.?$", "F. Supp."),
        (r"^F\.?\s?Supp\.?\s?2d$", "F. Supp. 2d"),
        (r"^F\.?\s?Supp\.?\s?3d$", "F. Supp. 3d"),
    ]
    for pat, repl in subs:
        m = re.match(pat, rep, re.IGNORECASE)
        if m:
            return repl(m) if callable(repl) else repl
    return rep


def extract_us_cases(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _US_CASE_RE.finditer(text):
        start = m.start()
        pre = text[max(0, start - 140):start]
        name_match = list(_CASE_NAME_RE.finditer(pre))
        case_name = name_match[-1].group(1).strip(" ,") if name_match else None
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.CASE,
                volume=m.group("volume"),
                reporter=_normalize_reporter(m.group("reporter")),
                page=m.group("page"),
                pincite=m.group("pincite"),
                year=int(m.group("year")) if m.group("year") else None,
                case_name=case_name,
                span=(start, m.end()),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# US Code and CFR
# --------------------------------------------------------------------------- #


_USC_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s+U\.?\s?S\.?\s?C\.?\s*§+\s*"
    r"(?P<section>\d+[A-Za-z]?(?:[-.]\d+)*)"
    r"(?:\((?P<sub>[a-z0-9]+)\))?",
)
_CFR_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s+C\.?\s?F\.?\s?R\.?\s*§+\s*"
    r"(?P<section>\d+(?:\.\d+)*)"
    r"(?:\((?P<sub>[a-z0-9]+)\))?",
)
# State statutes — loose catch-all. Each state has its own abbreviation style;
# we capture the common ones and tag jurisdiction so specialized clients can
# dispatch.
_STATE_STATUTE_RE = re.compile(
    r"\b(?P<jur>Cal\.|N\.Y\.|Tex\.|Fla\.|Ill\.|Pa\.|Ohio)\s+"
    r"(?P<code>[A-Z][A-Za-z\.\s&]{2,40}?)\s+"
    r"(?:Code\s+)?§+\s*(?P<section>\d+(?:[-.]\d+)*)",
)


def extract_statutes(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _USC_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.STATUTE,
                title=m.group("title"),
                section=m.group("section"),
                subsection=m.group("sub"),
                jurisdiction="US",
                span=(m.start(), m.end()),
            )
        )
    for m in _CFR_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.REGULATION,
                title=m.group("title"),
                section=m.group("section"),
                subsection=m.group("sub"),
                jurisdiction="US",
                span=(m.start(), m.end()),
            )
        )
    for m in _STATE_STATUTE_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.STATE_STATUTE,
                section=m.group("section"),
                jurisdiction=m.group("jur").rstrip("."),
                notes=f"code={m.group('code').strip()}",
                span=(m.start(), m.end()),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Foreign jurisdictions
# --------------------------------------------------------------------------- #


# UK: [2019] UKSC 4, [2023] EWCA Civ 456, [2020] EWHC 1234 (Admin)
_UK_RE = re.compile(
    r"\[(?P<year>\d{4})\]\s+(?P<court>UKSC|UKHL|UKPC|EWCA(?:\s+Civ|\s+Crim)?|EWHC)\s+(?P<num>\d+)"
    r"(?:\s+\((?P<div>[A-Za-z]+)\))?",
)
# EU: Case C-123/19 P, Joined Cases C-293/12 and C-594/12, ECLI:EU:...
_EU_RE = re.compile(
    r"(?:(?:Joined\s+)?Cases?\s+)?C[-‑]\d+/\d{2}(?:\s+P)?"
    r"|ECLI:[A-Z]{2}:[A-Z]+:\d{4}:\d+",
)
# Canada neutral: 2020 SCC 7, 2019 ONCA 512, 2018 BCCA 100
_CA_RE = re.compile(
    r"\b(?P<year>\d{4})\s+(?P<court>SCC|FCA|FC|ONCA|ONSC|BCCA|BCSC|ABCA|QCCA|NSCA)\s+(?P<num>\d+)",
)
# Australia: [2020] HCA 14, [2019] FCA 123
_AU_RE = re.compile(
    r"\[(?P<year>\d{4})\]\s+(?P<court>HCA|FCA|FCAFC|NSWCA|VSCA)\s+(?P<num>\d+)",
)


def extract_foreign_cases(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _UK_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.FOREIGN_CASE,
                jurisdiction="UK",
                year=int(m.group("year")),
                court=m.group("court"),
                neutral=m.group(0),
                span=(m.start(), m.end()),
            )
        )
    for m in _EU_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.FOREIGN_CASE,
                jurisdiction="EU",
                neutral=m.group(0),
                span=(m.start(), m.end()),
            )
        )
    for m in _CA_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.FOREIGN_CASE,
                jurisdiction="CA",
                year=int(m.group("year")),
                court=m.group("court"),
                neutral=m.group(0),
                span=(m.start(), m.end()),
            )
        )
    for m in _AU_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.FOREIGN_CASE,
                jurisdiction="AU",
                year=int(m.group("year")),
                court=m.group("court"),
                neutral=m.group(0),
                span=(m.start(), m.end()),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Federal dockets (PACER)
# --------------------------------------------------------------------------- #


# 1:23-cv-04456, 2:20-cr-00123-ABC, No. 20-1234
_DOCKET_RE = re.compile(
    r"\b(?:No\.\s*)?"
    r"(?:(?P<div>\d{1,2}):)?"
    r"(?P<year>\d{2})[-‑](?P<type>cv|cr|mc|md|mj|bk)[-‑](?P<num>\d{3,6})"
    r"(?:[-‑][A-Z]{2,4})?",
    re.IGNORECASE,
)


def extract_dockets(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _DOCKET_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.DOCKET,
                docket_number=m.group(0),
                jurisdiction="US",
                span=(m.start(), m.end()),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Short forms — id., supra, party-only references
# --------------------------------------------------------------------------- #


_ID_RE = re.compile(r"\bId\.(?:\s+at\s+(?P<pin>\d+))?", re.IGNORECASE)
_SUPRA_RE = re.compile(
    rf"(?P<party>{_CAP}(?:\s+{_CAP}){{0,3}}),\s+supra(?:\s+at\s+(?P<pin>\d+))?",
)
# "*Varghese*, 925 F.3d at 1342" — short form referring to an earlier full cite.
_SHORT_VOLUME_RE = re.compile(
    rf"(?P<party>{_CAP}(?:\s+{_CAP}){{0,3}}),\s+"
    r"(?P<volume>\d{1,4})\s+(?P<reporter>[A-Z][A-Za-z0-9\.\s]{1,15}?)\s+at\s+(?P<pin>\d+)",
)


def extract_short_forms(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _ID_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.SHORT_FORM,
                pincite=m.group("pin"),
                notes="id",
                span=(m.start(), m.end()),
            )
        )
    for m in _SUPRA_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.SHORT_FORM,
                case_name=m.group("party"),
                pincite=m.group("pin"),
                notes="supra",
                span=(m.start(), m.end()),
            )
        )
    for m in _SHORT_VOLUME_RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.SHORT_FORM,
                case_name=m.group("party"),
                volume=m.group("volume"),
                reporter=_normalize_reporter(m.group("reporter")),
                pincite=m.group("pin"),
                notes="short_volume",
                span=(m.start(), m.end()),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Unified extractor
# --------------------------------------------------------------------------- #


def extract_all(text: str) -> list[Citation]:
    """Run every extractor and deduplicate by span (earlier kinds win)."""
    found: list[Citation] = []
    found.extend(extract_us_cases(text))
    found.extend(extract_statutes(text))
    found.extend(extract_foreign_cases(text))
    found.extend(extract_dockets(text))
    found.extend(extract_short_forms(text))
    # Sort by start span, drop overlaps (prefer the earliest-added = higher priority).
    found.sort(key=lambda c: (c.span[0] if c.span else 0, _kind_priority(c.kind)))
    out: list[Citation] = []
    for c in found:
        if not c.span:
            out.append(c)
            continue
        if _overlaps(c.span, [x.span for x in out if x.span]):
            continue
        out.append(c)
    return out


_KIND_ORDER = [
    CitationKind.CASE,
    CitationKind.STATUTE,
    CitationKind.REGULATION,
    CitationKind.STATE_STATUTE,
    CitationKind.FOREIGN_CASE,
    CitationKind.DOCKET,
    CitationKind.SECONDARY,
    CitationKind.SHORT_FORM,
]


def _kind_priority(k: CitationKind) -> int:
    try:
        return _KIND_ORDER.index(k)
    except ValueError:
        return 99


def _overlaps(span: tuple[int, int], existing: Iterable[tuple[int, int]]) -> bool:
    s, e = span
    for es, ee in existing:
        if s < ee and es < e:
            return True
    return False
