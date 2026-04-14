"""PII / client-data redaction for outbound LLM calls.

Policy: every byte that leaves the firm's network must be inspectable by
the firm's GC. This module implements a reversible redaction layer —
client names, matter numbers, dollar amounts, addresses, SSNs, emails,
phone numbers, and dates are replaced with opaque tokens before going to
the LLM, and the tokens are swapped back in the response.

Critical invariant: **legal citations are never redacted.** The detection
pipeline needs them intact. We run citation extraction first to mark
protected spans, then redact only non-overlapping regions.

Usage:

    redactor = Redactor(
        client_names={"Acme Corp.", "John Doe"},
        matter_number_patterns=[r"MATTER-\d{6}"],
    )
    provider = RedactingProvider(AnthropicProvider(), redactor)
    # provider.complete(...) looks identical to the inner provider but
    # scrubs+restores transparently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .citation_extractors import extract_all
from .providers import LLMProvider, LLMResponse


# --------------------------------------------------------------------------- #
# Canonical patterns - expand as needed per firm policy.
# --------------------------------------------------------------------------- #


# Order matters: longer/more specific patterns first so they consume their
# substrings before shorter patterns can match.
_BUILTIN_PATTERNS: list[tuple[str, str]] = [
    # SSN
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    # Credit card (loose)
    ("CARD", r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    # Email
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # US phone numbers
    ("PHONE", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # Dollar amounts
    ("AMOUNT", r"\$\s?\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\$\s?\d+(?:\.\d{2})?"),
    # Street addresses - loose
    (
        "ADDRESS",
        r"\b\d{1,6}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Plaza|Pl)\.?\b",
    ),
    # Dates - ISO and common US formats
    ("DATE", r"\b\d{4}-\d{2}-\d{2}\b"),
    ("DATE", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    # Honorifics + name (a weak proxy for named individuals)
    (
        "PERSON",
        r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Hon\.?|Atty\.?)\s+"
        r"[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,2}",
    ),
]


@dataclass
class RedactionMap:
    """Holds the bidirectional mapping so responses can be restored."""

    forward: dict[str, str] = field(default_factory=dict)   # token -> original
    counter: dict[str, int] = field(default_factory=dict)

    def fresh_token(self, kind: str) -> str:
        n = self.counter.get(kind, 0) + 1
        self.counter[kind] = n
        return f"[[{kind}_{n:03d}]]"

    def record(self, token: str, original: str) -> None:
        self.forward[token] = original

    def restore(self, text: str) -> str:
        # Sort by token length desc so longer tokens get replaced first
        # (prevents [[PERSON_1]] being partially matched by [[PERSON_10]]).
        for token in sorted(self.forward, key=len, reverse=True):
            text = text.replace(token, self.forward[token])
        return text


@dataclass
class Redactor:
    """Reversible redactor that preserves legal citations."""

    patterns: list[tuple[str, str]] = field(
        default_factory=lambda: list(_BUILTIN_PATTERNS)
    )
    # Firm-specific literal strings to scrub (client names, matter numbers).
    # Treated as whole-word literal matches, case-insensitive.
    client_names: set[str] = field(default_factory=set)
    matter_number_patterns: list[str] = field(default_factory=list)
    # Compiled lazily.
    _compiled: list[tuple[str, re.Pattern[str]]] | None = None

    def _compile(self) -> list[tuple[str, re.Pattern[str]]]:
        if self._compiled is not None:
            return self._compiled
        out: list[tuple[str, re.Pattern[str]]] = []
        for kind, pat in self.patterns:
            out.append((kind, re.compile(pat)))
        for pat in self.matter_number_patterns:
            out.append(("MATTER", re.compile(pat)))
        for name in self.client_names:
            # Lookarounds instead of \b: \b fails when a name ends with
            # punctuation like "Acme Corp." because "." is a non-word char.
            out.append(
                (
                    "CLIENT",
                    re.compile(
                        r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE
                    ),
                )
            )
        self._compiled = out
        return out

    def redact(self, text: str) -> tuple[str, RedactionMap]:
        """Return ``(redacted_text, map)``. Citations are preserved verbatim."""
        rmap = RedactionMap()

        # Mark citation spans as protected so we never rewrite them.
        protected = [c.span for c in extract_all(text) if c.span]

        # Collect all candidate replacement spans (non-overlapping with
        # citations and with each other; earlier patterns win by priority).
        replacements: list[tuple[int, int, str, str]] = []
        for kind, pat in self._compile():
            for m in pat.finditer(text):
                s, e = m.span()
                if _overlaps_any((s, e), protected):
                    continue
                if _overlaps_any((s, e), [(rs, re_) for rs, re_, _, _ in replacements]):
                    continue
                token = rmap.fresh_token(kind)
                rmap.record(token, m.group(0))
                replacements.append((s, e, token, m.group(0)))

        if not replacements:
            return text, rmap

        replacements.sort(key=lambda r: r[0])
        out: list[str] = []
        cursor = 0
        for s, e, token, _orig in replacements:
            out.append(text[cursor:s])
            out.append(token)
            cursor = e
        out.append(text[cursor:])
        return "".join(out), rmap

    def restore(self, text: str, rmap: RedactionMap) -> str:
        return rmap.restore(text)


def _overlaps_any(span: tuple[int, int], others: Iterable[tuple[int, int]]) -> bool:
    s, e = span
    for os_, oe in others:
        if s < oe and os_ < e:
            return True
    return False


# --------------------------------------------------------------------------- #
# Transparent provider wrapper
# --------------------------------------------------------------------------- #


@dataclass
class RedactingProvider:
    """Wraps any :class:`LLMProvider` and scrubs PII in/out transparently.

    The inner provider sees only redacted tokens; callers see the original
    strings in responses. Logs whatever the inner provider emits to
    ``redacted_log`` (a callable) so the firm's GC can inspect every
    outbound prompt byte.
    """

    inner: LLMProvider
    redactor: Redactor
    redacted_log: callable | None = None  # invoked with the redacted prompt

    @property
    def name(self) -> str:
        return f"redacting:{self.inner.name}"

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        logprobs: bool = False,
    ) -> LLMResponse:
        r_prompt, pmap = self.redactor.redact(prompt)
        r_system = None
        smap: RedactionMap | None = None
        if system:
            r_system, smap = self.redactor.redact(system)

        if self.redacted_log:
            try:
                self.redacted_log({"prompt": r_prompt, "system": r_system, "model": self.model})
            except Exception:  # noqa: BLE001 - logging must not break generation
                pass

        resp = self.inner.complete(
            r_prompt,
            system=r_system,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
        )
        # Restore tokens in the response. Merge maps so both system and
        # prompt tokens resolve.
        merged = RedactionMap(
            forward={**pmap.forward, **(smap.forward if smap else {})}
        )
        restored_text = merged.restore(resp.text)
        resp.text = restored_text
        return resp
