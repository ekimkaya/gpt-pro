"""Restatement / Model Rules / UCC lookup using a shipped JSON corpus.

Secondary-source citations (Restatement (Second) of Contracts § 90;
Model Rules of Pro. Conduct r. 1.6) are finite, slow-changing, and
routinely cited in legal briefs. We ship a small verified corpus and
validate existence locally. The firm can expand the corpus via JSON.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..models import Citation, CitationKind


_RE = re.compile(
    r"(?P<work>Restatement(?:\s*\((?:First|Second|Third|Fourth)\))?\s+of\s+[A-Z][A-Za-z\s]+?"
    r"|Model\s+Rules?\s+of\s+Pro(?:fessional)?\.?\s+Conduct"
    r"|U\.?C\.?C\.?)"
    r"\s*(?:§|r\.)\s*(?P<section>[\d\-\.]+)"
)


def extract_secondary_sources(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _RE.finditer(text):
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.SECONDARY,
                title=m.group("work").strip(),
                section=m.group("section"),
                span=(m.start(), m.end()),
            )
        )
    return out


_DEFAULT_CORPUS_PATH = Path(__file__).with_name("restatement_corpus.json")


class RestatementCorpus:
    """A local KV lookup over a JSON corpus. Default corpus ships with the package."""

    def __init__(self, corpus_path: str | None = None):
        path = Path(corpus_path) if corpus_path else _DEFAULT_CORPUS_PATH
        if path.exists():
            with path.open() as f:
                raw = json.load(f)
        else:
            raw = _BUILTIN_SEED
        # Normalize keys to lowercase "<work>|<section>".
        self._db: dict[str, str] = {
            f"{k.lower().strip()}|{s}": v for k, sections in raw.items() for s, v in sections.items()
        }

    def supports(self, c: Citation) -> bool:
        return c.kind == CitationKind.SECONDARY

    def lookup(self, c: Citation) -> dict[str, Any] | None:
        if not (c.title and c.section):
            return None
        key = f"{c.title.lower().strip()}|{c.section}"
        text = self._db.get(key)
        if text is None:
            return None
        return {"canonical_title": f"{c.title} § {c.section}", "source": "corpus", "text": text}


# Minimal built-in seed so the default client is useful without shipping data files.
_BUILTIN_SEED: dict[str, dict[str, str]] = {
    "Restatement (Second) of Contracts": {
        "1": "A contract is a promise or a set of promises for the breach of which the law gives a remedy.",
        "90": "A promise which the promisor should reasonably expect to induce action or forbearance ... is binding if injustice can be avoided only by enforcement of the promise.",
        "205": "Every contract imposes upon each party a duty of good faith and fair dealing in its performance and its enforcement.",
    },
    "Restatement (Second) of Torts": {
        "46": "One who by extreme and outrageous conduct intentionally or recklessly causes severe emotional distress to another is subject to liability.",
        "402A": "One who sells any product in a defective condition unreasonably dangerous to the user or consumer ... is subject to liability.",
    },
    "Model Rules of Professional Conduct": {
        "1.1": "A lawyer shall provide competent representation to a client.",
        "1.6": "A lawyer shall not reveal information relating to the representation of a client unless the client gives informed consent.",
        "3.3": "A lawyer shall not knowingly make a false statement of fact or law to a tribunal.",
    },
    "U.C.C.": {
        "2-207": "A definite and seasonable expression of acceptance ... operates as an acceptance even though it states terms additional to or different from those offered.",
        "2-314": "Unless excluded or modified, a warranty that the goods shall be merchantable is implied in a contract for their sale.",
    },
}


# Convenience: materialize the seed corpus to disk the first time someone wants it.
def write_seed_corpus(path: str) -> None:
    with open(path, "w") as f:
        json.dump(_BUILTIN_SEED, f, indent=2)
