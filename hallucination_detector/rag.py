"""Layer 1 — Retrieval-Augmented Generation with forced citations.

Designed for legal workflows: the retriever pulls candidate cases, statutes,
or filings from an authoritative corpus, then the model is instructed to
answer *only* from that retrieved context and to tag every factual claim
with the document ID it came from. Any citation that does not resolve to a
retrieved document is flagged as an ungrounded claim.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .models import Finding, Severity
from .providers import LLMProvider


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class Document:
    """A single item in the authoritative legal corpus."""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RAGResult:
    answer: str
    retrieved: list[Document]
    cited_ids: list[str]
    ungrounded_citations: list[str]
    findings: list[Finding]


class RAGIndex:
    """A tiny, dependency-free BM25-ish retriever.

    Real deployments should swap this for a vector DB (pgvector, Weaviate,
    Pinecone) backed by an authoritative law firm corpus: Westlaw exports,
    internal memos, client filings. The interface (:meth:`search`) is what
    :func:`run_rag_with_citations` depends on.
    """

    def __init__(self, documents: list[Document] | None = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: dict[str, Document] = {}
        self._tokens: dict[str, list[str]] = {}
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0
        if documents:
            for d in documents:
                self.add(d)

    def add(self, doc: Document) -> None:
        if doc.doc_id in self._docs:
            return
        tokens = _tokenize(f"{doc.title}\n{doc.text}")
        self._docs[doc.doc_id] = doc
        self._tokens[doc.doc_id] = tokens
        for t in set(tokens):
            self._df[t] += 1
        self._avg_len = sum(len(v) for v in self._tokens.values()) / max(1, len(self._tokens))

    def _bm25(self, query_tokens: list[str], doc_id: str) -> float:
        tokens = self._tokens[doc_id]
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        N = max(1, len(self._docs))
        score = 0.0
        dl = len(tokens)
        for q in query_tokens:
            f = tf.get(q, 0)
            if f == 0:
                continue
            df = self._df.get(q, 0)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        q_tokens = _tokenize(query)
        scored = [(self._bm25(q_tokens, did), did) for did in self._docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._docs[did] for s, did in scored[:top_k] if s > 0]

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)


_RAG_SYSTEM = """You are a legal research assistant for a licensed law firm.

You will receive a question and a set of numbered source documents pulled
from the firm's authoritative corpus (case law, filings, internal memos).

Rules - these are non-negotiable:
1. Answer ONLY using facts present in the supplied documents. If the
   documents do not contain the answer, reply exactly:
   "I cannot answer from the provided sources."
2. After every factual sentence, append an inline citation of the form
   [doc_id] identifying the source document(s). Multiple ids are allowed:
   [doc_id_1][doc_id_2].
3. Never invent case names, docket numbers, judges, parties, holdings,
   or quotations. If you are not certain, say so.
4. Do not use outside knowledge, even if it seems helpful.
"""

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-\./:]+)\]")


def run_rag_with_citations(
    question: str,
    index: RAGIndex,
    provider: LLMProvider,
    *,
    top_k: int = 5,
    max_tokens: int = 1024,
) -> RAGResult:
    """Retrieve docs, generate an answer, then verify every citation resolves.

    Any ``[doc_id]`` tag in the answer that isn't in the retrieved set is
    treated as an ungrounded claim and surfaced as a HIGH-severity finding -
    that's the "capture" for this layer.
    """

    retrieved = index.search(question, top_k=top_k)
    if not retrieved:
        finding = Finding(
            layer="rag",
            severity=Severity.HIGH,
            message="No authoritative sources retrieved; model would have to rely on parametric knowledge.",
        )
        return RAGResult(
            answer="",
            retrieved=[],
            cited_ids=[],
            ungrounded_citations=[],
            findings=[finding],
        )

    context_blocks = []
    for d in retrieved:
        meta = " | ".join(f"{k}={v}" for k, v in d.metadata.items())
        header = f"[{d.doc_id}] {d.title}" + (f" ({meta})" if meta else "")
        context_blocks.append(f"{header}\n{d.text}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        f"Question:\n{question}\n\n"
        f"Authoritative sources:\n{context}\n\n"
        "Produce the answer now, citing [doc_id] after every factual sentence."
    )
    response = provider.complete(
        prompt, system=_RAG_SYSTEM, temperature=0.0, max_tokens=max_tokens
    )
    answer = response.text

    cited_ids = list(dict.fromkeys(_CITATION_RE.findall(answer)))
    valid_ids = {d.doc_id for d in retrieved}
    ungrounded = [c for c in cited_ids if c not in valid_ids]

    findings: list[Finding] = []
    if ungrounded:
        findings.append(
            Finding(
                layer="rag",
                severity=Severity.CRITICAL,
                message=(
                    f"Answer cites {len(ungrounded)} document id(s) that were not in "
                    f"the retrieved corpus: {ungrounded}. Treat as fabricated."
                ),
                evidence={"ungrounded": ungrounded, "retrieved": sorted(valid_ids)},
            )
        )

    # Heuristic: a factual-looking answer with zero citations is also suspect.
    if not cited_ids and answer.strip() and "cannot answer" not in answer.lower():
        findings.append(
            Finding(
                layer="rag",
                severity=Severity.HIGH,
                message="Answer contains no [doc_id] citations despite making factual claims.",
            )
        )

    return RAGResult(
        answer=answer,
        retrieved=retrieved,
        cited_ids=cited_ids,
        ungrounded_citations=ungrounded,
        findings=findings,
    )
