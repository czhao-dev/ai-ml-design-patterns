"""Hand-rolled Okapi BM25 keyword search over the local synthetic knowledge base.

Not scikit-learn TF-IDF and not an embedding-based approach -- BM25 is a small,
fully explainable ranking function, in keeping with this repo's "build it by
hand" ethos, and is more than sufficient for unambiguous multi-hop lookups over
a 15-20 document corpus. Embedding-based retrieval is left as Future Work.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .base import CallContext, ToolResult

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "by", "as", "that", "this", "it", "its", "from", "which", "who",
    "does", "do", "did", "has", "have", "had", "what",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        """documents: [{"doc_id": str, "title": str, "text": str}, ...]"""
        self.k1 = k1
        self.b = b
        self.documents = {doc["doc_id"]: doc for doc in documents}
        self._doc_term_freqs: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_length = 0.0
        self._build_index()

    def _build_index(self) -> None:
        total_length = 0
        for doc_id, doc in self.documents.items():
            terms = tokenize(doc["text"])
            total_length += len(terms)
            self._doc_lengths[doc_id] = len(terms)
            term_freqs: dict[str, int] = {}
            for term in terms:
                term_freqs[term] = term_freqs.get(term, 0) + 1
            self._doc_term_freqs[doc_id] = term_freqs
            for term in term_freqs:
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._avg_doc_length = total_length / len(self.documents) if self.documents else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self._doc_freq.get(term, 0)
        # Standard Okapi BM25 IDF with +1 smoothing to keep it non-negative.
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_terms: list[str], doc_id: str) -> float:
        term_freqs = self._doc_term_freqs.get(doc_id, {})
        doc_length = self._doc_lengths.get(doc_id, 0)
        total = 0.0
        for term in query_terms:
            freq = term_freqs.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf(term)
            denom = freq + self.k1 * (1 - self.b + self.b * doc_length / (self._avg_doc_length or 1))
            total += idf * (freq * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        scored = [(doc_id, self.score(query_terms, doc_id)) for doc_id in self.documents]
        scored = [(doc_id, s) for doc_id, s in scored if s > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def load_knowledge_base(kb_dir: Path) -> list[dict]:
    manifest = json.loads((kb_dir / "manifest.json").read_text())
    documents = []
    for entry in manifest:
        text = (kb_dir / entry["path"]).read_text()
        documents.append({"doc_id": entry["doc_id"], "title": entry["title"], "text": text})
    return documents


class RetrievalTool:
    name = "search_knowledge_base"
    description = (
        "Search the local knowledge base for documents relevant to a query. "
        "Returns up to top_k document titles and excerpts. Use this to look up "
        "facts about companies, people, or products described in the knowledge base."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "top_k": {
                "type": "integer",
                "description": "Maximum number of documents to return.",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    def __init__(self, index: BM25Index, default_top_k: int = 3) -> None:
        self.index = index
        self.default_top_k = default_top_k

    def __call__(self, call_context: CallContext, query: str, top_k: int | None = None) -> ToolResult:
        k = top_k or self.default_top_k
        hits = self.index.search(query, top_k=k)
        if not hits:
            return ToolResult(content="No matching documents found.")
        blocks = []
        for doc_id, _score in hits:
            doc = self.index.documents[doc_id]
            blocks.append(f"### {doc['title']}\n{doc['text']}")
        return ToolResult(content="\n\n".join(blocks))
