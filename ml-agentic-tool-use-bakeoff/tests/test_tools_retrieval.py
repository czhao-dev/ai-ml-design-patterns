"""Tests for src/tools/retrieval.py."""

from src.tools.base import CallContext
from src.tools.retrieval import BM25Index, RetrievalTool, tokenize


def _make_index() -> BM25Index:
    documents = [
        {
            "doc_id": "doc_a",
            "title": "Widget X1",
            "text": "Widget X1 is made by AstraCorp and uses FenCore battery "
            "technology licensed from Northfen Robotics.",
        },
        {
            "doc_id": "doc_b",
            "title": "AstraCorp",
            "text": "AstraCorp is a hardware company founded in 2011 by Jane "
            "Okoye, headquartered in Lisbon.",
        },
        {
            "doc_id": "doc_c",
            "title": "Northfen Robotics",
            "text": "Northfen Robotics is a robotics company founded in 2008 "
            "by Marcus Lindqvist, headquartered in Toronto.",
        },
    ]
    return BM25Index(documents)


def test_tokenize_lowercases_and_drops_stopwords():
    tokens = tokenize("Who is the founder of AstraCorp?")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "astracorp" in tokens


def test_bm25_ranks_relevant_doc_first():
    index = _make_index()
    hits = index.search("Who founded AstraCorp?", top_k=3)
    assert hits[0][0] == "doc_b"


def test_multihop_query_surfaces_both_needed_docs():
    index = _make_index()
    hits = index.search("Widget X1 AstraCorp founder", top_k=3)
    doc_ids = {doc_id for doc_id, _score in hits}
    assert "doc_a" in doc_ids
    assert "doc_b" in doc_ids


def test_retrieval_tool_formats_results():
    index = _make_index()
    tool = RetrievalTool(index, default_top_k=1)
    result = tool(CallContext.for_task(), query="Who founded AstraCorp?")
    assert "AstraCorp" in result.content
    assert not result.is_error


def test_retrieval_tool_no_match_returns_message_not_error():
    index = _make_index()
    tool = RetrievalTool(index, default_top_k=3)
    result = tool(CallContext.for_task(), query="zzz nonexistent qqq")
    assert not result.is_error
    assert "no matching" in result.content.lower()
