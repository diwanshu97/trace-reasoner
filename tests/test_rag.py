import importlib.util
import json

import pytest

rag_available = importlib.util.find_spec("langchain_community") is not None
pytestmark = pytest.mark.skipif(not rag_available, reason="rag extra (langchain) not installed")


def fake_retriever(floor=0.55):
    # Deterministic fake embeddings exercise the LangChain FAISS path offline,
    # with no model download. Scores are not semantic, so tests use floor extremes.
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from trace_reasoner.rag.retriever import PrecedentRetriever

    return PrecedentRetriever(DeterministicFakeEmbedding(size=64), floor=floor)


def test_corpus_loads_as_documents():
    from trace_reasoner.rag.retriever import load_corpus

    docs = load_corpus()
    assert len(docs) == 12
    assert all(d.metadata.get("id") for d in docs)
    assert any(d.metadata["fault_family"] == "saturation" for d in docs)


def test_returns_at_most_k_when_floor_is_open():
    hits = fake_retriever(floor=-1e9).retrieve("redis connection pool saturation", k=3)
    assert 0 < len(hits) <= 3
    assert all(hasattr(h, "fault_family") for h in hits)


def test_high_floor_returns_no_precedent():
    from trace_reasoner.rag.retriever import retrieve_precedents

    out = json.loads(retrieve_precedents(fake_retriever(floor=1e9), "anything", k=5))
    assert out["result"] == "no precedent retrieved"
    assert out["precedents"] == []


def test_output_shape():
    from trace_reasoner.rag.retriever import retrieve_precedents

    out = json.loads(retrieve_precedents(fake_retriever(floor=-1e9), "kafka consumer lag", k=2))
    assert out["precedents"]
    precedent = out["precedents"][0]
    assert set(precedent) >= {"id", "score", "source", "fault_family", "service_class", "text"}


def test_session_uses_injected_retriever():
    from trace_reasoner.mcp.session import TraceReasonerSession

    session = TraceReasonerSession(retriever=fake_retriever(floor=-1e9))
    out = json.loads(session.retrieve_precedents("redis pool", k=2))
    assert "precedents" in out
