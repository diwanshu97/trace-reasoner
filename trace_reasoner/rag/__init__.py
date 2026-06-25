"""Retrieval-augmented precedent search (Checkpoint 3), built on LangChain.

Importing this package requires the 'rag' optional dependency (langchain + faiss). The
research loop and the MCP session load it lazily, so the rest of the package does not
depend on LangChain being installed.
"""

from trace_reasoner.rag.retriever import (
    Precedent,
    PrecedentRetriever,
    load_corpus,
    retrieve_precedents,
)

__all__ = ["Precedent", "PrecedentRetriever", "load_corpus", "retrieve_precedents"]
