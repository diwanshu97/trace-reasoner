"""Semantic precedent retrieval (Checkpoint 3), built on LangChain.

retrieve_precedents finds past incidents that read like the agent's current hypothesis.
This is the analogical-reasoning step from Checkpoint 3: it does not give the answer, it
moves the agent away from the obvious wrong one. The corpus loads into LangChain Documents,
embeds with BGE-small-en, and indexes in a LangChain FAISS vector store. We use the
MAX_INNER_PRODUCT strategy with normalized embeddings, so the score the store returns is
cosine similarity; hits below a 0.55 floor are dropped and the result is "no precedent
retrieved", the Checkpoint 3 mitigation against lexical-similarity-but-causal-mismatch.

The embedding model is injected. The production path uses BGE (HuggingFaceEmbeddings); tests
inject a deterministic fake embedding so they run offline without downloading a model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

_CORPUS_PATH = Path(__file__).with_name("corpus.jsonl")
DEFAULT_FLOOR = 0.55
BGE_MODEL = "BAAI/bge-small-en-v1.5"
_META_KEYS = ("id", "source", "fault_family", "service_class")


@dataclass
class Precedent:
    id: str
    score: float
    source: str
    fault_family: str
    service_class: str
    text: str


def load_corpus(path: str | Path = _CORPUS_PATH) -> list[Document]:
    """Read corpus.jsonl into LangChain Documents (text plus fault-family metadata)."""
    docs: list[Document] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        docs.append(
            Document(
                page_content=record["text"],
                metadata={key: record.get(key, "") for key in _META_KEYS},
            )
        )
    return docs


def bge_embeddings() -> Embeddings:
    """BGE-small-en with normalized vectors. Lazy import: pulls torch/sentence-transformers."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=BGE_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


class PrecedentRetriever:
    """Cosine-similarity precedent search over the Checkpoint 3 corpus, on a LangChain FAISS store."""

    def __init__(
        self,
        embeddings: Embeddings,
        docs: list[Document] | None = None,
        floor: float = DEFAULT_FLOOR,
    ) -> None:
        self.floor = floor
        self._store = FAISS.from_documents(
            docs if docs is not None else load_corpus(),
            embeddings,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

    @classmethod
    def production(cls, floor: float = DEFAULT_FLOOR) -> "PrecedentRetriever":
        """The real retriever: BGE embeddings over the bundled corpus."""
        return cls(bge_embeddings(), floor=floor)

    def retrieve(self, query: str, k: int = 5) -> list[Precedent]:
        precedents: list[Precedent] = []
        for doc, score in self._store.similarity_search_with_score(query, k=k):
            cosine = float(score)  # inner product of normalized vectors == cosine similarity
            if cosine < self.floor:
                continue
            meta = doc.metadata
            precedents.append(
                Precedent(
                    id=meta.get("id", ""),
                    score=round(cosine, 3),
                    source=meta.get("source", ""),
                    fault_family=meta.get("fault_family", ""),
                    service_class=meta.get("service_class", ""),
                    text=doc.page_content,
                )
            )
        return precedents


def retrieve_precedents(retriever: PrecedentRetriever, query: str, k: int = 5) -> str:
    """Tool surface: return top-k precedents above the floor as JSON, or 'no precedent retrieved'."""
    precedents = retriever.retrieve(query, k=k)
    if not precedents:
        return json.dumps({"query": query, "result": "no precedent retrieved", "precedents": []})
    return json.dumps({"query": query, "precedents": [asdict(p) for p in precedents]})
