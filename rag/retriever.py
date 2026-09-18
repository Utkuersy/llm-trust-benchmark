"""Track B — the retrieval layer and retrieval quality metrics.

Two modes can be compared:
    * ``dense``  : embedding similarity only
    * ``hybrid`` : a weighted combination of dense + BM25 lexical scores
      (``rag.hybrid_alpha``; 0 = pure lexical, 1 = pure dense)

Metrics measured (against gold source labels):
    * **Context precision** : what fraction of retrieved chunks are actually relevant
    * **Context recall**    : what fraction of relevant sources were retrieved
    * **MRR**               : the average of the reciprocal rank of the first relevant result
    * **Hit rate**          : the fraction of queries with at least one relevant result retrieved

CLI::

    python -m rag.retriever --query "what is the password policy"
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.config import Settings, get_settings
from core.logging_setup import get_logger
from core.schemas import RetrievalResult, Status
from rag.vector_store import (
    ChromaVectorStore,
    Document,
    NumpyVectorStore,
    SearchHit,
    bm25_scores,
    build_embedder,
    build_vector_store,
)

logger = get_logger(__name__)


@dataclass
class RetrievalRecord:
    """The retrieval result for a single query, plus evaluation inputs."""

    question: str
    hits: list[SearchHit] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)

    @property
    def contexts(self) -> list[str]:
        """The retrieved chunk texts."""
        return [hit.text for hit in self.hits]

    @property
    def sources(self) -> list[str]:
        """The source files of the retrieved chunks."""
        return [hit.source for hit in self.hits]


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scales scores to the 0-1 range (before combining)."""
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class Retriever:
    """Runs dense or hybrid search over the vector store."""

    def __init__(
        self,
        store: NumpyVectorStore | ChromaVectorStore | None = None,
        settings: Settings | None = None,
        mode: str = "hybrid",
    ) -> None:
        self.settings = settings or get_settings()
        self.mode = mode if mode in {"dense", "hybrid"} else "hybrid"
        if store is None:
            embedder = build_embedder(self.settings)
            store = build_vector_store(self.settings, embedder)
            store.load()
        self.store = store
        self._documents: list[Document] = []

    @property
    def documents(self) -> list[Document]:
        """All documents in the store (cached for lexical scoring)."""
        if not self._documents:
            self._documents = self.store.all_documents()
        return self._documents

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Returns the best ``top_k`` chunks for the query."""
        top_k = top_k or self.settings.rag.top_k
        dense_hits = self.store.query(query, top_k * 3 if self.mode == "hybrid" else top_k)
        if self.mode == "dense":
            return dense_hits[:top_k]

        documents = self.documents
        if not documents:
            return dense_hits[:top_k]

        lexical = _minmax(bm25_scores(query, documents))
        lexical_by_id = {
            doc.doc_id: float(score) for doc, score in zip(documents, lexical, strict=True)
        }

        dense_scores = _minmax(np.asarray([hit.score for hit in dense_hits], dtype=np.float32))
        alpha = float(self.settings.rag.hybrid_alpha)

        combined: dict[str, SearchHit] = {}
        for hit, dense_score in zip(dense_hits, dense_scores, strict=True):
            score = alpha * float(dense_score) + (1 - alpha) * lexical_by_id.get(hit.doc_id, 0.0)
            combined[hit.doc_id] = SearchHit(
                doc_id=hit.doc_id, text=hit.text, source=hit.source,
                score=round(score, 6), metadata=hit.metadata,
            )

        # Also include chunks that missed the dense results but score strongly lexically.
        for doc, score in sorted(
            zip(documents, lexical, strict=True), key=lambda pair: -pair[1]
        )[:top_k]:
            if doc.doc_id not in combined and score > 0:
                combined[doc.doc_id] = SearchHit(
                    doc_id=doc.doc_id, text=doc.text, source=doc.source,
                    score=round((1 - alpha) * float(score), 6), metadata=doc.metadata,
                )

        return sorted(combined.values(), key=lambda hit: -hit.score)[:top_k]

    def batch(
        self, questions: Sequence[str], expected: Sequence[Sequence[str]] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalRecord]:
        """Produces retrieval records for multiple queries."""
        records: list[RetrievalRecord] = []
        for index, question in enumerate(questions):
            gold = list(expected[index]) if expected and index < len(expected) else []
            records.append(
                RetrievalRecord(
                    question=question, hits=self.search(question, top_k), expected_sources=gold
                )
            )
        return records


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def _is_relevant(hit_source: str, expected: Sequence[str]) -> bool:
    """Says whether a chunk belongs to one of the gold sources."""
    normalized = hit_source.strip().lower()
    return any(normalized == item.strip().lower() for item in expected)


def evaluate_retrieval(
    records: Sequence[RetrievalRecord], settings: Settings | None = None
) -> RetrievalResult:
    """Computes retrieval quality metrics and a 0-100 sub-score."""
    settings = settings or get_settings()
    started = time.perf_counter()

    graded = [record for record in records if record.expected_sources]
    if not graded:
        return RetrievalResult(
            status=Status.SKIPPED,
            message="dataset has no gold source labels",
            queries_evaluated=len(records),
        )

    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    hits_count = 0

    for record in graded:
        flags = [_is_relevant(source, record.expected_sources) for source in record.sources]
        retrieved = len(flags) or 1
        relevant_retrieved = sum(flags)

        precisions.append(relevant_retrieved / retrieved)
        found_sources = {
            source.lower() for source, flag in zip(record.sources, flags, strict=True) if flag
        }
        expected_set = {item.lower() for item in record.expected_sources}
        recalls.append(len(found_sources & expected_set) / max(1, len(expected_set)))

        rank = next((i + 1 for i, flag in enumerate(flags) if flag), 0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        hits_count += int(any(flags))

    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    mrr = float(np.mean(reciprocal_ranks))
    hit_rate = hits_count / len(graded)

    score = round(100.0 * (0.35 * precision + 0.35 * recall + 0.30 * mrr), 2)
    return RetrievalResult(
        score=score,
        status=Status.OK,
        duration_sec=round(time.perf_counter() - started, 3),
        context_precision=round(precision, 4),
        context_recall=round(recall, 4),
        mrr=round(mrr, 4),
        hit_rate=round(hit_rate, 4),
        queries_evaluated=len(graded),
        message=f"{len(graded)} queries evaluated",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retriever trial")
    parser.add_argument("--query", required=True, help="The question to search for")
    parser.add_argument("--mode", default="hybrid", choices=["dense", "hybrid"])
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    retriever = Retriever(mode=args.mode)
    for rank, hit in enumerate(retriever.search(args.query, args.top_k), start=1):
        preview: Any = hit.text[:160].replace("\n", " ")
        print(f"{rank}. [{hit.score:.4f}] {hit.source} :: {preview}")


if __name__ == "__main__":
    main()
