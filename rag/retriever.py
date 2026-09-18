"""Track B — Retrieval katmanı ve retrieval kalite metrikleri.

İki mod karşılaştırılabilir:
    * ``dense``  : yalnızca embedding benzerliği
    * ``hybrid`` : dense + BM25 lexical skorlarının ağırlıklı birleşimi
      (``rag.hybrid_alpha``; 0 = saf lexical, 1 = saf dense)

Ölçülen metrikler (altın kaynak etiketleri üzerinden):
    * **Context precision** : getirilen chunk'ların kaçı gerçekten ilgili
    * **Context recall**    : ilgili kaynakların kaçı getirilebildi
    * **MRR**               : ilk ilgili sonucun sırasının tersinin ortalaması
    * **Hit rate**          : en az bir ilgili sonuç getirilen sorgu oranı

CLI::

    python -m rag.retriever --query "parola politikasi nedir"
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

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
    """Tek bir sorgunun retrieval sonucu ve değerlendirme girdileri."""

    question: str
    hits: list[SearchHit] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)

    @property
    def contexts(self) -> list[str]:
        """Getirilen chunk metinleri."""
        return [hit.text for hit in self.hits]

    @property
    def sources(self) -> list[str]:
        """Getirilen chunk'ların kaynak dosyaları."""
        return [hit.source for hit in self.hits]


def _minmax(values: np.ndarray) -> np.ndarray:
    """Skorları 0-1 aralığına ölçekler (birleştirme öncesi)."""
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class Retriever:
    """Vektör deposu üzerinde dense veya hybrid arama yapar."""

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
        """Depodaki tüm dokümanlar (lexical skorlama için cache'li)."""
        if not self._documents:
            self._documents = self.store.all_documents()
        return self._documents

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Sorgu için en iyi ``top_k`` chunk'ı döndürür."""
        top_k = top_k or self.settings.rag.top_k
        dense_hits = self.store.query(query, top_k * 3 if self.mode == "hybrid" else top_k)
        if self.mode == "dense":
            return dense_hits[:top_k]

        documents = self.documents
        if not documents:
            return dense_hits[:top_k]

        lexical = _minmax(bm25_scores(query, documents))
        lexical_by_id = {doc.doc_id: float(score) for doc, score in zip(documents, lexical)}

        dense_scores = _minmax(np.asarray([hit.score for hit in dense_hits], dtype=np.float32))
        alpha = float(self.settings.rag.hybrid_alpha)

        combined: dict[str, SearchHit] = {}
        for hit, dense_score in zip(dense_hits, dense_scores):
            score = alpha * float(dense_score) + (1 - alpha) * lexical_by_id.get(hit.doc_id, 0.0)
            combined[hit.doc_id] = SearchHit(
                doc_id=hit.doc_id, text=hit.text, source=hit.source,
                score=round(score, 6), metadata=hit.metadata,
            )

        # Dense sonuçlara girmemiş ama lexical olarak güçlü chunk'ları da ekle.
        for doc, score in sorted(
            zip(documents, lexical), key=lambda pair: -pair[1]
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
        """Birden çok sorgu için retrieval kayıtları üretir."""
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
# Retrieval metrikleri
# --------------------------------------------------------------------------- #
def _is_relevant(hit_source: str, expected: Sequence[str]) -> bool:
    """Bir chunk'ın altın kaynaklardan birine ait olup olmadığını söyler."""
    normalized = hit_source.strip().lower()
    return any(normalized == item.strip().lower() for item in expected)


def evaluate_retrieval(
    records: Sequence[RetrievalRecord], settings: Settings | None = None
) -> RetrievalResult:
    """Retrieval kalite metriklerini ve 0-100 alt puanı hesaplar."""
    settings = settings or get_settings()
    started = time.perf_counter()

    graded = [record for record in records if record.expected_sources]
    if not graded:
        return RetrievalResult(
            status=Status.SKIPPED,
            message="altin kaynak etiketi olmayan veri seti",
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
            source.lower() for source, flag in zip(record.sources, flags) if flag
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
        message=f"{len(graded)} sorgu degerlendirildi",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retriever denemesi")
    parser.add_argument("--query", required=True, help="Aranacak soru")
    parser.add_argument("--mode", default="hybrid", choices=["dense", "hybrid"])
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    retriever = Retriever(mode=args.mode)
    for rank, hit in enumerate(retriever.search(args.query, args.top_k), start=1):
        preview: Any = hit.text[:160].replace("\n", " ")
        print(f"{rank}. [{hit.score:.4f}] {hit.source} :: {preview}")  # noqa: T201


if __name__ == "__main__":
    main()
