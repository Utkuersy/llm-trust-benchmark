"""Track B — Vektör deposu ve embedding katmanı.

**Tasarım kararı (bilinçli):** tek bir vektör DB'sine bağlanmak yerine
takılabilir bir arka uç kullanılır. Sıra: **Chroma → FAISS → saf NumPy**.
Gerekçe: staj/demo ortamında (ve CI'da) ağır bağımlılıklar her zaman
kurulamaz; NumPy fallback sayesinde Track B testleri her koşulda çalışır.
``config/settings.yaml`` içindeki ``rag.vector_backend`` ile sabitlenebilir.

Embedding için de aynı yaklaşım: ``sentence-transformers`` varsa gerçek
model, yoksa deterministik **hashing embedding** (bag-of-hashed-ngrams +
L2 normalizasyon) kullanılır. Fallback kalitesi düşüktür ama pipeline'ın
uçtan uca doğrulanmasına yeter ve sonuçta ``backend`` alanı raporlanır.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger

logger = get_logger(__name__)

HASH_DIM = 384
_TOKEN_PATTERN = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+")


def tokenize(text: str) -> list[str]:
    """Basit, dile duyarsız tokenizer (lexical arama ve hashing için)."""
    return [token.lower() for token in _TOKEN_PATTERN.findall(text or "")]


@dataclass
class Document:
    """Vektör deposuna yazılan tek bir chunk."""

    doc_id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    """Arama sonucu."""

    doc_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    """Embedding sağlayıcısı arayüzü."""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Metin listesini (n, d) boyutlu float32 matrise çevirir."""


class HashingEmbedder:
    """Bağımlılıksız, deterministik hashing tabanlı embedding."""

    name = "hashing"

    def __init__(self, dim: int = HASH_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def encode(self, texts: list[str]) -> np.ndarray:
        """Metinleri hashing uzayına gömer."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts]).astype(np.float32)


class SentenceTransformerEmbedder:
    """``sentence-transformers`` tabanlı gerçek embedding."""

    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Metinleri model uzayına gömer (L2 normalize)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Konfigürasyona göre kullanılabilir en iyi embedder'ı seçer."""
    settings = settings or get_settings()
    backend = settings.rag.embedding_backend.lower()
    if backend in {"auto", "sentence_transformers"}:
        try:
            embedder = SentenceTransformerEmbedder(settings.rag.embedding_model)
            logger.info("embedding backend", extra={"backend": embedder.name})
            return embedder
        except Exception as exc:  # noqa: BLE001 - opsiyonel bagimlilik
            if backend == "sentence_transformers":
                raise
            logger.warning(
                "sentence-transformers kullanilamadi, hashing embedding'e dusuluyor",
                extra={"error": str(exc)},
            )
    logger.info("embedding backend", extra={"backend": "hashing"})
    return HashingEmbedder()


# --------------------------------------------------------------------------- #
# Vektör deposu
# --------------------------------------------------------------------------- #
class NumpyVectorStore:
    """Bellek içi + JSON kalıcı, kosinüs benzerliğine dayalı basit depo."""

    backend = "numpy"

    def __init__(self, persist_dir: Path, collection: str, embedder: Embedder) -> None:
        self.embedder = embedder
        self.path = persist_dir / f"{collection}_numpy.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._documents: list[Document] = []
        self._matrix: np.ndarray = np.zeros((0, 1), dtype=np.float32)

    def add(self, documents: list[Document]) -> None:
        """Dokümanları depoya ekler (embedding hesaplanır)."""
        if not documents:
            return
        vectors = self.embedder.encode([doc.text for doc in documents])
        self._documents.extend(documents)
        self._matrix = (
            vectors if self._matrix.size == 0 else np.vstack([self._matrix, vectors])
        )

    def query(self, text: str, top_k: int) -> list[SearchHit]:
        """Kosinüs benzerliğine göre en yakın chunk'ları döndürür."""
        if not self._documents:
            return []
        query_vector = self.embedder.encode([text])[0]
        scores = self._matrix @ query_vector
        order = np.argsort(-scores)[:top_k]
        return [
            SearchHit(
                doc_id=self._documents[i].doc_id,
                text=self._documents[i].text,
                source=self._documents[i].source,
                score=float(scores[i]),
                metadata=self._documents[i].metadata,
            )
            for i in order
        ]

    def all_documents(self) -> list[Document]:
        """Depodaki tüm dokümanlar (lexical/BM25 katmanı için)."""
        return list(self._documents)

    def count(self) -> int:
        """Doküman sayısı."""
        return len(self._documents)

    def reset(self) -> None:
        """Depoyu sıfırlar."""
        self._documents = []
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        self.path.unlink(missing_ok=True)

    def persist(self) -> None:
        """Depoyu diske yazar."""
        payload = {
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "text": d.text,
                    "source": d.source,
                    "metadata": d.metadata,
                }
                for d in self._documents
            ],
            "vectors": self._matrix.tolist(),
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        """Diskteki depoyu yükler; başarılıysa True döner."""
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        self._documents = [
            Document(
                doc_id=item["doc_id"],
                text=item["text"],
                source=item.get("source", ""),
                metadata=item.get("metadata", {}),
            )
            for item in payload.get("documents", [])
        ]
        self._matrix = np.asarray(payload.get("vectors", []), dtype=np.float32)
        return bool(self._documents)


class ChromaVectorStore:
    """ChromaDB tabanlı kalıcı depo."""

    backend = "chroma"

    def __init__(self, persist_dir: Path, collection: str, embedder: Embedder) -> None:
        import chromadb  # noqa: PLC0415

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._cache: list[Document] = []

    def add(self, documents: list[Document]) -> None:
        """Dokümanları koleksiyona ekler."""
        if not documents:
            return
        vectors = self.embedder.encode([doc.text for doc in documents])
        self._collection.upsert(
            ids=[doc.doc_id for doc in documents],
            documents=[doc.text for doc in documents],
            embeddings=[vector.tolist() for vector in vectors],
            metadatas=[{"source": doc.source, **doc.metadata} for doc in documents],
        )
        self._cache.extend(documents)

    def query(self, text: str, top_k: int) -> list[SearchHit]:
        """Koleksiyonda benzerlik araması yapar."""
        vector = self.embedder.encode([text])[0].tolist()
        response = self._collection.query(
            query_embeddings=[vector], n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        ids = (response.get("ids") or [[]])[0]
        docs = (response.get("documents") or [[]])[0]
        metas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        for index, doc_id in enumerate(ids):
            metadata = dict(metas[index] or {}) if index < len(metas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                SearchHit(
                    doc_id=str(doc_id),
                    text=docs[index] if index < len(docs) else "",
                    source=str(metadata.pop("source", "")),
                    score=1.0 - distance,
                    metadata=metadata,
                )
            )
        return hits

    def all_documents(self) -> list[Document]:
        """Koleksiyondaki tüm dokümanlar."""
        if self._cache:
            return list(self._cache)
        payload = self._collection.get(include=["documents", "metadatas"])
        documents: list[Document] = []
        for index, doc_id in enumerate(payload.get("ids", [])):
            metadata = dict((payload.get("metadatas") or [{}])[index] or {})
            documents.append(
                Document(
                    doc_id=str(doc_id),
                    text=(payload.get("documents") or [""])[index],
                    source=str(metadata.pop("source", "")),
                    metadata=metadata,
                )
            )
        self._cache = documents
        return documents

    def count(self) -> int:
        """Doküman sayısı."""
        return int(self._collection.count())

    def reset(self) -> None:
        """Koleksiyonu boşaltır."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        self._cache = []

    def persist(self) -> None:
        """Chroma PersistentClient otomatik kalıcıdır."""

    def load(self) -> bool:
        """Var olan koleksiyonda veri olup olmadığını bildirir."""
        return self.count() > 0


def build_vector_store(
    settings: Settings | None = None, embedder: Embedder | None = None
) -> NumpyVectorStore | ChromaVectorStore:
    """Konfigürasyona göre vektör deposunu kurar (fallback zinciriyle)."""
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)

    persist_dir = settings.rag.persist_dir
    if not persist_dir.is_absolute():
        persist_dir = PROJECT_ROOT / persist_dir

    backend = settings.rag.vector_backend.lower()
    if backend in {"auto", "chroma"}:
        try:
            store = ChromaVectorStore(persist_dir, settings.rag.collection_name, embedder)
            logger.info("vector store backend", extra={"backend": "chroma"})
            return store
        except Exception as exc:  # noqa: BLE001 - opsiyonel bagimlilik
            if backend == "chroma":
                raise
            logger.warning(
                "chroma kullanilamadi, numpy deposuna dusuluyor", extra={"error": str(exc)}
            )

    logger.info("vector store backend", extra={"backend": "numpy"})
    return NumpyVectorStore(persist_dir, settings.rag.collection_name, embedder)


# --------------------------------------------------------------------------- #
# Lexical skorlama (hybrid retrieval için)
# --------------------------------------------------------------------------- #
def bm25_scores(query: str, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Doküman listesi için BM25 skorlarını hesaplar."""
    if not documents:
        return np.zeros(0, dtype=np.float32)

    corpus = [tokenize(doc.text) for doc in documents]
    lengths = np.array([len(tokens) for tokens in corpus], dtype=np.float32)
    average_length = float(lengths.mean()) if len(lengths) else 1.0
    document_count = len(corpus)

    frequencies: list[dict[str, int]] = []
    document_frequency: dict[str, int] = {}
    for tokens in corpus:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        frequencies.append(counts)
        for token in counts:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    scores = np.zeros(document_count, dtype=np.float32)
    for token in tokenize(query):
        df = document_frequency.get(token, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        for index, counts in enumerate(frequencies):
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            denominator = tf + k1 * (1 - b + b * (lengths[index] / max(average_length, 1e-6)))
            scores[index] += idf * (tf * (k1 + 1)) / max(denominator, 1e-6)
    return scores
