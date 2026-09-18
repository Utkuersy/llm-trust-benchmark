"""Track B — document loading, chunking, and indexing.

Flow::

    data/rag_corpus/*.md|*.txt  →  chunk (with overlap)  →  embedding  →  vector store

If the corpus is empty, the ``--seed`` flag generates a small sample
document set; this set deliberately contains *verifiable facts* (dates,
numbers, names), so that faithfulness and hallucination measurement is
meaningful.

CLI::

    python -m rag.ingest --seed        # generate and index the sample corpus
    python -m rag.ingest --reset       # reset the index and rebuild it
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from rag.vector_store import Document, build_embedder, build_vector_store

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}

SEED_CORPUS: dict[str, str] = {
    "security_policy.md": """# Corporate Information Security Policy

Policy version 4.2, effective March 12, 2024. All employees must complete
a total of 6 hours of security awareness training twice a year.

Passwords must be at least 14 characters long and changed every 90 days.
Multi-factor authentication (MFA) is mandatory for all administrator
accounts.

Suspected security incidents must be reported to the Security Operations
Center no later than 4 hours after detection.
""",
    "data_retention.md": """# Data Retention and Disposal Procedure

Customer transaction records are retained for 10 years. Behavioral data
collected for marketing purposes may be retained for at most 24 months.

Deletion requests (under applicable data protection regulation) are
completed within 30 days; removal from backups may take an additional
60 days.

Data is classified into four levels: Public, Internal, Confidential, and
Highly Confidential. Highly Confidential data is stored only on encrypted
disks.
""",
    "model_lifecycle.md": """# ML Model Lifecycle Standard

Every model deployed to production requires a Model Card. A Model Card
must contain at least the following sections: purpose, training data,
metrics, known limitations, and ethical review.

Production models are re-evaluated every 3 months. If performance drops
more than 5 points below baseline, the model is withdrawn.

The data drift monitoring threshold, measured by Population Stability
Index (PSI), is set at 0.20.
""",
    "access_management.md": """# Access Management Guide

Access requests are evaluated in two stages: manager approval and data
owner approval. Access is never granted without both approvals.

Privileged account sessions are terminated after 15 minutes of
inactivity. Privileged access logs are retained for 5 years.

All access for departing employees is revoked by the end of their last
working day. This action is verified by Human Resources.
""",
    "incident_response.md": """# Incident Response Plan

Incidents are classified into four severity levels: P1 (critical),
P2 (high), P3 (medium), P4 (low). For P1 incidents, the response team
assembles within 30 minutes.

The root cause analysis report is published within 10 business days
after the incident is closed. The report is written in a blameless
tone.

The notification obligation for incidents affecting customers is 72
hours.
""",
}


def ensure_seed_corpus(settings: Settings) -> Path:
    """Writes the sample corpus files to disk (if not already present)."""
    corpus_dir = settings.paths.absolute(settings.paths.rag_corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in SEED_CORPUS.items():
        path = corpus_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            logger.info("sample document written", extra={"file": filename})
    return corpus_dir


def load_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    """Reads the text files in the corpus folder as (name, content) pairs."""
    if not corpus_dir.exists():
        return []
    documents: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        try:
            documents.append((path.name, path.read_text(encoding="utf-8", errors="replace")))
        except OSError as exc:
            logger.warning("could not read document", extra={"file": str(path), "error": str(exc)})
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into overlapping chunks that respect paragraph boundaries."""
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{paragraph}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= chunk_size:
            buffer = paragraph
        else:
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_size])
                start += chunk_size - overlap
            buffer = ""
    if buffer:
        chunks.append(buffer)

    # Apply overlap between chunks too (reduces context loss).
    if overlap and len(chunks) > 1:
        merged = [chunks[0]]
        for previous, current in itertools.pairwise(chunks):
            merged.append((previous[-overlap:] + "\n" + current).strip())
        chunks = merged
    return [chunk for chunk in chunks if chunk.strip()]


def build_documents(pairs: list[tuple[str, str]], settings: Settings) -> list[Document]:
    """Converts raw files into chunked ``Document`` objects."""
    documents: list[Document] = []
    for source, content in pairs:
        chunks = chunk_text(content, settings.rag.chunk_size, settings.rag.chunk_overlap)
        for index, chunk in enumerate(chunks):
            digest = hashlib.blake2b(
                f"{source}:{index}:{chunk[:64]}".encode(), digest_size=8
            ).hexdigest()
            documents.append(
                Document(
                    doc_id=f"{Path(source).stem}-{index:03d}-{digest}",
                    text=chunk,
                    source=source,
                    metadata={"chunk_index": index},
                )
            )
    return documents


def ingest(
    reset: bool = False, seed: bool = False, settings: Settings | None = None
) -> dict[str, object]:
    """Reads the corpus and indexes it into the vector store; returns a summary."""
    settings = settings or get_settings()
    corpus_dir = settings.paths.absolute(settings.paths.rag_corpus_dir)

    corpus_is_empty = (not corpus_dir.exists()) or not any(corpus_dir.glob("*"))
    if seed or corpus_is_empty:
        corpus_dir = ensure_seed_corpus(settings)

    pairs = load_documents(corpus_dir)
    if not pairs:
        logger.error("corpus is empty", extra={"corpus_dir": str(corpus_dir)})
        return {"documents": 0, "chunks": 0, "backend": "none"}

    embedder = build_embedder(settings)
    store = build_vector_store(settings, embedder)

    if reset:
        store.reset()
    elif store.load() and store.count() > 0:
        logger.info("using existing index", extra={"chunks": store.count()})
        return {
            "documents": len(pairs),
            "chunks": store.count(),
            "backend": store.backend,
            "embedder": embedder.name,
            "reused": True,
        }

    documents = build_documents(pairs, settings)
    store.add(documents)
    store.persist()

    summary = {
        "documents": len(pairs),
        "chunks": len(documents),
        "backend": store.backend,
        "embedder": embedder.name,
        "corpus_dir": str(corpus_dir.relative_to(PROJECT_ROOT)),
        "reused": False,
    }
    logger.info("indexing completed", extra=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the RAG corpus")
    parser.add_argument("--reset", action="store_true", help="Delete the existing index and rebuild it")
    parser.add_argument("--seed", action="store_true", help="Generate the sample corpus files")
    args = parser.parse_args()
    print(ingest(reset=args.reset, seed=args.seed))


if __name__ == "__main__":
    main()
