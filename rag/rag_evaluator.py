"""Track B — Üretim kalitesi değerlendirmesi (faithfulness / answer relevance).

İki arka uç:

``ragas``
    Kurulu ve bir LLM sağlayıcısı yapılandırılmışsa RAGAS'ın
    ``faithfulness`` ve ``answer_relevancy`` metrikleri kullanılır.

``heuristic`` (varsayılan fallback)
    LLM gerektirmeyen, tekrarlanabilir ve maliyetsiz bir yaklaşım:

    * **Faithfulness**: cevap cümlelere ayrılır; her cümlenin içerik
      kelimeleri getirilen bağlamda ne oranda destekleniyor diye bakılır.
      Ayrıca sayısal iddialar (tarih, yüzde, süre) bağlamda birebir
      aranır — RAG halüsinasyonlarının en sık görüldüğü yer burasıdır.
    * **Answer relevance**: cevap ile soru arasındaki içerik kelimesi
      örtüşmesi + cevabın kaçamak/boş olup olmadığı.

Heuristic mod, LLM-as-judge kadar hassas değildir; bu bilinçli bir
takastır ve sonuçlarda ``backend`` alanıyla şeffaf biçimde raporlanır.

CLI::

    python -m rag.rag_evaluator --outputs llm_outputs/gpt4
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.schemas import GenerationQualityResult, Status

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMERIC = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]{3,}")

# İçerik taşımayan, örtüşme hesabını şişiren kelimeler.
STOPWORDS = {
    "ve", "veya", "ile", "bir", "bu", "sun", "için", "icin", "olarak", "gibi",
    "daha", "çok", "cok", "ise", "ancak", "ama", "the", "and", "for", "that",
    "this", "with", "are", "was", "her", "tüm", "tum", "olan", "göre", "gore",
}

EVASIVE_PATTERNS = (
    "bilmiyorum",
    "bilgi bulunmuyor",
    "bilgim yok",
    "cevap veremem",
    "i don't know",
    "no information",
    "cannot answer",
)


def _content_words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text or "")} - STOPWORDS


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def content_words(text: str) -> set[str]:
    """İçerik kelimelerini (stopword'süz) döndürür — diğer modüller için genel API."""
    return _content_words(text)


def split_sentences(text: str) -> list[str]:
    """Metni cümlelere böler — diğer modüller için genel API."""
    return _sentences(text)


def sentence_support(sentence: str, context_words: set[str], context_text: str) -> float:
    """Bir cümlenin bağlam tarafından desteklenme oranını (0-1) hesaplar."""
    words = _content_words(sentence)
    if not words:
        return 1.0
    lexical = len(words & context_words) / len(words)

    numbers = set(_NUMERIC.findall(sentence))
    if numbers:
        supported = sum(1 for number in numbers if number in context_text)
        numeric_ratio = supported / len(numbers)
        # Sayısal iddialar ağır basar: uydurulmuş bir tarih/oran en ciddi
        # halüsinasyon türüdür.
        return round(0.5 * lexical + 0.5 * numeric_ratio, 4)
    return round(lexical, 4)


def heuristic_faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Cevabın bağlama sadakatini 0-1 aralığında tahmin eder."""
    context_text = "\n".join(contexts)
    if not context_text.strip():
        return 0.0
    context_words = _content_words(context_text)
    sentences = _sentences(answer)
    if not sentences:
        return 0.0
    scores = [sentence_support(sentence, context_words, context_text) for sentence in sentences]
    return float(np.mean(scores))


def heuristic_relevance(question: str, answer: str) -> float:
    """Cevabın soruyla alaka düzeyini 0-1 aralığında tahmin eder."""
    if not answer.strip():
        return 0.0
    lowered = answer.lower()
    if any(pattern in lowered for pattern in EVASIVE_PATTERNS) and len(answer) < 200:
        return 0.15

    question_words = _content_words(question)
    answer_words = _content_words(answer)
    if not question_words:
        return 0.5
    overlap = len(question_words & answer_words) / len(question_words)

    # Aşırı kısa veya aşırı uzun cevaplar cezalandırılır.
    length = len(answer.split())
    length_factor = 1.0
    if length < 5:
        length_factor = 0.6
    elif length > 400:
        length_factor = 0.85
    return float(min(1.0, overlap * 0.85 + 0.15) * length_factor)


# --------------------------------------------------------------------------- #
# Veri yükleme
# --------------------------------------------------------------------------- #
def load_llm_outputs(path: Path) -> list[dict[str, Any]]:
    """``llm_outputs/<model>/`` altındaki JSON/JSONL cevap kayıtlarını okur.

    Beklenen kayıt şeması::

        {
          "question": "...",
          "answer": "...",
          "contexts": ["...", "..."],          # opsiyonel
          "expected_sources": ["dosya.md"],    # opsiyonel (retrieval icin)
          "ground_truth": "..."                # opsiyonel
        }
    """
    records: list[dict[str, Any]] = []
    if path.is_file():
        files = [path]
    else:
        files = sorted([p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl"}])

    for file_path in files:
        try:
            raw = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("cikti dosyasi okunamadi", extra={"file": str(file_path), "error": str(exc)})
            continue

        if file_path.suffix.lower() == ".jsonl":
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("gecersiz JSON", extra={"file": str(file_path)})
            continue
        if isinstance(payload, list):
            records.extend([item for item in payload if isinstance(item, dict)])
        elif isinstance(payload, dict):
            items = payload.get("records", payload.get("answers"))
            if isinstance(items, list):
                records.extend([item for item in items if isinstance(item, dict)])
            else:
                records.append(payload)

    normalized: list[dict[str, Any]] = []
    for record in records:
        question = str(record.get("question", record.get("query", ""))).strip()
        answer = str(record.get("answer", record.get("response", ""))).strip()
        if not question:
            continue
        contexts = record.get("contexts", record.get("retrieved_contexts", []))
        if isinstance(contexts, str):
            contexts = [contexts]
        normalized.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [str(c) for c in contexts if str(c).strip()],
                "expected_sources": [
                    str(s) for s in record.get("expected_sources", []) if str(s).strip()
                ],
                "ground_truth": str(record.get("ground_truth", "")),
            }
        )
    return normalized


# --------------------------------------------------------------------------- #
# RAGAS arka ucu
# --------------------------------------------------------------------------- #
def _try_ragas(records: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    """RAGAS ile faithfulness/answer_relevancy hesaplamayı dener."""
    try:
        from datasets import Dataset  # noqa: PLC0415
        from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
        from ragas.metrics import answer_relevancy, faithfulness  # noqa: PLC0415
    except ImportError:
        logger.info("ragas yuklu degil, heuristic degerlendirmeye dusuluyor")
        return None

    try:
        dataset = Dataset.from_dict(
            {
                "question": [r["question"] for r in records],
                "answer": [r["answer"] for r in records],
                "contexts": [r["contexts"] or [""] for r in records],
                "ground_truth": [r.get("ground_truth", "") for r in records],
            }
        )
        scores = ragas_evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        frame = scores.to_pandas()
        return {
            "faithfulness": float(frame["faithfulness"].fillna(0).mean()),
            "answer_relevance": float(frame["answer_relevancy"].fillna(0).mean()),
        }
    except Exception as exc:  # noqa: BLE001 - LLM saglayicisi yoksa normal
        logger.warning(
            "ragas calistirilamadi, heuristic'e dusuluyor", extra={"error": str(exc)[:200]}
        )
        return None


# --------------------------------------------------------------------------- #
# Ana değerlendirme
# --------------------------------------------------------------------------- #
def evaluate_generation(
    records: Sequence[dict[str, Any]], settings: Settings | None = None
) -> GenerationQualityResult:
    """Faithfulness / relevance / halüsinasyon oranını hesaplar."""
    settings = settings or get_settings()
    started = time.perf_counter()

    if not records:
        return GenerationQualityResult(status=Status.SKIPPED, message="degerlendirilecek cevap yok")

    limited = list(records)[: settings.rag_evaluation.max_samples]
    backend = settings.rag_evaluation.backend.lower()
    aggregate: dict[str, float] | None = None
    if backend in {"auto", "ragas"}:
        aggregate = _try_ragas(limited)
        if aggregate is None and backend == "ragas":
            return GenerationQualityResult(
                status=Status.ERROR, message="ragas arka ucu kullanilamadi"
            )

    per_record: list[dict[str, Any]] = []
    for record in limited:
        faith = heuristic_faithfulness(record["answer"], record["contexts"])
        relevance = heuristic_relevance(record["question"], record["answer"])
        per_record.append(
            {
                "question": record["question"][:200],
                "answer": record["answer"][:300],
                "faithfulness": round(faith, 4),
                "answer_relevance": round(relevance, 4),
            }
        )

    if aggregate is not None:
        faithfulness_mean = aggregate["faithfulness"]
        relevance_mean = aggregate["answer_relevance"]
        backend_name = "ragas"
    else:
        faithfulness_mean = float(np.mean([r["faithfulness"] for r in per_record]))
        relevance_mean = float(np.mean([r["answer_relevance"] for r in per_record]))
        backend_name = "heuristic"

    threshold = settings.rag_evaluation.faithfulness_threshold
    hallucinated = sum(1 for r in per_record if r["faithfulness"] < threshold)
    hallucination_rate = hallucinated / len(per_record)

    score = round(
        100.0 * (0.6 * faithfulness_mean + 0.3 * relevance_mean + 0.1 * (1 - hallucination_rate)),
        2,
    )
    worst = sorted(per_record, key=lambda r: r["faithfulness"])[:5]

    result = GenerationQualityResult(
        score=score,
        status=Status.OK,
        duration_sec=round(time.perf_counter() - started, 3),
        faithfulness=round(faithfulness_mean, 4),
        answer_relevance=round(relevance_mean, 4),
        hallucination_rate=round(hallucination_rate, 4),
        answers_evaluated=len(per_record),
        backend=backend_name,
        worst_examples=worst,
        message=f"{len(per_record)} cevap / {hallucinated} supheli",
    )
    logger.info(
        "uretim kalitesi degerlendirildi",
        extra={
            "backend": backend_name,
            "faithfulness": result.faithfulness,
            "hallucination_rate": result.hallucination_rate,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG uretim kalitesi degerlendirmesi")
    parser.add_argument("--outputs", required=True, help="llm_outputs/<model> klasoru veya JSON")
    args = parser.parse_args()

    path = Path(args.outputs)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    records = load_llm_outputs(path)
    print(evaluate_generation(records).model_dump_json(indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
