"""Matematik yeteneği değerlendirmesi.

Modelin bir soru setine verdiği cevapları referans cevaplarla karşılaştırır.
Zorluk, doğruluğu ölçmekten çok **cevabı serbest metinden çıkarmak ve
denkliği doğru tanımlamaktır**:

* ``1/2``, ``0.5``, ``0,5`` ve ``\\frac{1}{2}`` aynı cevaptır
* ``2x + 4`` ile ``4 + 2x`` aynı ifadedir
* ``12 elma`` ile ``12`` aynı cevaptır
* ``$1,200`` ile ``1200`` aynı sayıdır

Bu yüzden karşılaştırma üç kademelidir:

1. **Normalize edilmiş string eşitliği** — en hızlı, en kesin
2. **Sayısal denklik** — tolerans dahilinde (``math_eval.tolerance``)
3. **Sembolik denklik** — SymPy varsa ``simplify(a - b) == 0``

Cevap çıkarılamayan durumlar *yanlış* sayılmaz, ayrı raporlanır
(``extraction_failures``): bu bir model hatası değil, çıktı formatı
uyumsuzluğudur ve ayrı bir aksiyon gerektirir.

CLI::

    python -m capability.math_eval --outputs llm_outputs/gemini
    python -m capability.math_eval --seed        # ornek soru seti olustur
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.schemas import MathEvalResult, Status

logger = get_logger(__name__)

MAX_EXAMPLES = 20

# Cevap çıkarma kalıpları, öncelik sırasına göre.
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
# Ayraç (: veya =) zorunludur: aksi halde "cevaplayamıyorum" gibi kelimelerin
# içindeki "cevap" hecesi yanlışlıkla işaretçi sayılır.
ANSWER_MARKERS = re.compile(
    r"\b(?:cevap|sonu[çc]|yan[ıi]t|answer|result)\b\s*[:=]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
FINAL_NUMBER = re.compile(r"(-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)")
LATEX_FRAC = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")

SEED_PROBLEMS: list[dict[str, str]] = [
    {
        "id": "M-01",
        "category": "Prealgebra",
        "problem": "Bir depoda 480 parça var. Parçaların 3/8'i sevk edildi. Kaç parça kaldı?",
        "answer": "300",
    },
    {
        "id": "M-02",
        "category": "Prealgebra",
        "problem": (
            "Bir ürünün fiyatı önce %20 artırılıp sonra %20 indirilmiştir. "
            "Başlangıç fiyatı 500 TL ise son fiyat kaç TL'dir?"
        ),
        "answer": "480",
    },
    {
        "id": "M-03",
        "category": "Algebra",
        "problem": "3x + 7 = 2x + 15 denkleminde x kaçtır?",
        "answer": "8",
    },
    {
        "id": "M-04",
        "category": "Algebra",
        "problem": "(x + 3)(x - 3) ifadesinin açılımı nedir?",
        "answer": "x**2 - 9",
    },
    {
        "id": "M-05",
        "category": "Algebra",
        "problem": (
            "Bir işi A tek başına 6 günde, B tek başına 12 günde bitiriyor. "
            "Birlikte kaç günde bitirirler?"
        ),
        "answer": "4",
    },
    {
        "id": "M-06",
        "category": "Counting & Probability",
        "problem": (
            "Bir torbada 4 kırmızı, 6 mavi bilye var. Rastgele çekilen bir bilyenin "
            "kırmızı olma olasılığı kaçtır?"
        ),
        "answer": "2/5",
    },
    {
        "id": "M-07",
        "category": "Geometry",
        "problem": "Kenar uzunluğu 7 cm olan bir karenin alanı kaç cm² dir?",
        "answer": "49",
    },
    {
        "id": "M-08",
        "category": "Prealgebra",
        "problem": "1'den 100'e kadar olan tam sayıların toplamı kaçtır?",
        "answer": "5050",
    },
    {
        "id": "M-09",
        "category": "Prealgebra",
        "problem": (
            "Bir sunucunun çalışma süresi %99.9 ise, 30 günlük bir ayda toplam kaç "
            "dakika kesinti beklenir?"
        ),
        "answer": "43.2",
    },
    {
        "id": "M-10",
        "category": "Algebra",
        "problem": "x² - 5x + 6 = 0 denkleminin kökleri toplamı kaçtır?",
        "answer": "5",
    },
    {
        "id": "M-11",
        "category": "Number Theory",
        "problem": "24 ve 36'nın en büyük ortak böleni (EBOB) kaçtır?",
        "answer": "12",
    },
    {
        "id": "M-12",
        "category": "Intermediate Algebra",
        "problem": "log2(8) + log3(9) toplamı kaçtır?",
        "answer": "5",
    },
    {
        "id": "M-13",
        "category": "Precalculus",
        "problem": "sin(30°) + cos(60°) toplamı kaçtır?",
        "answer": "1",
    },
]


def ensure_seed_dataset(settings: Settings) -> Path:
    """Örnek soru setini (yoksa) diske yazar."""
    path = settings.math_eval.dataset_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            for problem in SEED_PROBLEMS:
                handle.write(json.dumps(problem, ensure_ascii=False) + "\n")
        logger.info("ornek matematik seti yazildi", extra={"path": str(path)})
    return path


def load_problems(path: Path) -> dict[str, dict[str, str]]:
    """JSONL soru setini ``id -> kayıt`` sözlüğü olarak okur."""
    problems: dict[str, dict[str, str]] = {}
    if not path.exists():
        return problems
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("id") and record.get("answer") is not None:
            problems[str(record["id"])] = {
                "id": str(record["id"]),
                "problem": str(record.get("problem", "")),
                "answer": str(record["answer"]),
                "category": str(record.get("category", "genel")),
            }
    return problems


def load_responses(model_dir: Path) -> dict[str, str]:
    """``math_answers.json`` dosyasından model cevaplarını okur.

    Kabul edilen iki format::

        {"M-01": "cevap metni", ...}
        [{"id": "M-01", "response": "cevap metni"}, ...]
    """
    path = model_dir / "math_answers.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("matematik cevaplari okunamadi", extra={"error": str(exc)})
        return {}

    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    if isinstance(payload, list):
        responses: dict[str, str] = {}
        for item in payload:
            if isinstance(item, dict) and item.get("id"):
                responses[str(item["id"])] = str(item.get("response", item.get("answer", "")))
        return responses
    return {}


# --------------------------------------------------------------------------- #
# Cevap çıkarma
# --------------------------------------------------------------------------- #
def extract_answer(response: str) -> str | None:
    """Serbest metinden nihai cevabı çıkarır."""
    if not response or not response.strip():
        return None
    text = response.strip()

    boxed = BOXED.findall(text)
    if boxed:
        return str(boxed[-1]).strip()

    marker = ANSWER_MARKERS.findall(text)
    if marker:
        candidate = str(marker[-1]).strip()
        if candidate:
            return candidate

    # Son satırdaki ifadeyi dene
    last_line = [line for line in text.splitlines() if line.strip()]
    if last_line:
        numbers = FINAL_NUMBER.findall(last_line[-1])
        if numbers:
            return str(numbers[-1]).strip()

    numbers = FINAL_NUMBER.findall(text)
    if numbers:
        return str(numbers[-1]).strip()
    return None


def normalize_expression(value: str) -> str:
    """Karşılaştırma öncesi ifadeyi sadeleştirir."""
    text = str(value).strip()
    text = LATEX_FRAC.sub(r"(\1)/(\2)", text)
    text = text.replace("$", "").replace("\\", "").replace("%", "")
    text = re.sub(r"\b(tl|cm|cm2|cm²|m|kg|adet|parça|parca|gün|gun|dakika|elma)\b", "", text,
                  flags=re.IGNORECASE)
    text = text.replace("^", "**").replace("×", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d)[  ](?=\d{3}\b)", "", text)   # 1 200 -> 1200
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)       # 1,200 -> 1200
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)           # 0,5   -> 0.5
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _as_float(value: str) -> float | None:
    try:
        if "/" in value and value.count("/") == 1:
            numerator, denominator = value.split("/")
            return float(numerator) / float(denominator)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _symbolically_equal(left: str, right: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False
    try:
        difference = simplify(parse_expr(left) - parse_expr(right))
        return bool(difference == 0)
    except Exception:
        return False


def answers_match(predicted: str, expected: str, tolerance: float, symbolic: bool) -> tuple[bool, str]:
    """İki cevabın denk olup olmadığını ve hangi yöntemle eşleştiğini döndürür."""
    left = normalize_expression(predicted)
    right = normalize_expression(expected)

    if left == right:
        return True, "string"

    left_value, right_value = _as_float(left), _as_float(right)
    if left_value is not None and right_value is not None:
        if abs(left_value - right_value) <= tolerance * max(1.0, abs(right_value)):
            return True, "numeric"
        return False, "numeric_mismatch"

    if symbolic and _symbolically_equal(left, right):
        return True, "symbolic"
    return False, "mismatch"


# --------------------------------------------------------------------------- #
# Değerlendirme
# --------------------------------------------------------------------------- #
def evaluate(
    responses: dict[str, str],
    problems: dict[str, dict[str, str]] | None = None,
    settings: Settings | None = None,
) -> MathEvalResult:
    """Model cevaplarını referans cevaplarla karşılaştırır."""
    settings = settings or get_settings()
    started = time.perf_counter()
    config = settings.math_eval

    if not config.enabled:
        return MathEvalResult(status=Status.SKIPPED, message="devre disi")

    if problems is None:
        problems = load_problems(ensure_seed_dataset(settings))
    if not problems:
        return MathEvalResult(status=Status.ERROR, message="soru seti bos")
    if not responses:
        return MathEvalResult(status=Status.SKIPPED, message="math_answers.json bulunamadi")

    correct = 0
    evaluated = 0
    extraction_failures = 0
    by_category: dict[str, list[int]] = {}
    examples: list[dict[str, Any]] = []

    for problem_id, problem in problems.items():
        response = responses.get(problem_id)
        if response is None:
            continue
        evaluated += 1
        category = problem["category"]
        by_category.setdefault(category, [])

        predicted = extract_answer(response)
        if predicted is None:
            extraction_failures += 1
            by_category[category].append(0)
            if len(examples) < MAX_EXAMPLES:
                examples.append(
                    {
                        "id": problem_id,
                        "category": category,
                        "expected": problem["answer"],
                        "extracted": None,
                        "correct": False,
                        "method": "extraction_failed",
                        "response_excerpt": response[:200],
                    }
                )
            continue

        matched, method = answers_match(
            predicted, problem["answer"], config.tolerance, config.symbolic_check
        )
        correct += int(matched)
        by_category[category].append(int(matched))

        if not matched and len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "id": problem_id,
                    "category": category,
                    "expected": problem["answer"],
                    "extracted": predicted,
                    "correct": False,
                    "method": method,
                    "response_excerpt": response[:200],
                }
            )

    if evaluated == 0:
        return MathEvalResult(status=Status.SKIPPED, message="eslesen soru bulunamadi")

    accuracy = correct / evaluated
    category_accuracy = {
        name: round(sum(values) / len(values), 4) for name, values in by_category.items() if values
    }

    result = MathEvalResult(
        score=round(accuracy * 100.0, 2),
        status=Status.OK,
        duration_sec=round(time.perf_counter() - started, 3),
        problems_total=len(problems),
        problems_evaluated=evaluated,
        correct=correct,
        accuracy=round(accuracy, 4),
        extraction_failures=extraction_failures,
        accuracy_by_category=category_accuracy,
        failures=examples,
        symbolic_available=config.symbolic_check and _sympy_available(),
        message=f"{correct}/{evaluated} dogru, {extraction_failures} cevap cikarilamadi",
    )
    logger.info(
        "matematik degerlendirmesi tamamlandi",
        extra={
            "evaluated": evaluated,
            "accuracy": result.accuracy,
            "extraction_failures": extraction_failures,
        },
    )
    return result


def _sympy_available() -> bool:
    try:
        import sympy  # noqa: F401
    except ImportError:
        return False
    return True


def evaluate_model(model_name: str, settings: Settings | None = None) -> MathEvalResult:
    """``llm_outputs/<model>/math_answers.json`` üzerinden değerlendirme yapar."""
    settings = settings or get_settings()
    model_dir = settings.paths.absolute(settings.paths.llm_outputs_dir) / model_name
    return evaluate(load_responses(model_dir), settings=settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matematik yetenek degerlendirmesi")
    parser.add_argument("--outputs", default=None, help="llm_outputs/<model> klasoru")
    parser.add_argument("--seed", action="store_true", help="Ornek soru setini olustur")
    args = parser.parse_args()

    settings = get_settings()
    if args.seed:
        print(f"soru seti: {ensure_seed_dataset(settings)}")
        if not args.outputs:
            return

    if not args.outputs:
        parser.error("--outputs veya --seed gerekli")

    path = Path(args.outputs)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    print(evaluate(load_responses(path), settings=settings).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
