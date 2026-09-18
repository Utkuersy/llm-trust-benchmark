"""Math capability evaluation.

Compares the model's answers to a problem set against reference answers.
The difficulty lies less in measuring accuracy than in **correctly
extracting the answer from free text and defining equivalence**:

* ``1/2``, ``0.5``, and ``\\frac{1}{2}`` are the same answer
* ``2x + 4`` and ``4 + 2x`` are the same expression
* ``12 apples`` and ``12`` are the same answer
* ``$1,200`` and ``1200`` are the same number

Comparison therefore happens in three tiers:

1. **Normalized string equality** — fastest, most precise
2. **Numeric equivalence** — within a tolerance (``math_eval.tolerance``)
3. **Symbolic equivalence** — ``simplify(a - b) == 0`` if SymPy is available

Cases where the answer cannot be extracted are not counted as *wrong*;
they are reported separately (``extraction_failures``): this is an output
format mismatch, not a model error, and it requires a different action.

CLI::

    python -m capability.math_eval --outputs llm_outputs/gemini
    python -m capability.math_eval --seed        # generate the sample problem set
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

# Answer-extraction patterns, in priority order.
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
# A delimiter (: or =) is required: otherwise the "answer" substring inside
# words like "I can't answer" would be mistakenly treated as a marker.
ANSWER_MARKERS = re.compile(
    r"\b(?:answer|result|solution)\b\s*[:=]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
FINAL_NUMBER = re.compile(r"(-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)")
LATEX_FRAC = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")

SEED_PROBLEMS: list[dict[str, str]] = [
    {
        "id": "M-01",
        "category": "Prealgebra",
        "problem": "A warehouse has 480 parts. 3/8 of the parts were shipped. How many parts remain?",
        "answer": "300",
    },
    {
        "id": "M-02",
        "category": "Prealgebra",
        "problem": (
            "A product's price is first increased by 20% and then decreased "
            "by 20%. If the starting price is $500, what is the final price?"
        ),
        "answer": "480",
    },
    {
        "id": "M-03",
        "category": "Algebra",
        "problem": "In the equation 3x + 7 = 2x + 15, what is x?",
        "answer": "8",
    },
    {
        "id": "M-04",
        "category": "Algebra",
        "problem": "What is the expansion of (x + 3)(x - 3)?",
        "answer": "x**2 - 9",
    },
    {
        "id": "M-05",
        "category": "Algebra",
        "problem": (
            "Worker A can finish a job alone in 6 days, worker B alone in "
            "12 days. How many days does it take them working together?"
        ),
        "answer": "4",
    },
    {
        "id": "M-06",
        "category": "Counting & Probability",
        "problem": (
            "A bag contains 4 red and 6 blue marbles. What is the "
            "probability that a randomly drawn marble is red?"
        ),
        "answer": "2/5",
    },
    {
        "id": "M-07",
        "category": "Geometry",
        "problem": "What is the area, in cm², of a square with a side length of 7 cm?",
        "answer": "49",
    },
    {
        "id": "M-08",
        "category": "Prealgebra",
        "problem": "What is the sum of the integers from 1 to 100?",
        "answer": "5050",
    },
    {
        "id": "M-09",
        "category": "Prealgebra",
        "problem": (
            "If a server's uptime is 99.9%, how many total minutes of "
            "downtime are expected in a 30-day month?"
        ),
        "answer": "43.2",
    },
    {
        "id": "M-10",
        "category": "Algebra",
        "problem": "In the equation x² - 5x + 6 = 0, what is the sum of the roots?",
        "answer": "5",
    },
    {
        "id": "M-11",
        "category": "Number Theory",
        "problem": "What is the greatest common divisor (GCD) of 24 and 36?",
        "answer": "12",
    },
    {
        "id": "M-12",
        "category": "Intermediate Algebra",
        "problem": "What is log2(8) + log3(9)?",
        "answer": "5",
    },
    {
        "id": "M-13",
        "category": "Precalculus",
        "problem": "What is sin(30°) + cos(60°)?",
        "answer": "1",
    },
]


def ensure_seed_dataset(settings: Settings) -> Path:
    """Writes the sample problem set to disk if it doesn't already exist."""
    path = settings.math_eval.dataset_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            for problem in SEED_PROBLEMS:
                handle.write(json.dumps(problem, ensure_ascii=False) + "\n")
        logger.info("sample math problem set written", extra={"path": str(path)})
    return path


def load_problems(path: Path) -> dict[str, dict[str, str]]:
    """Reads the JSONL problem set as an ``id -> record`` dict."""
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
                "category": str(record.get("category", "general")),
            }
    return problems


def load_responses(model_dir: Path) -> dict[str, str]:
    """Reads model answers from ``math_answers.json``.

    Two accepted formats::

        {"M-01": "answer text", ...}
        [{"id": "M-01", "response": "answer text"}, ...]
    """
    path = model_dir / "math_answers.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read math answers", extra={"error": str(exc)})
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
# Answer extraction
# --------------------------------------------------------------------------- #
def extract_answer(response: str) -> str | None:
    """Extracts the final answer from free text."""
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

    # Try the expression on the last line
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
    """Simplifies an expression before comparison."""
    text = str(value).strip()
    text = LATEX_FRAC.sub(r"(\1)/(\2)", text)
    text = text.replace("$", "").replace("\\", "").replace("%", "")
    text = re.sub(
        r"\b(usd|tl|cm|cm2|cm²|m|kg|pieces?|parts?|days?|minutes?|apples?)\b",
        "", text, flags=re.IGNORECASE,
    )
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
    """Returns whether two answers are equivalent and by which method they matched."""
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
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    responses: dict[str, str],
    problems: dict[str, dict[str, str]] | None = None,
    settings: Settings | None = None,
) -> MathEvalResult:
    """Compares model answers against reference answers."""
    settings = settings or get_settings()
    started = time.perf_counter()
    config = settings.math_eval

    if not config.enabled:
        return MathEvalResult(status=Status.SKIPPED, message="disabled")

    if problems is None:
        problems = load_problems(ensure_seed_dataset(settings))
    if not problems:
        return MathEvalResult(status=Status.ERROR, message="problem set is empty")
    if not responses:
        return MathEvalResult(status=Status.SKIPPED, message="math_answers.json not found")

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
        return MathEvalResult(status=Status.SKIPPED, message="no matching problems found")

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
        message=f"{correct}/{evaluated} correct, {extraction_failures} answers could not be extracted",
    )
    logger.info(
        "math evaluation completed",
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
    """Evaluates using ``llm_outputs/<model>/math_answers.json``."""
    settings = settings or get_settings()
    model_dir = settings.paths.absolute(settings.paths.llm_outputs_dir) / model_name
    return evaluate(load_responses(model_dir), settings=settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Math capability evaluation")
    parser.add_argument("--outputs", default=None, help="llm_outputs/<model> folder")
    parser.add_argument("--seed", action="store_true", help="Generate the sample problem set")
    args = parser.parse_args()

    settings = get_settings()
    if args.seed:
        print(f"problem set: {ensure_seed_dataset(settings)}")
        if not args.outputs:
            return

    if not args.outputs:
        parser.error("--outputs or --seed is required")

    path = Path(args.outputs)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    print(evaluate(load_responses(path), settings=settings).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
