"""Human calibration scaffolding.

**An honesty note — read this first.** This module is a **tool**, not a
**result**. No real human evaluation study has been conducted in this
environment, because no real human evaluators are available. The
"calibration report" produced by ``run_calibration_demo()`` runs **on
synthetic data** and marks this explicitly with the ``is_synthetic:
True`` field. Presenting this report as a genuine finding — saying "our
system agrees with human evaluation 87% of the time" — is the exact
opposite of what this module exists for, and if discovered, it would
destroy the credibility of the entire project.

**The real workflow has three steps:**

1. ``sample_for_labeling()`` — draws a random sample from existing model
   outputs and produces a CSV/JSON template for a human labeler to fill
   in. The system's own score is **not shown** in this template — if a
   human labeler labels while knowing what the system said, an anchoring
   bias results.
2. Human labeler(s), **provided by the organization, with real
   evaluators**, fill in this template. This step is outside the scope
   of this library.
3. ``analyze_calibration()`` — reads the filled-in template, compares it
   against the system score, and computes the Spearman correlation and
   (for binary labels) precision/recall.

CLI::

    python -m core.calibration sample --outputs llm_outputs/gpt4 --n 20
    python -m core.calibration analyze --labels filled_template.json
    python -m core.calibration demo   # ONLY to demonstrate the mechanism
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT
from core.logging_setup import get_logger

logger = get_logger(__name__)

MIN_RECOMMENDED_SAMPLE = 100


# --------------------------------------------------------------------------- #
# Step 1 — sampling and template generation
# --------------------------------------------------------------------------- #
def sample_for_labeling(
    model_dir: Path,
    n: int = 30,
    dimension: str = "faithfulness",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Selects samples to be labeled and produces a template with empty label fields.

    The system's own score is **not written** into the template — this
    is a deliberate design decision (see the module docstring).
    """
    from rag.rag_evaluator import load_llm_outputs

    records = load_llm_outputs(model_dir)
    if not records:
        return []

    rng = random.Random(seed)
    selected = rng.sample(records, k=min(n, len(records)))

    template: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        template.append(
            {
                "item_id": f"CAL-{index:04d}",
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "contexts": record.get("contexts", []),
                "dimension": dimension,
                # Filled in by the human labeler — 0 (no) / 1 (yes), or a 1-5 scale.
                "human_label": None,
                "human_notes": "",
            }
        )

    if n < MIN_RECOMMENDED_SAMPLE:
        logger.warning(
            "sample size is below the recommended minimum",
            extra={"n": n, "recommended_minimum": MIN_RECOMMENDED_SAMPLE},
        )
    return template


def write_labeling_template(template: list[dict[str, Any]], output_path: Path) -> None:
    """Writes the template as JSON (the human labeler fills this in)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "labeling template written",
        extra={"path": str(output_path), "items": len(template)},
    )


# --------------------------------------------------------------------------- #
# Step 3 — analysis
# --------------------------------------------------------------------------- #
def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """A dependency-free Spearman rank correlation (no SciPy required)."""
    n = len(x)
    if n < 2:
        return float("nan")

    def _ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = average_rank
            i = j + 1
        return ranks

    rank_x = _ranks(x)
    rank_y = _ranks(y)
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n

    numerator: float = sum(
        (rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y, strict=True)
    )
    denom_x: float = sum((rx - mean_x) ** 2 for rx in rank_x) ** 0.5
    denom_y: float = sum((ry - mean_y) ** 2 for ry in rank_y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return float("nan")
    return numerator / (denom_x * denom_y)


def _binary_confusion(system_flags: Sequence[bool], human_flags: Sequence[bool]) -> dict[str, int]:
    """A confusion matrix for binary labels (the basis for precision/recall)."""
    tp = sum(1 for s, h in zip(system_flags, human_flags, strict=True) if s and h)
    fp = sum(1 for s, h in zip(system_flags, human_flags, strict=True) if s and not h)
    fn = sum(1 for s, h in zip(system_flags, human_flags, strict=True) if not s and h)
    tn = sum(1 for s, h in zip(system_flags, human_flags, strict=True) if not s and not h)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn}


def analyze_calibration(
    labeled_items: list[dict[str, Any]],
    system_scores: dict[str, float],
    is_synthetic: bool = False,
) -> dict[str, Any]:
    """Compares human labels against system scores.

    ``labeled_items`` are records each containing an ``item_id`` and a
    filled-in ``human_label`` (see ``sample_for_labeling``).
    ``system_scores`` is an ``item_id -> system score`` dict (0-1 or 0-100).
    """
    paired = [
        (system_scores[item["item_id"]], float(item["human_label"]))
        for item in labeled_items
        if item.get("human_label") is not None and item["item_id"] in system_scores
    ]

    unlabeled = sum(1 for item in labeled_items if item.get("human_label") is None)
    missing_system_score = sum(
        1 for item in labeled_items
        if item.get("human_label") is not None and item["item_id"] not in system_scores
    )

    if len(paired) < 2:
        return {
            "is_synthetic": is_synthetic,
            "n_labeled": len(paired),
            "n_unlabeled": unlabeled,
            "n_missing_system_score": missing_system_score,
            "status": "insufficient_data",
            "message": "at least 2 matched labels are required for correlation",
        }

    system_values = [pair[0] for pair in paired]
    human_values = [pair[1] for pair in paired]
    correlation = _spearman(system_values, human_values)

    result: dict[str, Any] = {
        "is_synthetic": is_synthetic,
        "n_labeled": len(paired),
        "n_unlabeled": unlabeled,
        "n_missing_system_score": missing_system_score,
        "spearman_correlation": round(correlation, 4) if correlation == correlation else None,
        "status": "completed",
    }

    unique_human = set(human_values)
    if unique_human.issubset({0.0, 1.0}):
        threshold = 0.5
        max_score = max(system_values) if system_values else 1.0
        normalized = [v / max_score if max_score > 1.5 else v for v in system_values]
        system_flags = [v >= threshold for v in normalized]
        human_flags = [v >= threshold for v in human_values]
        confusion = _binary_confusion(system_flags, human_flags)
        tp, fp, fn = confusion["true_positive"], confusion["false_positive"], confusion["false_negative"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        result["confusion_matrix"] = confusion
        result["precision"] = round(precision, 4) if precision == precision else None
        result["recall"] = round(recall, 4) if recall == recall else None

    if len(paired) < MIN_RECOMMENDED_SAMPLE:
        result["warning"] = (
            f"sample size ({len(paired)}) is below the recommended minimum "
            f"({MIN_RECOMMENDED_SAMPLE}); the result should be treated as a "
            "preliminary finding, not a definitive calibration"
        )

    return result


def format_report(analysis: dict[str, Any]) -> str:
    """Converts the analysis result into a readable report."""
    lines = ["=== Calibration Report ==="]
    if analysis.get("is_synthetic"):
        lines.append(
            "*** THIS IS A DEMONSTRATION — RUNNING ON SYNTHETIC DATA ***"
        )
        lines.append(
            "*** For real calibration, a study must be run with real "
            "human labelers. ***"
        )
    lines.append(f"Matched label count: {analysis.get('n_labeled', 0)}")
    lines.append(f"Unlabeled: {analysis.get('n_unlabeled', 0)}")
    if analysis.get("status") == "insufficient_data":
        lines.append(f"Status: {analysis['message']}")
        return "\n".join(lines)

    lines.append(f"Spearman correlation: {analysis.get('spearman_correlation')}")
    if "precision" in analysis:
        lines.append(f"Precision: {analysis.get('precision')}")
        lines.append(f"Recall: {analysis.get('recall')}")
        lines.append(f"Confusion matrix: {analysis.get('confusion_matrix')}")
    if analysis.get("warning"):
        lines.append(f"WARNING: {analysis['warning']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Demonstration — must never be presented as a real finding
# --------------------------------------------------------------------------- #
def run_calibration_demo(seed: int = 7) -> dict[str, Any]:
    """Demonstrates the mechanism with synthetic data.

    The numbers this function produces are **not real human evaluation**.
    It runs only on randomly generated data, to demonstrate that the
    ``analyze_calibration`` function works correctly.
    """
    rng = random.Random(seed)
    n = 40
    items = []
    scores = {}
    for i in range(n):
        item_id = f"DEMO-{i:03d}"
        true_quality = rng.random()
        system_score = max(0.0, min(1.0, true_quality + rng.gauss(0, 0.15)))
        human_label = 1.0 if true_quality > 0.5 else 0.0
        items.append({"item_id": item_id, "human_label": human_label})
        scores[item_id] = system_score

    analysis = analyze_calibration(items, scores, is_synthetic=True)
    analysis["_disclaimer"] = (
        "This report uses synthetic (artificially generated) data. It is "
        "not a real calibration finding and must not be presented as one."
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Human calibration scaffolding")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Generate a labeling template")
    sample_parser.add_argument("--outputs", required=True)
    sample_parser.add_argument("--n", type=int, default=30)
    sample_parser.add_argument("--dimension", default="faithfulness")
    sample_parser.add_argument("--out", default="calibration_template.json")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a filled-in template")
    analyze_parser.add_argument("--labels", required=True)
    analyze_parser.add_argument("--scores", required=True, help="a JSON file of item_id -> score")

    subparsers.add_parser("demo", help="Demonstrate the mechanism with SYNTHETIC data")

    args = parser.parse_args()

    if args.command == "sample":
        path = Path(args.outputs)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        template = sample_for_labeling(path, n=args.n, dimension=args.dimension)
        write_labeling_template(template, Path(args.out))
        print(f"a {len(template)}-item template was written: {args.out}")
        print("The human labeler must fill in the 'human_label' fields.")

    elif args.command == "analyze":
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        analysis = analyze_calibration(labels, scores)
        print(format_report(analysis))

    else:
        analysis = run_calibration_demo()
        print(format_report(analysis))


if __name__ == "__main__":
    main()
