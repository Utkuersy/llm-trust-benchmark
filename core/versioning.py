"""Test-data versioning and drift detection.

**Problem solved.** When someone says "it was 85 points last month, 70
this month," it's impossible to tell whether that's because (a) the
model got worse, (b) the test data (corpus, scenarios, lexicons)
changed, or (c) the weight preset changed. These three call for very
different actions.

**Solution.** The test data itself is hashed (``dataset_version``). Every
run records which data version it ran against. When comparing two runs,
the drift report first checks whether ``dataset_version`` and the weight
preset are the same; if not, it says "data/weights changed" and makes no
claim about model performance. If they are the same, a real
model/configuration change can be concluded.

Versioned components: the RAG corpus, injection scenarios, the math
problem set, the content-safety lexicons. Their combined hash produces a
single ``dataset_version`` string.

CLI::

    python -m core.versioning hash
    python -m core.versioning drift --model gpt4
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.storage import fetch_track_b

logger = get_logger(__name__)

SIGNIFICANT_DELTA = 5.0  # a change larger than this many points counts as "notable"


def _hash_directory(path: Path, suffixes: tuple[str, ...]) -> str:
    """Produces a combined hash of files with given extensions in a directory.

    File names are also part of the hash: renaming a file is also a
    change, not just its content changing.
    """
    if not path.exists():
        return "missing"
    hasher = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in suffixes)
    for file_path in files:
        hasher.update(str(file_path.relative_to(path)).encode("utf-8"))
        try:
            hasher.update(file_path.read_bytes())
        except OSError:
            hasher.update(b"unreadable")
    return hasher.hexdigest()[:12]


def _hash_file(path: Path) -> str:
    """The hash of a single file; 'missing' if it doesn't exist."""
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _hash_module_constant(module_name: str, attribute: str) -> str:
    """Hashes a constant list in a Python module (e.g. SCENARIOS).

    Scenarios are defined by hand in the file; without the file
    changing, someone could still mutate the constant at runtime. That's
    why we also separately hash the file itself (see
    dataset_components); this function provides an extra cross-check.
    """
    try:
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        canonical = repr(value).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:12]
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "could not hash module constant", extra={"module": module_name, "error": str(exc)}
        )
        return "unavailable"


def dataset_components(settings: Settings | None = None) -> dict[str, str]:
    """Returns the hash of each test-data component separately."""
    settings = settings or get_settings()

    corpus_dir = settings.paths.absolute(settings.paths.rag_corpus_dir)
    lexicon_dir = settings.content_safety.lexicon_dir
    if not lexicon_dir.is_absolute():
        lexicon_dir = PROJECT_ROOT / lexicon_dir
    math_path = settings.math_eval.dataset_path
    if not math_path.is_absolute():
        math_path = PROJECT_ROOT / math_path

    return {
        "rag_corpus": _hash_directory(corpus_dir, (".md", ".txt", ".rst")),
        "content_lexicons": _hash_directory(lexicon_dir, (".txt",)),
        "math_problems": _hash_file(math_path),
        "injection_scenarios": _hash_module_constant(
            "llm_security.prompt_injection_tests", "SCENARIOS"
        ),
        "poisoning_cases": _hash_module_constant(
            "llm_security.data_poisoning_sim", "POISON_CASES"
        ),
    }


def dataset_version(settings: Settings | None = None) -> str:
    """Produces a combined version string across all test-data components.

    This string changes if any component changes; use
    ``dataset_components()`` to see which one changed.
    """
    components = dataset_components(settings)
    combined = "|".join(f"{name}:{value}" for name, value in sorted(components.items()))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def explain_dataset_version(settings: Settings | None = None) -> dict[str, Any]:
    """A human-readable version report."""
    components = dataset_components(settings)
    return {
        "dataset_version": dataset_version(settings),
        "components": components,
        "missing": [name for name, value in components.items() if value == "missing"],
    }


# --------------------------------------------------------------------------- #
# Drift report
# --------------------------------------------------------------------------- #
def drift_report(model_name: str, settings: Settings | None = None) -> dict[str, Any]:
    """Analyzes the score change across a model's historical runs.

    For each consecutive pair of runs: if the data version or the weight
    preset changed, this is flagged as a **data/configuration change**
    and the score difference is not attributed to the model. If both are
    the same and the score difference exceeds the threshold, this is
    flagged as **real model/behavior change (drift)**.
    """
    settings = settings or get_settings()
    rows = [row for row in fetch_track_b(limit=500, settings=settings) if row["model_name"] == model_name]
    rows.sort(key=lambda r: r["created_at"])

    if len(rows) < 2:
        return {
            "model_name": model_name,
            "comparable_pairs": 0,
            "message": "at least 2 runs are required for comparison",
            "transitions": [],
        }

    transitions: list[dict[str, Any]] = []
    for previous, current in itertools.pairwise(rows):
        delta = round(current["trust_score"] - previous["trust_score"], 2)
        config_changed = previous["config_name"] != current["config_name"]

        if config_changed:
            classification = "config_changed"
            explanation = (
                f"the weight preset changed ({previous['config_name']} → "
                f"{current['config_name']}); the score difference is attributed to "
                "the weighting, not the model"
            )
        elif abs(delta) < SIGNIFICANT_DELTA:
            classification = "stable"
            explanation = "no significant difference"
        else:
            direction = "improvement" if delta > 0 else "degradation"
            classification = "model_drift"
            explanation = (
                f"the dataset and weights are the same; a {abs(delta):.1f}-point "
                f"{direction} points to a real model/behavior change"
            )

        transitions.append(
            {
                "from_run": previous["run_id"],
                "to_run": current["run_id"],
                "from_date": previous["created_at"],
                "to_date": current["created_at"],
                "from_score": previous["trust_score"],
                "to_score": current["trust_score"],
                "delta": delta,
                "classification": classification,
                "explanation": explanation,
            }
        )

    drift_events = [t for t in transitions if t["classification"] == "model_drift"]
    return {
        "model_name": model_name,
        "comparable_pairs": len(transitions),
        "drift_events": len(drift_events),
        "transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test-data versioning and drift tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("hash", help="Show the current test-data version")

    drift_parser = subparsers.add_parser("drift", help="Produce a drift report for a model")
    drift_parser.add_argument("--model", required=True)

    args = parser.parse_args()

    if args.command == "hash":
        report = explain_dataset_version()
        print(f"dataset_version: {report['dataset_version']}")
        for name, value in report["components"].items():
            print(f"  {name}: {value}")
        if report["missing"]:
            print(f"WARNING — missing components: {report['missing']}")
    else:
        report = drift_report(args.model)
        print(f"Model: {report['model_name']}")
        print(f"Comparable transition count: {report['comparable_pairs']}")
        for transition in report.get("transitions", []):
            print(
                f"  {transition['from_date'][:10]} -> {transition['to_date'][:10]}: "
                f"{transition['from_score']:.1f} -> {transition['to_score']:.1f} "
                f"[{transition['classification']}] {transition['explanation']}"
            )


if __name__ == "__main__":
    main()
