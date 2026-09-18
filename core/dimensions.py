"""Dimension registry.

**Problem solved.** The seven dimensions used to be baked into the
engine: ``benchmark_model()`` called each one by name, and the weighting
had to be kept in sync with a separate constant list (``DIMENSIONS``).
Adding an eighth dimension required changes in four places: the engine,
the weight list, storage, the dashboard.

**Solution.** Each dimension **registers itself** with this registry (a
single call):

    register(Dimension(
        key="content_safety",
        label="Content safety",
        iso_pillar="safety",
        owasp_ref=None,
        evaluator=lambda ctx: content_safety_scan.scan_records(ctx.records, ctx.settings),
    ))

The engine runs the registered dimensions in a loop
(``for dim in all_dimensions(): setattr(result, dim.key, dim.evaluator(ctx))``).
Adding a new dimension now means: write an evaluator function, add a
single ``register()`` call, add a field to the ``EvaluationResult``
schema. The engine code does not change.

**Limitation.** This is Python; it is not a "real" plugin system
(dynamic discovery, version-compatibility checks, sandbox isolation).
The goal is to eliminate hardcoded duplication and establish a single
source of truth — not a general-purpose plugin marketplace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core.config import Settings
from core.schemas import BaseResult
from rag.retriever import RetrievalRecord


@dataclass
class DimensionContext:
    """Gathers everything an evaluator might need in one place.

    All evaluators share the same signature (``Callable[[DimensionContext],
    BaseResult]``); they ignore the fields they don't need. This keeps the
    registration interface uniform — some dimensions use ``records``,
    others ``model_name``, others neither.
    """

    model_name: str
    records: list[dict]
    retrieval_records: list[RetrievalRecord]
    settings: Settings
    skip_poisoning: bool = False


@dataclass
class Dimension:
    """Registration information for a single measurement dimension."""

    key: str
    """Must match a field name on the EvaluationResult schema exactly (e.g. 'content_safety')."""

    label: str
    """Human-readable name (used in the dashboard and reports)."""

    iso_pillar: str
    """'safety' | 'security' | 'functional' — one of the 3 pillars drawn from ISO 25010."""

    owasp_ref: str | None
    """OWASP LLM Top 10 reference if applicable (e.g. 'LLM01'), otherwise None."""

    evaluator: Callable[[DimensionContext], BaseResult]
    """A function taking the context and returning a BaseResult subclass."""

    enabled_check: Callable[[Settings], bool] = field(default=lambda _settings: True)
    """Determines whether this dimension is active for this run (e.g. math_eval.enabled)."""


_REGISTRY: dict[str, Dimension] = {}


def register(dimension: Dimension, *, replace: bool = False) -> None:
    """Adds a dimension to the registry.

    Registering a second time under the same ``key`` raises unless
    ``replace=True`` is given — a silent overwrite would make it unclear
    which definition is active and would make debugging harder.
    """
    if dimension.key in _REGISTRY and not replace:
        raise ValueError(
            f"'{dimension.key}' is already registered. If you intend to "
            "replace it, pass replace=True."
        )
    if dimension.iso_pillar not in {"safety", "security", "functional"}:
        raise ValueError(
            f"invalid iso_pillar: {dimension.iso_pillar!r} "
            "(must be safety | security | functional)"
        )
    _REGISTRY[dimension.key] = dimension


def unregister(key: str) -> None:
    """Removes a dimension from the registry (for test isolation)."""
    _REGISTRY.pop(key, None)


def all_dimensions() -> list[Dimension]:
    """Returns all registered dimensions."""
    return list(_REGISTRY.values())


def get(key: str) -> Dimension | None:
    """Retrieves a single dimension by name."""
    return _REGISTRY.get(key)


def dimension_keys() -> tuple[str, ...]:
    """An unordered tuple of registered dimension keys (for weight mapping)."""
    return tuple(_REGISTRY.keys())


def pillar_membership() -> dict[str, list[str]]:
    """Maps ISO pillar -> the dimension keys belonging to it (for reporting)."""
    grouping: dict[str, list[str]] = {"safety": [], "security": [], "functional": []}
    for dimension in _REGISTRY.values():
        grouping[dimension.iso_pillar].append(dimension.key)
    return grouping


def reset_registry() -> None:
    """Clears the registry entirely (for tests only)."""
    _REGISTRY.clear()


def register_builtin_dimensions() -> None:
    """Registers the platform's seven built-in dimensions.

    This function is idempotent: it does nothing if already registered.
    It is called at engine startup; tests can request a clean registry
    with ``reset_registry()``.
    """
    if _REGISTRY:
        return

    from capability import math_eval
    from core.schemas import PoisoningResult, Status
    from llm_security import (
        content_safety_scan,
        data_poisoning_sim,
        pii_leakage_scan,
        prompt_injection_tests,
    )
    from rag import rag_evaluator
    from rag.retriever import evaluate_retrieval

    register(
        Dimension(
            key="content_safety",
            label="Content safety",
            iso_pillar="safety",
            owasp_ref=None,
            evaluator=lambda ctx: content_safety_scan.scan_records(ctx.records, ctx.settings),
        )
    )
    register(
        Dimension(
            key="injection",
            label="Injection resistance",
            iso_pillar="security",
            owasp_ref="LLM01",
            evaluator=lambda ctx: prompt_injection_tests.run_combined_for_model(
                ctx.model_name, ctx.settings
            ),
        )
    )
    register(
        Dimension(
            key="pii",
            label="PII safety",
            iso_pillar="security",
            owasp_ref="LLM02",
            evaluator=lambda ctx: pii_leakage_scan.scan_records(ctx.records, ctx.settings),
        )
    )
    register(
        Dimension(
            key="poisoning",
            label="Poisoning resistance",
            iso_pillar="security",
            owasp_ref="LLM04",
            evaluator=lambda ctx: (
                PoisoningResult(status=Status.SKIPPED, message="skipped via CLI")
                if ctx.skip_poisoning
                else data_poisoning_sim.simulate(settings=ctx.settings)
            ),
        )
    )
    register(
        Dimension(
            key="retrieval",
            label="Retrieval",
            iso_pillar="functional",
            owasp_ref="LLM08",
            evaluator=lambda ctx: evaluate_retrieval(ctx.retrieval_records, ctx.settings),
        )
    )
    register(
        Dimension(
            key="generation",
            label="Faithfulness",
            iso_pillar="functional",
            owasp_ref="LLM09",
            evaluator=lambda ctx: rag_evaluator.evaluate_generation(ctx.records, ctx.settings),
        )
    )
    register(
        Dimension(
            key="math",
            label="Math",
            iso_pillar="functional",
            owasp_ref=None,
            evaluator=lambda ctx: math_eval.evaluate_model(ctx.model_name, ctx.settings),
            enabled_check=lambda settings: settings.math_eval.enabled,
        )
    )
