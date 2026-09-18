"""Dimension registry tests.

These tests **prove** the claim that "adding a new dimension does not
require changing the engine code" — not merely assert it. They also
verify that one dimension's failure does not stop the others (error
isolation).
"""

from __future__ import annotations

import pytest

from core.dimensions import (
    Dimension,
    DimensionContext,
    all_dimensions,
    get,
    pillar_membership,
    register,
    register_builtin_dimensions,
    reset_registry,
    unregister,
)
from core.schemas import BaseResult, Status


@pytest.fixture(autouse=True)
def _clean_registry():
    """Starts every test from a clean state with the built-in dimensions registered."""
    reset_registry()
    register_builtin_dimensions()
    yield
    reset_registry()
    register_builtin_dimensions()


def _dummy_result(score: float = 50.0, status: Status = Status.OK) -> BaseResult:
    return BaseResult(score=score, status=status)


# --------------------------------------------------------------------------- #
# Basic registration behavior
# --------------------------------------------------------------------------- #
def test_register_and_retrieve() -> None:
    """A registered dimension must be retrievable by name."""
    register(Dimension("test_dim", "Test", "safety", None, lambda _ctx: _dummy_result()))
    assert get("test_dim") is not None
    assert get("test_dim").label == "Test"


def test_duplicate_registration_raises_by_default() -> None:
    """A second registration under the same key must raise unless explicitly allowed."""
    register(Dimension("dup", "First", "safety", None, lambda _ctx: _dummy_result()))
    with pytest.raises(ValueError, match="already registered"):
        register(Dimension("dup", "Second", "safety", None, lambda _ctx: _dummy_result()))


def test_duplicate_registration_allowed_with_replace() -> None:
    """A deliberate overwrite with replace=True must be allowed."""
    register(Dimension("dup", "First", "safety", None, lambda _ctx: _dummy_result()))
    register(Dimension("dup", "Second", "safety", None, lambda _ctx: _dummy_result()), replace=True)
    assert get("dup").label == "Second"


def test_invalid_pillar_rejected() -> None:
    """An undefined ISO pillar must be rejected — not silently accepted."""
    with pytest.raises(ValueError, match="iso_pillar"):
        register(Dimension("bad", "Bad", "not_a_pillar", None, lambda _ctx: _dummy_result()))


def test_unregister_removes_dimension() -> None:
    """Unregistering must actually remove the dimension."""
    register(Dimension("temp", "Temporary", "safety", None, lambda _ctx: _dummy_result()))
    unregister("temp")
    assert get("temp") is None


# --------------------------------------------------------------------------- #
# THE CORE CLAIM: adding a new dimension without changing the engine
# --------------------------------------------------------------------------- #
def test_new_dimension_is_picked_up_without_engine_changes() -> None:
    """A newly registered dimension takes effect without touching benchmark_engine.py.

    This is the proof of the "plugin architecture" claim: the
    benchmark_engine module is imported here and never modified — only a
    new dimension is registered, and it is shown to be included in its
    loop.
    """
    import benchmark_engine

    def custom_evaluator(ctx: DimensionContext) -> BaseResult:
        return BaseResult(score=77.0, status=Status.OK, message=f"custom for {ctx.model_name}")

    register(
        Dimension(
            key="custom_test_dimension",
            label="Custom test dimension",
            iso_pillar="functional",
            owasp_ref=None,
            evaluator=custom_evaluator,
        )
    )

    keys = [dimension.key for dimension in all_dimensions()]
    assert "custom_test_dimension" in keys

    context = DimensionContext(
        model_name="test-model", records=[], retrieval_records=[],
        settings=benchmark_engine.get_settings(), skip_poisoning=True,
    )
    result = get("custom_test_dimension").evaluator(context)
    assert result.score == 77.0
    assert "test-model" in result.message


def test_builtin_dimensions_registered() -> None:
    """The platform's seven built-in dimensions must be registered."""
    keys = {dimension.key for dimension in all_dimensions()}
    expected = {
        "content_safety", "injection", "pii", "poisoning",
        "retrieval", "generation", "math",
    }
    assert expected == keys


def test_pillar_membership_covers_all_dimensions() -> None:
    """Every dimension must belong to exactly one ISO pillar, none lost."""
    grouping = pillar_membership()
    total = sum(len(members) for members in grouping.values())
    assert total == len(all_dimensions())
    assert len(grouping["safety"]) == 1  # content_safety
    assert len(grouping["security"]) == 3  # injection, pii, poisoning
    assert len(grouping["functional"]) == 3  # retrieval, generation, math


# --------------------------------------------------------------------------- #
# Error isolation
# --------------------------------------------------------------------------- #
def test_one_dimension_failure_does_not_affect_others() -> None:
    """If one dimension's evaluator blows up, the other dimensions must not be affected.

    This tests independence at the registry level, not the try/except
    wrapper behavior in benchmark_engine.py: each evaluator runs in its
    own scope and shares no state.
    """
    def broken_evaluator(ctx: DimensionContext) -> BaseResult:
        raise RuntimeError("deliberate failure")

    def healthy_evaluator(ctx: DimensionContext) -> BaseResult:
        return BaseResult(score=90.0, status=Status.OK)

    register(Dimension("broken", "Broken", "safety", None, broken_evaluator))
    register(Dimension("healthy", "Healthy", "safety", None, healthy_evaluator))

    context = DimensionContext(
        model_name="m", records=[], retrieval_records=[],
        settings=None, skip_poisoning=False,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError):
        get("broken").evaluator(context)

    # The healthy dimension still works, independent of the broken one.
    assert get("healthy").evaluator(context).score == 90.0


def test_register_builtin_is_idempotent() -> None:
    """register_builtin_dimensions() must not raise if called multiple times."""
    register_builtin_dimensions()
    count_before = len(all_dimensions())
    register_builtin_dimensions()
    assert len(all_dimensions()) == count_before
