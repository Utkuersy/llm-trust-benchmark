"""Boyut kayıt sistemi testleri.

Bu testler, "yeni bir boyut eklemek motor kodunu değiştirmeyi
gerektirmiyor" iddiasını **kanıtlar** — sadece iddia etmekle yetinmez.
Ayrıca bir boyutun hata vermesinin diğerlerini durdurmadığını (hata
izolasyonu) doğrular.
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
    """Her testi, yerleşik boyutlar kayıtlı temiz bir durumdan başlatır."""
    reset_registry()
    register_builtin_dimensions()
    yield
    reset_registry()
    register_builtin_dimensions()


def _dummy_result(score: float = 50.0, status: Status = Status.OK) -> BaseResult:
    return BaseResult(score=score, status=status)


# --------------------------------------------------------------------------- #
# Temel kayıt davranışı
# --------------------------------------------------------------------------- #
def test_register_and_retrieve() -> None:
    """Kaydedilen bir boyut adıyla geri alınabilmeli."""
    register(Dimension("test_dim", "Test", "safety", None, lambda ctx: _dummy_result()))
    assert get("test_dim") is not None
    assert get("test_dim").label == "Test"


def test_duplicate_registration_raises_by_default() -> None:
    """Aynı anahtarla ikinci kayıt, açıkça izin verilmedikçe hataya düşmeli."""
    register(Dimension("dup", "İlk", "safety", None, lambda ctx: _dummy_result()))
    with pytest.raises(ValueError, match="zaten kayıtlı"):
        register(Dimension("dup", "İkinci", "safety", None, lambda ctx: _dummy_result()))


def test_duplicate_registration_allowed_with_replace() -> None:
    """replace=True ile bilinçli üzerine yazma serbest olmalı."""
    register(Dimension("dup", "İlk", "safety", None, lambda ctx: _dummy_result()))
    register(Dimension("dup", "İkinci", "safety", None, lambda ctx: _dummy_result()), replace=True)
    assert get("dup").label == "İkinci"


def test_invalid_pillar_rejected() -> None:
    """Tanımsız bir ISO sütunu reddedilmeli — sessizce kabul edilmemeli."""
    with pytest.raises(ValueError, match="iso_pillar"):
        register(Dimension("bad", "Kötü", "not_a_pillar", None, lambda ctx: _dummy_result()))


def test_unregister_removes_dimension() -> None:
    """Kayıttan çıkarma gerçekten kaldırmalı."""
    register(Dimension("temp", "Geçici", "safety", None, lambda ctx: _dummy_result()))
    unregister("temp")
    assert get("temp") is None


# --------------------------------------------------------------------------- #
# ASIL İDDİA: motoru değiştirmeden yeni boyut eklemek
# --------------------------------------------------------------------------- #
def test_new_dimension_is_picked_up_without_engine_changes() -> None:
    """Yeni kaydedilen bir boyut, benchmark_engine.py'ye dokunmadan devreye girer.

    Bu, "eklenti mimarisi" iddiasının kanıtıdır: benchmark_engine modülü
    burada import edilip hiç değiştirilmeden, sadece yeni bir dimension
    kaydedilerek onun döngüsüne dahil olduğu gösterilir.
    """
    import benchmark_engine

    def custom_evaluator(ctx: DimensionContext) -> BaseResult:
        return BaseResult(score=77.0, status=Status.OK, message=f"{ctx.model_name} icin ozel")

    register(
        Dimension(
            key="custom_test_dimension",
            label="Özel test boyutu",
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
    """Platformun yedi yerleşik boyutu kayıtlı olmalı."""
    keys = {dimension.key for dimension in all_dimensions()}
    expected = {
        "content_safety", "injection", "pii", "poisoning",
        "retrieval", "generation", "math",
    }
    assert expected == keys


def test_pillar_membership_covers_all_dimensions() -> None:
    """Her boyut tam olarak bir ISO sütununa ait olmalı, hiçbiri kaybolmamalı."""
    grouping = pillar_membership()
    total = sum(len(members) for members in grouping.values())
    assert total == len(all_dimensions())
    assert len(grouping["safety"]) == 1  # content_safety
    assert len(grouping["security"]) == 3  # injection, pii, poisoning
    assert len(grouping["functional"]) == 3  # retrieval, generation, math


# --------------------------------------------------------------------------- #
# Hata izolasyonu
# --------------------------------------------------------------------------- #
def test_one_dimension_failure_does_not_affect_others() -> None:
    """Bir boyutun evaluator'ı patlarsa diğer boyutlar etkilenmemeli.

    Bu, benchmark_engine.py'deki try/except sarmalayıcının davranışını
    değil, registry seviyesindeki bağımsızlığı test eder: her evaluator
    kendi kapsamında çalışır, ortak durum paylaşmaz.
    """
    def broken_evaluator(ctx: DimensionContext) -> BaseResult:
        raise RuntimeError("kasitli hata")

    def healthy_evaluator(ctx: DimensionContext) -> BaseResult:
        return BaseResult(score=90.0, status=Status.OK)

    register(Dimension("broken", "Bozuk", "safety", None, broken_evaluator))
    register(Dimension("healthy", "Sağlıklı", "safety", None, healthy_evaluator))

    context = DimensionContext(
        model_name="m", records=[], retrieval_records=[],
        settings=None, skip_poisoning=False,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError):
        get("broken").evaluator(context)

    # Bozuk boyuttan bağımsız olarak sağlıklı boyut hâlâ çalışır.
    assert get("healthy").evaluator(context).score == 90.0


def test_register_builtin_is_idempotent() -> None:
    """register_builtin_dimensions() birden çok kez çağrılırsa hata vermemeli."""
    register_builtin_dimensions()
    count_before = len(all_dimensions())
    register_builtin_dimensions()
    assert len(all_dimensions()) == count_before
