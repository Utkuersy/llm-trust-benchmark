"""Boyut kayıt sistemi (dimension registry).

**Çözülen problem.** Yedi boyut motorun içine gömülüydü:
``benchmark_model()`` her boyutu ismen çağırıyordu, ağırlıklandırma ayrı bir
sabit listeyle (``DIMENSIONS``) senkron tutulmak zorundaydı. Sekizinci bir
boyut eklemek dört yerde değişiklik gerektiriyordu: motor, ağırlık listesi,
depolama, dashboard.

**Çözüm.** Her boyut kendini bu kayda **kaydeder** (tek çağrı):

    register(Dimension(
        key="content_safety",
        label="İçerik güvenliği",
        iso_pillar="safety",
        owasp_ref=None,
        evaluator=lambda ctx: content_safety_scan.scan_records(ctx.records, ctx.settings),
    ))

Motor, kayıtlı boyutların listesini döngüyle çalıştırır
(``for dim in all_dimensions(): setattr(result, dim.key, dim.evaluator(ctx))``).
Yeni bir boyut eklemek artık şu demektir: bir değerlendirme fonksiyonu yaz,
tek ``register()`` çağrısı ekle, ``EvaluationResult`` şemasına bir alan ekle.
Motor kodu değişmez.

**Sınır.** Bu Python; "gerçek" bir plugin sistemi (dinamik keşif,
sürüm uyumluluğu kontrolü, sandbox izolasyonu) değildir. Amaç, sabit
kod tekrarını ortadan kaldırmak ve tek bir doğruluk kaynağı (single
source of truth) kurmaktır — genel amaçlı bir eklenti pazarı değil.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core.config import Settings
from core.schemas import BaseResult
from rag.retriever import RetrievalRecord


@dataclass
class DimensionContext:
    """Bir değerlendiricinin ihtiyaç duyabileceği her şeyi tek yerde toplar.

    Tüm değerlendiriciler aynı imzayı paylaşır (``Callable[[DimensionContext],
    BaseResult]``); ihtiyaç duymadıkları alanları yok sayarlar. Bu, kayıt
    arayüzünü tek tip tutar — bazı boyutlar ``records`` alır, bazıları
    ``model_name``, bazıları hiçbirini almaz.
    """

    model_name: str
    records: list[dict]
    retrieval_records: list[RetrievalRecord]
    settings: Settings
    skip_poisoning: bool = False


@dataclass
class Dimension:
    """Tek bir ölçüm boyutunun kayıt bilgisi."""

    key: str
    """EvaluationResult şemasındaki alan adıyla birebir eşleşmeli (ör. 'content_safety')."""

    label: str
    """İnsan tarafından okunabilir ad (dashboard ve raporlarda kullanılır)."""

    iso_pillar: str
    """'safety' | 'security' | 'functional' — ISO 25010'dan seçilen 3 sütundan biri."""

    owasp_ref: str | None
    """İlgiliyse OWASP LLM Top 10 referansı (ör. 'LLM01'), değilse None."""

    evaluator: Callable[[DimensionContext], BaseResult]
    """Bağlamı alıp bir BaseResult alt sınıfı döndüren fonksiyon."""

    enabled_check: Callable[[Settings], bool] = field(default=lambda _settings: True)
    """Bu boyutun bu koşuda aktif olup olmadığını belirler (ör. math_eval.enabled)."""


_REGISTRY: dict[str, Dimension] = {}


def register(dimension: Dimension, *, replace: bool = False) -> None:
    """Bir boyutu kayda ekler.

    Aynı ``key`` ile ikinci kez kayıt, ``replace=True`` verilmedikçe hataya
    yol açar — sessiz üzerine yazma, hangi tanımın etkin olduğunu
    belirsizleştirir ve hata ayıklamayı zorlaştırır.
    """
    if dimension.key in _REGISTRY and not replace:
        raise ValueError(
            f"'{dimension.key}' zaten kayıtlı. Kasıtlı değiştiriyorsan "
            "replace=True ver."
        )
    if dimension.iso_pillar not in {"safety", "security", "functional"}:
        raise ValueError(
            f"gecersiz iso_pillar: {dimension.iso_pillar!r} "
            "(safety | security | functional olmali)"
        )
    _REGISTRY[dimension.key] = dimension


def unregister(key: str) -> None:
    """Bir boyutu kayıttan çıkarır (testlerde izolasyon için)."""
    _REGISTRY.pop(key, None)


def all_dimensions() -> list[Dimension]:
    """Kayıtlı tüm boyutları döndürür."""
    return list(_REGISTRY.values())


def get(key: str) -> Dimension | None:
    """Tek bir boyutu adıyla getirir."""
    return _REGISTRY.get(key)


def dimension_keys() -> tuple[str, ...]:
    """Kayıtlı boyut anahtarlarının sırasız listesi (ağırlık eşleme için)."""
    return tuple(_REGISTRY.keys())


def pillar_membership() -> dict[str, list[str]]:
    """ISO sütunu -> o sütundaki boyut anahtarları eşlemesi (raporlama için)."""
    grouping: dict[str, list[str]] = {"safety": [], "security": [], "functional": []}
    for dimension in _REGISTRY.values():
        grouping[dimension.iso_pillar].append(dimension.key)
    return grouping


def reset_registry() -> None:
    """Kaydı tamamen boşaltır (yalnızca testler için)."""
    _REGISTRY.clear()


def register_builtin_dimensions() -> None:
    """Platformun yedi yerleşik boyutunu kaydeder.

    Bu fonksiyon idempotenttir: zaten kayıtlıysa tekrar kaydetmez. Motor
    başlangıcında çağrılır; testler ``reset_registry()`` ile temiz bir
    kayıt isteyebilir.
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
            label="İçerik güvenliği",
            iso_pillar="safety",
            owasp_ref=None,
            evaluator=lambda ctx: content_safety_scan.scan_records(ctx.records, ctx.settings),
        )
    )
    register(
        Dimension(
            key="injection",
            label="Injection direnci",
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
            label="PII güvenliği",
            iso_pillar="security",
            owasp_ref="LLM02",
            evaluator=lambda ctx: pii_leakage_scan.scan_records(ctx.records, ctx.settings),
        )
    )
    register(
        Dimension(
            key="poisoning",
            label="Zehirlenme direnci",
            iso_pillar="security",
            owasp_ref="LLM04",
            evaluator=lambda ctx: (
                PoisoningResult(status=Status.SKIPPED, message="CLI ile atlandi")
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
            label="Matematik",
            iso_pillar="functional",
            owasp_ref=None,
            evaluator=lambda ctx: math_eval.evaluate_model(ctx.model_name, ctx.settings),
            enabled_check=lambda settings: settings.math_eval.enabled,
        )
    )
