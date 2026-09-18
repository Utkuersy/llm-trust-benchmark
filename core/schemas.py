"""Benchmark sonuç şemaları (Pydantic v2).

Her analiz modülü buradaki bir modeli döndürür; orkestratörler
(``benchmark_engine.py`` / ``llm_benchmark_engine.py``) bunları birleştirip
``core/storage.py`` üzerinden SQLite'a ve MLflow'a yazar.

Tüm alt sonuçlarda ortak sözleşme:
    * ``score``  : 0-100 arası normalize edilmiş alt puan
    * ``status`` : analiz adımının teknik durumu (ok/error/skipped/timeout)
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """Zaman damgası üretir (UTC, timezone-aware)."""
    return datetime.now(UTC)


class Track(StrEnum):
    """Değerlendirme hattı."""

    A = "A"  # Code Trustworthiness
    B = "B"  # LLM/RAG Trustworthiness


class Status(StrEnum):
    """Bir analiz adımının teknik sonucu."""

    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class BaseResult(BaseModel):
    """Tüm alt sonuçların ortak tabanı."""

    model_config = ConfigDict(protected_namespaces=())

    score: float = Field(default=0.0, ge=0.0, le=100.0)
    status: Status = Status.OK
    message: str = ""
    duration_sec: float = 0.0


# --------------------------------------------------------------------------- #
# Track A
# --------------------------------------------------------------------------- #
class CodeQualityResult(BaseResult):
    """Pylint + Radon çıktıları."""

    pylint_score: float = 0.0  # 0-10
    error_count: int = 0
    warning_count: int = 0
    convention_count: int = 0
    refactor_count: int = 0
    average_complexity: float = 0.0
    max_complexity: int = 0
    maintainability_index: float = 0.0
    lines_of_code: int = 0
    files_analyzed: int = 0
    top_issues: list[dict[str, Any]] = Field(default_factory=list)


class CorrectnessResult(BaseResult):
    """pytest çalıştırma sonucu."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    pass_rate: float = 0.0
    failed_tests: list[str] = Field(default_factory=list)


class StaticSecurityResult(BaseResult):
    """Bandit statik güvenlik taraması."""

    high: int = 0
    medium: int = 0
    low: int = 0
    total_findings: int = 0
    findings: list[dict[str, Any]] = Field(default_factory=list)


class DependencySecurityResult(BaseResult):
    """pip-audit bağımlılık taraması."""

    packages_scanned: int = 0
    vulnerable_packages: int = 0
    total_vulnerabilities: int = 0
    findings: list[dict[str, Any]] = Field(default_factory=list)


class RuntimeSecurityResult(BaseResult):
    """Sandbox içinde çalıştırma sonucu."""

    executed: bool = False
    exit_code: int | None = None
    timed_out: bool = False
    memory_exceeded: bool = False
    peak_memory_mb: float = 0.0
    file_write_violations: list[str] = Field(default_factory=list)
    file_read_violations: list[str] = Field(default_factory=list)
    network_attempts: list[str] = Field(default_factory=list)
    subprocess_attempts: list[str] = Field(default_factory=list)
    dynamic_code_events: list[str] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


class TrackAResult(BaseModel):
    """Bir AI model kod klasörü için birleşik Track A sonucu."""

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    model_name: str
    target_path: str
    created_at: datetime = Field(default_factory=utcnow)
    quality: CodeQualityResult = Field(default_factory=CodeQualityResult)
    correctness: CorrectnessResult = Field(default_factory=CorrectnessResult)
    static_security: StaticSecurityResult = Field(default_factory=StaticSecurityResult)
    dependency_security: DependencySecurityResult = Field(
        default_factory=DependencySecurityResult
    )
    runtime_security: RuntimeSecurityResult = Field(default_factory=RuntimeSecurityResult)
    trust_score: float = 0.0
    subscores: dict[str, float] = Field(default_factory=dict)
    scoring_version: str = ""
    weight_preset: str = ""
    weights: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Track B
# --------------------------------------------------------------------------- #
class RetrievalResult(BaseResult):
    """Retrieval kalitesi metrikleri."""

    context_precision: float = 0.0
    context_recall: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    queries_evaluated: int = 0


class GenerationQualityResult(BaseResult):
    """Üretim kalitesi (faithfulness / relevance)."""

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    hallucination_rate: float = 0.0
    answers_evaluated: int = 0
    backend: str = "heuristic"
    worst_examples: list[dict[str, Any]] = Field(default_factory=list)


class InjectionResult(BaseResult):
    """Prompt injection dayanıklılığı."""

    scenarios_run: int = 0
    scenarios_failed: int = 0
    failure_rate: float = 0.0
    resistance_rate: float = 0.0
    failures: list[dict[str, Any]] = Field(default_factory=list)


class PiiLeakageResult(BaseResult):
    """Cevaplarda hassas veri sızıntısı."""

    answers_scanned: int = 0
    total_hits: int = 0
    hits_by_type: dict[str, int] = Field(default_factory=dict)
    examples: list[dict[str, Any]] = Field(default_factory=list)


class PoisoningResult(BaseResult):
    """Veri zehirlenmesi dayanıklılığı."""

    poisoned_docs: int = 0
    queries_run: int = 0
    retrieved_poison: int = 0
    answers_influenced: int = 0
    susceptibility_rate: float = 0.0
    examples: list[dict[str, Any]] = Field(default_factory=list)


class ContentSafetyResult(BaseResult):
    """Zararlı içerik taraması (küfür, hakaret, dini saldırı, tehdit, cinsel)."""

    answers_scanned: int = 0
    flagged_answers: int = 0
    flagged_rate: float = 0.0
    total_hits: int = 0
    hits_by_category: dict[str, int] = Field(default_factory=dict)
    hits_by_severity: dict[str, int] = Field(default_factory=dict)
    active_categories: list[str] = Field(default_factory=list)
    inactive_categories: list[str] = Field(default_factory=list)
    classifier_used: bool = False
    examples: list[dict[str, Any]] = Field(default_factory=list)


class MathEvalResult(BaseResult):
    """Matematik yetenek değerlendirmesi."""

    problems_total: int = 0
    problems_evaluated: int = 0
    correct: int = 0
    accuracy: float = 0.0
    extraction_failures: int = 0
    accuracy_by_category: dict[str, float] = Field(default_factory=dict)
    symbolic_available: bool = False
    failures: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Bir LLM/RAG konfigürasyonu için birleşik değerlendirme sonucu.

    Yedi boyut üç ISO/IEC 25010 karakteristiğine dağılır:
        Safety              -> content_safety
        Security            -> injection, pii, poisoning
        Functional          -> retrieval, generation, math

    ``pipeline`` alanı, aşama bazlı izleme özetini taşır: hangi aşamada
    ne kadar süre geçti, nerede hata oldu, guardrail bulguları hangi
    aşamada tetiklendi.
    """

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    model_name: str
    config_name: str = "default"
    created_at: datetime = Field(default_factory=utcnow)
    retrieval: RetrievalResult = Field(default_factory=RetrievalResult)
    generation: GenerationQualityResult = Field(default_factory=GenerationQualityResult)
    injection: InjectionResult = Field(default_factory=InjectionResult)
    pii: PiiLeakageResult = Field(default_factory=PiiLeakageResult)
    poisoning: PoisoningResult = Field(default_factory=PoisoningResult)
    content_safety: ContentSafetyResult = Field(default_factory=ContentSafetyResult)
    math: MathEvalResult = Field(default_factory=MathEvalResult)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    trust_score: float = 0.0
    subscores: dict[str, float] = Field(default_factory=dict)
    scoring_version: str = ""
    weight_preset: str = ""
    weights: dict[str, float] = Field(default_factory=dict)


# Track A kaldırıldıktan sonra tek değerlendirme hattı kaldı. Eski adı,
# mevcut depolama katmanını kırmamak için takma ad olarak korunuyor.
TrackBResult = EvaluationResult
