"""LLM çıktı güvenilirliği değerlendirme motoru.

``llm_outputs/<model>/`` altındaki her model için yedi boyut ölçülür ve
ağırlıklı bir Trust Score (0-100) hesaplanır:

    Safety      (ISO 25010)  -> content_safety
    Security    (OWASP LLM)  -> injection, pii, poisoning
    Functional  (ISO 25010)  -> retrieval, generation, math

Ağırlıklar elle yazılmaz; ``core/scoring.py`` içindeki, her biri bir
kaynağa bağlanmış ön ayarlardan seçilir. Kullanılan ön ayarın adı ve
puanlama şeması sürümü her sonuç kaydına yazılır (izlenebilirlik).

**Pipeline izleme.** Her soru kaydı için bir ``PipelineTrace`` üretilir:
girdi guardrail'i, retrieval, üretim ve çıktı guardrail'i ayrı aşamalar
olarak ölçülür. Böylece "cevap kötü" demek yerine *hangi aşamada* bozulduğu
söylenebilir — girdide yakalanan bir injection ile modelin kendi ürettiği
bir ihlal farklı risklerdir ve farklı aksiyon gerektirir.

**Atlanan boyut cezalandırılmaz.** Bir analiz adımı çalışamazsa (sözlük
boş, matematik cevap dosyası yok) o boyutun ağırlığı paydadan düşülür ve
kalan boyutlara orantılı dağıtılır.

CLI::

    python benchmark_engine.py
    python benchmark_engine.py --models gemini --preset owasp_rank
    python benchmark_engine.py --skip-poisoning --no-trace
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Sequence

from capability import math_eval
from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.schemas import (
    EvaluationResult,
    MathEvalResult,
    PoisoningResult,
    RetrievalResult,
    Status,
    Track,
)
from core.scoring import SCORING_VERSION, aggregate as weighted_score, resolve_preset
from core.storage import init_db, save_track_b
from core.trace import PipelineTrace, aggregate as aggregate_traces
from core.tracking import ExperimentTracker, get_tracker
from llm_security import (
    content_safety_scan,
    data_poisoning_sim,
    pii_leakage_scan,
    prompt_injection_tests,
)
from rag import rag_evaluator
from rag.retriever import RetrievalRecord, evaluate_retrieval
from rag.vector_store import SearchHit

logger = get_logger(__name__)

from core.dimensions import DimensionContext, all_dimensions, register_builtin_dimensions

register_builtin_dimensions()
DIMENSIONS = tuple(dimension.key for dimension in all_dimensions())


# --------------------------------------------------------------------------- #
# Kayıtlardan retrieval değerlendirme formatı
# --------------------------------------------------------------------------- #
def _infer_source(context: str, expected: Sequence[str]) -> str:
    """Bağlam metninden kaynak dosya adını tahmin eder (başlık eşleşmesi)."""
    head = context.strip().splitlines()[0].lower() if context.strip() else ""
    for candidate in expected:
        stem = Path(candidate).stem.replace("_", " ").lower()
        if stem and any(token in head for token in stem.split()):
            return candidate
    return "unknown"


def records_to_retrieval(records: Sequence[dict[str, Any]]) -> list[RetrievalRecord]:
    """Kayıtlı cevapları retrieval değerlendirme formatına çevirir."""
    output: list[RetrievalRecord] = []
    for record in records:
        expected = list(record.get("expected_sources", []))
        sources = record.get("context_sources")
        hits: list[SearchHit] = []
        for index, context in enumerate(record.get("contexts", [])):
            if isinstance(sources, list) and index < len(sources) and sources[index]:
                source = str(sources[index])
            else:
                source = _infer_source(context, expected)
            hits.append(
                SearchHit(
                    doc_id=f"ctx-{index}",
                    text=context,
                    source=source,
                    score=1.0 - index * 0.01,
                )
            )
        output.append(
            RetrievalRecord(
                question=record.get("question", ""),
                hits=hits,
                expected_sources=expected,
            )
        )
    return output


# --------------------------------------------------------------------------- #
# Pipeline izleme
# --------------------------------------------------------------------------- #
def build_traces(
    records: Sequence[dict[str, Any]], model_name: str, settings: Settings
) -> list[PipelineTrace]:
    """Her soru kaydı için aşama bazlı pipeline izi üretir.

    Kayıtlı çıktılar üzerinden çalışıldığı için aşamalar yeniden
    *çalıştırılmaz*; gözlemlenebilir olanlar ölçülür:

        input_guardrail   — sorudaki zararlı içerik / PII izleri
        retrieval         — bağlam getirildi mi, kaç parça
        generation        — cevap üretildi mi, uzunluğu
        output_guardrail  — cevaptaki içerik ihlali ve PII sızıntısı

    Bu ayrım, riskin kullanıcıdan mı yoksa modelden mi geldiğini gösterir.
    """
    lexicon_dir = settings.content_safety.lexicon_dir
    if not lexicon_dir.is_absolute():
        lexicon_dir = PROJECT_ROOT / lexicon_dir
    lexicons = [
        lexicon
        for lexicon in content_safety_scan.load_lexicons(
            lexicon_dir, settings.content_safety.category_severity
        )
        if len(lexicon) > 0
    ]

    traces: list[PipelineTrace] = []
    for index, record in enumerate(records):
        trace = PipelineTrace(
            pipeline="rag",
            model_name=model_name,
            query_id=f"Q-{index + 1:03d}",
            redactor=pii_leakage_scan.redact,
        )
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))
        contexts = list(record.get("contexts", []))

        with trace.stage("input_guardrail") as stage:
            stage.set_input(question)
            stage.add_findings(
                "content_safety", content_safety_scan.scan_text(question, lexicons)
            )
            stage.add_findings("pii", pii_leakage_scan.scan_text(question))

        with trace.stage("retrieval") as stage:
            stage.set_output(contexts, retrieved=len(contexts))
            if not contexts:
                stage.mark_skipped("kayitta baglam yok")

        with trace.stage("generation") as stage:
            stage.set_output(answer, answer_chars=len(answer))
            if not answer.strip():
                stage.mark_skipped("bos cevap")

        with trace.stage("output_guardrail") as stage:
            stage.add_findings(
                "content_safety", content_safety_scan.scan_text(answer, lexicons)
            )
            stage.add_findings("pii", pii_leakage_scan.scan_text(answer))

        traces.append(trace.finish())
    return traces


# --------------------------------------------------------------------------- #
# Puanlama
# --------------------------------------------------------------------------- #
def compute_trust_score(
    result: EvaluationResult, settings: Settings, preset: str
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Ağırlıklı Trust Score, alt puanlar ve kullanılan ağırlıkları döndürür."""
    weights = resolve_preset("B", preset)
    components = {name: getattr(result, name) for name in DIMENSIONS}

    subscores = {
        name: round(float(component.score), 2) for name, component in components.items()
    }
    usable = [name for name, component in components.items() if component.status is Status.OK]

    missing = [name for name in usable if name not in weights]
    if missing:
        logger.warning(
            "on ayarda agirligi tanimsiz boyut var; puana katkisi olmayacak",
            extra={"preset": preset, "dimensions": missing},
        )

    trust = weighted_score(subscores, weights, usable)
    return round(trust, 2), subscores, weights


def collect_findings(result: EvaluationResult) -> list[dict[str, Any]]:
    """Alt sonuçlardan ortak bulgu kayıtları üretir."""
    findings: list[dict[str, Any]] = []

    for example in result.content_safety.examples:
        findings.append(
            {
                "category": "content_safety",
                "severity": example.get("severity", "HIGH"),
                "title": f"Zararli icerik: {example.get('category')}",
                "detail": f"{example.get('matched_root', '')} — {example.get('context', '')[:200]}",
                "location": example.get("question", ""),
            }
        )
    for failure in result.injection.failures:
        findings.append(
            {
                "category": "prompt_injection",
                "severity": failure.get("severity", "MEDIUM"),
                "title": f"{failure.get('scenario_id')} {failure.get('name')}",
                "detail": failure.get("reason", ""),
                "location": failure.get("category", ""),
            }
        )
    for example in result.pii.examples:
        findings.append(
            {
                "category": "pii_leakage",
                "severity": example.get("severity", "MEDIUM"),
                "title": f"PII sizintisi: {example.get('type')}",
                "detail": f"{example.get('masked_value')} — {example.get('context', '')[:200]}",
                "location": example.get("question", ""),
            }
        )
    for example in result.poisoning.examples:
        if example.get("answer_influenced"):
            findings.append(
                {
                    "category": "data_poisoning",
                    "severity": "HIGH",
                    "title": f"Zehirli iddia benimsendi: {example.get('case_id')}",
                    "detail": str(example.get("matched_claims")),
                    "location": example.get("query", ""),
                }
            )
    for example in result.generation.worst_examples:
        if example.get("faithfulness", 1.0) < 0.5:
            findings.append(
                {
                    "category": "hallucination",
                    "severity": "MEDIUM",
                    "title": "Dusuk faithfulness",
                    "detail": example.get("answer", "")[:300],
                    "location": example.get("question", ""),
                }
            )
    for failure in result.math.failures:
        findings.append(
            {
                "category": "math_error",
                "severity": "LOW" if failure.get("method") == "extraction_failed" else "MEDIUM",
                "title": f"Matematik hatasi: {failure.get('id')}",
                "detail": f"beklenen={failure.get('expected')} bulunan={failure.get('extracted')}",
                "location": failure.get("category", ""),
            }
        )

    for stage, count in (result.pipeline.get("stage_findings") or {}).items():
        if count:
            findings.append(
                {
                    "category": "pipeline",
                    "severity": "INFO",
                    "title": f"Asama bazli bulgu: {stage}",
                    "detail": f"{count} bulgu bu asamada tetiklendi",
                    "location": stage,
                }
            )
    return findings


# --------------------------------------------------------------------------- #
# Orkestrasyon
# --------------------------------------------------------------------------- #
def discover_models(settings: Settings) -> list[Path]:
    """``llm_outputs/`` altındaki model klasörlerini bulur."""
    root = settings.paths.absolute(settings.paths.llm_outputs_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith((".", "_"))
        and any(p.suffix.lower() in {".json", ".jsonl"} for p in path.rglob("*"))
    )


def benchmark_model(
    model_dir: Path,
    settings: Settings,
    *,
    preset: str,
    skip_poisoning: bool = False,
    with_trace: bool = True,
) -> EvaluationResult:
    """Tek bir model için yedi boyutu ölçer."""
    model_name = model_dir.name
    run_id = f"E-{model_name}-{uuid.uuid4().hex[:10]}"
    logger.info("degerlendirme basladi", extra={"model": model_name, "run_id": run_id})

    result = EvaluationResult(run_id=run_id, model_name=model_name, config_name=preset)
    records = rag_evaluator.load_llm_outputs(model_dir)

    if not records:
        result.retrieval = RetrievalResult(status=Status.ERROR, message="cevap dosyasi yok")
        result.trust_score, result.subscores, result.weights = compute_trust_score(
            result, settings, preset
        )
        result.scoring_version = SCORING_VERSION
        result.weight_preset = preset
        return result

    context = DimensionContext(
        model_name=model_name,
        records=records,
        retrieval_records=records_to_retrieval(records),
        settings=settings,
        skip_poisoning=skip_poisoning,
    )
    for dimension in all_dimensions():
        if not dimension.enabled_check(settings):
            setattr(
                result, dimension.key,
                type(getattr(result, dimension.key))(status=Status.SKIPPED, message="devre disi"),
            )
            continue
        try:
            setattr(result, dimension.key, dimension.evaluator(context))
        except Exception as exc:  # noqa: BLE001 - bir boyutun hatasi digerlerini durdurmasin
            logger.exception(
                "boyut degerlendirmesi basarisiz",
                extra={"dimension": dimension.key, "model": model_name, "error": str(exc)},
            )
            error_type = type(getattr(result, dimension.key))
            setattr(result, dimension.key, error_type(status=Status.ERROR, message=str(exc)[:300]))

    if with_trace:
        traces = build_traces(records, model_name, settings)
        result.pipeline = aggregate_traces(traces)
        result.pipeline["sample_trace"] = traces[0].to_dict() if traces else {}

    result.trust_score, result.subscores, result.weights = compute_trust_score(
        result, settings, preset
    )
    result.scoring_version = SCORING_VERSION
    result.weight_preset = preset

    logger.info(
        "degerlendirme tamamlandi",
        extra={
            "model": model_name,
            "trust_score": result.trust_score,
            "bottleneck": result.pipeline.get("bottleneck", ""),
            **result.subscores,
        },
    )
    return result


def persist(result: EvaluationResult, tracker: ExperimentTracker, settings: Settings) -> None:
    """Sonucu MLflow'a loglar, SQLite'a, results/ altına yazar ve denetim izine kaydeder."""
    from core.audit import record_run
    from core.versioning import dataset_version

    record_run(
        run_id=result.run_id,
        preset=result.weight_preset,
        dataset_version=dataset_version(settings),
        extra={"model_name": result.model_name, "trust_score": result.trust_score},
        settings=settings,
    )

    payload = json.loads(result.model_dump_json())
    mlflow_run_id: str | None = None

    with tracker.start_run(f"eval-{result.model_name}", Track.B) as run_id:
        mlflow_run_id = run_id
        tracker.set_tags(
            {
                "model": result.model_name,
                "preset": result.weight_preset,
                "scoring_version": result.scoring_version,
            }
        )
        tracker.log_params(
            {
                "model_name": result.model_name,
                "weight_preset": result.weight_preset,
                "scoring_version": result.scoring_version,
                "eval_backend": result.generation.backend,
                "classifier_used": result.content_safety.classifier_used,
                "top_k": settings.rag.top_k,
            }
        )
        metrics: dict[str, Any] = {"trust_score": result.trust_score}
        metrics.update({f"score_{k}": v for k, v in result.subscores.items()})
        metrics.update(
            {
                "content_flagged_rate": result.content_safety.flagged_rate,
                "content_hits": result.content_safety.total_hits,
                "injection_failure_rate": result.injection.failure_rate,
                "pii_hits": result.pii.total_hits,
                "poisoning_susceptibility": result.poisoning.susceptibility_rate,
                "faithfulness": result.generation.faithfulness,
                "hallucination_rate": result.generation.hallucination_rate,
                "context_precision": result.retrieval.context_precision,
                "math_accuracy": result.math.accuracy,
                "pipeline_findings": result.pipeline.get("total_findings", 0),
            }
        )
        tracker.log_metrics(metrics)
        tracker.log_json_artifact(payload, f"evaluation_{result.model_name}.json")

    save_track_b(result, collect_findings(result), mlflow_run_id, settings)

    results_dir = settings.paths.absolute(settings.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{result.run_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def run_benchmark(
    models: list[str] | None = None,
    *,
    preset: str | None = None,
    skip_poisoning: bool = False,
    with_trace: bool = True,
    settings: Settings | None = None,
) -> list[EvaluationResult]:
    """Değerlendirmeyi seçili (veya tüm) modeller için çalıştırır."""
    settings = settings or get_settings()
    settings.offline.apply()
    init_db(settings)
    tracker = get_tracker(settings)
    preset = preset or settings.scoring.track_b_preset

    directories = discover_models(settings)
    if models:
        wanted = {name.lower() for name in models}
        directories = [d for d in directories if d.name.lower() in wanted]
    if not directories:
        logger.error(
            "llm_outputs altinda model bulunamadi; once "
            "python -m scripts.generate_llm_outputs calistirin"
        )
        return []

    results: list[EvaluationResult] = []
    for model_dir in directories:
        try:
            result = benchmark_model(
                model_dir,
                settings,
                preset=preset,
                skip_poisoning=skip_poisoning,
                with_trace=with_trace,
            )
            persist(result, tracker, settings)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - tek model hatasi kosuyu durdurmasin
            logger.exception(
                "model degerlendirmesi basarisiz",
                extra={"model": model_dir.name, "error": str(exc)},
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM cikti guvenilirligi degerlendirmesi")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--preset", default=None, help="Agirlik on ayari (core/scoring.py)")
    parser.add_argument("--skip-poisoning", action="store_true")
    parser.add_argument("--no-trace", action="store_true", help="Pipeline izlemeyi kapat")
    parser.add_argument("--config", default=None, help="Alternatif settings.yaml yolu")
    args = parser.parse_args()

    if args.config:
        os.environ["AITB_CONFIG_PATH"] = args.config

    results = run_benchmark(
        models=args.models,
        preset=args.preset,
        skip_poisoning=args.skip_poisoning,
        with_trace=not args.no_trace,
    )
    if not results:
        return

    preset = results[0].weight_preset
    print(f"\n=== Trust Score ozeti (on ayar: {preset}, v{SCORING_VERSION}) ===")  # noqa: T201
    for result in sorted(results, key=lambda r: r.trust_score, reverse=True):
        parts = " ".join(f"{k}={v:.0f}" for k, v in result.subscores.items())
        print(f"{result.model_name:<10} {result.trust_score:6.2f}   {parts}")  # noqa: T201

    print("\n=== Pipeline: asama bazli bulgu ===")  # noqa: T201
    for result in results:
        stages = result.pipeline.get("stage_findings", {})
        if stages:
            summary = " ".join(f"{k}={v}" for k, v in stages.items())
            print(f"{result.model_name:<10} {summary}")  # noqa: T201


if __name__ == "__main__":
    main()
