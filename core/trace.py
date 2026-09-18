"""Pipeline izleme (trace) katmanı.

Bir RAG hattı zincirdir: soru alınır, bağlam getirilir, prompt kurulur,
cevap üretilir, çıktı filtrelenir. Yalnızca son çıktıyı ölçmek **nerede
bozulduğunu göstermez**: kötü bir cevabın sorumlusu retrieval mı, model
mi, yoksa filtre mi?

Bu modül her aşamayı ayrı ayrı kaydeder:

* süre ve durum (ok / error / skipped)
* girdi ve çıktının **özeti** (ham içerik değil)
* o aşamada tetiklenen guardrail bulguları
* hata durumunda istisna türü ve kısaltılmış mesaj

Tasarım kararları:

**Ham içerik saklanmaz.** Aşama özetleri uzunluk, hash ve isteğe bağlı
redaksiyondan geçmiş kısa bir önizlemeden oluşur. Trace kayıtları
paylaşılabilir artefaktlardır; ham model çıktısını içlerine yazmak, PII
sızıntısını ölçmek yerine çoğaltmak olurdu.

**Kontrol karakterleri temizlenir.** Model çıktısı düşmanca kabul edilir;
önizlemeye giren metin log satırı bölemez (CWE-117).

**core paketi analiz modüllerine bağımlı değildir.** Redaksiyon işlevi
dışarıdan enjekte edilir (``redactor`` parametresi), böylece bağımlılık
yönü tek yönlü kalır.

Kullanım::

    trace = PipelineTrace(pipeline="rag", model_name="gpt4")
    with trace.stage("retrieval") as stage:
        hits = retriever.search(question)
        stage.set_output(hits, count=len(hits))
        stage.add_findings("input_guardrail", injection_findings)
    trace.finish()
"""

from __future__ import annotations

import hashlib
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.logging_setup import get_logger

logger = get_logger(__name__)

PREVIEW_CHARS = 160
MAX_ERROR_CHARS = 500
Redactor = Callable[[str], str]


def _clean(text: str) -> str:
    """Kontrol karakterlerini boşluğa çevirir (CWE-117)."""
    return "".join(" " if ch in "\r\n\t" or ord(ch) < 32 else ch for ch in text)


def summarize(value: Any, redactor: Redactor | None = None) -> dict[str, Any]:
    """Bir aşama girdisini/çıktısını ham içerik saklamadan özetler.

    Hash, aynı girdinin farklı koşularda tekrar edip etmediğini
    karşılaştırmayı sağlar; önizleme ise hata ayıklama içindir.
    """
    if value is None:
        return {"type": "none", "length": 0}

    if isinstance(value, str):
        text = value
        kind = "text"
    elif isinstance(value, (list, tuple)):
        text = "\n".join(str(item) for item in value)
        kind = "sequence"
    elif isinstance(value, dict):
        text = "\n".join(f"{k}={v}" for k, v in value.items())
        kind = "mapping"
    else:
        text = str(value)
        kind = type(value).__name__

    preview = _clean(text[:PREVIEW_CHARS])
    if redactor is not None:
        try:
            preview = redactor(preview)
        except Exception:
            preview = "[redaksiyon basarisiz]"

    summary: dict[str, Any] = {
        "type": kind,
        "length": len(text),
        "sha256_12": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12],
        "preview": preview,
    }
    if isinstance(value, (list, tuple, dict)):
        summary["items"] = len(value)
    return summary


@dataclass
class StageRecord:
    """Tek bir pipeline aşamasının kaydı."""

    name: str
    index: int
    status: str = "ok"
    started_at: str = ""
    duration_sec: float = 0.0
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    findings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""

    @property
    def finding_count(self) -> int:
        """Bu aşamada tetiklenen toplam guardrail bulgusu sayısı."""
        return sum(len(items) for items in self.findings.values())

    def to_dict(self) -> dict[str, Any]:
        """Serileştirilebilir sözlük gösterimi."""
        return {
            "name": self.name,
            "index": self.index,
            "status": self.status,
            "started_at": self.started_at,
            "duration_sec": round(self.duration_sec, 4),
            "input": self.input_summary,
            "output": self.output_summary,
            "findings": self.findings,
            "finding_count": self.finding_count,
            "metrics": self.metrics,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class StageHandle:
    """``with trace.stage(...)`` bloğu içinde kullanılan yazma arayüzü."""

    def __init__(self, record: StageRecord, redactor: Redactor | None) -> None:
        self._record = record
        self._redactor = redactor

    def set_input(self, value: Any, **metrics: float) -> None:
        """Aşamanın girdisini özetler."""
        self._record.input_summary = summarize(value, self._redactor)
        self._record.metrics.update({k: float(v) for k, v in metrics.items()})

    def set_output(self, value: Any, **metrics: float) -> None:
        """Aşamanın çıktısını özetler."""
        self._record.output_summary = summarize(value, self._redactor)
        self._record.metrics.update({k: float(v) for k, v in metrics.items()})

    def add_findings(self, guardrail: str, findings: Sequence[dict[str, Any]]) -> None:
        """Bu aşamada tetiklenen guardrail bulgularını ekler.

        Girdi guardrail'i (kullanıcı sorusundaki injection/küfür) ve çıktı
        guardrail'i (üretilen cevaptaki ihlal) ayrı ayrı kaydedilir; bu
        ayrım olmadan riskin nereden geldiği anlaşılamaz.
        """
        if not findings:
            return
        bucket = self._record.findings.setdefault(guardrail, [])
        bucket.extend(dict(item) for item in findings)

    def add_metrics(self, **metrics: float) -> None:
        """Aşamaya özgü sayısal metrik ekler (ör. getirilen chunk sayısı)."""
        self._record.metrics.update({k: float(v) for k, v in metrics.items()})

    def mark_skipped(self, reason: str = "") -> None:
        """Aşamayı atlanmış olarak işaretler."""
        self._record.status = "skipped"
        self._record.error_message = reason[:MAX_ERROR_CHARS]


class PipelineTrace:
    """Bir sorgunun pipeline boyunca izini tutar."""

    def __init__(
        self,
        pipeline: str,
        model_name: str = "",
        query_id: str = "",
        redactor: Redactor | None = None,
    ) -> None:
        self.trace_id = f"T-{uuid.uuid4().hex[:12]}"
        self.pipeline = pipeline
        self.model_name = model_name
        self.query_id = query_id
        self.created_at = datetime.now(UTC).isoformat()
        self.stages: list[StageRecord] = []
        self._redactor = redactor
        self._start = time.perf_counter()
        self._finished = False
        self.total_duration_sec = 0.0

    @contextmanager
    def stage(self, name: str) -> Iterator[StageHandle]:
        """Bir aşamayı ölçer; istisnayı kaydeder ve yeniden yükseltir.

        İstisna yutulmaz: pipeline'ın gerçekten kırıldığı durumda çağıran
        katman haberdar olmalıdır. Trace yalnızca *kaydeder*.
        """
        record = StageRecord(
            name=name,
            index=len(self.stages),
            started_at=datetime.now(UTC).isoformat(),
        )
        self.stages.append(record)
        handle = StageHandle(record, self._redactor)
        started = time.perf_counter()
        try:
            yield handle
        except Exception as exc:
            record.status = "error"
            record.error_type = type(exc).__name__
            record.error_message = _clean(str(exc))[:MAX_ERROR_CHARS]
            record.duration_sec = time.perf_counter() - started
            logger.warning(
                "pipeline asamasi hata verdi",
                extra={
                    "trace_id": self.trace_id,
                    "stage": name,
                    "error_type": record.error_type,
                    "traceback": traceback.format_exc(limit=3)[-400:],
                },
            )
            raise
        else:
            record.duration_sec = time.perf_counter() - started
            logger.debug(
                "pipeline asamasi tamamlandi",
                extra={
                    "trace_id": self.trace_id,
                    "stage": name,
                    "duration_sec": round(record.duration_sec, 4),
                    "findings": record.finding_count,
                },
            )

    def finish(self) -> PipelineTrace:
        """İzlemeyi kapatır ve toplam süreyi sabitler."""
        if not self._finished:
            self.total_duration_sec = time.perf_counter() - self._start
            self._finished = True
        return self

    # ----------------------------------------------------------------- #
    # Türetilmiş göstergeler
    # ----------------------------------------------------------------- #
    @property
    def failed_stages(self) -> list[str]:
        """Hata veren aşamaların adları."""
        return [stage.name for stage in self.stages if stage.status == "error"]

    @property
    def total_findings(self) -> int:
        """Tüm aşamalardaki guardrail bulgusu sayısı."""
        return sum(stage.finding_count for stage in self.stages)

    def slowest_stage(self) -> str:
        """En uzun süren aşamanın adı (darboğaz tespiti)."""
        if not self.stages:
            return ""
        return max(self.stages, key=lambda stage: stage.duration_sec).name

    def stage_durations(self) -> dict[str, float]:
        """Aşama adı → süre eşlemesi."""
        return {stage.name: round(stage.duration_sec, 4) for stage in self.stages}

    def findings_by_stage(self) -> dict[str, int]:
        """Aşama adı → bulgu sayısı eşlemesi.

        Riskin hangi aşamada ortaya çıktığını gösterir: girdide yakalanan
        bir injection ile çıktıda yakalanan bir ihlal aynı şey değildir.
        """
        return {stage.name: stage.finding_count for stage in self.stages}

    def to_dict(self) -> dict[str, Any]:
        """Serileştirilebilir tam gösterim."""
        return {
            "trace_id": self.trace_id,
            "pipeline": self.pipeline,
            "model_name": self.model_name,
            "query_id": self.query_id,
            "created_at": self.created_at,
            "total_duration_sec": round(self.total_duration_sec, 4),
            "stage_count": len(self.stages),
            "failed_stages": self.failed_stages,
            "total_findings": self.total_findings,
            "slowest_stage": self.slowest_stage(),
            "stages": [stage.to_dict() for stage in self.stages],
        }


def aggregate(traces: Sequence[PipelineTrace]) -> dict[str, Any]:
    """Birden çok izlemeden pipeline sağlık özeti çıkarır.

    Tek bir sorgunun izi hata ayıklama içindir; **toplu görünüm** ise
    pipeline'ın sistematik olarak nerede bozulduğunu ve nerede yavaşladığını
    gösterir. Motor bunu her model için hesaplayıp sonuca yazar.
    """
    if not traces:
        return {
            "traces": 0,
            "stage_count": 0,
            "stage_error_rate": {},
            "stage_avg_duration_sec": {},
            "stage_findings": {},
            "bottleneck": "",
            "total_findings": 0,
        }

    durations: dict[str, list[float]] = {}
    errors: dict[str, int] = {}
    attempts: dict[str, int] = {}
    findings: dict[str, int] = {}

    for trace in traces:
        for stage in trace.stages:
            durations.setdefault(stage.name, []).append(stage.duration_sec)
            attempts[stage.name] = attempts.get(stage.name, 0) + 1
            if stage.status == "error":
                errors[stage.name] = errors.get(stage.name, 0) + 1
            findings[stage.name] = findings.get(stage.name, 0) + stage.finding_count

    average = {
        name: round(sum(values) / len(values), 4) for name, values in durations.items()
    }
    error_rate = {
        name: round(errors.get(name, 0) / count, 4) for name, count in attempts.items()
    }
    bottleneck = max(average, key=lambda name: average[name]) if average else ""

    return {
        "traces": len(traces),
        "stage_count": len(average),
        "failed_stages": sorted(errors),
        "stage_error_rate": error_rate,
        "stage_avg_duration_sec": average,
        "stage_findings": findings,
        "bottleneck": bottleneck,
        "total_findings": sum(findings.values()),
        "avg_total_duration_sec": round(
            sum(trace.total_duration_sec for trace in traces) / len(traces), 4
        ),
    }
