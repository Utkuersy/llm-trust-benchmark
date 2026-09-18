"""Pipeline izleme katmanı testleri.

Trace katmanı, sonuçların *nerede* bozulduğunu söylemekle yükümlüdür.
Bu yüzden testler üç şeyi garanti eder:

1. Aşama sırası, süresi ve durumu doğru kaydedilir
2. Hata yutulmaz — kaydedilir ve yeniden yükseltilir
3. Ham içerik ve kontrol karakteri trace'e sızmaz (CWE-117 ve veri sızıntısı)
"""

from __future__ import annotations

import time

import pytest

from core.trace import PipelineTrace, aggregate, summarize


def test_stages_recorded_in_order() -> None:
    """Aşamalar çağrıldıkları sırayla kaydedilmeli."""
    trace = PipelineTrace(pipeline="rag", model_name="test")
    for name in ("retrieval", "prompt", "generation", "output_guardrail"):
        with trace.stage(name):
            pass
    trace.finish()

    assert [stage.name for stage in trace.stages] == [
        "retrieval", "prompt", "generation", "output_guardrail"
    ]
    assert [stage.index for stage in trace.stages] == [0, 1, 2, 3]
    assert all(stage.status == "ok" for stage in trace.stages)


def test_durations_are_measured() -> None:
    """Aşama süresi ölçülmeli ve toplam süreye yansımalı."""
    trace = PipelineTrace(pipeline="rag")
    with trace.stage("yavas"):
        time.sleep(0.05)
    with trace.stage("hizli"):
        pass
    trace.finish()

    durations = trace.stage_durations()
    assert durations["yavas"] >= 0.04
    assert durations["yavas"] > durations["hizli"]
    assert trace.slowest_stage() == "yavas"
    assert trace.total_duration_sec >= durations["yavas"]


def test_error_is_recorded_and_reraised() -> None:
    """Hata kaydedilmeli ama yutulmamalı.

    Trace bir hata yönetim katmanı değildir; yalnızca gözlemler. İstisnayı
    yutmak, kırık pipeline'ın sessizce başarılı görünmesine yol açardı.
    """
    trace = PipelineTrace(pipeline="rag")
    with pytest.raises(ValueError, match="retrieval coktu"), trace.stage("retrieval"):
        raise ValueError("retrieval coktu")
    trace.finish()

    stage = trace.stages[0]
    assert stage.status == "error"
    assert stage.error_type == "ValueError"
    assert "retrieval coktu" in stage.error_message
    assert trace.failed_stages == ["retrieval"]


def test_findings_are_attributed_to_stage() -> None:
    """Bulgular hangi aşamada tetiklendiyse oraya yazılmalı.

    Girdide yakalanan bir injection ile çıktıda yakalanan bir ihlal farklı
    risklerdir; ayrım kaybolursa kök neden analizi yapılamaz.
    """
    trace = PipelineTrace(pipeline="rag")
    with trace.stage("input_guardrail") as stage:
        stage.add_findings("injection", [{"scenario_id": "INJ-01"}])
    with trace.stage("output_guardrail") as stage:
        stage.add_findings("content_safety", [{"category": "profanity"}])
        stage.add_findings("pii", [{"type": "email"}])
    trace.finish()

    per_stage = trace.findings_by_stage()
    assert per_stage["input_guardrail"] == 1
    assert per_stage["output_guardrail"] == 2
    assert trace.total_findings == 3
    assert "injection" in trace.stages[0].findings
    assert "content_safety" in trace.stages[1].findings


def test_skipped_stage_is_not_an_error() -> None:
    """Atlanan aşama hata sayılmamalı."""
    trace = PipelineTrace(pipeline="rag")
    with trace.stage("reranking") as stage:
        stage.mark_skipped("reranker yapilandirilmamis")
    trace.finish()

    assert trace.stages[0].status == "skipped"
    assert trace.failed_stages == []


# --------------------------------------------------------------------------- #
# Veri sızıntısı ve kontrol karakteri
# --------------------------------------------------------------------------- #
def test_summary_does_not_store_raw_content() -> None:
    """Özet, ham içeriğin tamamını taşımamalı."""
    secret = "gizli belge icerigi " * 100
    summary = summarize(secret)

    assert summary["length"] == len(secret)
    assert len(summary["preview"]) <= 160
    assert secret not in summary["preview"]
    assert len(summary["sha256_12"]) == 12


def test_summary_strips_control_characters() -> None:
    """Önizleme satır sonu içermemeli (CWE-117)."""
    payload = "normal\nsatir\r\nSAHTE LOG: yetki verildi\ttab"
    summary = summarize(payload)
    assert "\n" not in summary["preview"]
    assert "\r" not in summary["preview"]
    assert "\t" not in summary["preview"]


def test_redactor_is_applied_to_preview() -> None:
    """Enjekte edilen redaksiyon işlevi önizlemeye uygulanmalı."""
    trace = PipelineTrace(pipeline="rag", redactor=lambda text: text.replace("gizli", "***"))
    with trace.stage("generation") as stage:
        stage.set_output("bu gizli bir cevaptir")
    trace.finish()

    assert "gizli" not in trace.stages[0].output_summary["preview"]
    assert "***" in trace.stages[0].output_summary["preview"]


def test_failing_redactor_does_not_break_trace() -> None:
    """Redaksiyon işlevi çökerse trace çalışmaya devam etmeli."""
    def broken(_: str) -> str:
        raise RuntimeError("redaksiyon hatasi")

    trace = PipelineTrace(pipeline="rag", redactor=broken)
    with trace.stage("generation") as stage:
        stage.set_output("herhangi bir cevap")
    trace.finish()

    assert trace.stages[0].status == "ok"
    assert "redaksiyon" in trace.stages[0].output_summary["preview"]


def test_same_input_produces_same_hash() -> None:
    """Aynı girdi aynı hash'i üretmeli (koşular arası karşılaştırma)."""
    assert summarize("aynı metin")["sha256_12"] == summarize("aynı metin")["sha256_12"]
    assert summarize("metin a")["sha256_12"] != summarize("metin b")["sha256_12"]


# --------------------------------------------------------------------------- #
# Toplu görünüm
# --------------------------------------------------------------------------- #
def test_aggregate_computes_error_rate_and_bottleneck() -> None:
    """Toplu özet, sistematik hata ve darboğaz göstermeli."""
    traces = []
    for index in range(4):
        trace = PipelineTrace(pipeline="rag")
        with trace.stage("retrieval"):
            time.sleep(0.02)
        if index < 2:
            with pytest.raises(RuntimeError), trace.stage("generation"):
                raise RuntimeError("model hatasi")
        else:
            with trace.stage("generation"):
                pass
        traces.append(trace.finish())

    summary = aggregate(traces)
    assert summary["traces"] == 4
    assert summary["stage_error_rate"]["generation"] == pytest.approx(0.5)
    assert summary["stage_error_rate"]["retrieval"] == 0.0
    assert summary["bottleneck"] == "retrieval"


def test_aggregate_handles_empty_input() -> None:
    """İzleme yoksa toplu özet çökmemeli."""
    summary = aggregate([])
    assert summary["traces"] == 0
    assert summary["bottleneck"] == ""


def test_trace_serializes_to_dict() -> None:
    """Tam gösterim JSON'a yazılabilir olmalı (SQLite payload'u için)."""
    import json

    trace = PipelineTrace(pipeline="rag", model_name="test", query_id="Q-1")
    with trace.stage("retrieval") as stage:
        stage.set_input("soru metni", top_k=4)
        stage.set_output(["chunk1", "chunk2"], retrieved=2)
    trace.finish()

    payload = trace.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "retrieval" in encoded
    assert payload["stage_count"] == 1
    assert payload["stages"][0]["metrics"]["top_k"] == 4.0
    assert payload["stages"][0]["output"]["items"] == 2
