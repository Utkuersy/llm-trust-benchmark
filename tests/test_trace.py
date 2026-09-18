"""Pipeline trace layer tests.

The trace layer is responsible for saying *where* results broke down.
Tests therefore guarantee three things:

1. Stage order, duration, and status are recorded correctly
2. Errors are not swallowed — they are recorded and re-raised
3. Raw content and control characters do not leak into the trace (CWE-117
   and data leakage)
"""

from __future__ import annotations

import time

import pytest

from core.trace import PipelineTrace, aggregate, summarize


def test_stages_recorded_in_order() -> None:
    """Stages must be recorded in the order they were called."""
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
    """Stage duration must be measured and reflected in the total duration."""
    trace = PipelineTrace(pipeline="rag")
    with trace.stage("slow"):
        time.sleep(0.05)
    with trace.stage("fast"):
        pass
    trace.finish()

    durations = trace.stage_durations()
    assert durations["slow"] >= 0.04
    assert durations["slow"] > durations["fast"]
    assert trace.slowest_stage() == "slow"
    assert trace.total_duration_sec >= durations["slow"]


def test_error_is_recorded_and_reraised() -> None:
    """An error must be recorded but not swallowed.

    Trace is not an error-handling layer; it only observes. Swallowing
    the exception would let a broken pipeline appear silently successful.
    """
    trace = PipelineTrace(pipeline="rag")
    with pytest.raises(ValueError, match="retrieval blew up"), trace.stage("retrieval"):
        raise ValueError("retrieval blew up")
    trace.finish()

    stage = trace.stages[0]
    assert stage.status == "error"
    assert stage.error_type == "ValueError"
    assert "retrieval blew up" in stage.error_message
    assert trace.failed_stages == ["retrieval"]


def test_findings_are_attributed_to_stage() -> None:
    """Findings must be written to the stage where they were triggered.

    An injection caught on input and a violation caught on output are
    different risks; if the distinction is lost, root-cause analysis
    becomes impossible.
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
    """A skipped stage must not count as an error."""
    trace = PipelineTrace(pipeline="rag")
    with trace.stage("reranking") as stage:
        stage.mark_skipped("reranker not configured")
    trace.finish()

    assert trace.stages[0].status == "skipped"
    assert trace.failed_stages == []


# --------------------------------------------------------------------------- #
# Data leakage and control characters
# --------------------------------------------------------------------------- #
def test_summary_does_not_store_raw_content() -> None:
    """The summary must not carry the full raw content."""
    secret = "secret document contents " * 100
    summary = summarize(secret)

    assert summary["length"] == len(secret)
    assert len(summary["preview"]) <= 160
    assert secret not in summary["preview"]
    assert len(summary["sha256_12"]) == 12


def test_summary_strips_control_characters() -> None:
    """The preview must not contain line breaks (CWE-117)."""
    payload = "normal\nline\r\nFAKE LOG: access granted\ttab"
    summary = summarize(payload)
    assert "\n" not in summary["preview"]
    assert "\r" not in summary["preview"]
    assert "\t" not in summary["preview"]


def test_redactor_is_applied_to_preview() -> None:
    """An injected redaction function must be applied to the preview."""
    trace = PipelineTrace(pipeline="rag", redactor=lambda text: text.replace("secret", "***"))
    with trace.stage("generation") as stage:
        stage.set_output("this is a secret answer")
    trace.finish()

    assert "secret" not in trace.stages[0].output_summary["preview"]
    assert "***" in trace.stages[0].output_summary["preview"]


def test_failing_redactor_does_not_break_trace() -> None:
    """If the redaction function crashes, the trace must keep working."""
    def broken(_: str) -> str:
        raise RuntimeError("redaction error")

    trace = PipelineTrace(pipeline="rag", redactor=broken)
    with trace.stage("generation") as stage:
        stage.set_output("some answer")
    trace.finish()

    assert trace.stages[0].status == "ok"
    assert "redaction" in trace.stages[0].output_summary["preview"]


def test_same_input_produces_same_hash() -> None:
    """The same input must produce the same hash (for cross-run comparison)."""
    assert summarize("same text")["sha256_12"] == summarize("same text")["sha256_12"]
    assert summarize("text a")["sha256_12"] != summarize("text b")["sha256_12"]


# --------------------------------------------------------------------------- #
# Aggregate view
# --------------------------------------------------------------------------- #
def test_aggregate_computes_error_rate_and_bottleneck() -> None:
    """The aggregate summary must show systematic errors and the bottleneck."""
    traces = []
    for index in range(4):
        trace = PipelineTrace(pipeline="rag")
        with trace.stage("retrieval"):
            time.sleep(0.02)
        if index < 2:
            with pytest.raises(RuntimeError), trace.stage("generation"):
                raise RuntimeError("model error")
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
    """The aggregate summary must not crash when there are no traces."""
    summary = aggregate([])
    assert summary["traces"] == 0
    assert summary["bottleneck"] == ""


def test_trace_serializes_to_dict() -> None:
    """The full representation must be JSON-serializable (for the SQLite payload)."""
    import json

    trace = PipelineTrace(pipeline="rag", model_name="test", query_id="Q-1")
    with trace.stage("retrieval") as stage:
        stage.set_input("question text", top_k=4)
        stage.set_output(["chunk1", "chunk2"], retrieved=2)
    trace.finish()

    payload = trace.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "retrieval" in encoded
    assert payload["stage_count"] == 1
    assert payload["stages"][0]["metrics"]["top_k"] == 4.0
    assert payload["stages"][0]["output"]["items"] == 2
