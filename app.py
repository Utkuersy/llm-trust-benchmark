"""LLM Output Reliability — Streamlit dashboard (lightweight edition).

Four tabs:

* **Overview** — Trust Score, a single chart of the seven dimensions,
  the riskiest findings, and where the pipeline runs into trouble — all
  on one screen, no scrolling.
* **Details** — the full metrics table, a filterable findings list, the
  raw record of a sample pipeline trace.
* **Governance** — the drift report, the audit trail summary.
* **Add Model** — upload a new model's answer files and trigger an
  evaluation from here (see ``render_add_model``); the same tab also has
  a section for permanently deleting a model (see ``render_delete_model``).

Compared to the previous version: the radar chart and repetitive bar
charts were removed (they showed the same information three times in
different forms), the chart count went from 15 to 4, the tab count from
4 to 2. The goal is to answer the most important question — "which
model is trustworthy, and why not" — on the very first screen.

The dashboard reads from ``db/benchmark.db`` by default; the "Add Model"
tab writes uploaded files under ``llm_outputs/<model>/`` and triggers a
synchronous run via ``benchmark_engine.run_benchmark`` — apart from this
one exception, the dashboard starts no other process.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import shutil
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import Settings, get_settings
from core.scoring import resolve_preset
from core.storage import delete_model, fetch_findings, fetch_runs, fetch_track_b, init_db
from core.versioning import drift_report

REQUIRED_UPLOAD = "rag_answers.json"
OPTIONAL_UPLOADS = ("injection_responses.json", "math_answers.json")


def _require_auth() -> None:
    """Password-protects the dashboard, since report findings can be sensitive.

    The password itself is never stored anywhere; it is only compared
    against the SHA-256 hash held in the ``AITB__DASHBOARD__PASSWORD_HASH``
    environment variable (see ``core/dashboard_auth.py``). Unless
    ``auth_enabled`` is turned off, access stays on the safe side and is
    fully denied when the hash is undefined.
    """
    settings = get_settings()
    if not settings.dashboard.auth_enabled:
        return
    if st.session_state.get("_authenticated"):
        return

    st.title("🛡️ LLM Output Reliability")
    if not settings.dashboard.password_hash:
        st.error(
            "Authentication is enabled but no password is defined. "
            "Generate a hash with `python -m core.dashboard_auth` and "
            "assign it to the `AITB__DASHBOARD__PASSWORD_HASH` environment variable."
        )
        st.stop()

    password = st.text_input("Dashboard password", type="password")
    if not password:
        st.stop()
    entered_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if hmac.compare_digest(entered_hash, settings.dashboard.password_hash):
        st.session_state["_authenticated"] = True
        st.rerun()
    st.error("Incorrect password.")
    st.stop()

DIMENSION_LABELS = {
    "content_safety_score": "Content safety",
    "injection_score": "Injection resistance",
    "pii_score": "PII safety",
    "poisoning_score": "Poisoning resistance",
    "retrieval_score": "Retrieval",
    "generation_score": "Faithfulness",
    "math_score": "Math",
}
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
SEVERITY_COLORS = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#f1c40f", "INFO": "#95a5a6"}

st.set_page_config(page_title="LLM Output Reliability", page_icon="🛡️", layout="wide")
_require_auth()


@st.cache_data(ttl=30)
def load_data() -> dict[str, Any]:
    """Reads the tables from the database (cached for 30s)."""
    init_db()
    runs = pd.DataFrame(fetch_runs())
    payloads: dict[str, dict[str, Any]] = {}
    if not runs.empty:
        latest = runs.sort_values("created_at", ascending=False).drop_duplicates("model_name")
        for _, row in latest.iterrows():
            try:
                payloads[row["model_name"]] = json.loads(row["payload"])
            except (ValueError, TypeError):
                payloads[row["model_name"]] = {}
    return {
        "results": pd.DataFrame(fetch_track_b()),
        "findings": pd.DataFrame(fetch_findings()),
        "payloads": payloads,
    }


def latest_per_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the latest run for each model."""
    if frame.empty or "model_name" not in frame.columns:
        return frame
    return frame.sort_values("created_at", ascending=False).drop_duplicates("model_name")


def score_color(value: float) -> str:
    """A color based on the Trust Score."""
    if value >= 80:
        return "#27ae60"
    if value >= 60:
        return "#f39c12"
    return "#c0392b"


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def render_overview(
    frame: pd.DataFrame, payloads: dict[str, dict[str, Any]], findings: pd.DataFrame
) -> None:
    """On one screen: Trust Score, dimensions, the riskiest findings, a pipeline summary."""
    ordered = frame.sort_values("trust_score", ascending=False)

    columns = st.columns(min(4, max(1, len(ordered))))
    for index, (_, row) in enumerate(ordered.iterrows()):
        with columns[index % len(columns)]:
            st.metric(str(row["model_name"]).upper(), f"{float(row['trust_score']):.1f}")
            st.markdown(
                f"<div style='height:5px;border-radius:3px;background:"
                f"{score_color(float(row['trust_score']))}'></div>",
                unsafe_allow_html=True,
            )

    _render_measurement_note(payloads)

    st.markdown("##### The seven dimensions")
    available = [c for c in DIMENSION_LABELS if c in frame.columns]
    melted = frame.melt(id_vars="model_name", value_vars=available, var_name="dimension", value_name="score")
    melted["dimension"] = melted["dimension"].map(DIMENSION_LABELS)
    st.plotly_chart(
        px.bar(
            melted, x="dimension", y="score", color="model_name", barmode="group", range_y=[0, 100],
        ).update_layout(height=320, margin={"t": 10, "b": 10}, legend_title=None),
        use_container_width=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Riskiest findings")
        _render_top_findings(findings, limit=6)
    with right:
        st.markdown("##### Pipeline: where does it break down?")
        _render_pipeline_mini(payloads)


def _render_measurement_note(payloads: dict[str, dict[str, Any]]) -> None:
    """Reports unmeasured categories and skipped dimensions in a single line."""
    inactive: set[str] = set()
    skipped_count = 0
    for payload in payloads.values():
        inactive.update((payload.get("content_safety") or {}).get("inactive_categories", []) or [])
        for dimension in DIMENSION_LABELS:
            key = dimension.replace("_score", "")
            status = (payload.get(key) or {}).get("status")
            if status in {"skipped", "error"}:
                skipped_count += 1
    if inactive:
        st.caption(
            f"⚠️ Unmeasured content categories: {', '.join(sorted(inactive))} "
            "— fill in the lexicons under `config/lexicons/`."
        )
    if skipped_count:
        st.caption(
            f"ℹ️ {skipped_count} dimension(s) skipped; their weight was removed "
            "from the denominator, not scored as zero."
        )


def _render_top_findings(findings: pd.DataFrame, limit: int) -> None:
    """Shows the highest-severity findings in a compact table."""
    if findings.empty:
        st.info("No findings recorded.")
        return
    view = findings.copy()
    view["_rank"] = view["severity"].map(SEVERITY_ORDER).fillna(9)
    view = view.sort_values("_rank").head(limit)
    st.dataframe(
        view[["model_name", "severity", "category", "title"]].rename(
            columns={"model_name": "model", "severity": "severity", "category": "category", "title": "finding"}
        ),
        use_container_width=True, hide_index=True, height=38 * min(limit, len(view)) + 38,
    )


def _render_pipeline_mini(payloads: dict[str, dict[str, Any]]) -> None:
    """Shows the stage-level finding count in a single chart; summarizes the bottleneck in one line."""
    rows = []
    bottlenecks = []
    for model, payload in payloads.items():
        pipeline = payload.get("pipeline") or {}
        for stage, count in (pipeline.get("stage_findings") or {}).items():
            rows.append({"model_name": model, "stage": stage, "findings": count})
        if pipeline.get("bottleneck"):
            bottlenecks.append(f"{model}: {pipeline['bottleneck']}")

    if not rows:
        st.info("No pipeline trace. Run without `--no-trace`.")
        return

    st.plotly_chart(
        px.bar(
            pd.DataFrame(rows), x="stage", y="findings", color="model_name", barmode="group",
        ).update_layout(height=280, margin={"t": 10, "b": 10}, showlegend=False),
        use_container_width=True,
    )
    if bottlenecks:
        st.caption("Bottleneck (slowest stage): " + " · ".join(bottlenecks))


# --------------------------------------------------------------------------- #
# Details
# --------------------------------------------------------------------------- #
def render_details(
    frame: pd.DataFrame, payloads: dict[str, dict[str, Any]], findings: pd.DataFrame
) -> None:
    """The full metrics table, filterable findings, a sample pipeline trace."""
    st.markdown("##### Full metrics table")
    metric_columns = [c for c in (
        "model_name", "trust_score", *DIMENSION_LABELS,
        "context_precision", "faithfulness", "hallucination_rate",
        "scenarios_failed", "pii_hits", "susceptibility_rate", "math_accuracy",
    ) if c in frame.columns]
    st.dataframe(frame[metric_columns], use_container_width=True, hide_index=True)

    st.markdown("##### Findings")
    if findings.empty:
        st.info("No findings recorded.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox(
                "Model", ["(all)", *sorted(findings["model_name"].unique().tolist())]
            )
        with col2:
            category_choice = st.selectbox(
                "Category", ["(all)", *sorted(findings["category"].unique().tolist())]
            )
        view = findings.copy()
        if model_choice != "(all)":
            view = view[view["model_name"] == model_choice]
        if category_choice != "(all)":
            view = view[view["category"] == category_choice]
        view["_rank"] = view["severity"].map(SEVERITY_ORDER).fillna(9)
        view = view.sort_values("_rank")
        st.dataframe(
            view[["model_name", "category", "severity", "title", "detail", "location"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("##### Sample pipeline trace")
    model_names = sorted(payloads)
    if model_names:
        selected = st.selectbox("Select model", model_names, key="trace_model")
        sample = (payloads[selected].get("pipeline") or {}).get("sample_trace") or {}
        if sample:
            with st.expander("Raw JSON record"):
                st.json(sample)
        else:
            st.caption("No sample trace recorded for this model.")


def render_governance(frame: pd.DataFrame) -> None:
    """Shows audit trail integrity and a model drift summary.

    Deliberately kept plain: this tab isn't "decoration," it answers the
    question "prove how you arrived at this result." Directly readable
    tables are preferred over heavy charts.
    """
    from core.audit import fetch_audit_log, verify_chain
    from core.versioning import explain_dataset_version

    st.markdown("##### Audit trail integrity")
    ok, problems = verify_chain()
    if ok:
        st.success("Chain integrity verified — no tampering detected.")
    else:
        st.error("WARNING: inconsistency detected in the chain.")
        for problem in problems:
            st.caption(f"• {problem}")

    entries = fetch_audit_log()
    if entries:
        recent = pd.DataFrame(entries[-10:])
        st.dataframe(
            recent[["sequence", "timestamp", "run_id", "triggered_by", "hostname", "code_version", "preset"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("No audit trail records yet.")

    st.markdown("##### Test data version")
    version_info = explain_dataset_version()
    st.caption(f"`dataset_version`: `{version_info['dataset_version']}`")
    if version_info["missing"]:
        st.warning(f"Missing components: {', '.join(version_info['missing'])}")

    st.markdown("##### Model drift")
    st.caption(
        "Whether a score change comes from a data/weight change or from "
        "real model behavior — the distinction is made here."
    )
    if not frame.empty:
        selected_model = st.selectbox("Model", sorted(frame["model_name"].unique().tolist()))
        report = drift_report(selected_model)
        if report["comparable_pairs"] == 0:
            st.info("At least 2 runs are needed for this model to compare.")
        else:
            for transition in report["transitions"]:
                icon = {"model_drift": "🔴", "config_changed": "🟡", "stable": "🟢"}.get(
                    transition["classification"], "⚪"
                )
                st.caption(
                    f"{icon} {transition['from_date'][:10]} → {transition['to_date'][:10]}: "
                    f"{transition['from_score']:.1f} → {transition['to_score']:.1f} "
                    f"— {transition['explanation']}"
                )


def render_add_model(settings: Settings) -> None:
    """Lets a new model's answer files be uploaded and triggers a synchronous evaluation run.

    The questions/scenarios are fixed (the RAG corpus, 12 injection
    scenarios, 13 math problems) — the user doesn't write new questions,
    only uploads the model's answers to this fixed set.
    ``rag_answers.json`` is required; if the other two are skipped,
    those dimensions are counted as "not measured," not scored as zero
    (see ``core/scoring.aggregate``).
    """
    if "model_tab_message" in st.session_state:
        st.success(st.session_state.pop("model_tab_message"))

    st.markdown(
        "Upload answer files to evaluate a new model. The questions are "
        "fixed; only the model's answers to these questions/scenarios are "
        "needed. To see the scenario texts: "
        "`python -m llm_security.prompt_injection_tests --list`"
    )

    model_name = st.text_input("Model name", placeholder="e.g. new_model", key="add_model_name")
    rag_upload = st.file_uploader(f"{REQUIRED_UPLOAD} (required)", type="json", key="add_model_rag")
    injection_upload = st.file_uploader(
        "injection_responses.json (optional — if skipped, the injection dimension is not measured)",
        type="json", key="add_model_injection",
    )
    math_upload = st.file_uploader(
        "math_answers.json (optional — if skipped, the math dimension is not measured)",
        type="json", key="add_model_math",
    )

    disabled = not (model_name and model_name.strip() and rag_upload is not None)
    if st.button("Evaluate and save", type="primary", disabled=disabled):
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", model_name.strip()).strip("_")
        if not safe_name:
            st.error("Enter a valid model name (letters/digits/_/-).")
            return

        uploads = {
            REQUIRED_UPLOAD: rag_upload,
            "injection_responses.json": injection_upload,
            "math_answers.json": math_upload,
        }
        parsed: dict[str, Any] = {}
        for filename, upload in uploads.items():
            if upload is None:
                continue
            try:
                parsed[filename] = json.loads(upload.getvalue().decode("utf-8"))
            except UnicodeDecodeError:
                st.error(f"{filename} could not be read as UTF-8 text.")
                return
            except json.JSONDecodeError as exc:
                st.error(f"{filename} is not valid JSON: {exc}")
                return

        model_dir = settings.paths.absolute(settings.paths.llm_outputs_dir) / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in parsed.items():
            (model_dir / filename).write_text(
                json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        with st.spinner(f"Evaluating '{safe_name}'…"):
            from benchmark_engine import run_benchmark

            try:
                results = run_benchmark(models=[safe_name], settings=settings)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                return

        if not results:
            st.error(
                "The evaluation produced no result. Check the uploaded "
                "files' content (make sure they aren't an empty list/dict)."
            )
            return

        # st.rerun() cuts this script's execution immediately; the message
        # is stashed in session_state to be shown at the top of the next
        # run — otherwise the user would never see it before the page reloads.
        st.session_state["model_tab_message"] = (
            f"'{safe_name}' evaluated — Trust Score: {results[0].trust_score}."
        )
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    render_delete_model(settings)


def render_delete_model(settings: Settings) -> None:
    """Permanently deletes a model that was added by mistake or is no longer needed.

    Deleting a ``runs`` row automatically clears the corresponding
    ``track_a_results``/``track_b_results``/``findings`` rows via
    ``ON DELETE CASCADE`` (see ``core/storage.delete_model``); the
    ``llm_outputs/<model>/`` folder is deleted separately by hand since
    the filesystem isn't tied to the database.
    """
    st.markdown("##### Delete a model")
    models = sorted({row["model_name"] for row in fetch_runs()})
    if not models:
        st.caption("No model to delete.")
        return

    to_delete = st.selectbox("Model to delete", models, key="delete_model_select")
    confirm = st.checkbox(
        f"I confirm I want to permanently delete '{to_delete}' and all its results",
        key="delete_model_confirm",
    )
    if st.button("Delete", disabled=not confirm, key="delete_model_button"):
        deleted = delete_model(to_delete, settings)
        model_dir = settings.paths.absolute(settings.paths.llm_outputs_dir) / to_delete
        if model_dir.is_dir():
            shutil.rmtree(model_dir)
        st.session_state["model_tab_message"] = f"'{to_delete}' deleted ({deleted} run records)."
        st.cache_data.clear()
        st.rerun()


def main() -> None:
    """The dashboard entry point."""
    settings = get_settings()
    st.title("🛡️ LLM Output Reliability")

    data = load_data()
    frame = latest_per_model(data["results"])

    with st.sidebar:
        st.caption(f"Weight preset: `{settings.scoring.track_b_preset}`")
        try:
            weights = resolve_preset("B", settings.scoring.track_b_preset)
            st.caption(" · ".join(f"{k}: {v:.2f}" for k, v in weights.items()))
        except (KeyError, ValueError):
            st.caption("⚠️ undefined preset")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    overview_tab, detail_tab, governance_tab, add_model_tab = st.tabs(
        ["📊 Overview", "🔍 Details", "🛡️ Governance", "➕ Add Model"]
    )

    if frame.empty:
        with overview_tab:
            st.warning(
                "No evaluation on record. Add and evaluate a model from the "
                "**Add Model** tab on the right, or run the following from "
                "the command line in order:\n\n"
                "```\npython -m rag.ingest --seed --reset\n"
                "python -m scripts.generate_llm_outputs\n"
                "python benchmark_engine.py\n```"
            )
    else:
        with overview_tab:
            render_overview(frame, data["payloads"], data["findings"])
        with detail_tab:
            render_details(frame, data["payloads"], data["findings"])
        with governance_tab:
            render_governance(frame)

    with add_model_tab:
        render_add_model(settings)


if __name__ == "__main__":
    main()
