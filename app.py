"""LLM Çıktı Güvenilirliği — Streamlit dashboard (hafif sürüm).

Dört sekme:

* **Genel bakış** — Trust Score, yedi boyutun tek grafiği, en riskli
  bulgular ve pipeline'ın nerede sorun çıkardığı — tek ekranda, kaydırmadan.
* **Detaylar** — tam metrik tablosu, filtrelenebilir bulgu listesi, örnek
  pipeline izinin ham kaydı.
* **Yönetişim** — drift raporu, denetim izi özeti.
* **Model Ekle** — yeni bir modelin cevap dosyalarını yükleyip değerlendirmeyi
  buradan tetikler (bkz. ``render_add_model``); aynı sekmede bir modeli
  kalıcı olarak silme bölümü de vardır (bkz. ``render_delete_model``).

Önceki sürüme göre: radar grafiği ve tekrarlı sütun grafikleri kaldırıldı
(aynı bilgiyi farklı biçimde üç kez gösteriyorlardı), grafik sayısı 15'ten
4'e indi, sekme sayısı 4'ten 2'ye indi. Amaç, en önemli soruya —
"hangi model güvenilir, neden değil" — ilk ekranda cevap vermek.

Dashboard varsayılan olarak ``db/benchmark.db``'yi okur; "Model Ekle"
sekmesi, yüklenen dosyaları ``llm_outputs/<model>/`` altına yazıp
``benchmark_engine.run_benchmark`` ile senkron bir koşu tetikler — bu
tek istisna dışında dashboard başka bir işlem başlatmaz.
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
    """Rapor bulguları hassas olabileceğinden dashboard'u şifreyle korur.

    Şifrenin kendisi hiçbir yerde saklanmaz; yalnızca
    ``AITB__DASHBOARD__PASSWORD_HASH`` ortam değişkenindeki SHA-256 hash'i
    ile karşılaştırılır (bkz. ``core/dashboard_auth.py``). ``auth_enabled``
    kapatılmadıkça, hash tanımlı değilse erişim güvenli tarafta kalıp
    tamamen reddedilir.
    """
    settings = get_settings()
    if not settings.dashboard.auth_enabled:
        return
    if st.session_state.get("_authenticated"):
        return

    st.title("🛡️ LLM Çıktı Güvenilirliği")
    if not settings.dashboard.password_hash:
        st.error(
            "Kimlik doğrulama etkin ama şifre tanımlanmamış. "
            "`python -m core.dashboard_auth` ile bir hash üretip "
            "`AITB__DASHBOARD__PASSWORD_HASH` ortam değişkenine atayın."
        )
        st.stop()

    password = st.text_input("Dashboard şifresi", type="password")
    if not password:
        st.stop()
    entered_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if hmac.compare_digest(entered_hash, settings.dashboard.password_hash):
        st.session_state["_authenticated"] = True
        st.rerun()
    st.error("Hatalı şifre.")
    st.stop()

DIMENSION_LABELS = {
    "content_safety_score": "İçerik güvenliği",
    "injection_score": "Injection direnci",
    "pii_score": "PII güvenliği",
    "poisoning_score": "Zehirlenme direnci",
    "retrieval_score": "Retrieval",
    "generation_score": "Faithfulness",
    "math_score": "Matematik",
}
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
SEVERITY_COLORS = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#f1c40f", "INFO": "#95a5a6"}

st.set_page_config(page_title="LLM Çıktı Güvenilirliği", page_icon="🛡️", layout="wide")
_require_auth()


@st.cache_data(ttl=30)
def load_data() -> dict[str, Any]:
    """Veritabanından tabloları okur (30 sn cache'li)."""
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
    """Her model için yalnızca en son koşuyu bırakır."""
    if frame.empty or "model_name" not in frame.columns:
        return frame
    return frame.sort_values("created_at", ascending=False).drop_duplicates("model_name")


def score_color(value: float) -> str:
    """Trust Score'a göre renk."""
    if value >= 80:
        return "#27ae60"
    if value >= 60:
        return "#f39c12"
    return "#c0392b"


# --------------------------------------------------------------------------- #
# Genel bakış
# --------------------------------------------------------------------------- #
def render_overview(
    frame: pd.DataFrame, payloads: dict[str, dict[str, Any]], findings: pd.DataFrame
) -> None:
    """Tek ekranda: Trust Score, boyutlar, en riskli bulgular, pipeline özeti."""
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

    st.markdown("##### Yedi boyut")
    available = [c for c in DIMENSION_LABELS if c in frame.columns]
    melted = frame.melt(id_vars="model_name", value_vars=available, var_name="boyut", value_name="puan")
    melted["boyut"] = melted["boyut"].map(DIMENSION_LABELS)
    st.plotly_chart(
        px.bar(
            melted, x="boyut", y="puan", color="model_name", barmode="group", range_y=[0, 100],
        ).update_layout(height=320, margin={"t": 10, "b": 10}, legend_title=None),
        use_container_width=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### En riskli bulgular")
        _render_top_findings(findings, limit=6)
    with right:
        st.markdown("##### Pipeline: nerede çıktı?")
        _render_pipeline_mini(payloads)


def _render_measurement_note(payloads: dict[str, dict[str, Any]]) -> None:
    """Ölçülmeyen kategori ve atlanan boyutları tek satırda bildirir."""
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
            f"⚠️ Ölçülmeyen içerik kategorileri: {', '.join(sorted(inactive))} "
            "— sözlükleri `config/lexicons/` altında doldurun."
        )
    if skipped_count:
        st.caption(
            f"ℹ️ {skipped_count} boyut atlandı; ağırlıkları paydadan düşüldü, sıfır puan verilmedi."
        )


def _render_top_findings(findings: pd.DataFrame, limit: int) -> None:
    """En yüksek önem derecesindeki bulguları kompakt bir tabloda gösterir."""
    if findings.empty:
        st.info("Kayıtlı bulgu yok.")
        return
    view = findings.copy()
    view["_rank"] = view["severity"].map(SEVERITY_ORDER).fillna(9)
    view = view.sort_values("_rank").head(limit)
    st.dataframe(
        view[["model_name", "severity", "category", "title"]].rename(
            columns={"model_name": "model", "severity": "önem", "category": "kategori", "title": "bulgu"}
        ),
        use_container_width=True, hide_index=True, height=38 * min(limit, len(view)) + 38,
    )


def _render_pipeline_mini(payloads: dict[str, dict[str, Any]]) -> None:
    """Aşama bazlı bulgu sayısını tek grafikte gösterir; darboğazı bir satırda özetler."""
    rows = []
    bottlenecks = []
    for model, payload in payloads.items():
        pipeline = payload.get("pipeline") or {}
        for stage, count in (pipeline.get("stage_findings") or {}).items():
            rows.append({"model_name": model, "aşama": stage, "bulgu": count})
        if pipeline.get("bottleneck"):
            bottlenecks.append(f"{model}: {pipeline['bottleneck']}")

    if not rows:
        st.info("Pipeline izi yok. `--no-trace` olmadan çalıştırın.")
        return

    st.plotly_chart(
        px.bar(
            pd.DataFrame(rows), x="aşama", y="bulgu", color="model_name", barmode="group",
        ).update_layout(height=280, margin={"t": 10, "b": 10}, showlegend=False),
        use_container_width=True,
    )
    if bottlenecks:
        st.caption("Darboğaz (en yavaş aşama): " + " · ".join(bottlenecks))


# --------------------------------------------------------------------------- #
# Detaylar
# --------------------------------------------------------------------------- #
def render_details(
    frame: pd.DataFrame, payloads: dict[str, dict[str, Any]], findings: pd.DataFrame
) -> None:
    """Tam metrik tablosu, filtrelenebilir bulgular, örnek pipeline izi."""
    st.markdown("##### Tam metrik tablosu")
    metric_columns = [c for c in (
        "model_name", "trust_score", *DIMENSION_LABELS,
        "context_precision", "faithfulness", "hallucination_rate",
        "scenarios_failed", "pii_hits", "susceptibility_rate", "math_accuracy",
    ) if c in frame.columns]
    st.dataframe(frame[metric_columns], use_container_width=True, hide_index=True)

    st.markdown("##### Bulgular")
    if findings.empty:
        st.info("Kayıtlı bulgu yok.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox(
                "Model", ["(tümü)", *sorted(findings["model_name"].unique().tolist())]
            )
        with col2:
            category_choice = st.selectbox(
                "Kategori", ["(tümü)", *sorted(findings["category"].unique().tolist())]
            )
        view = findings.copy()
        if model_choice != "(tümü)":
            view = view[view["model_name"] == model_choice]
        if category_choice != "(tümü)":
            view = view[view["category"] == category_choice]
        view["_rank"] = view["severity"].map(SEVERITY_ORDER).fillna(9)
        view = view.sort_values("_rank")
        st.dataframe(
            view[["model_name", "category", "severity", "title", "detail", "location"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("##### Örnek pipeline izi")
    model_names = sorted(payloads)
    if model_names:
        selected = st.selectbox("Model seç", model_names, key="trace_model")
        sample = (payloads[selected].get("pipeline") or {}).get("sample_trace") or {}
        if sample:
            with st.expander("Ham JSON kaydı"):
                st.json(sample)
        else:
            st.caption("Bu model için örnek iz kaydedilmemiş.")


def render_governance(frame: pd.DataFrame) -> None:
    """Denetim izi bütünlüğü ve model drift özetini gösterir.

    Bilinçli olarak sade tutulmuştur: bu sekme "süsleme" değil, "bu sonuca
    nasıl ulaştığını kanıtla" sorusuna cevap verir. Ağır grafik yerine
    doğrudan okunabilir tablolar tercih edilmiştir.
    """
    from core.audit import fetch_audit_log, verify_chain
    from core.versioning import explain_dataset_version

    st.markdown("##### Denetim izi bütünlüğü")
    ok, problems = verify_chain()
    if ok:
        st.success("Zincir bütünlüğü doğrulandı — kurcalama tespit edilmedi.")
    else:
        st.error("UYARI: zincirde tutarsızlık tespit edildi.")
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
        st.caption("Henüz denetim izi kaydı yok.")

    st.markdown("##### Test verisi sürümü")
    version_info = explain_dataset_version()
    st.caption(f"`dataset_version`: `{version_info['dataset_version']}`")
    if version_info["missing"]:
        st.warning(f"Eksik bileşenler: {', '.join(version_info['missing'])}")

    st.markdown("##### Model drift")
    st.caption(
        "Puan değişimi veri/ağırlık değişikliğinden mi, gerçek model "
        "davranışından mı kaynaklanıyor — ayrım burada yapılır."
    )
    if not frame.empty:
        selected_model = st.selectbox("Model", sorted(frame["model_name"].unique().tolist()))
        report = drift_report(selected_model)
        if report["comparable_pairs"] == 0:
            st.info("Karşılaştırma için bu modelde en az 2 koşu gerekiyor.")
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
    """Yeni bir modelin cevap dosyalarını yükletip senkron bir değerlendirme koşusu tetikler.

    Sorular/senaryolar sabittir (RAG korpusu, 12 injection senaryosu, 13
    matematik sorusu) — kullanıcı yeni soru yazmaz, sadece modelin bu sabit
    kümeye verdiği cevapları yükler. ``rag_answers.json`` zorunludur; diğer
    ikisi atlanırsa o boyutlar "ölçülmedi" sayılır, sıfır puan verilmez
    (bkz. ``core/scoring.aggregate``).
    """
    if "model_tab_message" in st.session_state:
        st.success(st.session_state.pop("model_tab_message"))

    st.markdown(
        "Yeni bir modeli değerlendirmek için cevap dosyalarını yükleyin. "
        "Sorular sabittir; sadece modelin bu sorulara/senaryolara verdiği "
        "cevaplar gerekir. Senaryo metinlerini görmek için: "
        "`python -m llm_security.prompt_injection_tests --list`"
    )

    model_name = st.text_input("Model adı", placeholder="ör. yeni_model", key="add_model_name")
    rag_upload = st.file_uploader(f"{REQUIRED_UPLOAD} (zorunlu)", type="json", key="add_model_rag")
    injection_upload = st.file_uploader(
        "injection_responses.json (opsiyonel — atlanırsa injection boyutu ölçülmez)",
        type="json", key="add_model_injection",
    )
    math_upload = st.file_uploader(
        "math_answers.json (opsiyonel — atlanırsa math boyutu ölçülmez)",
        type="json", key="add_model_math",
    )

    disabled = not (model_name and model_name.strip() and rag_upload is not None)
    if st.button("Değerlendir ve kaydet", type="primary", disabled=disabled):
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", model_name.strip()).strip("_")
        if not safe_name:
            st.error("Geçerli bir model adı girin (harf/rakam/_/-).")
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
                st.error(f"{filename} UTF-8 metin olarak okunamadı.")
                return
            except json.JSONDecodeError as exc:
                st.error(f"{filename} geçerli bir JSON değil: {exc}")
                return

        model_dir = settings.paths.absolute(settings.paths.llm_outputs_dir) / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in parsed.items():
            (model_dir / filename).write_text(
                json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        with st.spinner(f"'{safe_name}' değerlendiriliyor…"):
            from benchmark_engine import run_benchmark

            try:
                results = run_benchmark(models=[safe_name], settings=settings)
            except Exception as exc:
                st.error(f"Değerlendirme başarısız oldu: {exc}")
                return

        if not results:
            st.error(
                "Değerlendirme sonuç üretmedi. Yüklenen dosyaların içeriğini "
                "(boş liste/sözlük olmadığından) kontrol edin."
            )
            return

        # st.rerun() bu script çalışmasını hemen keser; mesaj bir sonraki
        # çalışmada (fonksiyonun başında) gösterilmek üzere session_state'e
        # yazılır, aksi halde kullanıcı mesajı hiç göremeden sayfa yenilenir.
        st.session_state["model_tab_message"] = (
            f"'{safe_name}' değerlendirildi — Trust Score: {results[0].trust_score}."
        )
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    render_delete_model(settings)


def render_delete_model(settings: Settings) -> None:
    """Yanlışlıkla eklenen veya artık gerekmeyen bir modeli kalıcı olarak siler.

    ``runs`` satırını silmek ``track_a_results``/``track_b_results``/
    ``findings`` satırlarını ``ON DELETE CASCADE`` ile otomatik temizler
    (bkz. ``core/storage.delete_model``); ``llm_outputs/<model>/`` klasörü
    ayrıca elle silinir çünkü dosya sistemi veritabanına bağlı değildir.
    """
    st.markdown("##### Model sil")
    models = sorted({row["model_name"] for row in fetch_runs()})
    if not models:
        st.caption("Silinecek model yok.")
        return

    to_delete = st.selectbox("Silinecek model", models, key="delete_model_select")
    confirm = st.checkbox(
        f"'{to_delete}' modelini ve tüm sonuçlarını kalıcı olarak silmek istediğimi onaylıyorum",
        key="delete_model_confirm",
    )
    if st.button("Sil", disabled=not confirm, key="delete_model_button"):
        deleted = delete_model(to_delete, settings)
        model_dir = settings.paths.absolute(settings.paths.llm_outputs_dir) / to_delete
        if model_dir.is_dir():
            shutil.rmtree(model_dir)
        st.session_state["model_tab_message"] = f"'{to_delete}' silindi ({deleted} koşu kaydı)."
        st.cache_data.clear()
        st.rerun()


def main() -> None:
    """Dashboard giriş noktası."""
    settings = get_settings()
    st.title("🛡️ LLM Çıktı Güvenilirliği")

    data = load_data()
    frame = latest_per_model(data["results"])

    with st.sidebar:
        st.caption(f"Ağırlık ön ayarı: `{settings.scoring.track_b_preset}`")
        try:
            weights = resolve_preset("B", settings.scoring.track_b_preset)
            st.caption(" · ".join(f"{k}: {v:.2f}" for k, v in weights.items()))
        except (KeyError, ValueError):
            st.caption("⚠️ ön ayar tanımsız")
        if st.button("Verileri yenile", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    overview_tab, detail_tab, governance_tab, add_model_tab = st.tabs(
        ["📊 Genel bakış", "🔍 Detaylar", "🛡️ Yönetişim", "➕ Model Ekle"]
    )

    if frame.empty:
        with overview_tab:
            st.warning(
                "Kayıtlı değerlendirme yok. Sağdaki **Model Ekle** sekmesinden "
                "bir model ekleyip değerlendirebilir, ya da komut satırından "
                "sırasıyla şunları çalıştırabilirsiniz:\n\n"
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
