"""Denetim izi (audit trail) ve test verisi versiyonlama testleri.

Denetim izinin varlık gerekçesi tek bir soruya cevap vermektir: "bu sonucu
nasıl elde ettin, kanıtla." Testler bu iddiayı iki şekilde sınar:
    1. Zincir gerçekten bozulunca ``verify_chain`` bunu yakalıyor mu
       (aksi halde "değiştirilemez" iddiası boş bir sözdür).
    2. Versiyonlama, veri değişikliği ile model değişikliğini gerçekten
       ayırt edebiliyor mu.
"""

from __future__ import annotations

import pytest

from core.audit import GENESIS_HASH, _compute_hash, fetch_audit_log, record_run, verify_chain
from core.storage import connect
from core.versioning import dataset_components, dataset_version, drift_report


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Her testi ayrı bir SQLite dosyasında çalıştırır (testler birbirini etkilemesin)."""
    db_path = tmp_path / "audit_test.db"
    monkeypatch.setenv("AITB__PATHS__DB_PATH", str(db_path))
    from core.config import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


# --------------------------------------------------------------------------- #
# Denetim izi — zincir doğruluğu
# --------------------------------------------------------------------------- #
def test_first_entry_chains_to_genesis() -> None:
    """İlk kayıt, sıfır hash'e (genesis) zincirlenmeli."""
    entry = record_run("R1", "preset-a")
    assert entry.previous_hash == GENESIS_HASH
    assert entry.sequence == 1


def test_entries_chain_sequentially() -> None:
    """Her yeni kayıt bir öncekinin hash'ine zincirlenmeli."""
    first = record_run("R1", "preset-a")
    second = record_run("R2", "preset-a")
    third = record_run("R3", "preset-a")

    assert second.previous_hash == first.entry_hash
    assert third.previous_hash == second.entry_hash


def test_never_fabricates_code_version() -> None:
    """Git bilgisi alınamazsa 'unknown' dönmeli, asla uydurulmuş bir hash değil."""
    entry = record_run("R1", "preset-a")
    # Bu ortamda git deposu olabilir de olmayabilir de; tek garanti,
    # dönen değerin ya gerçek bir git tanımlayıcısı ya da "unknown" olması.
    assert entry.code_version == "unknown" or len(entry.code_version) >= 4


def test_verify_chain_passes_on_untampered_log() -> None:
    """Dokunulmamış bir zincir her zaman doğrulanmalı."""
    for i in range(5):
        record_run(f"R{i}", "preset-a")
    ok, problems = verify_chain()
    assert ok is True
    assert problems == []


def test_verify_chain_detects_content_tampering() -> None:
    """Bir kaydın içeriği elle değiştirilirse tespit edilmeli.

    Bu, 'değiştirilemez' iddiasının gerçekten sınandığı testtir. Kayıt
    doğrudan SQL ile değiştirilir (uygulama API'si bunu engellemez, çünkü
    engellemek veritabanı erişimi olan biri için imkansızdır) ve doğrulama
    fonksiyonunun bunu yakalaması beklenir.
    """
    record_run("R1", "preset-a")
    record_run("R2", "preset-a")

    with connect() as connection:
        connection.execute(
            "UPDATE audit_log SET triggered_by = 'saldirgan' WHERE run_id = 'R1'"
        )

    ok, problems = verify_chain()
    assert ok is False
    assert any("hash" in problem.lower() for problem in problems)


def test_verify_chain_detects_deleted_middle_entry() -> None:
    """Zincirin ortasından bir kayıt silinirse sonraki kayıtların zinciri kopmalı."""
    record_run("R1", "preset-a")
    record_run("R2", "preset-a")
    record_run("R3", "preset-a")

    with connect() as connection:
        connection.execute("DELETE FROM audit_log WHERE run_id = 'R2'")

    ok, problems = verify_chain()
    assert ok is False
    assert len(problems) > 0


def test_verify_chain_empty_log_is_valid() -> None:
    """Boş bir denetim izi geçerli sayılmalı (henüz koşu yapılmamış)."""
    ok, problems = verify_chain()
    assert ok is True
    assert problems == []


def test_compute_hash_is_deterministic() -> None:
    """Aynı içerik her zaman aynı hash'i üretmeli."""
    payload = {"run_id": "X", "timestamp": "2026-01-01"}
    assert _compute_hash(GENESIS_HASH, payload) == _compute_hash(GENESIS_HASH, payload)


def test_compute_hash_changes_with_previous_hash() -> None:
    """Aynı içerik farklı önceki hash ile farklı sonuç üretmeli (zincirleme çalışıyor)."""
    payload = {"run_id": "X"}
    hash_a = _compute_hash(GENESIS_HASH, payload)
    hash_b = _compute_hash("f" * 64, payload)
    assert hash_a != hash_b


def test_fetch_audit_log_filters_by_run_id() -> None:
    """run_id filtresi yalnızca ilgili kayıtları döndürmeli."""
    record_run("TARGET", "preset-a")
    record_run("OTHER", "preset-a")

    filtered = fetch_audit_log(run_id="TARGET")
    assert len(filtered) == 1
    assert filtered[0]["run_id"] == "TARGET"


def test_audit_entry_records_who_and_when() -> None:
    """Kim ve ne zaman bilgisi boş olmamalı."""
    entry = record_run("R1", "preset-a")
    assert entry.triggered_by
    assert entry.hostname
    assert entry.timestamp


# --------------------------------------------------------------------------- #
# Versiyonlama
# --------------------------------------------------------------------------- #
def test_dataset_version_is_deterministic() -> None:
    """Aynı test verisi her seferinde aynı sürüm dizgesini üretmeli."""
    assert dataset_version() == dataset_version()


def test_dataset_components_lists_all_pieces() -> None:
    """Tüm bileşenler ayrı ayrı raporlanmalı (hangi parça değişti sorusu için)."""
    components = dataset_components()
    expected = {
        "rag_corpus", "content_lexicons", "math_problems",
        "injection_scenarios", "poisoning_cases",
    }
    assert expected.issubset(components.keys())


def test_dataset_version_changes_when_scenario_count_differs() -> None:
    """Farklı bir senaryo kümesi farklı bir hash üretmeli."""
    from core.versioning import _hash_module_constant
    from llm_security.prompt_injection_tests import InjectionScenario

    original = _hash_module_constant("llm_security.prompt_injection_tests", "SCENARIOS")

    import llm_security.prompt_injection_tests as module

    extra_scenario = InjectionScenario(
        scenario_id="TEST-EXTRA", name="test", category="test", severity="LOW", prompt="x",
    )
    original_scenarios = module.SCENARIOS
    module.SCENARIOS = (*original_scenarios, extra_scenario)
    try:
        changed = _hash_module_constant("llm_security.prompt_injection_tests", "SCENARIOS")
        assert changed != original
    finally:
        module.SCENARIOS = original_scenarios


# --------------------------------------------------------------------------- #
# Drift raporu
# --------------------------------------------------------------------------- #
def _insert_run(model: str, score: float, preset: str, created_at: str) -> None:
    """Test için doğrudan track_b_results tablosuna minimal bir satır ekler."""
    from core.schemas import EvaluationResult
    from core.storage import init_db, save_track_b

    init_db()
    result = EvaluationResult(run_id=f"run-{created_at}-{model}", model_name=model, config_name=preset)
    result.trust_score = score
    result.created_at = result.created_at.__class__.fromisoformat(created_at)
    save_track_b(result)


def test_drift_report_requires_at_least_two_runs() -> None:
    """Tek koşuyla drift raporu üretilmemeli."""
    _insert_run("gpt4", 80.0, "preset-a", "2026-01-01T00:00:00+00:00")
    report = drift_report("gpt4")
    assert report["comparable_pairs"] == 0


def test_drift_report_flags_config_change_not_model() -> None:
    """Ağırlık ön ayarı değiştiyse fark modele değil konfigürasyona atfedilmeli."""
    _insert_run("gpt4", 80.0, "preset-a", "2026-01-01T00:00:00+00:00")
    _insert_run("gpt4", 40.0, "preset-b", "2026-02-01T00:00:00+00:00")

    report = drift_report("gpt4")
    assert report["comparable_pairs"] == 1
    assert report["transitions"][0]["classification"] == "config_changed"


def test_drift_report_flags_real_drift_when_config_stable() -> None:
    """Konfigürasyon aynıyken büyük puan farkı gerçek drift sayılmalı."""
    _insert_run("gpt4", 85.0, "preset-a", "2026-01-01T00:00:00+00:00")
    _insert_run("gpt4", 60.0, "preset-a", "2026-02-01T00:00:00+00:00")

    report = drift_report("gpt4")
    assert report["transitions"][0]["classification"] == "model_drift"
    assert report["drift_events"] == 1


def test_drift_report_ignores_small_fluctuation() -> None:
    """Eşiğin altındaki küçük dalgalanmalar drift sayılmamalı."""
    _insert_run("gpt4", 85.0, "preset-a", "2026-01-01T00:00:00+00:00")
    _insert_run("gpt4", 83.0, "preset-a", "2026-02-01T00:00:00+00:00")

    report = drift_report("gpt4")
    assert report["transitions"][0]["classification"] == "stable"
    assert report["drift_events"] == 0
