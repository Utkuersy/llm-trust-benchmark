"""Audit trail and test-data versioning tests.

The audit trail exists to answer a single question: "how did you arrive
at this result, prove it." The tests probe this claim in two ways:
    1. When the chain is genuinely tampered with, does ``verify_chain``
       catch it (otherwise the "immutable" claim is an empty promise)?
    2. Can versioning actually distinguish a data change from a model
       change?
"""

from __future__ import annotations

import pytest

from core.audit import GENESIS_HASH, _compute_hash, fetch_audit_log, record_run, verify_chain
from core.storage import connect
from core.versioning import dataset_components, dataset_version, drift_report


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Runs each test against its own SQLite file (so tests don't affect each other)."""
    db_path = tmp_path / "audit_test.db"
    monkeypatch.setenv("AITB__PATHS__DB_PATH", str(db_path))
    from core.config import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


# --------------------------------------------------------------------------- #
# Audit trail — chain correctness
# --------------------------------------------------------------------------- #
def test_first_entry_chains_to_genesis() -> None:
    """The first record must chain to the zero hash (genesis)."""
    entry = record_run("R1", "preset-a")
    assert entry.previous_hash == GENESIS_HASH
    assert entry.sequence == 1


def test_entries_chain_sequentially() -> None:
    """Each new record must chain to the previous one's hash."""
    first = record_run("R1", "preset-a")
    second = record_run("R2", "preset-a")
    third = record_run("R3", "preset-a")

    assert second.previous_hash == first.entry_hash
    assert third.previous_hash == second.entry_hash


def test_never_fabricates_code_version() -> None:
    """If git info can't be obtained, must return 'unknown', never a fabricated hash."""
    entry = record_run("R1", "preset-a")
    # This environment may or may not have a git repository; the only
    # guarantee is that the returned value is either a real git
    # identifier or "unknown".
    assert entry.code_version == "unknown" or len(entry.code_version) >= 4


def test_verify_chain_passes_on_untampered_log() -> None:
    """An untouched chain must always verify."""
    for i in range(5):
        record_run(f"R{i}", "preset-a")
    ok, problems = verify_chain()
    assert ok is True
    assert problems == []


def test_verify_chain_detects_content_tampering() -> None:
    """If a record's content is manually altered, it must be detected.

    This is the test that actually exercises the 'immutable' claim. The
    record is modified directly via SQL (the application API does not
    prevent this, because preventing it is impossible for anyone with
    database access), and the verification function is expected to
    catch it.
    """
    record_run("R1", "preset-a")
    record_run("R2", "preset-a")

    with connect() as connection:
        connection.execute(
            "UPDATE audit_log SET triggered_by = 'attacker' WHERE run_id = 'R1'"
        )

    ok, problems = verify_chain()
    assert ok is False
    assert any("hash" in problem.lower() for problem in problems)


def test_verify_chain_detects_deleted_middle_entry() -> None:
    """Deleting a record from the middle of the chain must break the chain for subsequent records."""
    record_run("R1", "preset-a")
    record_run("R2", "preset-a")
    record_run("R3", "preset-a")

    with connect() as connection:
        connection.execute("DELETE FROM audit_log WHERE run_id = 'R2'")

    ok, problems = verify_chain()
    assert ok is False
    assert len(problems) > 0


def test_verify_chain_empty_log_is_valid() -> None:
    """An empty audit trail must be considered valid (no run has happened yet)."""
    ok, problems = verify_chain()
    assert ok is True
    assert problems == []


def test_compute_hash_is_deterministic() -> None:
    """The same content must always produce the same hash."""
    payload = {"run_id": "X", "timestamp": "2026-01-01"}
    assert _compute_hash(GENESIS_HASH, payload) == _compute_hash(GENESIS_HASH, payload)


def test_compute_hash_changes_with_previous_hash() -> None:
    """The same content with a different previous hash must produce a different result (chaining works)."""
    payload = {"run_id": "X"}
    hash_a = _compute_hash(GENESIS_HASH, payload)
    hash_b = _compute_hash("f" * 64, payload)
    assert hash_a != hash_b


def test_fetch_audit_log_filters_by_run_id() -> None:
    """The run_id filter must return only the matching records."""
    record_run("TARGET", "preset-a")
    record_run("OTHER", "preset-a")

    filtered = fetch_audit_log(run_id="TARGET")
    assert len(filtered) == 1
    assert filtered[0]["run_id"] == "TARGET"


def test_audit_entry_records_who_and_when() -> None:
    """The who and when fields must not be empty."""
    entry = record_run("R1", "preset-a")
    assert entry.triggered_by
    assert entry.hostname
    assert entry.timestamp


# --------------------------------------------------------------------------- #
# Versioning
# --------------------------------------------------------------------------- #
def test_dataset_version_is_deterministic() -> None:
    """The same test data must produce the same version string every time."""
    assert dataset_version() == dataset_version()


def test_dataset_components_lists_all_pieces() -> None:
    """All components must be reported separately (for "which piece changed" questions)."""
    components = dataset_components()
    expected = {
        "rag_corpus", "content_lexicons", "math_problems",
        "injection_scenarios", "poisoning_cases",
    }
    assert expected.issubset(components.keys())


def test_dataset_version_changes_when_scenario_count_differs() -> None:
    """A different scenario set must produce a different hash."""
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
# Drift report
# --------------------------------------------------------------------------- #
def _insert_run(model: str, score: float, preset: str, created_at: str) -> None:
    """Inserts a minimal row directly into the track_b_results table for testing."""
    from core.schemas import EvaluationResult
    from core.storage import init_db, save_track_b

    init_db()
    result = EvaluationResult(run_id=f"run-{created_at}-{model}", model_name=model, config_name=preset)
    result.trust_score = score
    result.created_at = result.created_at.__class__.fromisoformat(created_at)
    save_track_b(result)


def test_drift_report_requires_at_least_two_runs() -> None:
    """A drift report must not be produced from a single run."""
    _insert_run("gpt4", 80.0, "preset-a", "2026-01-01T00:00:00+00:00")
    report = drift_report("gpt4")
    assert report["comparable_pairs"] == 0


def test_drift_report_flags_config_change_not_model() -> None:
    """If the weight preset changed, the difference must be attributed to configuration, not the model."""
    _insert_run("gpt4", 80.0, "preset-a", "2026-01-01T00:00:00+00:00")
    _insert_run("gpt4", 40.0, "preset-b", "2026-02-01T00:00:00+00:00")

    report = drift_report("gpt4")
    assert report["comparable_pairs"] == 1
    assert report["transitions"][0]["classification"] == "config_changed"


def test_drift_report_flags_real_drift_when_config_stable() -> None:
    """A large score gap with an unchanged configuration must count as real drift."""
    _insert_run("gpt4", 85.0, "preset-a", "2026-01-01T00:00:00+00:00")
    _insert_run("gpt4", 60.0, "preset-a", "2026-02-01T00:00:00+00:00")

    report = drift_report("gpt4")
    assert report["transitions"][0]["classification"] == "model_drift"
    assert report["drift_events"] == 1


def test_drift_report_ignores_small_fluctuation() -> None:
    """Small fluctuations below the threshold must not count as drift."""
    _insert_run("gpt4", 85.0, "preset-a", "2026-01-01T00:00:00+00:00")
    _insert_run("gpt4", 83.0, "preset-a", "2026-02-01T00:00:00+00:00")

    report = drift_report("gpt4")
    assert report["transitions"][0]["classification"] == "stable"
    assert report["drift_events"] == 0
