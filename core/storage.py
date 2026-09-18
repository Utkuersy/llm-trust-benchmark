"""SQLite depolama katmanı — Track A ve Track B'nin ortak sonuç deposu.

Şema (``db/benchmark.db``):

``runs``
    Her benchmark koşusunun başlık kaydı (track, model, trust score, ham JSON).
``track_a_results``
    Track A alt metriklerinin düzleştirilmiş hali (dashboard sorguları için).
``track_b_results``
    Track B alt metriklerinin düzleştirilmiş hali.
``findings``
    Her iki track'in ürettiği bulgular (Bandit, CVE, injection, PII...).

Tasarım kararları:
    * Yazmalar tek transaction içinde, ``INSERT OR REPLACE`` ile idempotenttir.
    * Tüm sorgular parametrelidir (SQL enjeksiyonu yüzeyi yok).
    * ``payload`` sütunu tam Pydantic modelini JSON olarak saklar; şema
      genişlediğinde geçmiş koşular kaybolmaz.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.schemas import Track, TrackAResult, TrackBResult

logger = get_logger(__name__)

SCHEMA_VERSION = 3

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    track        TEXT NOT NULL,
    model_name   TEXT NOT NULL,
    config_name  TEXT NOT NULL DEFAULT 'default',
    created_at   TEXT NOT NULL,
    trust_score  REAL NOT NULL DEFAULT 0,
    subscores    TEXT NOT NULL DEFAULT '{}',
    payload      TEXT NOT NULL DEFAULT '{}',
    mlflow_run_id TEXT
);

CREATE TABLE IF NOT EXISTS track_a_results (
    run_id                TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    model_name            TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    target_path           TEXT NOT NULL,
    quality_score         REAL DEFAULT 0,
    pylint_score          REAL DEFAULT 0,
    average_complexity    REAL DEFAULT 0,
    max_complexity        INTEGER DEFAULT 0,
    maintainability_index REAL DEFAULT 0,
    lines_of_code         INTEGER DEFAULT 0,
    correctness_score     REAL DEFAULT 0,
    tests_total           INTEGER DEFAULT 0,
    tests_passed          INTEGER DEFAULT 0,
    tests_failed          INTEGER DEFAULT 0,
    pass_rate             REAL DEFAULT 0,
    static_score          REAL DEFAULT 0,
    bandit_high           INTEGER DEFAULT 0,
    bandit_medium         INTEGER DEFAULT 0,
    bandit_low            INTEGER DEFAULT 0,
    dependency_score      REAL DEFAULT 0,
    cve_count             INTEGER DEFAULT 0,
    runtime_score         REAL DEFAULT 0,
    network_attempts      INTEGER DEFAULT 0,
    file_violations       INTEGER DEFAULT 0,
    subprocess_attempts   INTEGER DEFAULT 0,
    dynamic_code_events   INTEGER DEFAULT 0,
    timed_out             INTEGER DEFAULT 0,
    trust_score           REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS track_b_results (
    run_id               TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    model_name           TEXT NOT NULL,
    config_name          TEXT NOT NULL DEFAULT 'default',
    created_at           TEXT NOT NULL,
    retrieval_score      REAL DEFAULT 0,
    context_precision    REAL DEFAULT 0,
    context_recall       REAL DEFAULT 0,
    mrr                  REAL DEFAULT 0,
    hit_rate             REAL DEFAULT 0,
    generation_score     REAL DEFAULT 0,
    faithfulness         REAL DEFAULT 0,
    answer_relevance     REAL DEFAULT 0,
    hallucination_rate   REAL DEFAULT 0,
    injection_score      REAL DEFAULT 0,
    injection_failure_rate REAL DEFAULT 0,
    scenarios_run        INTEGER DEFAULT 0,
    scenarios_failed     INTEGER DEFAULT 0,
    pii_score            REAL DEFAULT 0,
    pii_hits             INTEGER DEFAULT 0,
    poisoning_score      REAL DEFAULT 0,
    susceptibility_rate  REAL DEFAULT 0,
    content_safety_score REAL DEFAULT 0,
    content_flagged_rate REAL DEFAULT 0,
    content_hits         INTEGER DEFAULT 0,
    math_score           REAL DEFAULT 0,
    math_accuracy        REAL DEFAULT 0,
    pipeline_stages      INTEGER DEFAULT 0,
    pipeline_failed      INTEGER DEFAULT 0,
    pipeline_bottleneck  TEXT DEFAULT '',
    pipeline_findings    INTEGER DEFAULT 0,
    trust_score          REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS findings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    track      TEXT NOT NULL,
    model_name TEXT NOT NULL,
    category   TEXT NOT NULL,
    severity   TEXT NOT NULL DEFAULT 'INFO',
    title      TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    location   TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_runs_track_model ON runs(track, model_name);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_findings_category ON findings(category);
"""


def resolve_db_path(settings: Settings | None = None) -> Path:
    """Konfigürasyondaki veritabanı yolunu mutlaklaştırır ve klasörü oluşturur."""
    settings = settings or get_settings()
    path = settings.paths.db_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connect(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    """Yapılandırılmış bir SQLite bağlantısı açar (foreign key açık)."""
    path = resolve_db_path(settings)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    # Şema v3'te eklenen sütunlar. Mevcut bir veritabanı CREATE TABLE
    # IF NOT EXISTS ile güncellenmez; eksik sütunlar ALTER TABLE ile eklenir.
    "track_b_results": [
        ("content_safety_score", "REAL DEFAULT 0"),
        ("content_flagged_rate", "REAL DEFAULT 0"),
        ("content_hits", "INTEGER DEFAULT 0"),
        ("math_score", "REAL DEFAULT 0"),
        ("math_accuracy", "REAL DEFAULT 0"),
        ("pipeline_stages", "INTEGER DEFAULT 0"),
        ("pipeline_failed", "INTEGER DEFAULT 0"),
        ("pipeline_bottleneck", "TEXT DEFAULT ''"),
        ("pipeline_findings", "INTEGER DEFAULT 0"),
    ],
}


def _migrate(connection: sqlite3.Connection) -> list[str]:
    """Eksik sütunları ekler; eski veritabanları veri kaybı olmadan güncellenir."""
    applied: list[str] = []
    for table, columns in _ADDED_COLUMNS.items():
        existing = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if not existing:
            continue
        for name, definition in columns:
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
                applied.append(f"{table}.{name}")
    return applied


def init_db(settings: Settings | None = None) -> Path:
    """Şemayı oluşturur, gerekirse göç uygular (idempotent)."""
    path = resolve_db_path(settings)
    with connect(settings) as connection:
        connection.executescript(_SCHEMA)
        migrated = _migrate(connection)
        if migrated:
            logger.info("sema gocu uygulandi", extra={"columns": migrated})
        connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
    logger.info("veritabani hazir", extra={"db_path": str(path), "schema": SCHEMA_VERSION})
    return path


def _insert_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    track: Track,
    model_name: str,
    config_name: str,
    created_at: str,
    trust_score: float,
    subscores: dict[str, float],
    payload: dict[str, Any],
    mlflow_run_id: str | None,
) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO runs
            (run_id, track, model_name, config_name, created_at,
             trust_score, subscores, payload, mlflow_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            track.value,
            model_name,
            config_name,
            created_at,
            float(trust_score),
            json.dumps(subscores, ensure_ascii=False),
            json.dumps(payload, ensure_ascii=False, default=str),
            mlflow_run_id,
        ),
    )


def _insert_findings(
    connection: sqlite3.Connection, run_id: str, track: Track, model_name: str,
    findings: list[dict[str, Any]],
) -> None:
    connection.execute("DELETE FROM findings WHERE run_id = ?", (run_id,))
    connection.executemany(
        """
        INSERT INTO findings (run_id, track, model_name, category, severity, title, detail, location)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                track.value,
                model_name,
                str(item.get("category", "other")),
                str(item.get("severity", "INFO")).upper(),
                str(item.get("title", ""))[:300],
                str(item.get("detail", ""))[:1000],
                str(item.get("location", ""))[:300],
            )
            for item in findings
        ],
    )


def save_track_a(
    result: TrackAResult,
    findings: list[dict[str, Any]] | None = None,
    mlflow_run_id: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Track A sonucunu runs + track_a_results + findings tablolarına yazar."""
    created_at = result.created_at.isoformat()
    with connect(settings) as connection:
        _insert_run(
            connection,
            run_id=result.run_id,
            track=Track.A,
            model_name=result.model_name,
            config_name="default",
            created_at=created_at,
            trust_score=result.trust_score,
            subscores=result.subscores,
            payload=json.loads(result.model_dump_json()),
            mlflow_run_id=mlflow_run_id,
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO track_a_results VALUES
            (?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?,?, ?)
            """,
            (
                result.run_id,
                result.model_name,
                created_at,
                result.target_path,
                result.quality.score,
                result.quality.pylint_score,
                result.quality.average_complexity,
                result.quality.max_complexity,
                result.quality.maintainability_index,
                result.quality.lines_of_code,
                result.correctness.score,
                result.correctness.total,
                result.correctness.passed,
                result.correctness.failed,
                result.correctness.pass_rate,
                result.static_security.score,
                result.static_security.high,
                result.static_security.medium,
                result.static_security.low,
                result.dependency_security.score,
                result.dependency_security.total_vulnerabilities,
                result.runtime_security.score,
                len(result.runtime_security.network_attempts),
                len(result.runtime_security.file_write_violations),
                len(result.runtime_security.subprocess_attempts),
                len(result.runtime_security.dynamic_code_events),
                int(result.runtime_security.timed_out),
                result.trust_score,
            ),
        )
        _insert_findings(connection, result.run_id, Track.A, result.model_name, findings or [])
    logger.info(
        "track A sonucu kaydedildi",
        extra={"run_id": result.run_id, "model": result.model_name, "score": result.trust_score},
    )


def save_track_b(
    result: TrackBResult,
    findings: list[dict[str, Any]] | None = None,
    mlflow_run_id: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Track B sonucunu runs + track_b_results + findings tablolarına yazar."""
    created_at = result.created_at.isoformat()
    with connect(settings) as connection:
        _insert_run(
            connection,
            run_id=result.run_id,
            track=Track.B,
            model_name=result.model_name,
            config_name=result.config_name,
            created_at=created_at,
            trust_score=result.trust_score,
            subscores=result.subscores,
            payload=json.loads(result.model_dump_json()),
            mlflow_run_id=mlflow_run_id,
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO track_b_results (
                run_id, model_name, config_name, created_at,
                retrieval_score, context_precision, context_recall, mrr, hit_rate,
                generation_score, faithfulness, answer_relevance, hallucination_rate,
                injection_score, injection_failure_rate, scenarios_run, scenarios_failed,
                pii_score, pii_hits,
                poisoning_score, susceptibility_rate,
                content_safety_score, content_flagged_rate, content_hits,
                math_score, math_accuracy,
                pipeline_stages, pipeline_failed, pipeline_bottleneck, pipeline_findings,
                trust_score
            ) VALUES (
                :run_id, :model_name, :config_name, :created_at,
                :retrieval_score, :context_precision, :context_recall, :mrr, :hit_rate,
                :generation_score, :faithfulness, :answer_relevance, :hallucination_rate,
                :injection_score, :injection_failure_rate, :scenarios_run, :scenarios_failed,
                :pii_score, :pii_hits,
                :poisoning_score, :susceptibility_rate,
                :content_safety_score, :content_flagged_rate, :content_hits,
                :math_score, :math_accuracy,
                :pipeline_stages, :pipeline_failed, :pipeline_bottleneck, :pipeline_findings,
                :trust_score
            )
            """,
            {
                "run_id": result.run_id,
                "model_name": result.model_name,
                "config_name": result.config_name,
                "created_at": created_at,
                "retrieval_score": result.retrieval.score,
                "context_precision": result.retrieval.context_precision,
                "context_recall": result.retrieval.context_recall,
                "mrr": result.retrieval.mrr,
                "hit_rate": result.retrieval.hit_rate,
                "generation_score": result.generation.score,
                "faithfulness": result.generation.faithfulness,
                "answer_relevance": result.generation.answer_relevance,
                "hallucination_rate": result.generation.hallucination_rate,
                "injection_score": result.injection.score,
                "injection_failure_rate": result.injection.failure_rate,
                "scenarios_run": result.injection.scenarios_run,
                "scenarios_failed": result.injection.scenarios_failed,
                "pii_score": result.pii.score,
                "pii_hits": result.pii.total_hits,
                "poisoning_score": result.poisoning.score,
                "susceptibility_rate": result.poisoning.susceptibility_rate,
                "content_safety_score": result.content_safety.score,
                "content_flagged_rate": result.content_safety.flagged_rate,
                "content_hits": result.content_safety.total_hits,
                "math_score": result.math.score,
                "math_accuracy": result.math.accuracy,
                "pipeline_stages": int(result.pipeline.get("stage_count", 0) or 0),
                "pipeline_failed": len(result.pipeline.get("failed_stages", []) or []),
                "pipeline_bottleneck": str(result.pipeline.get("bottleneck", "")),
                "pipeline_findings": int(result.pipeline.get("total_findings", 0) or 0),
                "trust_score": result.trust_score,
            },
        )
        _insert_findings(connection, result.run_id, Track.B, result.model_name, findings or [])
    logger.info(
        "track B sonucu kaydedildi",
        extra={"run_id": result.run_id, "model": result.model_name, "score": result.trust_score},
    )


# --------------------------------------------------------------------------- #
# Okuma yardımcıları (dashboard için)
# --------------------------------------------------------------------------- #
def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def fetch_runs(track: Track | str | None = None, limit: int = 500,
               settings: Settings | None = None) -> list[dict[str, Any]]:
    """Koşu başlıklarını (en yeni önce) döndürür."""
    track_value = track.value if isinstance(track, Track) else track
    with connect(settings) as connection:
        if track_value:
            rows = connection.execute(
                "SELECT * FROM runs WHERE track = ? ORDER BY created_at DESC LIMIT ?",
                (track_value, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return _rows_to_dicts(rows)


def fetch_track_a(limit: int = 500, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Track A düzleştirilmiş sonuçları."""
    with connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM track_a_results ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return _rows_to_dicts(rows)


def fetch_track_b(limit: int = 500, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Track B düzleştirilmiş sonuçları."""
    with connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM track_b_results ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return _rows_to_dicts(rows)


def delete_model(model_name: str, settings: Settings | None = None) -> int:
    """Bir modele ait tüm koşu kayıtlarını siler.

    ``runs`` satırları silinince ``track_a_results``, ``track_b_results``
    ve ``findings`` tablolarındaki ilgili satırlar ``ON DELETE CASCADE``
    ile otomatik silinir (bkz. ``connect()`` — foreign_keys açık).
    """
    with connect(settings) as connection:
        cursor = connection.execute("DELETE FROM runs WHERE model_name = ?", (model_name,))
        return cursor.rowcount


def fetch_findings(
    run_id: str | None = None, track: Track | str | None = None,
    limit: int = 2000, settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Bulguları filtreli olarak döndürür."""
    clauses: list[str] = []
    params: list[Any] = []
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if track:
        clauses.append("track = ?")
        params.append(track.value if isinstance(track, Track) else track)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with connect(settings) as connection:
        rows = connection.execute(
            f"SELECT * FROM findings {where} ORDER BY id DESC LIMIT ?", params  # noqa: S608
        ).fetchall()
    return _rows_to_dicts(rows)


def latest_run_payload(
    track: Track | str, model_name: str, settings: Settings | None = None
) -> dict[str, Any] | None:
    """Bir model için en son koşunun tam JSON payload'unu döndürür."""
    track_value = track.value if isinstance(track, Track) else track
    with connect(settings) as connection:
        row = connection.execute(
            """
            SELECT payload FROM runs
            WHERE track = ? AND model_name = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (track_value, model_name),
        ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["payload"])
    except (ValueError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    print(f"veritabani hazirlandi: {init_db()}")  # noqa: T201
