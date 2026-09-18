"""Denetim izi (audit trail) katmanı.

Her benchmark koşusu, **kim, ne zaman, hangi kod ve konfigürasyonla**
çalıştırdı sorusuna kanıt sunan bir kayıt bırakır. Bu, NIST AI RMF'nin
Govern fonksiyonu ve ISO/IEC 42001'in 9.2 (iç denetim) maddesinin somut
karşılığıdır — bkz. ``docs/GOVERNANCE_ALIGNMENT.md``.

**Değiştirilemezlik (tamper-evidence).** Bu bir blockchain değildir; amaç
dağıtık mutabakat değil, **sonradan sessizce değiştirilmiş bir kaydı
tespit edebilmektir**. Her girdi bir önceki girdinin hash'ini içerir
(basit hash zinciri). Zincirin ortasındaki bir satır değiştirilirse veya
silinirse, ondan sonraki tüm satırların hash'i tutmaz ve ``verify_chain()``
bunu tespit eder.

Kayıt edilenler:
    * kim   — işletim sistemi kullanıcısı, hostname
    * ne zaman — UTC zaman damgası
    * hangi kod — git commit hash (varsa), yoksa "unknown" (asla uydurulmaz)
    * hangi konfigürasyon — settings.yaml içeriğinin SHA-256'sı
    * hangi test verisi — korpus + senaryo + sözlük dosyalarının birleşik
      hash'i (bkz. ``core/versioning.py``)

Kullanım::

    entry = record_run(run_id="E-gpt4-...", preset="output_safety_first")
    verify_chain()  # -> (True, []) ya da (False, ["satır 4 tutarsız"])
"""

from __future__ import annotations

import getpass
import hashlib
import json
import socket
import subprocess  # nosec B404 - sabit git komutu, shell=False
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.storage import connect

logger = get_logger(__name__)

GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    """Tek bir denetim izi kaydı."""

    sequence: int
    entry_hash: str
    previous_hash: str
    run_id: str
    timestamp: str
    triggered_by: str
    hostname: str
    code_version: str
    config_hash: str
    dataset_version: str
    preset: str
    scoring_version: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serileştirilebilir gösterim."""
        return {
            "sequence": self.sequence,
            "entry_hash": self.entry_hash,
            "previous_hash": self.previous_hash,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "triggered_by": self.triggered_by,
            "hostname": self.hostname,
            "code_version": self.code_version,
            "config_hash": self.config_hash,
            "dataset_version": self.dataset_version,
            "preset": self.preset,
            "scoring_version": self.scoring_version,
            "extra": self.extra,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    sequence         INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_hash       TEXT NOT NULL UNIQUE,
    previous_hash    TEXT NOT NULL,
    run_id           TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    triggered_by     TEXT NOT NULL,
    hostname         TEXT NOT NULL,
    code_version     TEXT NOT NULL,
    config_hash      TEXT NOT NULL,
    dataset_version  TEXT NOT NULL,
    preset           TEXT NOT NULL,
    scoring_version  TEXT NOT NULL,
    extra            TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_log(run_id);
"""


def init_audit_schema(settings: Settings | None = None) -> None:
    """Denetim izi tablosunu oluşturur (idempotent)."""
    with connect(settings) as connection:
        connection.executescript(_SCHEMA)


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - bazi konteynerlerde kullanici adi cozulemez
        return "unknown"


def _current_host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


@lru_cache(maxsize=1)
def _git_commit() -> str:
    """Mevcut git commit hash'ini döndürür; yoksa 'unknown' — asla uydurulmaz.

    Süreç başına bir kez hesaplanıp önbelleğe alınır: git durumu bir
    sürecin ömrü boyunca değişmez, her kayıtta yeniden subprocess
    çağırmanın hem performans hem dosya tanıtıcısı maliyeti var.
    """
    try:
        result = subprocess.run(  # noqa: S603 # nosec B603, B607
            ["git", "describe", "--always", "--dirty", "--broken"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _config_hash(settings: Settings) -> str:
    """settings.yaml içeriğinin SHA-256'sı — konfigürasyon değişikliği tespiti."""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if not config_path.exists():
        return "missing"
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def _last_entry(connection: Any) -> tuple[int, str]:
    """Zincirdeki son kaydın (sequence, hash) değerini döndürür."""
    row = connection.execute(
        "SELECT sequence, entry_hash FROM audit_log ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return 0, GENESIS_HASH
    return int(row["sequence"]), str(row["entry_hash"])


def _compute_hash(previous_hash: str, payload: dict[str, Any]) -> str:
    """Bir kaydın hash'ini önceki hash + kendi içeriğinden hesaplar."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest_input = f"{previous_hash}|{canonical}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def record_run(
    run_id: str,
    preset: str,
    dataset_version: str = "unknown",
    extra: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> AuditEntry:
    """Bir benchmark koşusu için denetim izi kaydı ekler (zincire eklenir)."""
    settings = settings or get_settings()
    init_audit_schema(settings)

    from core.scoring import SCORING_VERSION  # noqa: PLC0415 - dairesel import onleme

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "triggered_by": _current_user(),
        "hostname": _current_host(),
        "code_version": _git_commit(),
        "config_hash": _config_hash(settings),
        "dataset_version": dataset_version,
        "preset": preset,
        "scoring_version": SCORING_VERSION,
        "extra": extra or {},
    }

    with connect(settings) as connection:
        sequence, previous_hash = _last_entry(connection)
        entry_hash = _compute_hash(previous_hash, payload)
        connection.execute(
            """
            INSERT INTO audit_log
                (entry_hash, previous_hash, run_id, timestamp, triggered_by,
                 hostname, code_version, config_hash, dataset_version, preset,
                 scoring_version, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_hash, previous_hash, payload["run_id"], payload["timestamp"],
                payload["triggered_by"], payload["hostname"], payload["code_version"],
                payload["config_hash"], payload["dataset_version"], payload["preset"],
                payload["scoring_version"], json.dumps(payload["extra"], ensure_ascii=False),
            ),
        )

    entry = AuditEntry(sequence=sequence + 1, entry_hash=entry_hash, previous_hash=previous_hash, **payload)
    logger.info(
        "denetim izi kaydedildi",
        extra={"run_id": run_id, "sequence": entry.sequence, "code_version": entry.code_version},
    )
    return entry


def fetch_audit_log(run_id: str | None = None, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Denetim izi kayıtlarını döndürür (opsiyonel run_id filtresiyle)."""
    settings = settings or get_settings()
    init_audit_schema(settings)
    with connect(settings) as connection:
        if run_id:
            rows = connection.execute(
                "SELECT * FROM audit_log WHERE run_id = ? ORDER BY sequence", (run_id,)
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM audit_log ORDER BY sequence").fetchall()
    entries = []
    for row in rows:
        item = dict(row)
        try:
            item["extra"] = json.loads(item.get("extra", "{}"))
        except (ValueError, json.JSONDecodeError):
            item["extra"] = {}
        entries.append(item)
    return entries


def verify_chain(settings: Settings | None = None) -> tuple[bool, list[str]]:
    """Zincirin bütünlüğünü doğrular; kurcalanmış satırları raporlar.

    Her kaydın hash'i (önceki_hash + kendi_içeriği)'nden yeniden hesaplanır
    ve saklanan hash ile karşılaştırılır. Uyuşmazlık, o satırın veya ondan
    önceki bir satırın değiştirildiği anlamına gelir.
    """
    entries = fetch_audit_log(settings=settings)
    if not entries:
        return True, []

    problems: list[str] = []
    expected_previous = GENESIS_HASH

    for entry in entries:
        if entry["previous_hash"] != expected_previous:
            problems.append(
                f"sequence {entry['sequence']}: previous_hash zincirle uyusmuyor "
                f"(beklenen {expected_previous[:12]}..., bulunan {entry['previous_hash'][:12]}...)"
            )

        payload = {
            "run_id": entry["run_id"],
            "timestamp": entry["timestamp"],
            "triggered_by": entry["triggered_by"],
            "hostname": entry["hostname"],
            "code_version": entry["code_version"],
            "config_hash": entry["config_hash"],
            "dataset_version": entry["dataset_version"],
            "preset": entry["preset"],
            "scoring_version": entry["scoring_version"],
            "extra": entry["extra"],
        }
        recomputed = _compute_hash(entry["previous_hash"], payload)
        if recomputed != entry["entry_hash"]:
            problems.append(
                f"sequence {entry['sequence']}: icerik hash'i tutmuyor — "
                f"kayit degistirilmis olabilir"
            )

        expected_previous = entry["entry_hash"]

    ok = not problems
    if ok:
        logger.info("denetim izi zinciri dogrulandi", extra={"entries": len(entries)})
    else:
        logger.error("denetim izi zincirinde tutarsizlik", extra={"problems": problems})
    return ok, problems


def main() -> None:
    """CLI: denetim izini listele veya doğrula."""
    import argparse

    parser = argparse.ArgumentParser(description="Denetim izi araci")
    parser.add_argument("action", choices=["list", "verify"])
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    if args.action == "verify":
        ok, problems = verify_chain()
        if ok:
            print("Zincir bütünlüğü doğrulandı, kurcalama tespit edilmedi.")  # noqa: T201
        else:
            print("UYARI — zincirde tutarsızlık tespit edildi:")  # noqa: T201
            for problem in problems:
                print(f"  - {problem}")  # noqa: T201
            sys.exit(1)
    else:
        for entry in fetch_audit_log(args.run_id):
            print(  # noqa: T201
                f"[{entry['sequence']:04d}] {entry['timestamp']} | {entry['run_id']} | "
                f"kullanıcı={entry['triggered_by']}@{entry['hostname']} | "
                f"kod={entry['code_version']} | preset={entry['preset']}"
            )


if __name__ == "__main__":
    main()
