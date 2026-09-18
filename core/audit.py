"""The audit trail layer.

Every benchmark run leaves a record that provides evidence for the
question **who ran it, when, with which code, and with which
configuration**. This is the concrete counterpart to the NIST AI RMF's
Govern function and ISO/IEC 42001's clause 9.2 (internal audit) — see
``docs/GOVERNANCE_ALIGNMENT.md``.

**Tamper-evidence.** This is not a blockchain; the goal is not
distributed consensus, but **being able to detect a record that was
silently altered afterward**. Each entry contains the hash of the
previous entry (a simple hash chain). If a line in the middle of the
chain is modified or deleted, the hash of every line after it stops
matching, and ``verify_chain()`` detects this.

What is recorded:
    * who        — the OS user, hostname
    * when       — a UTC timestamp
    * which code — the git commit hash (if available), otherwise
      "unknown" (never fabricated)
    * which configuration — the SHA-256 of settings.yaml's contents
    * which test data — the combined hash of the corpus + scenario +
      lexicon files (see ``core/versioning.py``)

Usage::

    entry = record_run(run_id="E-gpt4-...", preset="output_safety_first")
    verify_chain()  # -> (True, []) or (False, ["line 4 inconsistent"])
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import socket
import subprocess  # nosec B404 - a fixed git command, shell=False
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.storage import connect

logger = get_logger(__name__)

GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    """A single audit trail record."""

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
        """A serializable representation."""
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
    """Creates the audit trail table (idempotent)."""
    with connect(settings) as connection:
        connection.executescript(_SCHEMA)


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _current_host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


@lru_cache(maxsize=1)
def _git_commit() -> str:
    """Returns the current git commit hash; 'unknown' if unavailable — never fabricated.

    Computed and cached once per process: git state doesn't change over
    a process's lifetime, and calling a subprocess again on every record
    has both a performance and a file-handle cost.
    """
    try:
        result = subprocess.run(  # nosec B603, B607
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


def _config_hash(_settings: Settings) -> str:
    """The SHA-256 of settings.yaml's contents — for detecting configuration changes.

    Hashes the custom config file given via ``AITB_CONFIG_PATH`` if set,
    otherwise the default — the same resolution as ``get_settings()``.
    """
    config_path = Path(os.environ.get("AITB_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        return "missing"
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def _last_entry(connection: Any) -> tuple[int, str]:
    """Returns the (sequence, hash) of the last entry in the chain."""
    row = connection.execute(
        "SELECT sequence, entry_hash FROM audit_log ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return 0, GENESIS_HASH
    return int(row["sequence"]), str(row["entry_hash"])


def _compute_hash(previous_hash: str, payload: dict[str, Any]) -> str:
    """Computes a record's hash from the previous hash + its own content."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest_input = f"{previous_hash}|{canonical}".encode()
    return hashlib.sha256(digest_input).hexdigest()


def record_run(
    run_id: str,
    preset: str,
    dataset_version: str = "unknown",
    extra: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> AuditEntry:
    """Adds an audit trail record for a benchmark run (appended to the chain)."""
    settings = settings or get_settings()
    init_audit_schema(settings)

    from core.scoring import SCORING_VERSION

    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
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

    entry = AuditEntry(
        sequence=sequence + 1,
        entry_hash=entry_hash,
        previous_hash=previous_hash,
        run_id=payload["run_id"],
        timestamp=payload["timestamp"],
        triggered_by=payload["triggered_by"],
        hostname=payload["hostname"],
        code_version=payload["code_version"],
        config_hash=payload["config_hash"],
        dataset_version=payload["dataset_version"],
        preset=payload["preset"],
        scoring_version=payload["scoring_version"],
        extra=payload["extra"],
    )
    logger.info(
        "audit trail record written",
        extra={"run_id": run_id, "sequence": entry.sequence, "code_version": entry.code_version},
    )
    return entry


def fetch_audit_log(run_id: str | None = None, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Returns audit trail records (optionally filtered by run_id)."""
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
    """Verifies the chain's integrity; reports any tampered lines.

    Each record's hash is recomputed from (previous_hash + its own
    content) and compared against the stored hash. A mismatch means that
    line, or a line before it, was modified.
    """
    entries = fetch_audit_log(settings=settings)
    if not entries:
        return True, []

    problems: list[str] = []
    expected_previous = GENESIS_HASH

    for entry in entries:
        if entry["previous_hash"] != expected_previous:
            problems.append(
                f"sequence {entry['sequence']}: previous_hash does not match the chain "
                f"(expected {expected_previous[:12]}..., found {entry['previous_hash'][:12]}...)"
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
                f"sequence {entry['sequence']}: content hash does not match — "
                f"the record may have been altered"
            )

        expected_previous = entry["entry_hash"]

    ok = not problems
    if ok:
        logger.info("audit trail chain verified", extra={"entries": len(entries)})
    else:
        logger.error("inconsistency found in the audit trail chain", extra={"problems": problems})
    return ok, problems


def main() -> None:
    """CLI: list or verify the audit trail."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit trail tool")
    parser.add_argument("action", choices=["list", "verify"])
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    if args.action == "verify":
        ok, problems = verify_chain()
        if ok:
            print("Chain integrity verified, no tampering detected.")
        else:
            print("WARNING — inconsistency detected in the chain:")
            for problem in problems:
                print(f"  - {problem}")
            sys.exit(1)
    else:
        for entry in fetch_audit_log(args.run_id):
            print(
                f"[{entry['sequence']:04d}] {entry['timestamp']} | {entry['run_id']} | "
                f"user={entry['triggered_by']}@{entry['hostname']} | "
                f"code={entry['code_version']} | preset={entry['preset']}"
            )


if __name__ == "__main__":
    main()
