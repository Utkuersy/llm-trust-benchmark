"""Yapılandırılmış (JSON) loglama katmanı.

Projede ``print()`` kullanımı yasaktır (CLI script'lerinin son çıktı satırı
hariç). Tüm modüller::

    from core.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("analiz basladi", extra={"model": "gpt4", "track": "A"})

``extra`` içindeki alanlar JSON log satırına düz alan olarak eklenir; bu
sayede loglar Loki/ELK gibi sistemlerde doğrudan sorgulanabilir.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT, get_settings

_CONFIGURED = False

# LogRecord'un standart alanları; bunların dışındakiler "extra" sayılır.
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    """LogRecord'u tek satır JSON'a çevirir."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = _safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class HumanFormatter(logging.Formatter):
    """Geliştirici konsolu için okunabilir format."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )


def _safe(value: Any) -> Any:
    """JSON'a serialize edilemeyen değerleri stringe düşürür."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


def configure_logging(force: bool = False) -> None:
    """Kök logger'ı konfigürasyona göre kurar (idempotent)."""
    global _CONFIGURED  # noqa: PLW0603 - modul seviyesi tekil kurulum bayragi
    if _CONFIGURED and not force:
        return

    settings = get_settings().logging
    root = logging.getLogger()
    root.setLevel(settings.level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter: logging.Formatter = JsonFormatter() if settings.json_format else HumanFormatter()

    if settings.console:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if settings.log_file is not None:
        log_path = Path(settings.log_file)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # Gürültülü üçüncü parti logger'ları kıs.
    for noisy in ("urllib3", "matplotlib", "chromadb", "httpx", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Kurulumu garanti edilmiş bir logger döndürür."""
    configure_logging()
    return logging.getLogger(name)
