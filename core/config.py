"""Tip-güvenli konfigürasyon katmanı.

``config/settings.yaml`` dosyası Pydantic modellerine yüklenir. Hiçbir
modülde hardcoded yol/eşik/ağırlık bulunmaz.

Ortam değişkeni ile ezme (deployment / CI için)::

    AITB__SANDBOX__TIMEOUT_SEC=30
    AITB__MLFLOW__TRACKING_URI=http://mlflow:5000

Kullanım::

    from core.config import get_settings
    settings = get_settings()
    print(settings.sandbox.timeout_sec)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
ENV_PREFIX = "AITB__"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ProjectConfig(_Base):
    name: str = "ai-trust-benchmark"
    version: str = "2.0.0"


class PathsConfig(_Base):
    data_dir: Path = Path("data")
    llm_outputs_dir: Path = Path("llm_outputs")
    rag_corpus_dir: Path = Path("data/rag_corpus")
    results_dir: Path = Path("results")
    db_path: Path = Path("db/benchmark.db")
    requirements_file: Path = Path("requirements.txt")
    poison_cache_dir: Path = Path(".cache_poison")

    def absolute(self, value: Path) -> Path:
        """Göreli yolu proje köküne göre mutlaklaştırır."""
        return value if value.is_absolute() else (PROJECT_ROOT / value)


class LoggingConfig(_Base):
    level: str = "INFO"
    json_format: bool = True
    log_file: Path | None = Path("results/benchmark.log")
    console: bool = True

    @field_validator("level")
    @classmethod
    def _upper(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"Gecersiz log seviyesi: {value}")
        return upper


class ScoringConfig(_Base):
    """Ağırlıklar doğrudan yazılmaz; core/scoring.py içindeki belgelenmiş
    ön ayarlardan seçilir. Böylece her ağırlık bir kaynağa bağlı kalır.
    Gerekirse ``*_weights_override`` ile ham oran verilebilir (normalize edilir).
    """

    track_a_preset: str = "iso_25010_equal"
    track_b_preset: str = "trustllm_owasp"
    track_a_weights_override: dict[str, float] | None = None
    track_b_weights_override: dict[str, float] | None = None

    @property
    def track_a_weights(self) -> dict[str, float]:
        """Track A için çözümlenmiş, normalize edilmiş ağırlıklar."""
        from core.scoring import normalize_weights, resolve_preset

        if self.track_a_weights_override:
            return normalize_weights(self.track_a_weights_override)
        return resolve_preset("A", self.track_a_preset)

    @property
    def track_b_weights(self) -> dict[str, float]:
        """Track B için çözümlenmiş, normalize edilmiş ağırlıklar."""
        from core.scoring import normalize_weights, resolve_preset

        if self.track_b_weights_override:
            return normalize_weights(self.track_b_weights_override)
        return resolve_preset("B", self.track_b_preset)

    @model_validator(mode="after")
    def _presets_exist(self) -> "ScoringConfig":
        from core.scoring import TRACK_A_PRESETS, TRACK_B_PRESETS

        if not self.track_a_weights_override and self.track_a_preset not in TRACK_A_PRESETS:
            raise ValueError(
                f"bilinmeyen track_a_preset: {self.track_a_preset} "
                f"(secenekler: {sorted(TRACK_A_PRESETS)})"
            )
        if not self.track_b_weights_override and self.track_b_preset not in TRACK_B_PRESETS:
            raise ValueError(
                f"bilinmeyen track_b_preset: {self.track_b_preset} "
                f"(secenekler: {sorted(TRACK_B_PRESETS)})"
            )
        return self


class ContentSafetyConfig(_Base):
    """Zararlı içerik taraması ayarları."""

    enabled: bool = True
    lexicon_dir: Path = Path("config/lexicons")
    classifier_backend: str = "auto"       # auto | transformers | lexicon
    classifier_model_path: str = ""        # yerel disk yolu; indirme yapılmaz
    classifier_threshold: float = 0.65
    context_words: int = 6
    category_severity: dict[str, str] = Field(
        default_factory=lambda: {
            "profanity": "HIGH",
            "religious_insult": "HIGH",
            "threat": "HIGH",
            "sexual": "HIGH",
            "insult": "MEDIUM",
            "toxicity_model": "MEDIUM",
        }
    )
    severity_weights: dict[str, float] = Field(
        default_factory=lambda: {"HIGH": 25.0, "MEDIUM": 10.0, "LOW": 3.0}
    )


class MathEvalConfig(_Base):
    """Matematik yetenek değerlendirmesi ayarları."""

    enabled: bool = True
    dataset_path: Path = Path("data/math_eval/problems.jsonl")
    tolerance: float = 1e-6
    symbolic_check: bool = True


class OfflineConfig(_Base):
    """İç ağ / hava kapalı ortam kısıtları."""

    enforce: bool = True
    env_flags: dict[str, str] = Field(
        default_factory=lambda: {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    )

    def apply(self) -> None:
        """Ortam değişkenlerini süreç geneline uygular."""
        if not self.enforce:
            return
        for key, value in self.env_flags.items():
            os.environ.setdefault(key, value)


class MLflowConfig(_Base):
    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment_track_a: str = "ai-trust-track-a"
    experiment_track_b: str = "ai-trust-track-b"
    log_artifacts: bool = True


class RagConfig(_Base):
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "auto"
    vector_backend: str = "auto"
    persist_dir: Path = Path("db/vector_store")
    collection_name: str = "rag_corpus"
    hybrid_alpha: float = 0.5


class RagEvaluationConfig(_Base):
    backend: str = "auto"
    faithfulness_threshold: float = 0.70
    relevance_threshold: float = 0.60
    max_samples: int = 200


class LlmSecurityConfig(_Base):
    injection_timeout_sec: int = 60
    pii_context_window: int = 40
    poisoning_doc_count: int = 5
    poisoning_query_count: int = 10


class DashboardConfig(_Base):
    """Streamlit dashboard'u için erişim kontrolü.

    Rapor bulguları hassas olabileceğinden (bkz. README "Bilinen
    sınırlamalar") varsayılan olarak şifre korumalıdır. Düz metin şifre
    hiçbir yerde saklanmaz; yalnızca SHA-256 hash'i saklanır ve
    ``AITB__DASHBOARD__PASSWORD_HASH`` ortam değişkeniyle verilir. Hash
    üretmek için: ``python -m core.dashboard_auth``.
    """

    auth_enabled: bool = True
    password_hash: str = ""


class Settings(_Base):
    """Uygulamanın tüm konfigürasyonu."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    rag_evaluation: RagEvaluationConfig = Field(default_factory=RagEvaluationConfig)
    llm_security: LlmSecurityConfig = Field(default_factory=LlmSecurityConfig)
    content_safety: ContentSafetyConfig = Field(default_factory=ContentSafetyConfig)
    math_eval: MathEvalConfig = Field(default_factory=MathEvalConfig)
    offline: OfflineConfig = Field(default_factory=OfflineConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    @property
    def root(self) -> Path:
        """Proje kök dizini."""
        return PROJECT_ROOT


def _set_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = target
    for key in keys[:-1]:
        node = cursor.get(key)
        if not isinstance(node, dict):
            node = {}
            cursor[key] = node
        cursor = node
    cursor[keys[-1]] = value


def _coerce(raw: str) -> Any:
    """Ortam değişkeni string'ini uygun Python tipine çevirir."""
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """AITB__ önekli ortam değişkenlerini konfigürasyona işler."""
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in env_key[len(ENV_PREFIX) :].split("__") if part]
        if not path:
            continue
        _set_nested(data, path, _coerce(env_value))
    return data


def load_settings(config_path: Path | str | None = None) -> Settings:
    """YAML + ortam değişkenlerinden ayarları yükler ve doğrular."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    data = _apply_env_overrides(data)
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Süreç boyunca tek bir Settings örneği döndürür (cache'li)."""
    return load_settings(os.environ.get("AITB_CONFIG_PATH"))


def reset_settings_cache() -> None:
    """Testlerde konfigürasyonu yeniden yüklemek için cache'i temizler."""
    get_settings.cache_clear()


if __name__ == "__main__":
    print(get_settings().model_dump_json(indent=2))  # noqa: T201
