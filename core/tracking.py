"""MLflow deney takibi katmanı.

Her benchmark koşusu bir MLflow "run" olarak loglanır:
    * **params**  : model adı, hedef yol, track, konfigürasyon adı, araç sürümleri
    * **metrics** : trust score + tüm alt metrikler
    * **artifacts**: ham sonuç JSON'u

MLflow kurulu değilse veya tracking sunucusuna erişilemiyorsa katman
sessizce devre dışı kalır (``NullTracker``); benchmark akışı bozulmaz.
Bu, CI ve air-gapped ortamlar için bilinçli bir tasarım kararıdır.

Kullanım::

    with tracker.start_run("gpt4", Track.A) as run_id:
        tracker.log_params({...})
        tracker.log_metrics({...})
        tracker.log_json_artifact(result.model_dump(), "track_a_result.json")
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.schemas import Track

logger = get_logger(__name__)


def _flatten_metrics(prefix: str, payload: dict[str, Any]) -> dict[str, float]:
    """İç içe sözlükten sayısal metrikleri düz anahtarlarla çıkarır."""
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, bool):
            flat[name] = float(value)
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
        elif isinstance(value, dict):
            flat.update(_flatten_metrics(name, value))
    return flat


class ExperimentTracker:
    """MLflow sarmalayıcısı; MLflow yoksa tüm çağrılar no-op olur."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._mlflow: Any = None
        self._active = False
        if not self.settings.mlflow.enabled:
            logger.info("mlflow konfigurasyonla devre disi birakildi")
            return
        try:
            import mlflow  # noqa: PLC0415 - opsiyonel bagimlilik
        except ImportError:
            logger.warning("mlflow yuklu degil, deney takibi atlanacak")
            return
        try:
            uri = self.settings.mlflow.tracking_uri
            if uri.startswith("file:") and not uri.startswith("file:/"):
                uri = "file:" + str(PROJECT_ROOT / uri.removeprefix("file:").lstrip("./"))
            mlflow.set_tracking_uri(uri)
            self._mlflow = mlflow
            logger.info("mlflow hazir", extra={"tracking_uri": uri})
        except Exception as exc:  # noqa: BLE001 - takip katmani asla akisi kirmaz
            logger.warning("mlflow baslatilamadi", extra={"error": str(exc)})

    @property
    def enabled(self) -> bool:
        """MLflow gerçekten kullanılabilir durumda mı?"""
        return self._mlflow is not None

    def _experiment_for(self, track: Track) -> str:
        return (
            self.settings.mlflow.experiment_track_a
            if track is Track.A
            else self.settings.mlflow.experiment_track_b
        )

    @contextmanager
    def start_run(self, run_name: str, track: Track) -> Iterator[str | None]:
        """MLflow run'ı başlatır; devre dışıysa ``None`` verir."""
        if not self.enabled:
            self._active = False
            yield None
            return
        try:
            self._mlflow.set_experiment(self._experiment_for(track))
            with self._mlflow.start_run(run_name=run_name) as run:
                self._active = True
                try:
                    yield run.info.run_id
                finally:
                    self._active = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("mlflow run baslatilamadi", extra={"error": str(exc)})
            self._active = False
            yield None

    def log_params(self, params: dict[str, Any]) -> None:
        """Parametreleri loglar (hatalar yutulur)."""
        if not (self.enabled and self._active):
            return
        try:
            self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        except Exception as exc:  # noqa: BLE001
            logger.debug("mlflow param loglanamadi", extra={"error": str(exc)})

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        """Sayısal metrikleri düzleştirip loglar."""
        if not (self.enabled and self._active):
            return
        flat = _flatten_metrics(prefix, metrics)
        try:
            self._mlflow.log_metrics(flat)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mlflow metrik loglanamadi", extra={"error": str(exc)})

    def log_json_artifact(self, payload: dict[str, Any], filename: str) -> None:
        """Sözlüğü JSON artifact olarak yükler."""
        if not (self.enabled and self._active and self.settings.mlflow.log_artifacts):
            return
        try:
            with tempfile.TemporaryDirectory(prefix="aitb_artifact_") as tmp:
                path = Path(tmp) / filename
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                self._mlflow.log_artifact(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.debug("mlflow artifact loglanamadi", extra={"error": str(exc)})

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Run etiketlerini ayarlar."""
        if not (self.enabled and self._active):
            return
        try:
            self._mlflow.set_tags({k: str(v)[:250] for k, v in tags.items()})
        except Exception as exc:  # noqa: BLE001
            logger.debug("mlflow tag loglanamadi", extra={"error": str(exc)})


def get_tracker(settings: Settings | None = None) -> ExperimentTracker:
    """Yeni bir tracker örneği döndürür."""
    return ExperimentTracker(settings)
