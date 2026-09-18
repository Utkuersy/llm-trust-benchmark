"""MLflow experiment tracking layer.

Every benchmark run is logged as an MLflow "run":
    * **params**  : model name, target path, track, config name, tool versions
    * **metrics** : trust score + all sub-metrics
    * **artifacts**: the raw result JSON

If MLflow is not installed or the tracking server is unreachable, this
layer silently disables itself (``NullTracker``); the benchmark flow is
not disrupted. This is a deliberate design decision for CI and
air-gapped environments.

Usage::

    with tracker.start_run("gpt4", Track.A) as run_id:
        tracker.log_params({...})
        tracker.log_metrics({...})
        tracker.log_json_artifact(result.model_dump(), "track_a_result.json")
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.schemas import Track

logger = get_logger(__name__)


def _flatten_metrics(prefix: str, payload: dict[str, Any]) -> dict[str, float]:
    """Extracts numeric metrics from a nested dict into flat keys."""
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, (bool, int, float)):
            flat[name] = float(value)
        elif isinstance(value, dict):
            flat.update(_flatten_metrics(name, value))
    return flat


class ExperimentTracker:
    """An MLflow wrapper; every call is a no-op if MLflow is unavailable."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._mlflow: Any = None
        self._active = False
        if not self.settings.mlflow.enabled:
            logger.info("mlflow disabled via configuration")
            return
        try:
            import mlflow
        except ImportError:
            logger.warning("mlflow not installed, experiment tracking will be skipped")
            return
        try:
            uri = self.settings.mlflow.tracking_uri
            if uri.startswith("file:") and not uri.startswith("file:/"):
                uri = "file:" + str(PROJECT_ROOT / uri.removeprefix("file:").lstrip("./"))
            mlflow.set_tracking_uri(uri)
            self._mlflow = mlflow
            logger.info("mlflow ready", extra={"tracking_uri": uri})
        except Exception as exc:
            logger.warning("mlflow could not be initialized", extra={"error": str(exc)})

    @property
    def enabled(self) -> bool:
        """Is MLflow actually usable?"""
        return self._mlflow is not None

    def _experiment_for(self, track: Track) -> str:
        return (
            self.settings.mlflow.experiment_track_a
            if track is Track.A
            else self.settings.mlflow.experiment_track_b
        )

    @contextmanager
    def start_run(self, run_name: str, track: Track) -> Iterator[str | None]:
        """Starts an MLflow run; yields ``None`` if disabled."""
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
        except Exception as exc:
            logger.warning("mlflow run could not be started", extra={"error": str(exc)})
            self._active = False
            yield None

    def log_params(self, params: dict[str, Any]) -> None:
        """Logs parameters (errors are swallowed)."""
        if not (self.enabled and self._active):
            return
        try:
            self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        except Exception as exc:
            logger.debug("mlflow param could not be logged", extra={"error": str(exc)})

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        """Flattens and logs numeric metrics."""
        if not (self.enabled and self._active):
            return
        flat = _flatten_metrics(prefix, metrics)
        try:
            self._mlflow.log_metrics(flat)
        except Exception as exc:
            logger.debug("mlflow metric could not be logged", extra={"error": str(exc)})

    def log_json_artifact(self, payload: dict[str, Any], filename: str) -> None:
        """Uploads a dict as a JSON artifact."""
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
        except Exception as exc:
            logger.debug("mlflow artifact could not be logged", extra={"error": str(exc)})

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Sets run tags."""
        if not (self.enabled and self._active):
            return
        try:
            self._mlflow.set_tags({k: str(v)[:250] for k, v in tags.items()})
        except Exception as exc:
            logger.debug("mlflow tag could not be logged", extra={"error": str(exc)})


def get_tracker(settings: Settings | None = None) -> ExperimentTracker:
    """Returns a new tracker instance."""
    return ExperimentTracker(settings)
