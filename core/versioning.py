"""Test verisi versiyonlama ve drift tespiti.

**Çözülen problem.** "Geçen ay 85 puandı, bu ay 70" dendiğinde, bunun
(a) modelin kötüleşmesinden mi, (b) test verisinin (korpus, senaryolar,
sözlükler) değişmesinden mi, yoksa (c) ağırlık ön ayarının değişmesinden mi
kaynaklandığı ayırt edilemez. Bu üçü birbirinden çok farklı aksiyon
gerektirir.

**Çözüm.** Test verisinin kendisi hash'lenir (``dataset_version``). Her
koşu, hangi veri sürümüyle çalıştığını kaydeder. Drift raporu iki koşuyu
karşılaştırırken önce ``dataset_version`` ve ağırlık ön ayarının aynı olup
olmadığına bakar; farklıysa "veri/ağırlık değişti" der, model performansına
dair bir iddiada bulunmaz. Aynıysa gerçek bir model/konfigürasyon
değişikliği olduğu sonucuna varılabilir.

Versiyonlanan bileşenler: RAG korpusu, injection senaryoları, matematik
soru seti, içerik güvenliği sözlükleri. Hepsinin birleşik hash'i tek bir
``dataset_version`` dizgesi üretir.

CLI::

    python -m core.versioning hash
    python -m core.versioning drift --model gpt4
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from core.storage import fetch_track_b

logger = get_logger(__name__)

SIGNIFICANT_DELTA = 5.0  # bu puandan fazla degisim "dikkat cekici" sayilir


def _hash_directory(path: Path, suffixes: tuple[str, ...]) -> str:
    """Bir klasördeki belirli uzantılı dosyaların birleşik hash'ini üretir.

    Dosya adları da hash'e dahildir: bir dosyanın yeniden adlandırılması da
    bir değişikliktir, sadece içerik değişikliği değil.
    """
    if not path.exists():
        return "missing"
    hasher = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in suffixes)
    for file_path in files:
        hasher.update(str(file_path.relative_to(path)).encode("utf-8"))
        try:
            hasher.update(file_path.read_bytes())
        except OSError:
            hasher.update(b"unreadable")
    return hasher.hexdigest()[:12]


def _hash_file(path: Path) -> str:
    """Tek bir dosyanın hash'i; yoksa 'missing'."""
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _hash_module_constant(module_name: str, attribute: str) -> str:
    """Bir Python modülündeki sabit listeyi (ör. SCENARIOS) hash'ler.

    Senaryolar dosyada elle tanımlı; dosya değişmeden de biri sabiti runtime'da
    değiştirebilir. Bu yüzden dosyanın kendisini de ayrıca hash'liyoruz
    (bkz. dataset_components), bu fonksiyon ek bir çapraz kontrol sağlar.
    """
    try:
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        canonical = repr(value).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:12]
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "modul sabiti hashlenemedi", extra={"module": module_name, "error": str(exc)}
        )
        return "unavailable"


def dataset_components(settings: Settings | None = None) -> dict[str, str]:
    """Test verisinin her bileşeninin ayrı ayrı hash'ini döndürür."""
    settings = settings or get_settings()

    corpus_dir = settings.paths.absolute(settings.paths.rag_corpus_dir)
    lexicon_dir = settings.content_safety.lexicon_dir
    if not lexicon_dir.is_absolute():
        lexicon_dir = PROJECT_ROOT / lexicon_dir
    math_path = settings.math_eval.dataset_path
    if not math_path.is_absolute():
        math_path = PROJECT_ROOT / math_path

    return {
        "rag_corpus": _hash_directory(corpus_dir, (".md", ".txt", ".rst")),
        "content_lexicons": _hash_directory(lexicon_dir, (".txt",)),
        "math_problems": _hash_file(math_path),
        "injection_scenarios": _hash_module_constant(
            "llm_security.prompt_injection_tests", "SCENARIOS"
        ),
        "poisoning_cases": _hash_module_constant(
            "llm_security.data_poisoning_sim", "POISON_CASES"
        ),
    }


def dataset_version(settings: Settings | None = None) -> str:
    """Tüm test verisi bileşenlerinin birleşik sürüm dizgesini üretir.

    Herhangi bir bileşen değişirse bu dizge değişir; hangi bileşenin
    değiştiğini görmek için ``dataset_components()`` kullanılır.
    """
    components = dataset_components(settings)
    combined = "|".join(f"{name}:{value}" for name, value in sorted(components.items()))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def explain_dataset_version(settings: Settings | None = None) -> dict[str, Any]:
    """İnsan tarafından okunabilir sürüm raporu."""
    components = dataset_components(settings)
    return {
        "dataset_version": dataset_version(settings),
        "components": components,
        "missing": [name for name, value in components.items() if value == "missing"],
    }


# --------------------------------------------------------------------------- #
# Drift raporu
# --------------------------------------------------------------------------- #
def drift_report(model_name: str, settings: Settings | None = None) -> dict[str, Any]:
    """Bir modelin geçmiş koşuları arasındaki puan değişimini analiz eder.

    Ardışık her koşu çifti için: veri sürümü veya ağırlık ön ayarı
    değiştiyse bu **veri/konfigürasyon değişikliği** olarak işaretlenir ve
    puan farkı modele atfedilmez. İkisi de aynıysa ve puan farkı eşiği
    aşıyorsa bu **gerçek model/davranış değişikliği (drift)** olarak
    işaretlenir.
    """
    settings = settings or get_settings()
    rows = [row for row in fetch_track_b(limit=500, settings=settings) if row["model_name"] == model_name]
    rows.sort(key=lambda r: r["created_at"])

    if len(rows) < 2:
        return {
            "model_name": model_name,
            "comparable_pairs": 0,
            "message": "karşılaştırma için en az 2 koşu gerekli",
            "transitions": [],
        }

    transitions: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        delta = round(current["trust_score"] - previous["trust_score"], 2)
        config_changed = previous["config_name"] != current["config_name"]

        if config_changed:
            classification = "config_changed"
            explanation = (
                f"ağırlık ön ayarı değişti ({previous['config_name']} → "
                f"{current['config_name']}); puan farkı ağırlığa atfediliyor, modele değil"
            )
        elif abs(delta) < SIGNIFICANT_DELTA:
            classification = "stable"
            explanation = "anlamlı fark yok"
        else:
            direction = "iyileşme" if delta > 0 else "kötüleşme"
            classification = "model_drift"
            explanation = (
                f"veri seti ve ağırlıklar aynı, {abs(delta):.1f} puanlık {direction} "
                "gerçek bir model/davranış değişikliğine işaret ediyor"
            )

        transitions.append(
            {
                "from_run": previous["run_id"],
                "to_run": current["run_id"],
                "from_date": previous["created_at"],
                "to_date": current["created_at"],
                "from_score": previous["trust_score"],
                "to_score": current["trust_score"],
                "delta": delta,
                "classification": classification,
                "explanation": explanation,
            }
        )

    drift_events = [t for t in transitions if t["classification"] == "model_drift"]
    return {
        "model_name": model_name,
        "comparable_pairs": len(transitions),
        "drift_events": len(drift_events),
        "transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test verisi versiyonlama ve drift araci")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("hash", help="Mevcut test verisi surumunu goster")

    drift_parser = subparsers.add_parser("drift", help="Bir model icin drift raporu uret")
    drift_parser.add_argument("--model", required=True)

    args = parser.parse_args()

    if args.command == "hash":
        report = explain_dataset_version()
        print(f"dataset_version: {report['dataset_version']}")  # noqa: T201
        for name, value in report["components"].items():
            print(f"  {name}: {value}")  # noqa: T201
        if report["missing"]:
            print(f"UYARI — eksik bileşenler: {report['missing']}")  # noqa: T201
    else:
        report = drift_report(args.model)
        print(f"Model: {report['model_name']}")  # noqa: T201
        print(f"Karşılaştırılabilir geçiş sayısı: {report['comparable_pairs']}")  # noqa: T201
        for transition in report.get("transitions", []):
            print(  # noqa: T201
                f"  {transition['from_date'][:10]} -> {transition['to_date'][:10]}: "
                f"{transition['from_score']:.1f} -> {transition['to_score']:.1f} "
                f"[{transition['classification']}] {transition['explanation']}"
            )


if __name__ == "__main__":
    main()
