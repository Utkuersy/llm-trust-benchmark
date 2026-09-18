"""İnsan kalibrasyonu iskeleti.

**Dürüstlük notu — önce bunu oku.** Bu modül bir **araçtır**, bir
**sonuç değildir**. Gerçek bir insan değerlendirme çalışması bu ortamda
yapılmamıştır çünkü gerçek insan değerlendiriciler mevcut değildir.
``run_calibration_demo()`` fonksiyonu ürettiği "kalibrasyon raporu"
**yapay/sentetik veri üzerinde** çalışır ve bunu ``is_synthetic: True``
alanıyla açıkça işaretler. Bu raporu gerçek bir bulgu gibi sunmak —
"sistemimiz insan değerlendirmesiyle %87 örtüşüyor" demek — bu modülün
var oluş amacının tam tersidir ve tespit edildiğinde projenin tamamının
güvenilirliğini götürür.

**Asıl akış üç adımdır:**

1. ``sample_for_labeling()`` — mevcut model çıktılarından rastgele bir
   örneklem çıkarır, insan etiketleyicinin dolduracağı bir CSV/JSON şablonu
   üretir. Sistemin kendi puanı bu şablonda **gösterilmez** — insan
   etiketleyici sistemin ne dediğini bilerek etiketlerse çapa etkisi
   (anchoring bias) oluşur.
2. İnsan(lar) bu şablonu **kurum tarafından, gerçek değerlendiricilerle**
   doldurur. Bu adım bu kütüphanenin kapsamı dışındadır.
3. ``analyze_calibration()`` — doldurulmuş şablonu okur, sistem puanıyla
   karşılaştırır, Spearman korelasyonu ve (ikili etiketse) precision/recall
   hesaplar.

CLI::

    python -m core.calibration sample --outputs llm_outputs/gpt4 --n 20
    python -m core.calibration analyze --labels doldurulmus_sablon.json
    python -m core.calibration demo   # SADECE mekanizmayi gostermek icin
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT
from core.logging_setup import get_logger

logger = get_logger(__name__)

MIN_RECOMMENDED_SAMPLE = 100


# --------------------------------------------------------------------------- #
# Adım 1 — örnekleme ve şablon üretimi
# --------------------------------------------------------------------------- #
def sample_for_labeling(
    model_dir: Path,
    n: int = 30,
    dimension: str = "faithfulness",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Etiketlenecek örnekleri seçer ve boş etiket alanlı şablon üretir.

    Sistemin kendi puanı şablona **yazılmaz** — bu bilinçli bir tasarım
    kararıdır (bkz. modül dokümanı).
    """
    from rag.rag_evaluator import load_llm_outputs

    records = load_llm_outputs(model_dir)
    if not records:
        return []

    rng = random.Random(seed)
    selected = rng.sample(records, k=min(n, len(records)))

    template: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        template.append(
            {
                "item_id": f"CAL-{index:04d}",
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "contexts": record.get("contexts", []),
                "dimension": dimension,
                # İnsan etiketleyici doldurur — 0 (hayır) / 1 (evet) veya 1-5 ölçek.
                "human_label": None,
                "human_notes": "",
            }
        )

    if n < MIN_RECOMMENDED_SAMPLE:
        logger.warning(
            "ornek boyutu onerilen minimumun altinda",
            extra={"n": n, "onerilen_minimum": MIN_RECOMMENDED_SAMPLE},
        )
    return template


def write_labeling_template(template: list[dict[str, Any]], output_path: Path) -> None:
    """Şablonu JSON olarak yazar (insan etiketleyici bunu doldurur)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "etiketleme sablonu yazildi",
        extra={"path": str(output_path), "items": len(template)},
    )


# --------------------------------------------------------------------------- #
# Adım 3 — analiz
# --------------------------------------------------------------------------- #
def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Bağımlılıksız Spearman sıra korelasyonu (SciPy gerektirmez)."""
    n = len(x)
    if n < 2:
        return float("nan")

    def _ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = average_rank
            i = j + 1
        return ranks

    rank_x = _ranks(x)
    rank_y = _ranks(y)
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n

    numerator: float = sum(
        (rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y, strict=True)
    )
    denom_x: float = sum((rx - mean_x) ** 2 for rx in rank_x) ** 0.5
    denom_y: float = sum((ry - mean_y) ** 2 for ry in rank_y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return float("nan")
    return numerator / (denom_x * denom_y)


def _binary_confusion(system_flags: Sequence[bool], human_flags: Sequence[bool]) -> dict[str, int]:
    """İkili etiketler için karışıklık matrisi (precision/recall temeli)."""
    tp = sum(1 for s, h in zip(system_flags, human_flags, strict=True) if s and h)
    fp = sum(1 for s, h in zip(system_flags, human_flags, strict=True) if s and not h)
    fn = sum(1 for s, h in zip(system_flags, human_flags, strict=True) if not s and h)
    tn = sum(1 for s, h in zip(system_flags, human_flags, strict=True) if not s and not h)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn}


def analyze_calibration(
    labeled_items: list[dict[str, Any]],
    system_scores: dict[str, float],
    is_synthetic: bool = False,
) -> dict[str, Any]:
    """İnsan etiketleri ile sistem puanlarını karşılaştırır.

    ``labeled_items`` her biri ``item_id`` ve dolu ``human_label`` içeren
    kayıtlardır (bkz. ``sample_for_labeling``). ``system_scores``,
    ``item_id -> sistem puanı`` (0-1 veya 0-100) sözlüğüdür.
    """
    paired = [
        (system_scores[item["item_id"]], float(item["human_label"]))
        for item in labeled_items
        if item.get("human_label") is not None and item["item_id"] in system_scores
    ]

    unlabeled = sum(1 for item in labeled_items if item.get("human_label") is None)
    missing_system_score = sum(
        1 for item in labeled_items
        if item.get("human_label") is not None and item["item_id"] not in system_scores
    )

    if len(paired) < 2:
        return {
            "is_synthetic": is_synthetic,
            "n_labeled": len(paired),
            "n_unlabeled": unlabeled,
            "n_missing_system_score": missing_system_score,
            "status": "yetersiz_veri",
            "message": "korelasyon icin en az 2 eslesen etiket gerekli",
        }

    system_values = [pair[0] for pair in paired]
    human_values = [pair[1] for pair in paired]
    correlation = _spearman(system_values, human_values)

    result: dict[str, Any] = {
        "is_synthetic": is_synthetic,
        "n_labeled": len(paired),
        "n_unlabeled": unlabeled,
        "n_missing_system_score": missing_system_score,
        "spearman_correlation": round(correlation, 4) if correlation == correlation else None,
        "status": "tamamlandi",
    }

    unique_human = set(human_values)
    if unique_human.issubset({0.0, 1.0}):
        threshold = 0.5
        max_score = max(system_values) if system_values else 1.0
        normalized = [v / max_score if max_score > 1.5 else v for v in system_values]
        system_flags = [v >= threshold for v in normalized]
        human_flags = [v >= threshold for v in human_values]
        confusion = _binary_confusion(system_flags, human_flags)
        tp, fp, fn = confusion["true_positive"], confusion["false_positive"], confusion["false_negative"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        result["confusion_matrix"] = confusion
        result["precision"] = round(precision, 4) if precision == precision else None
        result["recall"] = round(recall, 4) if recall == recall else None

    if len(paired) < MIN_RECOMMENDED_SAMPLE:
        result["warning"] = (
            f"örneklem boyutu ({len(paired)}) önerilen minimumun "
            f"({MIN_RECOMMENDED_SAMPLE}) altında; sonuç ön bulgu sayılmalı, "
            "kesin kalibrasyon değil"
        )

    return result


def format_report(analysis: dict[str, Any]) -> str:
    """Analiz sonucunu okunabilir bir rapora çevirir."""
    lines = ["=== Kalibrasyon Raporu ==="]
    if analysis.get("is_synthetic"):
        lines.append(
            "*** BU BİR GÖSTERİMDİR — SENTETİK VERİ ÜZERİNDE ÇALIŞIYOR ***"
        )
        lines.append(
            "*** Gerçek kalibrasyon için gerçek insan etiketleyicilerle "
            "bir çalışma yürütülmelidir. ***"
        )
    lines.append(f"Eşleşen etiket sayısı: {analysis.get('n_labeled', 0)}")
    lines.append(f"Etiketlenmemiş: {analysis.get('n_unlabeled', 0)}")
    if analysis.get("status") == "yetersiz_veri":
        lines.append(f"Durum: {analysis['message']}")
        return "\n".join(lines)

    lines.append(f"Spearman korelasyonu: {analysis.get('spearman_correlation')}")
    if "precision" in analysis:
        lines.append(f"Precision: {analysis.get('precision')}")
        lines.append(f"Recall: {analysis.get('recall')}")
        lines.append(f"Karışıklık matrisi: {analysis.get('confusion_matrix')}")
    if analysis.get("warning"):
        lines.append(f"UYARI: {analysis['warning']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Gösterim (demo) — asla gerçek bulgu olarak sunulmamalı
# --------------------------------------------------------------------------- #
def run_calibration_demo(seed: int = 7) -> dict[str, Any]:
    """Mekanizmayı sentetik veriyle gösterir.

    Bu fonksiyonun ürettiği sayılar **gerçek insan değerlendirmesi
    değildir**. Yalnızca ``analyze_calibration`` fonksiyonunun doğru
    çalıştığını göstermek için rastgele üretilmiş verilerle çalışır.
    """
    rng = random.Random(seed)
    n = 40
    items = []
    scores = {}
    for i in range(n):
        item_id = f"DEMO-{i:03d}"
        true_quality = rng.random()
        system_score = max(0.0, min(1.0, true_quality + rng.gauss(0, 0.15)))
        human_label = 1.0 if true_quality > 0.5 else 0.0
        items.append({"item_id": item_id, "human_label": human_label})
        scores[item_id] = system_score

    analysis = analyze_calibration(items, scores, is_synthetic=True)
    analysis["_disclaimer"] = (
        "Bu rapor sentetik (yapay üretilmiş) veri kullanır. Gerçek bir "
        "kalibrasyon bulgusu değildir ve bu şekilde sunulmamalıdır."
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Insan kalibrasyonu iskeleti")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Etiketleme sablonu uret")
    sample_parser.add_argument("--outputs", required=True)
    sample_parser.add_argument("--n", type=int, default=30)
    sample_parser.add_argument("--dimension", default="faithfulness")
    sample_parser.add_argument("--out", default="calibration_template.json")

    analyze_parser = subparsers.add_parser("analyze", help="Doldurulmus sablonu analiz et")
    analyze_parser.add_argument("--labels", required=True)
    analyze_parser.add_argument("--scores", required=True, help="item_id -> puan JSON dosyasi")

    subparsers.add_parser("demo", help="SENTETIK veriyle mekanizmayi goster")

    args = parser.parse_args()

    if args.command == "sample":
        path = Path(args.outputs)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        template = sample_for_labeling(path, n=args.n, dimension=args.dimension)
        write_labeling_template(template, Path(args.out))
        print(f"{len(template)} öğelik şablon yazıldı: {args.out}")
        print("İnsan etiketleyici 'human_label' alanlarını doldurmalı.")

    elif args.command == "analyze":
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        analysis = analyze_calibration(labels, scores)
        print(format_report(analysis))

    else:
        analysis = run_calibration_demo()
        print(format_report(analysis))


if __name__ == "__main__":
    main()
