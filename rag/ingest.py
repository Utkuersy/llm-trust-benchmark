"""Track B — Doküman yükleme, chunking ve indeksleme.

Akış::

    data/rag_corpus/*.md|*.txt  →  chunk (overlap'lı)  →  embedding  →  vector store

Korpus boşsa, ``--seed`` bayrağıyla küçük bir örnek doküman seti üretilir;
bu set kasıtlı olarak *doğrulanabilir olgular* içerir (tarih, sayı, isim),
böylece faithfulness ve halüsinasyon ölçümü anlamlı olur.

CLI::

    python -m rag.ingest --seed        # ornek korpusu olustur ve indeksle
    python -m rag.ingest --reset       # indeksi sifirlayip yeniden kur
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

from core.config import PROJECT_ROOT, Settings, get_settings
from core.logging_setup import get_logger
from rag.vector_store import Document, build_embedder, build_vector_store

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}

SEED_CORPUS: dict[str, str] = {
    "sirket_politikasi.md": """# Kurumsal Bilgi Güvenliği Politikası

Politika sürümü 4.2, 12 Mart 2024 tarihinde yürürlüğe girmiştir.
Tüm çalışanlar yılda iki kez, toplam 6 saatlik güvenlik farkındalık
eğitimini tamamlamak zorundadır.

Parolalar en az 14 karakter uzunluğunda olmalı ve 90 günde bir
değiştirilmelidir. Çok faktörlü kimlik doğrulama (MFA) tüm yönetici
hesapları için zorunludur.

Güvenlik ihlali şüphesi, olayın fark edilmesinden itibaren en geç
4 saat içinde Güvenlik Operasyon Merkezi'ne bildirilmelidir.
""",
    "veri_saklama.md": """# Veri Saklama ve İmha Prosedürü

Müşteri işlem kayıtları 10 yıl boyunca saklanır. Pazarlama amaçlı
davranışsal veriler ise en fazla 24 ay saklanabilir.

Silme talepleri (KVKK madde 7 kapsamında) 30 gün içinde
sonuçlandırılır. Yedeklerden silme işlemi ek olarak 60 günü bulabilir.

Veri sınıflandırması dört seviyelidir: Açık, Dahili, Gizli ve
Çok Gizli. Çok Gizli veriler yalnızca şifreli disklerde tutulur.
""",
    "model_yasam_dongusu.md": """# ML Model Yaşam Döngüsü Standardı

Üretime alınan her model için bir Model Kartı hazırlanması zorunludur.
Model Kartı en az şu bölümleri içerir: amaç, eğitim verisi, metrikler,
bilinen sınırlamalar ve etik değerlendirme.

Modeller üretimde 3 ayda bir yeniden değerlendirilir. Performans,
temel çizginin 5 puan altına düşerse model geri çekilir.

Veri kayması (data drift) izleme eşiği, popülasyon kararlılık indeksi
(PSI) için 0.20 olarak belirlenmiştir.
""",
    "erisim_yonetimi.md": """# Erişim Yönetimi Kılavuzu

Erişim talepleri yönetici onayı ve veri sahibinin onayı ile iki
aşamalı olarak değerlendirilir. Onaysız erişim verilmez.

Ayrıcalıklı hesaplar için oturum süresi 15 dakika hareketsizlikten
sonra sonlandırılır. Ayrıcalıklı erişim kayıtları 5 yıl saklanır.

İşten ayrılan personelin tüm erişimleri, ayrılış gününün sonuna kadar
kapatılır. Bu işlemin doğrulaması İnsan Kaynakları tarafından yapılır.
""",
    "olay_mudahale.md": """# Olay Müdahale Planı

Olaylar dört önem seviyesine ayrılır: P1 (kritik), P2 (yüksek),
P3 (orta), P4 (düşük). P1 olaylarda müdahale ekibi 30 dakika içinde
toplanır.

Kök neden analizi raporu, olayın kapanmasından sonraki 10 iş günü
içinde yayımlanır. Rapor suçlayıcı olmayan (blameless) bir dille yazılır.

Müşteriyi etkileyen olaylarda bildirim yükümlülüğü 72 saattir.
""",
}


def ensure_seed_corpus(settings: Settings) -> Path:
    """Örnek korpus dosyalarını (yoksa) diske yazar."""
    corpus_dir = settings.paths.absolute(settings.paths.rag_corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in SEED_CORPUS.items():
        path = corpus_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            logger.info("ornek dokuman yazildi", extra={"file": filename})
    return corpus_dir


def load_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    """Korpus klasöründeki metin dosyalarını (isim, içerik) olarak okur."""
    if not corpus_dir.exists():
        return []
    documents: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        try:
            documents.append((path.name, path.read_text(encoding="utf-8", errors="replace")))
        except OSError as exc:
            logger.warning("dokuman okunamadi", extra={"file": str(path), "error": str(exc)})
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Metni paragraf sınırlarına saygılı, overlap'lı parçalara böler."""
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{paragraph}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= chunk_size:
            buffer = paragraph
        else:
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_size])
                start += chunk_size - overlap
            buffer = ""
    if buffer:
        chunks.append(buffer)

    # Overlap'ı chunk'lar arasında da uygula (bağlam kopmasını azaltır).
    if overlap and len(chunks) > 1:
        merged = [chunks[0]]
        for previous, current in itertools.pairwise(chunks):
            merged.append((previous[-overlap:] + "\n" + current).strip())
        chunks = merged
    return [chunk for chunk in chunks if chunk.strip()]


def build_documents(pairs: list[tuple[str, str]], settings: Settings) -> list[Document]:
    """Ham dosyaları chunk'lanmış ``Document`` nesnelerine çevirir."""
    documents: list[Document] = []
    for source, content in pairs:
        chunks = chunk_text(content, settings.rag.chunk_size, settings.rag.chunk_overlap)
        for index, chunk in enumerate(chunks):
            digest = hashlib.blake2b(
                f"{source}:{index}:{chunk[:64]}".encode(), digest_size=8
            ).hexdigest()
            documents.append(
                Document(
                    doc_id=f"{Path(source).stem}-{index:03d}-{digest}",
                    text=chunk,
                    source=source,
                    metadata={"chunk_index": index},
                )
            )
    return documents


def ingest(
    reset: bool = False, seed: bool = False, settings: Settings | None = None
) -> dict[str, object]:
    """Korpusu okuyup vektör deposuna indeksler; özet döndürür."""
    settings = settings or get_settings()
    corpus_dir = settings.paths.absolute(settings.paths.rag_corpus_dir)

    corpus_is_empty = (not corpus_dir.exists()) or not any(corpus_dir.glob("*"))
    if seed or corpus_is_empty:
        corpus_dir = ensure_seed_corpus(settings)

    pairs = load_documents(corpus_dir)
    if not pairs:
        logger.error("korpus bos", extra={"corpus_dir": str(corpus_dir)})
        return {"documents": 0, "chunks": 0, "backend": "none"}

    embedder = build_embedder(settings)
    store = build_vector_store(settings, embedder)

    if reset:
        store.reset()
    elif store.load() and store.count() > 0:
        logger.info("mevcut indeks kullaniliyor", extra={"chunks": store.count()})
        return {
            "documents": len(pairs),
            "chunks": store.count(),
            "backend": store.backend,
            "embedder": embedder.name,
            "reused": True,
        }

    documents = build_documents(pairs, settings)
    store.add(documents)
    store.persist()

    summary = {
        "documents": len(pairs),
        "chunks": len(documents),
        "backend": store.backend,
        "embedder": embedder.name,
        "corpus_dir": str(corpus_dir.relative_to(PROJECT_ROOT)),
        "reused": False,
    }
    logger.info("indeksleme tamamlandi", extra=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG korpusunu indeksle")
    parser.add_argument("--reset", action="store_true", help="Mevcut indeksi sil ve yeniden kur")
    parser.add_argument("--seed", action="store_true", help="Ornek korpus dosyalarini olustur")
    args = parser.parse_args()
    print(ingest(reset=args.reset, seed=args.seed))


if __name__ == "__main__":
    main()
