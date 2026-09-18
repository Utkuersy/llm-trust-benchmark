"""İçerik güvenliği tarayıcısı — işlevsel test paketi.

Yöntem, Röttger vd. (2021) HateCheck çalışmasından uyarlanmıştır: model
genel bir test kümesinde ortalama başarı ile değil, **her biri tek bir
davranışı sınayan işlevsel testlerle** ölçülür. HateCheck 29 işlevsellik
altında 3.728 vaka tanımlar ve kritik olarak her vakaya "ihlal / ihlal
değil" altın etiketi verir; ihlal olmayan karşıt vakalar (contrast cases)
olmadan yanlış pozitif oranı ölçülemez.

    Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., &
    Pierrehumbert, J. (2021). HateCheck: Functional Tests for Hate Speech
    Detection Models. ACL-IJCNLP 2021, 41-58.
    https://doi.org/10.18653/v1/2021.acl-long.4

Kaçırma (evasion) testlerinin dayanağı Hosseini vd. (2017): Perspective
API'nin, küfürlü kelimelerin yanlış yazılması veya harf araya noktalama
konulmasıyla atlatılabildiği gösterilmiştir ("idiot" → "idiiot" toksisite
skorunu %84'ten %20'ye düşürmüştür). Aynı çalışma olumsuzlama duyarlılığı
eksikliğini de raporlar.

    Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017).
    Deceiving Google's Perspective API Built for Detecting Toxic Comments.
    arXiv:1702.08138.

Bu dosyadaki testler proje sözlüğüne bağımlı olmamak için kendi izole
sözlüğünü kurar; kurumun gerçek sözlüğü değişse de testler geçerli kalır.
"""

from __future__ import annotations

import pytest

from llm_security.content_safety_scan import Lexicon, normalize, scan_text

# İzole test sözlüğü. Kasıtlı olarak hakaret düzeyinde ama kaba olmayan
# terimler seçildi; amaç mekanizmayı sınamak, terim listesi doğrulamak değil.
TEST_ROOTS = ("aptal", "salak", "gerizekal")
TEST_LEXICON = Lexicon("profanity_test", "HIGH", TEST_ROOTS)
LEXICONS = [TEST_LEXICON]


def categories(text: str) -> list[str]:
    """Metindeki bulguların eşleşen köklerini döndürür."""
    return [finding["matched_root"] for finding in scan_text(text, LEXICONS)]


def flagged(text: str) -> bool:
    """Metin en az bir ihlal üretiyor mu?"""
    return bool(scan_text(text, LEXICONS))


# --------------------------------------------------------------------------- #
# F1 — Düz ihlal tespiti (HateCheck: derogation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "Sen tam bir aptalsın.",
        "Bu ne salak bir öneri.",
        "Gerizekalı gibi davranıyorsun.",
    ],
)
def test_f1_plain_violation_detected(text: str) -> None:
    """Açık ihlaller yakalanmalı."""
    assert flagged(text), f"ihlal yakalanmadi: {text}"


# --------------------------------------------------------------------------- #
# F2 — Kaçırma teknikleri (Hosseini vd. 2017)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,technique",
    [
        ("Sen bir @pt@lsın.", "harf degistirme (leetspeak)"),
        ("Sen bir a.p.t.a.l.s.ı.n", "harf arasi noktalama"),
        ("Sen bir a p t a l sın", "harf arasi bosluk"),
        ("Sen bir aptaaaalsın", "harf tekrari"),
        ("SEN BİR APTALSIN", "buyuk harf"),
        ("Sen bir s@l@ksın", "karisik degistirme"),
    ],
)
def test_f2_evasion_techniques_detected(text: str, technique: str) -> None:
    """Kaçırma teknikleri normalize edilip yakalanmalı."""
    assert flagged(text), f"kacirma teknigi atlatti: {technique} -> {text}"


# --------------------------------------------------------------------------- #
# F3 — Türkçe biçimbilim (eklemeli dil)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "aptal",
        "aptallar",
        "aptalca",
        "aptallık",
        "salaklar",
        "salakça",
    ],
)
def test_f3_turkish_morphology_detected(text: str) -> None:
    """Kök + ek türevleri yakalanmalı (tam kelime eşleşmesi yetersizdir)."""
    assert flagged(text), f"ek almis tureve takildi: {text}"


def test_f3_excessive_suffix_not_matched() -> None:
    """Kök + 6 karakterden uzun ek eşleşmemeli — aşırı genelleme koruması."""
    assert not flagged("aptalxyzabcdefgh")


# --------------------------------------------------------------------------- #
# F4 — İhlal OLMAYAN karşıt vakalar (HateCheck contrast cases)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,reason",
    [
        ("Parolalar en az 14 karakter olmalıdır.", "notr kurumsal metin"),
        ("Müşteri işlem kayıtları 10 yıl saklanır.", "notr politika metni"),
        ("Bu yaklaşımın teknik gerekçesi zayıf görünüyor.", "nazik elestiri"),
        ("Kullanıcı davranış analizi yapılmalıdır.", "notr teknik oneri"),
    ],
)
def test_f4_non_violating_not_flagged(text: str, reason: str) -> None:
    """İhlal olmayan metinler işaretlenmemeli (yanlış pozitif kontrolü)."""
    assert not flagged(text), f"yanlis pozitif ({reason}): {text}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BİLİNEN SINIRLAMA: sözlük katmanı bağlam duyarlı değildir. Terimin "
        "akademik, alıntı veya karşı-söylem bağlamında geçmesi ihlal değildir "
        "ama sözlük bunu ayırt edemez. HateCheck'in 'non-hateful contrast case' "
        "kategorisine karşılık gelir. Bağlam ayrımı sınıflandırıcı katmanının "
        "işidir; sözlük tek başına karar mercii olarak kullanılmamalıdır."
    ),
)
@pytest.mark.parametrize(
    "text",
    [
        "Aptallık üzerine bir psikoloji makalesi okudum.",
        "Kullanıcı 'sen aptalsın' yazdığında sistem uyarı vermelidir.",
    ],
)
def test_f4_context_dependent_false_positive(text: str) -> None:
    """Bağlam gerektiren vakalar — şu an yanlış pozitif üretiliyor."""
    assert not flagged(text)


def test_f4_short_root_does_not_overmatch() -> None:
    """Kısa kökler yanlış pozitif üretir; minimum uzunluk eşiği çalışmalı."""
    short = Lexicon("test", "LOW", ("ap", "sa"))
    assert len(short) == 0, "4 karakterden kisa kokler sozluge alinmamali"


# --------------------------------------------------------------------------- #
# F5 — Olumsuzlama (Hosseini vd. 2017'nin raporladığı zayıflık)
# --------------------------------------------------------------------------- #
def test_f5_negation_is_known_limitation() -> None:
    """Olumsuzlanmış ifade de işaretlenir — bu bilinen bir sınırlamadır.

    "aptal değilsin" cümlesi hakaret değildir ama sözlük tabanlı katman
    bunu ayırt edemez. Test, davranışı *belgelemek* için vardır: sözlük
    katmanı bağlam duyarlı değildir ve tek başına karar mercii olamaz.
    Bağlam ayrımı sınıflandırıcı katmanının işidir.
    """
    assert flagged("Sen aptal değilsin, tam tersine çok yeteneklisin."), (
        "davranis degisti: olumsuzlama artik ayirt ediliyorsa bu testi guncelle"
    )


# --------------------------------------------------------------------------- #
# F6 — Normalizasyon birim testleri
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("APTAL", "aptal"),
        ("@pt@l", "aptal"),
        ("a.p.t.a.l", "aptal"),
        ("IĞDIR", "ığdır"),
        ("İSTANBUL", "istanbul"),
    ],
)
def test_f6_normalization(raw: str, expected: str) -> None:
    """Normalizasyon Türkçe büyük/küçük harf kurallarına uymalı."""
    assert normalize(raw) == expected


# --------------------------------------------------------------------------- #
# F7 — Sözlük yüklenmemişse sessizce geçmemeli
# --------------------------------------------------------------------------- #
def test_f7_empty_lexicon_produces_no_findings() -> None:
    """Boş sözlük bulgu üretmez — ama bu 'temiz' anlamına gelmez.

    Motor bu durumu ``inactive_categories`` altında raporlamalıdır;
    bu testin amacı boş sözlüğün çökme değil sessizlik ürettiğini
    doğrulamaktır.
    """
    empty = Lexicon("bos", "HIGH", ())
    assert scan_text("Sen bir aptalsın.", [empty]) == []
