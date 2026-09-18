"""Doğrulayıcı ve puanlama testleri — standart tabanlı.

Bu dosya, sayısal kimlik doğrulayıcılarının ve cevap denklik kontrolünün
ilgili standartlara uygun davrandığını sınar.

Dayanaklar:

* **Luhn algoritması** — ISO/IEC 7812-1, kart tanımlayıcı numaraların
  kontrol hanesi. Yalnızca desen eşleştirmek yanlış pozitif üretir;
  16 haneli her sayı kart numarası değildir.
* **IBAN mod-97 kontrolü** — ISO 13616-1. Kontrol hanesi doğrulaması
  olmadan IBAN tespiti güvenilmez.
* **T.C. Kimlik Numarası** — 10. ve 11. hane kontrol algoritması.
* **Cevap denkliği** — Hendrycks vd. (2021), *Measuring Mathematical
  Problem Solving With the MATH Dataset* (NeurIPS Datasets & Benchmarks),
  matematik değerlendirmesinde ham string karşılaştırmanın yetersiz
  olduğunu ve cevapların normalize edilerek denklik kontrolü yapılması
  gerektiğini ortaya koyar; ``1/2`` ile ``0.5`` aynı cevaptır.
* **Ağırlık normalizasyonu** — Barron, F. H. & Barrett, B. E. (1996),
  *Decision Quality Using Ranked Attribute Weights*, Management Science
  42(11), sıralı özniteliklerden ağırlık türetme yöntemleri.
"""

from __future__ import annotations

import pytest

from capability.math_eval import answers_match, extract_answer
from core.scoring import normalize_weights, resolve_preset
from llm_security.pii_leakage_scan import (
    scan_text,
    validate_iban,
    validate_luhn,
    validate_tckn,
)

# --------------------------------------------------------------------------- #
# Luhn — ISO/IEC 7812-1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "number",
    ["4111111111111111", "5500005555555559", "4012888888881881", "378282246310005"],
)
def test_luhn_accepts_valid(number: str) -> None:
    """Standart test kart numaraları kontrol hanesini sağlamalı."""
    assert validate_luhn(number)


@pytest.mark.parametrize(
    "number",
    ["4111111111111112", "1234567890123456", "0000000000000001", "123", "9" * 25],
)
def test_luhn_rejects_invalid(number: str) -> None:
    """Kontrol hanesi tutmayan veya uzunluk dışı sayılar reddedilmeli."""
    assert not validate_luhn(number)


# --------------------------------------------------------------------------- #
# IBAN — ISO 13616-1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "iban",
    ["GB82WEST12345698765432", "DE89370400440532013000", "FR1420041010050500013M02606"],
)
def test_iban_accepts_valid(iban: str) -> None:
    """mod-97 kontrolünü geçen IBAN'lar kabul edilmeli."""
    assert validate_iban(iban)


@pytest.mark.parametrize(
    "iban",
    ["GB82WEST12345698765433", "XX00INVALID", "1234567890123456", ""],
)
def test_iban_rejects_invalid(iban: str) -> None:
    """Kontrol hanesi tutmayan dizeler reddedilmeli."""
    assert not validate_iban(iban)


# --------------------------------------------------------------------------- #
# TCKN — kontrol hanesi algoritması
# --------------------------------------------------------------------------- #
def _make_valid_tckn(prefix: str = "123456789") -> str:
    digits = [int(ch) for ch in prefix]
    tenth = ((sum(digits[0:9:2]) * 7) - sum(digits[1:8:2])) % 10
    eleventh = (sum(digits) + tenth) % 10
    return prefix + str(tenth) + str(eleventh)


def test_tckn_accepts_algorithmically_valid() -> None:
    """Algoritmayı sağlayan numara kabul edilmeli."""
    assert validate_tckn(_make_valid_tckn())


@pytest.mark.parametrize(
    "value",
    [
        "00000000000",   # sifirla baslayamaz
        "12345678901",   # kontrol hanesi tutmuyor
        "1234567890",    # 10 hane
        "123456789012",  # 12 hane
        "abcdefghijk",
    ],
)
def test_tckn_rejects_invalid(value: str) -> None:
    """Geçersiz numaralar reddedilmeli."""
    assert not validate_tckn(value)


def test_random_eleven_digits_rarely_pass() -> None:
    """Rastgele 11 haneli sayıların çoğu TCKN değildir.

    Bu test doğrulayıcının var olma gerekçesidir: salt desen eşleştirme
    kullanılsaydı her 11 haneli sayı yanlış pozitif üretirdi.
    """
    candidates = [str(10_000_000_000 + i) for i in range(0, 1000, 7)]
    passing = sum(validate_tckn(value) for value in candidates)
    assert passing < len(candidates) * 0.2


def test_pii_scan_ignores_invalid_identifiers() -> None:
    """Geçersiz kimlik numaraları bulgu olarak raporlanmamalı."""
    findings = scan_text("Sipariş numarası 12345678901 olarak kaydedildi.")
    assert not [f for f in findings if f["type"] == "tckn"]


# --------------------------------------------------------------------------- #
# Matematik cevap denkliği — Hendrycks vd. (2021)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "predicted,expected,reason",
    [
        ("1/2", "0.5", "kesir - ondalik"),
        ("0,5", "0.5", "turkce ondalik ayraci"),
        ("1,200", "1200", "binlik ayraci"),
        ("$1200", "1200", "para birimi simgesi"),
        ("12 elma", "12", "birim eki"),
        ("x^2 - 9", "x**2 - 9", "us gosterimi"),
        ("  480  ", "480", "bosluk"),
        ("\\frac{1}{4}", "0.25", "latex kesir"),
    ],
)
def test_equivalent_answers_match(predicted: str, expected: str, reason: str) -> None:
    """Farklı biçimde yazılmış aynı cevaplar denk sayılmalı."""
    matched, _ = answers_match(predicted, expected, 1e-6, True)
    assert matched, f"denk sayilmadi ({reason}): {predicted} vs {expected}"


@pytest.mark.parametrize(
    "predicted,expected",
    [("7", "8"), ("0.5", "0.6"), ("x + 1", "x + 2"), ("100", "1000")],
)
def test_different_answers_do_not_match(predicted: str, expected: str) -> None:
    """Farklı cevaplar denk sayılmamalı — yanlış pozitif doğruluğu şişirir."""
    matched, _ = answers_match(predicted, expected, 1e-6, True)
    assert not matched


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Hesaplama: 8 x 60 = 480. \\boxed{480}", "480"),
        ("Adım adım çözelim... Cevap: 2/5", "2/5"),
        ("Sonuç: 5050", "5050"),
        ("Answer = 42", "42"),
    ],
)
def test_answer_extraction(response: str, expected: str) -> None:
    """Cevap serbest metinden doğru çıkarılmalı."""
    assert extract_answer(response) == expected


def test_extraction_failure_is_distinguishable() -> None:
    """Cevap çıkarılamaması ile yanlış cevap farklı durumlardır.

    Çıkarılamama bir model hatası değil format uyumsuzluğudur ve ayrı
    raporlanmalıdır; ``None`` dönüşü bu ayrımı mümkün kılar.
    """
    assert extract_answer("Bu soruyu cevaplayamıyorum.") is None


# --------------------------------------------------------------------------- #
# Ağırlıklar — Barron & Barrett (1996)
# --------------------------------------------------------------------------- #
def test_normalize_weights_sums_to_one() -> None:
    """Ham oranlar normalize edilince toplam 1.0 olmalı."""
    weights = normalize_weights({"a": 3, "b": 2, "c": 1})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] > weights["b"] > weights["c"]


def test_normalize_weights_preserves_ratio() -> None:
    """Normalizasyon oranları korumalı."""
    weights = normalize_weights({"a": 3, "b": 1})
    assert weights["a"] / weights["b"] == pytest.approx(3.0)


@pytest.mark.parametrize("preset", ["output_safety_first", "owasp_rank", "trustllm_equal"])
def test_presets_are_normalized(preset: str) -> None:
    """Her ön ayar 1.0'a normalize olmalı; aksi halde puanlar kıyaslanamaz."""
    weights = resolve_preset("B", preset)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(value >= 0 for value in weights.values())


def test_output_safety_preset_prioritizes_content_safety() -> None:
    """Kurumsal ön ayarda içerik güvenliği en ağır boyut olmalı.

    Bu, görevlinin belirttiği önceliğin puanlamaya yansıdığının testidir;
    ağırlıklar yanlışlıkla değiştirilirse bu test kırmızıya döner.
    """
    weights = resolve_preset("B", "output_safety_first")
    assert weights["content_safety"] == max(weights.values())
    assert weights["content_safety"] > weights["injection"]
    assert weights["injection"] > weights["pii"] > weights["poisoning"]


def test_unknown_preset_raises() -> None:
    """Tanımsız ön ayar sessizce varsayılana düşmemeli."""
    with pytest.raises((KeyError, ValueError)):
        resolve_preset("B", "boyle_bir_preset_yok")
