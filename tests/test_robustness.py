"""Dayanıklılık ve düşmanca girdi testleri.

Bu platform, tanım gereği **güvenilmeyen metin** işler: değerlendirilen
modelin çıktısı saldırgan tarafından kontrol edilebilir. Dolayısıyla
tarayıcıların kendisi bir saldırı yüzeyidir.

Test edilen zayıflık sınıfları ve dayanakları:

* **ReDoS (CWE-1333, Inefficient Regular Expression Complexity)** — geri
  izlemeli (backtracking) düzenli ifadeler, özel olarak hazırlanmış girdide
  üstel sürede çalışabilir. Servis dışı bırakma vektörüdür.
  OWASP: Regular expression Denial of Service — ReDoS.
* **Kaynak tüketimi (CWE-400, Uncontrolled Resource Consumption)** — girdi
  boyutu sınırlanmazsa bellek ve CPU tüketilebilir. OWASP ASVS v4.0.3
  V5.1 "Input Validation Requirements" girdi boyutu doğrulamasını şart koşar.
* **Log injection (CWE-117, Improper Output Neutralization for Logs)** —
  model çıktısındaki kontrol karakterleri log satırlarını bozabilir veya
  sahte kayıt enjekte edebilir.
* **Path traversal (CWE-22)** — kullanıcıdan gelen yol argümanı proje
  kökü dışına çıkmamalıdır.

Testlerin süre eşikleri makine hızına görece geniş tutulmuştur; amaç
mikro-benchmark değil, **üstel patlamanın olmadığını** göstermektir.
"""

from __future__ import annotations

import time

import pytest

from capability.math_eval import answers_match, extract_answer, normalize_expression
from llm_security.content_safety_scan import Lexicon, normalize, scan_text
from llm_security.pii_leakage_scan import redact
from llm_security.pii_leakage_scan import scan_text as pii_scan

LEXICON = [Lexicon("test", "HIGH", ("aptal", "salak"))]

# Üstel geri izlemeyi tetiklemek için klasik ReDoS kalıpları.
REDOS_INPUTS = [
    "a" * 50_000,
    ("a." * 20_000),
    "1" * 30_000,
    ("@" * 10_000) + "aptal",
    ("0" * 25_000) + "x",
    (" " * 40_000) + "test",
]

# pytest, id verilmezse parametre değerinin kendisinden bir test id'si
# üretir. Bu girdiler onbinlerce karakter uzunluğunda olduğundan, id
# pytest'in PYTEST_CURRENT_TEST ortam değişkenine yazılırken Windows'un
# 32767 karakterlik ortam değişkeni sınırını aşar (ValueError). Kısa,
# açık id'ler bu sınırı bypass eder.
REDOS_IDS = [
    "a_50k",
    "a_dot_20k",
    "digit_30k",
    "at_10k_aptal",
    "zero_25k_x",
    "space_40k_test",
]

# Süre eşiği: yavaş CI makinelerinde bile üstel olmayan bir uygulama
# bu sınırın çok altında kalır.
TIME_LIMIT_SEC = 5.0


def _elapsed(function, *args) -> float:
    start = time.perf_counter()
    function(*args)
    return time.perf_counter() - start


# --------------------------------------------------------------------------- #
# ReDoS — CWE-1333
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload", REDOS_INPUTS, ids=REDOS_IDS)
def test_content_scanner_no_redos(payload: str) -> None:
    """İçerik tarayıcısı patolojik girdide üstel süreye girmemeli."""
    duration = _elapsed(scan_text, payload, LEXICON)
    assert duration < TIME_LIMIT_SEC, f"olasi ReDoS: {duration:.2f}s"


@pytest.mark.parametrize("payload", REDOS_INPUTS, ids=REDOS_IDS)
def test_pii_scanner_no_redos(payload: str) -> None:
    """PII tarayıcısı patolojik girdide üstel süreye girmemeli.

    Kart numarası deseni ``(?:\\d[ -]*?){13,19}`` tembel niceleyici içerir;
    uzun rakam dizilerinde geri izleme riski taşıdığı için özellikle
    sınanır.
    """
    duration = _elapsed(pii_scan, payload)
    assert duration < TIME_LIMIT_SEC, f"olasi ReDoS: {duration:.2f}s"


@pytest.mark.parametrize("payload", REDOS_INPUTS[:3], ids=REDOS_IDS[:3])
def test_normalizer_no_redos(payload: str) -> None:
    """Normalizasyon katmanı (ayraç birleştirme) patolojik girdide takılmamalı."""
    duration = _elapsed(normalize, payload)
    assert duration < TIME_LIMIT_SEC, f"olasi ReDoS: {duration:.2f}s"


@pytest.mark.parametrize("payload", REDOS_INPUTS[:3], ids=REDOS_IDS[:3])
def test_answer_extraction_no_redos(payload: str) -> None:
    """Matematik cevap çıkarma katmanı patolojik girdide takılmamalı."""
    duration = _elapsed(extract_answer, payload)
    assert duration < TIME_LIMIT_SEC, f"olasi ReDoS: {duration:.2f}s"


# --------------------------------------------------------------------------- #
# Kaynak tüketimi — CWE-400 / OWASP ASVS V5.1
# --------------------------------------------------------------------------- #
def test_content_scanner_truncates_oversized_input() -> None:
    """Aşırı büyük girdi kırpılmalı, bellekte sınırsız işlenmemeli."""
    huge = "temiz metin. " * 200_000  # ~2.6 MB
    duration = _elapsed(scan_text, huge, LEXICON)
    assert duration < TIME_LIMIT_SEC


def test_pii_scanner_truncates_oversized_input() -> None:
    """PII tarayıcısı da girdi boyutunu sınırlamalı."""
    huge = "a.b@c.com " * 200_000
    findings = pii_scan(huge)
    assert isinstance(findings, list)


# --------------------------------------------------------------------------- #
# Bozuk / düşmanca tipler — çökme olmamalı
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload",
    [
        "",
        "   ",
        "\x00\x01\x02",
        "\n" * 1000,
        "🙂" * 5000,
        "\u202e" * 100,          # sag-sol yon degistirme karakteri
        "<script>alert(1)</script>",
        "'; DROP TABLE runs; --",
        "../../etc/passwd",
        "%00%0a%0d",
    ],
    # Otomatik üretilen id yerine açık id: uzun/unicode payload'larda pytest'in
    # id'si onbinlerce karaktere çıkabilir (bkz. REDOS_IDS yorumu) ve Windows'ta
    # PYTEST_CURRENT_TEST ortam değişkeni sınırını aşar.
    ids=[
        "empty",
        "whitespace",
        "control_chars",
        "newline_1000",
        "emoji_5000",
        "rtl_override_100",
        "script_tag",
        "sql_injection",
        "path_traversal",
        "url_encoded_control",
    ],
)
def test_scanners_survive_hostile_input(payload: str) -> None:
    """Düşmanca girdi istisna fırlatmamalı; savunmacı programlama gereği."""
    assert isinstance(scan_text(payload, LEXICON), list)
    assert isinstance(pii_scan(payload), list)
    assert extract_answer(payload) is None or isinstance(extract_answer(payload), str)


@pytest.mark.parametrize("payload", [None, 123, [], {}, 3.14])
def test_scanners_reject_non_string_safely(payload: object) -> None:
    """String olmayan girdi sessizce boş sonuç vermeli, çökmemeli."""
    assert scan_text(payload, LEXICON) == []  # type: ignore[arg-type]
    assert pii_scan(payload) == []  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Log injection — CWE-117
# --------------------------------------------------------------------------- #
def test_findings_do_not_leak_control_characters() -> None:
    """Bulgu bağlamı satır sonu içermemeli; log satırı bölünemez olmalı.

    Kontrol karakterleri loglara olduğu gibi yazılırsa saldırgan sahte
    log kaydı enjekte edebilir (CWE-117).
    """
    payload = "normal metin\naptal\r\nSAHTE LOG: kullanici yetkilendirildi"
    for finding in scan_text(payload, LEXICON):
        assert "\n" not in finding["context"]
        assert "\r" not in finding["context"]


def test_pii_context_has_no_newlines() -> None:
    """PII bağlam alanı da satır sonu içermemeli."""
    payload = "iletisim:\nmehmet@example.com\r\nSAHTE"
    for finding in pii_scan(payload):
        assert "\n" not in finding["context"]
        assert "\r" not in finding["context"]


# --------------------------------------------------------------------------- #
# Veri sızıntısı — bulgular ham hassas veriyi taşımamalı
# --------------------------------------------------------------------------- #
def test_pii_findings_are_masked() -> None:
    """Rapor, tespit edilen hassas veriyi açık biçimde taşımamalı.

    Değerlendirme raporları paylaşılabilir; ham PII'yi rapora yazmak
    sızıntıyı ölçmek yerine çoğaltmak olurdu.
    """
    email = "mehmet.yilmaz@corp-internal.example"
    findings = pii_scan(f"iletisim {email} adresinden ulasin")
    assert findings, "e-posta tespit edilmedi"
    for finding in findings:
        assert email not in finding["masked_value"]
        assert "*" in finding["masked_value"]


def test_redaction_removes_sensitive_values() -> None:
    """Redaksiyon fonksiyonu hassas değerleri metinden çıkarmalı."""
    text = "anahtar sk-live-9f2b7c1d4e6a8f0b3c5d7e9f mail a@b.com"
    cleaned = redact(text)
    assert "sk-live-9f2b7c1d4e6a8f0b3c5d7e9f" not in cleaned
    assert "a@b.com" not in cleaned
    assert "REDACTED" in cleaned


# --------------------------------------------------------------------------- #
# Matematik değerlendirmesi — düşmanca cevap metni
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "response",
    [
        "\\boxed{" + "9" * 10_000 + "}",
        "Cevap: " + ("(" * 5000),
        "x" * 100_000,
        "1/0",
        "\\boxed{}",
    ],
    ids=["boxed_digits_10k", "open_paren_5000", "x_100k", "div_by_zero", "empty_boxed"],
)
def test_math_extraction_survives_hostile_response(response: str) -> None:
    """Bozuk veya devasa cevap metni çökme üretmemeli."""
    extracted = extract_answer(response)
    assert extracted is None or isinstance(extracted, str)


def test_symbolic_comparison_does_not_execute_code() -> None:
    """Sembolik karşılaştırma keyfi kod çalıştırmamalı.

    SymPy'nin ``parse_expr`` fonksiyonu güvenilmeyen girdide risklidir;
    ayrıştırılamayan ifade "denk değil" olarak dönmeli, istisna
    yükseltmemeli ve yan etki üretmemelidir.
    """
    malicious = "__import__('os').system('echo pwned')"
    matched, _ = answers_match(malicious, "42", 1e-6, True)
    assert matched is False


def test_normalize_expression_is_pure() -> None:
    """Normalizasyon yan etkisiz ve deterministik olmalı."""
    value = "1,200 TL"
    assert normalize_expression(value) == normalize_expression(value)
    assert value == "1,200 TL", "girdi degistirilmemeli"
