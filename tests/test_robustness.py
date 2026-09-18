"""Robustness and adversarial-input tests.

This platform, by definition, processes **untrusted text**: the output
of the model under evaluation can be controlled by an attacker.
Consequently, the scanners themselves are an attack surface.

Weakness classes tested and their rationale:

* **ReDoS (CWE-1333, Inefficient Regular Expression Complexity)** —
  backtracking regular expressions can run in exponential time on
  specially crafted input. A denial-of-service vector.
  OWASP: Regular expression Denial of Service — ReDoS.
* **Resource consumption (CWE-400, Uncontrolled Resource Consumption)**
  — memory and CPU can be exhausted if input size is not bounded. OWASP
  ASVS v4.0.3 V5.1 "Input Validation Requirements" mandates input size
  validation.
* **Log injection (CWE-117, Improper Output Neutralization for Logs)**
  — control characters in model output can corrupt log lines or inject
  fake entries.
* **Path traversal (CWE-22)** — a user-supplied path argument must not
  escape the project root.

The time thresholds in these tests are set generously relative to
machine speed; the goal is not micro-benchmarking, but demonstrating
**the absence of exponential blowup**.
"""

from __future__ import annotations

import time

import pytest

from capability.math_eval import answers_match, extract_answer, normalize_expression
from llm_security.content_safety_scan import Lexicon, normalize, scan_text
from llm_security.pii_leakage_scan import redact
from llm_security.pii_leakage_scan import scan_text as pii_scan

LEXICON = [Lexicon("test", "HIGH", ("idiot", "moron"))]

# Classic ReDoS patterns meant to trigger exponential backtracking.
REDOS_INPUTS = [
    "a" * 50_000,
    ("a." * 20_000),
    "1" * 30_000,
    ("@" * 10_000) + "idiot",
    ("0" * 25_000) + "x",
    (" " * 40_000) + "test",
]

# pytest generates a test id from the parameter value itself when none is
# given. Since these inputs are tens of thousands of characters long, the
# id exceeds Windows's 32767-character environment-variable limit when
# pytest writes it to PYTEST_CURRENT_TEST (ValueError). Short, explicit
# ids bypass this limit.
REDOS_IDS = [
    "a_50k",
    "a_dot_20k",
    "digit_30k",
    "at_10k_idiot",
    "zero_25k_x",
    "space_40k_test",
]

# Time threshold: even on slow CI machines, a non-exponential
# implementation stays well under this limit.
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
    """The content scanner must not enter exponential time on pathological input."""
    duration = _elapsed(scan_text, payload, LEXICON)
    assert duration < TIME_LIMIT_SEC, f"possible ReDoS: {duration:.2f}s"


@pytest.mark.parametrize("payload", REDOS_INPUTS, ids=REDOS_IDS)
def test_pii_scanner_no_redos(payload: str) -> None:
    """The PII scanner must not enter exponential time on pathological input.

    The card-number pattern ``(?:\\d[ -]*?){13,19}`` contains a lazy
    quantifier; it is tested specifically because it carries a
    backtracking risk on long digit sequences.
    """
    duration = _elapsed(pii_scan, payload)
    assert duration < TIME_LIMIT_SEC, f"possible ReDoS: {duration:.2f}s"


@pytest.mark.parametrize("payload", REDOS_INPUTS[:3], ids=REDOS_IDS[:3])
def test_normalizer_no_redos(payload: str) -> None:
    """The normalization layer (separator joining) must not hang on pathological input."""
    duration = _elapsed(normalize, payload)
    assert duration < TIME_LIMIT_SEC, f"possible ReDoS: {duration:.2f}s"


@pytest.mark.parametrize("payload", REDOS_INPUTS[:3], ids=REDOS_IDS[:3])
def test_answer_extraction_no_redos(payload: str) -> None:
    """The math answer-extraction layer must not hang on pathological input."""
    duration = _elapsed(extract_answer, payload)
    assert duration < TIME_LIMIT_SEC, f"possible ReDoS: {duration:.2f}s"


# --------------------------------------------------------------------------- #
# Resource consumption — CWE-400 / OWASP ASVS V5.1
# --------------------------------------------------------------------------- #
def test_content_scanner_truncates_oversized_input() -> None:
    """Oversized input must be truncated, not processed unbounded in memory."""
    huge = "clean text. " * 200_000  # ~2.6 MB
    duration = _elapsed(scan_text, huge, LEXICON)
    assert duration < TIME_LIMIT_SEC


def test_pii_scanner_truncates_oversized_input() -> None:
    """The PII scanner must also bound its input size."""
    huge = "a.b@c.com " * 200_000
    findings = pii_scan(huge)
    assert isinstance(findings, list)


# --------------------------------------------------------------------------- #
# Malformed / adversarial types — must not crash
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload",
    [
        "",
        "   ",
        "\x00\x01\x02",
        "\n" * 1000,
        "🙂" * 5000,
        "‮" * 100,          # right-to-left override character
        "<script>alert(1)</script>",
        "'; DROP TABLE runs; --",
        "../../etc/passwd",
        "%00%0a%0d",
    ],
    # Explicit ids instead of the auto-generated one: for long/unicode
    # payloads pytest's id can run to tens of thousands of characters
    # (see the REDOS_IDS comment) and exceed the Windows
    # PYTEST_CURRENT_TEST environment-variable limit.
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
    """Adversarial input must not raise an exception; defensive programming requirement."""
    assert isinstance(scan_text(payload, LEXICON), list)
    assert isinstance(pii_scan(payload), list)
    assert extract_answer(payload) is None or isinstance(extract_answer(payload), str)


@pytest.mark.parametrize("payload", [None, 123, [], {}, 3.14])
def test_scanners_reject_non_string_safely(payload: object) -> None:
    """Non-string input must silently return an empty result, not crash."""
    assert scan_text(payload, LEXICON) == []  # type: ignore[arg-type]
    assert pii_scan(payload) == []  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Log injection — CWE-117
# --------------------------------------------------------------------------- #
def test_findings_do_not_leak_control_characters() -> None:
    """A finding's context must not contain line breaks; a log line must not be splittable.

    If control characters are written to logs as-is, an attacker could
    inject a fake log entry (CWE-117).
    """
    payload = "normal text\nidiot\r\nFAKE LOG: user authorized"
    for finding in scan_text(payload, LEXICON):
        assert "\n" not in finding["context"]
        assert "\r" not in finding["context"]


def test_pii_context_has_no_newlines() -> None:
    """The PII context field must also contain no line breaks."""
    payload = "contact:\njohn.doe@example.com\r\nFAKE"
    for finding in pii_scan(payload):
        assert "\n" not in finding["context"]
        assert "\r" not in finding["context"]


# --------------------------------------------------------------------------- #
# Data leakage — findings must not carry the raw sensitive data
# --------------------------------------------------------------------------- #
def test_pii_findings_are_masked() -> None:
    """A report must not carry detected sensitive data in the clear.

    Evaluation reports are shareable; writing raw PII into the report
    would multiply the leak instead of measuring it.
    """
    email = "john.doe@corp-internal.example"
    findings = pii_scan(f"reach out at {email}")
    assert findings, "email was not detected"
    for finding in findings:
        assert email not in finding["masked_value"]
        assert "*" in finding["masked_value"]


def test_redaction_removes_sensitive_values() -> None:
    """The redaction function must remove sensitive values from text."""
    text = "key sk-live-9f2b7c1d4e6a8f0b3c5d7e9f mail a@b.com"
    cleaned = redact(text)
    assert "sk-live-9f2b7c1d4e6a8f0b3c5d7e9f" not in cleaned
    assert "a@b.com" not in cleaned
    assert "REDACTED" in cleaned


# --------------------------------------------------------------------------- #
# Math evaluation — adversarial answer text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "response",
    [
        "\\boxed{" + "9" * 10_000 + "}",
        "Answer: " + ("(" * 5000),
        "x" * 100_000,
        "1/0",
        "\\boxed{}",
    ],
    ids=["boxed_digits_10k", "open_paren_5000", "x_100k", "div_by_zero", "empty_boxed"],
)
def test_math_extraction_survives_hostile_response(response: str) -> None:
    """Malformed or oversized answer text must not produce a crash."""
    extracted = extract_answer(response)
    assert extracted is None or isinstance(extracted, str)


def test_symbolic_comparison_does_not_execute_code() -> None:
    """Symbolic comparison must not execute arbitrary code.

    SymPy's ``parse_expr`` function is risky on untrusted input; an
    unparseable expression must come back as "not equivalent", never
    raise an exception or produce a side effect.
    """
    malicious = "__import__('os').system('echo pwned')"
    matched, _ = answers_match(malicious, "42", 1e-6, True)
    assert matched is False


def test_normalize_expression_is_pure() -> None:
    """Normalization must be side-effect-free and deterministic."""
    value = "1,200 USD"
    assert normalize_expression(value) == normalize_expression(value)
    assert value == "1,200 USD", "input must not be mutated"
