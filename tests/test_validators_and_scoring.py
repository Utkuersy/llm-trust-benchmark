"""Validator and scoring tests — standards-based.

This file tests that the numeric identity validators and the answer
equivalence check behave in accordance with the relevant standards.

Basis:

* **The Luhn algorithm** — ISO/IEC 7812-1, the check digit for card
  identification numbers. Pattern matching alone produces false
  positives; not every 16-digit number is a card number.
* **IBAN mod-97 check** — ISO 13616-1. IBAN detection is unreliable
  without check-digit validation.
* **Turkish National ID Number (TCKN)** — the 10th/11th-digit checksum
  algorithm.
* **Answer equivalence** — Hendrycks et al. (2021), *Measuring
  Mathematical Problem Solving With the MATH Dataset* (NeurIPS Datasets
  & Benchmarks), shows that raw string comparison is insufficient for
  math evaluation and that answers must be normalized before checking
  equivalence; ``1/2`` and ``0.5`` are the same answer.
* **Weight normalization** — Barron, F. H. & Barrett, B. E. (1996),
  *Decision Quality Using Ranked Attribute Weights*, Management Science
  42(11), methods for deriving weights from ranked attributes.
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
    """Standard test card numbers must satisfy the check digit."""
    assert validate_luhn(number)


@pytest.mark.parametrize(
    "number",
    ["4111111111111112", "1234567890123456", "0000000000000001", "123", "9" * 25],
)
def test_luhn_rejects_invalid(number: str) -> None:
    """Numbers with a failing check digit or invalid length must be rejected."""
    assert not validate_luhn(number)


# --------------------------------------------------------------------------- #
# IBAN — ISO 13616-1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "iban",
    ["GB82WEST12345698765432", "DE89370400440532013000", "FR1420041010050500013M02606"],
)
def test_iban_accepts_valid(iban: str) -> None:
    """IBANs that pass the mod-97 check must be accepted."""
    assert validate_iban(iban)


@pytest.mark.parametrize(
    "iban",
    ["GB82WEST12345698765433", "XX00INVALID", "1234567890123456", ""],
)
def test_iban_rejects_invalid(iban: str) -> None:
    """Strings with a failing check digit must be rejected."""
    assert not validate_iban(iban)


# --------------------------------------------------------------------------- #
# TCKN — the check-digit algorithm
# --------------------------------------------------------------------------- #
def _make_valid_tckn(prefix: str = "123456789") -> str:
    digits = [int(ch) for ch in prefix]
    tenth = ((sum(digits[0:9:2]) * 7) - sum(digits[1:8:2])) % 10
    eleventh = (sum(digits) + tenth) % 10
    return prefix + str(tenth) + str(eleventh)


def test_tckn_accepts_algorithmically_valid() -> None:
    """A number satisfying the algorithm must be accepted."""
    assert validate_tckn(_make_valid_tckn())


@pytest.mark.parametrize(
    "value",
    [
        "00000000000",   # cannot start with zero
        "12345678901",   # check digit fails
        "1234567890",    # 10 digits
        "123456789012",  # 12 digits
        "abcdefghijk",
    ],
)
def test_tckn_rejects_invalid(value: str) -> None:
    """Invalid numbers must be rejected."""
    assert not validate_tckn(value)


def test_random_eleven_digits_rarely_pass() -> None:
    """Most random 11-digit numbers are not a valid TCKN.

    This test is the validator's reason for existing: with plain pattern
    matching alone, every 11-digit number would be a false positive.
    """
    candidates = [str(10_000_000_000 + i) for i in range(0, 1000, 7)]
    passing = sum(validate_tckn(value) for value in candidates)
    assert passing < len(candidates) * 0.2


def test_pii_scan_ignores_invalid_identifiers() -> None:
    """Invalid identity numbers must not be reported as findings."""
    findings = scan_text("Order number was recorded as 12345678901.")
    assert not [f for f in findings if f["type"] == "tckn"]


# --------------------------------------------------------------------------- #
# Math answer equivalence — Hendrycks et al. (2021)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "predicted,expected,reason",
    [
        ("1/2", "0.5", "fraction - decimal"),
        ("0,5", "0.5", "comma decimal separator"),
        ("1,200", "1200", "thousands separator"),
        ("$1200", "1200", "currency symbol"),
        ("12 apples", "12", "unit suffix"),
        ("x^2 - 9", "x**2 - 9", "exponent notation"),
        ("  480  ", "480", "whitespace"),
        ("\\frac{1}{4}", "0.25", "latex fraction"),
    ],
)
def test_equivalent_answers_match(predicted: str, expected: str, reason: str) -> None:
    """Answers written differently but equal in value must be treated as equivalent."""
    matched, _ = answers_match(predicted, expected, 1e-6, True)
    assert matched, f"not treated as equivalent ({reason}): {predicted} vs {expected}"


@pytest.mark.parametrize(
    "predicted,expected",
    [("7", "8"), ("0.5", "0.6"), ("x + 1", "x + 2"), ("100", "1000")],
)
def test_different_answers_do_not_match(predicted: str, expected: str) -> None:
    """Different answers must not be treated as equivalent — false positives inflate accuracy."""
    matched, _ = answers_match(predicted, expected, 1e-6, True)
    assert not matched


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Let's compute: 8 x 60 = 480. \\boxed{480}", "480"),
        ("Let's solve step by step... Answer: 2/5", "2/5"),
        ("Result: 5050", "5050"),
        ("Answer = 42", "42"),
    ],
)
def test_answer_extraction(response: str, expected: str) -> None:
    """The answer must be correctly extracted from free text."""
    assert extract_answer(response) == expected


def test_extraction_failure_is_distinguishable() -> None:
    """Failure to extract an answer and a wrong answer are different situations.

    An extraction failure is a format mismatch, not a model error, and
    must be reported separately; returning ``None`` makes this
    distinction possible.
    """
    assert extract_answer("I cannot answer this question.") is None


# --------------------------------------------------------------------------- #
# Weights — Barron & Barrett (1996)
# --------------------------------------------------------------------------- #
def test_normalize_weights_sums_to_one() -> None:
    """Raw ratios must sum to 1.0 once normalized."""
    weights = normalize_weights({"a": 3, "b": 2, "c": 1})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] > weights["b"] > weights["c"]


def test_normalize_weights_preserves_ratio() -> None:
    """Normalization must preserve ratios."""
    weights = normalize_weights({"a": 3, "b": 1})
    assert weights["a"] / weights["b"] == pytest.approx(3.0)


@pytest.mark.parametrize("preset", ["output_safety_first", "owasp_rank", "trustllm_equal"])
def test_presets_are_normalized(preset: str) -> None:
    """Every preset must normalize to 1.0; otherwise scores are not comparable."""
    weights = resolve_preset("B", preset)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(value >= 0 for value in weights.values())


def test_output_safety_preset_prioritizes_content_safety() -> None:
    """In the enterprise preset, content safety must be the heaviest dimension.

    This tests that the priority the stakeholder specified is actually
    reflected in the scoring; if the weights are accidentally changed,
    this test goes red.
    """
    weights = resolve_preset("B", "output_safety_first")
    assert weights["content_safety"] == max(weights.values())
    assert weights["content_safety"] > weights["injection"]
    assert weights["injection"] > weights["pii"] > weights["poisoning"]


def test_unknown_preset_raises() -> None:
    """An undefined preset must not silently fall back to a default."""
    with pytest.raises((KeyError, ValueError)):
        resolve_preset("B", "no_such_preset_exists")
