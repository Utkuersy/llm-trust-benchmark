"""Content safety scanner — functional test suite.

The method is adapted from Röttger et al. (2021)'s HateCheck study: a
model is measured not by average accuracy on a generic test set, but by
**functional tests that each probe a single behavior**. HateCheck defines
3,728 cases across 29 functional test types and, critically, gives every
case a "violation / not a violation" gold label; without non-violating
contrast cases, the false-positive rate cannot be measured.

    Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., &
    Pierrehumbert, J. (2021). HateCheck: Functional Tests for Hate Speech
    Detection Models. ACL-IJCNLP 2021, 41-58.
    https://doi.org/10.18653/v1/2021.acl-long.4

The basis for the evasion tests is Hosseini et al. (2017): they showed
that Perspective API can be evaded by misspelling profane words or by
inserting punctuation between letters ("idiot" -> "idiiot" dropped the
toxicity score from 84% to 20%). The same study also reports a lack of
negation sensitivity.

    Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017).
    Deceiving Google's Perspective API Built for Detecting Toxic Comments.
    arXiv:1702.08138.

The tests in this file build their own isolated lexicon so they do not
depend on the project's real lexicon; they remain valid even if the
organization's actual lexicon changes.
"""

from __future__ import annotations

import pytest

from llm_security.content_safety_scan import Lexicon, normalize, scan_text

# Isolated test lexicon. Deliberately chosen to be insult-level but not
# crude language; the goal is to test the mechanism, not validate a term
# list.
TEST_ROOTS = ("idiot", "moron", "stupid")
TEST_LEXICON = Lexicon("profanity_test", "HIGH", TEST_ROOTS)
LEXICONS = [TEST_LEXICON]


def categories(text: str) -> list[str]:
    """Returns the matched roots for the findings in a text."""
    return [finding["matched_root"] for finding in scan_text(text, LEXICONS)]


def flagged(text: str) -> bool:
    """Does the text produce at least one violation?"""
    return bool(scan_text(text, LEXICONS))


# --------------------------------------------------------------------------- #
# F1 — Plain violation detection (HateCheck: derogation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "You are a complete idiot.",
        "What a moronic suggestion.",
        "You're acting stupid.",
    ],
)
def test_f1_plain_violation_detected(text: str) -> None:
    """Plain violations must be caught."""
    assert flagged(text), f"violation not caught: {text}"


# --------------------------------------------------------------------------- #
# F2 — Evasion techniques (Hosseini et al. 2017)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,technique",
    [
        ("You are an 1d10t.", "character substitution (leetspeak)"),
        ("You are an i.d.i.o.t", "inter-letter punctuation"),
        ("You are an i d i o t", "inter-letter spacing"),
        ("You are an idiooot", "letter repetition"),
        ("YOU ARE AN IDIOT", "uppercase"),
        ("You are a m0r0n", "mixed substitution"),
    ],
)
def test_f2_evasion_techniques_detected(text: str, technique: str) -> None:
    """Evasion techniques must be normalized and caught."""
    assert flagged(text), f"evasion technique bypassed detection: {technique} -> {text}"


# --------------------------------------------------------------------------- #
# F3 — Suffix morphology
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "idiot",
        "idiots",
        "idiotic",
        "idiotically",
        "morons",
        "stupidly",
    ],
)
def test_f3_morphology_detected(text: str) -> None:
    """Root + suffix derivatives must be caught (exact word matching is insufficient)."""
    assert flagged(text), f"suffixed derivative slipped through: {text}"


def test_f3_excessive_suffix_not_matched() -> None:
    """Root + suffix longer than 6 characters should not match — over-generalization guard."""
    assert not flagged("idiotxyzabcdefgh")


# --------------------------------------------------------------------------- #
# F4 — Non-violating contrast cases (HateCheck contrast cases)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,reason",
    [
        ("Passwords must be at least 14 characters long.", "neutral corporate text"),
        ("Customer transaction records are retained for 10 years.", "neutral policy text"),
        ("The technical rationale for this approach seems weak.", "polite criticism"),
        ("A user behavior analysis should be conducted.", "neutral technical suggestion"),
    ],
)
def test_f4_non_violating_not_flagged(text: str, reason: str) -> None:
    """Non-violating text must not be flagged (false-positive check)."""
    assert not flagged(text), f"false positive ({reason}): {text}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN LIMITATION: the lexicon layer is not context-aware. A term "
        "appearing in an academic, quoted, or counter-speech context is not "
        "a violation, but the lexicon cannot tell the difference. This "
        "corresponds to HateCheck's 'non-hateful contrast case' category. "
        "Context disambiguation is the classifier layer's job; the lexicon "
        "must not be used as the sole decision authority."
    ),
)
@pytest.mark.parametrize(
    "text",
    [
        "I read a psychology article about stupidity.",
        "The system should warn when a user types 'you are an idiot'.",
    ],
)
def test_f4_context_dependent_false_positive(text: str) -> None:
    """Cases that require context — currently produce a false positive."""
    assert not flagged(text)


def test_f4_short_root_does_not_overmatch() -> None:
    """Short roots produce false positives; the minimum length threshold must hold."""
    short = Lexicon("test", "LOW", ("id", "mo"))
    assert len(short) == 0, "roots shorter than 4 characters must not enter the lexicon"


# --------------------------------------------------------------------------- #
# F5 — Negation (the weakness reported by Hosseini et al. 2017)
# --------------------------------------------------------------------------- #
def test_f5_negation_is_known_limitation() -> None:
    """Negated statements are also flagged — this is a known limitation.

    "you are not an idiot" is not an insult, but the lexicon-based layer
    cannot tell the difference. This test exists to *document* the
    behavior: the lexicon layer is not context-aware and must not be the
    sole decision authority. Context disambiguation is the classifier
    layer's job.
    """
    assert flagged("You are not an idiot, quite the opposite, you're very talented."), (
        "behavior changed: update this test if negation is now disambiguated"
    )


# --------------------------------------------------------------------------- #
# F6 — Normalization unit tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("IDIOT", "idiot"),
        ("1d10t", "idiot"),
        ("i.d.i.o.t", "idiot"),
        ("MORON", "moron"),
        ("StUpId", "stupid"),
    ],
)
def test_f6_normalization(raw: str, expected: str) -> None:
    """Normalization must lowercase and resolve evasion characters correctly."""
    assert normalize(raw) == expected


# --------------------------------------------------------------------------- #
# F7 — An unloaded lexicon must not silently pass as "clean"
# --------------------------------------------------------------------------- #
def test_f7_empty_lexicon_produces_no_findings() -> None:
    """An empty lexicon produces no findings — but that doesn't mean "clean".

    The engine must report this case under ``inactive_categories``; the
    purpose of this test is to confirm that an empty lexicon produces
    silence, not a crash.
    """
    empty = Lexicon("empty", "HIGH", ())
    assert scan_text("You are an idiot.", [empty]) == []
