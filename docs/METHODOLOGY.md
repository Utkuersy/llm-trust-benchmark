# Methodology and sources

This document lists the basis for every methodological choice in the
platform. The goal is to be able to give a traceable answer to questions
like "why did you pick 0.65 for this threshold."

The sources below justify the **method**. None of them *prescribes* a
numeric weight for this platform; the numbers are derived by applying
the stated methods to this project's dimensions. This is not a
calibration — real calibration requires a human-expert-labeled reference
set.

---

## 1. Weight derivation

| Decision | Basis |
|---|---|
| Equal weight among the top-level dimensions | Dawes, R. M. (1979). *The robust beauty of improper linear models in decision making.* American Psychologist, 34(7), 571-582. Shows that, absent calibration data, equal (unit) weights produce results as robust as estimated weights. |
| Mapping the dimension set onto ISO characteristics | ISO/IEC 25010:2023, *Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model.* The standard defines 9 quality characteristics (functional suitability, performance efficiency, compatibility, interaction capability, reliability, security, maintainability, flexibility, safety). Of these 9, the 3 directly relevant to this project (Safety, Security, Functional suitability) were selected; the standard defines no priority or subset among them — the selection and the equal weighting between them is this project's decision. |
| Moving from ranked risk to a numeric weight (rank-sum) | Barron, F. H. & Barrett, B. E. (1996). *Decision Quality Using Ranked Attribute Weights.* Management Science, 42(11), 1515-1523. |
| Ranking of the security sub-dimensions | OWASP Foundation (2024). *OWASP Top 10 for Large Language Model Applications 2025* (v2.0, November 18, 2024). LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure, LLM04 Data and Model Poisoning. |
| Decomposition of LLM trustworthiness dimensions | Sun, L. et al. (2024). *TrustLLM: Trustworthiness in Large Language Models.* ICML 2024. |

**Warning.** ISO 25010 and OWASP assign no numeric weight. Saying "I
measured the weights" is incorrect; the correct phrasing is "I derived
the weights using a documented method."

---

## 2. Content safety tests

| Decision | Basis |
|---|---|
| The functional test suite approach (one behavior = one test) | Röttger, P. et al. (2021). *HateCheck: Functional Tests for Hate Speech Detection Models.* ACL-IJCNLP 2021, 41-58. DOI: 10.18653/v1/2021.acl-long.4. 29 functionalities, 3,728 validated cases. |
| The requirement for non-violating contrast cases | Same source. Testing only with violation examples makes the false-positive rate invisible. |
| Normalizing evasion techniques (misspelling, inter-letter punctuation, letter repetition) | Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017). *Deceiving Google's Perspective API Built for Detecting Toxic Comments.* arXiv:1702.08138. Shows that changing "idiot" → "idiiot" dropped the toxicity score from 84% to 20%. |
| Documenting negation-sensitivity as a known limitation | Same source; reports that negated profane statements do not receive a lower score. |
| The lexicon term-selection rationale (short/collision-prone roots excluded) | LDNOOBW (*List of Dirty, Naughty, Obscene, and Otherwise Bad Words*), a 75-language open-source community list. github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words. **Not an organization-approved final list**, see the note in the file headers. |
| The profanity (untargeted crude language) / insult (targeted individual put-down) / threat (violence threat) distinction | Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social Media.* LREC 2020 — the OffensEval hierarchical labeling scheme (offensive/not → targeted/untargeted → target type); the categorization logic in this project's lexicon layer follows the same taxonomy regardless of language. |

**Model selection note.** Perspective API cannot be used in this
project: it sends requests to an external network, which conflicts with
the internal-network constraint. The classifier layer is loaded from
local disk (`content_safety.classifier_model_path`).

---

## 3. Robustness and security tests

| Decision | Basis |
|---|---|
| A time-limit test on regular expressions | CWE-1333: *Inefficient Regular Expression Complexity.* OWASP: *Regular expression Denial of Service (ReDoS).* |
| Input size limiting | CWE-400: *Uncontrolled Resource Consumption.* OWASP ASVS v4.0.3, V5.1 Input Validation Requirements. |
| Stripping control characters from log fields | CWE-117: *Improper Output Neutralization for Logs.* |
| Path argument validation | CWE-22: *Improper Limitation of a Pathname to a Restricted Directory.* |
| `shell=False` in subprocesses | CWE-78: *OS Command Injection.* |
| Masking sensitive data in reports | An evaluation report is a shareable artifact; writing raw PII into a report multiplies the leak instead of measuring it. |

---

## 4. Identity and account number validation

| Decision | Basis |
|---|---|
| Card number check digit (Luhn) | ISO/IEC 7812-1, *Identification cards — Identification of issuers.* |
| IBAN mod-97 check | ISO 13616-1, *Financial services — International Bank Account Number (IBAN).* |
| Turkish National ID Number (TCKN) check digit | The 10th/11th-digit algorithm. |

Rationale: pattern matching alone produces false positives. Not every
11-digit number is a national ID, not every 16-digit number is a card
number.

---

## 5. Math evaluation

| Decision | Basis |
|---|---|
| Extracting the answer from free text and comparing it normalized | Hendrycks, D. et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS Datasets and Benchmarks Track. |
| Problem categories (Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, Precalculus) | Same source (Hendrycks et al., 2021) — the MATH dataset's official 7-topic taxonomy. `data/math_eval/problems.jsonl` contains at least one problem from each of these 7 categories; the content is written specifically for this project, not copied from the dataset. |
| Symbolic equivalence checking | Lewkowycz, A. et al. (2022). *Solving Quantitative Reasoning Problems with Language Models* (Minerva). NeurIPS 2022. Establishes that answer equivalence cannot be measured by string equality. |
| Not counting an unextractable answer as wrong | A format mismatch and a reasoning error call for different actions; they are reported separately (`extraction_failures`). |

---

## 6. Governance frameworks

| Framework | Its counterpart in this project |
|---|---|
| NIST AI Risk Management Framework 1.0 (2023) | The Measure function — the system implements the "measure" step of the risk-measurement cycle. |
| ISO/IEC 42001:2023 (AI management system) | Evaluation records and version traceability (`SCORING_VERSION`, run records). |
| OWASP Top 10 for LLM Applications 2025 | The taxonomy and ranking of the security dimensions. |

---

## 7. Calibration status — must be stated explicitly

Currently **not calibrated.** The system ranks (is model A more
trustworthy than B), it does not set an absolute threshold (is a score
of 70 production-ready). What calibration requires:

1. A gold set of at least 100 samples, with two independent labelers
2. An inter-labeler agreement report (Cohen's kappa)
3. A precision/recall comparison of system output against human labels
4. Threshold selection justified via an ROC curve

Until these are done, the system is a **comparison tool, not a
certification tool.**
