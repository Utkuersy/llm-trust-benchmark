# Working Paper
## LLM Output Reliability Evaluation Platform

**Prepared by:** [name] · **Date:** [date] · **Scope:** Internship project interim report

---

## 1. The project in one paragraph

An evaluation platform designed to run on an internal network, measuring
the reliability of a corporate RAG assistant's **outputs**. Seven
dimensions are measured (content safety, prompt injection resistance,
PII leakage, data poisoning, retrieval quality, faithfulness, math
capability) and combined into a single Trust Score (0-100). The
dimensions are also reported separately. The system also shows at which
pipeline stage a risk arose. The weights are not arbitrary; they are
derived through a documented method based on ISO/IEC 25010 and the OWASP
LLM Top 10.

---

## 2. Problem statement and positioning

### 2.1 The problem solved

An LLM assistant giving a **wrong** answer is a fixable error. But it
producing an answer containing profanity, insults, an attack on
religious values, or leaked personal data is a corporate incident. This
second risk class is not measured in most evaluation setups.

A second problem: RAG is a chain (question → context retrieval →
generation → filter). Looking only at the final output, it's impossible
to tell *where* a problem arose. Root-cause analysis can't be done.

### 2.2 Relationship to existing solutions — LLM-Stats

[llm-stats.com/benchmarks](https://llm-stats.com/benchmarks), which the
supervisor pointed to, is a **benchmark aggregator** that produces model
rankings across 680 benchmarks and 55 capability areas. Results within
each category are combined using a conservative TrueSkill rating.

LLM-Stats's own methodology note directly states this project's
rationale: rankings can shift with prompt format, harness version,
contamination, missing runs, and model updates; for that reason,
**multiple relevant tests should be used instead of a single universal
score**.

| | LLM-Stats | This platform |
|---|---|---|
| What it measures | The model's general capability | The output of our pipeline |
| Its data | Public benchmark sets | The organization's own corpus and questions |
| Its language | Mostly English | Multilingual-capable content-safety layer |
| Its scope | Reasoning, coding, math, vision… | Content safety, PII, injection, RAG quality |
| Where it runs | Public, online | Internal network / air-gapped |
| Question it answers | "Which model is better?" | "Is our deployment production-ready?" |

**Conclusion:** These are complementary, not competing. LLM-Stats helps
with model *selection*; this platform measures what the selected model
produces under our conditions. A public benchmark cannot tell you
whether a model leaks a national ID number while summarizing internal
documents.

**Math is the bridge.** The math category the supervisor specifically
pointed to is also indexed by LLM-Stats (the MATH, AIME, MMLU-Pro
families). This platform's math module uses the MATH benchmark family's
answer-extraction and equivalence-checking approach; so the internal
measurement sits on the same methodological ground as the public
rankings.

---

## 3. Architecture — seven layers

Dependency is one-way: the layer above uses the one below, never the
reverse. There is no circular dependency.

```
┌──────────────────────────────────────────────────────────┐
│  7. PRESENTATION     app.py (Streamlit + Plotly)          │
│                      read-only, triggers no run           │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  6. STORAGE          SQLite · MLflow · results/*.json     │
│                      every run is reproducible            │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  5. SCORING          core/scoring.py                      │
│                      source-traceable weight presets      │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  4. PIPELINE TRACING core/trace.py                        │
│     input_guardrail → retrieval → generation → output_gr. │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  3. ANALYSIS (7 dimensions, each an independent module)   │
│     content_safety · injection · pii · poisoning          │
│     retrieval · faithfulness · math                       │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  2. CORE             schemas · JSON logging · subprocess  │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  1. CONFIGURATION    settings.yaml + Pydantic             │
│                      no hardcoded value in any module     │
└──────────────────────────────────────────────────────────┘
```

### The rationale for each layer

| Layer | Responsibility | Why it must be separate |
|---|---|---|
| 1 Configuration | All thresholds, weights, and paths | A weight change needs no code change; overridable via env var |
| 2 Core | Schema, logging, safe subprocess | A shared need for every analysis module; written once |
| 3 Analysis | 7 dimensions, 7 separate modules | Every module can run standalone; if one breaks, the others still work |
| 4 Pipeline tracing | Stage-by-stage trace | Root-cause analysis; where a risk came from |
| 5 Scoring | Weight presets + versioning | The weight scheme is versioned; old scores stay comparable |
| 6 Storage | SQLite + MLflow + raw JSON | Traceability; every run is reproducible |
| 7 Presentation | Dashboard | Read-only; needs no heavy dependency, can be deployed separately |

**The practical payoff of this separation:** If the organization says
"content safety weight should be 50%," a single YAML line changes. If
they say "add a new injection scenario," a single file changes. If they
say "use Qdrant instead of Chroma," a single class is added. None of
these break the others.

---

## 4. Evaluation framework

### 4.1 The seven dimensions and their weights

The dimensions are mapped onto the 3 characteristics directly relevant
to the project — Safety, Security, Functional suitability — out of the 9
quality characteristics ISO/IEC 25010:2023 defines (functional
suitability, performance efficiency, compatibility, interaction
capability, reliability, security, maintainability, flexibility,
safety). The other 6 characteristics (performance, compatibility,
portability, etc.) are outside the scope of this evaluation.

| ISO pillar | Dimension | What it measures | Weight |
|---|---|---|---|
| **Safety** | Content safety | Profanity, insults, attacks on religious values, threats, sexual content | 0.33 |
| **Security** | Injection resistance | The defense's break rate across 12 attack scenarios | 0.17 |
| | PII leakage | National ID, IBAN, card, API key, email, phone | 0.11 |
| | Poisoning resistance | The rate of being fooled by a fake document injected into the corpus | 0.06 |
| **Functional** | Retrieval | Context precision / recall, MRR | 0.11 |
| | Faithfulness | Hallucination rate, answer relevance | 0.11 |
| | Math | Answer accuracy (with symbolic equivalence checking) | 0.11 |

### 4.2 How the weights were derived

**A two-step method:**

**Step 1 — top level: equal weight.** An equal split (1/3 each) among
the 3 selected from ISO's 9 characteristics (Safety, Security,
Functional suitability). This selection and equalization is the
project's decision; the standard itself does not group these 3
separately, it defines all 9 at the same level and assigns no priority
among them. The literature shows that, absent calibration data, equal
weighting is a robust choice.

> Dawes, R. M. (1979). *The robust beauty of improper linear models in
> decision making.* American Psychologist, 34(7), 571-582.

**Step 2 — the split within security: rank-sum.** The OWASP ranking is
converted to a numeric weight: `w_i = (n + 1 − r_i) / Σ(n + 1 − r)` → 3 : 2 : 1.

> Barron, F. H. & Barrett, B. E. (1996). *Decision Quality Using Ranked
> Attribute Weights.* Management Science, 42(11), 1515-1523.
>
> OWASP Top 10 for LLM Applications 2025 (v2.0, November 18, 2024):
> LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure,
> LLM04 Data and Model Poisoning.

**The arithmetic:** `0.33 (Safety) + [0.17 + 0.11 + 0.06] (Security) +
[0.11 × 3] (Functional) = 1.00`

### 4.3 The critical scoring rule

If a dimension can't be measured (an empty lexicon, no answer file), it
is **not scored as zero**; that dimension's weight is removed from the
denominator and distributed proportionally to the remaining dimensions.
Otherwise a missing piece of infrastructure would unfairly penalize the
model — a common mistake in evaluation systems.

Unmeasured categories also appear in the report under
`inactive_categories`. **"No violation found" is never conflated with
"not scanned."**

---

## 5. What was done — and on what basis

### 5.1 Normalization against content-filter evasion techniques

**Problem.** Lexicon-based filters are bypassed by misspelling a word or
inserting punctuation between letters.

**Solution.** A normalization layer runs before scanning: lowercasing,
resolving character substitution (`@→a`, `1→i`), merging inter-letter
separators (`i.d.i.o.t → idiot`), collapsing three or more repeated
letters into one, Unicode NFKC. Root + suffix-tolerant matching is also
applied for morphologically rich languages.

**Source.**
> Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017). *Deceiving
> Google's Perspective API Built for Detecting Toxic Comments.*
> arXiv:1702.08138.

The study shows the system can be bypassed by misspelling profane words
or inserting punctuation between letters; "idiot" → "idiiot" dropped
that same sentence's toxicity score from **84% to 20%**. The same study
also reports that the system does not give negated profane statements a
low score.

---

### 5.2 A functional test suite and contrast cases

**Problem.** Testing a content filter only with violation examples is
misleading: a filter that flags everything also looks 100% successful.

**Solution.** Each test probes a single behavior. Some tests are
**non-violating** texts (neutral corporate text, polite criticism,
academic context) and are expected not to be flagged.

**Source.**
> Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., &
> Pierrehumbert, J. (2021). *HateCheck: Functional Tests for Hate Speech
> Detection Models.* ACL-IJCNLP 2021, 41-58.
> DOI: 10.18653/v1/2021.acl-long.4

The study defines 29 model functionalities drawn from a review of prior
research and interviews with civil-society stakeholders, validates test
cases through a structured labeling process, and publishes **3,728
cases** across the 29 functional tests; each case carries a hateful/
non-hateful gold label. Testing with this method surfaced critical
weaknesses in both academic and commercial models.

---

### 5.3 Identity number validation

**Problem.** With plain pattern matching, every 11-digit number is
treated as a national ID, every 16-digit number as a card number; order
numbers get mistakenly flagged as PII.

**Solution.** Every numeric identity is run through its own check
algorithm; a match that fails validation is not reported.

**Sources.** ISO/IEC 7812-1 (the Luhn check digit) · ISO 13616-1 (IBAN
mod-97) · the Turkish National ID Number's 10th/11th-digit algorithm.

---

### 5.4 Math answer equivalence

**Problem.** `1/2` and `0.5`, `1,200` and `1200`, `x^2-9` and `x**2-9`
are the same answer. Raw string comparison understates accuracy.

**Solution.** A three-tier check: normalized string equality → numeric
equivalence within a tolerance → symbolic equivalence via SymPy. The
answer is extracted from free text (`\boxed{}`, "Answer:", last-line
patterns). Questions whose answer **can't be extracted** are **not
counted as wrong**, they're reported separately — a format mismatch and
a reasoning error call for different actions.

**Sources.**
> Hendrycks, D. et al. (2021). *Measuring Mathematical Problem Solving With the
> MATH Dataset.* NeurIPS Datasets and Benchmarks Track.
>
> Lewkowycz, A. et al. (2022). *Solving Quantitative Reasoning Problems with
> Language Models* (Minerva). NeurIPS 2022.

---

### 5.5 Pipeline tracing

**Problem.** If an answer is bad, it's unclear whether retrieval, the
model, or the filter is responsible.

**Solution.** Four stages are measured separately for every query;
duration, status, an input/output summary, and the guardrail findings
triggered at that stage are recorded.

**Concrete gain.** Instead of "there are 6 findings," you can say **"all
6 findings are at the output stage — the risk isn't coming from the
user, the model itself is leaking PII."** The two situations call for
different actions: one is an input filter, the other a model/prompt
change.

**Basis.** Adapting the span/trace model from distributed systems to a
RAG pipeline. OWASP LLM Top 10 2025 defines LLM05 (Improper Output
Handling) and LLM08 (Vector and Embedding Weaknesses) as separate risks
— meaning output and retrieval must be audited separately.

---

### 5.6 The platform's own security

**Problem.** This tool, by definition, processes untrusted text; the
output of the model under evaluation can be controlled by an attacker.

**Solution.** Automated tests for four weakness classes:

| Weakness | Source | Mitigation |
|---|---|---|
| ReDoS — exponential backtracking in a regex | CWE-1333, OWASP ReDoS | A time-limit test on pathological input |
| Uncontrolled resource consumption | CWE-400, OWASP ASVS v4.0.3 V5.1 | Input size truncation |
| Log injection | CWE-117 | A ban on control characters in the context field |
| Command injection | CWE-78 | `shell=True` is used nowhere |

Reports additionally never carry raw sensitive data; findings are
masked. Rationale: an evaluation report is a shareable artifact; writing
raw PII into it would multiply the leak instead of measuring it.

---

### 5.7 Internal-network / air-gapped operation

**Problem.** Modern NLP libraries download model weights at runtime;
this causes silent failures in an air-gapped environment.

**Solution.** A three-tier fallback chain — vector store (Chroma →
NumPy), embedding (sentence-transformers → hashing), faithfulness
(RAGAS → heuristic). The `offline.enforce` setting applies the
`HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` variables. In Docker, the
evaluation service runs with `network_mode: none`.

**A critical point.** Which backend was used is reported in the
`backend` field of the output. Quality does not silently degrade, it is
transparently labeled.

---

## 6. Validation

### 6.1 The test suite

135 automated tests, three groups:

- **Functional tests** — the HateCheck method, including contrast cases
- **Robustness tests** — ReDoS, resource consumption, log injection,
  adversarial input types
- **Validator tests** — Luhn, IBAN, TCKN, math equivalence, weight
  normalization

### 6.2 Real bugs the test suite found

| Bug | Impact |
|---|---|
| Letter-repetition normalization worked incorrectly | The security filter let a known evasion technique through |
| The PII context field kept a line break | A log injection hole (CWE-117) |
| Answer extraction had no word boundary | The "answer" substring inside "I can't answer" was mistaken for a marker |
| Database field mapping was wrong | Pipeline data was silently written as zero |

Without the tests, all four would have kept silently producing wrong results.

### 6.3 Automated checks in CI

1. **A discrimination test** — a model that leaks PII and falls for
   injection must score lower than clean models; the build breaks if
   not.
2. **A pipeline test** — every model must have a stage trace recorded.
3. **A weight-sensitivity test** — the model ranking must not change
   across three different weight presets.

The third is especially important: **the weight choice is debatable, but
the result has been shown not to be sensitive to it.** Unlike the
derived-weight claim, this is a claim of *actual measurement*.

---

## 7. Known limitations

The most important property of an evaluation tool is knowing what it
cannot measure.

1. **Not calibrated.** The system *ranks*, it does not set an *absolute
   threshold*. Calibration requires a gold set of at least 100 samples
   with two independent labelers, and an inter-labeler agreement
   (Cohen's kappa) report. **It's a comparison tool, not a certification
   tool.**
2. **The lexicon layer is not context-aware.** A term appearing in an
   academic or quoted context is not a violation, but the lexicon can't
   tell the difference. This behavior is documented via `xfail` tests,
   not hidden.
3. **Heuristic faithfulness is not an LLM-as-judge.** RAGAS doesn't work
   without an LLM provider; the fallback mode catches numeric
   hallucinations well but can miss semantic contradictions.
4. **The lexicons are still empty.** Until the organization fills them
   in, the content safety score means partly "not scanned," not "clean."
5. **There is no access control.** [Note: this limitation has since been
   addressed — see the dashboard password protection in
   `core/dashboard_auth.py`.]
6. **No independent security audit has been performed.** The robustness
   tests cover specific weakness classes; they do not replace a
   penetration test.

---

## 7.5. The enterprise-maturity layer (added per the supervisor's feedback)

At the supervisor's request, the system was moved from being "a tool
that works well" to "a reference the organization trusts":

| Added | What it solves | File |
|---|---|---|
| Governance framework mapping | Answers "which standard does this rest on" | `docs/GOVERNANCE_ALIGNMENT.md` |
| Audit trail | "Prove how you arrived at this result" | `core/audit.py` |
| Versioning + drift | "Why did the score change — data or model" | `core/versioning.py` |
| Human calibration scaffolding | "Is this number actually meaningful" (infrastructure ready, awaiting the real study) | `core/calibration.py` |
| A plugin architecture | "Does adding a new dimension break the engine" — no, proven | `core/dimensions.py` |
| Multi-turn attack tests | The gradual attacks a single-message scan misses | `llm_security/prompt_injection_tests.py` |

**A critical correction.** In the original plan, "EU AI Act compliance"
was the top priority. Research led to changing this framing: AI Act
Article 2(3) excludes military/defense-purpose systems from scope, and
there's also a jurisdiction question. NIST AI RMF and ISO/IEC 42001
(voluntary, geography-independent) were made the primary anchor instead,
with the EU AI Act moved to a secondary "non-binding but a good design
reference" position. Be sure to explain this distinction to the
supervisor — making an incorrect legal claim is worse than making no
claim at all.

## 8. Next steps (in priority order)

1. **Filling in the lexicons.** Written guidance from the organization:
   20-30 examples counted as violations **and** 20-30 borderline
   examples not counted as violations. Without the second list, the
   false-positive rate cannot be measured.
2. **Calibration.** A 100-sample gold set, two labelers, a kappa report,
   a precision/recall comparison of system output against human labels.
3. **Authentication.** Before the dashboard goes live on the internal network.
4. **A local classifier.** For implicit toxicity the lexicon can't
   catch; with model weights downloaded to disk beforehand.

---

## 9. Likely questions and their answers

**"How did you determine the weights?"**
A two-step derivation: equal weight among the ISO/IEC 25010
characteristics (Dawes 1979), then a rank-sum applied to the OWASP LLM
Top 10 ranking within security (Barron & Barrett 1996). **I didn't
measure them, I derived them** — measuring would require an outcome
variable and regression, and no such data exists.

**"Would the result change if the weights were different?"**
The absolute scores change, the ranking doesn't. This was tested with
three presets and is automatically verified in CI.

**"Is this system secure, can we run it on the internal network?"**
There are automated tests against four weakness classes, and they pass.
However, no independent penetration test has been done. I can say
"tested," not "audited."

**"Content safety comes out at 100, is the system working?"**
The lexicons are still empty; this score largely means "not scanned."
The dashboard shows this as a yellow warning. This dimension produces no
meaningful result until a real list is entered.

**"Why do we need this when LLM-Stats exists?"**
LLM-Stats measures the model's general capability, not our pipeline's
output. A public benchmark cannot tell you whether a model leaks a
national ID number while summarizing our internal documents. Also,
LLM-Stats's own methodology recommends multiple relevant tests instead
of a single universal score.

**"Why did you remove the code-analysis track?"**
The organization's problem isn't the code the AI writes, it's what the
assistant tells the user. Instead of shipping a system half of which
goes unused, I focused on the one track that matters.

---

## 10. Bibliography

**Sources whose content was directly verified:**

1. Röttger, P. et al. (2021). *HateCheck: Functional Tests for Hate Speech
   Detection Models.* ACL-IJCNLP 2021, 41-58. DOI 10.18653/v1/2021.acl-long.4
2. Hosseini, H. et al. (2017). *Deceiving Google's Perspective API Built for
   Detecting Toxic Comments.* arXiv:1702.08138
3. OWASP Foundation (2024). *OWASP Top 10 for Large Language Model
   Applications 2025* (v2.0, November 18, 2024)
4. llm-stats.com/benchmarks — scope and methodology note

**Standard references** (verify the citation yourself before presenting):

5. ISO/IEC 25010:2023 — the software quality model (9 characteristics; 3
   relevant ones used in this project)
6. ISO/IEC 7812-1 — card number identification (Luhn)
7. ISO 13616-1 — IBAN
8. CWE-1333, CWE-400, CWE-117, CWE-78 — weakness classes
9. OWASP ASVS v4.0.3 — input validation requirements
10. Dawes, R. M. (1979). American Psychologist, 34(7), 571-582
11. Barron, F. H. & Barrett, B. E. (1996). Management Science, 42(11), 1515-1523
12. Hendrycks, D. et al. (2021). MATH Dataset. NeurIPS D&B
13. Lewkowycz, A. et al. (2022). Minerva. NeurIPS
14. Sun, L. et al. (2024). TrustLLM. ICML
15. Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social
    Media.* LREC 2020

> **Warning:** If you plan to cite a source in a presentation, open and
> read it first. Saying "I did this based on that paper" and then not
> being able to answer a question about the paper is worse than citing
> no source at all.
