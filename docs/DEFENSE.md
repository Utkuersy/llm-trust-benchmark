# Defense notes — the basis for every decision

This document answers a single question: **"Why did you do it this way?"**

Every line contains a problem, the solution applied to it, and the
source that solution rests on. You can keep this document open during a
presentation and answer any question about a decision from here.

---

## 0. Where this project sits: what LLM-Stats does, what this platform does

[llm-stats.com/benchmarks](https://llm-stats.com/benchmarks), which the
supervisor pointed to, is a **benchmark aggregator** that produces model
rankings across 680 benchmarks and 55 capability areas. Benchmark
results within each capability category are combined into a single
index using a conservative TrueSkill rating.

LLM-Stats's own methodology note directly states this platform's reason
for existing: rankings can shift with prompt format, harness version,
contamination, missing runs, and model updates; for that reason,
**multiple relevant tests should be used instead of a single universal
score**.

The difference is this:

| | LLM-Stats | This platform |
|---|---|---|
| What's measured | The model's **general capability** | The **output** of our pipeline |
| Data | Public benchmark sets | The organization's own corpus and questions |
| Language | Mostly English | Multilingual-capable content-safety layer |
| Scope | Reasoning, coding, math, vision… | Content safety, PII, injection, RAG quality |
| Where it runs | Public, online | Internal network / air-gapped |
| Question answered | "Which model is better?" | "Is our deployment production-ready?" |

**These are complementary, not competing.** LLM-Stats helps with model
selection; this platform measures what the selected model produces
*with our corpus, our prompt, for our users*. A public benchmark cannot
tell you whether a model leaks personal data while summarizing the
organization's internal documents.

**The math dimension is the bridge.** The math category the supervisor
specifically pointed to is also a capability area LLM-Stats indexes.
This platform's `capability/math_eval.py` module uses the answer
extraction and equivalence-checking approach of the MATH benchmark
family — so the internal measurement sits on the same methodological
ground as the public rankings, and is comparable to them.

---

## 1. Problem → solution → source mapping

### 1.1 "A single score is misleading"

**Problem.** A common mistake in model evaluation is collapsing
everything into one number. A model with a high overall score can still
be bad on the dimension the organization actually cares about (content
safety).

**Solution.** Seven separate dimensions are measured, each keeping its
own sub-score; the combined Trust Score is their weighted average, and
sub-scores are always reported separately. The dashboard shows the
dimensions first, then the total.

**Basis.** The LLM-Stats methodology note: multiple relevant tests
instead of one universal score. Also, TrustLLM (Sun et al., ICML 2024)
defines trustworthiness not as a single number but as separately
reported dimensions.

---

### 1.2 "The weights look arbitrary"

**Problem.** Combining dimensions requires weights. If the question "why
is content safety 33%" can't be answered, the entire score becomes
contestable.

**Solution — a two-layer derivation.**
1. ISO/IEC 25010:2023 defines 9 quality characteristics (functional
   suitability, performance efficiency, compatibility, interaction
   capability, reliability, security, maintainability, flexibility,
   safety). Of these, the 3 directly relevant to the project were
   selected — Safety, Security, Functional suitability — and **weighted
   equally between them**. This selection and equal weighting is this
   project's decision, not the standard's own.
2. The split within Security is done by applying a **rank-sum** method
   to the OWASP LLM Top 10 ranking: `w_i = (n+1-r_i) / Σ(n+1-r)`.

**Basis.**
- ISO/IEC 25010:2023 — the full list of 9 characteristics; the 3
  relevant ones were selected for this project. The standard also
  assigns no priority ordering among the 6 unselected characteristics
  (performance efficiency, compatibility, interaction capability,
  reliability, maintainability, flexibility) — so weighting the selected
  3 equally is in the spirit of the standard, though not something the
  standard directly mandates.
- Dawes, R. M. (1979), *The robust beauty of improper linear models in
  decision making*, American Psychologist 34(7) — shows that, absent
  calibration data, equal weights are robust.
- Barron & Barrett (1996), *Decision Quality Using Ranked Attribute
  Weights*, Management Science 42(11) — deriving weight from ranked preference.
- OWASP Top 10 for LLM Applications 2025 (v2.0, November 18, 2024) — the
  LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure, LLM04
  Data and Model Poisoning ranking.

**The sentence to say.** *"I didn't measure the weights, I derived them
with a documented method. This is not a calibration, it's a justified
starting point."*

**What I did measure:** the effect of the weight choice on the outcome.
It was run with three different presets (`output_safety_first`,
`owasp_rank`, `trustllm_equal`) and the model ranking did not change.
This test runs in CI; the build breaks if the ranking changes. **This is
a genuine measurement claim.**

---

### 1.3 "The content filter is easily bypassed"

**Problem.** Lexicon-based filters are bypassed by misspelling a word or
inserting punctuation between letters. This is not a theoretical risk,
it is a documented attack.

**Solution — a normalization layer.** Text is normalized before
scanning: lowercasing, resolving character substitution (`@→a`, `1→i`),
merging inter-letter separators (`i.d.i.o.t → idiot`), collapsing three
or more repeated letters into one (`idiooot → idiot`), Unicode NFKC
normalization.

**Basis.** Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. (2017),
*Deceiving Google's Perspective API Built for Detecting Toxic Comments*,
arXiv:1702.08138. The study shows the system can be bypassed by
misspelling profane words or inserting punctuation between letters;
turning "idiot" into "idiiot" dropped that same sentence's toxicity
score from 84% to 20%.

**Evidence.** This behavior is tested under
`tests/test_content_safety.py::test_f2_*` with six separate evasion
techniques. When the test suite was written, a real bug was found and
fixed in the letter-repetition normalization — the system was letting
`idiooot` slip through.

---

### 1.4 "The filter's false-positive rate is unknown"

**Problem.** Testing a content filter only with violation examples is
misleading: a filter that flags everything also looks 100% successful.
A filter cannot go to production without its false-positive rate being measured.

**Solution — a functional test suite + contrast cases.** Each test
probes a single behavior. Some of the tests are **non-violating** texts
(neutral corporate text, polite criticism, academic context), and these
are expected not to be flagged.

**Basis.** Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H.,
& Pierrehumbert, J. (2021), *HateCheck: Functional Tests for Hate Speech
Detection Models*, ACL-IJCNLP 2021, 41-58, DOI 10.18653/v1/2021.acl-long.4.
The study defines 29 model functionalities drawn from a review of prior
research and interviews with civil-society stakeholders, validates the
test cases through a structured labeling process, and publishes 3,728
cases across the 29 functional tests; every case carries a hateful/
non-hateful gold label. Testing with this method surfaced critical
weaknesses in both academic and commercial models.

**An honesty note.** Two contrast cases currently **fail** and are
marked `xfail`: the sentence "I read a psychology article about
stupidity" produces a false positive. This is not hidden — the
lexicon layer's lack of context-awareness is an architectural limit and
is documented. The test halts with `strict=True`: if a classifier is
added later and the behavior is fixed, the test will raise a warning.

---

### 1.5 "Identity number detection produces false positives"

**Problem.** With plain pattern matching, every 11-digit number is
treated as a national ID, every 16-digit number as a card number. Order
numbers, product codes, and dates get mistakenly flagged as PII.

**Solution — structural validation.** Every numeric identity type is
run through its own check algorithm; a match that fails validation is
not reported.

**Basis.**
- ISO/IEC 7812-1 — the Luhn check digit for card numbers
- ISO 13616-1 — the IBAN mod-97 check
- The Turkish National ID Number's 10th/11th-digit algorithm

**Evidence.** Tested with valid and invalid examples in
`tests/test_validators_and_scoring.py`. A separate test confirms that
most random 11-digit numbers are rejected — this is the validator's
reason for existing.

---

### 1.6 "The math answer is correct but the system counts it wrong"

**Problem.** `1/2` and `0.5`, `1,200` and `1200`, `x^2-9` and `x**2-9`
are the same answer. Raw string comparison counts these as wrong and
understates the accuracy rate.

**Solution — a three-tier equivalence check.** First normalized string
equality, then numeric equivalence within a tolerance, then symbolic
equivalence via SymPy. The answer is also extracted from free text
(`\boxed{}`, "Answer:", last-line patterns).

**Basis.**
- Hendrycks, D. et al. (2021), *Measuring Mathematical Problem Solving
  With the MATH Dataset*, NeurIPS Datasets & Benchmarks — the need for
  answer normalization and equivalence checking
- Lewkowycz, A. et al. (2022), *Solving Quantitative Reasoning Problems
  with Language Models* (Minerva), NeurIPS 2022

**An additional decision.** Questions whose answer **cannot be
extracted** are not counted as wrong; they are reported separately
(`extraction_failures`). Rationale: a format mismatch and a reasoning
error call for different actions — one is a prompt fix, the other a
model change.

**Evidence.** The test suite found a real bug in this layer too: the
`extract_answer` function was mistaking the "answer" substring inside a
word like "I can't answer this" for a marker and returning garbage. A
word boundary and a mandatory delimiter were added.

---

### 1.7 "The answer is bad but it's unclear where it broke down"

**Problem.** RAG is a chain. Looking only at the final output, it's
impossible to tell whether retrieval, the model, or the filter is
responsible. Root-cause analysis can't be done.

**Solution — a pipeline tracing layer.** Four stages are measured
separately for every query: `input_guardrail → retrieval → generation →
output_guardrail`. Each stage's duration, status, input/output summary,
and the guardrail findings triggered at that stage are recorded.

**Concrete gain.** Instead of saying "there are 6 findings," you can say
**"all 6 findings are at the output stage — the risk isn't coming from
the user, the model itself is leaking PII."** These two situations call
for different actions: one is an input filter, the other a model or
prompt change.

**Basis.** This is the standard approach to LLM observability; it's the
span/trace model from distributed systems adapted to a RAG pipeline.
OWASP LLM Top 10 2025 defines LLM05 (Improper Output Handling) and LLM08
(Vector and Embedding Weaknesses) as separate risks — meaning output and
retrieval must be audited separately.

---

### 1.8 "This tool is itself an attack surface"

**Problem.** By definition, the platform processes untrusted text: the
output of the model under evaluation can be controlled by an attacker.
The scanner itself can be a target.

**Solution — a robustness test suite.** Four weakness classes are tested:

| Weakness | Source | Test |
|---|---|---|
| ReDoS — exponential backtracking in a regex | CWE-1333; OWASP ReDoS | A time limit on pathological input |
| Uncontrolled resource consumption | CWE-400; OWASP ASVS v4.0.3 V5.1 | Input size truncation |
| Log injection | CWE-117 | A ban on control characters in the context field |
| Command injection | CWE-78 | `shell=True` is used nowhere |

**An additional decision.** Reports never carry raw sensitive data;
findings are masked (`joh***@***le`). Rationale: an evaluation report is
a shareable artifact; writing raw PII into a report would multiply the
leak instead of measuring it.

---

### 1.9 "An unmeasured dimension looks like a clean one"

**Problem.** If a lexicon is left empty, no violation is found and the
dimension gets a score of 100. This is a false sense of assurance — the
most dangerous failure mode of an evaluation tool.

**Solution.** Unmeasured categories are reported under
`inactive_categories` and shown as a yellow warning on the dashboard.
Also, a dimension in `skipped` status has its weight removed from the
denominator and distributed to the remaining dimensions — it is not
scored as zero, it is removed from the calculation.

**The sentence to say.** *"'No violation found' and 'not scanned' are
not the same thing; the system never conflates the two."*

---

### 1.10 "It can't run on an internal network, it tries to download a model"

**Problem.** Modern NLP libraries download model weights at runtime. In
an air-gapped environment, this causes silent failures.

**Solution — a three-tier fallback + enforced offline mode.**
- Vector store: Chroma → FAISS → pure NumPy
- Embedding: sentence-transformers → hashing embedding
- Faithfulness: RAGAS → heuristic

The `offline.enforce: true` setting applies the `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` variables process-wide. The classifier model is
loaded from local disk. In Docker, the evaluation service runs with
`network_mode: none`.

**A critical point.** Which backend was used is reported in the
`backend` field of the output. Quality does not silently degrade, it is
transparently labeled.

---

## 2. Layer architecture — "how many layers, and why"

Seven layers, dependency flows one way (the layer above imports the one
below, never the reverse):

| # | Layer | Responsibility | Why separate |
|---|---|---|---|
| 1 | **Configuration** | `settings.yaml` + Pydantic | No module has a hardcoded threshold/weight; overridable via env var |
| 2 | **Core infrastructure** | Schemas, JSON logging, safe subprocess | A shared need for every analysis module; written once |
| 3 | **Analysis** | 7 dimensions, each its own module | Every module can run standalone; if one breaks, the others still work |
| 4 | **Pipeline tracing** | Stage-by-stage trace | Root-cause analysis; where things broke down |
| 5 | **Scoring** | Source-traceable weight presets | Changing a weight needs no code change; the version is tagged |
| 6 | **Storage and traceability** | SQLite + MLflow + JSON | Every run is reproducible; the scoring version is recorded |
| 7 | **Presentation** | Streamlit dashboard | Read-only; needs no heavy dependency |

**Why this separation matters.** If the organization says tomorrow
"content safety weight should be 50%," a single YAML line changes. If
they say "add a new injection scenario," a single file changes. If they
say "use Qdrant instead of Chroma," a single class is added. None of
these break the others.

---

## 3. Provable claims (what you can confidently say in a presentation)

1. **"The test suite found and fixed four real bugs."** One was an
   evasion technique that bypassed the security filter, one a log
   injection vector, one an answer-extraction error, one a field-mapping
   error that wrote zero into the database.
2. **"The result is not sensitive to the weight choice."** The model
   ranking is the same across three different presets; this is verified
   automatically in CI.
3. **"The system can tell an unsafe example apart."** A model that leaks
   PII and falls for injection scores markedly lower than clean models;
   this stands as an assertion in CI.
4. **"I can say at which stage of the pipeline a risk arose."**
   Stage-by-stage finding distribution and bottleneck detection are
   recorded.
5. **"135 automated tests, 2 documented limitations."** The limitations
   aren't hidden, they're marked `xfail` with a reason.

---

## 4. Unprovable claims (never say these)

1. ❌ *"I measured the weights."* → I derived them. Measuring would
   require an outcome variable and regression; no such data exists.
2. ❌ *"The system is calibrated."* → It is not. Calibration requires a
   gold set of at least 100 samples with two independent labelers, and a
   Cohen's kappa report.
3. ❌ *"It's secure enough to run on an internal network."* → The
   robustness tests cover specific weakness classes; they do not replace
   an independent penetration test. Also, the dashboard has no
   authentication if it's disabled by configuration.
4. ❌ *"Content safety is 100%."* → Until the lexicons are filled in by
   the organization, this number means "not scanned."

**State these four limits yourself, don't wait to be asked.** An
evaluation tool that knows its own limits is more trustworthy than one
that doesn't.

---

## 5. Source list

Sources whose content was directly verified in this session:

- Röttger et al. (2021), HateCheck, ACL-IJCNLP 2021, DOI 10.18653/v1/2021.acl-long.4
- Hosseini et al. (2017), Deceiving Google's Perspective API, arXiv:1702.08138
- OWASP Top 10 for LLM Applications 2025 (v2.0, November 18, 2024)
- llm-stats.com/benchmarks — methodology and scope

Standard references (verify the citation yourself before using it):

- ISO/IEC 25010:2023 — the software quality model
- ISO/IEC 7812-1 — the Luhn check digit
- ISO 13616-1 — IBAN
- CWE-1333, CWE-400, CWE-117, CWE-78 — weakness classes
- OWASP ASVS v4.0.3 — input validation requirements
- Dawes (1979), American Psychologist 34(7), 571-582
- Barron & Barrett (1996), Management Science 42(11), 1515-1523
- Hendrycks et al. (2021), MATH Dataset, NeurIPS D&B
- Lewkowycz et al. (2022), Minerva, NeurIPS
- Sun et al. (2024), TrustLLM, ICML
- Çöltekin (2020), A Corpus of Turkish Offensive Language, LREC

**Warning.** If you plan to cite a source in a presentation, open and
read it first. Saying "I did this based on that paper" and then not
being able to answer a question about the paper is worse than citing no
source at all.
