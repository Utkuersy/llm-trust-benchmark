# 🛡️ LLM Output Reliability Evaluation Platform

[![CI](https://github.com/Utkuersy/llm-trust-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Utkuersy/llm-trust-benchmark/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-172%20passing-brightgreen)

An evaluation platform designed to run on an internal network, measuring the reliability of a corporate RAG assistant's **outputs**. Seven dimensions, a single Trust Score (0-100).

> **The problem:** An LLM assistant giving a wrong answer is a fixable error. It producing an answer containing profanity, insults, or leaked personal data is a corporate incident. This platform makes the second one measurable — and tells you at which pipeline stage the risk came from.

https://github.com/user-attachments/assets/e2d28287-368d-4d79-93b4-553bfb72ea04

## Table of contents

- [Dimensions measured](#dimensions-measured)
- [Pipeline tracing](#pipeline-tracing)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluating your own data](#evaluating-your-own-data)
- [Internal-network / air-gapped operation](#internal-network--air-gapped-operation)
- [Tests and validation](#tests-and-validation)
- [Docker](#docker)
- [Documents](#documents)
- [The enterprise-maturity layer](#the-enterprise-maturity-layer)
- [Known limitations](#known-limitations)
- [Project structure](#project-structure)
- [License](#license)

---

## Dimensions measured

The dimensions are distributed across ISO/IEC 25010:2023 quality characteristics. Weights are never hand-written; they are derived using a documented method (see [docs/METHODOLOGY.md](docs/METHODOLOGY.md)).

| Pillar | Dimension | What it measures | Weight |
|---|---|---|---|
| **Safety** | Content safety | Profanity, insults, attacks on religious values, threats, sexual content | 0.33 |
| **Security** | Injection resistance | The defense's break rate against 12 attack scenarios | 0.17 |
| | PII leakage | National ID, IBAN, card, API key, email in answers | 0.11 |
| | Poisoning resistance | The rate of being fooled by a fake document injected into the corpus | 0.06 |
| **Functional** | Retrieval | Context precision / recall, MRR | 0.11 |
| | Faithfulness | Hallucination rate, answer relevance | 0.11 |
| | Math | Answer accuracy (with symbolic equivalence checking) | 0.11 |

Weights come from the `output_safety_first` preset. Alternative presets exist in `core/scoring.py` (`owasp_rank`, `trustllm_equal`); switch with the `--preset` flag.

---

## Pipeline tracing

Looking only at the final output doesn't show *where* things broke down. Four stages are measured separately for every query:

```
  input_guardrail  ->  retrieval  ->  generation  ->  output_guardrail
       │                   │              │                 │
   violation/PII      the context      the answer      violation/PII
   in the question    retrieved        generated        in the answer
```

This distinction lets you say **"all 6 findings are at the output stage — the risk isn't coming from the user, it's coming from the model itself"** instead of just "there are 6 findings." Stage durations are also recorded; the bottleneck is detected automatically.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If the optional packages (`chromadb`, `sentence-transformers`, `transformers`, `ragas`) aren't installed, the system falls back in order to a NumPy vector store, a hashing embedding, and a heuristic faithfulness mode. Which backend was used is reported in the `backend` field of the output — quality doesn't silently degrade.

---

## Usage

```bash
# 1) Corpus and sample outputs
python -m rag.ingest --seed --reset
python -m scripts.generate_llm_outputs

# 2) Evaluation
python benchmark_engine.py
python benchmark_engine.py --models gemini --preset owasp_rank

# 3) Dashboard and experiment tracking
streamlit run app.py                            # http://localhost:8501
mlflow ui --backend-store-uri file:./mlruns     # http://localhost:5000
```

### Individual modules

```bash
python -m llm_security.content_safety_scan --outputs llm_outputs/gemini
python -m llm_security.pii_leakage_scan --outputs llm_outputs/gemini
python -m llm_security.prompt_injection_tests --model gemini
python -m llm_security.data_poisoning_sim
python -m capability.math_eval --outputs llm_outputs/gemini
python -m rag.retriever --query "what is the minimum password length"
```

---

## Evaluating your own data

### The easiest way: through the dashboard

In the dashboard opened with `streamlit run app.py`, the **"➕ Add Model"**
tab lets you add and evaluate a model without ever touching a terminal:
type the model name, upload `rag_answers.json` (required) and, if you
want, `injection_responses.json` / `math_answers.json` (optional), and
click "Evaluate and save" — the evaluation runs synchronously via
`benchmark_engine.run_benchmark` and the result appears instantly on the
**"📊 Overview"** tab. From the **"Delete a model"** section on the same
tab, you can permanently remove a model added by mistake (along with its
files and database records).

For the full list of which questions/scenarios you need to prepare
answers for, and JSON format examples: see the steps below, or run
`python -m llm_security.prompt_injection_tests --list` directly.

### By hand / from the command line

**1. Corpus.** Put your `.md` / `.txt` files into `data/rag_corpus/`, run `python -m rag.ingest --reset`.

**2. Model answers.** `llm_outputs/<model_name>/rag_answers.json`:

```json
[{
  "question": "The question asked",
  "answer": "The model's answer",
  "contexts": ["retrieved chunk 1", "chunk 2"],
  "expected_sources": ["correct_file.md"],
  "ground_truth": "The correct answer"
}]
```

`question` and `answer` are required. Without `contexts`, faithfulness can't be measured; without `expected_sources`, the retrieval dimension is skipped.

**3. Injection responses.** `injection_responses.json` — view the scenario texts with `python -m llm_security.prompt_injection_tests --list`, ask your own model, and save the answers as `{"INJ-01": "...", ...}`.

**4. Math.** `math_answers.json` — `{"M-01": "the model's answer", ...}`. The problem set is `data/math_eval/problems.jsonl`.

**5. Content lexicons.** `config/lexicons/*.txt` — **must be filled in by the organization.** A category left empty stays inactive and shows up in the report as "not measured"; it is not scored as zero.

> `religious_insult.txt` requires special care. Criticism of religion, theological debate, and academic study are not violations. Before filling this file in, get written guidance from the organization consisting of examples that **do and don't** count as violations.

---

## Internal-network / air-gapped operation

`config/settings.yaml` → `offline.enforce: true` (the default) applies the `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` variables process-wide. An evaluation run makes no request to any external service.

If the classifier layer is used, model weights are downloaded ahead of time and placed on disk:

```yaml
content_safety:
  classifier_backend: transformers
  classifier_model_path: /opt/models/turkish-offensive-bert
```

On the Docker side, the `evaluate` service runs with `network_mode: none`, `read_only: true`, `cap_drop: ALL`.

---

## Tests and validation

```bash
pytest tests/ -q          # 172 tests, 2 known limitations (xfail)
```

The test suite is split into three groups:

- **Functional tests** (`test_content_safety.py`) — the HateCheck method: each test probes a single behavior and includes **non-violating contrast cases**. Testing only with violation examples makes the false-positive rate invisible.
- **Robustness tests** (`test_robustness.py`) — ReDoS (CWE-1333), resource consumption (CWE-400), log injection (CWE-117), adversarial input types.
- **Validator tests** (`test_validators_and_scoring.py`) — Luhn (ISO/IEC 7812-1), IBAN mod-97 (ISO 13616-1), the TCKN check digit, math equivalence, weight normalization.

CI also runs a **weight-sensitivity test**: it runs with three different presets and verifies the model ranking doesn't change. The build goes red if the ranking changes with the weights.

---

## Docker

```bash
docker compose build
docker compose run --rm prepare       # corpus + sample outputs
docker compose run --rm evaluate      # evaluation (network disabled)
docker compose up dashboard           # :8501
docker compose up mlflow              # :5000
```

---

## Documents

- [docs/WORKING_PAPER.md](docs/WORKING_PAPER.md) — a single-file summary for presentations: architecture, frameworks, source mapping
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — the source basis for every parameter
- [docs/DEFENSE.md](docs/DEFENSE.md) — the problem/solution/source mapping for every design decision, and positioning against LLM-Stats
- [docs/GOVERNANCE_ALIGNMENT.md](docs/GOVERNANCE_ALIGNMENT.md) — the NIST AI RMF, ISO/IEC 42001, and (conditional) EU AI Act mapping

---

## The enterprise-maturity layer

Four layers added to move from being a "tool" to being a "standard":

**Audit trail** (`core/audit.py`) — every run immutably records, via a
hash chain, who/when/which code version/which configuration it ran
with. `python -m core.audit verify` checks chain integrity.

**Versioning and drift** (`core/versioning.py`) — the test data (corpus,
scenarios, lexicons) is hashed. Whether a "85 last month, 70 this month"
difference comes from a data change or from real model behavior is
disambiguated automatically: `python -m core.versioning drift --model gpt4`.

**Human calibration scaffolding** (`core/calibration.py`) — sampling,
labeling template, and correlation analysis tools are ready. **A real
human labeling study has not yet been done**; the `demo` command only
demonstrates the mechanism on synthetic data and marks it explicitly
with `is_synthetic: true`.

**A plugin architecture** (`core/dimensions.py`) — the seven dimensions
are no longer baked into the engine, they self-register. Adding a new
dimension does not require changing `benchmark_engine.py`; this claim is
proven by
`tests/test_dimension_registry.py::test_new_dimension_is_picked_up_without_engine_changes`.

**Multi-turn attack tests** — in addition to the 12 single-turn
scenarios, 4 multi-turn scenarios (`MULTI_TURN_SCENARIOS`) test attacks
built up over several messages, such as gradual authority-building and a
fake approval history.

**Alignment with governance frameworks** — see
[docs/GOVERNANCE_ALIGNMENT.md](docs/GOVERNANCE_ALIGNMENT.md). The
primary anchor is NIST AI RMF and ISO/IEC 42001 (voluntary,
geography-independent); the EU AI Act is directly relevant only if there
is an EU-market connection and the Article 2(3) military/defense
exemption doesn't apply — this document is not legal advice.

---

## Known limitations

The most important property of an evaluation tool is knowing what it cannot measure.

1. **Not calibrated.** The system *ranks* (is model A more trustworthy than B), it does not set an *absolute threshold* (is a score of 70 production-ready). Calibration requires a gold set of at least 100 samples with two independent labelers, and an inter-labeler agreement (Cohen's kappa) report. **As it stands, this is a comparison tool, not a certification tool.**
2. **The lexicon layer is not context-aware.** A term appearing in an academic, quoted, or counter-speech context is not a violation, but the lexicon can't tell the difference. This behavior is documented via `xfail` tests. Context disambiguation is the classifier layer's job.
3. **Heuristic faithfulness is not an LLM-as-judge.** RAGAS doesn't work without an LLM provider. The fallback mode catches numeric hallucinations well but can miss semantic contradictions.
4. **The injection scenarios are a closed set.** 12 scenarios represent six categories; the real attack surface keeps expanding. Regular updates are needed.
5. **Access control relies on a single shared password.** The dashboard is protected via `AITB__DASHBOARD__PASSWORD_HASH` (see `core/dashboard_auth.py`) and fully denies access if no password is defined (fail-closed), but it has no per-user roles or SSO. A multi-user authentication layer should be evaluated before an enterprise rollout.
6. **No independent security audit has been performed.** The robustness tests cover specific weakness classes; they do not replace a penetration test.

---

## Project structure

```
├── core/                    config, logging, schemas, storage, tracking, scoring, trace, process
├── llm_security/            content_safety_scan, pii_leakage_scan, prompt_injection_tests, data_poisoning_sim
├── rag/                     ingest, vector_store, retriever, rag_evaluator
├── capability/              math_eval
├── config/                  settings.yaml, lexicons/
├── tests/                   functional, robustness, validator, and trace tests
├── docs/METHODOLOGY.md      the source basis for every parameter
├── benchmark_engine.py      orchestration
└── app.py                   Streamlit dashboard
```

---

## License

MIT — see [LICENSE](LICENSE).
