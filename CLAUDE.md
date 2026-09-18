# CLAUDE.md

This file is the context file Claude Code automatically reads when
working on this project. Purpose: to hand Claude Code past design
decisions and known limits, without having to re-explain the project
from scratch.

## What the project is

An evaluation platform designed to run on an internal network, measuring
the **output** reliability of a corporate RAG assistant. It audits the
model's answers, not its code. It measures across seven dimensions and
produces a single 0-100 Trust Score.

## Critical history — don't try to re-propose these

**Track A (the code-trustworthiness track) was deliberately removed.**
The project was originally set up as two parallel tracks: code analysis
(Pylint/Bandit/sandbox) and LLM output analysis. Based on the internship
supervisor's feedback, the code track was fully deleted because the
organization's priority is *what the assistant says*, not *how the AI
writes code*. Unless a task explicitly asks for something like "also add
code quality," don't bring this back up.

**The "EU AI Act compliance" claim was deliberately abandoned.** The
original plan's top priority was EU AI Act compliance. Research showed
this was the wrong framing: Article 2(3) excludes military/defense
systems from scope, and the company is also based outside the EU.
**NIST AI RMF** and **ISO/IEC 42001** (voluntary, geography-independent)
were made the primary anchor instead; the EU AI Act is kept only as a
secondary "non-binding but a good design reference." Details:
`docs/GOVERNANCE_ALIGNMENT.md`. Never loosen this distinction — don't
write a sentence like "we are subject to the AI Act."

**Weights were "derived," not "measured."** 3 of ISO/IEC 25010's 9
characteristics (Safety, Security, Functional suitability) were selected
for this project and weighted equally between them — this is the
project's decision, not the standard's own mandate. The sub-split within
Security (injection:pii:poisoning = 3:2:1) came from a rank-sum method
applied to the OWASP LLM Top 10 ranking. See `docs/METHODOLOGY.md`.
Never write "we measured the weights"; say "we derived them" or "we
selected them with a documented method."

**Human calibration is scaffolding, not real.** `core/calibration.py`
contains sampling + analysis tools, but there is no real human labeler
data. The `demo` command marks every number it produces with
`is_synthetic: true`. Never present this module as "calibrated"; that
claim is not true until a real human labeling study is done.

## Architecture — seven layers, one-way dependency

```
config/settings.yaml (Pydantic)
  → core/ (schemas, logging, storage, tracking, scoring, trace, audit, versioning)
    → llm_security/, rag/, capability/ (the seven dimensions' analysis modules)
      → core/dimensions.py (the registry — dimensions register themselves)
        → benchmark_engine.py (orchestration)
          → app.py (Streamlit dashboard, read-only)
```

**Dimensions are registered, not hardcoded into the engine (registry
pattern).** Adding a new measurement dimension does not require changing
`benchmark_engine.py` — a `register(Dimension(...))` call inside
`core/dimensions.py` is enough. This claim is proven by
`tests/test_dimension_registry.py`; if you add a new dimension, don't
break that test's logic.

## The seven dimensions and their ISO pillars

| Pillar | Dimension | Weight |
|---|---|---|
| Safety | content_safety | 0.33 |
| Security | injection (OWASP LLM01) | 0.17 |
| Security | pii (OWASP LLM02) | 0.11 |
| Security | poisoning (OWASP LLM04) | 0.06 |
| Functional | retrieval | 0.11 |
| Functional | generation (faithfulness) | 0.11 |
| Functional | math | 0.11 |

## Fallback chains — deliberate design, not "incomplete"

Heavy dependencies (chromadb, sentence-transformers, ragas) are
optional. When absent, the system falls back in order to a NumPy vector
store, a hashing embedding, a heuristic faithfulness mode. Which one was
used is reported in the `backend` field of every result. Don't assume
this is "broken" and try to force-install the heavy packages; this
behavior is deliberate and needed for the internal-network (offline)
scenario.

## Audit trail and versioning

`core/audit.py` keeps a hash-chained, immutable record (who, when, which
code/config version). `core/versioning.py` hashes the test data (corpus,
scenarios, lexicons) and automatically disambiguates "is a score
difference from a data change or a model change." Don't remove or
simplify these two — they satisfy the "auditability" requirement the
supervisor specifically asked for.

## Test suite

172 tests pass, 2 are deliberate `xfail` (the lexicon-based content
filter's context-insensitivity — a known and documented limitation, not
something to "fix"). Test categories: functional (the HateCheck method,
including contrast cases), robustness (ReDoS, log injection, resource
consumption), validators (Luhn, IBAN, TCKN, math equivalence), the
registry system, the audit trail.

```bash
pytest tests/ -q
```

## Frequently used commands

```bash
python -m rag.ingest --seed --reset
python -m scripts.generate_llm_outputs
python benchmark_engine.py                    # the main run
python benchmark_engine.py --preset owasp_rank  # a different weight preset
python -m core.audit verify                   # audit trail integrity
python -m core.versioning drift --model gpt4  # drift report
streamlit run app.py                          # dashboard (4 tabs)
```

## `config/lexicons/` does not yet contain real data

The content-safety lexicons (`profanity.txt`, `religious_insult.txt`,
etc.) are filled with a handful of sample terms for testing the
mechanism, not a real organizational list. When you see a result like
`content_safety=100`, remember it can largely mean "not yet scanned"
rather than "clean" — the `inactive_categories` field in the result
shows which category is empty.

`religious_insult.txt` is especially sensitive: the line between
criticism of religion and an insult is not a technical decision, the
organization must define it. Don't fill this file in on your own
initiative.

## Documents (read when you need detail)

- `README.md` — setup, usage, a summary of the enterprise-maturity layer
- `docs/METHODOLOGY.md` — the source/clause reference for every parameter
- `docs/DEFENSE.md` — the problem/solution/source mapping for every design decision
- `docs/GOVERNANCE_ALIGNMENT.md` — the NIST/ISO/EU AI Act mapping and why in this order
- `docs/WORKING_PAPER.md` — a single-file summary for presentations
