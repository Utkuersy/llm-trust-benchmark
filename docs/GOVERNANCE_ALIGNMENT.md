# Alignment with governance frameworks

This document maps what the platform measures onto international AI
governance frameworks. The goal is **not to claim a legal obligation** —
it is to show that the organization has voluntarily aligned itself with
recognized good-governance standards. This distinction is preserved
throughout the document.

---

## 0. A correction up front — why "EU AI Act compliance" is the wrong framing

In the first draft, this document was framed as "compliance with the EU
AI Act." That framing was abandoned for two reasons:

**Article 2(3) — scope exclusion.** Article 2, paragraph 3 of the EU AI
Act **excludes** AI systems used exclusively for military, defense, or
national security purposes — not moving them into the "high-risk"
category, but taking them outside the regulation's scope entirely.
Recital 24 notes that a system re-enters scope if it is also used for
civilian or dual-use purposes; but the exemption for purely defense use
is clear.

> Source: Regulation (EU) 2024/1689, Article 2(3); Recital 24.
> "AI systems ... used exclusively for military, defence or national
> security purposes ... this Regulation does not apply."

**Jurisdiction.** The AI Act does not bind systems that are not placed
on the EU market or whose output is not used in the EU (Article 2). If
the company is established outside the EU and the system runs only on
an internal network for internal users, the AI Act's *direct*
applicability is open to debate.

**Conclusion.** The sentence "this system falls into the AI Act's
high-risk category, so we must comply with Article 15" can easily be
refuted in front of an AI Governance team. The correct and defensible
sentence instead is:

> **"Although not a legal obligation, we voluntarily apply the
> practices of internationally recognized governance frameworks (NIST AI
> RMF, ISO/IEC 42001); we also use the EU AI Act's technical
> requirements — even though non-binding — as a good design reference.
> If there is ever a product/service open to the EU market, this mapping
> becomes a ready starting point."**

This second positioning is stronger, because (a) it is a verifiable
claim, (b) voluntary compliance is itself a sign of maturity, (c) if the
legal claim is refuted, it does not shake the credibility of the rest of
the project.

**The final call belongs to legal/compliance.** This document is not a
legal opinion; whether a specific product/service of the company falls
under the AI Act requires qualified legal counsel.

---

## 1. Primary anchor: NIST AI Risk Management Framework 1.0 (2023)

A voluntary, US-originated but geography-independent, four-function
framework: **Govern, Map, Measure, Manage.**

| NIST function | Its counterpart in the platform |
|---|---|
| **Govern** — is risk-management culture and process institutionalized | Versioning of weight presets (`SCORING_VERSION`), the methodology document, the audit trail (see §4) |
| **Map** — are context and risk sources identified | Mapping the seven dimensions onto OWASP LLM Top 10 and ISO 25010 (`docs/METHODOLOGY.md`) |
| **Measure** — are risks measured, is the measurement reliable | Trust Score computation, a 135-test validation suite, the human calibration scaffolding (see §5) |
| **Manage** — are risks prioritized and mitigated | Pipeline tracing (at which stage a risk arose), surfacing the riskiest findings on the dashboard |

**Evidence location.** `docs/METHODOLOGY.md`, `docs/DEFENSE.md`,
`core/scoring.py` (the SCORING_VERSION constant), `core/trace.py`.

---

## 2. Primary anchor: ISO/IEC 42001:2023 (AI Management System)

A voluntary, international, ISO 9001-style management system standard —
it answers the question "how do you responsibly manage AI" at the
organizational-process level.

| ISO 42001 clause | Its counterpart in the platform |
|---|---|
| 6.1 — risk and opportunity assessment | The seven-dimension risk taxonomy |
| 8.1 — operational planning and control | Central configuration via `config/settings.yaml`, no hardcoded thresholds |
| 9.1 — monitoring, measurement, analysis, evaluation | Trust Score + pipeline tracing + the drift report (see §4) |
| 9.2 — internal audit | The audit trail table (see §4), the 135-test validation suite |
| 10.1 — continual improvement | The test suite catching and fixing four real bugs; a versioned scoring scheme |

---

## 3. Secondary and conditional reference: EU AI Act

It becomes **directly binding** only if one of the following conditions
holds; otherwise it is used as a **voluntary good-design reference**:

- The system is placed on the EU market or its output is used in the EU
  (Article 2), **and**
- The system is not used exclusively for military/defense/national
  security purposes, or it also has a civilian/dual-use application
  (falls outside the Article 2(3) exemption).

Even if these conditions hold, whether the system falls into one of the
high-risk categories in Annex III must be separately assessed — an
internal RAG assistant of the kind covered by this platform is not
automatically "high-risk."

**Conditionally relevant articles and their platform counterpart:**

| Article | Topic | Its counterpart in the platform |
|---|---|---|
| Article 9 | Risk management system | The seven-dimension risk taxonomy, weighted scoring |
| Article 12 | Record-keeping (automatic logs) | JSON structured logging, the audit trail table |
| Article 13 | Transparency | Reporting the `backend` field on every result (whether heuristic or RAGAS was used, etc.), METHODOLOGY.md |
| Article 14 | Human oversight | The human calibration scaffolding (see §5); surfacing findings on the dashboard for human review |
| **Article 15** | **Accuracy, robustness, cybersecurity** | **A direct three-part mapping below** |

### Article 15's three components and the platform's counterpart

Article 15(1): *"...appropriate level of accuracy, robustness, and
cybersecurity, and that they perform consistently ... throughout their
lifecycle."*

| Article 15 component | Platform dimension |
|---|---|
| **Accuracy** | `generation` (faithfulness/hallucination), `math` (math accuracy), `retrieval` (finding the right source) |
| **Robustness** | `poisoning` (data poisoning resistance), the robustness test suite (ReDoS, resource consumption, adversarial input) |
| **Cybersecurity** | `injection` (prompt injection resistance — the "unauthorized third-party interference" Article 15(5) specifically mentions) |
| **Consistency throughout lifecycle** | The drift report (see §4) — tracks score changes when the model or configuration changes |

This table ties Article 15's abstract requirements to concrete, measured
metrics. It is presented not as "we are legally subject to this," but as
**"we already measure the three properties Article 15 describes,
binding or not."**

---

## 4. Audit trail and reproducibility

See `core/audit.py` and `core/versioning.py`. Every run is recorded
immutably (via a hash chain) with the following information:

- Who triggered it (the OS user, hostname)
- When (a UTC timestamp)
- Which code version (the git commit hash, if available)
- Which configuration (the hash of settings.yaml)
- Which test data version (the combined hash of the corpus + scenario + lexicon files)

This is the counterpart to the NIST RMF's Govern function and ISO
42001's clause 9.2: it provides evidence for the question "how did you
arrive at this result."

---

## 5. Human calibration — an honesty note

`core/calibration.py` is a **calibration scaffold** — sampling, template
generation, and correlation-computation tools are ready. However, **no
real human labeling study has been carried out in this session**,
because no real human evaluators are available in this environment.

This must be stated plainly: *"The calibration infrastructure is ready
and tested; the actual study must be carried out by the organization
with human evaluators."* Producing a fake correlation number and
presenting it as real is the exact opposite of the honesty principle
this document advocates, and if discovered, it would destroy the
credibility of the entire project.

---

## 6. Sources

1. Regulation (EU) 2024/1689 (EU AI Act) — Article 2(3), Article 9, Article
   12, Article 13, Article 14, Article 15, Recital 24.
2. NIST AI Risk Management Framework (AI RMF 1.0), January 2023.
3. ISO/IEC 42001:2023 — Information technology — Artificial intelligence —
   Management system.
4. OWASP Top 10 for LLM Applications 2025 (v2.0).
5. ISO/IEC 25010:2023 — Product quality model.

> **Warning.** This document is not a legal opinion. To what extent a
> specific product/service of the company is subject to which
> frameworks requires qualified legal and compliance counsel.
