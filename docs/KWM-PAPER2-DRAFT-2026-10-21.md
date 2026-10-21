# Diagnosing a Failed Confirmatory Pilot for Paired Validity Manipulations of Agent Failure Memory

**Tech report draft v0.4 (2026-10-21; supersedes v0.3 after third external review)**

> **Status**: DRAFT — internal diagnostic record, not submission-ready.
> **What this document is**: the record of a pre-registered confirmatory pilot that
> failed to produce its pre-registered effect, the diagnostic exploration that
> followed, and a retroactive outcome-layer (L3) scoring of **all 43 runs**
> (10 surviving JSONs + 33 reconstructed from retained logs).
> **What this document is not**: it does not establish that "the memory-value window
> is empty" (Section 7 states this as a *hypothesis with a concrete test design*),
> does not claim a structural defect of any research program, does not claim
> primacy for the instrument, and **does not claim that the pre-registered primary
> outcome (H-A) was fully tested** — the pilot stopped at a pre-registered
> *behavioral* stopping gate (Section 3.4 separates the two).
> **Evidence binding**: every numeric claim maps to a source in Appendix A. Claims
> from adaptive post-hoc harness variants are labeled [exploratory]; per-cell n=1
> throughout; no effect estimates are asserted anywhere. Two ladder rows that have
> no retained artifacts are marked *unverifiable* (Section 6.2).
> **Correction log**: v0.1 mischaracterized the strong model's behavior; v0.2
> rebuilt the matrix and added retroactive L3 scoring for the 10 surviving runs.
> v0.3 corrects **three further factual errors found by a full
> evidence–claim reconciliation against the retained logs** (C1–C3 below; audit in
> `docs/KWM-EVIDENCE-AUDIT-2026-10-21.md`): the confirmatory pilot's behavior
> description, the Gemma multi-attempt outcome, and the Gemma rule-variant outcome.
> v0.4 (this version) adds a machine-generated **unique run manifest** (43 rows;
> disjoint JSON/log-only evidence pools verified; Appendix A) and bounds every
> inventory-wide claim to the 43 traceable runs; capability-tier labels are replaced
> by model names; all claim strengths are aligned with n=1 evidence.

---

## Abstract

Agent memory systems increasingly store *failure experiences* so that downstream
agents avoid repeating costly mistakes. Recent work shows that attaching explicit
*applicability conditions* to memories changes behavior. A complementary question
remained untested: when the environment changes and a previously dead path becomes
viable again, what information helps an agent correctly revise an old lesson — rather
than mechanically obeying it (wrong abandonment) or forgetting it (repeating the
failure)? We formalized this as a paired validity manipulation (V-dead vs. V-revived)
crossed with four memory representations (NONE / UNCOND / COND / COND-MINUS), built a
frozen instrument — three data-analysis task families with verified gold standards,
contamination-checked fact-matched memory templates, trap tools whose frozen
implementations invert task-success across validity variants, and a method-menu
harness that makes method choice an explicit logged decision — and ran the
pre-registered confirmatory pilot on the family with the cleanest statistical
contrast.

The pilot triggered its pre-registered behavioral stopping gate: among the 43 traceable runs of the
reconciled inventory, the directional contrast the design hinges on (UNCOND vs.
NONE, expected to reverse across validity variants) was never observed — two early
ladder variants without retained artifacts are excluded from this statement and
marked unverifiable (§6.2). A subsequent adaptive diagnostic across harness
variants and three models (Qwen2.5-7B, Gemma3-12B, DeepSeek-V4.1-Flash) —
reported here as *exploratory*, not confirmatory evidence — never produced that
contrast either. Where method choice *did* move, the observed movement was
identical between COND and COND-MINUS in the relevant cells; these single-run
observations do not identify an effect attributable to applicability information —
they neither establish such an effect nor rule one out. Under the
single-commitment budget no run could, by construction, switch after a failed
attempt; under multi-attempt budgets runs could and sometimes did. Retroactive
scoring of all 43 runs against the frozen outcome checkpoints shows
DeepSeek-V4.1-Flash deterministically passes V-dead and fails V-revived (it
commits its single attempt to a tool that crashes under revival), while
Qwen2.5-7B's unreflective default is exactly correct under revival and biased under
V-dead. We additionally document a treatment confound: memory conditions received a
memory-usage instruction that NONE did not, so all memory-vs-NONE comparisons
contrast *memory + instruction* against *no memory*, not memory content alone
(Section 4.4).

We contribute: (1) a reusable, gate-verified instrument for paired validity
manipulations of failure memory (released with all traces); (2) a documented
negative confirmatory pilot with a complete, reconciled diagnostic trail; and
(3) a precisely stated hypothesis — the *memory-value window*: a memory can only add
decision value when the agent neither already knows the lesson nor lacks the
capability to act on it — together with a proposed confirmatory protocol (draft,
to be pre-registered before any run: task-private information, uniform base
prompts, judge and scoring-field tests built in advance, and attempt budget as a
pre-registered factor). We explicitly do not claim the window is empty in general; we report
that no tested configuration occupied it, and that the present pilot cannot test
whether any could.

---

## 1. Introduction

Long-horizon knowledge-work agents accumulate memory across sessions. A growing
literature stores *negative knowledge*: failed paths with reasons and evidence
[2606.21024], boundary fields on skills [2608.22339], failure reflections [MCMA],
and pre-execution guardrails distilled from failed computations [2608.11224]. The
demonstrated value is avoidance: do not redo what failed.

The reverse question — pre-registered in our design document before any Stage-A data
were collected — asks what happens when the world changes under the memory:

> **RQ (frozen v2.4)**: *When a previously failed path becomes viable again in a new
> environment, what information helps an agent correctly revise the old lesson —
> rather than mechanically obeying it (wrong abandonment) or forgetting it entirely
> (repeating the failure)?*

Answering it requires manipulating the *current validity* of one and the same failure
memory. In our audit of eight direct competitors (Section 2), none constructs this
paired manipulation; the nearest neighbor, BASM [2608.22339], conditions skill
memories on applicability but encodes only the "should suppress" cell — it never
builds the "should retry" cell a revived path requires. (This is a description of the
audited corpus as of 2026-10-21, not a primacy claim; the audit's screening criterion
and its boundary against diagnostic benchmarks such as MemFail [2605.26667] are
stated in Section 2. See also Section 8.)

We designed a 2×4 experiment (validity variant {V-dead, V-revived} × memory
representation {NONE, UNCOND, COND, COND-MINUS}) with pre-registered directional
hypotheses, built the complete instrument, and ran the confirmatory pilot. It failed
in an instructive way, and this report is organized around that failure:

- **Section 4–5** document the instrument and the pilot protocol.
- **Section 6** rebuilds the result matrix from primary artifacts (run JSONs and
  retained logs — all 43 runs inventoried, 43 scored or explicitly marked
  unjudgeable/unverifiable), and separates the confirmatory pilot from the
  exploratory diagnostics.
- **Section 7** extracts what the failure *does* and *does not* license us to say,
  and states the hypothesis-level takeaway (the memory-value window) with the design
  that would test it.

**Contributions.**

1. **An instrument** (engineering contribution, verifiable from the repository and
   its gate records): three task families with frozen gold standards and trap tools;
   nine fact-matched memory templates under five-part contamination control; a
   method-menu harness turning method choice into an explicit behavioral metric; and
   gate checks V1–V6.
2. **A documented negative confirmatory pilot with a reconciled evidence trail**:
   the pre-registered directional contrast (UNCOND vs. NONE) was not observed in
   any of the 43 traceable runs (uniqueness machine-verified; disjoint evidence
   pools); the full inventory is scored under frozen checkpoints or explicitly
   marked unjudgeable; the gaps between earlier draft narratives and the primary
   artifacts are documented as corrections rather than silently patched.
3. **A hypothesis with a test design**: the *memory-value window* — memory adds
   decision value only where (i) the lesson is absent from model weights and (ii) the
   agent can act on the memory — plus a proposed confirmatory protocol (draft,
   pending pre-registration) targeting task-private information domains with
   uniform prompts and two dedicated validity checks, one for each conjunct of the
   hypothesis.

---

## 2. Related Work

We audited eight direct competitors by reading their experimental sections in full
(2026-10-21; four dimensions: memory object D1, paired validity manipulation D2,
cost-side measurement D3, representation intervention D4). **Screening criterion**:
a paper enters the audited set if it (a) stores task experience (success or failure)
for reuse by LLM agents and (b) reports a controlled behavioral measurement of memory
content or structure. Design/engineering papers without controlled measurement
(e.g., [2607.24759]) and pure diagnostic benchmarks without a representation
intervention (e.g., MemFail [2605.26667]) are treated as background or adjacent
work, not competitors. Table 1 summarizes the audit; it is a corpus description, not
a completeness claim — the supportable statement is "no audited work constructs the
paired validity manipulation", not any field-wide claim.

**Table 1 — Eight audited competitors (D2 paired validity manipulation: absent in all).**

| # | Work | D1 memory object | D2 paired validity | D3 cost side | D4 representation |
|---|---|---|---|---|---|
| 1 | Negative Knowledge [2606.21024] | synthesized typed failure records; adopt/reject | none (one observed negative-transfer case) | none (token cost only) | none (296/795/1109-token forms confounded) |
| 2 | Experience-Following [2505.16067, ACL 2026] | query–execution pairs (success/failure not separated) | none (distribution shift via task reordering) | indirect only | none (intervention at storage admission) |
| 3 | MeClear [2609.09115] | stale/misleading retrieved records | none | over-selection of harmless records ≠ wrong abandonment | none (query-level suppression) |
| 4 | Remember When It Matters [2607.08716] | status/knowledge/procedural | none | indirect (Injection-only −6.0pp, unconceptualized) | none (injection timing) |
| 5 | Useful Memories Become Faulty [2605.12978] | episodic traces + consolidated abstractions | none (erosion curves observed) | utility below baseline ≠ wrong abandonment | partial (abstraction level as control) |
| 6 | BASM [2608.22339, EMNLP 2026 Findings] | success skills (failure trajectories only as boundary evidence) | none ("inapplicable states" constructed across tasks) | none systematic; App.-Skill dip unmeasured as cost | full (4-level + equal-length padding + equal-token pairing + attention knockout) |
| 7 | MCMA [2026.findings-acl.1535, ACL 2026 Findings] | tree-structured knowledge; Sum + failure-Reflection copilots | none (seen/unseen splits) | none (knowledge-count non-monotonicity) | partial (structure type / abstraction level) |
| 8 | Materials-science partner [2608.11224] | facts incl. boundary knowledge; skills with conditions/guardrails | none (within-family generalization) | partial (token cost; harmful cross-model transplant) | none (conditioning is format default) |

**Differentiation, stated narrowly.** BASM's skill memories *do* include
applicability-condition, risk, avoidance and recovery fields — we do not claim
applicability conditioning is unstudied. The narrow gap we target is: **the same
failed path, under paired controlled environmental change, flipping from
should-avoid to should-retry** — BASM's boundary fields come from failure
trajectories but failures are never stored as entries, its T_wrong cell encodes
"should suppress" only, and no audited work measures wrong abandonment of a revived
path (the "should retry" cell) or its cost. MemFail [2605.26667] is the nearest
diagnostic benchmark: it stress-tests failure modes of memory systems
(summarization/storage/retrieval) but performs no representation intervention and no
paired validity manipulation; it belongs in the adjacent set and would share our
released diagnostic tooling.

**The seam.** The conjunction — failure experience *itself* as memory object (D1) +
within-family paired validity manipulation (D2) + benefit–cost dual measurement (D3) —
is occupied by none of the eight audited works. BASM is the strongest neighbor on D4:
its boundary fields are a causally validated anti-imitation channel (same-prompt
knockout recovers 69.4% of the imitation-trap effect; wrong-tool margin 7.31 → 10.71
→ 2.26 across memory-free / procedure-skill / BASM), and it reports AppWorld +23.8pp,
BFCL +5.0pp, AgentDojo attack-success −4.6pp. Its intermediate configuration — adding
*only* the applicability-condition field to procedure skills ("App. Skill") — is
non-monotonic and model-dependent (AppWorld-Qwen3-32B 66.07→60.12, −5.95pp;
BFCL-Qwen3.5-397B 53.38→51.63, −1.75pp; AppWorld-Qwen3-14B 46.43→53.57, +7.14pp),
a phenomenon its authors neither measure as a cost nor explain. Our diagnostic
results are consistent with this pattern (Section 7.1) but do not explain it; that
would require the confirmatory experiment this pilot failed to enable.

Adjacent signals that conditioning information can *hurt*: MCMA disables its
success-summary copilot on ScienceWorld; [2607.08716] reports Injection-only at
−6.0pp on one environment; [2608.11224] reports memory transplantation that can be
neutral-to-harmful when source memory is weaker than the target's knowledge. None
treats these as the phenomenon under study. Motivation-level work [2607.24759]
argues for retaining dead paths in append-only wikis (design + case study, no
controlled measurement).

---

## 3. Problem Formalization

### 3.1 Design

**2×4 within-task-family design** (pre-registered, frozen v2.3/v2.4 before Stage-A
data):

- **Validity variant (within family, paired)**:
  - *V-dead*: environment unchanged; the old failed path X is still broken → avoiding
    X is correct; the failure memory should *help*.
  - *V-revived*: the disqualifying condition is removed → X is now correct; avoiding
    it is an opportunity loss; the failure memory should *harm* unless conditioned.
  - Construction requirement (pre-registered): X passes checkpoints in V-revived and
    genuinely fails in V-dead; variants match item-by-item except the manipulated
    condition.
- **Memory representation (fact-matched templates)**: NONE (no memory) / UNCOND
  (full failure facts: path + cause + evidence + consequence; no scope statement) /
  COND (same facts + explicit applicability condition and invalidating change) /
  COND-MINUS (equal length, same facts, applicability replaced by non-diagnostic
  content).
- **Pre-registered hypotheses**: **H-A** (primary, on task success rate per the
  frozen weighted checkpoints, Section 3.3, with directional expectations per
  variant: UNCOND vs NONE ↑ in V-dead, ↓ in V-revived); **H-B** (COND vs UNCOND
  mitigates damage in V-revived, non-inferior in V-dead); **H-C** (COND vs
  COND-MINUS isolates conditional information beyond length/genericity).

### 3.2 Behavioral metrics

Three layers, all logged per run: **L1 usage** (memory injected / retrieved /
explicitly cited), **L2 behavior** (method-choice; as of the reconciliation audit,
logged as two separate fields — `methods_tried` (exploration sequence) and
`method_adopted` (the method whose output the FINAL answer commits to) — so that
"explored X" is never conflated with "adopted X"), **L3 outcome** (frozen
deterministic checkpoints vs. gold standards, Section 3.3).

### 3.3 Frozen outcome checkpoints (family A; task-set doc §3.6, quoted verbatim)

| ID | weight | criterion | tolerance (frozen) |
|---|---|---|---|
| C-A1 | 0.4 | Δ point estimate vs. gold 2.1 | ≤ 0.3 (absolute) |
| C-A2 | 0.2 | CI width in [0.9, 3.5] (absent CI → 0) | pre-registered band |
| C-A3 | 0.2 | semantic consistency of the conclusion (judge prompt §7) | **judge not built → unjudgeable** |
| C-A4 | 0.2 | tool calls ≤ 10 | visible in task description |

Composite = 0.4·s1 + 0.2·s2 + 0.2·s3 + 0.2·s4; pass if ≥ 0.6. Because C-A3 was never
built (the stop-loss preceded it), we report the deterministic subtotal and the
composite *bounds* [subtotal, subtotal + 0.2], with verdicts `pass_guaranteed`
(lower bound ≥ 0.6), `fail_guaranteed` (upper bound < 0.6), and `unjudgeable`
(otherwise — C-A3 is decisive). **Terminology.** `pass_guaranteed` means
*frozen-score pass* — the frozen weights and threshold guarantee a pass under the
recorded deterministic components; it is **not** a claim of reliable task
completion, and the two notions are kept apart throughout this report. Families B/C checkpoints are analogously frozen
(B: total = 500,000 exact, month-row count exact; C: total = 50,000 exact, output
customer count = *that variant's* frozen unique-customer gold — 991 in v1, 992 in
v2 — so a per-variant-correct output is not a cross-variant mismatch).

**Known metric-validity defect (recorded, not patched).** Under these frozen weights,
C-A1 + C-A4 alone reach the 0.6 threshold: a run with a passing point estimate and
≤10 tool calls is `pass_guaranteed` even with no CI and no semantic check. This makes
"task success" weaker than the reader's intuitive notion of a reliable result. Per our freeze discipline the rule
is not edited post hoc; the next protocol replaces it with a *necessary-condition*
design (point estimate within tolerance as a hard requirement; CI, semantic
consistency, and cost as weighted additions) and additionally requires automated
scoring-field tests proving that every permitted method emits every field the
scoring protocol consumes — or a pre-registered interpretation of absent fields —
before any confirmatory run (see the protocol v2 document).

### 3.4 Primary endpoint vs. stopping gate (separation added in v0.3)

Two pre-registered quantities are often conflated; this report keeps them apart:

- **Primary endpoint (H-A)**: task success rate under the frozen checkpoints —
  the quantity the hypotheses are about.
- **G-core stopping gate**: a *behavioral viability* gate defined by the pilot's own
  protocol — if the behavioral layer (method choice) shows no movement in the
  pre-registered contrast, the pilot's continuation is not justified by its
  pre-registered logic and it stops. This is a **stopping decision about this
  pilot**, not a demonstration that other task families, models, or attempt budgets
  could not measure an outcome-layer memory effect.

The pilot was stopped by G-core. Consequently **H-A was not fully tested**: the
outcome layer was scored where artifacts permit (Section 6.4), but the pilot's stop
means the confirmatory evidence about H-A is limited to one family, n=1 cells, and a
harness configuration whose behavioral layer had already failed its viability gate.
All outcome-layer statements below are therefore *descriptive of the surviving
artifacts*, not confirmatory tests of H-A.

---

## 4. Instrument

All artifacts are frozen in the repository (paths in Appendix A).

### 4.1 Task families

Three data-analysis families, each with a naive method X (the NONE-condition natural
default), a memory-suggested alternative Y (a project trap tool whose frozen
implementation is task-passing in V-dead and *fails* in V-revived — by assertion
crash in families A/B, by **silent data loss** in family C), and gold standards:

| Family | Failure mode of X | X (naive) | Gold | X in V-dead | X in V-revived | Y (trap tool) in V-dead | Y in V-revived |
|---|---|---|---|---|---|---|---|
| A — missing data | MAR by (group, z2); deletion biases the group contrast | `dropna` + group means | Δ = 2.1 | Δ = 8.105 (C-A1 fail: dev +6.005) | Δ = 1.882 (C-A1 pass: dev −0.218 ≤ 0.3) | mi_tool: 2.031 ✓ | assertion crash |
| B — timestamp parsing | 12% unparseable timestamps silently dropped | month `groupby` on coerced ts | total = 500,000 | 440,000 (fail) | 500,000 ✓ | legacy_tool: 500,000 ✓ | column-position crash |
| C — SQL join fanout | duplicated customer rows fan out JOIN; Y merges distinct customers by name case | direct JOIN aggregated by id | (50,000; 991 in v1 / 992 in v2) | (59,280; 991) | (50,000; 992) ✓ | dedup_by_name: (50,000; 991 ✓) | (50,000; **904**) — silent customer loss |

Notes on the numbers: (i) the frozen C-A1 tolerance is *absolute* (±0.3), so the MI
tool's finite-sample deviation (2.0306, dev −0.069) passes and the naive estimator in
V-revived (1.882, dev −0.218) also passes — statistical unbiasedness of an estimator
and passing a single-realization checkpoint are distinguished exactly, and the
tolerance table was frozen before any agent run; (ii) family C's V-dead Y output
(50,000; 991) is checked against v1's own frozen gold (991), not v2's (992) — the
checkpoint table freezes per-variant golds.

### 4.2 Frozen memory templates and contamination control

Nine record templates (3 families × UNCOND/COND/COND-MINUS), fact-matched per family;
COND-MINUS equalized in length with neutral archival filler. A five-part
contamination check verifies: no gold *absolute* value in any template; COND-MINUS–
COND embedding distance below a 0.5 threshold, computed as cosine *distance*
(1 − cosine similarity) over bge-m3 (1024-d) embeddings; measured 0.487. This is a
leakage *heuristic*: it shows the replacement filler is not a near-duplicate of the
COND text, but an embedding distance alone cannot prove the absence of applicability
information — it is therefore one signal alongside the template-consistency audit,
and a blind human review of all nine templates is added as a protocol-v2
requirement before any further run. The check also verifies: length
parity within 10%; template consistency; no leak words in task descriptions.
All checks green before any agent run.

### 4.3 Memory layer and retrieval

A purpose-built memory layer (dense retrieval over frozen records with bge-m3,
top-k=3, hit threshold 0.30) injects the family-appropriate template at session
start; injection and retrieval events are logged (L1: `memory_injected`,
`memory_hit_condition` fields in every run JSON; NONE runs show `False`/`None`).

### 4.4 Agent harness evolution (v1 → v3) — and a treatment confound

- **v1 free-code ReAct**: prompt-based ACTION/OBSERVATION loop with sandboxed
  tools. Guards: `tool_call_id` on tool-result messages; FINAL rejected if no
  compute tool was called or placeholders present (≤2 nudges). Fixed a parser bug
  where models narrate an entire ReAct trace as text and a FINAL-priority parser
  never executes it.
- **v3 method menu**: the agent only emits `METHOD: <name>` (frozen implementation
  executes; OBSERVATION returned) or `FINAL:`; budget = 1 method execution (single
  commitment). `methods_tried`/`method_adopted` are the L2 metrics. The menu freezes
  method implementations, removing code-quality confounds across conditions.

**Treatment confound (disclosed; not removable in this data).** In all harness
variants, the memory-usage instruction ("judge whether the injected experience
applies to the current data before committing") was rendered **only when memory
exists** — NONE runs never saw it. Therefore every memory-condition vs. NONE
comparison in this report contrasts **memory content + usage instruction** against
**no memory and no instruction**, not memory content alone. Observed
memory-vs-NONE differences (or their absence) cannot be decomposed into the two
components; e.g., a null difference is compatible with the instruction boosting
caution while the memory content pushes the other way. We did not intend this as a
manipulated factor; it is a design flaw w.r.t. the stated question. The next-round
protocol renders a structurally identical prompt in every condition (NONE receives
an explicitly empty memory slot plus the same usage rule), making the memory payload
the only difference — see the protocol v2 document.

### 4.5 Pre-run gates

| Gate | Content | Status |
|---|---|---|
| V1 data reproduction | simulators reproduce designed golds | pass |
| V2 revival | deletion-based methods unbiased in V-revived | pass |
| V3 trap tools | Y passes in V-dead, fails in V-revived (A/B: assertion crash; C: silent customer loss) | pass |
| V4 templates | contamination checks all green | pass |
| V5 family-B exploration | v1/v2 asymmetries verified end-to-end | pass |
| V6 natural default | NONE-condition agents first-choose X | pass (qwen2.5:7b, 2/2; Δ=8.105) |
| V7 UNCOND implied-condition pretest | not reached (stop-loss preceded it) | — |

---

## 5. Stage-A Procedure and Provenance

All runs executed locally (Ollama) or via an Anthropic-protocol cloud endpoint
(DeepSeek-V4.1-Flash). Every run writes a JSON with condition tags, method-choice
fields, tool calls/results, and the final answer.

**Provenance defect (recorded, not silently patched):** the runner did not write the
model identifier into the run JSON, and later harness variants reused the same
result tags, *overwriting* earlier JSONs. Attribution and reconstruction therefore
rely on retained execution logs (committed under `results/kwm_stage_a/logs/`):
`kwm_cloud.log` (8 runs, DeepSeek-V4.1-Flash) matches the eight `*_r0` JSONs;
`kwm_stab.log` (4 runs, Qwen2.5-7B) covers the two surviving `*_r1` JSONs and two
overwritten `r0`s. The complete 43-run inventory, with per-run log citations, is in
the evidence audit (`docs/KWM-EVIDENCE-AUDIT-2026-10-21.md`); 33 log-only runs are
scored by `experiments/kwm_score_logonly.py` →
`results/kwm_stage_a/l3_scoring_logonly.json`. For future runs the runner must write
a `model` field, per-variant tag namespacing, and full OBSERVATION payloads (not
retroactively applied — no history was rewritten).

---

## 6. Results

### 6.1 Confirmatory pilot (family A, 2×4, menu + multi-attempt harness, Qwen2.5-7B) — **G-core failed** (rebuilt in v0.3)

Source: retained log `kwm_pilot24.log` (8 runs). v0.2 described this pilot as "all
eight cells chose the naive method X"; **that description is wrong** and is
corrected here from the log.

**L2 — exploration vs. adoption (per cell, n=1):**

| Cell | methods_tried | method_adopted | Adopted estimate |
|---|---|---|---|
| NONE / V-dead | X | **X** (naive_delete) | 8.1051 |
| NONE / V-revived | X | **X** | 1.8818 |
| UNCOND / V-dead | X | **X** | 8.1051 |
| UNCOND / V-revived | X | **X** | 1.88 |
| COND / V-dead | X → impute → mi_tool | **mi_tool** | 2.0306 |
| COND / V-revived | X → impute | **impute_group_mean** | 1.8818 |
| COND-MINUS / V-dead | X → impute → mi_tool | **mi_tool** | 2.0306 |
| COND-MINUS / V-revived | X → impute | **impute_group_mean** | 1.88 |

**L3 — frozen-checkpoint scores (log-only scoring; full table in §6.4/audit):**
V-dead: NONE/UNCOND `unjudgeable` (0.4), COND/COND-MINUS `pass_guaranteed` (0.6);
V-revived: all four cells `pass_guaranteed` (0.8).

**What this does and does not show.**

1. The **pre-registered contrast (UNCOND vs. NONE) never moved**: the two conditions
   are behaviorally identical in all four cells. Per the pre-registered G-core rule
   the pilot stops. *The verdict stands; v0.2's stated reason ("the behavioral
   metric does not move") does not.*
2. Movement *did* occur — in COND and COND-MINUS cells, **identically**. Because
   COND-MINUS carries non-diagnostic content, the observed movement is identical
   between the two conditions in the relevant cells; these single-run observations
   do not identify an effect attributable to the applicability information — they
   neither establish such an effect nor rule one out — and the COND/COND-MINUS
   vs. UNCOND difference is equally consistent with run-to-run stochasticity.
   No directional claim is made.
3. The L3 layering in V-dead (memory-present cells 0.6 vs. NONE/UNCOND 0.4) is
   directionally consistent with "memory presence helped", but the same
   COND-MINUS≡COND collapse rules out content attribution, the UNCOND anomaly is
   unexplained, and n=1 forbids any effect statement. Recorded as observation only.
4. In no cell did any run use the COND applicability condition *prospectively*
   (before choosing) to infer the feared path had become safe.

### 6.2 Exploratory diagnostics [exploratory — adaptive variants, n=1 per cell; rebuilt from retained logs in v0.3]

Because the confirmatory pilot stopped, we ran an adaptive ladder of harness variants
to determine whether the failure was instrumental (fixable) or systematic. Each
variant ran only after the previous failed; variants differ in prompts and are not
poolable. v0.3 rebuilds every row from its retained log; **rows without retained
artifacts are marked *unverifiable*** rather than asserted.

| # | Harness variant (log) | Model | Outcome (rebuilt) |
|---|---|---|---|
| 1 | Free-code ReAct (`kwm_pilot/uncond/gem.log`) | Qwen2.5-7B / Gemma3-12B | all three runs land on the deletion estimator (8.105); Gemma reaches it via full-column `pandas.mean` (implicitly skips NaN = same estimator) and reports "I did not use dropna" |
| 2 | Menu + multi-attempt, qwen — *no retained log* | Qwen2.5-7B | **unverifiable** (v0.2 claimed "4/4 choose X"; no supporting artifact survives; possibly conflated with the confirmatory pilot's V-dead half, which measured 2/4) |
| 2′ | Menu + multi-attempt (`kwm_gem_menu.log`) | Gemma3-12B | explores all three methods, **arbitrates back to the biased X (8.1051) in both V-dead cells**; in UNCOND it *reverses* the memory ("historical experience: bias relatively small"). (v0.2's "self-arbitrates to the correct answer" was wrong — corrected) |
| 3 | Menu + multi-attempt + memory-usage rule (`kwm_gem_rule.log`) | Gemma3-12B | **switches to mi_tool in V-dead cells** (NONE, UNCOND, COND; UNCOND/COND explicitly citing the memory) and adopts mi_tool/impute in V-revived cells — *including recovering to X after the mi_tool crash*; NONE/V-dead also adopts mi_tool, so switching is not memory-specific. (v0.2's "8/8 choose X" was wrong — corrected; that description matches row 4's harness) |
| 4 | Decision-point binding (`kwm_bind.log`) | Gemma3-12b | 4/4 adopt X; V-revived finals *fabricate* applicability ("current data has similar missing pattern to train_v1.csv") to justify X |
| 5 | Memory relocated to first user message — *no retained log* | Gemma3-12B | **unverifiable** (v0.2 claimed "no effect") |
| 6 | v3 single commitment (budget=1) + cloud model (`kwm_cloud.log`) | DeepSeek-V4.1-Flash | all 8 cells commit to `mi_tool` — see §6.3; in V-revived this is *not* "choosing the optimal method" (v0.1's wording, corrected in v0.2) |
| — | v3 single commitment (`kwm_v3.log`, `kwm_stab.log`) | Qwen2.5-7B | 8/8 + 2 stability reruns all adopt X, citing budget; V-revived cells score 0.8 pass_guaranteed, V-dead cells 0.4 unjudgeable |

Reading the ladder honestly: **no traceable configuration produced the
pre-registered contrast**; movement, where it occurred (rows 2′/3 vs. the pilot),
was inconsistent across harness variants and not specific to conditional content
(NONE cells move under row 3; COND-MINUS moves identically to COND in the pilot).
Two structural correspondences are observed, not causal findings: the
single-commitment configuration *by design* permits no switch after a failed
attempt, and that is exactly where switching is absent; multi-attempt
configurations permit switching and recovery, and that is where they occur.
Attempt budget is accordingly treated as a design factor the next protocol must
pre-register and manipulate — not as an identified cause of model behavior.

### 6.3 The strong-model matrix (rebuilt from primary artifacts; unchanged from v0.2)

**DeepSeek-V4.1-Flash, harness v3, family A, all 8 cells commit to `mi_tool`
(`A_*_r0.json`; log `kwm_cloud.log`):**

| Variant | Conditions | Method | Tool result | Behavior |
|---|---|---|---|---|
| V-dead | NONE / UNCOND / COND / COND-MINUS | mi_tool | exit 0, Δ = 2.0306, no CI | passes task |
| V-revived | NONE / UNCOND / COND / COND-MINUS | mi_tool | **exit 1, assertion crash** | honest "no estimate available"; never falls back to X (budget = 1) |

Three observations, each verified against the JSONs: (i) the choice is *invariant
across all four memory conditions* — including NONE (`memory_injected: False`) — so
the model's behavior is a **mechanical preference for the robust tool**
(strategy inertia / over-generalization), *not* condition adaptation and *not*
demonstrably "prior knowledge covers the memory's value"; (ii) in V-revived the
chosen tool is **not** the optimal method — the naive method would have passed the
frozen checkpoints (dev −0.218 ≤ 0.3) — so DeepSeek-V4.1-Flash deterministically
*fails* the entire V-revived column; (iii) its failure reports are honest (no
fabricated numbers) but never use the COND applicability condition to *anticipate*
that the feared path had become safe.

**Qwen2.5-7B (surviving `*_r1.json`; log `kwm_stab.log`):** both surviving
stability-rerun cells (UNCOND, COND × V-dead) adopt `naive_delete`, Δ = 8.1051,
CI [7.0432, 9.1670] (N=1440); the UNCOND final answer cites the memory ("历史经验
表明此方法的偏差约为 6.0pp") *while adopting the biased estimate* — the
acknowledge-but-not-act pattern.

### 6.4 Retroactive L3 scoring — all 43 runs

**Scoring coverage.** (a) 10 surviving JSONs: `experiments/kwm_score_retro.py` →
`l3_scoring_retroactive.json`. (b) 33 log-only runs (JSONs overwritten or never
written): `experiments/kwm_score_logonly.py` → `l3_scoring_logonly.json`; every
entry carries its log file and run index. (c) C-A3 unjudgeable everywhere (judge
never built); verdicts are interval-valued. Scoring convention: the adopted method's
output is scored; exploration is L2-only.

**Uniqueness audit (added in v0.4).** A machine-generated manifest
(`experiments/kwm_run_manifest.py` → `results/kwm_stage_a/run_manifest.json`,
43 rows) verifies the three conditions an evidence audit must satisfy: **(V-a)**
exactly 43 unique run IDs; **(V-b)** the 10 JSON-evidenced and 33 log-only runs are
disjoint — the four `kwm_stab.log` runs are split explicitly (runs 2/4 *are* the
surviving r1 JSONs and are counted only in the JSON pool; runs 1/3 are the
overwritten r0s, counted only as log-only); **(V-c)** every grouping number in
Table 6.4b regenerates from the manifest alone. The earlier "8 + 2 stab" phrasing
referred only to the two overwritten r0 stability runs; no run is double-counted.

**Table 6.4a — surviving JSONs (unchanged from v0.2):**

| Run | Model | Adopted | Est. | Dev vs 2.1 | CI width | s1 | s2 | s4 | Det. subtotal | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| NONE / V-dead | deepseek | mi_tool | 2.0306 | −0.069 | — | 1 | 0 | 1 | 0.60 | pass_guaranteed |
| UNCOND / V-dead | deepseek | mi_tool | 2.0306 | −0.069 | — | 1 | 0 | 1 | 0.60 | pass_guaranteed |
| COND / V-dead | deepseek | mi_tool | 2.0306 | −0.069 | — | 1 | 0 | 1 | 0.60 | pass_guaranteed |
| COND-MINUS / V-dead | deepseek | mi_tool | 2.0306 | −0.069 | — | 1 | 0 | 1 | 0.60 | pass_guaranteed |
| NONE / V-revived | deepseek | mi_tool | CRASH | — | — | 0 | 0 | 1 | 0.20 | **fail_guaranteed** |
| UNCOND / V-revived | deepseek | mi_tool | CRASH | — | — | 0 | 0 | 1 | 0.20 | **fail_guaranteed** |
| COND / V-revived | deepseek | mi_tool | CRASH | — | — | 0 | 0 | 1 | 0.20 | **fail_guaranteed** |
| COND-MINUS / V-revived | deepseek | mi_tool | CRASH | — | — | 0 | 0 | 1 | 0.20 | **fail_guaranteed** |
| UNCOND / V-dead (r1) | qwen | naive_delete | 8.1051 | +6.005 | 2.124 | 0 | 1 | 1 | 0.40 | unjudgeable (C-A3 decisive) |
| COND / V-dead (r1) | qwen | naive_delete | 8.1051 | +6.005 | 2.124 | 0 | 1 | 1 | 0.40 | unjudgeable (C-A3 decisive) |

(s2 = 0 for all deepseek V-dead cells because the frozen mi_tool implementation
returns a point estimate only — an instrument gap recorded in Section 8. The qwen
V-dead cells are *unjudgeable*, not "lean failing": they pass iff s3 = 1.0, and no
prior reason sets s3.)

**Table 6.4b — log-only runs, summarized (33 runs; per-run rows in the audit doc and
`l3_scoring_logonly.json`):**

| Harness × model | Cells | pass_guaranteed | fail_guaranteed | unjudgeable |
|---|---|---|---|---|
| Confirmatory pilot (multi-attempt, qwen) | 8 | 6 (2 V-revived X-cells + 4 COND/COND-MINUS cells) | 0 | 2 (NONE, UNCOND × V-dead) |
| v1 free-code (qwen, gemma) | 3 | 0 | 1 (NONE/V-dead, no CI) | 2 |
| Gemma multi-attempt (menu / +rule / binding) | 12 | 10 | 0 | 2 |
| v3 single commitment (qwen: 8 v3-validation + 2 overwritten r0 stability runs) | 10 | 4 (V-revived) | 0 | 6 (V-dead) |

**What the reconciled matrix licenses.** Within the *v3 single-commitment*
configuration (the only one with structured JSONs), success is determined by
model × variant with zero variance from memory representation. Across the full
43-run manifest that statement does *not* generalize: outcome patterns *correspond
with* harness budget structure and model identity — factors that were never crossed
in a full factorial design, so the correspondence is not a causal attribution.
The places where memory conditions diverged (pilot COND/COND-MINUS vs. UNCOND;
Gemma row 3) are n=1, unpredicted, and do not identify a content effect
(COND-MINUS ≡ COND; NONE moves too). Among the 43 traceable runs: (1) the
UNCOND-vs-NONE contrast was never observed; (2) prospective use of the
applicability condition occurred in zero of 43 runs; (3) switching is absent
exactly where the single-commitment design forbids it and present where
multi-attempt budgets allow it — a design-property correspondence, not a finding
about model reasoning; (4) DeepSeek-V4.1-Flash deterministically fails the revived
column by committing to a crashing tool, while Gemma3-12B — under a multi-attempt
budget — recovers to X after the same crash.

### 6.5 Evidence strength

- The confirmatory pilot is one family (A), one model, n=1 per cell. The diagnostics
  are exploratory, adaptive, and non-poolable.
- Run uniqueness is machine-verified (manifest, §6.4): 43 unique runs, disjoint
  JSON/log-only evidence pools, group totals regenerable from the manifest alone.
- Model attribution relies on logs because the runner omitted the model field
  (defect recorded, Section 5); attribution is unambiguous for the surviving JSONs,
  but we mark it as reconstructed.
- Two ladder claims from v0.2 (rows 2 and 5) have no retained artifacts and are
  marked unverifiable rather than retained as findings.
- The scoring package for future runs remains unbuilt except for the retroactive
  family-A scorers; families B/C have frozen checkpoints but no agent runs, hence no
  scores ("not applicable" rather than "failed").

---

## 7. Discussion

### 7.1 What this pilot does and does not establish

**Does not establish**: that failure memory cannot help agents; that the
memory-value window is empty in general; that the manipulation is structurally
infeasible; that any model "already knows" the lesson; **or that the pre-registered
primary outcome H-A has been tested to a negative conclusion** — the pilot stopped
at the behavioral gate, so H-A remains untested at confirmatory strength (§3.4).

**Does support (as observations, within scope)**:

1. The pre-registered contrast (UNCOND vs. NONE) was not observed in any of the 43
   traceable runs across all harness variants and all three models (two early variants
   without retained artifacts are excluded from this statement and marked
   unverifiable). Whatever would mediate the designed memory→behavior transfer, no
   traceable configuration engaged it.
2. In the retained exploratory runs there is insufficient evidence to attribute
   any method-choice movement to memory *content*: COND-MINUS behaved identically
   to COND wherever both moved, and NONE cells moved under at least one
   configuration. Observed behavior *corresponds with* budget structure and model
   identity — factors never crossed factorially, so this is a correspondence, not
   a causal attribution.
3. Prospective use of applicability conditions occurred in zero of the 43 traceable
   runs in any model: no run inferred from COND that the feared path had become safe
   before choosing. The closest behaviors are post-hoc: DeepSeek-V4.1-Flash's honest
   post-crash report; Gemma3-12B's post-crash inference that "the missingness
   pattern changed" (which it used to justify caution, not to conclude the path was
   safe).
4. Under frozen checkpoints, DeepSeek-V4.1-Flash deterministically fails the
   *entire revived column* by committing its only attempt to a crashing tool — the
   cell type that motivates conditional memory in the first place — while
   Qwen2.5-7B's unreflective default is exactly correct under revival (model names
   are descriptive here; no independent capability testing establishes a strict
   hierarchy on this task). Recovery from the crash is impossible *by design* under
   the single-commitment budget and occurs under a multi-attempt budget
   (Gemma3-12B); this pattern motivates pre-registering attempt budget as an
   independent factor in future designs rather than explaining behavior by it post
   hoc.

### 7.2 The memory-value window: a hypothesis, with a test

The observations motivate a precisely stated hypothesis:

> **H-window**: A failure memory can add decision value only for agents that
> (i) do not already encode the lesson (otherwise the memory is redundant) and
> (ii) can act on memory instructions (otherwise the memory is inert). On the
> tested axis these populations did not overlap.

Status: **hypothesis, not finding.** Neither conjunct has been independently
measured — (i) is inferred indirectly and unfalsifiably from behavior (a model that
ignores a memory may know the lesson *or* may fail to use memories at all), and the
pilot is equally consistent with strategy inertia, tool preference, or budget
effects. The concrete test follows from the diagnosis and is now specified as a
protocol (v2): rebuild the natural default around information that is *absent from
model weights by construction* — project-private parameters, randomly generated
environment conventions, or session-local state, regenerated per instance — while
holding task structure, models, and harness fixed, with uniform base prompts across
conditions (§4.4 confound removed). Under that design, a NONE-condition agent cannot
succeed by prior knowledge, so condition (i) holds by construction and the
manipulation tests the memory's *incremental* value rather than re-testing textbook
statistics.

Because neither conjunct of H-window is measured by the main manipulation alone,
the proposed protocol (v2) adds two dedicated checks, one per conjunct:

- **(a) Information-unobtainability check (conjunct i).** Every task instance's
  project rules/parameters are randomly generated per instance, with the generation
  seed, the prior environment, and the change record retained as artifacts; task
  descriptions and tool interfaces are audited so the instance-level answer cannot
  be leaked through the prompt or inferred from fixed tool names. This makes "the
  memory carries instance-level information absent from the weights" verifiable
  rather than assumed.
- **(b) Positive control for memory-use ability (conjunct ii).** A separate
  condition in which the memory is unambiguous, correct, and clearly applicable to
  the current task. A model that cannot act on memory even under this condition
  should not have the main experiment's null result interpreted as "the memory
  lacks incremental value" — the failure would lie in memory use, not memory
  content.

Testing the two conjuncts separately has more explanatory power than comparing
"weaker" and "stronger" models. The full design is the proposed confirmatory
protocol (draft v2, pending pre-registration); it reuses the released instrument
except for the task families' information content and the prompt structure.

### 7.3 Consilience with prior observations

BASM's unexplained App.-Skill dip (−5.95pp on a 32B model, +7.14pp on a 14B model)
is *consistent with* H-window — conditioning information being load-bearing for one
model and redundant-or-encumbering for another — but our pilot does not explain it;
establishing that link requires the confirmatory experiment above. MCMA's
environment-dependent copilot harm and [2607.08716]'s Injection-only −6.0pp are
additional signals that memory-side interventions can be non-monotonic; none of
these works measured wrong abandonment of revived paths.

### 7.4 Design guidance (revised, narrower than v0.1)

1. *Information selection*: target weight-absent information (private parameters,
   generated conventions, session state), not textbook statistics.
2. *Uniform prompts across conditions*: the memory payload must be the only
   difference; an empty-slot + same-rule NONE condition is mandatory (this pilot's
   confound, §4.4).
3. *Separate primary endpoint from stopping gate*: pre-register both, and never
   report a stopped pilot as having tested the primary outcome (§3.4).
4. *Conflict with procedure, not with knowledge*: an SOP-vs-memory design measures
   compliance, a different construct — keep it separate from applicability
   reasoning.
5. *Prospective condition use must be elicited structurally* (e.g., a required
   condition-check step) — and reported as instruction-following, not spontaneous
   discrimination.
6. *Harness logging and budget*: record the model, per-variant result tags, full
   OBSERVATION payloads, and `methods_tried` vs `method_adopted` separately; treat
   attempt budget as a pre-registered variable (this pilot: single commitment
   freezes all switching; multi-attempt permits recovery).

---

## 8. Limitations

1. **Single task family in all agent runs** (A); B/C instruments verified but never
   exercised with agents. Family A's bias mechanism is the most textbook-adjacent of
   the three; a less standard mechanism could change DeepSeek-V4.1-Flash's NONE
   behavior.
2. **n=1 per cell; three models; one prompt family per harness variant.** All
   diagnostic claims are qualitative and direction-level.
3. **No semantic judge** → C-A3 unjudgeable; composite scores reported as bounds.
   Additionally, the frozen weight structure lets C-A1 + C-A4 alone reach the pass
   threshold — `pass_guaranteed` runs may lack both CI and any semantic check
   (metric-validity defect, §3.3; to be redesigned as a necessary-condition rule in
   the next protocol, not patched here).
4. **Treatment confound**: memory conditions received a memory-usage instruction
   that NONE did not (§4.4). All memory-vs-NONE comparisons are
   memory+instruction vs. nothing; content effects cannot be isolated in this data.
5. **The frozen mi_tool returns no CI** → C-A2 is structurally unreachable via that
   method (instrument gap; frozen implementations must not be edited post hoc, so
   this is recorded rather than patched).
6. **Model attribution is log-reconstructed** (runner defect, Section 5); JSONs were
   overwritten across harness variants, so 33 of 43 runs are scored from logs, and
   two ladder claims (rows 2 and 5) are unverifiable for lack of any retained
   artifact.
7. **Memories were injected verbatim** with dense-retrieval logging; no consolidation
   noise. Findings concern memory *use*, not *formation*.
8. **Neither conjunct of H-window is independently measured in this data**: "knowledge
   absent from weights" is inferred indirectly (a NONE-condition failure can also
   mean the model cannot execute the task or cannot use memories), and no positive
   control for memory-use ability exists. The proposed protocol adds both checks
   (Section 7.2).
9. **No "first" claims**: the audit (Table 1) describes eight works read in full on
   2026-10-21 under a stated screening criterion (§2); a systematic pre-registration
   sweep (e.g., SciReg / OSF registries) has not been performed, so the seam
   statement is corpus-bounded.

---

## 9. Conclusion

We built and released a gate-verified instrument for paired validity manipulation of
agent failure memory and ran the pre-registered confirmatory pilot, which triggered
its pre-registered behavioral stopping gate: among the 43 traceable runs of the
reconciled inventory (uniqueness machine-verified; two early variants without
retained artifacts excluded as unverifiable), the designed directional contrast
never appeared. A full evidence–claim reconciliation against retained artifacts
scored all 43 runs (or marked them unjudgeable/unverifiable) and corrected three
residual narrative errors from earlier drafts; the surviving observations are
narrow — no UNCOND-vs-NONE contrast among the traceable runs, zero prospective use
of applicability conditions, and behavior that *corresponds with* budget structure
and model identity (non-causally: the factors were never crossed) rather than
evidence identifying memory-content effects. We report the failure with its
complete diagnostic trail, a disclosed treatment confound, and a precisely stated
hypothesis — the memory-value window — together with a proposed confirmatory
protocol (draft, pending pre-registration) with per-conjunct validity checks that
would test it. The instrument, the negative result, and the correction history
(v0.1's strong-model mischaracterization; v0.3's three log-based corrections) are
released in full.

**Artifacts**: simulators and gold standards (3 families), nine frozen record
templates + contamination checker, trap tools, method-menu agent + runner,
retroactive scorers (JSON + log-only), the unique run manifest, all run JSONs,
retained execution logs, the evidence audit, the protocol v2 draft, and the
decision record (master design doc v2.4, task-set v1, harness change log, stage-A
closure report + errata).

---

## References (as audited 2026-10-21; re-verify before any submission)

1. [2606.21024] H. Wang. Negative Knowledge as Failure-aware Shared Memory for
   AutoResearch. arXiv:2606.21024, 2026.
2. [2505.16067] Xiong et al. How Memory Management Impacts LLM Agents: An Empirical
   Study of Experience-Following Behavior. ACL 2026 (2026.acl-long.27) /
   arXiv:2505.16067.
3. [2609.09115] MeClear: Task-Conditioned Memory Clearance. arXiv:2609.09115, 2026.
4. [2607.08716] *Remember When It Matters: Proactive Memory Agent for Long-Horizon
   Agents.* arXiv:2607.08716, 2026. (title corrected in v0.3 per external review;
   v0.1–v0.2 erroneously listed "Selective Memory Grounding")
5. [2605.12978] Useful Memories Become Faulty. arXiv:2605.12978, 2026.
6. [2608.22339] When Not to Imitate: Boundary-Aware Skill Memory for Reliable
   Tool-Use LLM Agents (BASM). EMNLP 2026 Findings / arXiv:2608.22339, 2026.
7. [2026.findings-acl.1535] Liang et al. Learning How to Remember: A Meta-Cognitive
   Management Method for Structured and Transferable Agent Memory (MCMA). ACL 2026
   Findings.
8. [2608.11224] Harnessing Agent Memory to Build Lifelong AI Partners for Materials
   Scientists. arXiv:2608.11224, 2026.
9. [2607.24759] Moreira & Sweet. Beyond Memory: A Templated Substrate for
   Heterogeneous Collaborative Knowledge Work with LLM Agents. arXiv:2607.24759, 2026.
10. [2605.26667] MemFail: Stress-Testing Failure Modes of LLM Memory Systems.
    arXiv:2605.26667, 2026.

---

## Appendix A — Evidence map (project-internal sources for every numeric claim)

| Claim | Source |
|---|---|
| DeepSeek 8×(method=mi_tool), V-dead exit-0 Δ=2.0306 / V-revived exit-1 assertion crash; NONE cells `memory_injected: False` | `results/kwm_stage_a/A_*_r0.json`; attribution: `logs/kwm_cloud.log` |
| Qwen r1: naive_delete, Δ=8.1051, CI [7.0432, 9.1670], N=1440; UNCOND final cites memory bias "6.0pp" | `A_{UNCOND,COND}_V_dead_r1.json`; `logs/kwm_stab.log` |
| Confirmatory pilot L2 (Table 6.1: COND/COND-MINUS adopt mi_tool/impute; NONE/UNCOND adopt X) | `logs/kwm_pilot24.log` runs 1–8 |
| Ladder rows 1–4, 6 (Table 6.2) | `logs/kwm_{pilot,uncond,gem,gem_menu,gem_rule,bind,cloud,v3}.log` |
| Rows 2 and 5 unverifiable | absence of retained artifacts; noted in harness-change record |
| Retroactive L3 scores — JSONs | `experiments/kwm_score_retro.py`; `results/kwm_stage_a/l3_scoring_retroactive.json` |
| Retroactive L3 scores — log-only (33 runs, per-run log citation) | `experiments/kwm_score_logonly.py`; `results/kwm_stage_a/l3_scoring_logonly.json`; `docs/KWM-EVIDENCE-AUDIT-2026-10-21.md` |
| Run uniqueness (43 unique; disjoint JSON/log pools; regenerable group totals) | `experiments/kwm_run_manifest.py`; `results/kwm_stage_a/run_manifest.json` + `run_manifest.md` |
| Frozen checkpoints & tolerances (C-A1 ±0.3; C-A2 [0.9,3.5]; weights 0.4/0.2/0.2/0.2; pass ≥0.6; C per-variant golds 991/992) | `docs/KWM-TASKSET-V1-2026-10-21.md` §3.6/§4.4/§5.4/§7 |
| Family golds & frozen-implementation outputs | `data/kwm_stage_a/gold_{a,b,c}.json`; `docs/KWM-HARNESS-CHANGE-2026-10-21.md` |
| Contamination check values (0.487; all green) | `experiments/kwm_contamination_check.py` output |
| Gate V6 (NONE→X, Δ=8.105) | `logs/kwm_pilot.log`; closure report §2 |
| BASM numbers (7.31/10.71/2.26; 69.4%; +23.8/−5.95/+7.14) | project audit `docs/KWM-MASTER-2026-10-20.md` §1.7 |
| RQ and H-A/H-B/H-C wording (frozen) | `docs/KWM-MASTER-2026-10-20.md` §3.1–3.2; task-set doc |
| Confound disclosure (usage rule only with memory) | `experiments/kwm_agent.py` (v3 prompt assembly); §4.4 |
| 2607.08716 title correction | external review #2 (2026-10-21); to re-verify against arXiv before submission |
| v0.1→v0.2→v0.3 corrections | this document's correction log; errata in closure & harness records; evidence audit C1–C3 |

## Appendix B — Decision record

1. Stop-loss chosen over SOP-conflict redesign (different construct) and over
   private-information rebuild (now specified as protocol v2, the H-window test).
2.    Cloud-model option was exercised before stop-loss — it produced the corrected
   DeepSeek data, not a rescue.
3. L3 scoring was retroactively completed for all surviving runs after external
   review #1 identified the gap between the pre-registered primary outcome and the
   v0.1 report; after external review #2, scoring was extended to all 33 log-only
   runs (`kwm_score_logonly.py`) and the semantic judge remains unbuilt and is
   reported as unjudgeable.
4. v0.1 claims ("8/8 optimal", "structurally infeasible", implied primacy) are
   retracted in v0.2. v0.3 additionally corrects: the confirmatory pilot's
   "8/8 chose X" description (C1), Gemma row 2′ "arbitrated to the correct answer"
   (C2), and Gemma row 3 "8/8 chose X" (C3) — all rebuilt from retained logs in
   `docs/KWM-EVIDENCE-AUDIT-2026-10-21.md`.
5. Primary endpoint and stopping gate are now formally separated (§3.4); per review
   #2, the report no longer states that H-A was tested to a negative conclusion.
6. v0.4 (third review): a unique run manifest was added and machine-verified —
   43 unique runs, disjoint JSON/log-only pools (the `kwm_stab.log` runs 2/4 are the
   surviving r1 JSONs; no double counting); all inventory-wide claims are bounded to
   the 43 traceable runs; capability-tier labels ("strongest/floor model") are
   replaced by model names; "ruled out" is replaced by "does not identify an effect"
   per n=1 evidence; the single-commitment switching absence is framed as a design
   property; G-core is framed strictly as a pilot-stopping decision; the next-round
   design is relabeled "proposed confirmatory protocol (draft, pending
   pre-registration)" and gains per-conjunct H-window validity checks (§7.2).
