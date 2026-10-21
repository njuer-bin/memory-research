# Diagnosing a Failed Confirmatory Pilot for Paired Validity Manipulations of Agent Failure Memory

**Tech report draft v0.2 (2026-10-21; supersedes v0.1 after external review)**

> **Status**: DRAFT — internal diagnostic record, not submission-ready.
> **What this document is**: the record of a pre-registered confirmatory pilot that
> failed to produce its expected effect, the diagnostic exploration that followed, and
> a retroactive outcome-layer (L3) scoring of every surviving run.
> **What this document is not**: it does not establish that "the memory-value window
> is empty" (Section 7 states this as a *hypothesis with a concrete test design*),
> does not claim a structural defect of any research program, and does not claim
> primacy for the instrument.
> **Evidence binding**: every numeric claim maps to a source in Appendix A. Claims
> from adaptive post-hoc harness variants are labeled [exploratory]; per-cell n=1
> throughout; no effect estimates are asserted anywhere.
> **Correction log**: v0.1 mischaracterized the strong model's behavior
> ("8/8 cells choose the optimal method") — in V-revived the chosen tool crashes and
> the run deterministically fails the frozen outcome criteria. v0.2 rebuilds the
> matrix from run JSONs + retained logs and adds retroactive L3 scoring. The
> mischaracterization and its correction are themselves recorded in Section 6.4 and
> in the errata of the closure/harness records.

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

The pilot failed: method choice was invariant across all four memory conditions
in every run. A subsequent adaptive diagnostic across five harness variants and three
model tiers (Qwen2.5-7B, Gemma3-12B, DeepSeek-V4.1-Flash) — reported here as
*exploratory*, not confirmatory evidence — found the same invariance everywhere, with
model identity, not memory content, determining behavior: weaker models always chose
the naive method (biased under V-dead, correct under V-revived); the strongest model
always chose the robust tool (correct under V-dead; crashes under V-revived, where it
honestly reports that no estimate is possible). Retroactive scoring against the frozen
outcome checkpoints makes the complementarity exact: under V-revived the strongest
model deterministically fails its task while the weaker models' default would pass;
under V-dead the reverse holds. Memory content changed nothing at either layer.

We contribute: (1) a reusable, gate-verified instrument for paired validity
manipulations of failure memory (released with all traces); (2) a documented negative
confirmatory result with a complete diagnostic trail; and (3) a precisely stated
hypothesis — the *memory-value window*: a memory can only add decision value when the
agent neither already knows the lesson nor lacks the capability to act on it —
together with the experiment design that would test it (task-private information
absent from model weights). We explicitly do not claim the window is empty in
general; we report that no tested configuration occupied it.

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
audited corpus as of 2026-10-21, not a primacy claim; see Section 8.)

We designed a 2×4 experiment (validity variant {V-dead, V-revived} × memory
representation {NONE, UNCOND, COND, COND-MINUS}) with pre-registered directional
hypotheses, built the complete instrument, and ran the confirmatory pilot. It failed
in an instructive way, and this report is organized around that failure:

- **Section 4–5** document the instrument and the pilot protocol.
- **Section 6** rebuilds the result matrix from primary artifacts (run JSONs and
  retained logs), scores every surviving run against the frozen outcome checkpoints,
  and separates the confirmatory pilot from the exploratory diagnostics.
- **Section 7** extracts what the failure *does* and *does not* license us to say,
  and states the hypothesis-level takeaway (the memory-value window) with the design
  that would test it.

**Contributions.**

1. **An instrument** (engineering contribution, verifiable from the repository and
   its gate records): three task families with frozen gold standards and trap tools;
   nine fact-matched memory templates under five-part contamination control; a
   method-menu harness turning method choice into the primary behavioral metric; and
   gate checks V1–V6.
2. **A documented negative confirmatory result**: the pre-registered directional
   memory effect (H-A) was not observed at the behavioral layer in any run, and a
   retroactive outcome-layer scoring shows task success is fully determined by
   model-identity × environment-variant, invariant to memory representation.
3. **A hypothesis with a test design**: the *memory-value window* — memory adds
   decision value only where (i) the lesson is absent from model weights and (ii) the
   agent can act on the memory — plus the concrete next experiment (task-private
   information domains) that would confirm or refute it.

---

## 2. Related Work

We audited eight direct competitors by reading their experimental sections in full
(2026-10-21; four dimensions: memory object D1, paired validity manipulation D2,
cost-side measurement D3, representation intervention D4). Table 1 summarizes the
audit; it is a corpus description, not a completeness claim.

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

**The seam.** The conjunction — failure experience *itself* as memory object (D1) +
within-family paired validity manipulation (D2) + benefit–cost dual measurement (D3) —
is occupied by none of the eight audited works. BASM is the strongest neighbor on D4:
its boundary fields are a causally validated anti-imitation channel (same-prompt
knockout recovers 69.4% of the imitation-trap effect; wrong-tool margin 7.31 → 10.71
→ 2.26 across memory-free / procedure-skill / BASM), and it reports AppWorld +23.8pp,
BFCL +5.0pp, AgentDojo attack-success −4.6pp. But its T_wrong cell encodes "should
suppress"; it never constructs the complementary "should retry" cell, and it measures
no cost side. Notably, its intermediate configuration — adding *only* the
applicability-condition field to procedure skills ("App. Skill") — is non-monotonic
and model-dependent (AppWorld-Qwen3-32B 66.07→60.12, −5.95pp; BFCL-Qwen3.5-397B
53.38→51.63, −1.75pp; AppWorld-Qwen3-14B 46.43→53.57, +7.14pp), a phenomenon its
authors neither measure as a cost nor explain. Our diagnostic results are consistent
with this pattern (Section 7.1) but do not explain it; that would require the
confirmatory experiment this pilot failed to enable.

Adjacent signals that conditioning information can *hurt*: MCMA disables its
success-summary copilot on ScienceWorld; [2607.08716] reports Injection-only at
−6.0pp on one environment; [2608.11224] reports memory transplantation that can be
neutral-to-harmful when source memory is weaker than the target's knowledge, and
converts failed computations into pre-execution guardrails. None treats these as the
phenomenon under study. Motivation-level work [2607.24759] argues for retaining dead
paths in append-only wikis (design + case study, no controlled measurement).

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
- **Primary outcome (H-A, pre-registered)**: task success rate (frozen weighted
  checkpoints, Section 3.3), with directional expectations per variant (UNCOND vs
  NONE: ↑ in V-dead, ↓ in V-revived). **H-B**: COND vs UNCOND mitigates damage in
  V-revived and is non-inferior in V-dead. **H-C**: COND vs COND-MINUS isolates
  conditional information beyond length/genericity.

### 3.2 Behavioral metrics

Three layers, all logged per run: **L1 usage** (memory injected / retrieved /
explicitly cited), **L2 behavior** (method-choice sequence; under the method-menu
harness this is the explicit `methods_chosen` sequence), **L3 outcome** (frozen
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
(otherwise — C-A3 is decisive). Families B/C checkpoints are analogously frozen
(B: total = 500,000 exact, month-row count exact; C: total = 50,000 exact, output
customer count = *that variant's* frozen unique-customer gold — 991 in v1, 992 in
v2 — so a per-variant-correct output is not a cross-variant mismatch).

---

## 4. Instrument

All artifacts are frozen in the repository (paths in Appendix A).

### 4.1 Task families

Three data-analysis families, each with a naive method X (the NONE-condition natural
default), a memory-suggested alternative Y (a project trap tool whose frozen
implementation is task-passing in V-dead and crashes in V-revived), and gold
standards:

| Family | Failure mode of X | X (naive) | Gold | X in V-dead | X in V-revived | Y (trap tool) in V-dead | Y in V-revived |
|---|---|---|---|---|---|---|---|
| A — missing data | MAR by (group, z2); deletion biases the group contrast | `dropna` + group means | Δ = 2.1 | Δ = 8.105 (C-A1 fail: dev +6.005) | Δ = 1.882 (C-A1 pass: dev −0.218 ≤ 0.3) | mi_tool: 2.031 ✓ | assertion crash |
| B — timestamp parsing | 12% unparseable timestamps silently dropped | month `groupby` on coerced ts | total = 500,000 | 440,000 (fail) | 500,000 ✓ | legacy_tool: 500,000 ✓ | column-position crash |
| C — SQL join fanout | duplicated customer rows fan out JOIN; Y merges distinct customers by name case | direct JOIN aggregated by id | (50,000; 991 in v1 / 992 in v2) | (59,280; 991) | (50,000; 992) ✓ | dedup_by_name: (50,000; 991 ✓) | (50,000; **904**) — silent customer loss |

Notes on the numbers, addressing two review questions directly: (i) the frozen
C-A1 tolerance is *absolute* (±0.3), so the MI tool's finite-sample deviation
(2.0306, dev −0.069) passes and the naive estimator in V-revived (1.882, dev
−0.218) also passes — statistical unbiasedness of an estimator and passing a
single-realization checkpoint are distinguished exactly as the reviewer requested,
and the tolerance table was frozen before any agent run (task-set doc, "禁止看结果
后调"); (ii) family C's V-dead Y output (50,000; 991) is checked against v1's own
frozen gold (991), not v2's (992) — the checkpoint table freezes per-variant golds.

### 4.2 Frozen memory templates and contamination control

Nine record templates (3 families × UNCOND/COND/COND-MINUS), fact-matched per family;
COND-MINUS equalized in length with neutral archival filler. A five-part
contamination check verifies: no gold *absolute* value in any template (stricter than
the original spec, resolving an internal task-set conflict in favor of anti-leakage);
COND-MINUS semantic distance from COND < 0.5 cosine (bge-m3, 1024-d; measured 0.487);
length parity within 10%; template consistency; no leak words in task descriptions.
All checks green before any agent run.

### 4.3 Memory layer and retrieval

A purpose-built memory layer (dense retrieval over frozen records with bge-m3,
top-k=3, hit threshold 0.30) injects the family-appropriate template at session
start; injection and retrieval events are logged (L1: `memory_injected`,
`memory_hit_condition` fields in every run JSON; NONE runs show `False`/`None`).

### 4.4 Agent harness evolution (v1 → v3)

- **v1 free-code ReAct**: prompt-based ACTION/OBSERVATION loop with sandboxed
  tools. Guards: `tool_call_id` on tool-result messages; FINAL rejected if no
  compute tool was called or placeholders present (≤2 nudges). Fixed a parser bug
  where models narrate an entire ReAct trace as text and a FINAL-priority parser
  never executes it.
- **v3 method menu**: the agent only emits `METHOD: <name>` (frozen implementation
  executes; OBSERVATION returned) or `FINAL:`; budget = 1 method execution (single
  commitment); a memory-usage rule is rendered *only* when memory exists (NONE
  unpolluted). `methods_chosen` is the L2 metric. The menu freezes method
  implementations, removing code-quality confounds across conditions.

### 4.5 Pre-run gates

| Gate | Content | Status |
|---|---|---|
| V1 data reproduction | simulators reproduce designed golds | pass |
| V2 revival | deletion-based methods unbiased in V-revived | pass |
| V3 trap tools | Y passes in V-dead, crashes in V-revived (A, B); C's Y-trap verified | pass |
| V4 templates | contamination checks all green | pass |
| V5 family-B exploration | v1/v2 asymmetries verified end-to-end | pass |
| V6 natural default | NONE-condition agents first-choose X | pass (qwen2.5:7b, 2/2; Δ=8.105) |
| V7 UNCOND implied-condition pretest | not reached (stop-loss preceded it) | — |

---

## 5. Stage-A Procedure and Provenance

All runs executed locally (Ollama) or via an Anthropic-protocol cloud endpoint
(DeepSeek-V4.1-Flash). Every run writes a JSON with condition tags,
`methods_chosen`, tool calls/results, and the final answer.

**Provenance defect (recorded, not silently patched):** the runner did not write the
model identifier into the run JSON. Attribution below therefore relies on retained
execution logs (now committed under `results/kwm_stage_a/logs/`): `kwm_cloud.log`
(17:18, 8 runs, DeepSeek-V4.1-Flash) matches the eight `*_r0` JSONs;
`kwm_stab.log` (16:26, 2 runs, Qwen2.5-7B) matches the two `*_r1` JSONs.
Additionally, later harness variants reused the same result tags and *overwrote*
earlier JSONs; surviving JSONs therefore represent only the final state of each tag.
Intermediate runs' evidence lives in the retained logs and the harness-change record.
For future runs the runner has been noted to require a `model` field and
per-variant tag namespacing (not retroactively applied — no history was rewritten).

---

## 6. Results

### 6.1 Confirmatory pilot (family A, 2×4, method-menu harness, Qwen2.5-7B) — **G-core failed**

The pre-planned falsification pilot (retained log `kwm_pilot24.log`): all eight
cells chose the naive method X, including UNCOND/V-dead where the memory explicitly
quantifies the bias; L1 memory-hit signals were positive and final answers *quoted*
the memory's content while committing to X. No directional reversal is measurable
when the behavioral metric does not move. Per the pre-registered G-core rule this is
a failed pilot, not a negative finding about memory in general.

### 6.2 Exploratory diagnostics [exploratory — adaptive variants, n=1 per cell]

Because the confirmatory pilot failed, we ran an adaptive ladder of harness variants
to determine whether the failure was instrumental (fixable) or systematic. Each
variant ran only after the previous failed; variants differ in prompts and are not
poolable.

| # | Harness variant | Model | Outcome |
|---|---|---|---|
| 1 | Free-code ReAct | Qwen2.5-7B / Gemma3-12B | rationalized bypass: after "never use dropna on this data," Gemma3 computed column means directly (pandas `mean` implicitly skips NaN = same deletion estimator), obtained 8.105, and reported "I did not use dropna" |
| 2 | Menu + multi-attempt | Qwen2.5-7B | never switches (4/4 cells choose X) |
| 2′ | Menu + multi-attempt | Gemma3-12B | explores all three methods and self-arbitrates to the correct answer; memory adds nothing |
| 3 | Menu + single commitment + memory-usage rule | Gemma3-12B | 8/8 cells choose X; final answers restate memory content ("bias ≈ 6.0pp", "48% missing in B") yet commit to the biased method citing budget; in V-revived it *fabricates* applicability ("missing pattern similar") to keep X |
| 4 | Decision-point binding instruction | Gemma3-12B | still 4/4 choose X |
| 5 | Memory relocated to first user message | Gemma3-12B | no effect (4/4 choose X) |
| 6 | v3 harness + cloud strong model | DeepSeek-V4.1-Flash | **all 8 cells choose `mi_tool`** — see §6.3; this is *not* "choosing the optimal method" in V-revived (v0.1's wording, corrected) |

### 6.3 The corrected result matrix (rebuilt from primary artifacts)

**DeepSeek-V4.1-Flash, harness v3, family A, all 8 cells choose `mi_tool`
(`A_*_r0.json`; log `kwm_cloud.log`):**

| Variant | Conditions | Method | Tool result | Behavior |
|---|---|---|---|---|
| V-dead | NONE / UNCOND / COND / COND-MINUS | mi_tool | exit 0, Δ = 2.0306, no CI | passes task |
| V-revived | NONE / UNCOND / COND / COND-MINUS | mi_tool | **exit 1, assertion crash** | honest "no estimate available"; never falls back to X (budget = 1) |

Three observations, each verified against the JSONs: (i) the choice is *invariant
across all four memory conditions* — including NONE (`memory_injected: False`) — so
the strong model's behavior is a **mechanical preference for the robust tool**
(strategy inertia / over-generalization), *not* condition adaptation and *not*
demonstrably "prior knowledge covers the memory's value"; (ii) in V-revived the
chosen tool is **not** the optimal method — the naive method would have passed the
frozen checkpoints (dev −0.218 ≤ 0.3) — so the strongest model deterministically
*fails* the entire V-revived column; (iii) its failure reports are honest (no
fabricated numbers) but never use the COND applicability condition to *anticipate*
that the feared path had become safe.

**Qwen2.5-7B (surviving `*_r1.json`; log `kwm_stab.log`):** both stability-rerun
cells (UNCOND, COND × V-dead) chose `naive_delete`, Δ = 8.1051, CI [7.0432, 9.1670]
(N=1440); the UNCOND final answer cites the memory ("历史经验表明此方法的偏差约为
6.0pp") *while adopting the biased estimate* — the acknowledge-but-not-act pattern.

### 6.4 Retroactive L3 scoring (frozen checkpoints; script `experiments/kwm_score_retro.py`; output `results/kwm_stage_a/l3_scoring_retroactive.json`)

| Run | Model | Method | Est. | Dev vs 2.1 | CI width | s1 | s2 | s4 | Det. subtotal | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| NONE / V-dead | deepseek | mi_tool | 2.0306 | −0.069 | — | 1 | 0 | 1 | 0.60 | pass_guaranteed |
| UNCOND / V-dead | deepseek | mi_tool | 2.0306 | −0.069 | — | 1 | 0 | 1 | 0.60 | pass_guaranteed |
| COND / V-dead | deepseek | mi_tool | 2.0306 | −0.069 | — | 1 | 0 | 1 | 0.60 | pass_guaranteed |
| COND-MINUS / V-dead | deepseek | mi_tool | 2.0306 | −0.069 | — | 1 | 0 | 1 | 0.60 | pass_guaranteed |
| NONE / V-revived | deepseek | mi_tool | CRASH | — | — | 0 | 0 | 1 | 0.20 | **fail_guaranteed** |
| UNCOND / V-revived | deepseek | mi_tool | CRASH | — | — | 0 | 0 | 1 | 0.20 | **fail_guaranteed** |
| COND / V-revived | deepseek | mi_tool | CRASH | — | — | 0 | 0 | 1 | 0.20 | **fail_guaranteed** |
| COND-MINUS / V-revived | deepseek | mi_tool | CRASH | — | — | 0 | 0 | 1 | 0.20 | **fail_guaranteed** |
| UNCOND / V-dead (r1) | qwen | naive_delete | 8.1051 | +6.005 | 2.124 | 0 | 1 | 1 | 0.40 | unjudgeable (C-A3 decisive; passes only if s3 = 1.0) |
| COND / V-dead (r1) | qwen | naive_delete | 8.1051 | +6.005 | 2.124 | 0 | 1 | 1 | 0.40 | unjudgeable (C-A3 decisive) |

(s2 = 0 for all deepseek V-dead cells because the frozen mi_tool implementation
returns a point estimate only — an instrument gap recorded in Section 8.)

**What the scored matrix shows.** Under the frozen outcome criteria, task success is
a deterministic function of *model identity × environment variant*, with zero
variance from the memory representation: DeepSeek passes V-dead (4/4) and
deterministically fails V-revived (4/4); Qwen's default would pass V-revived and
fail V-dead (its V-revived cells under the earlier harness, retained only in
`kwm_pilot24.log`, chose X with Δ ≈ 1.88). The 2×4 collapses to
(model × variant); the memory manipulation did not move either layer. This is the
honest statement of the pilot's failure — narrower than v0.1's narrative and now
fully scoreable.

### 6.5 Evidence strength

- The confirmatory pilot is one family (A), one model, n=1 per cell (two stability
  reruns). The diagnostics are exploratory, adaptive, and non-poolable.
- Model attribution relies on logs because the runner omitted the model field
  (defect recorded, Section 5); attribution is unambiguous for the surviving JSONs
  (distinct harness signatures and timestamps), but we mark it as reconstructed.
- The scoring package for future runs remains unbuilt except for the retroactive
  family-A scorer; families B/C have frozen checkpoints but no agent runs, hence no
  scores ("not applicable" rather than "failed").

---

## 7. Discussion

### 7.1 What this pilot does and does not establish

**Does not establish**: that failure memory cannot help agents; that the
memory-value window is empty in general; that the manipulation is structurally
infeasible; or that any model "already knows" the lesson (the strong model's
behavior is equally consistent with mechanical robust-tool preference). The
confirmatory effect (H-A) was simply not observed, and per-protocol the correct
response is to report the failure with its diagnostics — which is what this report
is.

**Does support (as observations, within scope)**:

1. Across five harness variants and three model tiers, *method choice never varied
   with memory representation* — including a strong model that quotes the memory's
   quantitative content in its answers (weaker models) and one that never needed it
   (strongest model). Whatever mediates memory→behavior transfer, it was not
   engaged by any tested configuration on this task family.
2. Outcome-layer success under frozen checkpoints is fully determined by
   model-identity × validity-variant (Section 6.4): the strongest model fails the
   *entire revived column* — the cell type that motivates conditional memory in the
   first place — by committing to a crashing tool with a single-commitment budget.
   Conversely, the weaker models' "unreflective" default is exactly correct under
   revival.
3. Applicability conditions were never used *prospectively* at any tier: no run
   inferred from COND that the feared path had become safe before choosing. The
   strongest model's honest post-crash reporting is the closest behavior, and it is
   post-hoc.

### 7.2 The memory-value window: a hypothesis, with a test

The observations motivate a precisely stated hypothesis:

> **H-window**: A failure memory can add decision value only for agents that
> (i) do not already encode the lesson (otherwise the memory is redundant) and
> (ii) can act on memory instructions (otherwise the memory is inert). On the
> tested axis these populations did not overlap.

Status: **hypothesis, not finding.** The pilot is consistent with it but cannot
exclude that a different task family, information type, or model population would
fill the window. The concrete test follows from the diagnosis: rebuild the natural
default around information that is *absent from model weights by construction* —
project-private parameters, randomly generated environment conventions, or
session-local state — while holding task structure, models, and harness fixed. Under
that design, a NONE-condition agent cannot succeed by prior knowledge, so condition
(i) holds by construction and the manipulation tests the memory's *incremental*
value rather than re-testing textbook statistics. This is the recommended next
experiment if this line continues; it reuses the released instrument unchanged
except for the task families' information content.

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
2. *Conflict with procedure, not with knowledge*: an SOP-vs-memory design measures
   compliance, a different construct — keep it separate from applicability
   reasoning.
3. *Prospective condition use must be elicited structurally* (e.g., a required
   condition-check step) — and reported as instruction-following, not spontaneous
   discrimination.
4. *Harness logging*: record the model, per-variant result tags, and full
   OBSERVATION payloads in run JSONs (three defects this pilot hit are all
   provenance-related).

---

## 8. Limitations

1. **Single task family in all agent runs** (A); B/C instruments verified but never
   exercised with agents. Family A's bias mechanism is the most textbook-adjacent of
   the three; a less standard mechanism could change the strong model's NONE
   behavior.
2. **n=1 per cell; three models; one prompt family per harness variant.** All
   diagnostic claims are qualitative and direction-level.
3. **No semantic judge** → C-A3 unjudgeable; composite scores reported as bounds.
   Two deepseek V-dead cells are `pass_guaranteed` only because their deterministic
   subtotals already reach the 0.6 threshold; qwen's V-dead cells are
   `unjudgeable` and lean failing.
4. **The frozen mi_tool returns no CI** → C-A2 is structurally unreachable via that
   method (instrument gap; frozen implementations must not be edited post hoc, so
   this is recorded rather than patched).
5. **Model attribution is log-reconstructed** (runner defect, Section 5); JSONs were
   overwritten across harness variants, so intermediate-run numbers survive only in
   logs.
6. **Memories were injected verbatim** with dense-retrieval logging; no consolidation
   noise. Findings concern memory *use*, not *formation*.
7. **"First" claims are avoided**: the audit (Table 1) describes eight works read in
   full on 2026-10-21; a systematic pre-registration sweep (e.g., SciReg / OSF
   registries) has not been performed, so the seam statement is corpus-bounded.

---

## 9. Conclusion

We built and released a gate-verified instrument for paired validity manipulation of
agent failure memory, ran the pre-registered confirmatory pilot, and it failed: task
success under frozen checkpoints is fully determined by model identity and
environment variant, invariant to memory representation, across every tested
configuration. We report the failure with its complete diagnostic trail, a
retroactive outcome-layer scoring of every surviving run, and a precisely stated
hypothesis — the memory-value window — together with the experiment that would test
it. The instrument, the negative result, and the correction history (including this
document's own v0.1 mischaracterization of the strong model, caught by review and
rebuilt from primary artifacts) are released in full.

**Artifacts**: simulators and gold standards (3 families), nine frozen record
templates + contamination checker, trap tools, method-menu agent + runner,
retroactive scorer, all run JSONs, retained execution logs, and the decision record
(master design doc v2.4, task-set v1, harness change log, stage-A closure report +
errata).

---

## References (as audited 2026-10-21; re-verify before any submission)

1. [2606.21024] H. Wang. Negative Knowledge as Failure-aware Shared Memory for
   AutoResearch. arXiv:2606.21024, 2026.
2. [2505.16067] Xiong et al. How Memory Management Impacts LLM Agents: An Empirical
   Study of Experience-Following Behavior. ACL 2026 (2026.acl-long.27) /
   arXiv:2505.16067.
3. [2609.09115] MeClear: Task-Conditioned Memory Clearance. arXiv:2609.09115, 2026.
4. [2607.08716] Remember When It Matters: Selective Memory Grounding. Meta AI,
   arXiv:2607.08716, 2026.
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
| DeepSeek 8×(method=mi_tool), V-dear exit-0 Δ=2.0306 / V-revived exit-1 assertion crash; NONE cells `memory_injected: False` | `results/kwm_stage_a/A_*_r0.json` (10 fields per run); attribution: `results/kwm_stage_a/logs/kwm_cloud.log` |
| Qwen r1: naive_delete, Δ=8.1051, CI [7.0432, 9.1670], N=1440; UNCOND final cites memory bias "6.0pp" | `results/kwm_stage_a/A_{UNCOND,COND}_V_dead_r1.json`; `logs/kwm_stab.log` |
| Retroactive L3 scores (table in §6.4) | `experiments/kwm_score_retro.py`; `results/kwm_stage_a/l3_scoring_retroactive.json` |
| Frozen checkpoints & tolerances (C-A1 ±0.3; C-A2 [0.9,3.5]; weights 0.4/0.2/0.2/0.2; pass ≥0.6; C per-variant golds 991/992) | `docs/KWM-TASKSET-V1-2026-10-21.md` §3.6/§4.4/§5.4/§7 |
| Family golds & frozen-implementation outputs | `data/kwm_stage_a/gold_{a,b,c}.json`; `docs/KWM-HARNESS-CHANGE-2026-10-21.md` |
| Contamination check values (0.487; all green) | `experiments/kwm_contamination_check.py` output |
| Gate V6 (NONE→X, Δ=8.105) | `logs/kwm_pilot.log`; closure report §2 |
| Ladder rows 1–5 | `logs/kwm_{pilot24,v3,bind,gem_rule,gem_menu,gem}.log`; harness-change record |
| Confirmatory pilot (qwen 2×4, 8/8 X) | `logs/kwm_pilot24.log`; closure report §2 |
| BASM numbers (7.31/10.71/2.26; 69.4%; +23.8/−5.95/+7.14) | project audit `docs/KWM-MASTER-2026-10-20.md` §1.7 |
| RQ and H-A/H-B/H-C wording (frozen) | `docs/KWM-MASTER-2026-10-20.md` §3.1–3.2; task-set doc |
| v0.1→v0.2 correction (DeepSeek wording) | this document's correction log; errata in closure & harness records |

## Appendix B — Decision record

1. Stop-loss chosen over SOP-conflict redesign (different construct) and over
   private-information rebuild (recommended as the H-window test if the line
   continues).
2. Cloud-model option was exercised before stop-loss — it produced the corrected
   ceiling-tier data, not a rescue.
3. L3 scoring was retroactively completed for all surviving runs after external
   review identified the gap between the pre-registered primary outcome and the
   v0.1 report; the semantic judge remains unbuilt and is reported as unjudgeable.
4. v0.1 claims ("8/8 optimal", "structurally infeasible", implied primacy) are
   retracted in v0.2; the correction is preserved in both documents' errata.
