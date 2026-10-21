# When Does Failure Memory Change Agent Behavior? Evidence That the Memory-Value Window Is Empty on the Competence Axis

**Tech report draft v0.1 (2026-10-21)**

> **Status**: DRAFT — not submission-ready. Evidence binding: every numeric claim is
> annotated with its source in the project record (docs / run JSONs); see Appendix A.
> Manuscript claims marked [exploratory] derive from adaptive post-hoc harness variants
> (n=1 per cell) and are reported as qualitative, direction-consistent observations,
> not effect estimates. Bibliographic metadata of the eight audited competitors was
> verified by reading full experimental sections on 2026-10-21 (project audit §1.7);
> a final citation re-verification against original sources is required before any
> submission.

---

## Abstract

Agent memory systems increasingly store *failure experiences* — paths that were tried
and failed, with reasons and evidence — so that downstream agents can avoid repeating
costly mistakes. Recent work shows such memories help, and that attaching explicit
*applicability conditions* to skill memories changes behavior. A complementary question
remains untested: when the environment changes and a previously dead path becomes
viable again, can an agent distinguish a *still-valid failure lesson* from an *expired
historical taboo*? We formalize this as a paired validity manipulation (V-dead vs.
V-revived) crossed with four memory representations (NONE / UNCOND / COND / COND-MINUS),
and build a fully frozen instrument: three data-analysis task families with verified
gold standards, contamination-checked fact-matched memory templates, trap tools whose
frozen implementations invert correctness across the two validity variants, and a
method-menu harness that makes method choice an explicit, logged decision.

In a falsification stage, the core manipulation failed *structurally rather than
instrumentally*. Across five harness variants and three model tiers (Qwen2.5-7B,
Gemma3-12B, DeepSeek-V4.1-Flash), no model occupied the "memory-value window" — the
region where an agent (i) does not already know that the naive path is biased, and
(ii) is capable of acting on a memory instruction. Weaker models acknowledge the
failure memory verbatim (repeating its quantitative content) yet keep choosing the
biased method, sometimes reinterpreting or fabricating applicability to justify doing
so; the strong model avoids the biased path in every condition *without* memory, because
prior statistical knowledge covers the memory's information value. Applicability
conditions were never used *prospectively* to infer that a feared path had become safe.
We release the instrument, all run traces, and the negative results, and argue that
paired validity manipulations of failure memory must target information that is absent
from model weights — a boundary condition that also explains the unexplained
non-monotonic dip of applicability-conditioned skills reported by prior work.

---

## 1. Introduction

Long-horizon knowledge-work agents accumulate memory across sessions. A growing
literature stores *negative knowledge*: failed paths with reasons and evidence
[2606.21024], boundary fields on skills [2608.22339], failure reflections [MCMA],
and boundary-knowledge guardrails in domain systems [2608.11224]. The demonstrated
value is avoidance: do not redo what failed.

This paper asks the reverse question, pre-registered in our design document before
any Stage-A data were collected:

> **RQ (frozen v2.4)**: *When a previously failed path becomes viable again in a new
> environment, what information helps an agent correctly revise the old lesson —
> rather than mechanically obeying it (wrong abandonment) or forgetting it entirely
> (repeating the failure)?*

Answering this requires manipulating the *current validity* of one and the same
failure memory. No prior work does so (Section 2): the nearest competitor, BASM
[2608.22339], conditions skill memories on applicability but never constructs the
"should-retry" cell; eight audited competitors lack within-family paired validity
manipulations and none measures wrong abandonment of a revived path.

We designed a 2×4 experiment (validity variant {V-dead, V-revived} × memory
representation {NONE, UNCOND, COND, COND-MINUS}) with pre-registered directional
hypotheses (H-A/H-B/H-C), built the complete instrument, and ran a falsification
stage. The instrument works: gold standards reproduce, trap tools crash exactly
where designed, contamination checks pass, and NONE-condition agents spontaneously
take the naive path. But the manipulation itself failed in an instructive way. Across
five harness variants — free-code ReAct, method menus with multi-attempt and
single-commitment budgets, decision-point binding instructions, memory salience
relocation — and three model tiers, method choice was *insensitive to failure-memory
content*:

- **Floor models (Qwen2.5-7B, Gemma3-12B)** exhibit an "acknowledge-but-do-not-act"
  pattern: they quote the memory's quantitative content (e.g., "bias ≈ 6.0pp",
  "48% missing rate in group B") in their final answers while still committing to the
  biased method, sometimes *reinterpreting* the memory ("the bias is relatively small")
  or *fabricating applicability* ("the missingness pattern is similar") to rationalize
  the choice.
- **Ceiling model (DeepSeek-V4.1-Flash)** selects the optimal method in all eight
  cells of the 2×4 *including the no-memory baseline*, because "deletion under MAR
  is biased" is already in its weights; in V-revived its preferred tool crashes and
  it honestly reports that no estimate is possible — but it never uses the COND
  applicability condition to *anticipate* that the revived path is safe.

We call the empty region the **memory-value window**: an agent must simultaneously
(i) not already know the lesson and (ii) be capable of acting on the memory. The two
requirements are anti-correlated on the observed competence axis. This is a boundary
result for the failure-memory program: it aligns with, and potentially explains, the
unexplained non-monotonic effect of applicability-conditioned skills in BASM
(−5.95pp on one model, +7.14pp on another), and it delimits where memory evaluations
must look — at information absent from weights (project-private parameters,
session-local state), not at textbook statistics.

**Contributions.**

1. **A formalization and instrument** (released) for paired validity manipulation of
   failure memory: three task families with verified gold standards and trap tools,
   fact-matched memory templates (UNCOND/COND/COND-MINUS) under contamination
   control, and a method-menu harness that turns method choice into an explicit,
   logged decision (the L2 behavioral metric).
2. **A five-variant × three-tier falsification result**: the method-choice
   manipulation is structurally infeasible on the tested competence axis — reported
   with full negative-result discipline (per-cell n=1, qualitative diagnosis, no
   effect estimates claimed).
3. **A construct** — the *memory-value window* — that organizes the floor/mid/ceiling
   pattern and connects it to prior unexplained observations (BASM's App.-Skill dip),
   yielding concrete design guidance for future memory evaluations.

---

## 2. Related Work

We audited eight direct competitors by reading their experimental sections in full
(2026-10-21; four dimensions: memory object D1, paired validity manipulation D2,
cost-side measurement D3, representation intervention D4). Table 1 summarizes.

**Table 1 — Eight audited competitors (D2 paired validity manipulation: all absent).**

| # | Work | D1 memory object | D2 paired validity | D3 cost side | D4 representation |
|---|---|---|---|---|---|
| 1 | Negative Knowledge [2606.21024] | synthesized typed failure records; adopt/reject | none (one observed negative-transfer case) | none (token cost only) | none (296/795/1109-token forms confounded) |
| 2 | Experience-Following [2505.16067, ACL 2026] | query–execution pairs (success/failure not separated) | none (distribution shift via task reordering) | indirect only | none (intervention at storage admission) |
| 3 | MeClear [2609.09115] | stale/misleading retrieved records | none | over-selection of harmless records ≠ wrong abandonment | none (query-level suppression) |
| 4 | Remember When It Matters [2607.08716] | status/knowledge/procedural | none | indirect (Injection-only −6.0pp, unconceptualized) | none (injection timing) |
| 5 | Useful Memories Become Faulty [2605.12978] | episodic traces + consolidated abstractions | none (erosion curves observed) | utility below baseline ≠ wrong abandonment | partial (abstraction level as control) |
| 6 | **BASM [2608.22339, EMNLP 2026 Findings]** | **success skills** (failure trajectories only as boundary evidence) | **none** ("inapplicable states" constructed across tasks) | none systematic; App.-Skill dip unmeasured as cost | **full** (4-level + equal-length padding + equal-token pairing + attention knockout) |
| 7 | MCMA [2026.findings-acl.1535, ACL 2026 Findings] | tree-structured knowledge; Sum + failure-Reflection copilots | none (seen/unseen splits) | none (knowledge-count non-monotonicity) | partial (structure type / abstraction level) |
| 8 | Materials-science partner [2608.11224] | facts incl. boundary knowledge; skills with conditions/guardrails | none (within-family generalization) | partial (token cost; harmful cross-model transplant) | none (conditioning is format default) |

**The seam.** The conjunction — failure experience *itself* as memory object (D1) +
within-family paired validity manipulation (D2) + benefit–cost dual measurement (D3) —
is occupied by none of the eight. BASM is the strongest neighbor on D4: its boundary
fields are a causally validated anti-imitation channel (same-prompt knockout recovers
69.4% of the imitation-trap effect; wrong-tool margin 7.31 → 10.71 → 2.26 across
memory-free / procedure-skill / BASM), and it reports AppWorld +23.8pp, BFCL +5.0pp,
AgentDojo attack-success −4.6pp. But its T_wrong cell encodes "should suppress"; it
never constructs the complementary "should retry" cell that a revived path requires,
and it measures no cost side. Notably, BASM's own intermediate configuration — adding
*only* the applicability-condition field to procedure skills ("App. Skill") — is
non-monotonic and model-dependent (AppWorld-Qwen3-32B 66.07→60.12, −5.95pp;
BFCL-Qwen3.5-397B 53.38→51.63, −1.75pp; but AppWorld-Qwen3-14B 46.43→53.57, +7.14pp),
a phenomenon its authors neither measure as a cost nor explain. Our results
(Section 6) provide a candidate explanation frame for exactly this pattern.

Adjacent signals that conditioning information can *hurt*: MCMA disables its
success-summary copilot on ScienceWorld because success summaries are harmful there;
[2607.08716] reports Injection-only at −6.0pp on one environment; [2608.11224]
reports memory transplantation that can be neutral-to-harmful when source memory is
weaker than the target's knowledge. None treats these as the phenomenon under study.

Motivation-level work [2607.24759] argues for retaining dead paths in append-only
wikis; it is a design + case study with no controlled measurement.

**Positioning.** This is a *measurement* paper, not a method proposal: conditional
representation of memories is BASM's contribution; our claims concern what such
representation does to the benefit–cost trade-off of *failure* memories when
environmental validity flips — including the cost side (wrong abandonment) nobody
measures — and, after the falsification stage, where that manipulation's preconditions
break.

---

## 3. Problem Formalization

### 3.1 Design

**2×4 within-task-family design** (pre-registered, frozen v2.3/v2.4 before Stage-A
data):

- **Validity variant (within family, paired)**:
  - *V-dead*: environment unchanged; the old failed path X is still broken → avoiding
    X is correct; the failure memory should *help*.
  - *V-revived*: the disqualifying condition is removed (library upgrade / data
    regeneration / schema fix) → X is now correct; avoiding it is an opportunity
    loss; the failure memory should *harm* unless conditioned.
  - Construction requirement (pre-registered): X passes checkpoints in V-revived and
    genuinely fails in V-dead; variants match item-by-item except the manipulated
    condition.
- **Memory representation (fact-matched templates)**: NONE (no memory) / UNCOND
  (full failure facts: path + cause + evidence + consequence; no scope statement) /
  COND (same facts + explicit applicability condition and invalidating change) /
  COND-MINUS (equal length, same facts, applicability replaced by non-diagnostic
  content). Templates share identical failure facts; only the scope statement varies.
- **Primary outcome (H-A, pre-registered)**: task success rate, with directional
  expectations per variant (UNCOND vs NONE: ↑ in V-dead, ↓ in V-revived).
  **H-B**: COND vs UNCOND mitigates damage in V-revived and is non-inferior in V-dead
  (δ frozen before confirmatory stage). **H-C**: COND vs COND-MINUS isolates the
  conditional information beyond length/genericity (cannot exclude "more task-relevant
  information" — stated up front).

### 3.2 Behavioral metrics

Three layers, all logged per run: **L1 usage** (memory injected / retrieved /
explicitly cited), **L2 behavior** (method-choice sequence; under the method-menu
harness this becomes the explicit `methods_chosen` sequence; naive-path avoidance and
explicit abandonment are directly readable), **L3 outcome** (frozen deterministic
implementations' outputs vs. gold standards). L3-valid-but-L1-low and L1-high-but-L3-null
are separately interpretable.

---

## 4. Instrument

All artifacts are frozen in the repository (paths in Appendix A).

### 4.1 Task families

Three data-analysis families, each with a naive method X (the NONE-condition natural
default), a memory-suggested alternative Y (a project trap tool whose frozen
implementation is correct in V-dead and crashes in V-revived), and a gold standard:

| Family | Failure mode of X | X (naive) | Gold | X in V-dead | X in V-revived | Y (trap tool) in V-dead | Y in V-revived |
|---|---|---|---|---|---|---|---|
| A — missing data | MAR by (group, z2); deletion biases the group contrast | `dropna` + group means | Δ = 2.1 | Δ = 8.105 (biased, +6.0pp) | Δ = 1.882 (unbiased) | mi_tool: 2.031 ✓ | assertion crash |
| B — timestamp parsing | 12% unparseable timestamps silently dropped by month-grouped sum | month `groupby` on coerced ts | total = 500,000 | 440,000 (biased) | 500,000 ✓ | legacy_tool: 500,000 ✓ | column-position crash |
| C — SQL join fanout | duplicated customer rows fan out JOIN; Y additionally merges distinct customers by name case | direct JOIN aggregated by id | (50,000; 992 customers) | (59,280; 991) | (50,000; 992) ✓ | dedup_by_name: (50,000; 991) ✓ | (50,000; **904**) — silent customer-loss trap |

Family C's V-revived implements the subtle case: after the uniqueness fix, the naive
join is fully correct, but the memory's *remedy* (name-case deduplication) now merges
88 distinct customers whose names differ only by case — the Y trap. X-in-V-revived
correctness is verified against the gold before any agent run.

### 4.2 Frozen memory templates and contamination control

Nine record templates (3 families × UNCOND/COND/COND-MINUS), fact-matched per family
(identical path/cause/evidence/consequence; only the scope statement differs;
COND-MINUS equalized in length with neutral archival filler). A five-part
contamination check verifies: (i) no gold *absolute* value in any template (relative
deviations allowed; stricter than the original spec, resolving an internal conflict in
the task-set document in favor of anti-leakage); (ii) COND-MINUS semantic distance
from COND < 0.5 cosine (bge-m3, 1024-d; measured 0.487); (iii) UNCOND/COND/COND-MINUS
length parity within 10%; (iv) template consistency across families; (v) task
descriptions contain no leak words. All checks green before any agent run.

### 4.3 Memory layer and retrieval

A purpose-built memory layer (dense retrieval over frozen records with bge-m3,
top-k=3, hit threshold 0.30) injects the family-appropriate template at session
start; injection and retrieval events are logged (L1). The rule-based extractor of the
project's memory engine does not emit FAIL records, hence the explicit layer.

### 4.4 Agent harness evolution (v1 → v3)

- **v1 free-code ReAct**: prompt-based ACTION/OBSERVATION loop with sandboxed
  `run_python`/`run_sql`/`read_file`. Guards: tool-result messages must carry
  `tool_call_id` (native-tools mode); a FINAL is rejected if no compute tool was
  called or placeholders are present (up to 2 nudges). Fixed an execution-order bug
  where models narrate an entire ReAct trace as text and a FINAL-priority parser
  never executes it.
- **v3 method menu**: the agent only emits `METHOD: <name>` (frozen implementation
  executes; OBSERVATION returned) or `FINAL:`; budget = 1 method execution
  (single commitment); a memory-usage rule ("if memory marks a method biased, do not
  adopt it unless you verify its conditions do not hold") is rendered *only* when
  memory exists — so NONE is unpolluted. `methods_chosen` is the L2 metric.

The menu freezes method implementations, removing code-quality confounds across
conditions and making "implicit NaN skipping" bypasses impossible.

### 4.5 Pre-run gates

| Gate | Content | Status |
|---|---|---|
| V1 data reproduction | simulators reproduce designed golds (A: 8.105/2.1; B: 440k/500k; C: 59,280/50,000) | pass |
| V2 revival | deletion-based methods unbiased in V-revived | pass |
| V3 trap tools | Y correct in V-dead, crashes in V-revived (A, B); C's Y-trap verified | pass |
| V4 templates | contamination checks all green | pass |
| V5 family-B exploration | v1/v2 asymmetries verified end-to-end | pass |
| V6 natural default | NONE-condition agents first-choose X | pass (qwen2.5:7b, 2/2; Δ=8.105 with N=1440/2000) |
| V7 UNCOND implied-condition pretest | can models infer the condition from UNCOND facts? | **not reached** (stop-loss preceded it) |

---

## 5. Stage-A Procedure

Stage A was pre-registered as a falsification pilot: estimate variance, verify the
three preconditions (directional reversal signal; measurable representation effect
under fact-matched templates; L2/L3 separability), *not* to produce paper-grade
claims. Family A was run first (the family with the cleanest statistical contrast);
families B/C instrument work is complete and retained. All runs executed locally
(Ollama) or via an Anthropic-protocol cloud endpoint (DeepSeek-V4.1-Flash); every run
writes a JSON with condition tags, `methods_chosen`, tool calls, and the final answer.

---

## 6. Results

### 6.1 Confirmatory pilot (family A, 2×4, method-menu harness, Qwen2.5-7B)

**G-core failed.** All eight cells chose the naive method X
(`naive_delete`), including UNCOND/V-dead where the memory explicitly quantifies the
bias. Memory-hit L1 signals were positive (memory retrieved and injected), and final
answers *quoted* the memory's content while committing to X. No directional reversal
is measurable when the behavioral metric does not move.

### 6.2 Falsification ladder [exploratory — adaptive variants, n=1 per cell]

Because the confirmatory pilot failed, we ran a pre-registered-in-spirit ladder of
harness variants to determine whether the failure was instrumental (fixable) or
structural. Each variant was run only after the previous one failed; variants differ
in prompts and must not be pooled.

| # | Harness variant | Model | Outcome |
|---|---|---|---|
| 1 | Free-code ReAct | Qwen2.5-7B / Gemma3-12B | rationalized bypass: after the instruction "never use dropna on this data," Gemma3 computed column means directly (pandas `mean` implicitly skips NaN = same deletion estimator), obtained 8.105, and reported "I did not use dropna" |
| 2 | Menu + multi-attempt | Qwen2.5-7B | never switches (4/4 cells choose X) |
| 2′ | Menu + multi-attempt | Gemma3-12B | explores all three methods and self-arbitrates to the correct answer; memory adds nothing |
| 3 | Menu + single commitment + memory-usage rule | Gemma3-12B | 8/8 cells choose X; final answers restate memory content ("bias ≈ 6.0pp", "48% missing in B") yet commit to the biased method citing budget; in V-revived it *fabricates* applicability ("missing pattern similar") to keep X |
| 4 | Decision-point binding ("before emitting METHOD, check memory; if marked biased you MUST choose another") | Gemma3-12B | still 4/4 choose X |
| 5 | Memory relocated to first user message (salience) | Gemma3-12B | no effect (4/4 choose X) |
| 6 | v3 harness + cloud strong model | DeepSeek-V4.1-Flash | 8/8 cells choose the optimal method `mi_tool` — *including NONE*; in V-revived all cells crash on the tool's assertion and the agent honestly reports "no estimate possible" (including COND, without using the applicability condition prospectively) |

### 6.3 The three-tier diagnosis

| Tier | Model | NONE first choice | Behavior under memory conditions |
|---|---|---|---|
| floor | Qwen2.5-7B | X | acknowledge-but-do-not-act across all harnesses; reinterprets memory ("bias relatively small") or fabricates applicability |
| mid | Gemma3-12B | X | identical choice pattern to floor, with articulated rationalization |
| ceiling | DeepSeek-V4.1-Flash | optimal tool | prior statistical knowledge (deletion bias under MAR) covers the memory's information value; honest failure reporting in V-revived; no prospective use of COND |

**Finding (the memory-value window).** The manipulation requires a model that (i)
does not already know X is biased and (ii) will act on a memory instruction. On the
observed axis these requirements are mutually exclusive: floor/mid models satisfy
(i) but not (ii); the ceiling model satisfies (ii) but not (i). The window is empty.
Given the ladder exhausted prompt salience, commitment structure, decision-point
binding, and model scale, we conclude the failure is **structural to method-choice
manipulations over textbook statistics**, not an artifact of any single harness.

**Secondary observation.** Applicability-condition *discrimination* (in V-revived,
inferring from COND that X is now safe) never occurred spontaneously at any tier.
The closest behavior is the ceiling model's honest crash reporting in V-revived —
post-hoc honesty without pre-decision use of the condition. With respect to the
frozen RQ ("can agents distinguish still-valid failure lessons from expired historical
taboos?"), the measured answer on this task type and model set is **no**.

### 6.4 Evidence strength [honest accounting]

- All decisive cells are n=1 (one stability re-run: UNCOND/COND × V-dead × 2
  repeats, both X). The diagnosis is qualitative and direction-consistent across
  variants; it supports no effect-size estimates.
- Harness variants differ in prompts; no pooling or meta-analysis is legitimate.
- The scoring package was intentionally not built: with no directional signal in the
  behavioral metric, L3 scoring could not change the conclusion (the L2 metric is
  degenerate at ceiling/floor). This is the stop-loss point.
- All runs, logs, and the negative results are retained (Appendix A); intermediate
  ladder-run JSONs were overwritten by later variants sharing result tags — the
  per-variant outcomes above are documented in the harness-change record, and final
  2×4 matrices (Qwen and DeepSeek) are preserved as JSONs.

---

## 7. Discussion

**Consilience with BASM's unexplained dip.** BASM reports that adding only the
applicability field to procedure skills *helps* a 14B model (+7.14pp) but *hurts* a
32B model (−5.95pp). Our construct predicts exactly this sign pattern: for a model
that cannot independently evaluate applicability (i), the condition is load-bearing
information; for a model whose weights already encode the boundary, an extra
conditional clause is at best redundant and at worst an invitation to rationalized
suppression of viable actions. Our ladder shows the stronger mechanism directly:
mid-tier models do not merely ignore conditions — they *actively reinterpret* memory
to license their default choice. The "window" gives the failure-memory program a
falsifiable placement rule: target information absent from weights.

**Design guidance for future validity manipulations.**

1. *Information selection*: the memory's content must be unavailable to the model a
   priori — project-private parameters, environment-local state, session-scoped
   identifiers — not general statistical knowledge. (Our stop-loss decision tree
   listed this as the "private-information domain" rebuild path.)
2. *Conflict with procedure, not with knowledge*: an SOP that prescribes the naive
   method while memory contradicts it measures memory-vs-procedure compliance — a
   different construct from applicability reasoning (retained as an alternative
   design, not mixed into this report's claims).
3. *Prospective condition use must be elicited structurally* (e.g., requiring a
   condition check as a harness step) — but note this converts the construct from
   spontaneous discrimination to instruction-following, which must be reported as
   such.

**What the instrument is good for.** The families, templates, trap tools, gates, and
menu harness are released and domain-reusable: they implement a paired validity
manipulation with verified golds and contamination control — the D2/D3 seam no
audited competitor occupies. The negative result sharpens, rather than discards, the
seam: any future occupant must first clear the window problem.

---

## 8. Limitations

1. **Single task family in the decisive ladder** (A); B/C instruments verified but
   unexercised with agents. The structural diagnosis rests on A's statistical
   contrast being the most textbook-adjacent; a family whose bias mechanism is
   *less* standard could shift the ceiling tier's NONE behavior.
2. **n=1 per cell, two local + one cloud model, one prompt family per harness
   variant.** The claim is a qualitative existence result ("no model in the window
   was found"), not an absence proof.
3. **Template recall vs. realistic memory systems**: memories were injected
   verbatim with dense-retrieval logging; no consolidation noise. Findings concern
   memory *use*, not memory *formation*.
4. **The ceiling model's "prior knowledge" is inferred behaviorally** (uniform
   optimal choice including NONE), not by probing weights.
5. **L3 scoring not applied**; outcome-layer claims are limited to frozen-method
   outputs, which is sufficient for the behavioral diagnosis but leaves task-success
   rates unmeasured.
6. **Pre-registration scope**: H-A/H-B/H-C were pre-registered for the confirmatory
   2×4; the ladder is exploratory and labeled as such throughout.

---

## 9. Conclusion

We built and released the first instrument for paired validity manipulation of agent
failure memory, and used it to establish a boundary result: method-choice
manipulations of failure memory have an empty memory-value window on the tested
competence axis — weak models acknowledge failure lessons without acting on them;
strong models do not need them; and no tier spontaneously uses applicability
conditions to retire an expired taboo. The result organizes prior unexplained
observations (BASM's non-monotonic App.-Skill effect), delimits where memory
evaluations must aim (weight-absent information), and provides the community a
contamination-controlled, gate-verified harness for the next attempt.

**Artifacts** (repository, MIT-style per project convention): simulators and gold
standards (3 families), nine frozen record templates + contamination checker, trap
tools, method-menu agent + runner, cloud/local protocol clients, all run JSONs and
logs, and the full decision record (master design doc v2.4, task-set v1, harness
change log, stage-A closure report).

---

## References (as audited 2026-10-21; re-verify before submission)

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
    arXiv:2605.26667, 2026. (Paper-1 neighbor; cited for framing.)

---

## Appendix A — Evidence map (project-internal sources for every numeric claim)

| Claim | Source |
|---|---|
| Family golds & frozen-implementation outputs (A 8.105/2.031/1.882; B 440k/500k; C (59,280,991)/(50,000,992)/(50,000,904)) | `data/kwm_stage_a/gold_{a,b,c}.json`; `docs/KWM-HARNESS-CHANGE-2026-10-21.md` §变更内容.4 |
| Contamination check values (0.487 cosine; all green) | `experiments/kwm_contamination_check.py` output; task-set §8.1 conflict resolution |
| Gate V6 (NONE→X, Δ=8.105, N=1440/2000) | `results/kwm_stage_a/A_NONE_V_dead_r0.json` (qwen, final state) |
| Ladder rows 1–5 | `docs/KWM-HARNESS-CHANGE-2026-10-21.md`; `docs/KWM-STAGE-A-CLOSURE-2026-10-21.md` §2 |
| Three-tier 2×4 matrices (Qwen; DeepSeek 8/8 mi_tool; V-revived crash-honesty) | `results/kwm_stage_a/*.json` (10 runs, final state); harness-change appendix |
| BASM numbers (7.31/10.71/2.26; 69.4%; +23.8/−5.95/+7.14 etc.) | project audit `docs/KWM-MASTER-2026-10-20.md` §1.7 (full-text read 2026-10-21) |
| MCMA ScienceWorld Ref-only; 2607.08716 −6.0pp; 2608.11224 transplant risk | project audit §1.7 |
| Competitor D1–D4 table | project audit §1.7 four-dimension table |
| RQ and H-A/H-B/H-C wording (frozen) | `docs/KWM-MASTER-2026-10-20.md` §3.1–3.2; `docs/KWM-TASKSET-V1-2026-10-21.md` |

## Appendix B — Decision record (why the manipulation failed; retained alternatives)

1. Stop-loss chosen over SOP-conflict redesign (measures memory-vs-procedure
   compliance, a different construct) and over private-information domain rebuild
   (~1–2 days; retained as the recommended next attempt).
2. Cloud-model option was exercised (DeepSeek-V4.1-Flash) before stop-loss —
   the ceiling tier, not a rescue.
3. Scoring package deliberately unbuilt (no directional L2 signal → L3 could not
   alter conclusions).
