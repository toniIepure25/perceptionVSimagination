# Mock NeurIPS 2026 E&D Review Pack

Paper: *Benchmarking Paired fMRI Perception–Imagery Decoding Under Overlap
Scarcity*

---

## Review A — Favorable but rigorous E&D reviewer

**Summary.**
This paper contributes a frozen benchmark and comparison ladder for paired
perception–imagery fMRI decoding under severe overlap scarcity (94 rows, 5
shared paired stimulus IDs, 1 held-out group, 4 NSD subjects). The main finding
is a deliberately qualified negative result: Ridge regression dominates all
neural decoders, shared-only is the strongest neural baseline, and
shared-private disentanglement variants remain below shared-only. The paper
frames this as an Evaluations and Datasets contribution centered on claim
discipline rather than model novelty.

**Strengths.**

1. The paper is unusually honest for a benchmark submission. The claim-boundary
   table (Table 2) is a model of how E&D papers should separate supported,
   partially supported, and unjustified conclusions. I have not seen this done
   this cleanly in prior neuro-AI benchmark papers.
2. The pre-declaration of evidential roles (reference / canonical / exploratory)
   before interpretation is a methodologically sound choice that strengthens
   the negative finding. It prevents post-hoc role reassignment.
3. The writing is precise and non-promotional. The epistemic discipline is
   high—sentences like "this distinction—between being unjustified on available
   evidence versus being wrong in general—is precisely the kind of claim
   boundary an E&D paper should make explicit" are exactly the right tone.
4. The reproducibility contract in Appendix A is specific enough to be useful
   (exact configs, commands, artifact paths).
5. Figure 1 cleanly visualizes the enormous Ridge–neural gap, which is itself
   an important finding for the paired decoding community.

**Weaknesses.**

1. The benchmark is very small. 94 rows and 1 held-out paired group are below
   the paper's own preflight threshold for paper-readiness (≥32 paired groups).
   While the authors acknowledge this, a skeptical reader could argue that a
   benchmark that fails its own readiness check is premature for NeurIPS.
2. No code or artifact release is provided. For an E&D submission whose core
   claim is reproducibility, this is a meaningful gap. The reproducibility
   contract in Appendix A describes what *would* be released but does not
   provide it.
3. There are no error bars, repeated trials, or seed sweeps. The ordering is a
   single frozen snapshot. While the authors are transparent about this, it
   means the ordering could in principle be unstable.
4. The Related Work section is thin (10 references, ~0.5 page). Recent work on
   fMRI-to-image reconstruction, cross-subject alignment, and multi-modal
   benchmark design is not discussed. This makes the paper's positioning feel
   incomplete.
5. The model diversity is limited. Only Ridge and one family of neural
   decoders are compared. Including even one additional baseline (e.g., MLP,
   SVM, or a simpler factorized model) would strengthen the ladder.

**Main questions for the authors.**

- Q1: Why should the community trust a benchmark that is below its own
  paper-readiness threshold? What is the principled argument for publishing a
  benchmark that the preflight policy classifies as underpowered?
- Q2: Will code and benchmark artifacts be released upon acceptance? If not,
  what is the path to a reusable artifact?
- Q3: Have you run even a small seed sweep (e.g., 3 seeds) on the shared-only
  model to check whether the Ridge > shared-only ordering is stable?

**Likely score range:** 5–6 (borderline accept, leaning positive).

**Confidence:** 4/5. I am familiar with fMRI decoding and ML evaluation
methodology.

**What would most likely move the score upward:**
An anonymized code-and-artifact release, or even a credible commitment to
release, would raise this to a 6. A small seed sweep (even 3 runs of
shared-only) demonstrating ordering stability would raise it further.

---

## Review B — Borderline skeptical reviewer

**Summary.**
The authors present a small paired perception–imagery fMRI benchmark (94 rows,
4 subjects) and report that Ridge regression outperforms both shared-only and
shared-private neural decoders. They frame the negative result as a contribution
to evaluation methodology and claim discipline. The paper is well-written and
honest but the core question is whether the evaluation surface is substantial
enough to justify a NeurIPS E&D publication.

**Strengths.**

1. The framing is mature. The paper does not oversell and is transparent about
   its limitations.
2. The claim-boundary table is a useful contribution to E&D methodology. If
   more papers adopted this format, the field would be healthier.
3. The pre-declared evidential roles are a good methodological choice.
4. The paper clearly documents what cannot be concluded, which is valuable.

**Weaknesses.**

1. The E&D novelty is unclear. The paper freezes a benchmark and reports
   results—this is what every benchmark paper does. What is specifically new
   about the evaluation methodology? Pre-declaring roles is good practice but
   not itself a novel contribution. The claim-boundary table is useful but is
   it enough to carry a NeurIPS paper?
2. The benchmark is extremely small. With 94 rows and 1 held-out group, the
   evaluation surface barely functions. The paper acknowledges this but does
   not adequately explain why *this* scale, rather than waiting for more data,
   is the right time to publish.
3. The results feel predetermined by the data size. When you have 94 rows and
   5 paired IDs, *of course* Ridge will beat complex neural models. The finding
   that "simple beats complex under extreme data scarcity" is not surprising and
   has been documented in other low-data regimes. What insight does this add
   beyond confirming an expected pattern?
4. The paper does not release a dataset, a model, a benchmark package, or code.
   For the E&D track, the question "what artifact does this leave behind?" has
   no concrete answer.
5. The Ridge cosine of 0.552 vs neural cosines of 0.06–0.14 suggests that the
   neural models are essentially non-functional at this scale. This makes the
   neural-model comparisons less informative than the paper implies.

**Main questions for the authors.**

- Q1: What does this paper contribute that could not be written as a 4-page
  workshop paper? What justifies the length and venue?
- Q2: "Simple beats complex under data scarcity" has been observed many times.
  What is specific to the perception–imagery paired setting that makes this
  finding non-trivial?
- Q3: The paper argues that the benchmark prevents overclaiming. Can you give
  a concrete example of a claim in prior work that this benchmark would have
  caught?
- Q4: What exactly would a future researcher *do* with this benchmark that they
  cannot do by simply running their own comparison on NSD?

**Likely score range:** 4–5 (borderline reject, could be moved).

**Confidence:** 3/5. I am moderately familiar with fMRI decoding but less so
with the paired perception–imagery setting specifically.

**What would most likely move the score upward:**
A concrete, anonymized benchmark release package (even a minimal one) with a
clear protocol for future researchers to plug in new models would raise this to
a 5 or 6. A more convincing "why now" argument—ideally referencing a specific
overclaiming failure mode that the benchmark catches—would help.

---

## Review C — Harsh reviewer

**Summary.**
This paper reports that Ridge regression outperforms neural decoders on a very
small fMRI benchmark (94 rows, 5 paired IDs). The authors frame this as a
benchmark and evaluation contribution rather than a methods paper. I find this
framing unconvincing. The benchmark is too small to be useful, the findings are
unsurprising, and the paper does not release any reusable artifacts.

**Strengths.**

1. The paper is clearly written and well-organized.
2. The authors are transparent about the limitations of their benchmark.
3. The claim-boundary table (Table 2) is a nice organizational device.

**Weaknesses.**

1. **Fatal: the benchmark is far too small.** 94 rows and 1 held-out paired
   group is not a benchmark—it is a pilot study. The paper's own preflight
   threshold requires ≥32 paired groups for paper-readiness; the benchmark has
   1. Publishing a benchmark that fails its own readiness criterion is
   contradictory. The authors should acquire more data before submitting.
2. **Fatal: no released artifacts.** An E&D paper that contributes a
   "benchmark" but releases no dataset, no code, no evaluation package, and no
   model checkpoints is not an E&D contribution. It is a report.
3. **The finding is trivial.** Ridge beating complex neural models on 94 rows
   is not informative. This is a well-known pattern in low-data regimes across
   many domains. The paper does not demonstrate that anything is specific to the
   perception–imagery setting.
4. **This reads like a methods paper that did not work.** The shared-private
   model family is clearly the intended scientific contribution, and it
   underperforms. Reframing this as a "benchmark contribution" does not change
   the underlying fact that the method was not successful.
5. **No statistical evidence.** Single runs, no error bars, no significance
   tests. The entire ordering could be noise.
6. The model comparison is narrow: only Ridge and variants of one neural
   architecture. Other strong baselines (MLP, SVM, PLS regression, etc.) are
   absent.

**Main questions for the authors.**

- Q1: If this benchmark fails its own readiness criterion, why should the
  community accept it?
- Q2: How is this different from a negative workshop paper?
- Q3: Can you demonstrate that the ordering is stable across seeds?
- Q4: What prevents someone from simply downloading NSD, running Ridge, and
  observing the same result in an afternoon?

**Likely score range:** 3–4 (reject).

**Confidence:** 4/5. Familiar with both ML evaluation and fMRI decoding.

**What would most likely move the score upward:**
Almost nothing short of (a) a substantially larger benchmark, (b) a released
artifact package, and (c) error bars. This paper is premature.

---

## Meta-review summary

**Score range across reviewers:** 3–6.

**Consensus strengths:** Honest, well-written, and methodologically careful.
The claim-boundary table and pre-declared evidential roles are appreciated by
all reviewers. The epistemic discipline is unusual and valued.

**Consensus weaknesses:** The benchmark is very small. No artifact release. No
error bars. The E&D novelty beyond "freezing a benchmark" is not sufficiently
articulated. The "simple beats complex" finding is not surprising in isolation.

**Key disagreement:** Reviewer A sees the claim-boundary methodology and
evaluation discipline as sufficient for E&D publication, even with a small
benchmark. Reviewer C sees the small scale and missing artifacts as fatal.
Reviewer B could go either way depending on the rebuttal.

**Most likely outcome:** Borderline. The paper would need a strong rebuttal to
survive. The following would most improve acceptance odds:

1. A concrete (even minimal) anonymized artifact release commitment.
2. A stronger "why now" argument tied to specific overclaiming risks.
3. A seed sweep or stability check (even small) to demonstrate ordering
   robustness.
4. Sharper articulation of what is novel about the evaluation methodology
   beyond freezing a standard benchmark.

**AC recommendation:** The paper is on the borderline. Recommend discussion.
The claim-discipline contribution is genuine but the scale and missing artifacts
are real weaknesses. If the authors can address artifacts and stability in
rebuttal, this is a borderline accept. Otherwise, suggest resubmission after
acquiring larger paired data or preparing a release package.

---

## Top 10 Acceptance Risks (ranked by severity)

### 1. Benchmark too small — FATAL without new data
The benchmark has 94 rows, 5 paired IDs, and 1 held-out group. The paper's own
preflight threshold requires ≥32 paired groups. This is the single largest
risk. **Cannot be fixed by writing alone.**

### 2. ~~No artifact release~~ — MITIGATED (was near-fatal)
An anonymized supplementary package now accompanies the submission, containing
exact YAML configs for every ladder rung, a machine-readable artifact manifest,
step-by-step reproduction instructions with commands for all 6 models, an
external-asset inventory with licenses, and a claim-to-artifact provenance
table.  Full source code is committed for release upon acceptance.  **This risk
is materially reduced but not eliminated**: configs are not runnable code, and
the codebase itself is not bundled in the anonymous submission.

### 3. No error bars or seed sweeps — SERIOUS
Single runs without repeated trials make the ordering a snapshot rather than a
stable finding. A reviewer can reasonably question whether the results are
noise. **Fixable by running a small seed sweep (e.g., 3 seeds on shared-only
and SP p16).**

### 4. Perceived as failed method paper — SERIOUS, MANAGEABLE BY WRITING
Reviewers may read the paper as "shared-private didn't work, so we called it a
benchmark paper." The framing must be strong enough to defeat this reading.
**Largely manageable by writing; the current draft is already good but could
be sharper.**

### 5. E&D novelty unclear — MODERATE, MANAGEABLE BY WRITING
"Freezing a benchmark" is not novel per se. The paper needs to articulate what
is specifically new: claim-boundary methodology, pre-declared roles, evaluation
under support scarcity. **Fully fixable by writing.**

### 6. "Too obvious" finding — MODERATE, MANAGEABLE BY WRITING
"Simple beats complex under extreme scarcity" is a known pattern. The paper
needs to explain what is specifically non-trivial about this finding in the
paired perception–imagery context. **Fixable by writing.**

### 7. Ridge dominance makes neural comparison seem uninformative — MODERATE
The 4× gap (0.552 vs 0.136) could read as "neural decoders don't work at this
scale" rather than "this is an interesting benchmark." **Partially manageable by
framing the gap itself as informative.**

### 8. Limited model diversity — MODERATE, NOT FIXABLE BY WRITING
Only Ridge + one neural family. Other baselines (MLP, PLS, SVM) would
strengthen the ladder. **Not fixable without new experiments.**

### 9. ~~Thin Related Work~~ — FIXED
Related Work now includes 17 references covering recent fMRI reconstruction
(MindEye2, Brain-Diffuser, Tang et al.), benchmark methodology (HELM,
Dehghani et al.), and early deep reconstruction (Shen et al.), in addition to
the original perception--imagery and multi-subject literature.

### 10. Paper's own preflight says not ready — MODERATE, MANAGEABLE BY WRITING
The paper honestly reports that the benchmark is below its own readiness
threshold. A reviewer may cite this against the paper. Needs careful framing:
the gap between the threshold and reality is itself part of the finding.
**Manageable by writing, with care.**

### Risk classification summary

| Risk | Severity | Fixable by writing? |
|------|----------|---------------------|
| 1. Benchmark too small | Fatal | No |
| 2. ~~No artifact release~~ | ~~Near-fatal~~ Mitigated | Done (supplementary package) |
| 3. No error bars | Serious | No (requires runs) |
| 4. Perceived as failed method | Serious | Mostly yes |
| 5. E&D novelty unclear | Moderate | Yes |
| 6. Finding too obvious | Moderate | Yes |
| 7. Ridge dominance uninformative | Moderate | Partially |
| 8. Limited model diversity | Moderate | No |
| 9. ~~Thin Related Work~~ | ~~Minor~~ Fixed | Done (17 references) |
| 10. Own preflight says not ready | Moderate | Yes, with care |
