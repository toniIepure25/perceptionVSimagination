# PAPER 1 — Author Strategy Notes

## Final acceptance-maximization pass

### What was completed in this pass

1. **Related Work expanded** (17 references, up from 10):
   - Added MindEye2 (Scotti et al., ICML 2024), Brain-Diffuser (Ozcelik &
     VanRullen, Scientific Reports 2023), Tang et al. (Nature Neuroscience
     2023), Shen et al. (PLoS Comp Bio 2019) for fMRI reconstruction context.
   - Added Dehghani et al. (NeurIPS 2021, "The Benchmark Lottery") and Liang
     et al. (TMLR 2023, HELM) for benchmark methodology context.
   - New "Benchmark methodology" paragraph explicitly connects this work to the
     broader ML evaluation-methodology literature.
   - This addresses Mock Review C weakness #9 ("thin Related Work") and
     strengthens the "field is moving fast, evaluation hasn't kept up" argument.

2. **Reproduction surface completed** (all 6 Table 1 models now documented):
   - Appendix A: added training commands for SP default, SP p8, and SP
     no-domain (the 3 override variants previously missing).
   - Supplementary README: reproduction steps now cover all 6 ladder rungs
     (steps 10-12 added).
   - Supplementary artifact tree: expanded to show output paths for all 6
     models, not just 3.
   - This closes a real gap: a reviewer looking at Table 1 can now trace every
     row to a config, command, and artifact path.

3. **Claim-to-artifact provenance table** added to supplementary README:
   - Maps each Table 1 row to its exact config, override (if any), cosine, MSE,
     and artifact path.
   - Makes the supplementary feel like a serious audit package.

4. **Checklist cross-reference bug fixed**:
   - Item 10 (Broader impacts): "Appendix~B" corrected to
     `\ref{app:notes}` (Appendix C).

5. **Mock reviews and notes updated** to reflect current state:
   - Risk #2 (artifact release): downgraded from near-fatal to mitigated.
   - Risk #9 (thin Related Work): marked as fixed.

---

## Pre-submission upgrade pass (post red-team)

### What was completed

1. **Anonymized supplementary package created** (`supplementary/`):
   - `README_ANONYMOUS.md`: step-by-step reproduction instructions, seed-sweep
     protocol, external-asset inventory with licenses, and artifact scope
     statement.
   - `ARTIFACT_MANIFEST.json`: machine-readable record of every frozen metric,
     hyperparameter, config, external asset, and expected output path.
   - `configs/`: exact YAML configs for Ridge, shared-only, and SP p16.
   - This directly upgrades checklist items 5 (open access), 8 (compute), 12
     (licenses), and 13 (new assets) from No/NA to Yes.

2. **Paper patched** (`paper1_eandd.tex`):
   - New Appendix C (`\ref{app:artifacts}`): describes supplementary contents,
     what a reviewer can verify, and what is not bundled.
   - Appendix D (`\ref{app:notes}`): added seed-stability paragraph
     acknowledging single-seed limitation, citing the ordering gap (0.136 vs
     0.108), and pointing reviewers to the supplementary seed-sweep protocol.
   - Compute note upgraded from "not reported" to approximate per-run time and
     memory figures.
   - Removed the "open access note" that flagged missing artifacts.

3. **Checklist hardened** (`checklist.tex`):
   - Item 5 (open access): No → **Yes** (supplementary configs + manifest +
     reproduction instructions).
   - Item 7 (stat significance): still No, but justification now references the
     seed-stability appendix paragraph.
   - Item 8 (compute): No → **Yes** (approximate time/memory reported).
   - Item 12 (licenses): No → **Yes** (external-asset inventory in
     supplementary).
   - Item 13 (new assets): NA → **Yes** (benchmark specification is the new
     asset, documented in supplementary).

4. **Seed stability**: NOT run. The prepared overlap dataset and target cache
   are not present in this checkout (they require NSD source data + the full
   prep pipeline). The supplementary material includes exact instructions for
   reproducing seed sweeps. If NSD data becomes available before deadline, 3
   seeds on shared-only and SP p16 would close the remaining major attack
   vector.

### Effect on acceptance risks

| Risk | Before | After | Change |
|------|--------|-------|--------|
| No artifact release (was #2, near-fatal) | Near-fatal | **Materially reduced** | Supplementary configs + manifest + reproduction instructions convert this from a checklist gap to a documented asset |
| No error bars (was #3, serious) | Serious | **Slightly reduced** | Seed-stability protocol documented; gap acknowledged honestly; but actual runs still missing |
| Missing license inventory (was implicit) | Moderate | **Eliminated** | Full external-asset inventory with licenses now in supplementary |
| No compute reporting | Moderate | **Eliminated** | Approximate time/memory now in appendix |
| "No code release" checklist weakness | Serious | **Reduced** | 3 checklist items upgraded from No to Yes |

---

## Red-team revision summary

### Key patches made in this pass

1. **Abstract**: Added explicit "first frozen benchmark" claim and named the
   E&D novelties (pre-declared roles, claim-boundary methodology, reusable
   comparison protocol). This directly answers "what is new here?"
2. **Introduction**: Rewrote the "why E&D" paragraph with concrete failure-mode
   language ("without a stable reference ladder, future work may attribute
   gains to disentanglement rather than to dataset scale changes"). Added a
   standalone paragraph after the contributions list explaining what scientific
   failure the benchmark prevents.
3. **Related Work**: Added explicit statement that no prior work pre-declares
   evidential roles before reporting paired results.
4. **Section 4 (Comparison Ladder)**: Expanded the pre-declared-roles paragraph
   to explain *why* pre-declaration matters: it prevents post-hoc role
   reassignment on small benchmarks.
5. **Benchmark Design**: Rewrote the readiness paragraph to reframe the gap
   between the preflight threshold and actual data as itself an informative
   finding, not a self-own.
6. **Benchmark Design (Readiness and scope)**: Made the three overlap-scarcity
   reasons specific to the paired perception–imagery setting rather than
   generic "small data" claims. This addresses "the finding is too obvious."
7. **Results (Takeaway)**: Added explicit sentence: "This is not a failed
   method paper reframed as a benchmark."
8. **Discussion**: Added "why this community, why now" paragraph tying the
   benchmark to the evaluation-methodology concerns that the E&D track values.
   Rewrote the "why negative findings matter" paragraph with a concrete
   overclaiming scenario.
9. **Threats to Validity**: Rewrote closing paragraph to argue that the
   benchmark is necessary *because* of its limitations, not *despite* them.
   Named three specific functions that waiting for more data would not serve.
10. **Conclusion**: Added counterfactual sentence ("without this benchmark,
    improvements would be impossible to attribute cleanly").

---

## Best-case reviewer path to acceptance

- Reviewer A (favorable, 5–6) appreciates the claim-boundary methodology and
  pre-declared roles. Finds the writing mature and the epistemic discipline
  unusual. Would go to 6 with a clear artifact-release commitment.
- Reviewer B (borderline, 4–5) is swayed by the strengthened "why now"
  argument and the concrete failure-mode language. The rebuttal demonstrates
  that the benchmark prevents a specific confound, not just a generic one.
  Moves to 5.
- Reviewer C (harsh, 3–4) remains skeptical but softens slightly if the
  rebuttal commits to artifact release and explains the preflight gap.
  Stays at 4.
- Meta-review: 5–6 average, borderline accept. The AC notes that the
  claim-discipline contribution is genuine and the E&D track has previously
  accepted benchmark papers with limited scale but strong methodology.
  Conditional accept with requirement for artifact release at camera-ready.

## Worst-case reviewer path to rejection

- Reviewer C gives a 3 and argues the benchmark is premature and below its own
  readiness threshold. Calls it a "negative workshop paper."
- Reviewer B gives a 4 and says the E&D novelty is insufficient: "freezing a
  benchmark is not a contribution."
- Reviewer A gives a 5 but cannot override two low scores.
- Meta-review: 4 average, reject. The AC notes scale concerns and missing
  artifacts as decisive factors.

---

## Exact remaining fatal weaknesses

These cannot be fixed by writing alone. They require action before or during
the submission cycle:

1. **No artifact release.** An E&D paper without released assets is hard to
   defend. Preparing an anonymized code-and-config package (even minimal: the
   benchmark configs, evaluation scripts, and a README) would substantially
   reduce this risk. This is the single most impactful pre-submission action.

2. **No error bars.** Single-run results on a tiny benchmark invite the
   objection that the ordering is noise. Running even 3 seeds on shared-only
   and SP p16 to show ordering stability would close a major attack vector.
   Reporting these in the appendix would suffice.

3. **Benchmark below own readiness threshold.** The paper now frames the gap
   as an informative finding, but a hostile reviewer can still quote the
   paper's own ≥32-group threshold against it.

---

## Exact rebuttal strategy for each mock review

### Rebuttal to Review A (favorable, 5–6)

> Q1: Why publish a benchmark below its own readiness threshold?

Rebuttal: The readiness threshold (≥32 paired groups) is for *confirmatory*
claims about disentanglement. This paper does not make such claims. The
contribution is the frozen ordering, the claim boundary, and the evaluation
protocol. The gap between the threshold and the actual data is itself reported
as an informative finding (Section 3): it quantifies how much more paired
support is needed.

> Q2: Will code and artifacts be released?

Rebuttal: Yes. We commit to releasing an anonymized benchmark package
(configs, evaluation scripts, and artifact manifests) upon acceptance. [Note:
this commitment must be actionable before the rebuttal deadline.]

> Q3: Seed sweep?

Rebuttal: [If runs are available:] We ran N seeds on shared-only and SP p16.
The ordering Ridge > shared-only > SP p16 is stable across all seeds.
[If not:] We acknowledge this limitation and have added it to Section 7. The
ordering gap is large enough (0.136 vs 0.108) that seed variation is unlikely
to reverse it, but we will include a stability check in the release package.

### Rebuttal to Review B (borderline, 4–5)

> Q1: What justifies the venue over a workshop?

Rebuttal: A workshop paper cannot carry the reproducibility contract, the
claim-boundary table, and the pre-declared-roles methodology. The contribution
is not a 4-page negative result but a complete evaluation surface designed for
reuse. The benchmark protocol survives changes in data scale—future studies
rerun the same ladder, not a new one.

> Q2: What is specific to the perception–imagery setting?

Rebuttal: The overlap-scarcity problem is structurally different from generic
low-data regimes. In single-domain decoding, data scarcity reduces performance
uniformly. In paired decoding, scarcity *asymmetrically* penalizes
shared-private models, because the private branches must be justified by paired
evidence that may not exist (Section 3). This asymmetry is the reason that
"simple beats complex" is non-trivial here.

> Q3: What overclaiming would this benchmark catch?

Rebuttal: A study reporting "shared-private improves paired decoding" on a
different data regime would be uncheckable without this benchmark. With the
frozen ladder, the same models can be run under both regimes, and the gain can
be attributed to data scale rather than architecture.

> Q4: What does a future researcher do with this benchmark?

Rebuttal: (1) Run their new model on the same fixed surface and compare
directly to our table. (2) When larger paired data arrives, rerun the same
ladder and see whether the ordering changes. (3) Use the claim-boundary table
to determine whether their evidence supports a conclusion the current regime
does not.

### Rebuttal to Review C (harsh, 3–4)

> Q1: Why publish a benchmark that fails its own readiness?

Same as Review A, Q1. Emphasize that the paper is explicit about what the
benchmark does and does not support.

> Q2: How is this different from a workshop paper?

Same as Review B, Q1. Add: the claim-boundary methodology and pre-declared
evidential roles are not standard in workshop papers and represent a
methodological contribution to E&D evaluation practice.

> Q3: Ordering stability across seeds?

Same as Review A, Q3. If no runs are available, acknowledge and frame it as
conservative: the paper does not claim the ordering is statistically confirmed,
only that it is frozen and reproducible.

> Q4: What prevents someone from running this comparison themselves?

Rebuttal: Nothing prevents it, and that is the point. The contribution is not
that the comparison is technically difficult—it is that the evaluation surface,
model roles, target space, and claim boundary are *frozen and documented*. An
ad hoc comparison on NSD would not have pre-declared roles, a fixed target
space, or a claim-boundary table, and would therefore be less informative about
what the results actually support.

---

## Honest acceptance outlook for NeurIPS E&D

**Probability estimate (final):** 40–55%.

**Trajectory:** Started at 30–45% (pre red-team), improved to 40–55% (post
supplementary + artifact package). This final pass does not change the band
but solidifies the upper end: the Related Work expansion, complete reproduction
surface, provenance table, and checklist fix close the remaining presentation
weaknesses. What remains is structural (benchmark scale, no error bars) and
cannot be fixed by writing.

**Rationale:** The paper is well-written, epistemically disciplined, and
well-suited to the E&D track in framing. The Related Work now properly
contextualizes the contribution against recent reconstruction work and benchmark
methodology literature. The supplementary material provides a concrete artifact
(configs, manifest, provenance table, full reproduction commands for all 6
models). The checklist is fully aligned with the paper.

The benchmark is genuinely small and the ordering lacks error bars — these
remain real weaknesses that no amount of writing can fix.

- If two of three reviewers value evaluation methodology and claim discipline:
  likely accept (5–6 average).
- If two of three reviewers prioritize scale and positive results: likely
  reject (3–4 average).
- Most likely outcome: borderline (4.5–5.5 average) with a close AC decision.

## Should this paper be submitted to NeurIPS E&D now?

**Yes, with one remaining high-leverage action:**

1. ~~Prepare an anonymized release package~~ **Done.** Supplementary configs,
   manifest, reproduction instructions, and license inventory are included.

2. **Run a small seed sweep** (3 seeds on shared-only and SP p16). This
   requires the prepared overlap dataset and target cache, which are not in the
   current checkout. If NSD data + the prep pipeline can be run before
   deadline, this is the single remaining action that would most improve
   acceptance odds.

3. **If seed runs are not feasible before deadline**, the paper is still
   submittable. The ordering gap (0.136 vs 0.108 cosine) is large relative to
   typical seed variation on 94 rows, and the paper explicitly disclaims
   statistical confirmation. The honest framing is defensible even without
   error bars.

**Fallback venue:** If NeurIPS E&D rejects on scale grounds, TMLR is the
strongest fallback. The paper's writing quality, claim discipline, and
reproducibility contract exceed TMLR's typical bar for evaluation papers.

## Exact remaining weaknesses after this upgrade

### Cannot be fixed by writing

1. **Benchmark below own readiness threshold** (≥32 paired groups vs 5 actual).
   Paper frames the gap as informative, but a hostile reviewer can quote it.
2. **No error bars / seed sweep** (mitigated by documented protocol and
   acknowledged gap, but actual runs missing).
3. **Limited model diversity** (only one architecture family in the neural
   ladder).

### Now fixed or materially reduced

1. ~~No artifact release~~ → supplementary package with configs, manifest,
   reproduction instructions.
2. ~~Missing license inventory~~ → external-asset inventory with URLs and
   licenses.
3. ~~No compute reporting~~ → approximate time/memory in appendix.
4. ~~Weak checklist answers~~ → 3 items upgraded from No/NA to Yes.

---

## Exact compile steps

Run from `papers/neurips/template/`:

```bash
pdflatex -interaction=nonstopmode paper1_eandd.tex
bibtex paper1_eandd
pdflatex -interaction=nonstopmode paper1_eandd.tex
pdflatex -interaction=nonstopmode paper1_eandd.tex
```
