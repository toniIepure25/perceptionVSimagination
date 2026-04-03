---
name: research-scout
description: "Use when scouting the next research direction for this repository: classify ideas into the practical Animus Core Decoder lane, the shared-private threshold-research lane, or external data acquisition, then map the proposal against CURRENT_EVIDENCE_FREEZE, BENCHMARK_LADDER, TOP_LEVEL_RESEARCH_DOSSIER, THRESHOLD_HYPOTHESIS, and the current overlap ceiling."
---

# Research Scout

Use this skill before experiment design or execution when the user asks "what should we test next?", "is this a real paper direction?", or "does this help Animus or the threshold paper?"

## Inputs

- the proposed direction or idea
- any stated motivation
- whether the user is optimizing for Animus utility, paper progress, or benchmark expansion

## Read First

1. `docs/CURRENT_EVIDENCE_FREEZE.md`
2. `docs/BENCHMARK_LADDER.md`
3. `docs/TOP_LEVEL_RESEARCH_DOSSIER.md`
4. `docs/THRESHOLD_HYPOTHESIS.md`
5. `docs/PAPER_1_CLAIMS_MAP.md`
6. `docs/DATA_ACQUISITION_PROGRAM.md`
7. `docs/LIMITATIONS.md`
8. `docs/ANIMUS_CORE_DECODER.md` if the idea touches practical subsystem work

## Lane Classification

Classify every proposal into exactly one primary lane:

- `Animus subsystem engineering`
  Improve the shared-only practical decoder, export path, ROI/materialization pipeline, preflight, or downstream-ready metadata.
- `Threshold research`
  Test when shared-private begins to beat shared-only or Ridge on a materially larger paired benchmark.
- `Data acquisition`
  Increase overlapable paired data or bring in a clearly separate benchmark.
- `Out of scope`
  Ideas that depend on claims or data the repository explicitly does not have yet.

If a proposal touches more than one lane, state the primary lane and the dependency chain.

## What To Produce

Return a short decision memo with:

- the exact question being tested
- the lane classification
- why the idea matters now given the current ceiling of `94` rows / `5` shared paired `nsdId`s / `1` held-out paired group
- whether it is `supported`, `partially supported`, or `not justified yet` under `docs/PAPER_1_CLAIMS_MAP.md`
- what data would be required
- which fixed baseline(s) it must compare against
- what code/config/docs would likely move

## Hard Rules

- Treat shared-only as the practical Animus lane and shared-private as the exploratory hypothesis family unless new evidence genuinely changes that.
- Do not treat perception-only data as a direct answer to the paired threshold question.
- Do not propose benchmark-ladder changes before the fixed ladder is rerun on larger paired data.
- Do not sell tiny-overlap diagnostics as neuroscientific conclusions.
- If the idea needs subjective labels, say that the current repo does not justify vividness/confidence claims yet.

## Good Outcomes

Strong scouting output usually ends in one of these recommendations:

- strengthen the Animus Core Decoder without changing scientific claims
- rerun the fixed ladder on a larger paired source
- run a narrow diagnostic that explains current shared-private underperformance
- reject or postpone an idea because it exceeds the current evidence boundary

## Handoff

- hand off to `experiment-design` when an idea survives triage
- hand off to `paper-drafter` only when the task is framing existing evidence rather than generating new work

## Boundaries

This skill does not design the exact run matrix, execute experiments, or draft final paper prose. Hand off to:

- `experiment-design` for an executable plan
- `ablation-runner` for controlled execution
- `paper-drafter` for manuscript-facing text
