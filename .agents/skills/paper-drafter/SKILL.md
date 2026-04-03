---
name: paper-drafter
description: "Use when drafting or revising manuscript-facing text for this repository, especially docs/paper1 sections, claims summaries, benchmark descriptions, or README-level research framing that must stay aligned with CURRENT_EVIDENCE_FREEZE, PAPER_1_CLAIMS_MAP, BENCHMARK_LADDER, and the repo’s separation between practical Animus subsystem work and exploratory threshold-research claims."
---

# Paper Drafter

This skill writes the paper the repository can honestly support now, not the one it hopes to support later.

## Inputs

- the target manuscript surface
- the governing evidence docs
- any recent run reports or audits that already exist
- the intended audience or deliverable type

## Read First

1. `docs/CURRENT_EVIDENCE_FREEZE.md`
2. `docs/PAPER_1_CLAIMS_MAP.md`
3. `docs/PAPER_1_OUTLINE.md`
4. `docs/BENCHMARK_LADDER.md`
5. `docs/REPRODUCIBILITY.md`
6. `docs/VALIDATION.md`
7. Relevant files under `docs/paper1/`

## Default Framing

Write this project as:

- a reproducible ROI-first benchmark platform
- an honest low-overlap evidence paper
- a practical shared-only Animus subsystem plus an exploratory shared-private threshold lane

Do not write it as:

- “shared-private wins”
- a vividness/confidence paper
- a full decoded-experience paper
- a strong neuroscientific private-latent interpretation paper

## Required Facts To Preserve

- current fixed benchmark: `94` rows, `5` shared paired `nsdId`s, `1` held-out paired group
- ranking: Ridge > shared-only > shared-private `private_dim=16` > other shared-private variants
- shared-only is the best current canonical neural baseline
- shared-private remains exploratory
- the threshold hypothesis is motivated, not confirmed

## Drafting Rules

- Tie every strong claim to an evidence document already in the repo.
- Use “supports”, “suggests”, “motivates”, and “not yet justified” precisely.
- Keep engineering claims separate from empirical claims.
- Mention Animus only when discussing the practical shared-only subsystem, export surface, or downstream reuse.
- If a sentence sounds like “disentanglement helps” or “private latents are meaningful,” check it against `docs/PAPER_1_CLAIMS_MAP.md` before keeping it.

## Typical Outputs

- section rewrites under `docs/paper1/`
- updated benchmark descriptions
- safer claim language in `README.md` or top-level docs
- figure/table caption alignment
- manuscript summaries that distinguish validated, exploratory, and future-work statements

## Sync Points

`START_HERE.md` is the exhaustive command surface. `README.md` is the professional public narrative. Do not turn the README into a command dump.

If results changed upstream, sync the relevant manuscript assets:

- `docs/paper1/tables/TABLE_1_MAIN_RESULTS.md`
- `docs/paper1/tables/TABLE_2_CLAIMS_EVIDENCE_BOUNDARY.md`
- `docs/paper1/tables/TABLE_3_REPRODUCIBILITY_ARTIFACT_CONTRACT.md`
- `docs/PAPER_1_FIGURES_AND_TABLES.md`
- `docs/paper1/assets/paper1_source_data.json`

## Handoff

- hand off to `repro-auditor` if reproducibility support for a claim is unclear
- hand off to `experiment-design` if the text depends on evidence the repo does not yet have

## Boundaries

- Do not invent new evidence.
- Do not change benchmark ordering unless the evidence docs were updated first.
- Do not let public-facing prose outrun `CURRENT_EVIDENCE_FREEZE.md`.
- Do not use manuscript drafting to hide unresolved reproducibility gaps; hand off to `repro-auditor` when the artifact story is uncertain.
