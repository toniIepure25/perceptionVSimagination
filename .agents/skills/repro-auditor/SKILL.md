---
name: repro-auditor
description: "Use when auditing whether a config, run, artifact bundle, or manuscript claim in this repository is actually reproducible, including preflight readiness, canonical artifact presence, output-path compliance, fixed target space, checkpoint/config compatibility, and whether a result is smoke-only, bootstrap-ready, paper-ready, blocked, or merely documented without runnable artifacts."
---

# Repro Auditor

Use this skill when the question is “can we really rerun this?” rather than “what should we study?”.

## Inputs

- the run, config, output directory, or claim being audited
- the exact command or report being cited
- the expected readiness level if one was already claimed

## Read First

1. `docs/REPRODUCIBILITY.md`
2. `docs/VALIDATION.md`
3. `docs/CURRENT_STATE.md`
4. `docs/CURRENT_EVIDENCE_FREEZE.md`
5. `docs/PAPER_1_CLAIMS_MAP.md` if the audit touches paper or README language
6. Relevant config(s) under `configs/canonical/`
7. Relevant output directory or report

## Audit Checklist

1. Confirm the run belongs to the canonical workflow surface, not an unsupported legacy path.
2. Confirm target space is still `vit_l14_image_768` unless a target-change task explicitly says otherwise.
3. Check for the expected prep artifacts:
   - imagery/perception indices
   - `*.source_report.json`
   - `*.report.json`
   - mixed ROI-ready parquet
   - target cache
   - `preflight.json`
4. Check training/eval artifacts:
   - `config_snapshot.json`
   - `roi_summary.json`
   - `target_summary.json`
   - `train_history.json`
   - `best_decoder.pt`
   - eval/transfer metrics
   - export manifest if export is claimed
5. Verify the documented command matches the config and output path.
6. Verify readiness labels honestly: `smoke_only`, `bootstrap_ready`, `paper_ready`, or `blocked`.
7. Compare any manuscript or README claim against the current trust boundary.

## Classification

Use one of these outcomes:

- `reproducible`
- `operational only`
- `bootstrap-ready only`
- `blocked by missing artifacts`
- `legacy / non-canonical`
- `claim exceeds evidence`

## Repo-Specific Red Flags

- no real target cache for a claimed real-data run
- missing `nsdId` coverage or overlap reports
- paper-like language attached to a `bootstrap_ready` run
- shared-private described as validated despite the frozen ranking
- local workspace only containing perception indices with remote `beta_path` files
- documentation referring to outputs that do not exist or do not match the checked-in config

## What To Produce

Return a short audit memo with:

- audit target
- exact files/commands checked
- outcome classification
- missing or mismatched artifacts
- whether the claim is safe, overstated, or unrerunnable
- the shortest path to make it reproducible

## Handoff

- hand off to `paper-drafter` if wording needs to be corrected
- hand off to `experiment-design` if reproducibility failure reveals a missing plan rather than a missing artifact

## Boundaries

- Do not redesign the experiment here.
- Do not patch manuscript language unless the task explicitly includes writing; otherwise hand off to `paper-drafter`.
- Do not equate a passing smoke fixture with real-data validation.
