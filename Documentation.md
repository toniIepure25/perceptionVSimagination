# Workflow Notes

Use this file as the working journal for milestones, decisions, and follow-up
actions that arise during active repository work. It complements, but does not
replace, `docs/PROJECT_MASTER_LOG.md`.

## How this file should be used

- Capture local working decisions before they become permanent docs changes.
- Track what changed, why it changed, and what still needs follow-up.
- Record doc-routing decisions so engineering changes do not accidentally turn
  into scientific claims.
- Summarize the next concrete action for the user or the next Codex session.

## Logging rules

- Keep entries short and date-stamped.
- Prefer decisions and follow-ups over narrative transcripts.
- If a change becomes a real milestone, append a concise version to
  `docs/PROJECT_MASTER_LOG.md`.
- If the change affects evidence, benchmark ordering, or paper framing, cite
  the governing docs explicitly.

## Entry template

```md
## YYYY-MM-DD - <short title>

- Scope: engineering | experiment | paper | reproducibility
- Status: in_progress | completed | blocked
- Surfaces touched: <files, configs, workflows, docs>
- Validation: <command run or reason no validation applied>
- Decision: <what was decided and why>
- Claim boundary: <why this does or does not change scientific interpretation>
- Follow-up: <next concrete action>
```

## Decision routing

Use this routing table when deciding where a result belongs:

- Workflow surface or command contract:
  `START_HERE.md`, `docs/REPRODUCIBILITY.md`, `docs/VALIDATION.md`
- Architecture or data contract:
  `docs/ARCHITECTURE.md`, `docs/CURRENT_STATE.md`
- Experiment result or rerun:
  run report first, then `docs/REPRODUCIBILITY.md`
- Evidence boundary or benchmark ordering:
  `docs/CURRENT_EVIDENCE_FREEZE.md`, `docs/BENCHMARK_LADDER.md`,
  `docs/PAPER_1_CLAIMS_MAP.md`
- Manuscript framing:
  `docs/paper1/`, related `docs/PAPER_1_*`, then `README.md`

## Current standing reminders

- Ridge remains the strongest fixed-benchmark baseline.
- Shared-only remains the best current canonical neural baseline.
- Shared-private remains exploratory on the current paired benchmark.
- Animus-oriented subsystem work should improve the shared-only practical lane
  without overstating scientific conclusions.

## Latest entry

## 2026-04-04 - Workflow setup refinement

- Scope: engineering, experiment, paper, reproducibility
- Status: completed
- Surfaces touched: `AGENTS.md`, `PLANS.md`, `Documentation.md`,
  `.agents/skills/*/SKILL.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: `pytest tests/test_canonical_workflows.py -q` currently fails in
  this workspace because `torchvision` is not installed, so the canonical
  smoke/import path errors before reaching workflow prerequisite checks
- Decision: tightened the Codex workflow contract around the actual canonical
  configs, lanes, docs, and readiness labels already used by the repository
- Claim boundary: no evidence-facing claims changed; this was an operational
  workflow cleanup only
- Follow-up: add repo-specific checks for skill-doc frontmatter or workflow-doc
  consistency, and decide whether `torchvision` is an intended canonical test
  dependency or an import-path regression
