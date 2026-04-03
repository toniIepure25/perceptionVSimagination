# Paper 1 Appendix Plan

The appendix/supplement should hold the material that is essential for
reproducibility but too detailed for the main narrative.

## Appendix A. Dataset and overlap details

Include:
- subject list for the frozen benchmark
- row counts and overlap counts by subject
- prepared artifact paths
- explanation of why the current benchmark remains only `bootstrap_ready`

Primary sources:
- [TABLE_3_REPRODUCIBILITY_ARTIFACT_CONTRACT.md](tables/TABLE_3_REPRODUCIBILITY_ARTIFACT_CONTRACT.md)
- [CURRENT_EVIDENCE_FREEZE.md](/home/tonystark/Desktop/perceptionVSimagination/docs/CURRENT_EVIDENCE_FREEZE.md)

## Appendix B. Canonical configs and workflow surface

Include:
- `configs/canonical/max_available_overlap.yaml`
- `configs/canonical/animus_core_decoder.yaml`
- `configs/canonical/threshold_shared_private_p16.yaml`
- exact workflow commands for:
  - acquisition
  - overlap assembly
  - target cache
  - preflight
  - Ridge
  - shared-only
  - shared-private p16

## Appendix C. Asset-generation and figure provenance

Include:
- `scripts/paper1/build_paper1_assets.py`
- `docs/paper1/assets/paper1_source_data.json`
- note that figures are generated from frozen benchmark evidence, not hand-edited graphics

## Appendix D. Additional benchmark variants

Include:
- shared-private default
- shared-private `private_dim=8`
- shared-private no-domain

Purpose:
- show the within-family diagnostic controls
- keep the main text focused on the frozen ladder

## Appendix E. Export and subsystem notes

Include:
- practical Animus Core Decoder export contract
- stability tier / role metadata
- content/source/confidence interface readiness language

Purpose:
- keep the practical subsystem story explicit without bloating the main paper

## Appendix F. Evidence boundary and claim audit

Include:
- supported vs partially supported vs not-yet-justified claims
- explicit warning that Paper 1 is not a positive disentanglement paper

Primary sources:
- [PAPER_1_CLAIMS_MAP.md](/home/tonystark/Desktop/perceptionVSimagination/docs/PAPER_1_CLAIMS_MAP.md)
- [CURRENT_EVIDENCE_FREEZE.md](/home/tonystark/Desktop/perceptionVSimagination/docs/CURRENT_EVIDENCE_FREEZE.md)
