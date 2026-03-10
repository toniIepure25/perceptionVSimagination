---
name: Research-Level Project Overhaul
overview: Elevate the project from a working codebase to a top-tier research repository by adding CI/CD, academic citation infrastructure, purging 20+ legacy docs from the old project, consolidating redundant scripts, rewriting the broken Makefile, and standardizing code quality.
todos:
  - id: research-infra
    content: Add CITATION.cff, .pre-commit-config.yaml, py.typed, and GitHub Actions CI
    status: completed
  - id: purge-legacy-docs
    content: Delete 22 legacy doc files, fix 6 stale USAGE_EXAMPLES references, rewrite docs/README.md
    status: completed
  - id: consolidate-scripts
    content: Convert 15 thin wrappers to pyproject.toml entry points, move 3 misplaced tests
    status: completed
  - id: rewrite-makefile
    content: Rewrite Makefile from 559 lines to ~150 with only working targets
    status: completed
  - id: code-quality
    content: Standardize logging (logger vs log), replace print() with logging in library modules
    status: completed
isProject: false
---

# Research-Level Project Overhaul

## Current State Assessment

The codebase has good bones (typed code, docstrings, 6 novel analysis directions, configs with inheritance) but carries significant baggage: **26 of 32 docs** are legacy from the old Brain-to-Image project, **15 of 26 scripts** are thin wrappers that duplicate src/ modules, the **Makefile has 8+ broken targets**, and there is **no CI, no CITATION.cff, no pre-commit config, and no py.typed marker** -- all standard for top research repos.

---

## Phase 1: Research Infrastructure

Add files that every top-tier research repository has.

### 1a. Add `CITATION.cff`

Create a [CITATION.cff](CITATION.cff) in the repo root. GitHub auto-renders a "Cite this repository" button from this file. Use CFF version 1.2.0 with the project title, authors, version 0.2.0, and a `preferred-citation` block for the eventual paper.

### 1b. Add `.pre-commit-config.yaml`

`pre-commit` is already a dev dependency but there is no config file. Create [.pre-commit-config.yaml](.pre-commit-config.yaml) with hooks for:

- `ruff` (linting + formatting, replaces black+isort+flake8)
- `ruff-format`
- trailing whitespace, end-of-file fixer, YAML lint
- no large files (prevent checkpoint commits)

### 1c. Add `py.typed` marker

Create an empty [src/fmri2img/py.typed](src/fmri2img/py.typed) file (PEP 561). This advertises that the package ships inline type hints, enabling type-checker support for consumers.

### 1d. Add GitHub Actions CI

Create [.github/workflows/ci.yml](.github/workflows/ci.yml) with:

- **Trigger**: push to main, PRs
- **Jobs**: lint (ruff check + ruff format --check), test (pytest tests/ -v), type-check (optional, mypy)
- **Matrix**: Python 3.10, 3.11
- **Cache**: pip cache for fast runs

---

## Phase 2: Purge Legacy Documentation

**26 of 32** doc files are from the old Brain-to-Image project and reference non-existent scripts, deleted files, or outdated workflows. Keeping them degrades credibility.

### Files to delete (22 files)

**docs/guides/** -- delete 11 of 12 files (keep only `ADAPTER_QUICK_START.md`):

- `PIPELINE_SCRIPT_GUIDE.md`, `REALISTIC_WORKFLOW.md`, `NOVEL_CONTRIBUTIONS_PIPELINE.md`, `EVALUATION_SUITE_GUIDE.md`, `REPORTING_RECONSTRUCTION.md`, `QUICK_START.md`, `GALLERY_SUPPORT.md`, `ADAPTER_TRAINING_GUIDE.md`, `GETTING_STARTED_DIFFUSION.md`, `MLP_IMPLEMENTATION.md`, `RIDGE_BASELINE.md`

**docs/architecture/** -- delete 4 of 5 files (keep only `IMAGERY_EXTENSION.md`):

- `PIPELINE_ARCHITECTURE.md`, `MODULARIZATION_COMPLETE.md`, `DIFFUSION_ROBUSTNESS.md`, `DIFFUSION_DECODER.md`

**docs/technical/** -- delete 7 of 9 files (keep `NSD_IMAGERY_DATASET_GUIDE.md` and `NSD_Dataset_Guide.md`):

- `ADAPTER_METADATA_SUMMARY.md`, `GET_ALL_SAMPLES_GUIDE.md`, `OPTIMAL_CONFIGURATION_GUIDE.md`, `UPGRADE_TO_30K_SAMPLES.md`, `MANUAL_MODEL_DOWNLOAD.md`, `DATA_VALIDATION_REAL_VS_FALLBACK.md`, `PREVENTING_MODEL_DOWNLOAD_BLOCKING.md`

**docs/ root** -- delete all 3:

- `NOVEL_CONTRIBUTIONS_IMPLEMENTATION.md`, `NOVEL_CONTRIBUTIONS_QUICK_REF.md`, `PAPER_GRADE_EVALUATION.md`

### Fix stale references

6 files still reference the deleted `USAGE_EXAMPLES.md`:

- [scripts/deployment/README.md](scripts/deployment/README.md) (2 refs)
- [docs/research/PERCEPTION_VS_IMAGERY_ROADMAP.md](docs/research/PERCEPTION_VS_IMAGERY_ROADMAP.md) (1 ref)
- [docs/architecture/MODULARIZATION_COMPLETE.md](docs/architecture/MODULARIZATION_COMPLETE.md) (1 ref -- file being deleted)
- [src/fmri2img/utils/quick_status.py](src/fmri2img/utils/quick_status.py) (1 ref)
- [scripts/check_setup.sh](scripts/check_setup.sh) (1 ref)

### Update [docs/README.md](docs/README.md)

Rewrite to reflect the drastically reduced, focused doc set. After the purge, the surviving docs are:

- `docs/guides/ADAPTER_QUICK_START.md`
- `docs/architecture/IMAGERY_EXTENSION.md`
- `docs/technical/NSD_IMAGERY_DATASET_GUIDE.md`, `NSD_Dataset_Guide.md`
- `docs/research/PAPER_DRAFT_OUTLINE.md`, `PERCEPTION_VS_IMAGERY_ROADMAP.md`

---

## Phase 3: Consolidate Scripts

### 3a. Convert 15 thin wrappers to `pyproject.toml` entry points

Currently, 15 files in `scripts/` are 1-line wrappers like:

```python
from fmri2img.training.train_ridge import main; main()
```

These should be declared as `[project.scripts]` entry points in [pyproject.toml](pyproject.toml), which generates proper CLI commands on `pip install -e .`. Delete the 15 wrapper files afterward. The non-wrapper scripts (11 files, including all imagery-specific scripts) stay in `scripts/`.

Entry points to add:

- `fmri2img-train-ridge`, `fmri2img-train-mlp`, `fmri2img-train-two-stage`, `fmri2img-train-adapter`, `fmri2img-train-smoke`
- `fmri2img-decode`, `fmri2img-eval`, `fmri2img-compare-evals`
- `fmri2img-ablate`, `fmri2img-build-index`, `fmri2img-build-clip-cache`
- `fmri2img-check-headers`, `fmri2img-download-sd`, `fmri2img-status`
- `fmri2img-build-target-cache`

### 3b. Move misplaced test files

Move 3 test files from [src/fmri2img/scripts/](src/fmri2img/scripts/) to [tests/](tests/):

- `test_compare_evals.py` -> `tests/test_compare_evals.py`
- `test_reliability.py` -> `tests/test_data_reliability.py` (avoid name conflict with existing `tests/test_reliability.py`)
- `test_preprocess.py` -> `tests/test_preprocess.py`

---

## Phase 4: Rewrite the Makefile

The current [Makefile](Makefile) is 559 lines with **8+ broken targets** referencing non-existent scripts. Rewrite it to be concise (~150 lines) with only working, verified targets organized into clear sections:

- **Setup**: `setup`, `setup-dev`, `setup-all`
- **Data**: `build-index`, `build-clip-cache`, `imagery-index`
- **Training**: `ridge`, `mlp`, `two-stage`, `adapter`, `imagery-adapter`
- **Evaluation**: `eval`, `imagery-eval`, `ablate`
- **Research**: `novel-analyses`, `novel-figures`
- **Quality**: `lint`, `format`, `test`, `typecheck`
- **Utility**: `clean`, `status`, `help`

Each target should use the new entry points where applicable. Remove all targets that reference scripts that don't exist.

---

## Phase 5: Code Quality Standardization

### 5a. Standardize logging

3 inconsistencies to fix across the codebase:

- Rename `log = logging.getLogger(...)` to `logger = logging.getLogger(__name__)` in the ~8 files that use `log` instead of `logger`
- Replace named loggers (e.g., `getLogger("build_clip_cache")`) with `getLogger(__name__)`
- Replace `__import__('logging').getLogger(__name__)` in `data/torch_dataset.py` with normal import

### 5b. Replace print() with logging in library modules

Convert `print()` calls to `logger.info()` in library-style modules (not CLI scripts):

- [src/fmri2img/data/nsd_imagery.py](src/fmri2img/data/nsd_imagery.py) (~15 prints)
- [src/fmri2img/data/nsd_index_builder.py](src/fmri2img/data/nsd_index_builder.py) (~6 prints)

CLI scripts (`run_reconstruct_and_eval.py`, `compare_evals.py`, etc.) can keep `print()` for user-facing output.

---

## Impact Summary

- **Before**: 32 docs (26 legacy), 26 scripts (15 redundant wrappers), 559-line Makefile with 8+ broken targets, no CI, no citation, no pre-commit
- **After**: 6 focused docs, 11 purposeful scripts + 15 entry points, ~150-line Makefile with all targets working, GitHub Actions CI, CITATION.cff, pre-commit hooks, py.typed

