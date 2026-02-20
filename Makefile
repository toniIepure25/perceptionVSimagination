.PHONY: help setup setup-dev setup-all \
       build-index build-clip-cache imagery-index \
       ridge mlp two-stage adapter smoke imagery-adapter \
       eval compare-evals imagery-eval ablate \
       novel-analyses novel-figures paper-figures \
       lint format test typecheck \
       clean status

PYTHON ?= python3

# ──────────────────────────────────────────────────────
#  Perception vs. Imagination — Project Makefile
# ──────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ──────────────── Setup ────────────────────────────────

setup: ## Install core package (editable)
	pip install -e .

setup-dev: ## Install with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

setup-all: ## Install all optional dependencies
	pip install -e ".[all]"
	pre-commit install

# ──────────────── Data ─────────────────────────────────

build-index: ## Build the NSD full index
	fmri2img-build-index

build-clip-cache: ## Build CLIP embedding cache
	fmri2img-build-clip-cache

imagery-index: ## Build NSD-Imagery Parquet index
	$(PYTHON) scripts/build_nsd_imagery_index.py

# ──────────────── Training ─────────────────────────────

ridge: ## Train ridge regression decoder
	fmri2img-train-ridge

mlp: ## Train MLP decoder
	fmri2img-train-mlp

two-stage: ## Train two-stage (ridge + MLP) decoder
	fmri2img-train-two-stage

adapter: ## Train CLIP adapter
	fmri2img-train-adapter

smoke: ## Run smoke test (fast training sanity check)
	fmri2img-train-smoke

imagery-adapter: ## Train imagery adapter (perception → imagination)
	$(PYTHON) scripts/train_imagery_adapter.py

# ──────────────── Evaluation ───────────────────────────

eval: ## Run reconstruction + evaluation pipeline
	fmri2img-eval

compare-evals: ## Compare multiple evaluation runs
	fmri2img-compare-evals

imagery-eval: ## Evaluate perception-to-imagery transfer
	$(PYTHON) scripts/eval_perception_to_imagery_transfer.py

ablate: ## Run preprocessing & ridge ablation study
	fmri2img-ablate

# ──────────────── Research ─────────────────────────────

novel-analyses: ## Run all six novel analysis directions
	$(PYTHON) scripts/run_novel_analyses.py

novel-figures: ## Generate figures for novel analyses
	$(PYTHON) scripts/make_novel_figures.py

paper-figures: ## Generate publication-ready paper figures
	$(PYTHON) scripts/make_paper_figures.py

imagery-ablations: ## Run imagery adapter ablation experiments
	$(PYTHON) scripts/run_imagery_ablations.py

# ──────────────── Code Quality ─────────────────────────

lint: ## Lint with ruff
	ruff check src/ tests/ scripts/

format: ## Format with ruff
	ruff format src/ tests/ scripts/

test: ## Run test suite
	pytest tests/ -v

typecheck: ## Type-check with mypy (optional)
	mypy src/fmri2img/ --ignore-missing-imports

# ──────────────── Utility ──────────────────────────────

status: ## Show project status (data, models, configs)
	fmri2img-status

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .ruff_cache/
