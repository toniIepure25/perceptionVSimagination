#!/usr/bin/env python3
"""
Run Imagery Adaptation Ablation Suite
======================================

Runs a complete ablation study comparing:
- A0: Baseline (no adapter) - perception checkpoint evaluated on imagery
- A1: Linear adapter trained on imagery
- A2: MLP adapter trained on imagery
- (Optional) A3: Adapter with condition tokens

Produces consolidated results table, metrics JSON, and paper-ready figures.

Usage:
    python scripts/run_imagery_ablations.py \\
        --index cache/indices/imagery/subj01.parquet \\
        --checkpoint checkpoints/two_stage/subj01/best.pt \\
        --model-type two_stage \\
        --output-dir outputs/imagery_ablations/subj01 \\
        --epochs 50
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: list, desc: str, dry_run: bool = False) -> dict:
    """Run a command and capture output."""
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Running: {desc}")
    logger.info(f"  Command: {' '.join(cmd)}")
    
    if dry_run:
        logger.info(f"  [DRY RUN] Skipping actual execution")
        return {'status': 'skipped', 'dry_run': True}
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        logger.info(f"  ✓ Completed in {elapsed:.1f}s")
        return {
            'status': 'success',
            'elapsed': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"  ✗ Failed after {elapsed:.1f}s")
        logger.error(f"  Error: {e.stderr}")
        return {
            'status': 'failed',
            'elapsed': elapsed,
            'error': str(e),
            'stderr': e.stderr
        }


def load_metrics(metrics_path: Path) -> dict:
    """Load metrics JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run imagery adaptation ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--index', type=str, required=True, help='Path to imagery index parquet')
    parser.add_argument('--checkpoint', type=str, required=True, help='Base model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['ridge', 'mlp', 'two_stage'], required=True)
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for all results')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs for adapters')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--cache-root', type=str, default='cache', help='Cache directory')
    
    # Ablation options
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline (A0) evaluation')
    parser.add_argument('--skip-linear', action='store_true', help='Skip linear adapter (A1)')
    parser.add_argument('--skip-mlp', action='store_true', help='Skip MLP adapter (A2)')
    parser.add_argument('--with-condition', action='store_true', help='Add condition token ablation (A3)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no actual training/eval)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'adapters').mkdir(exist_ok=True)
    (output_dir / 'eval').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("IMAGERY ADAPTATION ABLATION SUITE")
    logger.info("=" * 80)
    logger.info(f"Index: {args.index}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")
    
    # Track all commands
    commands_log = []
    results = []
    
    # ========================================================================
    # A0: Baseline (no adapter)
    # ========================================================================
    if not args.skip_baseline:
        logger.info("-" * 80)
        logger.info("A0: Baseline (No Adapter)")
        logger.info("-" * 80)
        
        baseline_eval_dir = output_dir / 'eval' / 'baseline'
        baseline_eval_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'python', 'scripts/eval_perception_to_imagery_transfer.py',
            '--index', args.index,
            '--checkpoint', args.checkpoint,
            '--model-type', args.model_type,
            '--mode', 'imagery',
            '--split', 'test',
            '--output-dir', str(baseline_eval_dir),
            '--device', args.device,
            '--cache-root', args.cache_root
        ]
        
        if args.dry_run:
            cmd.append('--dry-run')
        
        commands_log.append({
            'name': 'A0_baseline_eval',
            'command': ' '.join(cmd)
        })
        
        result = run_command(cmd, "A0: Baseline evaluation", dry_run=args.dry_run)
        
        if not args.dry_run and result['status'] == 'success':
            metrics = load_metrics(baseline_eval_dir / 'metrics.json')
            results.append({
                'ablation': 'A0_baseline',
                'adapter': 'none',
                'condition_token': False,
                **metrics
            })
    
    # ========================================================================
    # A1: Linear Adapter
    # ========================================================================
    if not args.skip_linear:
        logger.info("-" * 80)
        logger.info("A1: Linear Adapter")
        logger.info("-" * 80)
        
        linear_adapter_dir = output_dir / 'adapters' / 'linear'
        linear_adapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        cmd_train = [
            'python', 'scripts/train_imagery_adapter.py',
            '--index', args.index,
            '--checkpoint', args.checkpoint,
            '--model-type', args.model_type,
            '--adapter', 'linear',
            '--output-dir', str(linear_adapter_dir),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--batch-size', str(args.batch_size),
            '--device', args.device,
            '--seed', str(args.seed),
            '--cache-root', args.cache_root
        ]
        
        commands_log.append({
            'name': 'A1_train_linear',
            'command': ' '.join(cmd_train)
        })
        
        result_train = run_command(cmd_train, "A1: Train linear adapter", dry_run=args.dry_run)
        
        # Eval
        linear_eval_dir = output_dir / 'eval' / 'linear_adapter'
        linear_eval_dir.mkdir(parents=True, exist_ok=True)
        
        cmd_eval = [
            'python', 'scripts/eval_perception_to_imagery_transfer.py',
            '--index', args.index,
            '--checkpoint', args.checkpoint,
            '--model-type', args.model_type,
            '--adapter-checkpoint', str(linear_adapter_dir / 'checkpoints' / 'adapter_best.pt'),
            '--adapter-type', 'linear',
            '--mode', 'imagery',
            '--split', 'test',
            '--output-dir', str(linear_eval_dir),
            '--device', args.device,
            '--cache-root', args.cache_root
        ]
        
        if args.dry_run:
            cmd_eval.append('--dry-run')
        
        commands_log.append({
            'name': 'A1_eval_linear',
            'command': ' '.join(cmd_eval)
        })
        
        result_eval = run_command(cmd_eval, "A1: Evaluate linear adapter", dry_run=args.dry_run)
        
        if not args.dry_run and result_eval['status'] == 'success':
            metrics = load_metrics(linear_eval_dir / 'metrics.json')
            results.append({
                'ablation': 'A1_linear_adapter',
                'adapter': 'linear',
                'condition_token': False,
                **metrics
            })
    
    # ========================================================================
    # A2: MLP Adapter
    # ========================================================================
    if not args.skip_mlp:
        logger.info("-" * 80)
        logger.info("A2: MLP Adapter")
        logger.info("-" * 80)
        
        mlp_adapter_dir = output_dir / 'adapters' / 'mlp'
        mlp_adapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        cmd_train = [
            'python', 'scripts/train_imagery_adapter.py',
            '--index', args.index,
            '--checkpoint', args.checkpoint,
            '--model-type', args.model_type,
            '--adapter', 'mlp',
            '--output-dir', str(mlp_adapter_dir),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--batch-size', str(args.batch_size),
            '--device', args.device,
            '--seed', str(args.seed),
            '--cache-root', args.cache_root
        ]
        
        commands_log.append({
            'name': 'A2_train_mlp',
            'command': ' '.join(cmd_train)
        })
        
        result_train = run_command(cmd_train, "A2: Train MLP adapter", dry_run=args.dry_run)
        
        # Eval
        mlp_eval_dir = output_dir / 'eval' / 'mlp_adapter'
        mlp_eval_dir.mkdir(parents=True, exist_ok=True)
        
        cmd_eval = [
            'python', 'scripts/eval_perception_to_imagery_transfer.py',
            '--index', args.index,
            '--checkpoint', args.checkpoint,
            '--model-type', args.model_type,
            '--adapter-checkpoint', str(mlp_adapter_dir / 'checkpoints' / 'adapter_best.pt'),
            '--adapter-type', 'mlp',
            '--mode', 'imagery',
            '--split', 'test',
            '--output-dir', str(mlp_eval_dir),
            '--device', args.device,
            '--cache-root', args.cache_root
        ]
        
        if args.dry_run:
            cmd_eval.append('--dry-run')
        
        commands_log.append({
            'name': 'A2_eval_mlp',
            'command': ' '.join(cmd_eval)
        })
        
        result_eval = run_command(cmd_eval, "A2: Evaluate MLP adapter", dry_run=args.dry_run)
        
        if not args.dry_run and result_eval['status'] == 'success':
            metrics = load_metrics(mlp_eval_dir / 'metrics.json')
            results.append({
                'ablation': 'A2_mlp_adapter',
                'adapter': 'mlp',
                'condition_token': False,
                **metrics
            })
    
    # ========================================================================
    # A3: MLP Adapter with Condition Token (optional)
    # ========================================================================
    if args.with_condition:
        logger.info("-" * 80)
        logger.info("A3: MLP Adapter + Condition Token")
        logger.info("-" * 80)
        
        mlp_cond_adapter_dir = output_dir / 'adapters' / 'mlp_condition'
        mlp_cond_adapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        cmd_train = [
            'python', 'scripts/train_imagery_adapter.py',
            '--index', args.index,
            '--checkpoint', args.checkpoint,
            '--model-type', args.model_type,
            '--adapter', 'mlp',
            '--condition-token',
            '--output-dir', str(mlp_cond_adapter_dir),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--batch-size', str(args.batch_size),
            '--device', args.device,
            '--seed', str(args.seed),
            '--cache-root', args.cache_root
        ]
        
        commands_log.append({
            'name': 'A3_train_mlp_condition',
            'command': ' '.join(cmd_train)
        })
        
        result_train = run_command(cmd_train, "A3: Train MLP adapter with condition", dry_run=args.dry_run)
        
        # Eval
        mlp_cond_eval_dir = output_dir / 'eval' / 'mlp_adapter_condition'
        mlp_cond_eval_dir.mkdir(parents=True, exist_ok=True)
        
        cmd_eval = [
            'python', 'scripts/eval_perception_to_imagery_transfer.py',
            '--index', args.index,
            '--checkpoint', args.checkpoint,
            '--model-type', args.model_type,
            '--adapter-checkpoint', str(mlp_cond_adapter_dir / 'checkpoints' / 'adapter_best.pt'),
            '--adapter-type', 'mlp',
            '--mode', 'imagery',
            '--split', 'test',
            '--output-dir', str(mlp_cond_eval_dir),
            '--device', args.device,
            '--cache-root', args.cache_root
        ]
        
        if args.dry_run:
            cmd_eval.append('--dry-run')
        
        commands_log.append({
            'name': 'A3_eval_mlp_condition',
            'command': ' '.join(cmd_eval)
        })
        
        result_eval = run_command(cmd_eval, "A3: Evaluate MLP adapter with condition", dry_run=args.dry_run)
        
        if not args.dry_run and result_eval['status'] == 'success':
            metrics = load_metrics(mlp_cond_eval_dir / 'metrics.json')
            results.append({
                'ablation': 'A3_mlp_adapter_condition',
                'adapter': 'mlp',
                'condition_token': True,
                **metrics
            })
    
    # ========================================================================
    # Save commands log
    # ========================================================================
    commands_file = output_dir / 'commands.txt'
    with open(commands_file, 'w') as f:
        f.write("# Imagery Adaptation Ablation Commands\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for cmd_entry in commands_log:
            f.write(f"# {cmd_entry['name']}\n")
            f.write(f"{cmd_entry['command']}\n\n")
    
    logger.info(f"✓ Commands saved to {commands_file}")
    
    if args.dry_run:
        logger.info("")
        logger.info("=" * 80)
        logger.info("DRY RUN COMPLETE - No actual training/evaluation performed")
        logger.info("=" * 80)
        logger.info(f"Commands saved to: {commands_file}")
        return
    
    # ========================================================================
    # Generate consolidated results
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("Generating consolidated results...")
    logger.info("=" * 80)
    
    if len(results) == 0:
        logger.warning("No results to consolidate (all ablations skipped or failed)")
        return
    
    # Create results table
    results_df = pd.DataFrame(results)
    
    # Select key metrics for summary
    key_metrics = ['ablation', 'adapter', 'condition_token', 'clip_cosine_mean', 'clip_cosine_std']
    
    # Add retrieval metrics if present
    for col in results_df.columns:
        if col.startswith('retrieval@'):
            key_metrics.append(col)
    
    summary_df = results_df[key_metrics]
    
    # Save CSV
    csv_path = output_dir / 'results_table.csv'
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Results table saved to {csv_path}")
    
    # Save markdown table
    md_path = output_dir / 'results_table.md'
    with open(md_path, 'w') as f:
        f.write("# Imagery Adaptation Ablation Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")
    logger.info(f"✓ Markdown table saved to {md_path}")
    
    # Save full metrics JSON
    metrics_path = output_dir / 'metrics_all.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Full metrics saved to {metrics_path}")
    
    # Generate figures
    logger.info("")
    logger.info("Generating figures...")
    cmd_figures = [
        'python', 'scripts/make_paper_figures.py',
        '--ablation-dir', str(output_dir),
        '--output-dir', str(output_dir / 'figures')
    ]
    
    result_figures = run_command(cmd_figures, "Generate paper figures", dry_run=False)
    
    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("ABLATION SUITE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results directory: {output_dir}")
    logger.info(f"  - results_table.csv: Summary table")
    logger.info(f"  - results_table.md: Markdown table")
    logger.info(f"  - metrics_all.json: Full metrics")
    logger.info(f"  - figures/: Paper-ready figures")
    logger.info(f"  - commands.txt: Exact commands executed")
    logger.info("")
    logger.info("Results summary:")
    print(summary_df.to_string(index=False))
    logger.info("")


if __name__ == "__main__":
    main()
