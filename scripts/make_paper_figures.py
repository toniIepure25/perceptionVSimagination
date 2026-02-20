#!/usr/bin/env python3
"""
Generate Paper-Ready Figures for Imagery Adaptation Ablation
=============================================================

Creates publication-quality figures and LaTeX tables from ablation results.

Usage:
    python scripts/make_paper_figures.py \\
        --ablation-dir outputs/imagery_ablations/subj01 \\
        --output-dir outputs/imagery_ablations/subj01/figures
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def load_results(ablation_dir: Path) -> pd.DataFrame:
    """Load results from CSV or JSON."""
    csv_path = ablation_dir / 'results_table.csv'
    json_path = ablation_dir / 'metrics_all.json'
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise FileNotFoundError(f"No results found in {ablation_dir}")


def plot_overall_performance(df: pd.DataFrame, output_dir: Path):
    """Create bar chart of overall performance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Prepare data
    ablations = df['ablation'].values
    cosine_mean = df['clip_cosine_mean'].values
    cosine_std = df['clip_cosine_std'].values if 'clip_cosine_std' in df.columns else np.zeros_like(cosine_mean)
    
    # Clean labels
    labels = []
    for abl in ablations:
        if 'baseline' in abl.lower():
            labels.append('Baseline\n(No Adapter)')
        elif 'linear' in abl.lower():
            labels.append('Linear\nAdapter')
        elif 'condition' in abl.lower():
            labels.append('MLP Adapter\n+ Condition')
        elif 'mlp' in abl.lower():
            labels.append('MLP\nAdapter')
        else:
            labels.append(abl)
    
    # Create bars
    x = np.arange(len(labels))
    bars = ax.bar(x, cosine_mean, yerr=cosine_std, capsize=5,
                   color=['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'][:len(labels)],
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, cosine_mean)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Styling
    ax.set_ylabel('CLIP Cosine Similarity', fontweight='bold')
    ax.set_title('Perception→Imagery Transfer Performance', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(cosine_mean) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'bar_overall_metric.png'
    plt.savefig(output_path)
    plt.close()
    
    print(f"✓ Saved overall performance figure to {output_path}")


def plot_by_stimulus_type(ablation_dir: Path, output_dir: Path):
    """Create grouped bar chart by stimulus type."""
    # Load per-trial results
    eval_dirs = list((ablation_dir / 'eval').glob('*'))
    
    if len(eval_dirs) == 0:
        print("⚠ No eval directories found, skipping stimulus type figure")
        return
    
    # Collect data
    stimulus_data = []
    
    for eval_dir in eval_dirs:
        per_trial_path = eval_dir / 'per_trial.csv'
        if not per_trial_path.exists():
            continue
        
        df = pd.read_csv(per_trial_path)
        
        # Get ablation name
        ablation_name = eval_dir.name.replace('_', ' ').title()
        
        # Compute per-stimulus-type means
        for stype in df['stimulus_type'].unique():
            mask = df['stimulus_type'] == stype
            mean_cosine = df.loc[mask, 'clip_cosine'].mean()
            
            stimulus_data.append({
                'ablation': ablation_name,
                'stimulus_type': stype,
                'clip_cosine_mean': mean_cosine
            })
    
    if len(stimulus_data) == 0:
        print("⚠ No stimulus type data found, skipping figure")
        return
    
    stimulus_df = pd.DataFrame(stimulus_data)
    
    # Pivot for grouped bar chart
    pivot = stimulus_df.pivot(index='stimulus_type', columns='ablation', values='clip_cosine_mean')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind='bar', ax=ax, width=0.8,
               color=['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'],
               edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Styling
    ax.set_ylabel('CLIP Cosine Similarity', fontweight='bold')
    ax.set_xlabel('Stimulus Type', fontweight='bold')
    ax.set_title('Performance by Stimulus Type', fontweight='bold', pad=15)
    ax.legend(title='Method', title_fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / 'bar_by_stimulus_type.png'
    plt.savefig(output_path)
    plt.close()
    
    print(f"✓ Saved stimulus type figure to {output_path}")


def create_latex_table(df: pd.DataFrame, output_dir: Path):
    """Create LaTeX table."""
    output_path = output_dir / 'table.tex'
    
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Imagery Adaptation Ablation Results}\n")
        f.write("\\label{tab:imagery_ablation}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Method & Adapter & Condition & CLIP Cosine & Retrieval@1 \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df.iterrows():
            ablation = row['ablation'].replace('_', ' ').title()
            adapter = row['adapter'].title() if row['adapter'] != 'none' else '—'
            condition = '✓' if row.get('condition_token', False) else '—'
            cosine = f"{row['clip_cosine_mean']:.3f}"
            retrieval = f"{row.get('retrieval@1', 0.0):.3f}" if 'retrieval@1' in row else '—'
            
            f.write(f"{ablation} & {adapter} & {condition} & {cosine} & {retrieval} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX table to {output_path}")


def create_markdown_table_enhanced(df: pd.DataFrame, output_dir: Path):
    """Create enhanced markdown table with formatting."""
    output_path = output_dir / 'table_formatted.md'
    
    with open(output_path, 'w') as f:
        f.write("# Imagery Adaptation Results\n\n")
        f.write("## Performance Summary\n\n")
        
        f.write("| Method | Adapter | Condition Token | CLIP Cosine | Retrieval@1 | Retrieval@5 |\n")
        f.write("|--------|---------|-----------------|-------------|-------------|-------------|\n")
        
        for _, row in df.iterrows():
            method = row['ablation'].replace('_', ' ').title()
            adapter = row['adapter'].title() if row['adapter'] != 'none' else '—'
            condition = '✓' if row.get('condition_token', False) else '—'
            cosine = f"{row['clip_cosine_mean']:.4f} ± {row.get('clip_cosine_std', 0.0):.4f}"
            ret1 = f"{row.get('retrieval@1', 0.0):.4f}" if 'retrieval@1' in row else '—'
            ret5 = f"{row.get('retrieval@5', 0.0):.4f}" if 'retrieval@5' in row else '—'
            
            f.write(f"| {method} | {adapter} | {condition} | {cosine} | {ret1} | {ret5} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Find best performing
        best_idx = df['clip_cosine_mean'].idxmax()
        best_row = df.loc[best_idx]
        baseline_row = df[df['adapter'] == 'none'].iloc[0] if 'none' in df['adapter'].values else None
        
        f.write(f"- **Best method**: {best_row['ablation']} ")
        f.write(f"(CLIP Cosine = {best_row['clip_cosine_mean']:.4f})\n")
        
        if baseline_row is not None:
            improvement = ((best_row['clip_cosine_mean'] - baseline_row['clip_cosine_mean']) 
                          / baseline_row['clip_cosine_mean'] * 100)
            f.write(f"- **Improvement over baseline**: {improvement:.1f}%\n")
        
        f.write(f"- **Total ablations**: {len(df)}\n")
    
    print(f"✓ Saved formatted markdown table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from ablation results")
    parser.add_argument('--ablation-dir', type=str, required=True, help='Ablation results directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for figures')
    
    args = parser.parse_args()
    
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING PAPER FIGURES")
    print("=" * 80)
    print(f"Input: {ablation_dir}")
    print(f"Output: {output_dir}")
    print("")
    
    # Load results
    try:
        df = load_results(ablation_dir)
        print(f"✓ Loaded {len(df)} ablation results")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_overall_performance(df, output_dir)
    plot_by_stimulus_type(ablation_dir, output_dir)
    
    # Generate tables
    print("\nGenerating tables...")
    create_latex_table(df, output_dir)
    create_markdown_table_enhanced(df, output_dir)
    
    print("")
    print("=" * 80)
    print("FIGURES COMPLETE")
    print("=" * 80)
    print(f"Generated files in: {output_dir}")
    print("  - bar_overall_metric.png")
    print("  - bar_by_stimulus_type.png")
    print("  - table.tex (LaTeX)")
    print("  - table_formatted.md (Markdown)")
    print("")


if __name__ == "__main__":
    main()
