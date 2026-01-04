#!/usr/bin/env python3
"""
Plot logistic regression results as matplotlib tables.

Reads CSV files from logistic_regression.py and creates combined figures.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"


def load_results(mode: str, no_dpo: bool, no_readability: bool) -> tuple[pd.DataFrame, str, int]:
    """
    Load regression results from CSV.
    
    Returns:
        Tuple of (DataFrame, label for the table, n_samples)
    """
    suffix = ""
    label_parts = [mode.capitalize()]
    
    if no_dpo:
        suffix += "_no_prompted_dpo"
        label_parts.append("(no Prompted-DPO)")
    if no_readability:
        suffix += "_no_readability"
        label_parts.append("(no readability)")
    
    filename = f"logistic_regression_{mode}{suffix}.csv"
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return None, None, None
    
    df = pd.read_csv(filepath)
    label = " ".join(label_parts)
    
    # Get n_samples from the DataFrame (if available)
    n_samples = None
    if 'n_samples' in df.columns:
        n_samples = int(df['n_samples'].iloc[0])
    
    return df, label, n_samples


def create_table_data(results_df: pd.DataFrame) -> list[list[str]]:
    """Convert results DataFrame to table data."""
    # Filter out intercept
    df = results_df[results_df['metric'] != 'intercept'].copy()
    
    table_data = []
    for _, row in df.iterrows():
        metric = row['metric'].replace('_v8', '').replace('_v2', '').replace('_', ' ').title()
        coef = f"{row['coefficient']:.3f}"
        se = f"{row['std_error']:.3f}" if not np.isnan(row['std_error']) else "N/A"
        p_val = row['p_value']
        if np.isnan(p_val):
            p_str = "N/A"
        elif p_val < 0.001:
            p_str = "<.001***"
        elif p_val < 0.01:
            p_str = f"{p_val:.3f}**"
        elif p_val < 0.05:
            p_str = f"{p_val:.3f}*"
        else:
            p_str = f"{p_val:.3f}"
        
        odds = f"{row['odds_ratio']:.2f}"
        ci_low = row['odds_ratio_ci_lower']
        ci_high = row['odds_ratio_ci_upper']
        ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]" if not np.isnan(ci_low) else "N/A"
        
        table_data.append([metric, coef, se, p_str, odds, ci_str])
    
    return table_data


def create_combined_figure(tables: list[tuple[pd.DataFrame, str, int]], output_path: Path):
    """
    Create a single figure with multiple regression tables.
    
    Args:
        tables: List of (results_df, label, n_samples) tuples
        output_path: Path to save the figure
    """
    n_tables = len(tables)
    if n_tables == 0:
        print("No tables to plot!")
        return
    
    # Calculate figure height based on number of tables
    fig_height = 2.5 * n_tables + 0.5
    fig, axes = plt.subplots(n_tables, 1, figsize=(10, fig_height))
    
    # Handle single table case
    if n_tables == 1:
        axes = [axes]
    
    columns = ['Metric', 'β', 'SE', 'p-value', 'OR', '95% CI']
    
    for ax, (results_df, label, n_samples) in zip(axes, tables):
        ax.axis('off')
        ax.axis('tight')
        
        table_data = create_table_data(results_df)
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            colColours=['#E6E6E6'] * len(columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)
        
        # Bold header and highlight significant rows
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight='bold')
            if row > 0:
                # Get p-value from the original data
                metric_idx = row - 1
                df_filtered = results_df[results_df['metric'] != 'intercept']
                if metric_idx < len(df_filtered):
                    p_val = df_filtered.iloc[metric_idx]['p_value']
                    if not np.isnan(p_val) and p_val < 0.05:
                        cell.set_facecolor('#E8F4E8')
        
        # Title for each table (include N if available)
        title = f"{label} (N={n_samples})" if n_samples else label
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle("Logistic Regression: Predicting Human Preference for Explanation A", 
                 fontsize=14, fontweight='bold', y=1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot logistic regression results as tables"
    )
    parser.add_argument(
        '--include-binarized',
        action='store_true',
        help="Include binarized mode results"
    )
    parser.add_argument(
        '--include-no-dpo',
        action='store_true',
        help="Include results that exclude prompted-DPO clusters"
    )
    parser.add_argument(
        '--include-no-readability',
        action='store_true',
        help="Include results that exclude readability metrics"
    )
    parser.add_argument(
        '--only-no-readability',
        action='store_true',
        help="Only show results that exclude readability metrics (ignores --include-no-readability)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output PNG path (default: auto-generated based on flags)"
    )
    
    args = parser.parse_args()
    
    # Determine which modes to include
    modes = ['continuous']
    if args.include_binarized:
        modes.append('binarized')
    
    # Determine which variants to include
    if args.only_no_readability:
        # Only show no-readability variants
        variants = [(False, True)]  # (no_dpo, no_readability)
        if args.include_no_dpo:
            variants.append((True, True))
    else:
        variants = [(False, False)]  # (no_dpo, no_readability)
        if args.include_no_dpo:
            variants.append((True, False))
        if args.include_no_readability:
            variants.append((False, True))
        if args.include_no_dpo and args.include_no_readability:
            variants.append((True, True))
    
    # Load all requested tables
    tables = []
    for mode in modes:
        for no_dpo, no_readability in variants:
            results_df, label, n_samples = load_results(mode, no_dpo, no_readability)
            if results_df is not None:
                # Add mode to label if showing multiple modes
                if len(modes) > 1:
                    full_label = f"{mode.capitalize()}: {label}" if label != mode.capitalize() else label
                else:
                    full_label = label.replace(f"{mode.capitalize()} ", "")
                tables.append((results_df, full_label, n_samples))
            else:
                print(f"Warning: Could not load {mode} (no_dpo={no_dpo}, no_readability={no_readability})")
    
    if not tables:
        print("No tables found to plot!")
        return
    
    # Generate output filename
    if args.output:
        output_path = Path(args.output)
    else:
        parts = ["logistic_regression_combined"]
        if args.include_binarized:
            parts.append("with_binarized")
        if args.include_no_dpo:
            parts.append("with_no_dpo")
        if args.only_no_readability:
            parts.append("only_no_readability")
        elif args.include_no_readability:
            parts.append("with_no_readability")
        output_path = DATA_DIR / f"{'_'.join(parts)}.png"
    
    print(f"Creating combined figure with {len(tables)} tables...")
    create_combined_figure(tables, output_path)


if __name__ == "__main__":
    main()

