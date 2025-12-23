"""
Aggregate v2: Category-based scoring system for science communication metrics.

Hierarchical structure (Baram-Tsabari & Lewenstein, 2012):
├── Basic Level
│   ├── Clarity: jargon, explanation_type, readability
│   ├── Content: connection to everyday life
│   └── Knowledge Organization: scaffolding
├── Intermediate Level
│   ├── Rhetorical Devices: analogy, metaphor
│   └── Style: humor
└── Advanced Level
    └── Dialogue: argumentation

Each metric has an individual weight defined in CLUSTERS.
Total score = weighted sum of metric scores.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROJECT_DIR

# Default run number
RUN_NUMBER = 9

# Level definitions with cluster groupings
LEVELS = {
    "basic": {
        "categories": ["clarity", "content", "knowledge_organization"],
        "description": "Foundational communication quality",
        "color_palette": ["#4C72B0", "#6A9BD1", "#8BB4E0", "#DD8452"],  # Blues + orange
    },
    "intermediate": {
        "categories": ["rhetorical_devices", "style"],
        "description": "Enhanced engagement techniques",
        "color_palette": ["#55A868", "#7BC48A", "#C44E52"],  # Greens + red
    },
    "advanced": {
        "categories": ["dialogue"],
        "description": "Sophisticated argumentation and dialogue",
        "color_palette": ["#8B5CF6", "#A78BFA"],  # Purple tones
    },
}

# Cluster definitions with metrics and their individual weights
CLUSTERS = {
    "clarity": {
        "metrics": {
            "jargon": {"weight": 0.10},
            "explanation_type_v2": {"weight": 0.15},
            # Readability metrics (equal weight each)
            "flesch_reading_ease": {"weight": 0.025},
            "flesch_kincaid": {"weight": 0.025},
            "dale_chall": {"weight": 0.025},
            "ari": {"weight": 0.025},
        },
        "description": "Clear, understandable explanations",
        "level": "basic",
    },
    "content": {
        "metrics": {
            "connection_to_everyday_life_v2": {"weight": 0.10},
        },
        "description": "Connection to real-world context",
        "level": "basic",
    },
    "knowledge_organization": {
        "metrics": {
            "scaffolding": {"weight": 0.15},
        },
        "description": "Knowledge organization and structure",
        "level": "basic",
    },
    "rhetorical_devices": {
        "metrics": {
            "analogy_v2": {"weight": 0.15},
            "metaphor_v2": {"weight": 0.15},
        },
        "description": "Use of analogies and metaphors",
        "level": "intermediate",
    },
    "style": {
        "metrics": {
            "humor_v2": {"weight": 0.10},
        },
        "description": "Engaging style (humor)",
        "level": "intermediate",
    },
    "dialogue": {
        "metrics": {
            "argumentation": {"weight": 0.0},
        },
        "description": "Argumentation quality and multiple perspectives",
        "level": "advanced",
    },
}

# Compute metric weights from CLUSTERS definition
def compute_metric_weights():
    """Compute weight for each metric from CLUSTERS definition."""
    weights = {}
    
    for cat_name, cat_info in CLUSTERS.items():
        level_name = cat_info["level"]
        for metric_name, metric_info in cat_info["metrics"].items():
            weights[metric_name] = {
                "weight": metric_info["weight"],
                "cluster": level_name,
                "category": cat_name,
            }
    
    return weights


def validate_weights(weights: dict, tolerance: float = 1e-6):
    """
    Validate that metric weights sum to 1.
    
    Args:
        weights: Dictionary of metric weights from compute_metric_weights()
        tolerance: Acceptable deviation from 1.0
    
    Raises:
        ValueError: If weights don't sum to 1 (within tolerance)
    """
    total = sum(info["weight"] for info in weights.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Metric weights must sum to 1.0, but got {total:.4f}. "
            f"Difference: {total - 1.0:+.4f}"
        )


METRIC_WEIGHTS = compute_metric_weights()
validate_weights(METRIC_WEIGHTS)

# Assign colors to metrics
def assign_metric_colors():
    """Assign colors to each metric based on cluster palette."""
    colors = {}
    for level_name, level_info in LEVELS.items():
        palette = level_info["color_palette"]
        idx = 0
        for cat_name in level_info["categories"]:
            for metric_name in CLUSTERS[cat_name]["metrics"]:
                colors[metric_name] = palette[idx % len(palette)]
                idx += 1
    return colors

METRIC_COLORS = assign_metric_colors()

# Metrics where lower values are better (will be inverted during normalization)
LOWER_IS_BETTER = ["ari", "dale_chall", "flesch_kincaid"]

# Custom normalization ranges: {metric_name: (min_val, max_val)}
# Values outside this range are clipped to 0 or 1
NORMALIZATION_RANGES = {
    "jargon": (0.65, 1.0),  # 0.65 → 0, 1.0 → 1 (min observed: ~0.659 in run 3)
    # Readability metrics (grade-level based, lower is better - will be inverted)
    # Range: grade 6 (elementary) to grade 16 (college graduate)
    "flesch_kincaid": (6, 16),  # Grade 6 → 1, Grade 16+ → 0
    "ari": (6, 16),  # Grade 6 → 1, Grade 16+ → 0
    "dale_chall": (7, 12),  # 7th grade → 1, college level → 0
    # Flesch Reading Ease: already 0-1 scaled, clip to reasonable range
    "flesch_reading_ease": (0.3, 0.7),  # 0.3 (difficult) → 0, 0.7+ (easy) → 1
    # Binary metrics already on 0-1 scale - use fixed range to avoid
    # min==max returning 0.5 when all values are 0 (or all 1)
    "humor_v2": (0, 1),  # 0 → 0, 1 → 1
    "analogy_v2": (0, 1),  # 0 → 0, 1 → 1
    "metaphor_v2": (0, 1),  # 0 → 0, 1 → 1
    "connection_to_everyday_life_v2": (0, 1),  # 0 → 0, 1 → 1
    "scaffolding": (0, 1),  # 0 → 0, 1 → 1
    "explanation_type_v2": (0, 1),  # 0 → 0, 1 → 1
}

# Backwards compatibility: map v2 metric names to v1 fallbacks for old runs
METRIC_FALLBACKS = {
    "explanation_type_v2": ["explanation_type", "explanation_type_explicit"],
    "connection_to_everyday_life_v2": ["connection_to_everyday_life", "connection_to_everyday_life_explicit"],
    "analogy_v2": ["analogy", "analogy_explicit"],
    "metaphor_v2": ["metaphor", "metaphor_explicit"],
    "humor_v2": ["humor_explicit", "humor"],
    "scaffolding": [],
    "argumentation": [],  # No v1 fallback (new metric)
    "jargon": [],  # No fallback needed
    "flesch_reading_ease": [],  # No fallback needed
    "flesch_kincaid": [],  # No fallback needed
    "dale_chall": [],  # No fallback needed
    "ari": [],  # No fallback needed
}


def load_bootstrap_confidence_intervals(directory: str) -> pd.DataFrame | None:
    """
    Load pre-computed bootstrap confidence intervals for v2 weighted scores.
    
    Returns DataFrame with columns: Model, Total_Score, Bootstrap_SE, CI_Lower, CI_Upper
    or None if file not found.
    """
    bootstrap_path = os.path.join(directory, "bootstrap", "bootstrap_v2_results.csv")
    if not os.path.exists(bootstrap_path):
        print(f"  Warning: V2 bootstrap results not found at {bootstrap_path}.")
        print(f"  Run: python bootstrap.py {directory}")
        return None
    
    df = pd.read_csv(bootstrap_path)
    print(f"  Loaded v2 bootstrap confidence intervals for {len(df)} models")
    return df[['Model', 'Total_Score', 'Bootstrap_SE', 'CI_Lower', 'CI_Upper']]


def normalize_metric(series: pd.Series, metric_name: str) -> pd.Series:
    """
    Normalize a metric series to [0, 1] range.
    
    Uses custom range from NORMALIZATION_RANGES if defined,
    otherwise normalizes between observed min/max.
    Inverts if metric is in LOWER_IS_BETTER list.
    """
    # Use custom range if defined, otherwise use observed min/max
    if metric_name in NORMALIZATION_RANGES:
        min_val, max_val = NORMALIZATION_RANGES[metric_name]
    else:
        min_val = series.min()
        max_val = series.max()
    
    if min_val == max_val:
        return pd.Series([0.5] * len(series), index=series.index)
    
    normalized = (series - min_val) / (max_val - min_val)
    
    # Clip to [0, 1] range (important for custom ranges)
    normalized = normalized.clip(0, 1)
    
    if metric_name in LOWER_IS_BETTER:
        normalized = 1 - normalized
    
    return normalized


def load_metric_data(directory: str, metric_name: str) -> tuple[pd.DataFrame, str]:
    """
    Load and return a metric CSV file.
    Tries the primary metric name first, then falls back to v1 names for backwards compatibility.
    
    Returns:
        Tuple of (DataFrame, actual_metric_name_loaded) or (None, None) if not found.
    """
    # Try primary metric name first
    filepath = os.path.join(directory, f"{metric_name}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        score_cols = [col for col in df.columns if col.endswith("__score")]
        return df[score_cols], metric_name
    
    # Try fallback names for backwards compatibility
    fallbacks = METRIC_FALLBACKS.get(metric_name, [])
    for fallback_name in fallbacks:
        fallback_path = os.path.join(directory, f"{fallback_name}.csv")
        if os.path.exists(fallback_path):
            print(f"  Using fallback: {fallback_name} for {metric_name}")
            df = pd.read_csv(fallback_path)
            score_cols = [col for col in df.columns if col.endswith("__score")]
            return df[score_cols], fallback_name
    
    print(f"Warning: Metric file not found: {metric_name} (no fallbacks available)")
    return None, None


def compute_metric_scores(directory: str) -> pd.DataFrame:
    """
    Compute normalized scores for each metric and model.
    
    Returns DataFrame with columns: model, metric, score, weight, cluster, category
    """
    results = []
    
    for metric_name, metric_info in METRIC_WEIGHTS.items():
        df, loaded_name = load_metric_data(directory, metric_name)
        if df is None:
            continue
        
        for col in df.columns:
            model_name = col.replace("__score", "")
            # Use the canonical metric name (not the fallback) for normalization lookup
            normalized = normalize_metric(df[col].dropna(), metric_name)
            
            results.append({
                "model": model_name,
                "metric": metric_name,  # Always use canonical name
                "loaded_from": loaded_name,  # Track which file was actually loaded
                "score": normalized.mean(),
                "weight": metric_info["weight"],
                "cluster": metric_info["cluster"],
                "category": metric_info["category"],
            })
    
    return pd.DataFrame(results)


def compute_total_scores(metric_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted total scores per model from metric-level data.
    
    Returns DataFrame with columns: model, total_score, basic_score, intermediate_score, 
                                    plus individual metric scores
    """
    if metric_df.empty:
        return pd.DataFrame()
    
    results = []
    
    for model in metric_df["model"].unique():
        model_data = metric_df[metric_df["model"] == model]
        
        row = {"model": model}
        
        # Compute total weighted score
        total_weighted = 0
        total_weight = 0
        
        # Also compute cluster subtotals
        cluster_scores = {cluster: {"weighted": 0, "weight": 0} for cluster in LEVELS.keys()}
        
        for _, metric_row in model_data.iterrows():
            score = metric_row["score"]
            weight = metric_row["weight"]
            cluster = metric_row["cluster"]
            metric_name = metric_row["metric"]
            
            if pd.notna(score):
                total_weighted += score * weight
                total_weight += weight
                cluster_scores[cluster]["weighted"] += score * weight
                cluster_scores[cluster]["weight"] += weight
                
                # Store individual metric score
                row[metric_name] = score
        
        row["total_score"] = total_weighted / total_weight if total_weight > 0 else np.nan
        
        # Cluster subtotals (normalized to their actual weight from metrics)
        for cluster_name, cluster_data in cluster_scores.items():
            if cluster_data["weight"] > 0:
                # Score as weighted average within the cluster
                row[f"{cluster_name}_score"] = cluster_data["weighted"] / cluster_data["weight"]
            else:
                row[f"{cluster_name}_score"] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_stacked_bar_chart(total_df: pd.DataFrame, metric_df: pd.DataFrame, output_dir: str, 
                          bootstrap_df: pd.DataFrame = None):
    """
    Create a stacked bar chart showing total score broken down by individual metrics.
    Metrics are grouped by cluster (Basic | Intermediate) with visual separator.
    Each segment's height represents its WEIGHTED contribution to the total.
    
    Args:
        total_df: DataFrame with total scores per model
        metric_df: DataFrame with individual metric scores
        output_dir: Directory to save output
        bootstrap_df: Optional DataFrame with bootstrap confidence intervals
    """
    # Sort models by total score
    df_sorted = total_df.dropna(subset=["total_score"]).sort_values("total_score", ascending=True)
    
    if df_sorted.empty:
        print("  No data for stacked bar chart")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = df_sorted["model"].tolist()
    y_pos = np.arange(len(models))
    
    # First pass: compute raw bar heights and scale factors for each model
    # This ensures the stacked bar height matches total_score
    raw_bar_heights = {model: 0.0 for model in models}
    total_scores = {model: df_sorted[df_sorted["model"] == model]["total_score"].values[0] for model in models}
    
    for cluster_name in ["basic", "intermediate", "advanced"]:
        for cat_name in LEVELS[cluster_name]["categories"]:
            for metric_name in CLUSTERS[cat_name]["metrics"].keys():
                if metric_name not in METRIC_WEIGHTS:
                    continue
                weight = METRIC_WEIGHTS[metric_name]["weight"]
                for model in models:
                    if metric_name in df_sorted.columns:
                        score = df_sorted[df_sorted["model"] == model][metric_name].values
                        if len(score) > 0 and pd.notna(score[0]):
                            raw_bar_heights[model] += score[0] * weight
    
    # Compute scale factors so bar heights match total_score
    scale_factors = {}
    for model in models:
        if raw_bar_heights[model] > 0:
            scale_factors[model] = total_scores[model] / raw_bar_heights[model]
        else:
            scale_factors[model] = 1.0
    
    # Stack bars from left to right, grouped by cluster
    left = np.zeros(len(models))
    legend_handles = []
    legend_labels = []
    
    # Plot clusters in order: Basic, Intermediate, Advanced
    cluster_boundaries = []
    for cluster_name in ["basic", "intermediate", "advanced"]:
        cluster_info = LEVELS[cluster_name]
        cluster_start = left.copy()
        
        for cat_name in cluster_info["categories"]:
            for metric_name in CLUSTERS[cat_name]["metrics"].keys():
                if metric_name not in METRIC_WEIGHTS:
                    continue
                
                metric_info = METRIC_WEIGHTS[metric_name]
                color = METRIC_COLORS[metric_name]
                
                # Get scores for each model (scaled to match total_score)
                scores = []
                weighted_scores_list = []
                for model in models:
                    if metric_name in df_sorted.columns:
                        score = df_sorted[df_sorted["model"] == model][metric_name].values
                        raw_score = score[0] if len(score) > 0 and pd.notna(score[0]) else 0
                    else:
                        raw_score = 0
                    scores.append(raw_score)
                    # Scale weighted contribution so total bar height = total_score
                    weighted_scores_list.append(raw_score * metric_info["weight"] * scale_factors[model])
                
                scores = np.array(scores)
                weighted_scores = np.array(weighted_scores_list)
                
                # Shorten metric name for display
                short_name = metric_name.replace("_v2", "").replace("_explicit", "").replace("_", " ").title()
                weight_pct = metric_info["weight"] * 100
                
                bars = ax.barh(y_pos, weighted_scores, left=left, 
                               color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
                
                legend_handles.append(bars[0])
                legend_labels.append(f'{short_name} ({weight_pct:.1f}%)')
                
                # Add value labels in the middle of each segment (if large enough)
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    width = bar.get_width()
                    if width > 0.025:  # Only label if segment is visible
                        ax.text(left[i] + width/2, bar.get_y() + bar.get_height()/2,
                               f'{score:.1f}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
                
                left += weighted_scores
        
        # Add vertical line to separate clusters (after basic and intermediate clusters)
        if cluster_name in ["basic", "intermediate"]:
            cluster_boundaries.append(left.copy())
            for i, x_pos in enumerate(left):
                ax.plot([x_pos, x_pos], [i - 0.4, i + 0.4], color='black', linewidth=2, linestyle='-')
    
    # Add error bars from v2 bootstrap confidence intervals if available
    error_bar_ends = {}  # Track CI upper bounds for label positioning
    if bootstrap_df is not None:
        bootstrap_mismatch_warned = False
        for i, model in enumerate(models):
            # Match model name (handle potential __score suffix)
            model_clean = model.replace('__score', '')
            match = bootstrap_df[bootstrap_df['Model'] == model_clean]
            if not match.empty:
                total = total_scores[model]  # Use the actual bar end position
                bootstrap_total = match.iloc[0]['Total_Score']
                bootstrap_se = match.iloc[0]['Bootstrap_SE']
                
                # Check if bootstrap data is stale (scores differ significantly)
                score_diff = abs(total - bootstrap_total)
                if score_diff > 0.05 and not bootstrap_mismatch_warned:
                    print(f"  Warning: Bootstrap data may be stale - score differences detected")
                    print(f"    (e.g., {model_clean}: current={total:.3f}, bootstrap={bootstrap_total:.3f})")
                    print(f"    Re-run bootstrap.py to update confidence intervals")
                    bootstrap_mismatch_warned = True
                
                # Scale SE proportionally if scores differ (rough approximation)
                if bootstrap_total > 0:
                    scaled_se = bootstrap_se * (total / bootstrap_total)
                else:
                    scaled_se = bootstrap_se
                
                ci_half_width = 1.96 * scaled_se  # 95% CI
                
                # Draw error bar at the actual bar end (total_score)
                ax.errorbar(total, i, xerr=ci_half_width, 
                           fmt='none', color='black', capsize=4, capthick=1.5, elinewidth=1.5)
                error_bar_ends[model] = total + ci_half_width
    
    # Add total score labels at the end of each bar
    for i, model in enumerate(models):
        total = total_scores[model]
        # Adjust label position if error bar is present
        if model in error_bar_ends:
            label_x = error_bar_ends[model] + 0.01
        else:
            label_x = total + 0.01
        ax.text(label_x, i, f'{total:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Weighted Score", fontsize=11)
    ax.set_title("Science Communication Quality Score\nBasic (50%)  |  Intermediate (30%)  |  Advanced (20%)", 
                fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.15)
    
    # Legend
    ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=8, 
              ncol=2, title="Metrics (weight)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stacked_total_scores.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: stacked_total_scores.png")


def plot_metric_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Create a heatmap showing all metrics x models.
    """
    if df.empty:
        return
    
    # Pivot to get metrics as rows, models as columns
    pivot_df = df.pivot(index="metric", columns="model", values="score")
    
    # Sort metrics by cluster then category
    metric_order = []
    for cluster_name in ["basic", "intermediate", "advanced"]:
        for cat_name in LEVELS[cluster_name]["categories"]:
            for m in CLUSTERS[cat_name]["metrics"].keys():
                if m in pivot_df.index:
                    metric_order.append(m)
    pivot_df = pivot_df.reindex(metric_order)
    
    # Sort models by average score
    model_order = pivot_df.mean().sort_values(ascending=False).index
    pivot_df = pivot_df[model_order]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn", 
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Normalized Score"})
    
    ax.set_title("Model Performance Across All Metrics\n(Basic → Intermediate → Advanced)", 
                fontsize=13, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: metric_heatmap.png")


def main(run_number: int = RUN_NUMBER):
    """Main aggregation and plotting pipeline."""
    directory = f"{PROJECT_DIR}/Benchmarking/deep_eval/data/run_{run_number}"
    output_dir = os.path.join(directory, "aggregations_v2")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Aggregate v2: Metric-weighted Scoring")
    print(f"{'='*60}")
    print(f"Run: {run_number}")
    print(f"Output: {output_dir}\n")
    
    # Print weight structure
    print("WEIGHT STRUCTURE:")
    for level_name, level_info in LEVELS.items():
        level_weight = sum(
            METRIC_WEIGHTS[m]["weight"] 
            for cat in level_info["categories"] 
            for m in CLUSTERS[cat]["metrics"].keys()
            if m in METRIC_WEIGHTS
        )
        print(f"\n  {level_name.upper()} LEVEL ({level_weight*100:.1f}%):")
        for cat_name in level_info["categories"]:
            print(f"    {cat_name.replace('_', ' ').title()}:")
            for metric in CLUSTERS[cat_name]["metrics"].keys():
                weight = METRIC_WEIGHTS[metric]["weight"]
                print(f"      • {metric}: {weight*100:.1f}%")
    
    # Compute metric-level scores
    print("\n" + "-"*60)
    print("Computing metric scores...")
    metric_df = compute_metric_scores(directory)
    metric_df.to_csv(os.path.join(output_dir, "metric_scores.csv"), index=False)
    print(f"  Saved: metric_scores.csv")
    
    # Compute total scores
    print("Computing total scores...")
    total_df = compute_total_scores(metric_df)
    total_df.to_csv(os.path.join(output_dir, "total_scores.csv"), index=False)
    print(f"  Saved: total_scores.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("TOTAL SCORES (sorted)")
    print("="*60)
    display_cols = ["model", "basic_score", "intermediate_score", "advanced_score", "total_score"]
    display_cols = [c for c in display_cols if c in total_df.columns]
    print(total_df[display_cols].sort_values("total_score", ascending=False).to_string(index=False, float_format="%.3f"))
    
    # Load bootstrap confidence intervals if available
    print("\n" + "-"*60)
    print("Loading bootstrap confidence intervals...")
    bootstrap_df = load_bootstrap_confidence_intervals(directory)
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_stacked_bar_chart(total_df, metric_df, output_dir, bootstrap_df)
    plot_metric_heatmap(metric_df, output_dir)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_num = int(sys.argv[1]) if len(sys.argv) > 1 else RUN_NUMBER
    main(run_num)

