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

Bootstrap confidence intervals are automatically generated if not found.
The script will create bootstrap/bootstrap_v2_results.csv on first run.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from evaluation.rubrics.settings import result_directory

# Default run number
RUN_NUMBER = 9


def generate_color_gradient(base_color: str, n_colors: int) -> list:
    """
    Generate a gradient of colors from light to dark based on a base color.
    
    Args:
        base_color: Base color in hex format (e.g., '#4C72B0')
        n_colors: Number of colors to generate
    
    Returns:
        List of hex color strings forming a gradient
    """
    if n_colors == 1:
        return [base_color]
    
    # Convert hex to RGB
    rgb = mcolors.hex2color(base_color)
    
    # Create lighter and darker versions
    colors = []
    for i in range(n_colors):
        # Scale from 0.6 (lighter) to 1.2 (can be darker via saturation)
        # Use i/max to go from light to dark
        factor = 0.7 + (i / max(n_colors - 1, 1)) * 0.5  # 0.7 to 1.2
        
        # Adjust brightness
        if factor <= 1.0:
            # Lighten
            new_rgb = tuple(min(1.0, c + (1 - c) * (1 - factor)) for c in rgb)
        else:
            # Darken
            new_rgb = tuple(max(0.0, c * (2 - factor)) for c in rgb)
        
        colors.append(mcolors.rgb2hex(new_rgb))
    
    return colors


# Level definitions with cluster groupings
LEVELS = {
    "basic": {
        # NOTE: Keep this in sync with enabled keys in CLUSTERS below.
        # "content" is currently disabled in CLUSTERS, so we omit it here.
        "categories": ["clarity", "knowledge_organization"],
        "description": "Foundational communication quality",
        "base_color": "#2E5C8A",  # Deep blue
        "color_palette": None,  # Will be generated dynamically
    },
    "intermediate": {
        "categories": ["rhetorical_devices", "style"],
        "description": "Enhanced engagement techniques",
        "base_color": "#C44E52",  # Red
        "color_palette": None,  # Will be generated dynamically
    },
    "advanced": {
        # "dialogue" is currently disabled in CLUSTERS, so we omit it here.
        "categories": [],
        "description": "Sophisticated argumentation and dialogue",
        "base_color": "#8B5CF6",  # Purple
        "color_palette": None,  # Will be generated dynamically
    },
}

# Cluster definitions with metrics and their individual weights
# Equal weighting: 6 effective metrics (readability counts as 1), each gets 1/6
# Readability's 1/6 is split among 4 metrics → 1/24 each
CLUSTERS = {
    "clarity": {
        "metrics": {
            "jargon": {"weight": 1/6},  # ≈ 0.1667
            # "explanation_type_v2": {"weight": 1/6},
            # Readability metrics (combined weight = 1/6, split 4 ways)
            "flesch_reading_ease": {"weight": 1/24},  # ≈ 0.0417
            "flesch_kincaid": {"weight": 1/24},
            "dale_chall": {"weight": 1/24},
            "ari": {"weight": 1/24},
        },
        "description": "Clear, understandable explanations",
        "level": "basic",
    },
    # "content": {
    #     "metrics": {
    #         "connection_to_everyday_life_v2": {"weight": 1/6},
    #     },
    #     "description": "Connection to real-world context",
    #     "level": "basic",
    # },
    "knowledge_organization": {
        "metrics": {
            "scaffolding_v2": {"weight": 1/6},  # ≈ 0.1667
        },
        "description": "Knowledge organization and structure",
        "level": "basic",
    },
    "rhetorical_devices": {
        "metrics": {
            "analogy_v2": {"weight": 1/6},  # ≈ 0.1667
            "metaphor_v8": {"weight": 1/6},  # ≈ 0.1667
        },
        "description": "Use of analogies and metaphors",
        "level": "intermediate",
    },
    "style": {
        "metrics": {
            "humor_v5": {"weight": 1/6},  # ≈ 0.1667
        },
        "description": "Engaging style (humor)",
        "level": "intermediate",
    },
    # "dialogue": {
    #     "metrics": {
    #         "argumentation": {"weight": 0.0},
    #     },
    #     "description": "Argumentation quality and multiple perspectives",
    #     "level": "advanced",
    # },
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

# Assign colors to metrics with gradients
def assign_metric_colors():
    """
    Assign colors to each metric based on level's base color.
    Generates a gradient for each level's metrics.
    """
    colors = {}
    
    for level_name, level_info in LEVELS.items():
        # Count metrics in this level
        n_metrics_in_level = 0
        for cat_name in level_info["categories"]:
            if cat_name in CLUSTERS:
                n_metrics_in_level += len(CLUSTERS[cat_name]["metrics"])
        
        if n_metrics_in_level == 0:
            continue
        
        # Generate gradient for this level
        base_color = level_info["base_color"]
        gradient = generate_color_gradient(base_color, n_metrics_in_level)
        
        # Store gradient in level_info for future use
        level_info["color_palette"] = gradient
        
        # Assign colors to metrics
        idx = 0
        for cat_name in level_info["categories"]:
            if cat_name in CLUSTERS:
                for metric_name in CLUSTERS[cat_name]["metrics"]:
                    colors[metric_name] = gradient[idx]
                    idx += 1
    
    return colors

METRIC_COLORS = assign_metric_colors()

# Metrics where lower values are better (will be inverted during normalization)
LOWER_IS_BETTER = ["ari", "dale_chall", "flesch_kincaid"]

# Readability metrics to group into a single block in visualizations
READABILITY_METRICS = ["flesch_reading_ease", "flesch_kincaid", "dale_chall", "ari"]
READABILITY_COMBINED_WEIGHT = sum(CLUSTERS["clarity"]["metrics"][m]["weight"] for m in READABILITY_METRICS)

# Custom normalization ranges: {metric_name: (min_val, max_val)}
# Values outside this range are clipped to 0 or 1
# DEFAULT: (0, 1) for all metrics not listed here
NORMALIZATION_RANGES = {
    "jargon": (0.65, 1.0),  # 0.65 → 0, 1.0 → 1 (min observed: ~0.659 in run 3)
    # Readability metrics (grade-level based, lower is better - will be inverted)
    # Range: grade 6 (elementary) to grade 16 (college graduate)
    "flesch_kincaid": (6, 16),  # Grade 6 → 1, Grade 16+ → 0
    "ari": (6, 16),  # Grade 6 → 1, Grade 16+ → 0
    "dale_chall": (7, 12),  # 7th grade → 1, college level → 0
    # Flesch Reading Ease: already 0-1 scaled, clip to reasonable range
    "flesch_reading_ease": (0.3, 0.7),  # 0.3 (difficult) → 0, 0.7+ (easy) → 1
    # All other metrics default to (0, 1) range - no need to add them here
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


def load_bootstrap_confidence_intervals(
    directory: str,
    expected_models: set = None,
    bootstrap_dir: str | None = None,
) -> pd.DataFrame | None:
    """
    Load pre-computed bootstrap confidence intervals for v2 weighted scores.
    If not found or missing expected models, automatically generate them.

    Args:
        directory: Run directory path (metric CSV source)
        expected_models: Set of model names expected to have confidence intervals
        bootstrap_dir: Optional scratch directory for bootstrap CSV read/write.
            When omitted, uses ``<directory>/bootstrap/`` (canonical location).

    Returns:
        DataFrame with columns: Model, Total_Score, Bootstrap_SE, CI_Lower, CI_Upper
        or None if generation fails.
    """
    if bootstrap_dir is not None:
        bootstrap_path = os.path.join(bootstrap_dir, "bootstrap_v2_results.csv")
    else:
        bootstrap_path = os.path.join(directory, "bootstrap", "bootstrap_v2_results.csv")
    should_regenerate = False
    
    # Check if bootstrap results exist
    if os.path.exists(bootstrap_path):
        df = pd.read_csv(bootstrap_path)
        existing_models = set(df['Model'].values)
        
        # Check if we have all expected models
        if expected_models:
            missing_models = expected_models - existing_models
            if missing_models:
                print(f"  Bootstrap file exists but missing {len(missing_models)} model(s): {missing_models}")
                print(f"  Regenerating bootstrap confidence intervals...")
                should_regenerate = True
            else:
                print(f"  Loaded v2 bootstrap confidence intervals for {len(df)} models")
                return df[['Model', 'Total_Score', 'Bootstrap_SE', 'CI_Lower', 'CI_Upper']]
        else:
            print(f"  Loaded v2 bootstrap confidence intervals for {len(df)} models")
            return df[['Model', 'Total_Score', 'Bootstrap_SE', 'CI_Lower', 'CI_Upper']]
    else:
        print(f"  V2 bootstrap results not found at {bootstrap_path}.")
        print(f"  Generating bootstrap confidence intervals...")
        should_regenerate = True
    
    # Generate bootstrap results if needed
    if should_regenerate:
        try:
            # Import bootstrap function with proper module path
            current_dir = os.path.dirname(__file__)
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from evaluation.rubrics.custom_metrics.bootstrap import bootstrap_analysis_v2
            
            # Generate bootstrap results
            bootstrap_df = bootstrap_analysis_v2(
                directory,
                n_bootstrap=10000,
                bootstrap_dir=bootstrap_dir,
            )
            
            if bootstrap_df is not None and not bootstrap_df.empty:
                print(f"  ✓ Successfully generated bootstrap confidence intervals for {len(bootstrap_df)} models")
                return bootstrap_df[['Model', 'Total_Score', 'Bootstrap_SE', 'CI_Lower', 'CI_Upper']]
            else:
                print(f"  Warning: Bootstrap generation returned no results")
                return None
                
        except Exception as e:
            import traceback
            print(f"  Warning: Failed to generate bootstrap confidence intervals: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            print(f"  Continuing without confidence intervals...")
            return None
    
    return None


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
        
        # Normalize across ALL models for this metric (not per-model)
        # Stack all model columns into one series for normalization
        all_values = pd.concat([df[col].dropna() for col in df.columns], ignore_index=True)
        
        # Use custom range if defined, otherwise default to (0, 1)
        if metric_name in NORMALIZATION_RANGES:
            min_val, max_val = NORMALIZATION_RANGES[metric_name]
        else:
            # Default to (0, 1) range for all binary metrics
            min_val, max_val = 0, 1
        
        # Now normalize each model using the same min/max
        for col in df.columns:
            model_name = col.replace("__score", "")
            model_values = df[col].dropna()
            
            # Apply normalization with consistent min/max
            if min_val == max_val:
                normalized = pd.Series([0.5] * len(model_values), index=model_values.index)
            else:
                normalized = (model_values - min_val) / (max_val - min_val)
                normalized = normalized.clip(0, 1)
                
                if metric_name in LOWER_IS_BETTER:
                    normalized = 1 - normalized
            
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
    # Sort models: unprompted variants at bottom, prompted at top
    # Within each group, sort by total score ascending
    df_valid = total_df.dropna(subset=["total_score"])
    
    # Separate prompted vs unprompted models
    # Prompted = has "prompt" OR "cot" in name
    has_prompt = df_valid["model"].str.contains("prompt", case=False)
    has_cot = df_valid["model"].str.contains("cot", case=False)
    is_prompted = has_prompt | has_cot
    
    unprompted_df = df_valid[~is_prompted].sort_values("total_score", ascending=True)
    prompted_df = df_valid[is_prompted].sort_values("total_score", ascending=True)
    
    # Track the separator position (number of unprompted models)
    separator_position = len(unprompted_df)
    
    # Combine: unprompted at bottom (first in list), prompted at top (last in list)
    df_sorted = pd.concat([unprompted_df, prompted_df], ignore_index=True)
    
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
                if weight == 0:  # Skip metrics with zero weight (e.g., argumentation)
                    continue
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
    readability_plotted = False  # Track if we've plotted the combined readability block
    
    for cluster_name in ["basic", "intermediate", "advanced"]:
        cluster_info = LEVELS[cluster_name]
        cluster_start = left.copy()
        
        for cat_name in cluster_info["categories"]:
            for metric_name in CLUSTERS[cat_name]["metrics"].keys():
                if metric_name not in METRIC_WEIGHTS:
                    continue
                
                metric_info = METRIC_WEIGHTS[metric_name]
                if metric_info["weight"] == 0:  # Skip metrics with zero weight
                    continue
                
                # Handle readability metrics as a combined block
                if metric_name in READABILITY_METRICS:
                    if readability_plotted:
                        continue  # Skip - already plotted as combined block
                    readability_plotted = True
                    
                    # Compute combined readability scores
                    combined_scores = []
                    combined_weighted_scores = []
                    for model in models:
                        model_avg = 0
                        count = 0
                        for rm in READABILITY_METRICS:
                            if rm in df_sorted.columns:
                                score = df_sorted[df_sorted["model"] == model][rm].values
                                if len(score) > 0 and pd.notna(score[0]):
                                    model_avg += score[0]
                                    count += 1
                        avg_score = model_avg / count if count > 0 else 0
                        combined_scores.append(avg_score)
                        combined_weighted_scores.append(avg_score * READABILITY_COMBINED_WEIGHT * scale_factors[model])
                    
                    scores = np.array(combined_scores)
                    weighted_scores = np.array(combined_weighted_scores)
                    short_name = "Readability"
                    weight_pct = READABILITY_COMBINED_WEIGHT * 100
                    color = METRIC_COLORS[READABILITY_METRICS[0]]  # Use first readability metric's color
                else:
                    # Regular metric handling
                    color = METRIC_COLORS[metric_name]
                    scores = []
                    weighted_scores_list = []
                    for model in models:
                        if metric_name in df_sorted.columns:
                            score = df_sorted[df_sorted["model"] == model][metric_name].values
                            raw_score = score[0] if len(score) > 0 and pd.notna(score[0]) else 0
                        else:
                            raw_score = 0
                        scores.append(raw_score)
                        weighted_scores_list.append(raw_score * metric_info["weight"] * scale_factors[model])
                    
                    scores = np.array(scores)
                    weighted_scores = np.array(weighted_scores_list)
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
    
    print(f"\n  DEBUG: bootstrap_df is {'None' if bootstrap_df is None else f'DataFrame with {len(bootstrap_df)} rows'}")
    if bootstrap_df is not None:
        print(f"  DEBUG: Bootstrap models: {list(bootstrap_df['Model'].values)}")
        print(f"  DEBUG: Plot models: {models}")
    
    if bootstrap_df is not None:
        bootstrap_mismatch_warned = False
        models_with_ci = set(bootstrap_df['Model'].values)
        models_without_ci = []
        
        for i, model in enumerate(models):
            # Match model name (handle potential __score suffix)
            model_clean = model.replace('__score', '')
            match = bootstrap_df[bootstrap_df['Model'] == model_clean]
            
            print(f"  DEBUG: Looking for '{model_clean}' in bootstrap... {'FOUND' if not match.empty else 'NOT FOUND'}")
            
            if not match.empty:
                total = total_scores[model]  # Use the actual bar end position
                bootstrap_total = match.iloc[0]['Total_Score']
                bootstrap_se = match.iloc[0]['Bootstrap_SE']
                
                # Check if bootstrap data is stale (scores differ significantly)
                score_diff = abs(total - bootstrap_total)
                if score_diff > 0.05 and not bootstrap_mismatch_warned:
                    print(f"  Warning: Bootstrap data may be stale - score differences detected")
                    print(f"    (e.g., {model_clean}: current={total:.3f}, bootstrap={bootstrap_total:.3f})")
                    print(f"    Re-run aggregate_v2.py to regenerate confidence intervals")
                    bootstrap_mismatch_warned = True
                
                # Scale SE proportionally if scores differ (rough approximation)
                if bootstrap_total > 0:
                    scaled_se = bootstrap_se * (total / bootstrap_total)
                else:
                    scaled_se = bootstrap_se
                
                ci_half_width = 1.96 * scaled_se  # 95% CI
                
                print(f"  DEBUG: Drawing CI for {model_clean}: total={total:.3f}, SE={scaled_se:.4f}, CI_width={ci_half_width:.4f}")
                
                # Draw error bar at the actual bar end (total_score)
                ax.errorbar(total, i, xerr=ci_half_width, 
                           fmt='none', color='black', capsize=4, capthick=1.5, elinewidth=1.5)
                error_bar_ends[model] = total + ci_half_width
            else:
                models_without_ci.append(model_clean)
        
        # Report models without confidence intervals
        if models_without_ci:
            print(f"  Note: {len(models_without_ci)} model(s) missing confidence intervals:")
            for m in models_without_ci:
                print(f"    - {m}")
            print(f"  This can happen if the model has insufficient data across metrics.")
            print(f"  Bootstrap will auto-regenerate on next run of aggregate_v2.py")
    
    # Add total score labels at the end of each bar
    for i, model in enumerate(models):
        total = total_scores[model]
        # Adjust label position if error bar is present
        if model in error_bar_ends:
            label_x = error_bar_ends[model] + 0.01
        else:
            label_x = total + 0.01
        ax.text(label_x, i, f'{total:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Add horizontal separator between unprompted (bottom) and prompted (top) groups
    if separator_position > 0 and separator_position < len(models):
        ax.axhline(y=separator_position - 0.5, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Weighted Score", fontsize=11)
    
    # Calculate actual level weights dynamically
    level_weights = {}
    for level_name, level_info in LEVELS.items():
        level_weight = sum(
            METRIC_WEIGHTS[m]["weight"] 
            for cat in level_info["categories"] 
            for m in CLUSTERS[cat]["metrics"].keys()
            if m in METRIC_WEIGHTS
        )
        level_weights[level_name] = level_weight
    
    # Build title with actual weights (only show levels that have weight > 0)
    title_parts = ["Science Communication Quality Score"]
    level_labels = []
    for level_name in ["basic", "intermediate", "advanced"]:
        if level_weights.get(level_name, 0) > 0:
            level_labels.append(f"{level_name.title()} ({level_weights[level_name]*100:.0f}%)")
    
    if level_labels:
        title_parts.append("  |  ".join(level_labels))
    
    ax.set_title("\n".join(title_parts), fontsize=13, fontweight="bold")
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
    
    # Combine readability metrics into a single row
    if all(m in pivot_df.index for m in READABILITY_METRICS):
        readability_row = pivot_df.loc[READABILITY_METRICS].mean()
        pivot_df = pivot_df.drop(READABILITY_METRICS)
        pivot_df.loc["readability"] = readability_row
    
    # Sort metrics by cluster then category, excluding zero-weight and individual readability metrics
    metric_order = []
    readability_added = False
    for cluster_name in ["basic", "intermediate", "advanced"]:
        for cat_name in LEVELS[cluster_name]["categories"]:
            for m in CLUSTERS[cat_name]["metrics"].keys():
                # Skip metrics with zero weight (e.g., argumentation)
                if m in METRIC_WEIGHTS and METRIC_WEIGHTS[m]["weight"] == 0:
                    continue
                # Replace individual readability metrics with combined "readability"
                if m in READABILITY_METRICS:
                    if not readability_added and "readability" in pivot_df.index:
                        metric_order.append("readability")
                        readability_added = True
                    continue
                if m in pivot_df.index:
                    metric_order.append(m)
    pivot_df = pivot_df.reindex(metric_order)
    
    # Sort models by average score
    model_order = pivot_df.mean().sort_values(ascending=False).index
    pivot_df = pivot_df[model_order]
    
    # Calculate figure width based on number of models (wider for more models)
    n_models = len(pivot_df.columns)
    n_metrics = len(pivot_df.index)
    fig_width = max(14, n_models * 1.2)  # At least 14, scale with models
    fig_height = max(8, n_metrics * 0.8)  # At least 8, scale with metrics
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn", 
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Normalized Score"})
    
    ax.set_title("Model Performance Across All Metrics\n(Basic → Intermediate)", 
                fontsize=13, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    
    # Rotate x-axis labels at an angle to prevent overlap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    # Keep y-axis labels horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: metric_heatmap.png")


def main(
    run_number: int = RUN_NUMBER,
    output_dir: str | None = None,
    bootstrap_dir: str | None = None,
):
    """Main aggregation and plotting pipeline."""
    directory = result_directory(run_number)
    canonical_output_dir = os.path.join(directory, "aggregations_v2")
    output_dir = output_dir or canonical_output_dir
    os.makedirs(output_dir, exist_ok=True)

    if output_dir == canonical_output_dir:
        print(
            "WARNING: writing aggregation outputs to the canonical directory "
            f"({canonical_output_dir}) may overwrite tracked publication artifacts. "
            "Use --output-dir under /tmp for safe reruns."
        )
    if bootstrap_dir is None:
        print(
            "WARNING: bootstrap confidence intervals load/generate under the "
            f"canonical directory ({directory}/bootstrap/) unless --bootstrap-dir "
            "points to a scratch location."
        )
    
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
    display_cols = ["model", "basic_score", "intermediate_score", "total_score"]
    display_cols = [c for c in display_cols if c in total_df.columns]
    print(total_df[display_cols].sort_values("total_score", ascending=False).to_string(index=False, float_format="%.3f"))
    
    # Load bootstrap confidence intervals if available
    print("\n" + "-"*60)
    print("Loading bootstrap confidence intervals...")
    expected_models = set(total_df["model"].unique())
    bootstrap_df = load_bootstrap_confidence_intervals(
        directory,
        expected_models=expected_models,
        bootstrap_dir=bootstrap_dir,
    )
    
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
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate rubric scores (v2).")
    parser.add_argument(
        "--run",
        type=int,
        default=RUN_NUMBER,
        help="canonical result run number (9=rubric scores, 10=preference metrics)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for aggregation CSVs and plots (default: <run-dir>/aggregations_v2)",
    )
    parser.add_argument(
        "--bootstrap-dir",
        default=None,
        help="scratch directory for bootstrap CSV load/generation (default: <run-dir>/bootstrap/)",
    )
    args = parser.parse_args()
    main(args.run, output_dir=args.output_dir, bootstrap_dir=args.bootstrap_dir)

