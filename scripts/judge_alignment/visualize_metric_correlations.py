#!/usr/bin/env python3
"""
Visualize correlations between metrics as a network graph.
Nodes = metrics, Edge width = correlation strength.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

THRESHOLD = 0.5
METRICS = [
    'humor_v2_score', 
    'metaphor_v2_score', 
    'analogy_v2_score', 
    'connection_to_everyday_life_v2_score', 
    'scaffolding_score'
]

SHORT_NAMES = {
    'humor_v2_score': 'Humor',
    'metaphor_v2_score': 'Metaphor', 
    'analogy_v2_score': 'Analogy',
    'connection_to_everyday_life_v2_score': 'Connection',
    'scaffolding_score': 'Scaffolding'
}


def main():
    script_dir = Path(__file__).parent
    metrics_path = script_dir / "balanced_dataset_v2_human" / "ask_science_human_metrics.csv"
    output_path = script_dir / "balanced_dataset_v2_human" / "metric_correlations.png"
    
    print(f"Loading data from: {metrics_path}")
    df = pd.read_csv(metrics_path)
    print(f"Total rows: {len(df)}")
    
    # Calculate correlation matrix
    corr_matrix = df[METRICS].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for metric in METRICS:
        G.add_node(SHORT_NAMES[metric])
    
    # Add edges with correlation as weight
    for i, m1 in enumerate(METRICS):
        for j, m2 in enumerate(METRICS):
            if i < j:  # Only upper triangle
                corr = corr_matrix.loc[m1, m2]
                if not np.isnan(corr):
                    G.add_edge(SHORT_NAMES[m1], SHORT_NAMES[m2], weight=abs(corr), corr=corr)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Position nodes in a circle
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='#4a90d9',
        node_size=3000,
        alpha=0.9,
        ax=ax
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=11,
        font_weight='bold',
        font_color='white',
        ax=ax
    )
    
    # Draw edges with width proportional to correlation
    edges = G.edges(data=True)
    
    # Separate positive and negative correlations
    pos_edges = [(u, v) for u, v, d in edges if d['corr'] > 0]
    neg_edges = [(u, v) for u, v, d in edges if d['corr'] < 0]
    
    pos_weights = [G[u][v]['weight'] * 8 for u, v in pos_edges]  # Scale for visibility
    neg_weights = [G[u][v]['weight'] * 8 for u, v in neg_edges]
    
    # Draw positive correlations (blue)
    if pos_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=pos_edges,
            width=pos_weights,
            edge_color='#2ecc71',
            alpha=0.7,
            ax=ax
        )
    
    # Draw negative correlations (red)
    if neg_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=neg_edges,
            width=neg_weights,
            edge_color='#e74c3c',
            style='dashed',
            alpha=0.7,
            ax=ax
        )
    
    # Add edge labels with correlation values
    edge_labels = {(u, v): f"{d['corr']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=9,
        font_color='#333',
        ax=ax
    )
    
    ax.set_title('Metric Correlations\n(Edge width ∝ |correlation|, Green=positive, Red=negative)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved correlation graph to: {output_path}")
    
    # Also print pairwise correlations sorted by strength
    print("\nPairwise correlations (sorted by absolute value):")
    pairs = []
    for i, m1 in enumerate(METRICS):
        for j, m2 in enumerate(METRICS):
            if i < j:
                corr = corr_matrix.loc[m1, m2]
                pairs.append((SHORT_NAMES[m1], SHORT_NAMES[m2], corr))
    
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for m1, m2, corr in pairs:
        print(f"  {m1} <-> {m2}: {corr:.3f}")


if __name__ == "__main__":
    main()

