"""
Convenience script to label clusters from a .fif file

Usage:
    python label_clusters_from_fif.py optimal_clusters.fif
    
    or in Python:
    
    from label_clusters_from_fif import label_clusters
    cluster, metadata = label_clusters('optimal_clusters.fif')
"""

import sys
from pathlib import Path
import numpy as np

from label_pycrostates_cluster import (
    label_and_reorder_cluster,
    load_cluster_from_fif,
)


def label_clusters(fif_file: str | Path):
    """
    Load clusters from a .fif file and label them based on MetaMaps template.
    
    Parameters
    ----------
    fif_file : str or Path
        Path to the .fif file containing cluster centers
        
    Returns
    -------
    cluster : pycrostates.cluster.ModKMeans or similar
        Labeled cluster object
    metadata : dict
        Metadata about the labeling including template matches and correlations
    """
    from label_pycrostates_cluster import (
        label_and_reorder_cluster,
        load_cluster_from_fif,
    )
    
    print(f"\n{'='*70}")
    print(f"Loading and labeling clusters from: {fif_file}")
    print(f"{'='*70}")
    
    # Load the cluster from .fif file
    print("\n1. Loading cluster from .fif file...")
    try:
        cluster = load_cluster_from_fif(fif_file)
        print(f"   ✓ Loaded successfully")
        print(f"   Number of clusters: {cluster.n_clusters}")
        print(f"   Number of channels: {cluster._cluster_centers_.shape[1]}")
    except Exception as e:
        print(f"   ✗ Error loading file: {e}")
        raise
    
    # Label and reorder based on MetaMaps
    print("\n2. Labeling clusters based on MetaMaps template...")
    try:
        cluster, metadata = label_and_reorder_cluster(cluster)
        print(f"   ✓ Labeled successfully")
        
    except Exception as e:
        print(f"   ✗ Error labeling clusters: {e}")
        raise
    
    # Display results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    print(f"\nCluster labels: {cluster._cluster_names}")
    print(f"Number of clusters: {metadata['n_clusters']}")
    print(f"Number of channels: {metadata['n_channels']}")
    
    print(f"\nTemplate matches:")
    print(f"  (Maps estimated clusters to MetaMaps K=7 solution labels: D, A, C, F, B, G, E)")
    for cluster_idx, template_label in metadata['template_matches'].items():
        corr = metadata['correlation_scores'][cluster_idx]
        print(f"    Estimated cluster {cluster_idx} → Template {template_label} "
              f"(correlation: {corr:.4f})")
    
    print(f"\nCorrelation scores (higher = better match):")
    scores = list(metadata['correlation_scores'].values())
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    print(f"  Mean: {mean_score:.4f}")
    print(f"  Min:  {min_score:.4f}")
    print(f"  Max:  {max_score:.4f}")
    
    # Quality assessment
    if mean_score > 0.7:
        quality = "✓ Good"
    elif mean_score > 0.5:
        quality = "~ Moderate"
    else:
        quality = "✗ Low"
    print(f"  Match quality: {quality}")
    
    return cluster, metadata


if __name__ == "__main__":
    # Command-line usage
    if len(sys.argv) < 2:
        print("Usage: python label_clusters_from_fif.py <fif_file>")
        print("\nExample:")
        print("  python label_clusters_from_fif.py optimal_clusters.fif")
        sys.exit(1)
    
    fif_file = sys.argv[1]
    
    try:
        cluster, metadata = label_clusters(fif_file)
        
        print(f"\n{'='*70}")
        print("✓ Cluster labeling completed successfully!")
        print(f"{'='*70}\n")
        
        # Optionally save or display more detailed results
        print("\nYou can now use the cluster for further analysis:")
        print("  - cluster._cluster_centers_: Array of labeled cluster topographies")
        print("  - cluster._cluster_names: List of labels (A, B, C, D, etc.)")
        print("  - metadata: Dictionary with detailed matching information")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
