"""
Example usage of label_pycrostates_cluster module

This script demonstrates how to:
1. Load EEG data
2. Fit a pycrostates ModKMeans cluster
3. Label and reorder the cluster based on MetaMaps template
4. Inspect the results
"""

import mne
import numpy as np
from pycrostates.cluster import ModKMeans
from label_pycrostates_cluster import label_and_reorder_cluster


def example_basic_usage():
    """Basic example: Load data, fit cluster, label it"""
    
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    # Load example data (you would use your own data)
    # For this example, we'll create synthetic data
    print("\n1. Creating synthetic EEG data...")
    n_channels = 71  # Matches MetaMaps montage
    n_samples = 10000
    sfreq = 250
    
    # Create realistic EEG-like data
    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_samples) * 10  # microvolts
    
    # Create MNE Raw object
    ch_names = [f'Ch{i:02d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    print(f"   Created raw data: {raw.get_data().shape}")
    
    # Fit a ModKMeans cluster
    print("\n2. Fitting ModKMeans cluster (K=4)...")
    cluster = ModKMeans(n_clusters=4, n_init=10, random_state=42)
    cluster.fit(raw)
    print(f"   Fitted cluster with {cluster.n_clusters} clusters")
    print(f"   Cluster names (before): {cluster._cluster_names}")
    
    # Label and reorder based on MetaMaps template
    print("\n3. Labeling and reordering cluster...")
    cluster, metadata = label_and_reorder_cluster(cluster)
    
    # Inspect results
    print(f"\n4. Results:")
    print(f"   Cluster names (after): {cluster._cluster_names}")
    print(f"   Template labels (ordered): {metadata['template_labels_ordered']}")
    print(f"\n   Matching details:")
    for cluster_idx, template_label in metadata['template_matches'].items():
        corr = metadata['correlation_scores'][cluster_idx]
        print(f"     Cluster {cluster_idx} → Template {template_label} "
              f"(correlation: {corr:.4f})")
    
    return cluster, metadata


def example_inspect_metadata():
    """Example: Inspect detailed metadata about the matches"""
    
    print("\n" + "=" * 70)
    print("Example 2: Inspecting Metadata")
    print("=" * 70)
    
    # (Using the cluster from example 1)
    cluster, metadata = example_basic_usage()
    
    print("\n" + "=" * 70)
    print("Metadata Inspection")
    print("=" * 70)
    
    print(f"\nMetadata keys: {list(metadata.keys())}")
    
    print(f"\nTemplate matches:")
    print(f"  {metadata['template_matches']}")
    
    print(f"\nCorrelation scores (higher = better match):")
    for cluster_idx, score in metadata['correlation_scores'].items():
        print(f"  Cluster {cluster_idx}: {score:.4f}")
    
    print(f"\nCluster information:")
    print(f"  Number of clusters: {metadata['n_clusters']}")
    print(f"  Number of channels: {metadata['n_channels']}")
    print(f"  Template file: {metadata['template_file']}")
    
    # Determine match quality
    print(f"\nMatch quality assessment:")
    scores = list(metadata['correlation_scores'].values())
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print(f"  Mean correlation: {mean_score:.4f}")
    print(f"  Min correlation: {min_score:.4f}")
    print(f"  Max correlation: {max_score:.4f}")
    
    if mean_score > 0.7:
        print(f"  ✓ Good match quality (mean > 0.7)")
    elif mean_score > 0.5:
        print(f"  ~ Moderate match quality (mean > 0.5)")
    else:
        print(f"  ✗ Low match quality (mean < 0.5)")


def example_working_with_cluster():
    """Example: Using the labeled cluster for further analysis"""
    
    print("\n" + "=" * 70)
    print("Example 3: Working with Labeled Cluster")
    print("=" * 70)
    
    # Create and label a cluster
    print("\n1. Creating and labeling cluster...")
    
    rng = np.random.RandomState(42)
    n_channels = 71
    n_samples = 10000
    data = rng.randn(n_channels, n_samples) * 10
    
    ch_names = [f'Ch{i:02d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    cluster = ModKMeans(n_clusters=4, n_init=10, random_state=42)
    cluster.fit(raw)
    cluster, metadata = label_and_reorder_cluster(cluster)
    
    print(f"   Cluster labeled with states: {cluster._cluster_names}")
    
    # Now you can use the cluster for various analyses
    print("\n2. Example: Using cluster for segmentation analysis...")
    
    # Get segmentation (labels for each sample in the original data)
    segmentation = cluster._labels_
    
    # Count samples in each state
    print(f"\n   Sample distribution across states:")
    for i, label in enumerate(cluster._cluster_names):
        n_samples_in_state = np.sum(segmentation == i)
        percentage = 100 * n_samples_in_state / len(segmentation)
        print(f"     State {label}: {n_samples_in_state:5d} samples ({percentage:.1f}%)")
    
    # Example: Calculate state duration metrics (if you had temporal info)
    print(f"\n   Cluster centers shape: {cluster._cluster_centers_.shape}")
    print(f"   (7x channels - one topographic map per state)")
    
    # You could now:
    # - Plot topographic maps for each state
    # - Analyze state transitions
    # - Compare to other datasets
    # - Export for external tools


def example_custom_template_file():
    """Example: Using a custom template file location"""
    
    print("\n" + "=" * 70)
    print("Example 4: Custom Template File")
    print("=" * 70)
    
    print("\nYou can specify a custom template file:")
    print("""
    cluster, metadata = label_and_reorder_cluster(
        cluster,
        template_file="/path/to/custom/MetaMaps_alternative.set"
    )
    """)
    
    print("\nThe default is: MetaMaps_2023_06.set")
    print("(assumed to be in the current directory)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("pycrostates Cluster Labeling Examples")
    print("=" * 70)
    
    # Run examples
    try:
        example_basic_usage()
        example_inspect_metadata()
        example_working_with_cluster()
        example_custom_template_file()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
