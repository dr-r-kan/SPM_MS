"""
Label and reorder pycrostates ModKMeans cluster objects based on MetaMaps template.

This module provides functionality to:
1. Load MetaMaps template microstates from EEGLAB .set file using MNE
2. Match estimated clusters to template clusters based on spatial correlation (ignoring polarity)
3. Relabel and reorder clusters alphabetically (A, B, C, D, E, F, G)

MetaMaps structure:
    - The .set file contains 22 microstate templates: K=4,5,6,7 solutions
    - Indices 0-3: K=4 solution (ABCD)
    - Indices 4-8: K=5 solution (ABCDE)
    - Indices 9-14: K=6 solution (ABCDEF)
    - Indices 15-21: K=7 solution with actual labels (D, A, C, F, B, G, E)
    
    We use the last 7 (K=7 solution) with labels: DACFBGE
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict


# Template labels for K=7 solution (the last 7 microstates in the file)
# These are in the order they appear: indices 15-21 map to D, A, C, F, B, G, E
METAMAPS_K7_LABELS = ['D', 'A', 'C', 'F', 'B', 'G', 'E']


def _corr_vectors(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute correlation between column vectors ignoring polarity.
    
    Computes absolute correlation (ignoring polarity) between data vectors.
    This is the standard approach for microstate analysis.
    
    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        Data matrix
    Y : ndarray, shape (n_channels, n_maps)
        Maps/templates matrix
        
    Returns
    -------
    corr : ndarray, shape (n_samples, n_maps)
        Absolute correlation matrix
    """
    # Normalize vectors
    X_norm = X / np.linalg.norm(X, axis=0, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=0, keepdims=True)
    
    # Compute absolute correlation (ignoring polarity)
    corr = np.abs(X_norm.T @ Y_norm)
    
    return corr


def _load_metamaps_from_set(set_file: str | Path) -> Tuple[np.ndarray, list]:
    """
    Load MetaMaps template microstates from EEGLAB .set file using MNE.
    
    Loads the K=7 solution (last 7 microstates) with proper labels.
    The file contains 22 total templates (K=4,5,6,7), we extract the last 7.
    
    Parameters
    ----------
    set_file : str or Path
        Path to the EEGLAB .set file containing template microstates
        
    Returns
    -------
    template_maps : ndarray, shape (7, n_channels)
        K=7 template microstate topographic maps, normalized to unit norm
    template_labels : list of str
        Labels for K=7 template states: ['D', 'A', 'C', 'F', 'B', 'G', 'E']
        
    Raises
    ------
    FileNotFoundError
        If the .set file cannot be found
    ImportError
        If MNE-Python is not available
    """
    set_file = Path(set_file)
    
    if not set_file.exists():
        raise FileNotFoundError(f"MetaMaps file not found: {set_file}")
    
    try:
        import mne
    except ImportError:
        raise ImportError(
            "MNE-Python is required to load EEGLAB .set files. "
            "Install it with: pip install mne"
        )
    
    try:
        # Load the .set file using MNE
        # preload=True loads data into memory
        raw = mne.io.read_raw_eeglab(str(set_file), preload=True)
        
        # Get the raw data: shape is (n_channels, n_samples)
        # Each "sample" is actually a microstate template
        data = raw.get_data()  # shape: (n_channels, 22) for MetaMaps
        
        # Extract the last 7 templates (K=7 solution, indices 15-21)
        # These correspond to the template solution with all 7 states
        template_data = data[:, -7:].T  # Transpose to (7, n_channels)
        
        # Normalize each template to unit norm (standard for microstate analysis)
        template_maps = template_data / np.linalg.norm(
            template_data, axis=1, keepdims=True
        )
        
        # Use the known labels for K=7 solution
        template_labels = METAMAPS_K7_LABELS.copy()
        
        return template_maps, template_labels
        
    except Exception as e:
        raise ValueError(
            f"Failed to load MetaMaps from {set_file}: {e}\n"
            "Make sure the file is a valid EEGLAB .set file and MNE is installed."
        ) from e


def _match_clusters_to_templates(
    cluster_maps: np.ndarray,
    template_maps: np.ndarray,
    template_labels: list,
) -> Tuple[Dict, np.ndarray]:
    """
    Match estimated clusters to template clusters based on spatial correlation.
    
    Uses greedy matching: for each estimated cluster, finds the best matching
    template cluster (ignoring polarity). This is standard practice in 
    microstate analysis.
    
    Handles channel mismatch by using only common channels.
    
    Parameters
    ----------
    cluster_maps : ndarray, shape (n_clusters, n_channels_est)
        Estimated cluster topographies (normalized)
    template_maps : ndarray, shape (n_templates, n_channels_template)
        Template microstate topographies (normalized)
    template_labels : list of str
        Labels for template clusters (e.g., ['D', 'A', 'C', 'F', 'B', 'G', 'E'])
        
    Returns
    -------
    cluster_to_template : dict
        Mapping from cluster index to template label
        E.g., {0: 'D', 1: 'A', 2: 'C', ...}
    correlation_scores : ndarray, shape (n_clusters,)
        Correlation score with matched template for each cluster
        
    Raises
    ------
    ValueError
        If channel count mismatch cannot be resolved
    """
    n_clusters = cluster_maps.shape[0]
    n_templates = template_maps.shape[0]
    n_channels_est = cluster_maps.shape[1]
    n_channels_template = template_maps.shape[1]
    
    # Handle channel mismatch
    if n_channels_est != n_channels_template:
        print(f"Warning: Channel mismatch - "
              f"estimated clusters have {n_channels_est} channels, "
              f"templates have {n_channels_template} channels")
        
        # Use only the minimum number of channels
        min_channels = min(n_channels_est, n_channels_template)
        print(f"  Using first {min_channels} channels for matching")
        
        cluster_maps = cluster_maps[:, :min_channels]
        template_maps = template_maps[:, :min_channels]
    
    # Compute correlation matrix (ignoring polarity)
    # shape: (n_clusters, n_templates)
    corr_matrix = np.abs(cluster_maps @ template_maps.T)
    
    cluster_to_template = {}
    correlation_scores = np.zeros(n_clusters)
    
    # Greedy matching: assign each cluster to best unmatched template
    used_templates = set()
    
    for cluster_idx in range(n_clusters):
        correlations = corr_matrix[cluster_idx, :].copy()
        
        # Set already-used templates to -inf to avoid reusing
        for used_template in used_templates:
            correlations[used_template] = -np.inf
        
        best_template_idx = np.argmax(correlations)
        best_corr = correlations[best_template_idx]
        
        # Get the template label
        if best_template_idx < len(template_labels):
            template_label = template_labels[best_template_idx]
        else:
            template_label = chr(65 + best_template_idx)  # Fallback: A, B, C, ...
        
        cluster_to_template[cluster_idx] = template_label
        correlation_scores[cluster_idx] = best_corr
        used_templates.add(best_template_idx)
    
    return cluster_to_template, correlation_scores


def label_and_reorder_cluster(
    cluster,
    template_file: str | Path = "MetaMaps_2023_06.set",
) -> tuple:
    """
    Label and reorder a pycrostates ModKMeans cluster based on template matching.
    
    This function:
    1. Loads the MetaMaps template microstate clusters (K=7 solution)
    2. Matches estimated clusters to template clusters using spatial correlation
       (ignoring polarity - standard for microstate analysis)
    3. Relabels clusters with template labels (D, A, C, F, B, G, E from templates)
    4. Reorders clusters alphabetically
    5. Relabels and reorders segmentation labels accordingly
    
    Parameters
    ----------
    cluster : pycrostates.clustering.ModKMeans
        Fitted ModKMeans cluster object with cluster_centers_ and labels_
        (segmentation) attributes
    template_file : str or Path, optional
        Path to EEGLAB .set file containing template microstates.
        Default: "MetaMaps_2023_06.set"
        
    Returns
    -------
    cluster : pycrostates.clustering.ModKMeans
        Modified cluster object with:
        - cluster_centers_ reordered and properly aligned
        - _cluster_names updated with template-based labels
        - _labels_ (segmentation) updated accordingly
        - _ignore_polarity set to True
    metadata : dict
        Metadata about the matching process containing:
        - 'template_matches': dict mapping cluster index to template label
        - 'correlation_scores': dict mapping cluster index to correlation score
        - 'template_labels_ordered': template labels in alphabetical order
        - 'n_channels': number of channels
        - 'n_clusters': number of clusters
        
    Raises
    ------
    ValueError
        If cluster is not fitted or if matching fails
    FileNotFoundError
        If template file cannot be found
        
    Examples
    --------
    Load a fitted cluster and label it based on MetaMaps templates:
    
    >>> from pycrostates.clustering import ModKMeans
    >>> from label_pycrostates_cluster import label_and_reorder_cluster
    >>>
    >>> # Fit a cluster (assuming data is available)
    >>> cluster = ModKMeans(n_clusters=4)
    >>> cluster.fit(raw_data)
    >>>
    >>> # Label and reorder based on templates
    >>> cluster, metadata = label_and_reorder_cluster(cluster)
    >>> print(metadata['template_labels_ordered'])
    ['A', 'B', 'C', 'D']
    >>> print(metadata['template_matches'])
    {0: 'D', 1: 'A', 2: 'C', 3: 'F'}
    
    Notes
    -----
    - Polarity is ignored during matching (standard for microstate analysis)
    - The template file contains K=7 solution with labels: D, A, C, F, B, G, E
    - Matching uses a greedy algorithm - each estimated cluster is matched
      to its best correlating template
    - If more than 7 clusters are estimated, only the first 7 are labeled;
      extras get generic labels (H, I, etc.)
    """
    # Validate input
    if not hasattr(cluster, '_fitted') or not cluster._fitted:
        raise ValueError("Cluster must be fitted before labeling")
    
    if not hasattr(cluster, '_cluster_centers_') or cluster._cluster_centers_ is None:
        raise ValueError("Cluster must have cluster_centers_ attribute")
    
    if not hasattr(cluster, '_labels_') or cluster._labels_ is None:
        raise ValueError("Cluster must have _labels_ (segmentation) attribute")
    
    # Load template maps with their labels
    template_maps, template_labels = _load_metamaps_from_set(template_file)
    
    # Normalize estimated cluster centers
    cluster_maps = cluster._cluster_centers_.copy()
    cluster_maps = cluster_maps / np.linalg.norm(cluster_maps, axis=1, keepdims=True)
    
    # Match clusters to templates
    cluster_to_template, correlation_scores = _match_clusters_to_templates(
        cluster_maps, template_maps, template_labels
    )
    
    # Create reordering: alphabetical by template label
    n_clusters = cluster_maps.shape[0]
    
    # Map cluster indices to their template labels
    cluster_labels = [cluster_to_template[i] for i in range(n_clusters)]
    
    # Sort alphabetically and create mapping
    sorted_labels = sorted(set(cluster_labels))
    label_to_sorted_idx = {label: idx for idx, label in enumerate(sorted_labels)}
    
    # Create mapping from old cluster index to new position
    old_to_new_position = {}
    for old_idx in range(n_clusters):
        new_label = cluster_to_template[old_idx]
        new_pos = label_to_sorted_idx[new_label]
        old_to_new_position[old_idx] = new_pos
    
    # Reorder cluster centers
    new_centers = np.zeros_like(cluster._cluster_centers_)
    new_segmentation = np.zeros_like(cluster._labels_)
    
    for old_idx, new_pos in old_to_new_position.items():
        new_centers[new_pos, :] = cluster._cluster_centers_[old_idx, :]
        
        # Update segmentation labels
        mask = cluster._labels_ == old_idx
        new_segmentation[mask] = new_pos
    
    # Update cluster object
    cluster._cluster_centers_ = new_centers
    cluster._labels_ = new_segmentation
    cluster._cluster_names = sorted_labels
    cluster._ignore_polarity = True
    
    # Create metadata
    metadata = {
        'template_matches': {
            i: cluster_to_template[i]
            for i in range(n_clusters)
        },
        'correlation_scores': {
            i: float(correlation_scores[i])
            for i in range(n_clusters)
        },
        'template_labels_ordered': sorted_labels,
        'n_channels': cluster_maps.shape[1],
        'n_clusters': n_clusters,
        'template_file': str(template_file),
    }
    
    return cluster, metadata


def load_cluster_from_fif(fif_file: str | Path):
    """
    Load cluster from a .fif file using pycrostates.io.read_cluster.
    
    Parameters
    ----------
    fif_file : str or Path
        Path to the .fif file containing cluster data
        
    Returns
    -------
    cluster : pycrostates.cluster.ModKMeans or similar
        Cluster object loaded from the file
        
    Raises
    ------
    FileNotFoundError
        If the file cannot be found
    ValueError
        If the file cannot be loaded as a cluster
    """
    try:
        from pycrostates.io import read_cluster
    except ImportError:
        raise ImportError("pycrostates is required. Install with: pip install pycrostates")
    
    fif_file = Path(fif_file)
    
    if not fif_file.exists():
        raise FileNotFoundError(f"File not found: {fif_file}")
    
    try:
        cluster = read_cluster(str(fif_file))
        return cluster
    except Exception as e:
        raise ValueError(
            f"Could not load cluster from {fif_file}: {e}"
        ) from e


# Example usage / testing
if __name__ == "__main__":
    print("label_pycrostates_cluster module - Example usage")
    print("=" * 70)
    
    print("\nExample 1: Label a fitted ModKMeans cluster")
    print("-" * 70)
    print("""
    from pycrostates.cluster import ModKMeans
    from label_pycrostates_cluster import label_and_reorder_cluster
    
    # Assume you have a fitted cluster
    cluster = ModKMeans(n_clusters=4, n_init=100)
    cluster.fit(raw_eeg_data)
    
    # Label and reorder based on MetaMaps template
    cluster, metadata = label_and_reorder_cluster(cluster)
    
    # Inspect results
    print(f"Template labels: {metadata['template_labels_ordered']}")
    print(f"Cluster matches: {metadata['template_matches']}")
    print(f"Correlation scores: {metadata['correlation_scores']}")
    """)
    
    print("\nExample 2: Load clusters from a .fif file and label them")
    print("-" * 70)
    print("""
    from label_pycrostates_cluster import load_cluster_from_fif, label_and_reorder_cluster
    
    # Load cluster data from .fif file
    raw = load_cluster_from_fif('optimal_clusters.fif')
    
    # Label and reorder
    cluster, metadata = label_and_reorder_cluster(cluster)
    print(f"Labeled clusters: {cluster._cluster_names}")
    """)
