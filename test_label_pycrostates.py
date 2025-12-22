"""
Test script for label_pycrostates_cluster module
"""

import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from label_pycrostates_cluster import (
    _load_metamaps_from_set,
    _match_clusters_to_templates,
    label_and_reorder_cluster,
    METAMAPS_K7_LABELS,
)


def test_load_metamaps():
    """Test loading MetaMaps from .set file"""
    print("=" * 70)
    print("TEST 1: Loading MetaMaps template")
    print("=" * 70)
    
    template_file = "MetaMaps_2023_06.set"
    
    try:
        template_maps, template_labels = _load_metamaps_from_set(template_file)
        
        print(f"✓ Successfully loaded MetaMaps")
        print(f"  Shape: {template_maps.shape}")
        print(f"  Expected: (7, n_channels)")
        print(f"  Template labels: {template_labels}")
        print(f"  Expected: {METAMAPS_K7_LABELS}")
        
        # Verify shape
        assert template_maps.shape[0] == 7, f"Expected 7 templates, got {template_maps.shape[0]}"
        
        # Verify labels
        assert template_labels == METAMAPS_K7_LABELS, "Labels don't match expected"
        
        # Verify normalization
        norms = np.linalg.norm(template_maps, axis=1)
        print(f"\n  Normalization check (all should be ~1.0):")
        for i, (label, norm) in enumerate(zip(template_labels, norms)):
            print(f"    Template {label}: norm = {norm:.6f}")
            assert np.isclose(norm, 1.0), f"Template {label} not normalized"
        
        print("\n✓ All tests passed for template loading")
        return True
        
    except Exception as e:
        print(f"✗ Error loading MetaMaps: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matching():
    """Test cluster matching to templates"""
    print("\n" + "=" * 70)
    print("TEST 2: Cluster matching")
    print("=" * 70)
    
    try:
        # Load templates
        template_maps, template_labels = _load_metamaps_from_set("MetaMaps_2023_06.set")
        
        # Create synthetic cluster maps (simulate 4 estimated clusters)
        # by randomly selecting 4 templates and slightly perturbing them
        np.random.seed(42)
        selected_indices = [0, 2, 5, 1]  # D, C, G, A from the 7 templates
        cluster_maps = template_maps[selected_indices, :].copy()
        
        # Add small noise while maintaining normalization
        noise = np.random.randn(*cluster_maps.shape) * 0.05
        cluster_maps = cluster_maps + noise
        cluster_maps = cluster_maps / np.linalg.norm(cluster_maps, axis=1, keepdims=True)
        
        print(f"Created synthetic cluster maps with shape: {cluster_maps.shape}")
        print(f"Based on templates: {[template_labels[i] for i in selected_indices]}")
        
        # Match clusters
        cluster_to_template, corr_scores = _match_clusters_to_templates(
            cluster_maps, template_maps, template_labels
        )
        
        print(f"\nMatching results:")
        for cluster_idx, template_label in cluster_to_template.items():
            corr = corr_scores[cluster_idx]
            print(f"  Cluster {cluster_idx} → Template {template_label} (correlation: {corr:.4f})")
        
        # Verify matching
        matched_labels = set(cluster_to_template.values())
        assert len(matched_labels) == len(cluster_to_template), "Duplicate template assignments!"
        
        print("\n✓ All tests passed for matching")
        return True
        
    except Exception as e:
        print(f"✗ Error in matching test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full labeling and reordering pipeline"""
    print("\n" + "=" * 70)
    print("TEST 3: Full pipeline (requires pycrostates cluster object)")
    print("=" * 70)
    
    try:
        from pycrostates.cluster import ModKMeans
        import mne
        
        print("✓ pycrostates and mne available")
        print("  Creating synthetic EEG data for testing...")
        
        # Create synthetic EEG data
        n_channels = 71  # Same as MetaMaps
        n_samples = 5000
        sfreq = 250
        
        # Create random EEG-like data
        rng = np.random.RandomState(42)
        data = rng.randn(n_channels, n_samples) * 10  # microvolts
        
        # Create info object
        ch_names = [f'Ch{i:02d}' for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        # Create Raw object
        raw = mne.io.RawArray(data, info)
        
        print(f"  Created synthetic raw data: {raw.get_data().shape}")
        
        # Fit a cluster
        print("  Fitting ModKMeans cluster with K=4...")
        cluster = ModKMeans(n_clusters=4, n_init=10, random_state=42)
        cluster.fit(raw)
        
        print(f"  ✓ Cluster fitted")
        print(f"    - Cluster centers shape: {cluster._cluster_centers_.shape}")
        print(f"    - Cluster names (before): {cluster._cluster_names}")
        
        # Label and reorder
        print("  Labeling and reordering cluster...")
        cluster, metadata = label_and_reorder_cluster(cluster)
        
        print(f"  ✓ Cluster labeled and reordered")
        print(f"    - Cluster names (after): {cluster._cluster_names}")
        print(f"    - Template labels (ordered): {metadata['template_labels_ordered']}")
        print(f"\n  Matching details:")
        for cluster_idx, template_label in metadata['template_matches'].items():
            corr = metadata['correlation_scores'][cluster_idx]
            print(f"    Cluster {cluster_idx} → Template {template_label} (correlation: {corr:.4f})")
        
        print("\n✓ Full pipeline test passed")
        return True
        
    except ImportError:
        print("⚠ pycrostates or mne not available, skipping full pipeline test")
        return True
    except Exception as e:
        print(f"✗ Error in full pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing label_pycrostates_cluster module")
    print("=" * 70)
    
    results = []
    results.append(("Loading MetaMaps", test_load_metamaps()))
    results.append(("Cluster matching", test_matching()))
    results.append(("Full pipeline", test_full_pipeline()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
