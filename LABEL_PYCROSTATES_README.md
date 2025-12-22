# Label and Reorder pycrostates Clusters

A Python module for labeling and reordering pycrostates `ModKMeans` cluster objects based on spatial correlation with the MetaMaps microstate template.

## Overview

This module provides a single, easy-to-use function that:

1. **Loads the MetaMaps template** - Extracts the K=7 microstate solution from the EEGLAB .set file
2. **Matches estimated clusters to templates** - Uses spatial correlation (ignoring polarity) to match each estimated cluster to the best-matching template
3. **Relabels clusters** - Assigns template labels (D, A, C, F, B, G, E) based on the matches
4. **Reorders alphabetically** - Sorts the clusters alphabetically for consistency

## Features

- ✅ Uses MNE-Python for robust EEGLAB .set file loading
- ✅ Polarity-invariant matching (standard for microstate analysis)
- ✅ Greedy one-to-one matching algorithm
- ✅ Proper handling of segmentation labels during reordering
- ✅ Comprehensive metadata about matching quality

## Installation

Requires Python 3.8+

```bash
pip install mne pycrostates numpy
```

## Usage

### Basic usage

```python
from pycrostates.cluster import ModKMeans
from label_pycrostates_cluster import label_and_reorder_cluster

# Assume you have a fitted ModKMeans cluster
cluster = ModKMeans(n_clusters=4, n_init=100)
cluster.fit(raw_eeg_data)

# Label and reorder based on MetaMaps template
cluster, metadata = label_and_reorder_cluster(cluster)

# Access the results
print(metadata['template_labels_ordered'])  # ['A', 'C', 'D', 'F']
print(metadata['template_matches'])          # {0: 'A', 1: 'C', 2: 'D', 3: 'F'}
print(metadata['correlation_scores'])        # {0: 0.92, 1: 0.89, 2: 0.91, 3: 0.88}
```

### Custom template file location

```python
cluster, metadata = label_and_reorder_cluster(
    cluster,
    template_file="/path/to/MetaMaps_2023_06.set"
)
```

## API Reference

### `label_and_reorder_cluster(cluster, template_file="MetaMaps_2023_06.set")`

Main function for labeling and reordering pycrostates clusters.

**Parameters:**
- `cluster` (pycrostates.cluster.ModKMeans): Fitted ModKMeans cluster object
- `template_file` (str or Path): Path to the MetaMaps template .set file

**Returns:**
- `cluster` (pycrostates.cluster.ModKMeans): Modified cluster object with:
  - `_cluster_centers_`: Reordered and properly aligned
  - `_cluster_names`: Updated with template labels
  - `_labels_`: Segmentation updated accordingly
  - `_ignore_polarity`: Set to True
  
- `metadata` (dict): Dictionary containing:
  - `template_matches`: Dict mapping cluster index → template label
  - `correlation_scores`: Dict mapping cluster index → correlation score (0-1)
  - `template_labels_ordered`: List of template labels in alphabetical order
  - `n_channels`: Number of channels
  - `n_clusters`: Number of clusters
  - `template_file`: Path to the template file used

**Raises:**
- `ValueError`: If cluster is not fitted
- `FileNotFoundError`: If template file cannot be found
- `ImportError`: If MNE-Python is not installed

## MetaMaps Template Structure

The MetaMaps_2023_06.set file contains 22 microstate templates representing K=4,5,6,7 solutions:
- Indices 0-3: K=4 solution (ABCD)
- Indices 4-8: K=5 solution (ABCDE)
- Indices 9-14: K=6 solution (ABCDEF)
- Indices 15-21: K=7 solution with labels (D, A, C, F, B, G, E)

This module uses the **last 7** templates (K=7 solution) with the actual labels: **D, A, C, F, B, G, E**

## Matching Algorithm

The algorithm uses **greedy one-to-one matching**:

1. Compute absolute spatial correlation between each estimated cluster and all template maps (ignoring polarity)
2. For each estimated cluster:
   - Find the template with highest correlation that hasn't been matched yet
   - Assign the template's label to this cluster
3. Reorder alphabetically by assigned label

## Example Output

Given a fitted 4-cluster model:

```
Template matching results:
  Cluster 0 → Template D (correlation: 0.9346)
  Cluster 1 → Template A (correlation: 0.9228)
  Cluster 2 → Template C (correlation: 0.9261)
  Cluster 3 → Template F (correlation: 0.9172)

After reordering:
  Cluster 0: A
  Cluster 1: C
  Cluster 2: D
  Cluster 3: F
```

## Requirements

- **numpy**: Numerical operations
- **mne**: EEGLAB .set file loading
- **pycrostates**: ModKMeans cluster class

## References

- **pycrostates**: https://github.com/vferat/pycrostates
- **MNE-Python**: https://mne.tools
- **Microstate analysis**: https://github.com/ThomasKoenigBern/MS-Template-Explorer

## License

This module is provided as-is for research purposes.
