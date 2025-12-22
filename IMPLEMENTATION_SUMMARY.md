# Implementation Summary: label_pycrostates_cluster

## What was created

A complete Python module for labeling and reordering pycrostates ModKMeans cluster objects based on the MetaMaps microstate template.

## Files created

1. **label_pycrostates_cluster.py** (364 lines)
   - Main module with all functionality
   - Fully documented with docstrings
   - Type hints for all functions

2. **test_label_pycrostates.py** (186 lines)
   - Comprehensive test suite
   - Tests template loading
   - Tests cluster matching
   - Tests full pipeline with synthetic data
   - All tests passing ✓

3. **LABEL_PYCROSTATES_README.md**
   - User documentation
   - Installation instructions
   - API reference
   - Usage examples

## Key Implementation Details

### MetaMaps Template Structure

The MetaMaps_2023_06.set file contains 22 microstate templates:
- Last 7 templates (indices 15-21) are the K=7 solution
- These 7 templates have specific labels: **D, A, C, F, B, G, E**

### Core Functions

1. **`_load_metamaps_from_set(set_file)`**
   - Loads MetaMaps using MNE-Python
   - Extracts the K=7 solution (last 7 templates)
   - Normalizes to unit norm
   - Returns templates with proper labels

2. **`_match_clusters_to_templates(cluster_maps, template_maps, template_labels)`**
   - Computes spatial correlation ignoring polarity (absolute correlation)
   - Implements greedy one-to-one matching
   - Returns mapping of clusters to template labels
   - Returns correlation scores for each match

3. **`label_and_reorder_cluster(cluster, template_file)`**
   - Main user-facing function
   - Takes a fitted pycrostates ModKMeans cluster
   - Labels it based on template matches
   - Reorders alphabetically
   - Returns modified cluster and metadata

### Key Features

✅ **Polarity-invariant matching** - Uses absolute correlation (standard for microstate analysis)

✅ **Greedy matching** - Each cluster matched to best unmatched template, ensuring one-to-one mapping

✅ **Proper segmentation handling** - When clusters are reordered, segmentation labels are updated consistently

✅ **Comprehensive metadata** - Returns:
- Template matches (cluster → label)
- Correlation scores (0-1 range)
- Ordered labels
- Channel and cluster counts

✅ **Error handling** - Validates input, provides clear error messages

## Test Results

All tests passing:

```
TEST 1: Loading MetaMaps template
  ✓ Loads 7 templates with 71 channels
  ✓ Correct labels: ['D', 'A', 'C', 'F', 'B', 'G', 'E']
  ✓ All templates properly normalized

TEST 2: Cluster matching
  ✓ Creates synthetic 4-cluster maps
  ✓ Correctly matches to best template
  ✓ Correlations in expected range (0.9+)

TEST 3: Full pipeline
  ✓ Fits ModKMeans on synthetic EEG
  ✓ Labels and reorders cluster
  ✓ Produces correct output format
```

## Usage Example

```python
from pycrostates.cluster import ModKMeans
from label_pycrostates_cluster import label_and_reorder_cluster

# Fit a cluster
cluster = ModKMeans(n_clusters=4, n_init=100)
cluster.fit(raw_eeg_data)

# Label and reorder
cluster, metadata = label_and_reorder_cluster(cluster)

# Results:
# cluster._cluster_names = ['A', 'C', 'D', 'F']  (alphabetically ordered)
# metadata['correlation_scores'] = {0: 0.92, 1: 0.89, ...}
```

## Dependencies

- numpy: Numerical operations
- mne: EEGLAB .set file loading
- pycrostates: ModKMeans cluster class

All installed and tested ✓

## Notes on Implementation

### Template Labels (D, A, C, F, B, G, E)

The template labels are **not alphabetical** but are the **empirical labels from the MetaMaps publication**. 
The last 7 templates in the file (K=7 solution) have these specific labels in order.

The module:
1. Matches clusters to these template labels
2. Then reorders clusters **alphabetically** (A, B, C, D, E, F, G)
3. This ensures consistent cluster ordering while preserving template identity information in metadata

### Why Polarity-Invariant?

Microstate maps have inherent polarity ambiguity—a map and its inverse represent the same microstate.
Using absolute correlation (ignoring sign) is the standard approach in microstate analysis.

### Greedy Matching

The algorithm ensures each template is used at most once:
- Process clusters in order
- For each cluster, assign the best-correlating unused template
- This is optimal for one-to-one assignment under greedy constraints

## Future Enhancements (if needed)

- Support for custom template files
- Batch processing of multiple clusters
- Visualization of matched templates
- Hungarian algorithm for globally optimal matching
- Handling of more than 7 clusters

## Files Modified

- label_pycrostates_cluster.py: Created ✓
- test_label_pycrostates.py: Created ✓
- LABEL_PYCROSTATES_README.md: Created ✓

## Ready for Use

The module is production-ready and can be imported and used as:

```python
from label_pycrostates_cluster import label_and_reorder_cluster
```
