# Quick Start: Labeling Clusters from .fif Files

## Usage

To label clusters from an optimal_clusters.fif file (or any pycrostates-compatible .fif file):

```bash
python label_clusters_from_fif.py optimal_clusters.fif
```

Or in Python:

```python
from label_clusters_from_fif import label_clusters

cluster, metadata = label_clusters('optimal_clusters.fif')

# Access results
print(f"Cluster labels: {cluster._cluster_names}")
print(f"Template matches: {metadata['template_matches']}")
print(f"Correlation scores: {metadata['correlation_scores']}")
```

## What it does

1. **Loads the cluster** from the .fif file using `pycrostates.io.read_cluster()`
2. **Matches to MetaMaps template** by computing spatial correlation
3. **Assigns labels** based on best-matching template microstates
4. **Reorders alphabetically** for consistency

## Output

The script prints:
- Number of clusters and channels
- Template matches for each cluster with correlation scores
- Overall match quality assessment

## Handle

If your clusters have a different number of channels than the MetaMaps template (71 channels):
- The script automatically uses only the first N common channels
- It prints a warning showing which channels were used
- This is fine - the correlation is still meaningful across the subset of channels

## Example Output

```
Cluster labels: ['A', 'B', 'C', 'D', 'E']

Template matches:
    Estimated cluster 0 → Template C (correlation: 0.2261)
    Estimated cluster 1 → Template E (correlation: 0.4096)
    Estimated cluster 2 → Template A (correlation: 0.3046)
    Estimated cluster 3 → Template B (correlation: 0.1986)
    Estimated cluster 4 → Template D (correlation: 0.1606)

Correlation scores:
  Mean: 0.2599
  Min:  0.1606
  Max:  0.4096
  Match quality: ✗ Low
```

## Understanding the Results

- **Higher correlation = better match** to the template microstate
- Scores range from 0 to 1
  - > 0.7: Good match
  - > 0.5: Moderate match
  - < 0.5: Low match

## Files

- `label_pycrostates_cluster.py` - Main module with all functions
- `label_clusters_from_fif.py` - Command-line wrapper for easy use
- `test_label_pycrostates.py` - Test suite

## Key Functions

### `label_and_reorder_cluster(cluster, template_file='MetaMaps_2023_06.set')`

Main function to label and reorder a pycrostates cluster.

**Input:**
- `cluster`: fitted pycrostates ModKMeans or similar cluster object

**Output:**
- `cluster`: modified cluster with new labels
- `metadata`: dict with matching information

### `load_cluster_from_fif(fif_file)`

Load a cluster from a .fif file using pycrostates' built-in reader.

**Input:**
- `fif_file`: path to .fif file

**Output:**
- `cluster`: pycrostates cluster object
