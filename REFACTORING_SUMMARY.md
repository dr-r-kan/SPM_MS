# Refactoring Summary

## Date: 2025-11-10

## Overview
This refactoring simplifies the codebase, removes redundant methods, and makes the pipeline more robust for real EEG data analysis.

## Changes Made

### 1. Files Deleted
- `fit_microstate_vb_kmeans.m` - Redundant VB k-means implementation (superseded by standard kmeans_koenig)
- `diag_kmeans.m` - Diagnostic script with references to non-existent functions
- `plot_microstates.asv` - MATLAB backup file

### 2. Files Modified

#### `Bayesian_MS_Comparison_Pipeline.m`
- **Reduced methods**: Changed from 4 methods to 2 core methods
  - Removed: `vb_kmeans`, `dp_mixture`
  - Kept: `kmeans_koenig`, `spm_vb`
- **GEV criterion filtering**: Added logic to skip GEV criterion for non-kmeans methods
- **Cleaner pipeline**: Removed method references in both parallel and sequential execution paths

#### `fit_microstate_kmeans_koenig.m`
- **Ground truth handling**: Added conditional checks for missing ground truth maps
- **Defensive field access**: Safe access to K_true, SNR_dB, duration_s fields
- **Empty recovery metrics**: Creates proper empty structure when ground truth unavailable
- **Improved logging**: Shows "Real data mode" when ground truth missing

#### `fit_microstate_spm_vb.m`
- **GEV validation**: Added error when GEV criterion is used (not supported for VB methods)
- **Ground truth handling**: Added conditional checks for missing ground truth maps
- **Bug fix**: Initialized `gev_vals` array (was previously uninitialized)
- **Defensive field access**: Safe access to K_true, SNR_dB, duration_s fields
- **Error path fixes**: Set gev_vals = 0 in error cases

#### `fit_microstate_dp_mixture.m`
- **Ground truth handling**: Added conditional checks (for consistency, though not used in main pipeline)
- **Defensive printing**: Safe printing of K_true when it may be NaN

### 3. Files Created

#### `analyze_single_eeg_file.m`
- **New functionality**: Standalone script for analyzing real EEG files
- **Flexible input**: Supports .set (EEGLAB) and .mat files
- **Optional ground truth**: Can run with or without ground truth maps
- **Auto-detection**: Automatically selects appropriate criterion based on method
- **Validation**: Prevents invalid method-criterion combinations (e.g., spm_vb + gev)
- **Clear output**: Displays results with recovery metrics when available

## Method Comparison: Evolution

### Original (Before November 2025)
- **4 methods**: kmeans_koenig, spm_vb, vb_kmeans, dp_mixture
- **Issues**:
  - vb_kmeans was redundant (similar to kmeans_koenig)
  - dp_mixture not fully validated
  - GEV criterion applied to all methods (invalid for VB)
  - No support for real EEG data without ground truth

### Refactored (November 2025)
- **2 core methods**: kmeans_koenig, spm_vb
- **Benefits**:
  - Clearer method distinction (classical k-means vs Bayesian VB)
  - GEV criterion only for k-means (as intended)
  - Robust handling of real EEG data
  - Simpler codebase, easier to maintain

### Current (December 2025) - Three-Method Comparison
- **3 methods**: kmeans_koenig, spm_kmeans, spm_vb
- **New addition: SPM K-means**
  - Implements K-means as GMM limit case (isotropic covariance, σ² → 0)
  - Validates theoretical equivalence: GMM → K-means
  - Uses SPM framework with hard assignments
  - Bridges classical and Bayesian approaches
- **Method relationships**:
  - **Koenig K-means** ↔ **SPM K-means**: Should show similar results (validates equivalence)
  - **SPM K-means** vs **VB GMM**: Demonstrates benefit of full covariance modeling
  - **Complete spectrum**: Classical → GMM Limit → Full Bayesian

## Supported Criteria by Method

| Criterion | kmeans_koenig | spm_kmeans | spm_vb |
|-----------|---------------|------------|---------|
| silhouette | ✓ | ✓ | ✓ |
| gev | ✓ | ✓ | ✗ (error) |
| elbow | ✓ | ✓ | ✓ |
| free_energy | ✗ (returns NaN) | ✗ (error) | ✓ |
| elbow_sil_combined | ✗ (returns NaN) | ✗ (error) | ✓ |

**Note**: SPM K-means explicitly rejects `free_energy` and `elbow_sil_combined` as they are not meaningful for degenerate GMMs (infinitesimal variance).

## Error Handling Improvements

1. **Missing ground truth**: All methods gracefully handle missing ground truth
2. **Invalid criteria**: spm_vb explicitly rejects GEV with clear error message
3. **Pipeline robustness**: GEV automatically skipped for non-kmeans methods
4. **Empty results**: Pipeline handles case where no valid results exist
5. **Field validation**: All optional Sim fields checked before access

## Usage Examples

### Running the main pipeline (synthetic data)
```matlab
% Runs with 2 methods: kmeans_koenig and spm_vb
T = Bayesian_MS_Comparison_Pipeline('reps', 1, 'K_true_vals', [4], 'SNR_dbs', [10]);
```

### Analyzing real EEG data
```matlab
% With kmeans (default)
Results = analyze_single_eeg_file('my_eeg.set');

% With SPM VB
Results = analyze_single_eeg_file('my_eeg.set', 'method', 'spm_vb');

% With ground truth for validation
Results = analyze_single_eeg_file('my_eeg.set', 'true_maps', ground_truth);

% SPM K-means method (GMM limit case)
Results = analyze_single_eeg_file('my_eeg.set', 'method', 'spm_kmeans');
```

## SPM K-Means Addition (December 2025)

### New File: `fit_microstate_spm_kmeans.m`

**Purpose**: Implements K-means as the limit case of Gaussian Mixture Models with:
- Isotropic (spherical) covariance constraint
- Infinitesimal variance (σ² = 1e-6) to force hard assignments
- SPM's `spm_mix` framework for consistency

**Key Features**:
1. **PCA dimensionality reduction** (same as `spm_vb`)
2. **GMM with isotropic covariance** via SPM
3. **Hard assignments** through argmax of responsibilities
4. **Koenig-compatible metrics**: GEV, silhouette, elbow detection
5. **Polarity-insensitive** silhouette calculation
6. **Recovery metrics** via `microstate_partial_alignment`

**Supported Criteria**:
- ✓ `silhouette`: Polarity-insensitive cosine-based
- ✓ `gev`: Global Explained Variance (Koenig method)
- ✓ `elbow`: Within-cluster sum of squares curvature
- ✗ `free_energy`: Not meaningful for degenerate GMM (explicitly rejected)
- ✗ `elbow_sil_combined`: Not supported (explicitly rejected)

**Mathematical Foundation**:
- References arXiv:1704.04812: "k-means as variational EM approximation of GMMs"
- References Celeux & Govaert, 1992: "A classification EM algorithm"
- Validates EII model (spherical, equal volume) ≈ K-means

### Integration Changes

**`Bayesian_MS_Comparison_Pipeline.m`**:
- Line 93: Added `'spm_kmeans'` to `method_names`
- Lines 244-246: Added method dispatch for `'spm_kmeans'`
- Criterion filtering: Automatically handled via `select_K_by_criterion` (returns NaN for unsupported criteria)

**`microstate_utilities_SHARED.m`**:
- Lines 221-222: Added display name formatting: `'spm_kmeans'` → `'SPM K-means'`

**Documentation**:
- `README.md`: Updated to explain 3-method comparison and theoretical foundation
- `REFACTORING_SUMMARY.md`: Documented evolution and method relationships

### Expected Behavior

1. **Equivalence validation**: `kmeans_koenig` and `spm_kmeans` should produce similar results
2. **Performance comparison**: `spm_kmeans` vs `spm_vb` demonstrates value of full covariance
3. **Complete spectrum**: Classical → GMM Limit → Full Bayesian
4. **Automatic integration**: All existing analysis tools work with 3 methods

## Testing

All changes preserve backward compatibility for:
- Pipeline execution with synthetic data
- Recovery metrics calculation
- Results structure format
- CSV output format

New functionality:
- Real EEG analysis without ground truth
- Explicit GEV validation

## Future Work

The following files remain in the repository but are not used in the main pipeline:
- `fit_microstate_dp_mixture.m` - For future research on Dirichlet Process methods
- `analyze_comparison_results.m` - Results analysis utilities

These can be cleaned up or integrated in future updates.
