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

## Method Comparison: Before vs After

### Before
- **4 methods**: kmeans_koenig, spm_vb, vb_kmeans, dp_mixture
- **Issues**:
  - vb_kmeans was redundant (similar to kmeans_koenig)
  - dp_mixture not fully validated
  - GEV criterion applied to all methods (invalid for VB)
  - No support for real EEG data without ground truth

### After
- **2 core methods**: kmeans_koenig, spm_vb
- **Benefits**:
  - Clearer method distinction (classical k-means vs Bayesian VB)
  - GEV criterion only for k-means (as intended)
  - Robust handling of real EEG data
  - Simpler codebase, easier to maintain

## Supported Criteria by Method

| Criterion | kmeans_koenig | spm_vb |
|-----------|---------------|---------|
| silhouette | ✓ | ✓ |
| gev | ✓ | ✗ (error) |
| elbow | ✓ | ✓ |
| free_energy | ✗ (returns NaN) | ✓ |
| elbow_sil_combined | ✗ (returns NaN) | ✓ |

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
```

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
