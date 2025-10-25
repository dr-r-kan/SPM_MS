# Variational Bayes-based microstate extraction and cluster number optimisation

A comprehensive comparison of Variational Bayes and classical methods for EEG microstate clustering and automatic K selection.

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)

[![SPM](https://github.com/spm/spm)]

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project addresses a fundamental challenge in EEG microstate analysis: **automatic determination of the optimal number of microstates (K)**. We compare multiple Variational Bayes (VB) approaches against classical methods.

### Key Innovation: The Elbow in the ELBO

The **ELBO** (Evidence Lower BOund) is the free energy in variational Bayes—a principled measure of model quality. However, simply maximizing ELBO tends to overfit (preferring complex models). Our solution:

**Detect the "elbow" (knee point) in the ELBO curve** where gains diminish—the sweet spot before overfitting. This is an implicit measure of the statistics of the fit itself, rather than a post-hoc heuristic, and so is an elegant solution for K finding.

This provides a principled, data-driven way to select K without manual inspection, combining the rigor of VB with practical robustness.

## Motivation

Traditional microstate analysis relies on:
- **Modified K-means** with polarity invariance (field standard)
- **Manual K selection** or heuristics (silhouette, GEV)
- **Visual inspection** of topographies

Can we do better with **Variational Bayes**?

This project systematically compares:
1. **VB approaches** (SPM's VB-GMM, VB K-means, Dirichlet Process)
2. **Classical approaches** (standard modified K-means)
3. **Model selection criteria** (ELBO elbow, silhouette, GEV)

Using **synthetic data with known ground truth**, we can definitively answer: which method works best?

## Features

✅ **Synthetic EEG generation** with realistic microstate structure  
✅ **Multiple VB methods** leveraging SPM's mixture toolbox for Bayesian methods and free energy minimisation
✅ **Elbow detection** on ELBO curves  
✅ **Comprehensive comparison** across SNR levels and true K values  
✅ **Recovery metrics** even when K is misestimated  

## Installation

### Prerequisites

- **MATLAB R2020a or later**
- **[SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)** (Statistical Parametric Mapping)

## Methods Compared

### 1. SPM VB-GMM (Our Main Focus)

**File:** `fit_microstate_spm_vb.m`

Uses SPM's `spm_mix` (proper Variational Bayes Gaussian Mixture Model) with:
- **PCA-reduced feature space** (95% variance)
- **Elbow detection on ELBO curve** (novel contribution)
- **Optional silhouette combination** (60% elbow + 40% silhouette)

**Criteria:**
- `elbow_sil_combined` - Combined elbow + silhouette (default)
- `elbow_only` - Pure elbow detection

**Advantages:**
- Principled Bayesian framework
- Automatic complexity control
- No manual tuning

### 2. Standard Modified K-Means

**File:** `fit_microstate_kmeans_standard.m`

The field standard for microstate analysis:
- **Polarity-invariant distance** (|correlation|)
- **Multiple random initializations** (10 restarts)
- **Classical model selection criteria**

**Criteria:**
- `silhouette` - Silhouette score (measures cluster separation)
- `gev` - Global Explained Variance (microstate literature standard)
- `elbow` - Elbow on within-cluster sum of squares

**Advantages:**
- Fast and interpretable
- Well-established in literature
- No dependencies

### 3. VB K-Means (Custom)

**File:** `fit_microstate_vb_kmeans.m`

Variational Bayes version of K-means:
- **Soft assignments** via responsibilities
- **Hyperparameter priors** (Dirichlet, precision)
- **Free energy objective**

**Criteria:**
- `free_energy` - Maximize variational free energy
- `silhouette` - Silhouette score

### 4. Dirichlet Process Mixture

**File:** `fit_microstate_dp_mixture.m`

Non-parametric Bayesian approach:
- **Automatic K determination** (stick-breaking)
- **Infinite mixture model** (truncated at K_max)
- **Concentration parameter** controls complexity

**Criteria:**
- `free_energy` - Maximize free energy
- `silhouette` - Silhouette score

## Key Files

| File | Description |
|------|-------------|
| `VBGMM_MS_Comparison_Pipeline.m` | **Main pipeline** - Runs full comparison |
| `generate_microstate_eeg.m` | **Synthetic data generator** with known ground truth |
| `fit_microstate_spm_vb.m` | **SPM VB-GMM** with elbow detection (main method) |
| `fit_microstate_kmeans_standard.m` | **Standard K-means** (baseline) |
| `fit_microstate_vb_kmeans.m` | **VB K-means** (custom implementation) |
| `fit_microstate_dp_mixture.m` | **Dirichlet Process** mixture model |

## Understanding Results

### Output Structure

```
./out_microstate_comparison_final/
├── results/
│   ├── comparison_results_final.csv    # Summary table
│   ├── run_001_K5_SNR+0_spm_vb_elbow_sil_combined.mat
│   ├── run_002_K5_SNR+0_kmeans_standard_silhouette.mat
│   └── ...
└── plots/
    └── method_comparison_final.png     # Visualization
```

### Key Metrics

1. **K Selection Accuracy** (`K_correct`)
   - Did the method choose the correct K?
   - Most important for practical use

2. **Mean Recovery** (`mean_recovery`)
   - Correlation between true and estimated maps (best match)
   - Only meaningful when K is correct

3. **Avg Recovery Per State** (`avg_recovery_per_state`)
   - Average quality of extracted maps (regardless of K)
   - Useful even when K is wrong

4. **Runtime** (`runtime_s`)
   - Computational cost

### Interpreting the Summary

```matlab
% Load results
T = readtable('out_microstate_comparison_final/results/comparison_results_final.csv');

% Overall accuracy by method
grpstats(T, {'method', 'criterion'}, 'mean', 'DataVars', 'K_correct')

% Performance vs SNR
grpstats(T, {'SNR_dB', 'method'}, 'mean', 'DataVars', 'mean_recovery')

% Best method overall
[~, best_idx] = max(T.K_correct);
fprintf('Winner: %s + %s (%.1f%% accuracy)\n', ...
    T.method{best_idx}, T.criterion{best_idx}, 100*T.K_correct(best_idx));
```

### Visualization

The pipeline generates a comprehensive comparison figure with:

1. **Overall Accuracy** - Bar chart by method+criterion
2. **Map Recovery** - Best-match correlation
3. **Avg Per State** - Recovery regardless of K
4. **Accuracy vs SNR** - Line plots showing robustness
5. **K Error Distribution** - How far off when wrong
6. **Recovery vs True K** - Performance scaling
7. **Runtime** - Computational cost
8. **Trade-offs** - Scatter plots

## Synthetic Data Generation

### How It Works

`generate_microstate_eeg.m` creates realistic EEG with:

1. **Spatial patterns** - K microstates on 64-channel sphere
2. **Temporal dynamics** - Geometric dwell times (~80ms mean)
3. **GFP modulation** - AR(1) amplitude envelope
4. **Realistic noise** - 1/f spatially correlated
5. **Artifacts** - Eye blinks, ECG

### Parameters

```matlab
[Sim, maps_true, pos] = generate_microstate_eeg(K_true, snr_db, duration_s, sfreq, seed);

% Example:
Sim = generate_microstate_eeg(5, 0, 60, 250, 42);
% K_true = 5 microstates
% SNR = 0 dB
% Duration = 60 seconds
% sfreq = 250 Hz
% seed = 42 (reproducibility)
```

### Output Structure

```matlab
Sim.X_clean      % Clean EEG [channels x time]
Sim.X_noisy      % Noisy EEG [channels x time]
Sim.maps_true    % True microstate topographies [K x channels]
Sim.z_true       % True state sequence [1 x time]
Sim.pos          % Channel positions [channels x 3]
Sim.sfreq        % Sampling frequency
```

## The Science: Elbow in the ELBO

### Why Not Just Maximize ELBO?

The **ELBO** (Evidence Lower BOund) is the objective in variational Bayes:

```
ELBO(K) = E_q[log p(X, Z)] - E_q[log q(Z)]
        = log p(X) - KL(q(Z) || p(Z|X))
```

**Problem:** ELBO always increases with K (more complex models fit better).

**Solution:** Find where returns diminish—the **elbow point**.

### Elbow Detection Algorithm

```matlab
% 1. Normalize ELBO and K to [0, 1]
elbo_norm = (elbo - min(elbo)) / (max(elbo) - min(elbo));
k_norm = (K - min(K)) / (max(K) - min(K));

% 2. Compute perpendicular distance to line
% connecting first and last points
for each K:
    distance(K) = perpendicular_distance(point, line)
end

% 3. K_elbow = argmax(distance)
```

This finds the point of **maximum curvature**—where the ELBO curve "bends" most.

### Combined Criterion (Default)

```matlab
score(K) = 0.6 * exp(-|K - K_elbow|) + 0.4 * (silhouette + 1) / 2
```

- **60% elbow penalty** - Prefer K near elbow
- **40% silhouette bonus** - Tiebreaker for cluster quality

## Advanced Usage

### Custom Method Development

```matlab
function Results = fit_microstate_custom(Sim, K_candidates, criterion)
    % Your method here
    
    % Required preprocessing
    [maps_norm, idx_peaks, gfp_vec, n_maps, C_dims] = preprocess_maps(Sim);
    
    % Fit models for each K
    for K = K_candidates
        % Your clustering algorithm
        [centers, labels] = your_clustering(maps_norm, K);
        
        % Compute scores
        scores(K) = your_criterion(maps_norm, labels, centers);
    end
    
    % Select best K
    [~, best_idx] = max(scores);
    K_estimated = K_candidates(best_idx);
    
    % Compute recovery
    true_maps_norm = normalize_maps(Sim.maps_true);
    recovery_corr = best_match_corr_hungarian_polarity_aware(...
        true_maps_norm, centers);
    mean_recovery = mean(recovery_corr);
    avg_recovery_per_state = mean(max(abs(centers * true_maps_norm'), [], 2));
    
    % Return results structure
    Results = struct(...
        'method', 'custom', ...
        'criterion', criterion, ...
        'K_estimated', K_estimated, ...
        'mean_recovery', mean_recovery, ...
        'avg_recovery_per_state', avg_recovery_per_state, ...
        'valid_fit', true);
end
```

### Parallel Processing

```matlab
% For large-scale comparisons
parpool(4);  % 4 workers

% Modify pipeline to use parfor
% (requires code modification in main loop)
```

### Batch Processing

```matlab
% Run multiple configurations
configs = {...
    struct('K_true_vals', [3 4 5], 'SNR_dbs', [-5 0 5]), ...
    struct('K_true_vals', [6 7 8], 'SNR_dbs', [0 5 10])};

for i = 1:length(configs)
    cfg = configs{i};
    T{i} = VBGMM_MS_Comparison_Pipeline(...
        'K_true_vals', cfg.K_true_vals, ...
        'SNR_dbs', cfg.SNR_dbs, ...
        'out_dir', sprintf('./batch_%d', i));
end
```


## Troubleshooting

### SPM Not Found

```matlab
Error: 'spm_mix' not found

Solution:
1. Download SPM development version (probably jsut git clone it somewhere) and get all the toolboxes
2. addpath('/path/to/spm')
```
## Citation

If you use this code in your research, please cite:

```bibtex
@software{microstate_vb_comparison_2025,
  author = {Kandasamy, Rohan},
  title = {Microstate VB Method Comparison},
  year = {2025},
  url = {https://github.com/dr-r-kan/SeizureMicrostateBehaviour},
  note = {MATLAB implementation comparing Variational Bayes and classical methods for EEG microstate analysis and K selection}
}
```

### Related Work

- **SPM12**: Friston et al., "Statistical Parametric Mapping" (SPM12 Manual)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Dr. R. Kan**  
GitHub: [@dr-r-kan](https://github.com/dr-r-kan)

For questions, issues, or collaboration:
- Open an [issue](https://github.com/dr-r-kan/SeizureMicrostateBehaviour/issues)
- Start a [discussion](https://github.com/dr-r-kan/SeizureMicrostateBehaviour/discussions)

## Acknowledgments

- **SPM team** at UCL for the mixture toolbox
- **Microstate community** for establishing methodology standards
- **MATLAB** for numerical computing infrastructure

---

**Made with ☕ and variational inference**

*"Finding the elbow in the ELBO—because sometimes the best puns are also the best science."*
