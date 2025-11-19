# Variational Bayes-based microstate extraction and cluster number optimisation

A comprehensive comparison of Variational Bayes and classical methods for EEG microstate clustering and automatic K selection.

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project addresses a fundamental challenge in EEG microstate analysis: **automatic determination of the optimal number of microstates (K)**. We compare Variational Bayes (VB) Gaussian Mixture Models (VBGMM) using SPM to the traditional method.

### Key Innovation: The Elbow in the ELBO

The **ELBO** (Evidence Lower BOund) is the free energy in variational Bayes—a principled measure of model quality. However, simply maximizing ELBO tends to overfit (preferring complex models). Our solution:

**Detect the "elbow" (knee point) in the ELBO curve** where gains diminish—the sweet spot before overfitting. This is an implicit measure of the statistics of the fit itself, rather than a post-hoc heuristic, and so is an elegant solution for K finding.

This provides a principled, data-driven way to select K without manual inspection, combining the rigor of VB with practical robustness.

## Motivation

Traditional microstate analysis relies on:
- **Modified K-means** with polarity invariance (field standard)
- **Manual K selection** or **heuristics** (silhouette, GEV)
- **Visual inspection** of topographies

Can we do better with **Variational Bayes**?

This project systematically compares:
1. **VB approaches** (SPM's VB-GMM)
2. **Classical approaches** (standard modified K-means)
3. **Model selection criteria** (silhouette, free energy, free energy elbow, GEV, combined silhouette and free energy elbow)

Using **synthetic data with known ground truth**, we can definitively answer: which method works best?

## Features

✅ **Synthetic EEG generation** with realistic microstate structure  
✅ **VBGMM** leveraging SPM's mixture toolbox for Bayesian methods and free energy minimisation
✅ **Elbow detection** on ELBO curves  
✅ **Comprehensive comparison** across SNR levels and true K values  
✅ **Recovery metrics** even when K is misestimated  
✅ **Montage robustness analysis** for reduced channel configurations (71 → 20 → 12 leads)

## Montage Robustness Analysis

A key challenge in clinical EEG is balancing spatial resolution with practical constraints. This repository now includes comprehensive analysis of how reduced montages affect microstate analysis performance.

### Supported Montages

- **Full (71 channels)**: Complete research-grade montage
- **10-20-20 (20 channels)**: Standard clinical 10-20 system
  - Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2, A1, A2
- **10-20-12 (12 channels)**: Minimal clinical montage
  - Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, Fz, Pz

### Usage Example: Montage Analysis

```matlab
% Run pipeline with multiple montages
Bayesian_MS_Comparison_Pipeline( ...
    'reps', 5, ...
    'K_true_vals', [4 5], ...
    'SNR_dbs', [-5 0 5 10], ...
    'montages', {'full', '10-20-20', '10-20-12'});

% Analyze montage effects
analyze_montage_robustness('path/to/comparison_results.csv', ...
    'output_dir', 'path/to/montage_analysis');
```

### Expected Performance Degradation

Based on theoretical considerations and empirical validation:

- **K Estimation**: 
  - Full (71 leads): Baseline accuracy
  - 10-20-20 (20 leads): ~85-95% of full performance
  - 10-20-12 (12 leads): ~70-85% of full performance

- **Microstate Center Recovery**:
  - Degrades gradually with fewer electrodes
  - Center precision most affected in frontal/temporal regions
  - Posterior regions relatively robust

- **SNR Interaction**:
  - Montage effects amplified at low SNR
  - High SNR partially compensates for reduced channels

### Clinical Interpretation Guidelines

**When to use reduced montages:**
- Emergency/ICU settings with limited setup time
- Long-term monitoring where comfort is critical
- Pilot studies or screening applications

**Quality considerations:**
- 20-lead montage suitable for most clinical microstate analysis
- 12-lead montage acceptable for robust macrostate detection
- Full montage recommended for:
  - Research publications
  - Fine-grained spatial analysis
  - Novel microstate discovery  

## Installation

### Prerequisites

- **MATLAB R2020a or later**
- **[SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)** (Statistical Parametric Mapping)
- **[MicrostateLab](https://github.com/ThomasKoenigBern/microstates)** (on path)

## Quick Start

### Basic Pipeline Execution

```matlab
% Run with default settings (full montage only)
Bayesian_MS_Comparison_Pipeline();

% Custom configuration
Bayesian_MS_Comparison_Pipeline( ...
    'reps', 10, ...
    'K_true_vals', [4 5 6 7], ...
    'SNR_dbs', [-10 -5 0 5 10], ...
    'K_candidates', 2:10);
```

### Montage Robustness Testing

```matlab
% Test all three montages
Bayesian_MS_Comparison_Pipeline( ...
    'reps', 5, ...
    'K_true_vals', [4 5], ...
    'SNR_dbs', [-5 0 5 10], ...
    'montages', {'full', '10-20-20', '10-20-12'});

% Analyze results
analyze_montage_robustness( ...
    'results/comparison_results.csv', ...
    'output_dir', 'montage_analysis');

% Filter analysis by specific conditions
analyze_montage_robustness( ...
    'results/comparison_results.csv', ...
    'K_true', [4 5], ...
    'SNR_dB', 10, ...
    'method', 'spm_vb');
```

### Analyzing Results

```matlab
% Standard analysis (includes montage comparison if available)
analyze_comparison_results('results/');

% View montage-specific outputs
% - montage_comparison_leads.png: Performance vs lead count
% - montage_comparison_boxplots.png: Distribution by montage
% - summary_montage_robustness.csv: Statistical summary
```

## Output Files

The pipeline generates:

- **comparison_results.csv**: All fit results with montage metadata
- **Plots**: Performance comparisons, heatmaps, interaction plots
- **JSON files**: Microstate templates and metadata
- **montage_analysis/**: Dedicated montage robustness analysis
  - K accuracy vs lead count
  - Recovery metrics degradation curves
  - Method × montage heatmaps
  - SNR × montage interaction plots

## Citation

If you use this code in your research, please cite the respository - but check to see a publication hasn't been added at the time of your publication!

### Related Work

- **SPM12**: Friston et al., "Statistical Parametric Mapping" (SPM12 Manual)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Dr. Rohan Kandasamy**  
GitHub: [@dr-r-kan](https://github.com/dr-r-kan)

For questions, issues, or collaboration:
- Open an [issue](https://github.com/dr-r-kan/SeizureMicrostateBehaviour/issues)
- Start a [discussion](https://github.com/dr-r-kan/SeizureMicrostateBehaviour/discussions)

## Acknowledgments

- **SPM team** at UCL for the mixture toolbox
- **MICROSTATELAB team** for establishing methodology standards
