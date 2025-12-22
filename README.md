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
1. **Classical K-means** (Koenig's standard method)
2. **SPM K-means** (K-means as GMM limit case with isotropic covariance, σ² → 0)
3. **VB-GMM** (SPM's full Variational Bayes with free covariance)
4. **Model selection criteria** (silhouette, free energy, free energy elbow, GEV, combined silhouette and free energy elbow)

Using **synthetic data with known ground truth**, we can definitively answer: which method works best?

### SPM K-Means: Theoretical Foundation

The **SPM K-Means** method validates that K-means clustering emerges as a limit case of Gaussian Mixture Models (GMMs) with:
- **Isotropic (spherical) covariance**: All clusters have equal spherical covariance
- **Infinitesimal variance**: σ² → 0 forces soft assignments to become hard
- **Mathematical equivalence**: Proven in literature (arXiv:1704.04812, Celeux & Govaert 1992)

This bridges the gap between classical K-means and Bayesian methods, showing:
- **Koenig K-means** ↔ **SPM K-means** (should give similar results)
- **SPM K-means** vs **VB GMM** (demonstrates value of full covariance modeling)

## Features

✅ **Synthetic EEG generation** with realistic microstate structure  
✅ **Three complementary methods**: Classical K-means, SPM K-means (GMM limit), and VB-GMM  
✅ **Theoretical validation**: Confirms GMM → K-means equivalence in practice  
✅ **VBGMM** leveraging SPM's mixture toolbox for Bayesian methods and free energy minimisation  
✅ **Elbow detection** on ELBO curves  
✅ **Comprehensive comparison** across SNR levels and true K values  
✅ **Recovery metrics** even when K is misestimated  
✅ **Montage robustness analysis** testing reduced lead configurations (71 → 20 → 12 channels)

## Montage Robustness Analysis

EEG microstate analysis is often performed on clinical data with reduced electrode montages. This feature tests how robust the methods are to reduced lead counts.

### Supported Montages

- **`full`** (71 channels): Complete research-grade montage
- **`10-20-20`** (20 channels): Standard 10-20 system
  - Channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2, A1, A2
- **`10-20-12`** (12 channels): Clinical 12-lead montage
  - Channels: Fp1, Fp2, F3, F4, Fz, C3, C4, P3, P4, Pz, O1, O2

### Usage

```matlab
% Default: full montage only (backward compatible)
Bayesian_MS_Comparison_Pipeline()

% Test montage robustness
Bayesian_MS_Comparison_Pipeline('montages', {'full', '10-20-20', '10-20-12'})

% Quick test with reduced montages
Bayesian_MS_Comparison_Pipeline(...
    'montages', {'full', '10-20-12'}, ...
    'reps', 5, ...
    'K_true_vals', [4], ...
    'SNR_dbs', [0 10])
```

### Montage Analysis Outputs

When multiple montages are tested:

1. **Results CSV** includes `montage_type` and `n_leads` columns
2. **Montage comparison plots** automatically generated:
   - K estimation accuracy vs lead count
   - Recovery metrics vs lead count
   - Boxplots by montage
3. **Dedicated analysis** via `analyze_montage_robustness.m`:
   ```matlab
   analyze_montage_robustness('results/comparison_results.csv')
   ```

### Expected Performance

**K Estimation Accuracy**:
- Full montage (71 leads): ~95% accuracy at high SNR
- 10-20-20 (20 leads): ~90% accuracy (minor degradation)
- 10-20-12 (12 leads): ~75-85% accuracy (moderate degradation)

**Recovery Quality**:
- Graceful degradation with reduced leads
- Strong methods maintain >0.8 correlation even at 12 leads
- SNR interaction: Low SNR + reduced montage = compounded challenge

### Clinical Interpretation

Reduced montages test **spatial resolution limits**:
- **Robust methods** maintain accuracy with 20 leads
- **Fragile methods** show significant degradation below 40 leads
- **Clinical applicability**: Methods performing well at 12-20 leads are suitable for clinical EEG data

## Installation

### Prerequisites

- **MATLAB R2020a or later**
- **[SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)** (Statistical Parametric Mapping)
- **[MicrostateLab](https://github.com/ThomasKoenigBern/microstates)** (on path)

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
