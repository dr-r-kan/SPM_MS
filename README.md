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
