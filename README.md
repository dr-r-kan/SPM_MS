# SPM Microstate Analysis

MATLAB tools for EEG microstate fitting, template alignment, meta-microstate dataset modelling, and simulation-based method comparison.

We use SPM's mixture toolbox for microstate analysis and compare it with traditional methods.

The repository is organised around a small number of public entry points and a shared utility layer. Defaults for paths, preprocessing, hierarchical fitting, simulation, and plotting are kept in `config/microstate_config.json`.

## Recommended Pipeline

For a shareable end-to-end run, use:

```matlab
R = run_benchmark_then_lemon_vb_pipeline( ...
    'manifest_csv', 'conditioned_lemon_sets.csv', ...
    'output_root', 'outputs/public_release');
```

This performs three stages:

1. `simulated_ms_retrieval_experiment` on simulated EEG.
2. `analyze_comparison_results` plus `summarize_first_line_spm_vb_metrics` on the simulation outputs, focusing on traditional K-means versus SPM-VB and on criterion comparisons within SPM-VB.
3. `metamicrostate_dataset_pipeline` on the LEMON manifest using the SPM-VB approach.

The runner writes a plain-text settings snapshot and a `pipeline_results.mat` summary under the chosen output root.

## Main Entry Points

### Single EEG Analysis

```matlab
[Results, json_file] = analyze_single_eeg_file('subject_EC.set', ...
    'method', 'spm_vb', ...
    'criterion', 'elbow_sil_combined', ...
    'align_template', true);
```

This loads an EEGLAB `.set` or MATLAB `.mat` file, applies the configured preprocessing, fits microstates, aligns to MetaMaps when requested, saves JSON, and writes topographic plots.

### Meta-Microstate Dataset Fit

```matlab
[R, output_csv] = metamicrostate_dataset_pipeline('conditioned_lemon_sets.csv', ...
    'method', 'spm_vb', ...
    'criterion', 'elbow_sil_combined');
```

The manifest must include `file_path`; `participant`, `condition`, and `group` are optional. The dataset workflow fits per-file solutions, clusters those solutions into dataset-level meta-microstates, and also writes pooled GFP-peak fits and subset summaries.

### Simulation Benchmark

```matlab
T = simulated_ms_retrieval_experiment();
analyze_comparison_results('outputs/simulations/results');
```

The simulation pipeline compares SPM-VB and traditional K-means across K, SNR, overlap, and montage conditions, and can also export per-K candidate metrics for criterion analysis.

## Configuration

Edit `config/microstate_config.json` rather than hard-coding local paths in functions.

Important keys:

- `paths.template_file`: MetaMaps `.set` file.
- `paths.spm_mixture_paths`: candidate SPM `toolbox/mixture` directories.
- `paths.koenig_code_dir`: local MicrostateLab helper directory containing exact vendored upstream files.
- `preprocessing`: shared defaults for average reference, filtering, GFP peak rejection, and optional spatial filtering.
- `hierarchical`: defaults for MetaMaps initialisation, canonical prior weight, and template-aligned GFP peak rejection.
- `simulation`: benchmark defaults, including the clean sanity profile used for the high-SNR zero-overlap traditional K-means fairness check.

## Inputs

- `conditioned_lemon_sets.csv`: default LEMON manifest used by the public pipeline.
- `build_lemon_manifest.py`: helper to generate a manifest CSV from preprocessed EEGLAB `.set` files.
- `examples/`: small example inputs and outputs for smoke-testing and inspection.

## Repository Layout

- `run_benchmark_then_lemon_vb_pipeline.m`: public three-stage runner for simulation, simulation analysis, and LEMON SPM-VB.
- `analyze_single_eeg_file.m`: single-file real EEG workflow.
- `metamicrostate_dataset_pipeline.m`: multi-file meta-microstate workflow.
- `simulated_ms_retrieval_experiment.m`: synthetic benchmark.
- `analyze_comparison_results.m`: simulation-result analysis and plotting.
- `summarize_first_line_spm_vb_metrics.m`: compact criterion summary for raw SPM-VB selectors.
- `fit_microstate_*.m`: low-level microstate fitting engines.
- `microstate_utilities.m`: shared preprocessing, geometry, config, sanitisation, and utility functions.
- `plot_microstate_*.m`: plotting helpers for global/group/condition summaries.
- `run_microstate_hierarchical_tanova.m`: non-parametric condition/group topography testing for dataset outputs.
- `Koenig_code/`: exact vendored upstream files from Thomas Koenig's MICROSTATELAB repository, kept separate from locally adapted code.
- `config/`: repository defaults.
- `build_lemon_manifest.py`: manifest builder for LEMON-style `.set` collections.
- `examples/`: small example data and manifests.
- `old/`: ignored legacy or off-scope code moved out of the active experiment surface.
- `outputs/`: generated results; ignored by git.

## Koenig Code Provenance

Any code copied exactly from Thomas Koenig's MICROSTATELAB repository should live in `Koenig_code/` rather than being duplicated elsewhere in the repo. Local files outside `Koenig_code/` should contain only project-specific wrappers, adaptations, or reimplementations.

Vendored upstream files currently in `Koenig_code/`:

- `Koenig_code/eeg_kMeans.m`
- `Koenig_code/pop_FitMSMaps.m`
- `Koenig_code/L2NormDim.m`
- `Koenig_code/mywaitbar.m`

Upstream source:

- https://github.com/ThomasKoenigBern/microstates

## Dependencies

- MATLAB R2020a or newer; developed with R2025a.
- EEGLAB for `.set` loading and `topoplot`.
- SPM mixture toolbox (`spm_mix`) for SPM/VB methods. This is included in the development version of SPM.
- Vendored MICROSTATELAB functions in `Koenig_code/` for the traditional K-means implementation.

## Development Notes

- Keep public workflows thin and put reusable logic in `microstate_utilities.m`.
- If an upstream MICROSTATELAB function is copied verbatim, place it in `Koenig_code/` and add it to the list above in this `README`.
- Prefer `run_benchmark_then_lemon_vb_pipeline.m` for shareable batch runs rather than the older meta-selector runner.
- Do not commit subject-level EEG data, generated `.mat` outputs, diagnostic folders, local manifests with absolute paths, or anything placed under `old/`.
- Prefer adding new defaults to `config/microstate_config.json` and reading them through `microstate_utilities().load_config()`.

## License

MIT License. See `LICENSE`.
