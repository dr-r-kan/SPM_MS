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

### Hierarchical Dataset-Wise SPM-VB Fit

```matlab
[HResults, results_mat] = fit_microstate_hierarchical_dataset('LEMON', ...
    'output_dir', 'outputs/lemon_hierarchical_spm_vb_all_peaks');
```

This is the direct dataset-wise hierarchy for participants, conditions, and groups. It accepts either a folder of `.set`/`.mat` files or a CSV manifest. A manifest must contain `file_path`; `participant`, `condition`, and `group` are optional. If no manifest is supplied, participant IDs are inferred from names such as `sub-032389_EC.set`, and LEMON-style condition labels are inferred as `EC` and `EO`.

How it works:

1. Loads each EEG file with EEGLAB or MATLAB loading.
2. Keeps scalp channels when `use_scalp_channels=true`.
3. Excludes channels in `exclude_channels`, default `{'PO9','PO10'}` for LEMON.
4. Builds the fitting montage. By default, `interpolate_missing_channels=false`, so only channels common to all files are used. If `interpolate_missing_channels=true`, the densest retained montage is used and missing channels are filled by inverse-distance weighted interpolation from observed channel positions.
5. Applies preprocessing before GFP extraction: average reference, optional spatial filter, temporal bandpass, GFP peak finding, and optional GFP outlier rejection.
6. Keeps all GFP peaks by default. Use `max_maps_per_file` only for fast smoke runs.
7. Pools the GFP peak maps across the whole dataset and fits the global model with SPM `spm_mix` over polarity-invariant PCA/projective features.
8. Selects global `K` from `K_candidates` using `criterion`; the selected global `K` is then fixed for child fits.
9. Fits groups, if a non-default `group` column exists, using the global templates as parent pseudo-prior maps.
10. Fits conditions, if a non-default `condition` column exists, using the global templates as parent pseudo-prior maps.
11. Fits pooled participants using the matching group templates when present, plus matching condition templates when present; otherwise it falls back to the global templates.
12. Fits participant-condition nodes, so a LEMON participant with both recordings gets separate EC and EO fits. These are stored in `HResults.participant_conditions` and under `participant_conditions/<participant>/<condition>/`.
13. Aligns every fitted solution to the MetaMaps template file, flips polarity where needed, and writes template-correlation quality columns.
14. Backfits each participant-condition template to its matching full EEG record by default with `backfit_microstate_timecourse`, falling back to the pooled participant template only if the exact participant-condition node is missing.
15. Optionally runs per-state axis-dynamics analysis on hard-backfit active samples.
16. Writes backfit state, pairwise, record-summary, and optional axis-dynamics metric CSVs.

The parent priors are passed to SPM by adding parent template maps to the peak-map bank `spm_prior_pseudocount` times before calling `spm_mix`. This keeps the implementation small and avoids forking SPM; it is a pseudo-prior over the fitted samples, not a modification of SPM's internal Dirichlet or mean-prior code.

Useful switches:

```matlab
% Fast check: fixed K, capped peaks, no interpolation
test_fit_microstate_hierarchical_dataset_smoke

% Disable full-record backfitting and metric CSV export
H = fit_microstate_hierarchical_dataset('LEMON', ...
    'run_backfit', false);

% Use densest montage and interpolate missing channels
H = fit_microstate_hierarchical_dataset('LEMON', ...
    'interpolate_missing_channels', true);

% Add per-state active-axis PCA/precession metrics
H = fit_microstate_hierarchical_dataset('LEMON', ...
    'run_backfit_axis_dynamics', true);

% Full LEMON sample run with all GFP peaks
H = fit_microstate_hierarchical_dataset('LEMON', ...
    'output_dir', 'outputs/lemon_hierarchical_spm_vb_all_peaks');
```

Main outputs under `output_dir`:

- `hierarchical_microstate_results.mat`: full `HResults` structure.
- `hierarchical_fit_summary.csv`: global/group/condition/participant/participant-condition rows with `K_estimated`, map counts, and template alignment scores.
- `common_channels.csv`: final montage after scalp filtering, exclusions, and optional interpolation target selection.
- `pooled_gfp_peak_manifest.csv`: every retained GFP peak with participant, condition, group, file, sample, and GFP value.
- `participant_condition_state_backfit_metrics.csv`: per-record, per-state hard and Gaussian-mixture backfit metrics, including occupancy, percent present, mean GFP, occurrences, rate, and template match.
- `participant_condition_state_pairwise_backfit_metrics.csv`: per-record, per-state-pair mutual information metrics for hard and Gaussian-mixture backfit traces.
- `participant_condition_record_backfit_summary.csv`: per-record hard and Gaussian-mixture backfit availability plus entropy summaries.
- `participant_condition_state_axis_dynamics.csv`: optional per-record, per-state hard-backfit active-axis metrics. For each participant/condition/state it takes the EEG samples assigned to that state, projects them onto the fitted microstate axis, reports projection energy/RMS/zero-crossing rate, runs PCA on the active maps, and estimates precession from the phase of the perpendicular residual in its first two-PC plane. Direction is reported with signed phase-speed columns plus `precession_directionality_index`; the sign depends on the residual PCA-plane convention, while the index is invariant to axis flips.
- `global/`, `groups/`, `conditions/`, `participants/`, `participant_conditions/`: `.mat`, center `.csv`, model-comparison `.csv`, peak manifests, and topographic `.png` plots for each fitted node.

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
- `fit_microstate_hierarchical_dataset.m`: direct dataset-wise global/group/condition/participant/participant-condition SPM-VB hierarchy.
- `metamicrostate_dataset_pipeline.m`: multi-file meta-microstate workflow.
- `test_fit_microstate_hierarchical_dataset_smoke.m`: fast LEMON smoke check for the hierarchical SPM-VB fitter.
- `simulated_ms_retrieval_experiment.m`: synthetic benchmark.
- `analyze_comparison_results.m`: simulation-result analysis and plotting.
- `summarize_first_line_spm_vb_metrics.m`: compact criterion summary for raw SPM-VB selectors.
- `fit_microstate_*.m`: low-level microstate fitting engines.
- `microstate_utilities.m`: shared preprocessing, geometry, config, sanitisation, and utility functions.
- `plot_microstate_*.m`: plotting helpers for global/group/condition summaries.
- `run_microstate_hierarchical_tanova.m`: non-parametric and Bayesian bootstrap condition/group topography testing for aligned dataset outputs.
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
