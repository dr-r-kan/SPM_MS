# SPM Microstate Analysis

MATLAB tools for EEG microstate fitting, template alignment, meta-microstate dataset modelling, and simulation-based method comparison.

The repository is organised around a small number of public entry points and a shared utility layer. Defaults for paths, preprocessing, hierarchical fitting, simulation, and plotting are kept in `config/microstate_config.json`.

## Main Workflows

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
[R, output_csv] = metamicrostate_dataset_pipeline('manifest.csv');
```

The manifest must include `file_path`; `participant`, `condition`, and `group` are optional. The dataset workflow fits per-file solutions, clusters those solutions into dataset-level meta-microstates, and also writes pooled GFP-peak fits and subset summaries.

### Simulation Benchmark

```matlab
T = simulated_ms_retrieval_experiment();
analyze_comparison_results('outputs/simulations/results/comparison_results.csv');
```

The simulation pipeline compares VB-GMM, Koenig k-means, and SPM k-means across K, SNR, overlap, and montage conditions.

## Configuration

Edit `config/microstate_config.json` rather than hard-coding local paths in functions.

Important keys:

- `paths.template_file`: MetaMaps `.set` file.
- `paths.spm_mixture_paths`: candidate SPM `toolbox/mixture` directories.
- `paths.koenig_code_dir`: local MicrostateLab/Koenig helper directory.
- `preprocessing`: shared defaults for average reference, filtering, GFP peak rejection, and optional spatial filtering.
- `hierarchical`: defaults for MetaMaps initialisation, canonical prior weight, and template-aligned GFP peak rejection.
- `simulation`: benchmark defaults. By default, simulation preprocessing is intentionally minimal to avoid biasing the benchmark.

## Experiment Scripts

- `scripts/download_lemon_preprocessed.sh`: clone/download the OpenNeuro `ds000221` dataset via DataLad, materialise preprocessed EEG derivatives, and build a manifest CSV.
- `scripts/build_lemon_manifest.py`: generate a LEMON manifest from a directory of `.set` files.
- `scripts/run_part1_simulated_cluster.m`: extensive Part 1 simulation entry point for cluster runs.
- `scripts/run_part2_lemon_cluster.m`: Part 2 LEMON pipeline entry point, with optional TANOVA.
- `scripts/run_experiment_smoke_tests.m`: small end-to-end smoke test for both parts.
- `scripts/myriad_*.qsub`: Myriad batch jobs.
- `scripts/submit_myriad_experiments.sh`: convenience submitter for the two main Myriad jobs.

## Repository Layout

- `analyze_single_eeg_file.m`: single-file real EEG workflow.
- `metamicrostate_dataset_pipeline.m`: multi-file meta-microstate workflow.
- `simulated_ms_retrieval_experiment.m`: synthetic benchmark.
- `fit_microstate_*.m`: low-level microstate fitting engines.
- `microstate_utilities.m`: shared preprocessing, geometry, config, sanitisation, and utility functions.
- `plot_microstate_*.m`: plotting helpers for global/group/condition summaries.
- `run_microstate_hierarchical_tanova.m`: non-parametric condition/group topography testing for dataset outputs.
- `config/`: repository defaults.
- `scripts/`: download, cluster, and smoke-test entry points.
- `old/`: ignored legacy or off-scope code moved out of the active experiment surface.
- `outputs/`: generated results; ignored by git.

## Dependencies

- MATLAB R2020a or newer; developed with R2025a.
- EEGLAB for `.set` loading and `topoplot`.
- SPM mixture toolbox (`spm_mix`) for SPM/VB methods. This is included in the development version of SPM.
- Koenig/MicrostateLab functions in `Koenig_code/` for the k-means implementation.

`Koenig_code/`, EEG datasets, and generated outputs are intentionally ignored by git.

## Development Notes

- Keep public workflows thin and put reusable logic in `microstate_utilities.m`.
- Do not commit subject-level EEG data, generated `.mat` outputs, diagnostic folders, local manifests with absolute paths, or anything placed under `old/`.
- Prefer adding new defaults to `config/microstate_config.json` and reading them through `microstate_utilities().load_config()`.

## License

MIT License. See `LICENSE`.
