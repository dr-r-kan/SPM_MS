# SPM Microstate Analysis

MATLAB tools for EEG microstate fitting, template alignment, hierarchical dataset modelling, backfitting, and simulation-based method comparison.

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

### Hierarchical Dataset Fit

```matlab
[H, mat_file] = fit_microstate_hierarchical_dataset('manifest.csv');
plot_hier_ms(mat_file);
```

The manifest must include `file_path`, `group`, and `condition`; `participant` is optional. The hierarchical workflow supports average reference, GFP outlier rejection, MetaMaps-seeded fitting, template-misaligned peak rejection, and global/group/condition/participant/file-level templates.

### Backfit Hierarchical Templates

```matlab
Results = ms_backfit('outputs/hierarchical_microstates/hierarchical_microstate_results.mat', ...
    'manifest_csv', 'manifest.csv');
```

Backfitting assigns continuous EEG samples to fitted templates and exports coverage, occurrence, duration, transition, and QC summaries.

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

## Repository Layout

- `analyze_single_eeg_file.m`: single-file real EEG workflow.
- `fit_microstate_hierarchical_dataset.m`: hierarchical multi-file workflow.
- `ms_backfit.m`: continuous backfitting from fitted hierarchical templates.
- `simulated_ms_retrieval_experiment.m`: synthetic benchmark.
- `fit_microstate_*.m`: low-level microstate fitting engines.
- `microstate_utilities.m`: shared preprocessing, geometry, config, sanitisation, and utility functions.
- `plot_hier_ms.m`, `plot_microstates.m`: plotting helpers.
- `config/`: repository defaults.
- `outputs/`: generated results; ignored by git.

## Dependencies

- MATLAB R2020a or newer; developed with R2025a.
- EEGLAB for `.set` loading and `topoplot`.
- SPM mixture toolbox (`spm_mix`) for SPM/VB methods.
- Koenig/MicrostateLab functions in `Koenig_code/` for the k-means implementation.

`Koenig_code/`, EEG datasets, and generated outputs are intentionally ignored by git.

## Development Notes

- Keep public workflows thin and put reusable logic in `microstate_utilities.m`.
- Do not commit subject-level EEG data, generated `.mat` outputs, diagnostic folders, or local manifests with absolute paths.
- Prefer adding new defaults to `config/microstate_config.json` and reading them through `microstate_utilities().load_config()`.

## License

MIT License. See `LICENSE`.
