# Repository Structure

This project keeps MATLAB public entry points in the repository root so they are easy to call after adding the repository to the MATLAB path.

## Source Files

- `analyze_single_eeg_file.m`: single real EEG analysis.
- `metamicrostate_dataset_pipeline.m`: multi-file meta-microstate fitting.
- `simulated_ms_retrieval_experiment.m`: simulation benchmark.
- `fit_microstate_*.m`: method-specific fitting engines.
- `microstate_utilities.m`: shared config, preprocessing, channel geometry, sanitisation, and common numerical helpers.
- `plot_*.m`: plotting entry points and wrappers.
- `run_microstate_hierarchical_tanova.m`: dataset-level topographic permutation testing.

## Script Entry Points

- `scripts/download_lemon_preprocessed.sh`: DataLad/OpenNeuro downloader plus manifest generation for LEMON.
- `scripts/build_lemon_manifest.py`: manifest builder for `.set` files.
- `scripts/run_part1_simulated_cluster.m`: extensive simulation driver for cluster runs.
- `scripts/run_part2_lemon_cluster.m`: LEMON dataset driver for cluster runs.
- `scripts/run_experiment_smoke_tests.m`: small environment smoke test.
- `scripts/myriad_*.qsub`: UCL Myriad batch jobs.

## Configuration

- `config/microstate_config.json`: repository defaults for paths, preprocessing, fitting, simulation, and plotting.

## Local-Only Directories

These should not be committed:

- `outputs/`: generated results, plots, JSON, and `.mat` files.
- `Koenig_code/`: local copy of external MicrostateLab/Koenig dependencies.
- `test_sets/`, `data/`, `raw_data/`, `derivatives/`: local datasets.
- `old/`: ignored legacy or off-scope files kept out of the active experiment surface.

## Refactoring Rule

When adding new code, keep reusable logic in `microstate_utilities.m` and keep workflow files focused on orchestration.
