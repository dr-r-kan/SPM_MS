#!/usr/bin/env bash
#SBATCH --job-name=spm-ms-combine
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load matlab

export OUT_ROOT="${OUT_ROOT:-Output/slurm_sim_all_methods}"

mkdir -p logs "${OUT_ROOT}"

matlab -nodisplay -nosplash -batch "addpath(genpath(pwd)); combine_slurm_array_results('${OUT_ROOT}')"
