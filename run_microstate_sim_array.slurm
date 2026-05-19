#!/usr/bin/env bash
#SBATCH --job-name=spm-ms-sim
#SBATCH --array=1-480%40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# One MATLAB process per array task. The default grid below has:
#   8 reps * 4 K values * 5 SNR values * 3 overlaps = 480 array tasks.
# If you edit REPS/K_TRUE_VALS/SNR_DBS/OVERLAP_PROBS or TASKS_PER_JOB,
# update the #SBATCH --array range above to ceil(total_conditions/TASKS_PER_JOB).

module purge
module load matlab

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export OUT_ROOT="${OUT_ROOT:-Output/slurm_sim_all_methods}"
export REPS="${REPS:-1:8}"
export K_TRUE_VALS="${K_TRUE_VALS:-4 5 6 7}"
export SNR_DBS="${SNR_DBS:--9 -3 0 1 3}"
export OVERLAP_PROBS="${OVERLAP_PROBS:-0 0.5 1.0}"
export K_CANDIDATES="${K_CANDIDATES:-2:10}"
export DURATION_S="${DURATION_S:-300}"
export SFREQ="${SFREQ:-250}"
export MONTAGES="${MONTAGES:-full,10-20-20,10-20-12}"
export N_WORKERS="${N_WORKERS:-1}"
export TASKS_PER_JOB="${TASKS_PER_JOB:-1}"
export VALIDATE_SIMULATION="${VALIDATE_SIMULATION:-true}"

# Set these if your cluster does not have SPM/EEGLAB on MATLAB's startup path.
export SPM_PATH="${SPM_PATH:-}"
export SET_FILE="${SET_FILE:-MetaMaps_2023_06.set}"

mkdir -p logs "${OUT_ROOT}"

echo "Job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo "Output root: ${OUT_ROOT}"

matlab -nodisplay -nosplash -batch "addpath(genpath(pwd)); run_slurm_array_task"
