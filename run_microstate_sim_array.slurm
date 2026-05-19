#!/bin/bash -l
#SBATCH --job-name=spm-ms-sim
#SBATCH --array=1-480%40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --licenses=matlab:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# One MATLAB process per array task. The default grid below has:
#   8 reps * 4 K values * 5 SNR values * 3 overlaps = 480 array tasks.
# If you edit REPS/K_TRUE_VALS/SNR_DBS/OVERLAP_PROBS or TASKS_PER_JOB,
# update the #SBATCH --array range above to ceil(total_conditions/TASKS_PER_JOB).
#
# Before submitting, make sure the Slurm log directory exists:
#   mkdir -p logs
#
# If your Slurm site uses a different MATLAB licence resource name, update
# or comment out the --licenses line above.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRATCH_DIR="${TMPDIR:-${PROJECT_DIR}}"

module purge
module load xorg-utils/X11R7.7
module load matlab/full/r2021a/9.10
module list

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/Output/slurm_sim_all_methods}"
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
export SET_FILE="${SET_FILE:-${PROJECT_DIR}/MetaMaps_2023_06.set}"

mkdir -p logs "${OUT_ROOT}"

echo "Job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"
echo "Project directory: ${PROJECT_DIR}"
echo "Scratch directory: ${SCRATCH_DIR}"
echo "Output root: ${OUT_ROOT}"

cd "${SCRATCH_DIR}"

cat > run_array_task_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.m <<MATLAB_EOF
addpath(genpath('${PROJECT_DIR}'));
run_slurm_array_task;
exit;
MATLAB_EOF

echo ""
echo "Running matlab -nosplash -nodesktop -nodisplay < run_array_task_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.m ..."
echo ""

matlab -nosplash -nodesktop -nodisplay < run_array_task_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.m
