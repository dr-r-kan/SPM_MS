#!/bin/bash -l
#SBATCH --job-name=spm-ms-combine
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --licenses=matlab:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRATCH_DIR="${TMPDIR:-${PROJECT_DIR}}"

module purge
module load xorg-utils/X11R7.7
module load matlab/full/r2021a/9.10
module list

export OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/Output/slurm_sim_all_methods}"

mkdir -p logs "${OUT_ROOT}"

cd "${SCRATCH_DIR}"

cat > combine_array_results_${SLURM_JOB_ID}.m <<MATLAB_EOF
addpath(genpath('${PROJECT_DIR}'));
combine_slurm_array_results('${OUT_ROOT}');
exit;
MATLAB_EOF

echo ""
echo "Running matlab -nosplash -nodesktop -nodisplay < combine_array_results_${SLURM_JOB_ID}.m ..."
echo ""

matlab -nosplash -nodesktop -nodisplay < combine_array_results_${SLURM_JOB_ID}.m
