#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-"${HOME}/Scratch/spm_ms"}
LEMON_MANIFEST=${LEMON_MANIFEST:-"${PROJECT_ROOT}/data/lemon/lemon_manifest.csv"}

mkdir -p "${SCRATCH_ROOT}"

RUN_PART1=${RUN_PART1:-1}
RUN_PART2=${RUN_PART2:-1}

if [[ "${RUN_PART1}" == "1" ]]; then
    qsub -wd "${SCRATCH_ROOT}" \
        -v PROJECT_ROOT="${PROJECT_ROOT}",PART1_OUTPUT_DIR="${PART1_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/cluster_runs/part1_simulated_extensive}",PART1_REPS="${PART1_REPS:-24}",PART1_THREADS="${PART1_THREADS:-12}" \
        "${SCRIPT_DIR}/myriad_part1.qsub"
fi

if [[ "${RUN_PART2}" == "1" ]]; then
    qsub -wd "${SCRATCH_ROOT}" \
        -v PROJECT_ROOT="${PROJECT_ROOT}",LEMON_MANIFEST="${LEMON_MANIFEST}",PART2_OUTPUT_DIR="${PART2_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/cluster_runs/part2_lemon}",PART2_THREADS="${PART2_THREADS:-12}",TANOVA_PERMS="${TANOVA_PERMS:-5000}" \
        "${SCRIPT_DIR}/myriad_part2.qsub"
fi
