#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

TARGET_ROOT=${1:-"${PROJECT_ROOT}/data/lemon"}
DATASET_URL=${DATASET_URL:-"https://github.com/OpenNeuroDatasets/ds000221.git"}
DATASET_DIR=${DATASET_DIR:-"${TARGET_ROOT}/ds000221"}
DERIV_ROOT_REL=${DERIV_ROOT_REL:-"derivatives"}
LINK_DIR=${LINK_DIR:-"${TARGET_ROOT}/EEG_Preprocessed"}
MANIFEST_CSV=${MANIFEST_CSV:-"${TARGET_ROOT}/lemon_manifest.csv"}
N_JOBS=${N_JOBS:-8}

if ! command -v datalad >/dev/null 2>&1; then
    echo "datalad is required for this downloader." >&2
    echo "Install datalad, then rerun this script." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for manifest generation." >&2
    exit 1
fi

mkdir -p "${TARGET_ROOT}"

if [[ ! -d "${DATASET_DIR}/.datalad" ]]; then
    datalad clone "${DATASET_URL}" "${DATASET_DIR}"
fi

cd "${DATASET_DIR}"

if [[ ! -d "${DERIV_ROOT_REL}" ]]; then
    echo "Derivative directory not found: ${DATASET_DIR}/${DERIV_ROOT_REL}" >&2
    echo "Set DERIV_ROOT_REL if the preprocessed EEG lives elsewhere in ds000221." >&2
    exit 1
fi

mapfile -t EEG_FILES < <(find "${DERIV_ROOT_REL}" -type f \( -name "*.set" -o -name "*.fdt" \) | sort)

if [[ ${#EEG_FILES[@]} -eq 0 ]]; then
    echo "No .set/.fdt files found under ${DATASET_DIR}/${DERIV_ROOT_REL}" >&2
    echo "Inspect the dataset layout and adjust DERIV_ROOT_REL before rerunning." >&2
    exit 1
fi

datalad get -J "${N_JOBS}" "${EEG_FILES[@]}"

mkdir -p "${LINK_DIR}"
while IFS= read -r relpath; do
    target_name=$(basename "${relpath}")
    ln -sfn "${DATASET_DIR}/${relpath}" "${LINK_DIR}/${target_name}"
done < <(printf '%s\n' "${EEG_FILES[@]}")

python3 "${SCRIPT_DIR}/build_lemon_manifest.py" \
    --input-dir "${LINK_DIR}" \
    --output "${MANIFEST_CSV}"

echo "LEMON download complete."
echo "Dataset clone: ${DATASET_DIR}"
echo "Flat symlink view: ${LINK_DIR}"
echo "Manifest: ${MANIFEST_CSV}"
