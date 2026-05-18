#!/bin/bash

set -euo pipefail

N_RUNS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
LOG_DIR="${PROJECT_ROOT}/logs"
OUTPUT_DIR="${PROJECT_ROOT}/results_randomness_exps"

if [ ! -d "${VENV_PATH}" ]; then
    echo "Virtual environment not found at ${VENV_PATH}"
    echo "Create it first with: python3.10 -m venv .venv"
    exit 1
fi

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

for i in $(seq 1 "${N_RUNS}"); do
    echo "=== RUN ${i} ==="
    RUN_ID="${i}" OUTPUT_CSV="${OUTPUT_DIR}/run_${i}.csv" \
        python src/main.py \
        2>&1 | tee "${LOG_DIR}/run_${i}.log"
done

# chmod +x Exps_runs_randomness/exps_seeds.sh
# ./Exps_runs_randomness/exps_seeds.sh



#bash Exps_runs_randomness/exps_seeds.sh