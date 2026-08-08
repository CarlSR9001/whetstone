#!/usr/bin/env bash
set -euo pipefail

# Windows mills with native Stockfish/KataGo. WSL owns CUDA training and fresh-
# load grading. Passing logic through this file avoids PowerShell/bash quoting.
phase="${1:-}"
if [[ -z "${phase}" ]]; then
  printf 'usage: bash scripts/run_gen4_wsl.sh bootstrap|train|grade [runner args...]\n' >&2
  exit 64
fi
shift
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
venv="${WHETSTONE_GEN4_VENV:-/home/${USER}/.local/share/whetstone-gen4-venv}"
source_root="${WHETSTONE_GEN4_SOURCE_ROOT:-${repo}/.bcv_runs/gen4_engine_student}"
run_root="${WHETSTONE_GEN4_RUN_ROOT:-/home/${USER}/whetstone-data/gen4-engine-student}"

if [[ "${phase}" == "bootstrap" ]]; then
  if [[ ! -d "${source_root}/bank" || ! -f "${source_root}/data/data_manifest.json" ]]; then
    printf 'error: mill and prepare the Windows source root first: %s\n' "${source_root}" >&2
    exit 66
  fi
  if [[ -e "${run_root}" ]]; then
    printf 'error: refusing to overwrite existing WSL run root: %s\n' "${run_root}" >&2
    exit 65
  fi
  mkdir -p "$(dirname "${run_root}")"
  cp -a "${source_root}" "${run_root}"
  printf 'bootstrapped_private_run=%s\n' "${run_root}"
  exit 0
fi

if [[ "${phase}" != "train" && "${phase}" != "grade" ]]; then
  printf 'error: WSL phase must be bootstrap, train, or grade\n' >&2
  exit 64
fi
if [[ ! -x "${venv}/bin/python" ]]; then
  printf 'error: run scripts/setup_gen4_wsl.sh first\n' >&2
  exit 69
fi
if [[ -z "${WHETSTONE_FASTCONTEXT_SNAPSHOT:-}" ]]; then
  printf 'error: set WHETSTONE_FASTCONTEXT_SNAPSHOT to the exact cached snapshot\n' >&2
  exit 64
fi
if [[ ! -d "${WHETSTONE_FASTCONTEXT_SNAPSHOT}" ]]; then
  printf 'error: snapshot directory not found: %s\n' "${WHETSTONE_FASTCONTEXT_SNAPSHOT}" >&2
  exit 66
fi

export PYTHONPATH="${repo}/src"
exec "${venv}/bin/python" "${repo}/scripts/run_gen4_engine_student.py" \
  --phase "${phase}" \
  --root "${run_root}" \
  --bank "${run_root}/bank" \
  --promotion-receipt "${repo}/results/gen4_engine_student_promotion_receipt.json" \
  --evaluation-receipt "${repo}/results/gen4_engine_student_evaluation_receipt.json" \
  "$@"
