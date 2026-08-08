#!/usr/bin/env bash
set -euo pipefail

# Preserve the host's verified CUDA torch; install only the Whetstone model stack
# into an isolated system-site-packages venv.
venv="${WHETSTONE_GEN4_VENV:-/home/${USER}/.local/share/whetstone-gen4-venv}"
if [[ -e "${venv}" && ! -x "${venv}/bin/python" ]]; then
  printf 'error: existing path is not a Python venv: %s\n' "${venv}" >&2
  exit 65
fi
python3 - <<'PY'
import torch

assert torch.cuda.is_available(), "host Python must expose CUDA before venv creation"
print(f"host_torch={torch.__version__} cuda={torch.version.cuda}")
PY
if [[ ! -x "${venv}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${venv}"
fi
"${venv}/bin/python" -m pip install --disable-pip-version-check \
  'transformers==5.14.1' \
  'peft==0.20.0' \
  'bitsandbytes==0.50.0' \
  'sentencepiece==0.2.2' \
  'python-chess==1.999'
"${venv}/bin/python" - <<'PY'
import bitsandbytes
import peft
import torch
import transformers

assert torch.cuda.is_available()
print(
    f"gen4_wsl_ready torch={torch.__version__} transformers={transformers.__version__} "
    f"peft={peft.__version__} bitsandbytes={bitsandbytes.__version__}"
)
PY
