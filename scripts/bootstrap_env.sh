#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -e .
.venv/bin/pip install numpy pandas scipy matplotlib seaborn pyarrow jinja2 pyyaml tqdm statsmodels pybaselines pyteomics pyopenms ifes_apt_tc_data_modeling pyccapt vispy adjustText scikit-learn
if [[ "${INSTALL_APYT:-0}" == "1" ]]; then
  .venv/bin/pip install "git+https://github.com/sebi-85/apyt.git"
fi

if command -v python3.12 >/dev/null 2>&1; then
  python3.12 -m venv .venv312
  .venv312/bin/pip install --upgrade pip setuptools wheel
  .venv312/bin/pip install -e .
  .venv312/bin/pip install numpy pandas scipy matplotlib seaborn pyarrow jinja2 pyyaml tqdm statsmodels pybaselines pyteomics ase ifes_apt_tc_data_modeling
  if command -v brew >/dev/null 2>&1 && [[ -d /opt/homebrew/opt/libomp ]]; then
    export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
    export CFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
    export CXXFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
  fi
  .venv312/bin/pip install apav ms-deisotope
fi
