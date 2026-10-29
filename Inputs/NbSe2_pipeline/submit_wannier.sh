#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 check_relaxations.py
python3 generate_wannier_12x12.py
sbatch submit_wannier_array.sh
