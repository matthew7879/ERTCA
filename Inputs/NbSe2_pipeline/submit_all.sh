#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -c 'from pipeline_utils import check_settings; check_settings()'
sbatch submit_relax_array.sh
