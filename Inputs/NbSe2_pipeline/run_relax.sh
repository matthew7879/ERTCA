#!/bin/bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${1:?Pass a shift directory}"

# A lock prevents two array/individual jobs from writing into the same folder.
mkdir .pipeline_lock || { echo 'Another job is running, or a stale .pipeline_lock remains.'; exit 1; }
trap 'rmdir .pipeline_lock' EXIT

if [[ -e relax.out ]]; then
    if python3 "$root/check_relaxations.py" --stacking .; then
        echo 'Relaxation already complete; skipping.'
        exit 0
    fi
    echo 'Existing relaxation failed validation. Archive relax.out before retrying (see README).'
    exit 1
fi
source "$root/cluster_env.sh" relax
mpirun -np "${SLURM_NTASKS:?Run through sbatch}" pw.x -in relax.in > relax.out
python3 "$root/check_relaxations.py" --stacking .
