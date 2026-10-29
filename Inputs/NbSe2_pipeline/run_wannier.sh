#!/bin/bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${1:?Pass a shift directory}"
mkdir .pipeline_lock || { echo 'Another job is running, or a stale .pipeline_lock remains.'; exit 1; }
trap 'rmdir .pipeline_lock' EXIT

python3 "$root/check_relaxations.py" --stacking .
if [[ -e wannier.complete ]]; then
    python3 "$root/check_wannier.py" --stacking .
    echo 'Wannier chain already complete; skipping.'
    exit 0
fi
# Never let an old successful output mask failure in a new run.
for file in scf.out nscf.out pw2wan.out NbSe2.wout NbSe2.werr NbSe2_hr.dat; do
    if [[ -e "$file" ]]; then
        echo "Existing $file: archive the failed Wannier outputs before retrying (see README)."
        exit 1
    fi
done
for file in scf.in nscf.in pw2wan.in NbSe2.win; do
    test -s "$file" || { echo "Missing $file: run generate_wannier_12x12.py first."; exit 1; }
done
source "$root/cluster_env.sh" wannier
np=${SLURM_NTASKS:?Run through sbatch}

echo 'SCF'
mpirun -np "$np" pw.x -in scf.in > scf.out
python3 "$root/check_wannier.py" --qe-file scf.out --scf
echo 'NSCF'
mpirun -np "$np" pw.x -in nscf.in > nscf.out
python3 "$root/check_wannier.py" --qe-file nscf.out
echo 'Wannier preprocessing'
wannier90.x -pp NbSe2 > wannier_pp.out
test -s NbSe2.nnkp
test ! -e NbSe2.werr
echo 'QE to Wannier90'
mpirun -np "$np" pw2wannier90.x -in pw2wan.in > pw2wan.out
python3 "$root/check_wannier.py" --qe-file pw2wan.out
echo 'Wannier Hamiltonian'
wannier90.x NbSe2 > wannier_run.out
python3 "$root/check_wannier.py" --stacking . --record

# Optional space saving, only AFTER the complete chain has passed validation.
# Default: retain the QE scratch data. Enable by exporting CLEANUP_QE=1 before sbatch.
if [[ "${CLEANUP_QE:-0}" == 1 ]]; then
    rm -f -- ./NbSe2.wfc*
    rm -rf -- ./NbSe2.save
fi
echo 'DONE'
