#!/bin/bash
# Sourced by the run scripts. These are your original Nibi module versions.
module purge
module load StdEnv/2023 intel/2023.2.1 openmpi/4.1.5 quantumespresso/7.3.1
if [[ "$1" == wannier ]]; then
    module load wannier90/3.1.0
fi
export OMP_NUM_THREADS=1
command -v python3 >/dev/null
command -v pw.x >/dev/null
