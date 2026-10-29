"""Generate portable Slurm scripts. Submit from the directory containing the script."""

import settings as s


def batch(stage, i=None, j=None):
    array = i is None
    name = f'{stage}_12x12' if array else f'{stage}_{i:02d}_{j:02d}'
    directives = f'#SBATCH --array=0-{s.N*s.N-1}%{s.MAX_CONCURRENT}\n' if array else ''
    if array:
        body = f'''root="$SLURM_SUBMIT_DIR"
i=$(( SLURM_ARRAY_TASK_ID / {s.N} ))
j=$(( SLURM_ARRAY_TASK_ID % {s.N} ))
folder=$(printf "%s/grid12/shift_i%02d_j%02d" "$root" "$i" "$j")
'''
    else:
        body = '''folder="$SLURM_SUBMIT_DIR"
root=$(cd "$folder/../.." && pwd)
'''
    return f'''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={s.ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={s.NTASKS}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={s.MEM_PER_CPU}
#SBATCH --time={s.WALLTIME}
#SBATCH --output={stage}-%A_%a.out
#SBATCH --error={stage}-%A_%a.err
{directives}
set -euo pipefail
{body}
bash "$root/run_{stage}.sh" "$folder"
'''
