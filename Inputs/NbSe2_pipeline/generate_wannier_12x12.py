#!/usr/bin/env python3
"""Generate SCF/NSCF/Wannier inputs only after every selected relaxation passes.

For an AA test: python3 generate_wannier_12x12.py --only 0 0
For the full grid: python3 generate_wannier_12x12.py
"""

import argparse
from pipeline_utils import GRID, check_settings, indices, shift_name, read_relax, write_same_or_new
from qe_inputs import pw_input, win_input, PW2WAN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', type=int, nargs=2, metavar=('I', 'J'))
    args = parser.parse_args()
    check_settings()
    jobs, failures = [], []
    for i, j in indices(args.only):
        folder = GRID / shift_name(i, j)
        try:
            atoms = read_relax(folder)['atoms']
            files = {'scf.in': pw_input('scf', atoms), 'nscf.in': pw_input('nscf', atoms),
                     'pw2wan.in': PW2WAN, 'NbSe2.win': win_input(atoms)}
            for name, text in files.items():
                if (folder / name).exists() and (folder / name).read_text() != text:
                    raise ValueError(f'{name} already exists with different content')
            jobs.append((folder, files))
        except (OSError, ValueError) as exc:
            failures.append(f'{folder.name}: {exc}')
    if failures:
        raise ValueError('No inputs written. Fix these relaxations first:\n' + '\n'.join(failures))
    for folder, files in jobs:
        for name, text in files.items():
            write_same_or_new(folder / name, text)
    print(f'Wrote inputs for {len(jobs)} converged stackings.')
    print('Test one from its shift directory with: sbatch batch_wannier.sh')
    print('Submit the complete grid from the pipeline folder with: bash submit_wannier.sh')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
