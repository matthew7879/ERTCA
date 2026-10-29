#!/usr/bin/env python3
"""Create a fresh 12x12 grid. Existing grids are deliberately never overwritten."""

import shutil
from pipeline_utils import ROOT, GRID, indices, shift_name
from qe_inputs import shifted_atoms, pw_input
from batch_scripts import batch


def main():
    for name in ['Nb.upf', 'Se.upf']:
        if not (ROOT / name).is_file():
            raise SystemExit(f'Missing {ROOT / name}')
    if GRID.exists():
        raise SystemExit(f'{GRID} already exists. Use a fresh copy of the pipeline folder.')
    GRID.mkdir()
    shutil.copyfile(ROOT / 'settings.py', GRID / 'settings_snapshot.py')
    for i, j in indices():
        folder = GRID / shift_name(i, j)
        folder.mkdir()
        for name in ['Nb.upf', 'Se.upf']:
            (folder / name).symlink_to('../../' + name)
        (folder / 'relax.in').write_text(pw_input('relax', shifted_atoms(i, j)))
        (folder / 'batch_relax.sh').write_text(batch('relax', i, j))
        (folder / 'batch_wannier.sh').write_text(batch('wannier', i, j))
    for stage in ['relax', 'wannier']:
        (ROOT / f'submit_{stage}_array.sh').write_text(batch(stage))
    print(f'Created {len(indices())} shifts in {GRID}')
    print('Test AA: cd grid12/shift_i00_j00 && sbatch batch_relax.sh')
    print('After checking the AA test, submit the grid from the pipeline folder: bash submit_all.sh')


if __name__ == '__main__':
    main()
