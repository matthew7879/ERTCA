#!/usr/bin/env python3
"""Check all 144 expected relaxations, or one folder with --stacking PATH."""

import argparse
from pathlib import Path
from pipeline_utils import GRID, check_settings, indices, shift_name, read_relax
import settings as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stacking', type=Path)
    args = parser.parse_args()
    check_settings()
    if args.stacking:
        result = read_relax(args.stacking)
        print(f'Converged: Nb-Nb = {result["distance"]:.8f} Angstrom; '
              f'max |Fz| = {result["max_fz"]:.3e} Ry/Bohr')
        return 0
    good, failed, details = [], [], []
    for i, j in indices():
        name = shift_name(i, j)
        try:
            r = read_relax(GRID / name)
            zs = ' '.join(f'{row[3]:.10f}' for row in r['atoms'])
            good.append(f'{i} {j} {i/s.N:.10f} {j/s.N:.10f} {zs} '
                        f'{r["energy"]:.10f} {r["distance"]:.10f} {r["max_fz"]:.8e}\n')
            print(f'{name}: OK  Nb-Nb={r["distance"]:.7f} Angstrom')
        except (OSError, ValueError) as exc:
            failed.append(name)
            details.append(f'{name}: {exc}\n')
            print(details[-1].rstrip())
    (GRID / 'relaxed_positions.dat').write_text(
        '# i j i/12 j/12 Nb1_z Se1_z Se2_z Nb2_z Se3_z Se4_z E_Ry NbNb_A max_abs_Fz_Ry_Bohr\n' + ''.join(good))
    (GRID / 'failed.txt').write_text(''.join(name + '\n' for name in failed))
    (GRID / 'relaxation_failures.txt').write_text(''.join(details))
    print(f'Converged: {len(good)}/{s.N*s.N}. Summary: {GRID / "relaxed_positions.dat"}')
    return int(bool(failed))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
