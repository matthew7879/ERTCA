#!/usr/bin/env python3
"""Check the completed QE/Wannier chain and required Hamiltonian outputs."""

import argparse
from pathlib import Path
from pipeline_utils import GRID, check_settings, check_wannier, record_completion, indices, shift_name, qe_finished


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stacking', type=Path)
    parser.add_argument('--record', action='store_true', help='record success after checking one finished job')
    parser.add_argument('--qe-file', type=Path, help='internal: check one QE stage')
    parser.add_argument('--scf', action='store_true', help='also require electronic convergence')
    args = parser.parse_args()
    if args.qe_file:
        qe_finished(args.qe_file, scf=args.scf)
        return 0
    check_settings()
    if args.record and not args.stacking:
        parser.error('--record requires --stacking')
    folders = [args.stacking] if args.stacking else [GRID / shift_name(i, j) for i, j in indices()]
    failed = []
    for folder in folders:
        try:
            check_wannier(folder, require_marker=not args.record)
            if args.record:
                record_completion(folder)
            print(f'{folder.name}: OK')
        except (OSError, ValueError, StopIteration) as exc:
            failed.append(folder.name)
            print(f'{folder.name}: FAILED: {exc}')
    if not args.stacking:
        (GRID / 'wannier_failed.txt').write_text(''.join(name + '\n' for name in failed))
    print(f'Complete: {len(folders)-len(failed)}/{len(folders)}')
    return int(bool(failed))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
