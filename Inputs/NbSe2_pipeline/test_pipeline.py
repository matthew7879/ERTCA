#!/usr/bin/env python3
"""Local regression checks; no QE, network access, or job submission."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parent


def run(root, script, *args, ok=True, env=None):
    p = subprocess.run([sys.executable, str(root / script), *map(str, args)],
                       cwd=root, capture_output=True, text=True, env=env)
    assert (p.returncode == 0) == ok, p.stdout + p.stderr
    return p


def relax_log(folder):
    positions = (folder / 'relax.in').read_text().split('ATOMIC_POSITIONS (crystal)\n')[1].split('K_POINTS')[0]
    forces = '\n'.join(f'atom {i} type 1 force = 0.00000000 0.00000000 0.00005000' for i in range(1, 7))
    return ('Program PWSCF\nconvergence has been achieved in 10 iterations\n'
            '! total energy = -1219.3363 Ry\nForces acting on atoms (cartesian axes, Ry/au):\n' + forces +
            '\nbfgs converged in 10 scf cycles and 9 bfgs steps\n'
            'Begin final coordinates\nATOMIC_POSITIONS (crystal)\n' + positions +
            'End final coordinates\nJOB DONE.\n')


def main():
    with tempfile.TemporaryDirectory(prefix='nbse2_pipeline_test_') as tmp:
        root = Path(tmp) / 'pipeline'
        shutil.copytree(SOURCE, root, ignore=shutil.ignore_patterns('__pycache__', 'grid12'))
        run(root, 'generate_grid_12x12.py')
        grid = root / 'grid12'
        folders = sorted(grid.glob('shift_i*_j*'))
        assert len(folders) == 144
        aa = grid / 'shift_i00_j00'
        run(root, 'generate_grid_12x12.py', ok=False)
        for folder in folders:
            i, j = map(int, [folder.name[7:9], folder.name[11:13]])
            text = (folder / 'relax.in').read_text()
            assert 'ecutwfc = 80, ecutrho = 800' in text
            assert 'forc_conv_thr = 1.00000000e-04' in text
            assert 'etot_conv_thr = 1.00000000e-05' in text
            atoms = text.split('ATOMIC_POSITIONS (crystal)\n')[1].split('K_POINTS')[0].splitlines()
            top = atoms[3].split()
            assert abs(float(top[1]) - i/12) < 1e-9
            assert abs(float(top[2]) - j/12) < 1e-9
            assert all(line.split()[-3:] == ['0', '0', '1'] for line in atoms)
            assert (folder / 'Nb.upf').read_bytes() == (root / 'Nb.upf').read_bytes()
        print('PASS: 144 unique shifts, z-only flags, cutoffs, thresholds and portable PP links')

        valid = relax_log(aa)
        bad_logs = [valid.replace('bfgs converged in', 'bfgs still running'),
                    valid.replace('End final coordinates', ''),
                    valid.replace('JOB DONE.', ''),
                    valid.replace('0.00005000', '0.00050000'),
                    valid + '\nProgram PWSCF\nstarted another run\n']
        for bad in bad_logs:
            (aa / 'relax.out').write_text(bad)
            run(root, 'check_relaxations.py', '--stacking', aa, ok=False)
            run(root, 'generate_wannier_12x12.py', '--only', 0, 0, ok=False)
            assert not (aa / 'scf.in').exists()
        # Verbose QE logs print individual D3 force contributions after TOTAL forces.
        # Accept a small total force even when the individual contribution is larger.
        d3 = ('\nDFT-D3 dispersion contribution to forces\n' +
              '\n'.join(f'atom {i} type 1 force = 0.0 0.0 0.00050000'
                        for i in range(1, 7)) + '\n')
        (aa / 'relax.out').write_text(valid.replace('\nbfgs converged', d3 + '\nbfgs converged'))
        run(root, 'check_relaxations.py', '--stacking', aa)
        # Conversely, a small D3 contribution must not mask a large TOTAL force.
        high_total = valid.replace('0.00005000', '0.00050000')
        small_d3 = d3.replace('0.00050000', '0.00005000')
        (aa / 'relax.out').write_text(high_total.replace('\nbfgs converged', small_d3 + '\nbfgs converged'))
        run(root, 'check_relaxations.py', '--stacking', aa, ok=False)
        print('PASS: total forces are distinguished from later D3 contributions')

        (aa / 'relax.out').write_text(valid)
        run(root, 'check_relaxations.py', '--stacking', aa)
        run(root, 'check_relaxations.py', ok=False)
        assert len((grid / 'failed.txt').read_text().splitlines()) == 143
        run(root, 'generate_wannier_12x12.py', ok=False)
        assert not (aa / 'scf.in').exists()  # full preflight fails before writing
        run(root, 'generate_wannier_12x12.py', '--only', 0, 0)
        for folder in folders:
            (folder / 'relax.out').write_text(relax_log(folder))
        run(root, 'generate_wannier_12x12.py')
        run(root, 'check_relaxations.py')
        for folder in folders:
            for stage in ['scf', 'nscf']:
                text = (folder / f'{stage}.in').read_text()
                assert 'ecutwfc = 80, ecutrho = 800' in text
                assert 'nbnd = 108' in text
            nscf = (folder / 'nscf.in').read_text().split('K_POINTS crystal\n')[1].splitlines()
            assert int(nscf[0]) == len(nscf[1:]) == 400
            assert abs(sum(float(row.split()[3]) for row in nscf[1:]) - 1) < 1e-10
            win = (folder / 'NbSe2.win').read_text()
            assert 'wannier_plot = .false.' in win and 'num_iter = 0' in win
            ks = win.split('begin kpoints\n')[1].split('end kpoints')[0].splitlines()
            assert len(ks) == 400
        for script in root.rglob('*.sh'):
            subprocess.run(['bash', '-n', str(script)], check=True)
            assert '/Users/' not in script.read_text()
        print('PASS: unfinished/high-force/missing jobs rejected; converged inputs and Bash syntax validated')

        # Simulate an SCF executable that exits zero but reports nonconvergence.
        bin_dir = root / 'fake_bin'
        bin_dir.mkdir()
        (root / 'cluster_env.sh').write_text('# Test fixture: no cluster modules\n')
        (bin_dir / 'mpirun').write_text('#!/bin/bash\nshift 2\nexec "$@"\n')
        (bin_dir / 'pw.x').write_text('#!/bin/bash\necho "convergence NOT achieved"\necho "JOB DONE."\n')
        for script in bin_dir.iterdir():
            script.chmod(0o755)
        env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], SLURM_NTASKS='8')
        p = subprocess.run(['bash', str(root / 'run_wannier.sh'), str(aa)], env=env, capture_output=True, text=True)
        assert p.returncode != 0, p.stdout + p.stderr
        assert not (aa / 'nscf.out').exists()
        assert not (aa / 'wannier.complete').exists()
        assert not (aa / '.pipeline_lock').exists()
        print('PASS: electronic failure halts chain before NSCF, even with executable exit code zero')

        for name in ['scf.out', 'nscf.out', 'pw2wan.out']:
            (aa / name).write_text('convergence has been achieved\nJOB DONE.\n')
        (aa / 'NbSe2.wout').write_text('Lattice Vectors (Ang)\nCartesian Coordinate (Ang)\nAll done\n')
        hr = 'synthetic fixture\n4\n1\n1\n' + ''.join(
            f'0 0 0 {i} {j} 0.1 0.0\n' for i in range(1, 5) for j in range(1, 5))
        (aa / 'NbSe2_hr.dat').write_text(hr)
        for name in ['NbSe2_wsvec.dat', 'NbSe2_band.dat']:
            (aa / name).write_text('synthetic nonempty fixture\n')
        run(root, 'check_wannier.py', '--stacking', aa, '--record')
        run(root, 'check_wannier.py', '--stacking', aa)
        (aa / 'NbSe2_hr.dat').write_text('\n'.join(hr.splitlines()[:-1]))
        run(root, 'check_wannier.py', '--stacking', aa, ok=False)
        (aa / 'NbSe2_hr.dat').write_text(hr)
        original = (aa / 'scf.in').read_bytes()
        (aa / 'scf.in').write_bytes(original.replace(b'nbnd = 108', b'nbnd = 109'))
        run(root, 'check_wannier.py', '--stacking', aa, ok=False)
        (aa / 'scf.in').write_bytes(original)
        run(root, 'check_wannier.py', '--stacking', aa)
        original_pp = (root / 'Nb.upf').read_bytes()
        (root / 'Nb.upf').write_bytes(bytes([original_pp[0] ^ 1]) + original_pp[1:])
        run(root, 'check_wannier.py', '--stacking', aa, ok=False)
        (root / 'Nb.upf').write_bytes(original_pp)
        run(root, 'check_wannier.py', '--stacking', aa)
        (aa / 'wannier.complete').write_bytes(b'invalid completion record')
        run(root, 'check_wannier.py', '--stacking', aa, ok=False)
        print('PASS: complete Hamiltonian accepted; truncated Hamiltonian and changed inputs rejected')
    print('All local checks passed. Physical convergence and Nibi execution still require real jobs.')


if __name__ == '__main__':
    main()
