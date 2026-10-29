#!/usr/bin/env python3
"""NbSe2 12×12 relaxation, SCF/NSCF and Wannier pipeline (Python standard library)."""

import argparse
from contextlib import contextmanager
import json
import math
from math import sqrt
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

# Edit settings before `generate`; scientific defaults match the supplied run.
class Settings:
    N = 12
    ECUTWFC = 80                 # Ry, used in relax, SCF, and NSCF
    ECUTRHO = 800                # Ry, used in relax, SCF, and NSCF
    ETOT_CONV_THR = 1.0e-5       # Ry, ionic energy-change threshold
    FORC_CONV_THR = 1.0e-4       # Ry/Bohr, about 0.002571 eV/Angstrom

    # Existing physical setup: fixed cell; only atomic z coordinates relax.
    ALAT_BOHR = 6.5195546606
    C_OVER_A = 5.797101449
    DEGAUSS = 0.02              # Ry, Gaussian smearing; test separately from ionic tolerances
    RELAX_K = 20
    SCF_K = 25
    WANNIER_K = 20
    RELAX_SCF_THR = 1.0e-10
    SCF_THR = 1.0e-8
    NBND = 108                 # The original exclude_bands assumes exactly 108 spinor bands
    BOHR_TO_ANG = 0.529177210903

    # Original projection gauge retained, including the three-decimal xy coordinates.
    PROJECTION_X = 0.667
    PROJECTION_Y = 0.333
    NUM_ITER = 0

    BOTTOM = [
        ('Nb', 0.0,          0.0,          -0.0204149343),
        ('Se', 0.3333333330, 0.6666666670,  0.0636847574),
        ('Se', 0.3333333330, 0.6666666670, -0.1050537796),
    ]
    TOP = [
        ('Nb', 0.0,          0.0,           0.3352708415),
        ('Se', 0.3333333330, 0.6666666670,  0.4199193132),
        ('Se', 0.3333333330, 0.6666666670,  0.2511811403),
    ]

    # Cluster settings: edit for your account before generating a fresh grid.
    ACCOUNT = 'rrg-vcrepel'
    NTASKS = 20
    MEM_PER_CPU = '4G'
    WALLTIME = '24:00:00'
    MAX_CONCURRENT = 20
    MODULES = ['StdEnv/2023', 'intel/2023.2.1', 'openmpi/4.1.5', 'quantumespresso/7.3.1']
    WANNIER_MODULE = 'wannier90/3.1.0'

s = Settings
ROOT = Path(__file__).resolve().parent
GRID = ROOT / 'grid12'
NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?'
COMPLETION_INPUTS = ('relax.out', 'scf.in', 'nscf.in', 'pw2wan.in',
                     'NbSe2.win', 'Nb.upf', 'Se.upf')

PW2WAN = """&INPUTPP
 outdir = './',
 prefix = 'NbSe2',
 seedname = 'NbSe2',
 write_mmn = .true.,
 write_amn = .true.,
 write_unk = .false.,
/
"""

def shifted_atoms(i, j):
    return s.BOTTOM + [(el, (x + i/s.N) % 1, (y + j/s.N) % 1, z)
                       for el, x, y, z in s.TOP]

def positions(atoms, relax=False):
    flags = '   0 0 1' if relax else ''
    return '\n'.join(f'{el:2s} {x:18.10f} {y:18.10f} {z:18.10f}{flags}'
                     for el, x, y, z in atoms)

def pw_input(stage, atoms):
    relax = stage == 'relax'
    ionic_control = (f' etot_conv_thr = {s.ETOT_CONV_THR:.8e},\n'
                     f' forc_conv_thr = {s.FORC_CONV_THR:.8e},\n nstep = 200,\n') if relax else ''
    band_control = '' if relax else f' nosym = .true.,\n nbnd = {s.NBND},\n'
    electrons = (f' conv_thr = {s.RELAX_SCF_THR:.8e},\n'
                 " mixing_beta = 0.2,\n startingpot = 'atomic',\n") if relax else f' conv_thr = {s.SCF_THR:.8e},\n'
    ions = "&IONS\n ion_dynamics = 'bfgs',\n/\n" if relax else ''
    if stage == 'nscf':
        k = s.WANNIER_K
        kpoints = f'K_POINTS crystal\n{k*k}\n' + '\n'.join(
            f'{i/k:.10f} {j/k:.10f} 0.0 {1/(k*k):.12e}'
            for i in range(k) for j in range(k))
    else:
        k = s.RELAX_K if relax else s.SCF_K
        kpoints = f'K_POINTS automatic\n{k} {k} 1 0 0 0'
    return f"""&CONTROL
 calculation = '{stage}',
 restart_mode = 'from_scratch',
 prefix = 'NbSe2',
 pseudo_dir = '.',
 outdir = './',
 verbosity = 'high',
 tprnfor = .true.,
{ionic_control}/
&SYSTEM
 ibrav = 4,
 celldm(1) = {s.ALAT_BOHR:.12f}, celldm(3) = {s.C_OVER_A:.12f},
 nat = 6, ntyp = 2,
 ecutwfc = {s.ECUTWFC}, ecutrho = {s.ECUTRHO},
 lspinorb = .true., noncolin = .true.,
{band_control} vdw_corr = 'grimme-d3',
 occupations = 'smearing', smearing = 'gaussian', degauss = {s.DEGAUSS},
/
&ELECTRONS
{electrons}/
{ions}ATOMIC_SPECIES
 Nb 92.90638 Nb.upf
 Se 78.96 Se.upf
ATOMIC_POSITIONS (crystal)
{positions(atoms, relax)}
{kpoints}
"""

def win_input(atoms):
    a, c = s.ALAT_BOHR, s.ALAT_BOHR * s.C_OVER_A
    k = s.WANNIER_K
    kp = '\n'.join(f'{i/k:.10f} {j/k:.10f} 0.0'
                   for i in range(k) for j in range(k))
    return f"""! Keep the same four-band projection gauge as the original pipeline.
num_bands = 4
num_wann = 4
exclude_bands : 1-88,93-108
num_iter = {s.NUM_ITER}
iprint = 3
num_print_cycles = 50
bands_plot = .true.
write_hr = .true.
! Real-space orbital pictures require UNK files, which we do not generate.
wannier_plot = .false.
use_ws_distance = .true.
spinors = true

begin kpoint_path
 G 0.0 0.0 0.0 M 0.5 0.0 0.0
 M 0.5 0.0 0.0 K 0.333333 0.333333 0.0
 K 0.333333 0.333333 0.0 G 0.0 0.0 0.0
end kpoint_path

begin unit_cell_cart
 bohr
 {a:.12f} 0.0 0.0
 {-a/2:.12f} {a*sqrt(3)/2:.12f} 0.0
 0.0 0.0 {c:.12f}
end unit_cell_cart

begin atoms_frac
{positions(atoms)}
end atoms_frac

begin projections
 f={s.PROJECTION_X:.3f},{s.PROJECTION_Y:.3f},{atoms[0][3]:.10f}:s:zona=4.0
 f={(s.PROJECTION_X+atoms[3][1])%1:.3f},{(s.PROJECTION_Y+atoms[3][2])%1:.3f},{atoms[3][3]:.10f}:s:zona=4.0
end projections

mp_grid : {k} {k} 1
begin kpoints
{kp}
end kpoints
"""

def number(value):
    result = float(value.replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(result):
        raise ValueError('non-finite number')
    return result

def shift_name(i, j):
    return f'shift_i{i:02d}_j{j:02d}'

def indices(only=None):
    if only is not None:
        if not all(0 <= v < s.N for v in only):
            raise ValueError(f'indices must be between 0 and {s.N-1}')
        return [tuple(only)]
    return [(i, j) for i in range(s.N) for j in range(s.N)]

def settings_text():
    return json.dumps({key: value for key, value in vars(s).items() if key.isupper()},
                      sort_keys=True, indent=2) + '\n'


def check_settings():
    if (GRID / 'settings_snapshot.json').read_text() != settings_text():
        raise ValueError('Settings changed after grid generation; use a fresh pipeline folder')


def qe_finished(path, scf=False):
    text = Path(path).read_text(errors='replace')
    # If a file was accidentally appended, only inspect its last QE run.
    text = re.split(r'Program\s+(?:PWSCF|PW2WANNIER)', text)[-1]
    if 'JOB DONE.' not in text or 'Error in routine' in text:
        raise ValueError(f'{Path(path).name}: missing JOB DONE or QE error')
    if 'convergence NOT achieved' in text:
        raise ValueError(f'{Path(path).name}: electronic convergence failed')
    if scf and 'convergence has been achieved' not in text:
        raise ValueError(f'{Path(path).name}: no electronic convergence report')
    return text

def final_total_forces(text, natoms=6):
    """Read the last TOTAL atomic-force table, not a verbose contribution table.

    QE prints DFT-D3 and other individual contributions after the total forces
    when verbosity='high'. Their rows have exactly the same 'atom ... force ='
    format, so taking the last natoms matching rows in the file is incorrect.
    """
    headers = list(re.finditer(
        r'^\s*Forces acting on atoms\s*\(cartesian axes,\s*Ry/au\):[^\S\n]*$',
        text, re.M))
    if not headers:
        raise ValueError('missing total-force table headed "Forces acting on atoms"')
    row_pattern = re.compile(
        rf'\s*atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*'
        rf'({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*')
    rows = []
    # Read only contiguous atom rows immediately after the last total-force header.
    # Do not fall back to an older step or continue into a contribution table.
    for line in text[headers[-1].end():].splitlines():
        if not line.strip() and not rows:
            continue
        match = row_pattern.fullmatch(line)
        if not match:
            break
        rows.append(match.groups())
    if len(rows) != natoms or [int(row[0]) for row in rows] != list(range(1, natoms + 1)):
        raise ValueError(f'missing or incomplete final {natoms}-atom total-force table')
    return [tuple(number(value) for value in row[1:]) for row in rows]

def read_relax(folder):
    folder = Path(folder)
    text = qe_finished(folder / 'relax.out', scf=True)
    if 'bfgs converged in' not in text.lower():
        raise ValueError('no BFGS ionic convergence report')
    blocks = re.findall(r'Begin final coordinates(.*?)End final coordinates', text, re.S)
    if not blocks:
        raise ValueError('no completed final-coordinate block')
    match = re.search(r'ATOMIC_POSITIONS\s*\(crystal\)\s*\n(.*)', blocks[-1], re.S)
    if not match:
        raise ValueError('final coordinates are not in crystal units')
    atoms = []
    for line in match[1].strip().splitlines():
        fields = line.split()
        if fields:
            atoms.append((fields[0], *map(number, fields[1:4])))
    if [row[0] for row in atoms] != ['Nb', 'Se', 'Se', 'Nb', 'Se', 'Se']:
        raise ValueError('expected Nb Se Se Nb Se Se in the final structure')
    if any(len(row) != 4 for row in atoms):
        raise ValueError('malformed final coordinates')
    inp = (folder / 'relax.in').read_text()
    for key, target in [('forc_conv_thr', s.FORC_CONV_THR), ('etot_conv_thr', s.ETOT_CONV_THR)]:
        m = re.search(rf'\b{key}\s*=\s*({NUMBER})', inp, re.I)
        if not m or number(m[1]) > target * (1 + 1e-10):
            raise ValueError(f'{key} missing or looser than pipeline.py settings')
    forces = final_total_forces(text, natoms=len(atoms))
    # x and y are constrained. Only z is an optimized coordinate in this pipeline.
    max_fz = max(abs(row[2]) for row in forces)
    if max_fz > s.FORC_CONV_THR + 5e-8:
        raise ValueError(f'final max |Fz| = {max_fz:.3e} exceeds requested force tolerance')
    energies = re.findall(rf'!\s+total energy\s*=\s*({NUMBER})\s+Ry', text)
    if not energies:
        raise ValueError('missing total energy')
    distance = abs(atoms[3][3] - atoms[0][3]) * s.ALAT_BOHR * s.C_OVER_A * s.BOHR_TO_ANG
    return dict(atoms=atoms, energy=number(energies[-1]), max_fz=max_fz, distance=distance)

def record_completion(folder):
    """Save exact input bytes after a successful validation, with compression."""
    folder = Path(folder)
    temporary = folder / 'wannier.complete.tmp'
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as saved:
            for name in COMPLETION_INPUTS:
                saved.write(folder / name, arcname=name)
        temporary.replace(folder / 'wannier.complete')
    finally:
        if temporary.exists():
            temporary.unlink()

def check_completion(folder):
    """Require every current input to match the saved bytes exactly."""
    folder = Path(folder)
    try:
        with ZipFile(folder / 'wannier.complete') as saved:
            if saved.namelist() != list(COMPLETION_INPUTS):
                raise ValueError('invalid Wannier completion record')
            for name in COMPLETION_INPUTS:
                if saved.read(name) != (folder / name).read_bytes():
                    raise ValueError(f'{name} changed since successful Wannier run')
    except BadZipFile as exc:
        raise ValueError('unreadable Wannier completion record; validate outputs before recording again') from exc

def check_wannier(folder, require_marker=True):
    folder = Path(folder)
    read_relax(folder)
    qe_finished(folder / 'scf.out', scf=True)
    qe_finished(folder / 'nscf.out')
    qe_finished(folder / 'pw2wan.out')
    wout = (folder / 'NbSe2.wout').read_text()
    if 'All done' not in wout:
        raise ValueError('Wannier90 did not report All done')
    if (folder / 'NbSe2.werr').exists():
        raise ValueError('NbSe2.werr exists; inspect the Wannier error')
    if 'Lattice Vectors (Ang)' not in wout or 'Cartesian Coordinate (Ang)' not in wout:
        raise ValueError('Wannier log lacks the cell or atomic site table')
    # Reject a truncated hr file even if some output files exist.
    with (folder / 'NbSe2_hr.dat').open() as inp:
        next(inp)
        nw, nr = int(next(inp)), int(next(inp))
        if nw != 4 or nr < 1:
            raise ValueError('expected a nonempty four-Wannier-function Hamiltonian')
        degeneracies = []
        while len(degeneracies) < nr:
            degeneracies.extend(map(int, next(inp).split()))
        if len(degeneracies) != nr or min(degeneracies) < 1:
            raise ValueError('invalid hr degeneracies')
        rows = [line.split() for line in inp if line.strip()]
        if len(rows) != nw*nw*nr or any(len(row) != 7 for row in rows):
            raise ValueError('truncated or malformed hr matrix')
        for row in rows:
            number(row[5]); number(row[6])
    for name in ['NbSe2_wsvec.dat', 'NbSe2_band.dat']:
        if (folder / name).stat().st_size == 0:
            raise ValueError(f'{name} is empty')
    if require_marker:
        check_completion(folder)

def write_same_or_new(path, content):
    """Allow regenerating identical inputs; never replace an existing different input."""
    if path.exists() and path.read_text() != content:
        raise ValueError(f'{path}: existing content differs; use a fresh directory')
    path.write_text(content)

def read_geometry(path):
    text = path.read_text()
    lattice_block = re.search(
        r"Lattice Vectors \(Ang\)(.*?)Unit Cell Volume", text, re.S
    )
    if lattice_block is None:
        raise ValueError("missing lattice vectors in Angstrom")
    lattice = {}
    for line in lattice_block[1].splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] in ("a_1", "a_2", "a_3"):
            lattice[fields[0]] = tuple(map(number, fields[1:]))
    if len(lattice) != 3:
        raise ValueError("expected three lattice vectors")

    lines = text.splitlines()
    start = next((k for k, line in enumerate(lines)
                  if "Site" in line and "Cartesian Coordinate (Ang)" in line), None)
    if start is None:
        raise ValueError("missing atomic Site table; reconstructed .wout files cannot be used")
    nb = {}
    for line in lines[start + 1:]:
        if line.strip().startswith("*---"):
            break
        fields = line.split("|")
        if len(fields) < 4:
            continue
        site = fields[1].split()
        if not site or site[0] != "Nb":
            continue
        cart = fields[2].split()
        if len(site) != 5 or len(cart) != 3:
            raise ValueError("malformed Nb atom row")
        index = int(site[1])
        if index in nb:
            raise ValueError("duplicate Nb atom index")
        nb[index] = tuple(map(number, cart))
    if len(nb) != 2:
        raise ValueError("expected exactly two Nb atoms, found {}".format(len(nb)))
    if not all(math.isfinite(v) for xyz in list(lattice.values()) + list(nb.values()) for v in xyz):
        raise ValueError("non-finite atomic coordinates or lattice vectors")
    if any(abs(lattice[key][2]) > 1e-6 for key in ("a_1", "a_2")):
        raise ValueError("a1/a2 are not in the xy plane; z is not the layer normal")
    z_bottom, z_top = sorted(xyz[2] for xyz in nb.values())
    return lattice, z_bottom, z_top

def extract(root, ngrid):
    rows, errors = [], []
    reference = None
    for i in range(ngrid):
        for j in range(ngrid):
            path = root / "shift_i{:02d}_j{:02d}".format(i, j) / "NbSe2.wout"
            try:
                lattice, z_bottom, z_top = read_geometry(path)
                if reference is None:
                    reference = lattice
                if any(abs(lattice[key][k] - reference[key][k]) > 1e-5
                       for key in reference for k in range(3)):
                    raise ValueError("lattice differs from the first stacking")
                u, v = i / ngrid, j / ngrid
                dx, dy = (u * lattice["a_1"][k] + v * lattice["a_2"][k] for k in (0, 1))
                rows.append((i, j, u, v, dx, dy, z_bottom, z_top, z_top - z_bottom))
            except (OSError, ValueError) as exc:
                errors.append("{}: {}".format(path, exc))
    if errors:
        raise ValueError("No output written: {} of {} stackings failed.\n{}".format(
            len(errors), ngrid * ngrid, "\n".join(errors)))
    return reference, rows

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
    modules = s.MODULES + ([s.WANNIER_MODULE] if stage == 'wannier' else [])
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
module purge
module load {' '.join(shlex.quote(m) for m in modules)}
export OMP_NUM_THREADS=1
python3 "$root/pipeline.py" run-{stage} "$folder"
'''


def generate():
    for name in ['Nb.upf', 'Se.upf']:
        if not (ROOT / name).is_file():
            raise ValueError(f'Missing {ROOT / name}')
    if GRID.exists():
        raise ValueError(f'{GRID} already exists. Use a fresh pipeline copy.')
    GRID.mkdir()
    (GRID / 'settings_snapshot.json').write_text(settings_text())
    for i, j in indices():
        folder = GRID / shift_name(i, j)
        folder.mkdir()
        for name in ['Nb.upf', 'Se.upf']:
            (folder / name).symlink_to('../../' + name)
        (folder / 'relax.in').write_text(pw_input('relax', shifted_atoms(i, j)))
        for stage in ['relax', 'wannier']:
            (folder / f'batch_{stage}.sh').write_text(batch(stage, i, j))
    for stage in ['relax', 'wannier']:
        (GRID / f'submit_{stage}_array.sh').write_text(batch(stage))
    print(f'Created {len(indices())} shifts in {GRID}')
    print('Test AA: python3 pipeline.py submit-relax --only 0 0')


def prepare(only=None):
    check_settings()
    jobs, failures = [], []
    for i, j in indices(only):
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


def check_grid(stage, only=None):
    check_settings()
    failed, good, details = [], [], []
    for i, j in indices(only):
        folder = GRID / shift_name(i, j)
        try:
            if stage == 'relax':
                r = read_relax(folder)
                zs = ' '.join(f'{row[3]:.10f}' for row in r['atoms'])
                good.append(f'{i} {j} {i/s.N:.10f} {j/s.N:.10f} {zs} '
                            f'{r["energy"]:.10f} {r["distance"]:.10f} {r["max_fz"]:.8e}\n')
                print(f'{folder.name}: OK  Nb-Nb={r["distance"]:.7f} Angstrom')
            else:
                check_wannier(folder)
                print(f'{folder.name}: OK')
        except (OSError, ValueError, StopIteration) as exc:
            failed.append(folder.name)
            details.append(f'{folder.name}: {exc}\n')
            print(details[-1].rstrip())
    if only is None:
        name = 'failed.txt' if stage == 'relax' else 'wannier_failed.txt'
        (GRID / name).write_text(''.join(item + '\n' for item in failed))
        if stage == 'relax':
            (GRID / 'relaxed_positions.dat').write_text(
                '# i j i/12 j/12 Nb1_z Se1_z Se2_z Nb2_z Se3_z Se4_z E_Ry NbNb_A max_abs_Fz_Ry_Bohr\n' + ''.join(good))
            (GRID / 'relaxation_failures.txt').write_text(''.join(details))
    total = len(indices(only))
    print(f'Complete: {total-len(failed)}/{total}')
    return int(bool(failed))


def submit(stage, only=None):
    check_settings()
    indices(only)  # validate before selecting a directory
    if stage == 'wannier':
        prepare(only)
    if only is None:
        folder = ROOT
        script = GRID / f'submit_{stage}_array.sh'
    else:
        folder = GRID / shift_name(*only)
        script = folder / f'batch_{stage}.sh'
    # Slurm records this cwd as SLURM_SUBMIT_DIR, used by the generated script.
    subprocess.run(['sbatch', str(script)], cwd=folder, check=True)


@contextmanager
def job_lock(folder):
    lock = folder / '.pipeline_lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f'{folder.name}: another job is running, or a stale .pipeline_lock remains')
    try:
        yield
    finally:
        lock.rmdir()


def execute(folder, argv, output):
    with (folder / output).open('w') as out:
        subprocess.run(argv, cwd=folder, stdout=out, check=True)


def run_stage(stage, folder):
    check_settings()
    folder = Path(folder).resolve()
    with job_lock(folder):
        if stage == 'relax':
            if (folder / 'relax.out').exists():
                read_relax(folder)
                print('Relaxation already complete; skipping.')
                return
            np = os.environ.get('SLURM_NTASKS')
            if not np:
                raise ValueError('Run through sbatch: SLURM_NTASKS is missing')
            execute(folder, ['mpirun', '-np', np, 'pw.x', '-in', 'relax.in'], 'relax.out')
            read_relax(folder)
            return
        read_relax(folder)
        if (folder / 'wannier.complete').exists():
            check_wannier(folder)
            print('Wannier chain already complete; skipping.')
            return
        for name in ['scf.out', 'nscf.out', 'pw2wan.out', 'NbSe2.wout', 'NbSe2.werr', 'NbSe2_hr.dat']:
            if (folder / name).exists():
                raise ValueError(f'Existing {name}: archive failed Wannier outputs before retrying')
        for name in ['scf.in', 'nscf.in', 'pw2wan.in', 'NbSe2.win']:
            if not (folder / name).is_file() or not (folder / name).stat().st_size:
                raise ValueError(f'Missing {name}: run pipeline.py prepare first')
        np = os.environ.get('SLURM_NTASKS')
        if not np:
            raise ValueError('Run through sbatch: SLURM_NTASKS is missing')
        mpi = ['mpirun', '-np', np]
        for qe_stage in ['scf', 'nscf']:
            print(qe_stage.upper(), flush=True)
            execute(folder, mpi + ['pw.x', '-in', f'{qe_stage}.in'], f'{qe_stage}.out')
            qe_finished(folder / f'{qe_stage}.out', scf=qe_stage == 'scf')
        print('Wannier preprocessing', flush=True)
        execute(folder, ['wannier90.x', '-pp', 'NbSe2'], 'wannier_pp.out')
        if not (folder / 'NbSe2.nnkp').stat().st_size or (folder / 'NbSe2.werr').exists():
            raise ValueError('Wannier preprocessing failed')
        print('QE to Wannier90', flush=True)
        execute(folder, mpi + ['pw2wannier90.x', '-in', 'pw2wan.in'], 'pw2wan.out')
        qe_finished(folder / 'pw2wan.out')
        print('Wannier Hamiltonian', flush=True)
        execute(folder, ['wannier90.x', 'NbSe2'], 'wannier_run.out')
        check_wannier(folder, require_marker=False)
        record_completion(folder)
        if os.environ.get('CLEANUP_QE') == '1':
            for path in folder.glob('NbSe2.wfc*'):
                path.unlink()
            scratch = folder / 'NbSe2.save'
            if scratch.is_symlink():
                scratch.unlink()
            elif scratch.exists():
                shutil.rmtree(scratch)
        print('DONE')


def distances(output):
    check_settings()
    lattice, rows = extract(GRID, s.N)
    with Path(output).open('w') as out:
        out.write('# Nb atomic layer separation from original NbSe2.wout Site tables\n')
        out.write(f'# source_grid: {GRID.resolve()}\n# ngrid: {s.N}\n')
        out.write('# d = u*a1 + v*a2; u=i/ngrid, v=j/ngrid\n')
        out.write('# distance_A = abs(z_Nb2-z_Nb1); no lateral distance or periodic wrapping\n')
        for key in ('a_1', 'a_2', 'a_3'):
            out.write('# {}_Ang: {:.8f} {:.8f} {:.8f}\n'.format(key, *lattice[key]))
        out.write('# i j u v dx_A dy_A z_bottom_A z_top_A distance_A\n')
        for row in rows:
            out.write('{:d} {:d} {:.10f} {:.10f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(*row))
    print(f'Wrote {len(rows)} stackings to {output}')
    print('Nb-Nb vertical distance: {:.5f} to {:.5f} Angstrom'.format(
        min(row[-1] for row in rows), max(row[-1] for row in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('generate', help='create a fresh 144-shift grid and job scripts')
    for command in ['prepare', 'submit-relax', 'submit-wannier', 'check-relax', 'check-wannier']:
        sub = commands.add_parser(command)
        sub.add_argument('--only', type=int, nargs=2, metavar=('I', 'J'))
    sub = commands.add_parser('distances', help='export atomic Nb-Nb separations')
    sub.add_argument('--output', type=Path, default=ROOT / 'NbSe2_interlayer_distance_regenerated.dat')
    for command in ['run-relax', 'run-wannier']:
        sub = commands.add_parser(command, help='internal: run one job inside a Slurm allocation')
        sub.add_argument('folder', type=Path)
    args = parser.parse_args()
    if args.command == 'generate':
        generate()
    elif args.command == 'prepare':
        prepare(args.only)
    elif args.command.startswith('submit-'):
        submit(args.command.split('-')[1], args.only)
    elif args.command.startswith('check-'):
        return check_grid(args.command.split('-')[1], args.only)
    elif args.command.startswith('run-'):
        run_stage(args.command.split('-')[1], args.folder)
    else:
        distances(args.output)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, StopIteration, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
