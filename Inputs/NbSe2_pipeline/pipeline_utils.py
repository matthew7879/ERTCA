"""Validation helpers: electronic convergence alone is NOT ionic convergence."""

import math
from pathlib import Path
import re
import settings as s
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

ROOT = Path(__file__).resolve().parent
GRID = ROOT / 'grid12'
NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?'


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


def check_settings():
    if (GRID / 'settings_snapshot.py').read_bytes() != (ROOT / 'settings.py').read_bytes():
        raise ValueError('settings.py changed after grid generation; use a fresh pipeline folder')


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
            raise ValueError(f'{key} missing or looser than settings.py')
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


COMPLETION_INPUTS = ('relax.out', 'scf.in', 'nscf.in', 'pw2wan.in',
                     'NbSe2.win', 'Nb.upf', 'Se.upf')


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
