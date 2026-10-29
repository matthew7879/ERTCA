#!/usr/bin/env python3
"""Generate and validate WSe2 12×12 relaxation, SCF/NSCF and Wannier inputs."""

import argparse
import itertools
import math
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from ase.io import read, write
from ase.units import Bohr

ROOT = Path(__file__).resolve().parent
GRID = ROOT / 'grid12'
N = 12
A_TARGET = 3.295003576       # Angstrom, fixed in-plane lattice parameter
INTERLAYER_GUESS = 6.5     # Angstrom, starting layer separation
VACUUM = 15.0             # Angstrom, added on either side of the bilayer
PSEUDO_DIR = '.'
PSEUDOS = {'W': 'W.upf', 'Se': 'Se.upf'}
MASSES = {'W': 183.84, 'Se': 78.971}
NBND = 110
SPREAD_THRESHOLD = 14.5   # Angstrom^2, original mean-spread check
MATCH_TOLERANCE = 0.5     # Angstrom, original projection-centre check
BOHR_TO_ANG = 0.529177210903

def build_bilayer(cif_path='WSe2.cif', a_target=A_TARGET,
                   shift_i=0, shift_j=0, nk=12,
                   interlayer=INTERLAYER_GUESS, vacuum=VACUUM):
    """Read the monolayer CIF, rescale it to a_target, stack a second
    (optionally in-plane-shifted) copy on top, and add vacuum."""

    layer1 = read(cif_path)
    layer1.pbc = (True, True, False)

    # Rescale ONLY the in-plane (a1, a2) vectors to the target lattice
    # constant -- leave the CIF's own (irrelevant) c-vector alone.
    cell = layer1.get_cell()
    a_old = np.linalg.norm(cell[0])
    scale = a_target / a_old
    new_cell = cell.copy()
    new_cell[0] = cell[0] * scale
    new_cell[1] = cell[1] * scale
    layer1.set_cell(new_cell, scale_atoms=True)

    # Top layer: same in-plane cell, optionally shifted by d = i/nk*a1 + j/nk*a2,
    # then displaced out-of-plane to sit above the bottom layer.
    layer2 = layer1.copy()
    if shift_i or shift_j:
        frac2 = layer2.get_scaled_positions(wrap=False)
        frac2[:, 0] += shift_i / nk
        frac2[:, 1] += shift_j / nk
        frac2 %= 1.0
        layer2.set_scaled_positions(frac2)
    layer2.translate((0, 0, interlayer))

    bilayer = layer1 + layer2
    bilayer.pbc = (True, True, False)
    bilayer.center(vacuum=vacuum, axis=2)
    return bilayer

def write_relax_in(bilayer, a_target=A_TARGET, outfile='relax.in', nosym=False):
    """Write the original fixed-cell, z-only relaxation input."""

    cell = bilayer.get_cell()
    c_len = cell[2, 2]                 # Angstrom
    celldm1 = a_target / Bohr          # a, in bohr
    celldm3 = c_len / a_target         # c/a (dimensionless -> unit-independent)

    symbols = bilayer.get_chemical_symbols()
    species = list(dict.fromkeys(symbols))     # preserve first-seen order
    frac = bilayer.get_scaled_positions(wrap=True)

    lines = []
    lines.append('&control')
    lines.append("    calculation='relax'")
    lines.append(f"    pseudo_dir = '{PSEUDO_DIR}'")
    lines.append("    outdir='./'")
    lines.append("    prefix='WSe2'")
    lines.append('    etot_conv_thr = 1.0D-4,')
    lines.append('    forc_conv_thr = 1.0D-3')
    lines.append(' /')
    lines.append(' &system')
    lines.append('    ibrav = 4,')
    lines.append(f'    celldm(1) = {celldm1:.10f}, celldm(3) = {celldm3:.10f},')
    lines.append(f'    nat = {len(bilayer)},')
    lines.append(f'    ntyp = {len(species)},')
    lines.append('    ecutwfc = 80, ecutrho = 800,')
    lines.append('    lspinorb = .true.,')
    lines.append('    noncolin = .true.,')
    if nosym:
        lines.append('    nosym = .true.,')
    lines.append("    vdw_corr = 'grimme-d3',")
    lines.append("    occupations = 'smearing', smearing = 'gaussian', degauss = 0.02")
    lines.append(' /')
    lines.append(' &electrons')
    lines.append('    conv_thr = 1.0e-8,')
    lines.append('    mixing_beta = 0.2,')
    lines.append("    startingpot = 'atomic',")
    lines.append(' /')
    lines.append('&ions')
    lines.append("    ion_dynamics = 'bfgs'")
    lines.append('/')
    lines.append('ATOMIC_SPECIES')
    for sp in species:
        lines.append(f'    {sp:<4s}{MASSES[sp]:>10.5f}   {PSEUDOS[sp]}')
    lines.append('ATOMIC_POSITIONS (crystal)')
    for sym, (fx, fy, fz) in zip(symbols, frac):
        lines.append(
            f'{sym:<4s}  {fx:>18.10f}  {fy:>18.10f}  {fz:>18.10f}   0 0 1'
        )
    lines.append('K_POINTS (automatic)')
    lines.append('    20 20 1 0 0 0')

    with open(outfile, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return outfile

def build_projection_fracs(atoms, shift_i, shift_j, nk):
    """Keep the original fractional projection gauge and relaxed layer heights."""
    symbols = atoms.get_chemical_symbols()
    frac = atoms.get_scaled_positions(wrap=True)
    positions = atoms.get_positions()

    w_indices = [k for k, s in enumerate(symbols) if s == 'W']
    if len(w_indices) != 2:
        raise RuntimeError(
            f"Expected exactly 2 W atoms (one per layer), found {len(w_indices)}."
        )
    # Order by cartesian z: lower z = bottom layer, higher z = top layer.
    w_indices.sort(key=lambda k: positions[k, 2])
    bottom_idx, top_idx = w_indices

    z_bottom = frac[bottom_idx, 2]
    z_top = frac[top_idx, 2]

    bottom_xy = (0.0, 0.0)                       # bottom layer never shifted
    top_xy = (shift_i / nk, shift_j / nk)        # top layer shifted by d

    return [(bottom_xy[0], bottom_xy[1], z_bottom),
            (top_xy[0], top_xy[1], z_top)]

def write_downstream(folder, wse2, shift_i, shift_j, nk):
    wse2.pbc = (True, True, False)

    pseudopotentials = {'W': 'W.upf', 'Se': 'Se.upf'}

    base_system = {
        'ecutwfc': 80,
        'ecutrho': 800,
        'occupations': 'smearing',
        'smearing': 'gaussian',
        'degauss': 0.02,
        'noncolin': True,
        'lspinorb': True,
        'vdw_corr': 'grimme-d3',
        'nosym': True,
    }
    base_electrons = {
        'conv_thr': 1.0e-8,
        'mixing_beta': 0.2,
        'startingpot': 'atomic',
    }

    # SCF: 25x25x1 k-grid.
    scf_input_data = {
        'control': {
            'calculation': 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir': PSEUDO_DIR,
            'outdir': './',
            'prefix': 'WSe2',
            'verbosity': 'high',
        },
        'system': base_system,
        'electrons': base_electrons,
    }

    write(folder / 'scf.in', wse2, format='espresso-in',
          pseudopotentials=pseudopotentials,
          kpts=(25, 25, 1),
          koffset=(0, 0, 0),
          input_data=scf_input_data)

    # Build the explicit uniform 20x20x1 k-grid shared by nscf.in
    # (QE "crystal" units, with weights) and WSe2.win (Wannier90
    # "kpoints" block, no weights) -- must match 1:1.
    NK_W90 = 20
    kgrid = [(i / NK_W90, j / NK_W90, 0.0) for i in range(NK_W90) for j in range(NK_W90)]
    weight = 1.0 / len(kgrid)

    kpoints_block_qe = 'K_POINTS crystal\n{}\n'.format(len(kgrid))
    for kx, ky, kz in kgrid:
        kpoints_block_qe += '  {:.8f}  {:.8f}  {:.8f}  {:.6e}\n'.format(kx, ky, kz, weight)

    kpoints_block_w90 = 'begin kpoints\n'
    for kx, ky, kz in kgrid:
        kpoints_block_w90 += '{:.8f} {:.8f} {:.8f}\n'.format(kx, ky, kz)
    kpoints_block_w90 += 'end kpoints\n'

    # NSCF: explicit 20x20x1 uniform k-grid (400 points).
    # Retain the original explicit NSCF band count and default SCF band count.
    nscf_system = dict(base_system, nbnd=NBND)

    nscf_input_data = {
        'control': {
            'calculation': 'nscf',
            'pseudo_dir': PSEUDO_DIR,
            'outdir': './',
            'prefix': 'WSe2',
            'verbosity': 'high',
        },
        'system': nscf_system,
        'electrons': base_electrons,
    }

    # Write with a placeholder k-point mesh first; overwrite the K_POINTS
    # block by hand right after (ASE needs *some* kpts argument).
    write(folder / 'nscf.in', wse2, format='espresso-in',
          pseudopotentials=pseudopotentials,
          kpts=(1, 1, 1),
          input_data=nscf_input_data)

    with open(folder / 'nscf.in', 'r') as f:
        text = f.read()

    # ASE's espresso-in writer places K_POINTS *before* ATOMIC_POSITIONS
    # and CELL_PARAMETERS. Only replace the existing K_POINTS block itself
    # (up to, but not including, the next block header or EOF).
    pattern = re.compile(
        r'K_POINTS.*?\n(?:.*\n)*?(?=\nATOMIC_POSITIONS|\nCELL_PARAMETERS|\Z)',
    )
    new_text, n_subs = pattern.subn(kpoints_block_qe, text, count=1)

    if n_subs != 1:
        raise RuntimeError(
            "Could not find a single K_POINTS block to replace in nscf.in "
            "-- check the file manually before trusting the output."
        )

    with open(folder / 'nscf.in', 'w') as f:
        f.write(new_text)


    # pw2wannier90 input.
    pw2wan_text = (
        "&inputpp\n"
        "   outdir = './'\n"
        "   prefix = 'WSe2'\n"
        "   seedname = 'WSe2'\n"
        "   write_mmn = .true.\n"
        "   write_amn = .true.\n"
        "   write_unk = .false.\n"
        "/\n"
    )
    with open(folder / 'pw2wan.in', 'w') as f:
        f.write(pw2wan_text)

    # Wannier90: preserve the original fractional projection positions.
    cell_bohr = wse2.get_cell()[:] / Bohr
    frac = wse2.get_scaled_positions(wrap=True)
    symbols = wse2.get_chemical_symbols()

    unit_cell_block = 'begin unit_cell_cart\n  bohr\n'
    for row in cell_bohr:
        unit_cell_block += '  {:>13.7f}  {:>13.7f}  {:>13.7f}\n'.format(*row)
    unit_cell_block += 'end unit_cell_cart\n'

    atoms_frac_block = 'begin atoms_frac\n'
    for sym, (fx, fy, fz) in zip(symbols, frac):
        atoms_frac_block += '{:<4s}  {:>18.10f}  {:>18.10f}  {:>18.10f}\n'.format(sym, fx, fy, fz)
    atoms_frac_block += 'end atoms_frac\n'

    proj_fracs = build_projection_fracs(wse2, shift_i, shift_j, nk)
    projections_block = 'begin projections\n'
    for fx, fy, fz in proj_fracs:
        projections_block += '  f={:.3f},{:.3f}, {:.10f} :s:zona=4.0\n'.format(fx, fy, fz)
    projections_block += 'end projections\n'

    num_wann = 2 * len(proj_fracs)   # spinors=true -> 2 WFs per s-projection

    win_text = f"""!---------------------------GENERAL
num_bands        =  {NBND}
num_wann         =  {num_wann}
!exclude_bands :

!------------------------------CONV
iprint = 3
num_iter         = 0
num_print_cycles = 50
dis_num_iter     = 1000

!-----------------------------bands
bands_plot = .true.
write_hr = .true.

dis_froz_min     = 0.6
dis_froz_max     = 1.2
dis_win_min      = -0.8
dis_win_max      = 1.2
begin kpoint_path
  G  0.0       0.0       0.0     M  0.5       0.0       0.0
  M  0.5       0.0       0.0     K  0.333333  0.333333  0.0
  K  0.333333  0.333333  0.0     G  0.0       0.0       0.0
end kpoint_path

!----------------------------SYSTEM
{unit_cell_block}
{atoms_frac_block}
use_ws_distance = .true.
guiding_centres = .true.

{projections_block}
spinors = true

!----------------------------KPOINTS
mp_grid : 20 20 1

{kpoints_block_w90}"""

    with open(folder / 'WSe2.win', 'w') as f:
        f.write(win_text)



def number(value):
    result = float(value.replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(result):
        raise ValueError('non-finite number')
    return result


def indices(only=None):
    if only is not None:
        if not all(0 <= v < N for v in only):
            raise ValueError(f'indices must be between 0 and {N-1}')
        return [tuple(only)]
    return [(i, j) for i in range(N) for j in range(N)]


def shift_name(i, j):
    return f'shift_i{i:02d}_j{j:02d}'


def check_settings():
    if (GRID / 'pipeline_snapshot.py').read_bytes() != Path(__file__).read_bytes():
        raise ValueError('pipeline.py changed after generation; use a fresh grid')
    if (GRID / 'structure_snapshot.cif').read_bytes() != (ROOT / 'WSe2.cif').read_bytes():
        raise ValueError('WSe2.cif changed after generation; use a fresh grid')


def qe_finished(path, scf=False):
    text = Path(path).read_text(errors='replace')
    text = re.split(r'Program\s+(?:PWSCF|PW2WANNIER)', text)[-1]
    if 'JOB DONE.' not in text or 'Error in routine' in text:
        raise ValueError(f'{Path(path).name}: missing normal QE termination')
    if 'convergence NOT achieved' in text:
        raise ValueError(f'{Path(path).name}: electronic convergence failed')
    if scf and 'convergence has been achieved' not in text:
        raise ValueError(f'{Path(path).name}: missing electronic convergence report')
    return text


def read_relax(folder):
    folder = Path(folder)
    text = qe_finished(folder / 'relax.out', scf=True)
    if 'bfgs converged in' not in text.lower():
        raise ValueError('missing BFGS ionic convergence report')
    if not re.search(r'Begin final coordinates.*?End final coordinates', text, re.S):
        raise ValueError('missing completed final coordinates')
    atoms = read(folder / 'relax.out', format='espresso-out', index=-1)
    atoms.pbc = (True, True, False)
    if atoms.get_chemical_symbols() != ['W', 'Se', 'Se', 'W', 'Se', 'Se']:
        raise ValueError('expected W Se Se W Se Se atom order')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
        raise ValueError('non-finite geometry')
    return atoms


def check_wannier(folder):
    folder = Path(folder)
    read_relax(folder)
    qe_finished(folder / 'scf.out', scf=True)
    qe_finished(folder / 'nscf.out')
    qe_finished(folder / 'pw2wan.out')
    wout = folder / 'WSe2.wout'
    if not check_wannier_done(wout) or (folder / 'WSe2.werr').exists():
        raise ValueError('missing normal Wannier completion or WSe2.werr exists')
    if not re.search(r'Disentanglement\s+convergence criteria satisfied', wout.read_text(errors='replace')):
        raise ValueError('Wannier disentanglement did not report convergence')
    total, count = parse_wout_spread(wout)
    if count != 4 or total is None or not math.isfinite(total) or total/4 >= SPREAD_THRESHOLD:
        raise ValueError('four-Wannier-function spread check failed')
    lattice, projections = parse_win(folder / 'WSe2.win')
    centres = parse_wf_centers(wout)
    if not lattice or len(projections) != 2 or len(centres) != 4:
        raise ValueError('missing cell, projection, or Wannier centre information')
    carts = [frac_to_cart(p, lattice) for p in projections]
    if any(min(min_periodic_distance(p, w, lattice) for p in carts) >= MATCH_TOLERANCE for w in centres):
        raise ValueError('Wannier centres do not match the supplied projections')
    with (folder / 'WSe2_hr.dat').open() as inp:
        next(inp)
        nw, nr = int(next(inp)), int(next(inp))
        if nw != 4 or nr < 1:
            raise ValueError('expected a four-orbital Hamiltonian')
        degeneracies = []
        while len(degeneracies) < nr:
            degeneracies.extend(map(int, next(inp).split()))
        if len(degeneracies) != nr or min(degeneracies) < 1:
            raise ValueError('invalid hopping degeneracies')
        rows = [line.split() for line in inp if line.strip()]
        if len(rows) != nw*nw*nr or any(len(row) != 7 for row in rows):
            raise ValueError('truncated or malformed Hamiltonian')
        vectors = set()
        for start in range(0, len(rows), nw*nw):
            block = rows[start:start + nw*nw]
            vector = tuple(map(int, block[0][:3]))
            if vector in vectors:
                raise ValueError('duplicate Hamiltonian lattice vector')
            vectors.add(vector)
            pairs = set()
            for row in block:
                if tuple(map(int, row[:3])) != vector:
                    raise ValueError('inconsistent Hamiltonian lattice-vector block')
                pairs.add(tuple(map(int, row[3:5])))
                number(row[5]); number(row[6])
            if pairs != set(itertools.product(range(1, nw+1), repeat=2)):
                raise ValueError('incomplete Hamiltonian matrix')
    for name in ('WSe2_wsvec.dat', 'WSe2_band.dat'):
        if not (folder / name).stat().st_size:
            raise ValueError(f'{name} is empty')

def check_wannier_done(path):
    if not Path(path).is_file():
        return False
    with open(path, errors="ignore") as f:
        content = f.read()
    return content.rstrip().endswith("All done: wannier90 exiting")

def parse_wout_spread(path):
    if not Path(path).is_file():
        return None, None
    with open(path, errors="ignore") as f:
        content = f.read()
    final_blocks = content.split("Final State")
    if len(final_blocks) < 2:
        return None, None
    final_block = final_blocks[-1]
    wf_lines = re.findall(
        r"WF centre and spread\s+\d+\s+\(([^)]+)\)\s+([\-0-9.]+)", final_block
    )
    num_wf = len(wf_lines)
    m = re.search(r"Omega Total\s*=\s*([\-0-9.]+)", final_block)
    if not m:
        return None, num_wf
    omega_total = float(m.group(1))
    return omega_total, num_wf

def parse_wf_centers(path):
    if not Path(path).is_file():
        return []
    with open(path, errors="ignore") as f:
        content = f.read()
    final_blocks = content.split("Final State")
    if len(final_blocks) < 2:
        return []
    final_block = final_blocks[-1]
    matches = re.findall(
        r"WF centre and spread\s+\d+\s+\(\s*([\-0-9.]+),\s*([\-0-9.]+),\s*([\-0-9.]+)\s*\)\s+([\-0-9.]+)",
        final_block,
    )
    return [(float(x), float(y), float(z), float(s)) for x, y, z, s in matches]

def parse_win(path):
    if not Path(path).is_file():
        return None, []
    with open(path, errors="ignore") as f:
        content = f.read()
    cell_match = re.search(
        r"begin unit_cell_cart\s*\n\s*(\S+)\s*\n(.*?)end unit_cell_cart",
        content, re.IGNORECASE | re.DOTALL
    )
    if not cell_match:
        return None, []
    unit = cell_match.group(1).strip().lower()
    vec_lines = cell_match.group(2).strip().splitlines()
    lattice = []
    for line in vec_lines[:3]:
        nums = [float(x) for x in line.split()]
        lattice.append(nums)
    if unit.startswith("bohr"):
        lattice = [[c * BOHR_TO_ANG for c in row] for row in lattice]
    proj_match = re.search(
        r"begin projections\s*\n(.*?)end projections",
        content, re.IGNORECASE | re.DOTALL
    )
    projections = []
    if proj_match:
        for line in proj_match.group(1).strip().splitlines():
            m = re.search(r"f\s*=\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)", line)
            if m:
                projections.append(tuple(float(x) for x in m.groups()))
    return lattice, projections

def frac_to_cart(frac, lattice):
    fx, fy, fz = frac
    a1, a2, a3 = lattice
    x = fx * a1[0] + fy * a2[0] + fz * a3[0]
    y = fx * a1[1] + fy * a2[1] + fz * a3[1]
    z = fx * a1[2] + fy * a2[2] + fz * a3[2]
    return (x, y, z)

def min_periodic_distance(cart_point, wf_center, lattice, image_range=1):
    best = None
    px, py, pz = cart_point
    wx, wy, wz = wf_center[0], wf_center[1], wf_center[2]
    a1, a2, a3 = lattice
    for n1, n2, n3 in itertools.product(range(-image_range, image_range + 1), repeat=3):
        tx = px + n1 * a1[0] + n2 * a2[0] + n3 * a3[0]
        ty = py + n1 * a1[1] + n2 * a2[1] + n3 * a3[1]
        tz = pz + n1 * a1[2] + n2 * a2[2] + n3 * a3[2]
        d = ((tx - wx) ** 2 + (ty - wy) ** 2 + (tz - wz) ** 2) ** 0.5
        if best is None or d < best:
            best = d
    return best

def generate():
    for name in ('WSe2.cif', 'W.upf', 'Se.upf'):
        if not (ROOT / name).is_file():
            raise ValueError(f'Missing {name}')
    if GRID.exists():
        raise ValueError(f'{GRID} already exists; use a fresh folder')
    with tempfile.TemporaryDirectory(prefix='wse2-inputs-', dir=ROOT) as staging:
        root = Path(staging)
        shutil.copyfile(__file__, root / 'pipeline_snapshot.py')
        shutil.copyfile(ROOT / 'WSe2.cif', root / 'structure_snapshot.cif')
        for i, j in indices():
            folder = root / shift_name(i, j)
            folder.mkdir()
            atoms = build_bilayer(ROOT / 'WSe2.cif', shift_i=i, shift_j=j, nk=N)
            write_relax_in(atoms, outfile=folder / 'relax.in')
            (folder / 'shift.dat').write_text(f'{i} {j} {N}\n')
            for name in ('W.upf', 'Se.upf'):
                (folder / name).symlink_to('../../' + name)
        root.rename(GRID)
    print(f'Created {N*N} relaxation inputs in {GRID}')


def prepare(only=None):
    check_settings()
    pending, failures = [], []
    with tempfile.TemporaryDirectory(prefix='wse2-prepare-') as staging:
        for i, j in indices(only):
            folder = GRID / shift_name(i, j)
            try:
                if (folder / 'shift.dat').read_text().split() != list(map(str, (i, j, N))):
                    raise ValueError('shift.dat does not match its grid directory')
                atoms = read_relax(folder)
                target = Path(staging) / folder.name
                target.mkdir()
                write_downstream(target, atoms, i, j, N)
                files = {name: (target / name).read_bytes() for name in ('scf.in','nscf.in','pw2wan.in','WSe2.win')}
                for name, data in files.items():
                    if (folder / name).exists() and (folder / name).read_bytes() != data:
                        raise ValueError(f'{name} exists with different contents')
                pending.append((folder, files))
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(f'{folder.name}: {exc}')
        if failures:
            raise ValueError('No inputs written:\n' + '\n'.join(failures))
        for folder, files in pending:
            for name, data in files.items():
                if not (folder / name).exists():
                    with (folder / name).open('xb') as out:
                        out.write(data)
    print(f'Prepared {len(pending)} stackings')


def check_grid(stage, only=None):
    check_settings()
    failed = []
    for i, j in indices(only):
        folder = GRID / shift_name(i,j)
        try:
            (read_relax if stage == 'relax' else check_wannier)(folder)
            print(f'{folder.name}: OK')
        except (OSError, ValueError, RuntimeError, StopIteration) as exc:
            failed.append(folder.name)
            print(f'{folder.name}: FAILED: {exc}')
    print(f'Complete: {len(indices(only))-len(failed)}/{len(indices(only))}')
    return int(bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('generate', help='generate 144 relaxation inputs')
    for command in ('prepare','check-relax','check-wannier'):
        sub = subs.add_parser(command)
        sub.add_argument('--only', type=int, nargs=2, metavar=('I','J'))
    args = parser.parse_args()
    if args.command == 'generate':
        generate()
    elif args.command == 'prepare':
        prepare(args.only)
    else:
        return check_grid(args.command.split('-')[1],args.only)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        raise SystemExit(str(exc))
