"""Small, readable input templates shared by the two generators."""

from math import sqrt
import settings as s


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


PW2WAN = """&INPUTPP
 outdir = './',
 prefix = 'NbSe2',
 seedname = 'NbSe2',
 write_mmn = .true.,
 write_amn = .true.,
 write_unk = .false.,
/
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
