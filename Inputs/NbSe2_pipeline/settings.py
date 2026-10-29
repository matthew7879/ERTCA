"""Edit these settings BEFORE generating a fresh grid (Python standard library only)."""

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

# Nibi settings carried over from your scripts. Modules are in cluster_env.sh.
ACCOUNT = 'rrg-vcrepel'
NTASKS = 20
MEM_PER_CPU = '4G'
WALLTIME = '24:00:00'
MAX_CONCURRENT = 20
