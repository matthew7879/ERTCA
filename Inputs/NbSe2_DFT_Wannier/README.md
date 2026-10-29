# NbSe2 12×12 inputs

`pipeline.py` generates input files for relaxation, SCF, NSCF and Wannier90, and
checks the outputs after you run the calculations. The scientific settings,
pseudopotentials and generated calculation inputs are unchanged.

## Files

- `pipeline.py`: scientific settings, input generation, output checks and distance extraction.
- `Nb.upf`, `Se.upf`: the exact pseudopotentials used in the calculations.
- `pack_grid.jl`: optional export of completed Hamiltonians to ERTCA's HDF5 format.
- `README.md` and `.gitignore`: instructions and generated-file exclusions.

## Generate inputs

Python 3.8+ with its standard library is sufficient to generate the inputs.
Run the calculations separately using your own Quantum ESPRESSO and Wannier90
installations. Edit the scientific settings at the top of `pipeline.py` before
generating a fresh grid.

From this folder:

```bash
python3 pipeline.py generate
```

This creates `grid12/shift_i00_j00/` through `grid12/shift_i11_j11/`, each containing
`relax.in` and links to the two pseudopotentials. Run each relaxation from its
shift directory and save its output as `relax.out`.

After the relaxations finish:

```bash
python3 pipeline.py check-relax
python3 pipeline.py prepare
```

`prepare` reads the converged atomic positions from each `relax.out` and writes
`scf.in`, `nscf.in`, `pw2wan.in` and `NbSe2.win` in that shift directory.
For a single-stacking test, use `--only 0 0` with `check-relax` and `prepare`.
Unfinished or unconverged relaxations are rejected; existing different inputs
are not overwritten.

## Run the generated inputs

Run the following stages in order within each shift directory, waiting for each
to finish successfully before proceeding:

| Stage | Input / operation | Output to retain |
| --- | --- | --- |
| Relaxation | Quantum ESPRESSO `pw.x` with `relax.in` | `relax.out` |
| SCF | `pw.x` with `scf.in` | `scf.out` and QE saved data |
| NSCF | `pw.x` with `nscf.in` | `nscf.out` and QE saved data |
| Wannier preprocessing | `wannier90.x -pp NbSe2` | `NbSe2.nnkp` |
| QE-to-Wannier conversion | `pw2wannier90.x` with `pw2wan.in` | `pw2wan.out`, `.amn`, `.mmn`, `.eig` |
| Wannier Hamiltonian | `wannier90.x NbSe2` | `NbSe2.wout`, `NbSe2_hr.dat`, `NbSe2_wsvec.dat`, `NbSe2_band.dat` |

Keep the QE saved data through conversion and retain the original Wannier logs
and Hamiltonians. The Python script does not launch calculations or delete output.
After all calculations finish:

```bash
python3 pipeline.py check-wannier
```

Use `--only 0 0` to check one stacking. Checks use the calculation logs and outputs;
no completion marker is required. Settings changes require a fresh grid.

## Calculation settings

Fixed six-atom cell; z-only relaxation; 12×12 registry grid; PBE with SOC and
Grimme D3; cutoffs 80/800 Ry; ionic thresholds 1e-5 Ry and 1e-4 Ry/Bohr; electronic
thresholds 1e-10 Ry (relax) and 1e-8 Ry (SCF/NSCF); Gaussian smearing 0.02 Ry;
k meshes 20×20×1 / 25×25×1 / 20×20×1; 108 spinor bands; Wannier bands 89–92;
four Wannier functions with the original projections and `num_iter=0`.

## Optional figure-input export

After all 144 calculations pass, from this folder:

```bash
python3 pipeline.py distances --output NbSe2_interlayer_distance_regenerated.dat
julia --project=../.. -e 'using Pkg; Pkg.instantiate()'
julia --project=../.. --startup-file=no pack_grid.jl grid12 NbSe2_grid_12x12_regenerated.h5
```

The packer preserves the printed Hamiltonians exactly and refuses to overwrite an
existing archive. It stores lattice vectors, Wannier centres/spreads and hoppings;
retain original logs for atomic positions and full calculation details. Compare
regenerated outputs with the existing inputs in `../` before replacing them.
Running the figures does not require rerunning these calculations.

## Pseudopotentials

The included files are unchanged copies of PseudoDojo's fully relativistic PBE
norm-conserving ONCVPSP v0.4 data:
[Nb-sp_r.upf](https://github.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4/blob/master/Nb/Nb-sp_r.upf)
(13 valence electrons) and
[Se-d_r.upf](https://github.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4/blob/master/Se/Se-d_r.upf)
(16 valence electrons), renamed `Nb.upf` and `Se.upf`. Together these give 90
valence electrons per bilayer cell. Keep these files for reproduction; substitutes
may change the results and band numbering.

Credit PseudoDojo and D. R. Hamann,
[Phys. Rev. B 88, 085117 (2013)](https://doi.org/10.1103/PhysRevB.88.085117).
These data retain their notices and PseudoDojo's
[CC BY 4.0 terms](https://github.com/PseudoDojo/pseudodojo#license), separate from
ERTCA's MIT license.
