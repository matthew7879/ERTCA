# WSe2 DFT and Wannier inputs

`pipeline.py` generates the 12×12 registry grid and the relaxation, SCF, NSCF and
Wannier90 input files. Run the calculations separately using your own Quantum
ESPRESSO and Wannier90 installations. The script generates inputs and checks
outputs; it does not launch calculations.

## Files and Python setup

- `pipeline.py`: input generation and output checks.
- `WSe2.cif`: the supplied starting monolayer structure.
- `W.upf`, `Se.upf`: the exact pseudopotentials used in the calculations.
- `pack_grid.jl`: optional export of completed Hamiltonians to HDF5.
- `README.md` and `.gitignore`: instructions and generated-file exclusions.

Use Python 3.9+ and ASE:

```bash
python3 -m pip install ase==3.26.0
```

The original ASE geometry construction and relaxed-output reader are retained.
In particular, downstream cell vectors use ASE's reading of the printed QE
lattice, including its original rounding. The pseudopotential directory is now
local (`.`); the scientific input settings are unchanged.

## Generate inputs

From this folder, with no existing `grid12/`:

```bash
python3 pipeline.py generate
```

This creates all 144 `grid12/shift_iII_jJJ/` directories, with `relax.in`,
`shift.dat` and links to the two pseudopotentials. The top layer is translated
by `(i/12)*a1 + (j/12)*a2`. Run each relaxation from its shift directory and save
the output as `relax.out`.

Once the relaxations finish:

```bash
python3 pipeline.py check-relax
python3 pipeline.py prepare
```

`prepare` reads the final relaxed structures and writes `scf.in`, `nscf.in`,
`pw2wan.in` and `WSe2.win`. Use `--only 0 0` with `check-relax` and `prepare`
to test one stacking first. Incomplete relaxations are rejected. Existing
different input files are not overwritten. Use a fresh grid after changing
the generator's scientific settings.

## Calculation order

Run each stage from its shift directory and wait for successful completion
before continuing:

| Stage | Input / operation | Output to retain |
| --- | --- | --- |
| Relaxation | `pw.x` with `relax.in` | `relax.out` |
| SCF | `pw.x` with `scf.in` | `scf.out` and QE saved data |
| NSCF | `pw.x` with `nscf.in` | `nscf.out` and QE saved data |
| Wannier preprocessing | `wannier90.x -pp WSe2` | `WSe2.nnkp` |
| QE-to-Wannier conversion | `pw2wannier90.x` with `pw2wan.in` | `pw2wan.out`, `.amn`, `.mmn`, `.eig` |
| Wannier Hamiltonian | `wannier90.x WSe2` | `WSe2.wout`, `WSe2_hr.dat`, `WSe2_wsvec.dat`, `WSe2_band.dat` |

Keep the QE saved data through conversion. After the calculations finish:

```bash
python3 pipeline.py check-wannier
```

Use `--only 0 0` to check one stacking. Retain the original `.wout` logs, `.win`
inputs and Hamiltonian outputs. The script does not delete them.

## Scientific settings

| Parameter | Value |
| --- | --- |
| Starting geometry | two translated copies of the supplied monolayer CIF |
| Target in-plane lattice constant | 3.295003576 Å |
| Initial layer offset / vacuum added on each side | 6.5 / 15 Å |
| Relaxation | fixed cell, atomic z coordinates only |
| Functional / SOC / dispersion | PBE / enabled / Grimme D3 |
| Wavefunction / density cutoffs | 80 / 800 Ry |
| Ionic energy / force thresholds | 1e-4 Ry / 1e-3 Ry/Bohr |
| Electronic threshold | 1e-8 Ry |
| k meshes, relax / SCF / NSCF | 20×20×1 / 25×25×1 / explicit 20×20×1 |
| Smearing | Gaussian, 0.02 Ry |
| NSCF bands / Wannier functions | 110 / 4 |
| Disentanglement outer window | −0.8 to 1.2 eV |
| Frozen window | 0.6 to 1.2 eV |
| Iterations | dis_num_iter=1000, num_iter=0 |
| Projections | original two fractional s projections; guiding_centres=true |

SCF keeps the original default band count; NSCF explicitly requests 110 bands.
All settings above are retained from the supplied WSe2 scripts.

## Optional HDF5 export

After all 144 calculations pass, from this folder:

```bash
julia --project=../.. -e 'using Pkg; Pkg.instantiate()'
julia --project=../.. --startup-file=no pack_grid.jl grid12 WSe2_grid_12x12_regenerated.h5
```

The packer verifies exact reconstruction of the printed Hamiltonians and refuses
to overwrite an existing archive. It stores the lattice, Wannier centres/spreads
and hopping matrices. Compare the result with `../WSe2_grid_12x12.h5` before
replacing any figure input. Keep original logs separately.

## Pseudopotentials

The supplied files match PseudoDojo's fully relativistic PBE norm-conserving
ONCVPSP v0.4 data byte for byte:
[W-sp_r.upf](https://github.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4/blob/master/W/W-sp_r.upf)
(14 valence electrons) and
[Se-d_r.upf](https://github.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4/blob/master/Se/Se-d_r.upf)
(16 valence electrons), renamed `W.upf` and `Se.upf`. The bilayer has 92 valence
electrons. Keep these files for reproduction; substitutes can change band
numbering and results.

Credit PseudoDojo and D. R. Hamann,
[Phys. Rev. B 88, 085117 (2013)](https://doi.org/10.1103/PhysRevB.88.085117).
These data retain their notices and PseudoDojo's
[CC BY 4.0 terms](https://github.com/PseudoDojo/pseudodojo#license), separate from
ERTCA's MIT license.
