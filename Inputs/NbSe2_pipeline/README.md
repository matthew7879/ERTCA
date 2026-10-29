# NbSe2 12×12 pipeline

`pipeline.py` generates the grid and runs relaxation → SCF → NSCF → Wannier90.
The calculation settings and generated QE/Wannier inputs are unchanged from the
supplied cluster pipeline. The Python helpers and job runners are consolidated
into this one file; Slurm scripts are generated inside `grid12/`.

## Files

- `pipeline.py`: settings, input generation, submission, execution and validation.
- `Nb.upf`, `Se.upf`: the exact pseudopotentials used in the calculations.
- `pack_grid.jl`: optional export of completed Hamiltonians to ERTCA's HDF5 format.
- `README.md` and `.gitignore`: instructions and generated-file exclusions.

## Run

Requires Python 3.8+ (standard library only), Bash, Slurm, MPI, Quantum ESPRESSO
7.3.1 and Wannier90 3.1.0. Review the account, resources and module settings at the
top of `pipeline.py` for your cluster **before** generating a fresh grid. Defaults
retain the Nibi setup: account `rrg-vcrepel`, 20 MPI tasks, 4 GB per task, 24 hours
and up to 20 concurrent jobs. Use an account you have access to.

From this folder on the cluster, first test one stacking:

```bash
python3 pipeline.py generate
python3 pipeline.py submit-relax --only 0 0
```

Wait for the relaxation job to finish, then:

```bash
python3 pipeline.py check-relax --only 0 0
python3 pipeline.py submit-wannier --only 0 0
```

Wait for its SCF/NSCF/Wannier job to finish, then:

```bash
python3 pipeline.py check-wannier --only 0 0
python3 pipeline.py submit-relax
```

Wait for all relaxation jobs to finish, then:

```bash
python3 pipeline.py check-relax
python3 pipeline.py submit-wannier
```

After all Wannier jobs finish:

```bash
python3 pipeline.py check-wannier
```

`generate` does not submit jobs. `submit-wannier` validates the selected relaxations
and prepares their SCF/NSCF/Wannier inputs before submitting. Use
`python3 pipeline.py prepare` to prepare inputs without submission. Completed,
validated jobs are skipped. Existing outputs are not silently overwritten.

Generated inputs, scratch files, job scripts and calculation outputs are inside
`grid12/`; array scheduler logs are written in the pipeline folder. Keep the
original `.wout`, `_hr.dat`, `_wsvec.dat` and `.win` files.
The runner stops on failure. Inspect and archive failed outputs before retrying;
remove a stale `.pipeline_lock` only after confirming its job has ended.
Settings changes require a fresh pipeline copy. QE scratch data is retained unless
`CLEANUP_QE=1` is exported before submission; cleanup runs only after validation.
Completion records compare compressed copies of exact input bytes.

## Calculation settings

Fixed six-atom cell; z-only relaxation; 12×12 registry grid; PBE with SOC and
Grimme D3; cutoffs 80/800 Ry; ionic thresholds 1e-5 Ry and 1e-4 Ry/Bohr; electronic
thresholds 1e-10 Ry (relax) and 1e-8 Ry (SCF/NSCF); Gaussian smearing 0.02 Ry;
k meshes 20×20×1 / 25×25×1 / 20×20×1; 108 spinor bands; Wannier bands 89–92;
four Wannier functions with the original projections and `num_iter=0`.

## Optional figure-input export

After all 144 jobs pass, from this folder:

```bash
python3 pipeline.py distances --output NbSe2_interlayer_distance_regenerated.dat
julia --project=../.. -e 'using Pkg; Pkg.instantiate()'
julia --project=../.. --startup-file=no pack_grid.jl grid12 NbSe2_grid_12x12_regenerated.h5
```

The packer preserves the printed Hamiltonians exactly and refuses to overwrite an
existing archive. It stores lattice vectors, Wannier centres/spreads and hoppings;
retain original logs for atomic positions and full calculation details. Compare
regenerated outputs with the existing inputs in `../` before replacing them.
Running the figures does not require rerunning this DFT workflow.

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
