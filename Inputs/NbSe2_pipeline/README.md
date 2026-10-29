# NbSe2 12×12 input-generation pipeline

Generate the 144 rigid in-plane registry shifts, relax atomic z coordinates, run
SCF → NSCF → Wannier90, and export the inputs used by ERTCA. This is optional:
`Fig1.jl`–`Fig10.jl` already use the supplied data and do not launch DFT jobs.

## Requirements and settings

Python 3.8+ (standard library only), Bash, Slurm, MPI, Quantum ESPRESSO 7.3.1
(`pw.x`, `pw2wannier90.x`) and Wannier90 3.1.0. Julia with ERTCA's project
is needed only to pack the final HDF5 archive. Cluster software is not installed
by these scripts.

Edit the Slurm account/resources in `settings.py` and module commands in
`cluster_env.sh` for your cluster **before generating the grid**. Defaults retain
the supplied Nibi configuration: 20 MPI tasks, 4 GB per task, 24 hours, and at
most 20 concurrent array tasks. The account `rrg-vcrepel` must be replaced if
you do not have access to it. Both array submission files and per-shift job
scripts are generated automatically; they are not stored in this repository.

The calculation settings are retained from the supplied cluster pipeline:

| Parameter | Value |
| --- | --- |
| Grid | (i/12, j/12), i,j = 0,…,11 |
| Structure | six-atom bilayer, fixed cell and x/y, z relaxation only |
| Functional / SOC / dispersion | PBE / enabled / Grimme D3 |
| Wavefunction / density cutoffs | 80 / 800 Ry, all stages |
| Ionic energy / force thresholds | 1e-5 Ry / 1e-4 Ry/Bohr |
| Electronic threshold, relax / SCF / NSCF | 1e-10 / 1e-8 / 1e-8 Ry |
| k meshes, relax / SCF / NSCF | 20×20×1 / 25×25×1 / 20×20×1 |
| Smearing | Gaussian, 0.02 Ry |
| Electrons / SCF and NSCF spinor bands | 90 / 108 |
| Wannier subspace | bands 89–92, four spinor Wannier functions |
| Projection gauge | supplied vacuum-site s projections, num_iter=0 |

Keep the supplied pseudopotentials for reproduction. The numerical setup,
including the three-decimal projection x/y coordinates, is unchanged. These
settings reproduce the supplied inputs; they are not a new convergence study.

## Run on the cluster

Use a fresh checkout with no `grid12/`. From `Inputs/NbSe2_pipeline`:

```bash
python3 test_pipeline.py
python3 generate_grid_12x12.py
(cd grid12/shift_i00_j00 && sbatch batch_relax.sh)
```

Wait for the AA relaxation to finish, then test its Wannier chain:

```bash
python3 check_relaxations.py --stacking grid12/shift_i00_j00
python3 generate_wannier_12x12.py --only 0 0
(cd grid12/shift_i00_j00 && sbatch batch_wannier.sh)
```

After that job finishes:

```bash
python3 check_wannier.py --stacking grid12/shift_i00_j00
bash submit_all.sh
```

Wait for all relaxation jobs to finish successfully, then:

```bash
python3 check_relaxations.py
bash submit_wannier.sh
```

After all Wannier jobs finish:

```bash
python3 check_wannier.py
python3 extract_nb_interlayer_distance.py grid12 --output NbSe2_interlayer_distance_regenerated.dat
```

Completed, validated AA jobs are skipped when the full arrays run. Submit array
scripts from the pipeline folder and individual jobs from their shift folder;
the commands above use the required working directories. Python generators only
write files; `sbatch` and the `submit_*.sh` wrappers submit jobs.

## Export the ERTCA archive

Retain the original `NbSe2.wout`, `NbSe2_hr.dat`, `NbSe2_wsvec.dat`, and `NbSe2.win`
from every stacking. Once all 144 chains pass, on a machine with Julia and the
ERTCA checkout, run from this folder:

```bash
julia --project=../.. -e 'using Pkg; Pkg.instantiate()'
julia --project=../.. --startup-file=no pack_grid.jl grid12 NbSe2_grid_12x12_regenerated.h5
```

The packer requires all 144 stackings and checks exact reconstruction of each
six-decimal Hamiltonian file before saving a new archive. It refuses to overwrite
an existing archive. It stores the lattice, Wannier centres/spreads, and hopping
matrices needed by the figure loader, not full calculation logs or atomic Site
tables. Extract layer distances from the original `.wout` logs first.

Compare the regenerated files with `../NbSe2_grid_12x12_tight.h5` and
`../NbSe2_interlayer_distance.dat` before deliberately replacing any figure input.
The output filenames above leave the supplied figure inputs untouched.

## Validation and retries

The checker requires normal QE termination, BFGS convergence, final coordinates,
and the **total** atomic z forces within tolerance. The Wannier runner stops after
failed stages and retains scratch files by default. `CLEANUP_QE=1` enables scratch
cleanup only after a complete, validated chain.

`wannier.complete` stores compressed copies of the exact inputs and pseudopotentials;
subsequent checks compare the current bytes directly. Old completion markers from
the cluster version are incompatible. This folder is intended for fresh runs;
never copy its scripts over an active calculation directory.

For a retry, first inspect the failure and confirm the job has ended. Archive
failed output files before resubmitting; the runners refuse to overwrite them.
A killed job can leave `.pipeline_lock`; remove that empty directory only after
confirming no job is using the stacking. Changed numerical settings require a
fresh pipeline copy. Keep successful `relax.out` and the generated `.in`/`.win`
files when retrying only the Wannier stages.

Local checks cover grid generation, rejection of incomplete/high-force relaxations,
separation of total and D3 contribution forces, failure propagation, and changed
inputs/pseudopotentials. The supplied test fixture was updated to include the
force-table header required by the supplied parser. No physical settings changed.

The five generated AA input files match the supplied cluster inputs byte for byte.
The supplied AA logs report QE 7.3.1, 90 electrons, 108 bands, 80/800 Ry cutoffs,
normal completion, and max |Fz| = 6.009e-5 Ry/Bohr. Their Wannier centres/spreads
match the AA entry in ERTCA's supplied archive. The full layer-distance table
matches the existing input file. These checks do not rerun the DFT grid or establish
fresh convergence for all 144 stackings. Sample logs are retained in the uploaded
review ZIP, not duplicated in this minimal folder.

## Pseudopotentials and attribution

`Nb.upf` and `Se.upf` are the exact supplied files. They match the following
[PseudoDojo PBE fully relativistic v0.4](https://github.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4)
files byte for byte:

| Local file | Original file | Valence electrons |
| --- | --- | --- |
| Nb.upf | [Nb/Nb-sp_r.upf](https://github.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4/blob/master/Nb/Nb-sp_r.upf) | 13 |
| Se.upf | [Se/Se-d_r.upf](https://github.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4/blob/master/Se/Se-d_r.upf) | 16 |

They are fully relativistic, norm-conserving ONCVPSP 3.3.0 pseudopotentials with
PBE exchange-correlation and spin-orbit data. The bilayer has 2×13 + 4×16 = 90
valence electrons. Substituting other pseudopotentials may change the electron
count, band numbering and results.

Credit the PseudoDojo project and D. R. Hamann,
[Phys. Rev. B 88, 085117 (2013)](https://doi.org/10.1103/PhysRevB.88.085117).
PseudoDojo distributes its pseudopotential files under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated in its
[license notice](https://github.com/PseudoDojo/pseudodojo#license).
These third-party data files retain their original notices and are not relicensed
under ERTCA's MIT license. The filenames were shortened; their contents are unchanged.
