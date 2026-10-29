# ERTCA

Minimal figure-reproduction code for **Moire metals and continuum models beyond a single valley**, Matthew Albert, Andrew Klekovkin-Hardy, and Valentin Crepel (manuscript dated October 29, 2026).

## Setup and run

Use Julia 1.11 or newer (validated with Julia 1.12.6). From this directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --startup-file=no -t auto Fig1.jl
```

Replace `Fig1.jl` with any script in the table below. Each runs independently from the supplied inputs, including `Fig4.jl`; no previously generated figures or serialized calculations are required. To run all ten in separate Julia processes:

```sh
for n in {1..10}; do
  julia --project=. --startup-file=no -t auto "Fig${n}.jl" || break
done
```

Outputs are `Figures/Fig1.pdf` through `Figures/Fig10.pdf`, with matching PNG previews. Some scripts also write numerical data and parameter reports. `Figures/` and `.cache/` are generated locally and ignored by Git. Full band, DOS, and spectral calculations can take appreciably longer than the simple supplementary plots. The numerical meshes and physical parameters match the source scripts; they are not reduced for speed.

Figures 1 and 2 also require `pdflatex` with `standalone` and `pgfplots`, and Poppler's `pdftoppm`. Figure 4 requires `pdfcrop` (and Ghostscript) and `pdftoppm`. On macOS, MacTeX supplies the TeX tools and `brew install poppler` supplies Poppler; ensure the commands are on PATH. The scripts also recognize the standard MacTeX location `/Library/TeX/texbin`.

Direct package versions are pinned in `Project.toml`. Setup generates an ignored local `Manifest.toml`; Julia manages its own package integrity checks. Calculation caches compare the input and source bytes directly.

## Figure numbering

The supplied PDF has **eight main figures and two supplementary figures**, S1 and S2. To provide the requested consecutive filenames, the latter are named `Fig9.jl` and `Fig10.jl` here.

| Script | Figure in the PDF | Content |
| --- | --- | --- |
| `Fig1.jl` | 1 | NbSe2 orbital, monolayer Fermi surface, and stacking-dependent DFT/Wannier bands |
| `Fig2.jl` | 2 | NbSe2 interlayer harmonics and shell convergence |
| `Fig3.jl` | 3 | WSe2 moire bands versus DFT and the published continuum model |
| `Fig4.jl` | 4 | WSe2 highest and Gamma-derived states unfolded into the monolayer BZ |
| `Fig5.jl` | 5 | NbSe2 bands and DOS at 5.09, 6.01, and 7.93 degrees |
| `Fig6.jl` | 6 | NbSe2 DOS benchmark at 9.43 degrees |
| `Fig7.jl` | 7 | NbSe2 spectral weight at the upper van Hove singularity |
| `Fig8.jl` | 8 | NbSe2 Gamma-valley band comparison |
| `Fig9.jl` | S1 | Nb-Nb separation versus registry |
| `Fig10.jl` | S2 | NbSe2 harmonic decay at Gamma and K-prime |

The original numerical algorithms, input values, plotting parameters, and energy references are retained. Packaging changes relocate paths, split the original combined entrypoints, calculate Figure 4's input on demand, and remove unrelated figures and optional comparison plots. Figure 1's orbital illustration is an input VESTA export, not generated from scratch by Julia.

## Inputs

To regenerate the registry data with Quantum ESPRESSO and Wannier90, see [Inputs/NbSe2_DFT_Wannier](Inputs/NbSe2_DFT_Wannier/README.md) and [Inputs/WSe2_DFT_Wannier](Inputs/WSe2_DFT_Wannier/README.md). Each includes the original pseudopotentials, scientific input generation, output checks, and HDF5 export. These optional workflows are separate from running the figures.

- `Inputs/NbSe2_grid_12x12_tight.h5` and `Inputs/WSe2_grid_12x12.h5`: the two 144-registry Wannier datasets. They expand on demand into `.cache/`; no unpacked 106 MB data tree is committed.
- `Inputs/NbSe2_monolayer*`: Wannier Hamiltonian, centers, checkpoint, spin matrix, and k-point input used by Figure 1.
- `Inputs/NbSe2_bands_DFT_*.dat`: the five stacking-resolved band references used by Figure 1.
- `Inputs/NbSe2_Wannier.pdf`: the VESTA orbital illustration used by Figure 1.
- `Inputs/NbSe2_9p43deg_DFT_DOS_gauss15meV.dat`: large-cell DOS reference for Figure 6.
- `Inputs/NbSe2_interlayer_distance.dat`: the 144 atomic Nb-Nb separations for Figure 9/S1.
- `Inputs/WSe2_DFT_reference/`: the k-path and band data from T. Devakul, V. Crepel, Y. Zhang and L. Fu, *Magic in twisted transition metal dichalcogenide bilayers*, Nature Communications 12, 6730 (2021), [arXiv:2106.11954](https://arxiv.org/abs/2106.11954). The CSVs contain only the seven plotted Kramers pairs (bands 6591–6604), retaining the original vacuum-referenced energies and reversed, deduplicated Gamma–K–M–Gamma path. The calculation has 762 atoms, NELECT=6604, and spinor bands; the plotted bands are selected as Kramers pairs.

Licensed under the MIT license; see `LICENSE`.

## Source layout

`src/` contains nine files: the module entrypoint, numerical utilities, geometry, Hamiltonian construction, Wannier input readers, ERTCA construction, band-path sampling, shared figure setup, and valley comparisons. 

## Validation

Checked on October 29, 2026 with Julia 1.12.6. All ten scripts completed after cleanup. Figures 1, 2, 9 and 10 ran at full settings; expensive checks used temporary copies with 12-point band paths (Figures 3/8), eight initial tracking steps with adaptive refinement (Figure 4), or 3-by-3 moire and 30-by-30 reference meshes (Figures 5–7). The committed scripts retain all original paper settings.

Full NbSe2 and WSe2 Hamiltonians at three momenta each were entry-for-entry identical before and after cleanup. Figure 4 reproduced its previous PDF drawing content, as did Figures 1, 2, 9 and 10 and the same-mesh checks for Figures 5–7. Module exports were verified.

## Push updates

```sh
git push origin main
```
