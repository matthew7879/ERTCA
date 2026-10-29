#############################################################
###### ERTCA: continuum models from Wannier90 registry scans
#############################################################

### The environment-resolved two-center approximation (ERTCA) pipeline: a family of
### Wannier90 calculations of an UNTWISTED bilayer at rigid registry shifts d is turned into
### the k-dependent registry harmonics W_{ββ'}(k̄; Ḡ), which parametrize the twisted-bilayer
### continuum model at ANY angle through the master formula
###
###     T^{ββ'}_{k,k'} = Σ_{g ∈ Λ*_m} W_{ββ'}(k*; Ḡ_g) δ_{k'−k, g},
###     k* = L_{ℓℓ'}^{-1} (k + k')/2,
###
### with L_{ℓℓ'} the mean orientation of the two layers involved (the layer's own rotation
### intralayer, the identity interlayer). The three stages, in order:
###
###     registry_scan       ingest + cross-validate the d-shifted family     (registry_scan.jl)
###     registry_harmonics  solve the e^{iḠ·d} system for the harmonics     (harmonics.jl)
###     ertca_model         realize the model on a plane-wave basis          (model.jl)
###
### The construction and conventions follow Albert, Klekovkin-Hardy & Crépel, "First
### principles continuum modeling of generic moiré materials": the package's atomic gauge
### with true Wannier centres IS the paper's Fourier convention (phase on the full orbital
### separation), and the k_envelope midpoint momentum IS the paper's k*.

include("registry_scan.jl")
include("harmonics.jl")
include("model.jl")
include("diagnostics.jl")
include("plotting_api.jl")
