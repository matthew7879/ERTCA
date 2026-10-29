##################################################################
###### export only the following functions from this module ######
##################################################################

export plot_harmonic_decay

### The `plot_harmonic_decay` NAME, declared here; the Plots METHOD lives in
### `ext/SingleParticleModelsPlotsExt.jl` and appears only once Plots is in the session.
###
### Plots is a package extension rather than a dependency: 127 packages and 79 JLLs of the
### resolved graph -- Qt6, GR, FFmpeg, the X11 stack -- were reachable ONLY through it, and
### `using Plots` was about two thirds of the load time (measured warm, five runs each: 1.90 s
### as a dependency, 0.60 s as an extension). The seam costs nothing
### structurally, because it is the compute/plot split this package already had: nothing in
### `src/` outside the plotting layer mentions `Plots.`, and nothing in `src/` calls a `plot_*`
### function.
###
### The DOCSTRINGS live here rather than beside the methods so that `checkdocs = :exports` is
### satisfied without Plots loaded, and so the reference pages keep rendering each function on
### the page it belongs to -- the `@autodocs` blocks filter by source file, and all six methods
### now share one extension file.
###
### Each name gets a catch-all method that refuses by name. A catch-all `f(args...; kwargs...)`
### is less specific than every method the extension defines, so loading Plots shadows it for
### real calls while an unloaded session gets a sentence telling the user what to type, rather
### than a bare `MethodError` listing zero methods.

"""
    plot_harmonic_decay(h::RegistryHarmonics; kwargs...) -> Plots.Plot

The registry-harmonic decay on a semilog axis: `max |W|` (the exact Bloch-sum bound of
[`harmonic_decay`](@ref)) against the harmonic length `|Ḡ|`, intralayer and interlayer
channels as separate series. The manuscript's decay-law figure -- and the pipeline's
gauge-health dashboard: a family of Wannierizations that does not share one smooth gauge
shows up as a decay that stalls. `kpoints` is forwarded to [`harmonic_decay`](@ref) (the
near-zone momenta the decay is evaluated at); every other keyword argument goes to
`Plots.plot!`.
"""
function plot_harmonic_decay(args...; kwargs...)
    throw(ArgumentError(
        "`plot_harmonic_decay`: this package ships its plotting as a package extension, so Plots.jl is " *
        "NOT installed alongside it and no method is defined yet. Add Plots to your own " *
        "project and run `using Plots` -- the extension loads itself the moment Plots is in " *
        "the session, and this call will then work unchanged. The compute half needs nothing: " *
        "`harmonic_decay` returns its result object with or without Plots."))
end
