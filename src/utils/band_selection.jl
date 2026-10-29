##################################################################
###### export only the following functions from this module ######
##################################################################

export EnergyWindow

### ------------------------------------------------------------------------------------
### Naming a band subspace, and the partial-spectrum bookkeeping that goes with it
### ------------------------------------------------------------------------------------
###
### This file owns the answer to one question -- WHICH BANDS -- in all the forms the package
### accepts it: an index, a range, a vector, or an `EnergyWindow` resolved per k-point against
### eigenvalues. With it comes the bookkeeping a PARTIAL spectrum forces, since `nev`, `which`
### and `sigma` mean the returned values are a slice whose absolute position must be recovered
### (`_partial_window`), whose truncated ends must be known (`_truncated_sides`), and whose
### gap to the complement is only meaningful when the slice is bracketed (`_complement_gap`).
###
### It lives beside `diagonalize.jl` because that is the contract it is written against: the
### ordering of `values`, the meaning of `which`, and the keyword set `_solver_kwargs`
### assembles for `bloch_solver`. It has no other dependency -- nothing here knows what a
### Hamiltonian, a cell or a mesh is.
###
### It was extracted from `analysis/topology/berry_and_chern.jl` (2026-09-16). Every consumer
### of band selection went through a file about Chern numbers: `spectra/diagonalize_over_bz.jl`
### and `spectra/form_factors.jl` reached up into `analysis/` for `EnergyWindow`,
### `_solver_kwargs`, `_window_selection` and `_band_selection`, and `interfaces/qe.jl` for
### `_band_selection`. That was the only reason `spectra` and `analysis/topology` referenced
### each other at all, and it is the largest false edge the dependency graph had.



#############################################################
###### Naming the subspace by energy                    #####
#############################################################

"""
    EnergyWindow(lo, hi)

Name a Berry subspace by ENERGY instead of by band index: every band whose energy lies strictly
inside `(lo, hi)`, resolved independently at each k-point. Pass it wherever `bands_sel` is
expected -- [`berry_curvature`](@ref), [`chern_number`](@ref), [`berry_flux`](@ref).

Why a type and not a tuple: `bands_sel` already accepts an `Int`, a range and a vector of
indices, so `(1, 2)` unavoidably means "bands 1 and 2". The window has to be spelled.

Purpose:
--------
Two things an index cannot do.

- On a large model the absolute index of the manifold you care about is not known in advance --
  a moiré flat-band pair sits wherever the plane-wave truncation and the orbital count put it.
  The energy window is how you say "the flat bands near `E_F`" without counting.
- It is the only selection that composes with SHIFT-INVERT. `nev` bands nearest `sigma` is a
  window whose absolute position cannot be recovered (see `_partial_window`), so index selection
  must refuse `sigma`; an energy window is already stated in the coordinate the solve returns, so
  `nev` + `sigma` is not only allowed but the intended combination:

      chern_number(model, EnergyWindow(-0.02, 0.02); nk=12, nev=24, sigma=0.0, matrix=:sparse)

Contract (checked at every k-point, never assumed):
---------------------------------------------------
- the window must hold the SAME NUMBER of bands everywhere. A band that crosses `lo` or `hi`
  changes the count, and the refusal names the k-point -- that is the honest signal that the
  window edge cuts through a band rather than a gap.
- the window must be BRACKETED by computed states on every truncated side: at least one computed
  band below `lo` and above `hi` when the solve stopped there rather than the spectrum ending.
  Without a bracket the gap to the complement is the truncation, not the physics, and `mingap`
  would read `Inf` straight through `chern_number`'s isolation guard.

Both refusals are cheap and per-point, which is what makes the k-dependent selection safe: the
subspace is verified to be the same rank and gapped everywhere it is used.
"""
struct EnergyWindow
    lo::Float64
    hi::Float64
    function EnergyWindow(lo::Real, hi::Real)
        (isfinite(lo) && isfinite(hi)) || throw(DomainError((lo, hi),
            "`EnergyWindow` needs two finite energies; got ($lo, $hi)."))
        lo < hi || throw(ArgumentError(
            "`EnergyWindow($lo, $hi)`: the lower edge must be strictly below the upper one. " *
            "An empty or inverted window selects no bands."))
        return new(Float64(lo), Float64(hi))
    end
end

EnergyWindow(w::Tuple{Real,Real}) = EnergyWindow(w[1], w[2])

Base.show(io::IO, w::EnergyWindow) = print(io, "EnergyWindow(", w.lo, ", ", w.hi, ")")



### Every entry point that names bands by INDEX funnels through here, so this is where an
### `EnergyWindow` handed to one of them is refused. The window is resolved against eigenvalues,
### per k-point (`_window_selection`), and the entry points that own a spectrum intercept it
### before this call; the ones that do not -- the `FormFactors` topology methods, the Wannier90
### and QE writers -- have no energies to resolve it against, and used to die on
### `MethodError: no method matching length(::EnergyWindow)` from `collect` below, one stack
### frame deep in a private helper, while `EnergyWindow`'s own docstring says it may be passed
### wherever `bands_sel` is expected.
function _band_selection(bands_sel, n::Int)::Vector{Int}
    bands_sel isa EnergyWindow && throw(ArgumentError(
        "The band selection is an `EnergyWindow($(bands_sel.lo), $(bands_sel.hi))`, which " *
        "names a subspace by ENERGY and has to be resolved against eigenvalues at every " *
        "k-point -- and this entry point selects by STORED band index only. An overlap " *
        "tensor (`FormFactors`) carries no energies at all: it is contracted from " *
        "eigenvectors, and its band axis is whatever the spectrum stored " *
        "(`band_selection(F)` reports that provenance). Resolve the window where the " *
        "energies live -- `diagonalize_over_bz(model, mesh; window = EnergyWindow(lo, hi))`, " *
        "then contract that spectrum, whose bands ARE the window -- or use the model method " *
        "(`chern_number(model, EnergyWindow(lo, hi); nk = ...)`), which resolves it per " *
        "k-point and checks the rank there."))
    sel = bands_sel isa Integer ? [Int(bands_sel)] : sort!(unique!(Int.(collect(bands_sel))))
    isempty(sel) && throw(ArgumentError("The band selection is empty."))
    (1 <= sel[1] && sel[end] <= n) || throw(ArgumentError(
        "Band selection $sel is outside 1:$n."))
    return sel
end

"""
    _partial_window(sel, n, nev, which, sigma, method; caller) -> Vector{Int}

Translate an ABSOLUTE band selection into indices of the partial spectrum `diagonalize` will
actually return, refusing every window that cannot carry a parallel-transported subspace.

`caller` names the public entry point in the refusals and has NO default, exactly like
`_solver_kwargs` and `_window_selection`: this validation is shared by `berry_curvature`,
`wilson_loop` and the spectra drivers, and a default would silently attribute one entry
point's refusal to another -- sending the reader to a docstring whose vocabulary (`nk`,
`seam`, ...) the function they actually called does not even have. Forgetting it is an
`UndefKeywordError` at the first call, not a wrong message at the user's.

Why this exists:
----------------
The subspace is named by absolute band index, but a partial solve returns only a WINDOW -- so
`r.vectors[:, s]` is column `s` of that window, not band `s`. Indexing one with the other selects
the wrong subspace, and `_complement_gap` then measures that subspace's gap, so `mingap` looks
healthy while the answer is wrong. Neither existing guard catches it: the Fukui-Hatsugai link
determinants only see whether the (wrongly) selected subspace is isolated, and it usually is. On
two uncoupled Haldane copies, asking for band 1 under `nev = 3, which = :largest` returned band 2's
curvature field with `chern = +1` instead of `-1`, `mingap = 1.27`, and no warning. Hence the
mapping is resolved HERE, once, or the request is refused by name.

(An index past the window's end is the benign case -- `copyto!` raises `BoundsError` from inside a
threaded closure. It is the in-range mis-mapping above that is dangerous.)

The mapping, and why it is method-independent:
----------------------------------------------
`_diag_lanczos` returns the `nev` extremal eigenvalues ascending, exactly matching the dense path's
`_select_indices`, so `:auto` flipping to Krylov at `_DENSE_CUTOFF` cannot change *which* bands
come back -- only how they were computed. Hence one mapping covers every method:

- `which = :smallest`: the window is bands `1:nev`         -> relative == absolute
- `which = :largest`:  the window is bands `n-nev+1:n`     -> relative = absolute - (n - nev)
- `sigma`:             the `nev` states nearest an energy, whose absolute position is unknowable
                       from the window alone -> REFUSED

Interiority:
------------
The selection must leave at least one computed band on every TRUNCATED side. That is what keeps
`_complement_gap` correct with no change: it scans sel/non-sel boundaries within the values it is
handed, and interiority guarantees the boundary it finds is a real gap rather than the truncation
edge, where the true neighbour was never computed.
"""
function _partial_window(sel::Vector{Int}, n::Int, nev::Union{Nothing,Int}, which::Symbol,
                         sigma::Union{Nothing,Real}, method::Symbol;
                         caller::String)::Vector{Int}
    isnothing(sigma) || throw(ArgumentError(
        "`$caller`: `sigma = $sigma` asks for the `nev` bands nearest an energy, but an index " *
        "selection names the subspace by ABSOLUTE band index and a shift-invert window's " *
        "absolute position cannot be recovered from the window alone -- indexing it with band " *
        "numbers would select the wrong subspace silently. Use `which = :smallest` or " *
        "`:largest` with `nev` (those windows do map onto band indices), select by " *
        "`EnergyWindow` (resolved per k-point, so it accepts `sigma`), or compute the full " *
        "spectrum."))
    method in (:auto, :dense, :lanczos, :shift_invert) || throw(ArgumentError(
        "`$caller`: unknown `method = :$method`; expected `:auto`, `:dense`, " *
        "`:lanczos` or `:shift_invert`."))
    method === :shift_invert && throw(ArgumentError(
        "`$caller`: `method = :shift_invert` targets an interior energy through " *
        "`sigma`, which an index selection cannot map onto absolute band indices -- use " *
        "`which = :smallest`/`:largest` with `nev`, an `EnergyWindow` selection, or the full " *
        "spectrum."))
    if isnothing(nev)
        method === :lanczos && throw(ArgumentError(
            "`$caller`: `method = :lanczos` computes a partial spectrum; give `nev` so " *
            "the band selection can be mapped onto the computed window."))
        return sel                                  # full spectrum: relative == absolute
    end
    which in (:smallest, :largest) || throw(ArgumentError(
        "`$caller`: `which` must be `:smallest` or `:largest`; got `:$which`. Use " *
        "`sigma=` for an interior window -- which an index selection refuses, see above."))
    1 <= nev <= n || throw(ArgumentError("`$caller`: `nev = $nev` is outside 1:$n."))
    offset = which === :smallest ? 0 : n - nev
    lo, hi = offset + 1, offset + nev               # the absolute bands the window holds
    ### the truncated side(s): where the neighbouring band was never computed, so a gap measured
    ### against it would be the truncation and not the physics
    tlo, thi = _truncated_sides(n, nev, which, sigma)
    ilo = lo + (tlo ? 1 : 0)
    ihi = hi - (thi ? 1 : 0)
    (sel[1] >= ilo && sel[end] <= ihi) || throw(ArgumentError(
        "`$caller`: the band selection $sel is not strictly interior to the window this " *
        "partial solve computes. `nev = $nev` with `which = :$which` returns bands $lo:$hi, and " *
        "the subspace must leave one COMPUTED band on every truncated side -- so $ilo:$ihi here. " *
        "The isolation guard behind `mingap` measures the gap from the subspace to its " *
        "complement, and the neighbour just outside this window was never computed, so that gap " *
        "would be the truncation rather than the physics. Raise `nev`, or compute the full " *
        "spectrum."))
    return sel .- offset
end

"""
    _solver_kwargs(matrix, method, nev, which, sigma; caller) -> NamedTuple

The partial-spectrum keywords to hand [`bloch_solver`](@ref), assembled once per call.

`method` is deliberately withheld on the sparse path: `bloch_solver`'s sparse closure chooses
`:lanczos` / `:shift_invert` itself from `sigma`, and passing our own `method` alongside would
either shadow that choice or -- for `:auto` -- densify the very `n × n` matrix the sparse mode
exists to avoid. So `matrix = :sparse` with an explicit `method` is refused rather than silently
ignored.

`caller` names the public entry point in the refusal messages, and has NO default: this
validation is shared by `berry_curvature`, `wilson_loop` and the spectra drivers, and an error
blaming `berry_curvature` from `diagonalize_over_bz` or `wilson_loop` would send the user to
the wrong docstring. A missing `caller` is an `UndefKeywordError` here rather than a
misattributed refusal there.
"""
function _solver_kwargs(matrix::Symbol, method::Symbol, nev::Union{Nothing,Int}, which::Symbol,
                        sigma::Union{Nothing,Real}; caller::String)
    matrix in (:dense, :sparse) || throw(ArgumentError(
        "`$caller`: `matrix` must be `:dense` or `:sparse`; got `:$matrix`."))
    matrix === :sparse || return (; nev=nev, which=which, sigma=sigma, method=method)
    isnothing(nev) && throw(ArgumentError(
        "`$caller`: `matrix = :sparse` reaches the spectrum only through the Krylov " *
        "paths, which compute a PARTIAL spectrum; give `nev` (with `which`, or with `sigma` and " *
        "an `EnergyWindow` selection). For the full spectrum use `matrix = :dense`."))
    method === :auto || throw(ArgumentError(
        "`$caller`: `matrix = :sparse` picks its own solver from `sigma` " *
        "(`:shift_invert` when given, `:lanczos` otherwise); passing `method = :$method` as " *
        "well would either contradict that or densify the matrix. Drop `method`."))
    return (; nev=nev, which=which, sigma=sigma)
end

"""
    _truncated_sides(n, nev, which, sigma) -> (below::Bool, above::Bool)

Which ends of the values `diagonalize` returns are a TRUNCATION rather than the end of the
spectrum -- i.e. where the neighbouring band exists but was never computed. `nev >= n` is the full
spectrum however it was asked for, so neither side is truncated there.

This is the one place the window geometry is decided; both the index path (`_partial_window`) and
the energy path (`_window_selection`) read it, so they cannot drift apart.
"""
@inline function _truncated_sides(n::Int, nev::Union{Nothing,Int}, which::Symbol,
                                  sigma::Union{Nothing,Real})::Tuple{Bool,Bool}
    (isnothing(nev) || nev >= n) && return (false, false)
    isnothing(sigma) || return (true, true)         # an interior window is cut on both sides
    return which === :smallest ? (false, true) : (true, false)
end

"""
    _window_selection(values, w::EnergyWindow, tlo, thi, k; caller) -> Vector{Int}

The indices of `values` lying strictly inside the energy window, refusing an unusable window by
name. `k` is quoted in the messages only; `caller` names the public entry point there and has
no default, for the reason given on `_solver_kwargs` (this validation is shared by
`berry_curvature`, `wilson_loop` and the spectra drivers).

The bracket condition is the energy-space twin of `_partial_window`'s interiority: a selection
that runs to a TRUNCATED end of the computed set has no measurable complement gap on that side,
because the band it would be gapped against was never computed. `_complement_gap` would then
report the gap to the wrong neighbour -- or `Inf` when the window swallows every computed state,
which passes `chern_number`'s isolation guard unnoticed.
"""
function _window_selection(values::Vector{Float64}, w::EnergyWindow, tlo::Bool, thi::Bool,
                           k::AbstractVector{<:Real};
                           caller::String)::Vector{Int}
    sel = findall(e -> w.lo < e < w.hi, values)
    isempty(sel) && throw(ArgumentError(
        "`$caller`: the energy window ($(w.lo), $(w.hi)) holds no band at k = " *
        "$(Tuple(k)), where the computed spectrum spans " *
        "($(minimum(values)), $(maximum(values))). Widen the window, or -- for a partial solve " *
        "-- move `sigma` onto the manifold you meant."))
    if tlo && sel[1] == 1
        throw(ArgumentError(
            "`$caller`: the energy window ($(w.lo), $(w.hi)) reaches the BOTTOM of the " *
            "computed window at k = $(Tuple(k)): its lowest band is the lowest state the solve " *
            "returned, so nothing was computed below it to measure a gap against, and the " *
            "subspace cannot be shown to be isolated. Raise `nev` so the solve brackets the " *
            "window from below, or move `sigma` up."))
    end
    if thi && sel[end] == length(values)
        throw(ArgumentError(
            "`$caller`: the energy window ($(w.lo), $(w.hi)) reaches the TOP of the " *
            "computed window at k = $(Tuple(k)): its highest band is the highest state the " *
            "solve returned, so nothing was computed above it to measure a gap against. Raise " *
            "`nev` so the solve brackets the window from above, or move `sigma` down."))
    end
    return sel
end

"""
    _complement_gap(values, sel) -> Float64

The smallest energy distance between the selected bands and any unselected one, `Inf` when the
selection spans the whole spectrum.
"""
function _complement_gap(values::Vector{Float64}, sel::Vector{Int})::Float64
    ### `values` is ascending (`diagonalize`'s ordering contract) and `sel` is sorted
    ### (`_band_selection`), so the closest selected/unselected pair is ADJACENT in energy
    ### order: only the boundaries between the two index sets need to be scanned --
    ### O(n log nb) and allocation-free, where the all-pairs scan was O(n·nb) plus a BitVector
    ### per k-point.
    g = Inf
    for b in 1:(length(values) - 1)
        if insorted(b, sel) != insorted(b + 1, sel)
            d = values[b + 1] - values[b]
            d < g && (g = d)
        end
    end
    return g
end
