##################################################################
###### export only the following functions from this module ######
##################################################################

export BlochSpectrum, diagonalize_over_bz, wrap_unitary
export n_bands, n_kpoints, states_matrix, is_conserved

### Contract satisfied here, fixed by the 2026-09-02 gauge adjudication (this file's former
### placeholder text):
###  1. Stored eigenvector grids carry a `gauge` tag, exactly like `bloch_solver` and
###     `eigenstates_along_line` results -- `BlochSpectrum.gauge`, always `:atomic` (see the
###     driver's Notes for why the atomic convention is the stored one).
###  2. Zone-boundary reuse through `bloch_periodicity` is :exact-cells-only and
###     constant-kernels-only -- `wrap_unitary` is that operator with the full guard set,
###     mirroring `berry_curvature`'s `seam=:glue` guards. The mesh itself needs no seam: it is
###     a fundamental-domain sample (no duplicated zone boundary), so wraps happen only at
###     consumption, through `kshift`'s exact carry plus `wrap_unitary`.




#############################################################
###### The eigen-grid container                          #####
#############################################################

"""
    BlochSpectrum{D}

Eigenvalues (and optionally eigenvectors) of a model on a [`BZMesh`](@ref) -- the result of
[`diagonalize_over_bz`](@ref), and the input to the form-factor layer.

Fields:
-------
- `mesh::BZMesh{D}`: the grid, with its index arithmetic ([`kshift`](@ref),
  [`momentum_conservation`](@ref)) -- the container knows its own discretization.
- `cell::UnitCell{D}`: the basis the eigenvectors are expressed in. This, not the model, is
  what the layer above actually needs: `form_factors` reads the cell and its k-periodicity, and
  the zone-wrap unitary is a property of the basis rather than of the Hamiltonian. Keeping it
  as its own field is what lets an eigenvector grid imported from an external band-structure
  code -- which has no compiled Hamiltonian at all -- go through the same machinery.
- `model::Union{Nothing,HamiltonianModel{D}}`: the ATOMIC-gauge view of the diagonalized model
  ([`change_gauge`](@ref) is O(1)), kept so consumers can reach [`gauge_matrix`](@ref),
  [`wrap_unitary`](@ref) and the cell without carrying the model separately.
- `energies::Matrix{Float64}`: `n_bands × n_kpoints`, ascending per column (the
  [`diagonalize`](@ref) ordering contract), columns in the mesh's K-ORDER.
- `states::Union{Nothing,Array{ComplexF64,3}}`: `n_states × n_bands × n_kpoints`, or `nothing`
  when computed with `vectors = false`. `states[:, b, i]` is the eigenvector of
  `energies[b, i]` at `k_fractional(mesh, i)`, components in the cell's state order
  (`STATE_ORDER`). The k axis is last and linear, so `view(states, :, :, i)` is a CONTIGUOUS
  block -- one BLAS-3 operand per k-point, which is what the form-factor contraction consumes.
- `gauge::Symbol`: the Bloch convention the stored eigenvectors are expressed in -- always
  `:atomic` (the binding grid contract; see [`diagonalize_over_bz`](@ref) for why). Eigenvectors
  in the two conventions differ by the k-dependent translator [`gauge_matrix`](@ref), and an
  untagged grid invites silently mixing conventions in downstream overlaps.
- `nev, which, sigma, window`: how the spectrum was selected -- `nothing`/`:smallest`/
  `nothing`/`nothing` for a full solve. Consumers that need the complement (gaps, band
  isolation) read these instead of assuming the full spectrum.
The type's `L` parameter rides along from the mesh (StaticArrays' D² length; see
[`BZMesh`](@ref)) so the field stays concretely typed.
"""
struct BlochSpectrum{D,L,K<:KPeriodicity}
    mesh::BZMesh{D,L}
    cell::UnitCell{D,K}
    model::Union{Nothing,HamiltonianModel{D,K}}
    energies::Matrix{Float64}
    states::Union{Nothing,Array{ComplexF64,3}}
    gauge::Symbol
    nev::Union{Nothing,Int}
    which::Symbol
    sigma::Union{Nothing,Float64}
    window::Union{Nothing,EnergyWindow}
end

"""
    n_bands(spectrum) -> Int

The number of stored bands per k-point (rows of `energies`).
"""
n_bands(sp::BlochSpectrum)::Int = size(sp.energies, 1)

"""
    conserved_qnums(spectrum) -> Union{Nothing,Vector{Symbol}}

The internal axes the source Hamiltonian conserves, or `nothing` when this spectrum carries no
model to derive them from (a Quantum ESPRESSO import, say).

`nothing` means UNKNOWN, and a consumer must not read it as "none". The two are different
claims: an imported plane-wave spectrum has symmetries nobody derived, while a compiled model
with a Rashba term genuinely conserves no spin axis, and a quantity that treats the first as
the second would silently block-diagonalize something that does not block.

Derived, never stored -- exactly as [`conserved_qnums`](@ref) is for a model, and for the same
reason: an asserted list goes stale the moment a symmetry-breaking term is added, and the
whole point of this accessor is to notice when that happens. The Hartree-Fock case is the
motivating one: spin U(1) is conserved until the antiferromagnetic mean field couples the
sectors, at which point the derivation stops reporting `:spin` on its own, with no API to
remember to call.

See [`is_conserved`](@ref) for the query that refuses on `nothing` rather than guessing.
"""
conserved_qnums(sp::BlochSpectrum)::Union{Nothing,Vector{Symbol}} =
    sp.model === nothing ? nothing : conserved_qnums(sp.model)

"""
    is_conserved(obj, key::Symbol) -> Bool

Whether `obj` -- a [`BlochSpectrum`](@ref) or a [`FormFactors`](@ref) -- conserves the internal
axis `key`.

REFUSES when the conservation facts are unknown (`conserved_qnums(obj) === nothing`), because
the caller is about to branch on the answer and "unknown" is not "no". That is the whole
difference between this and `key in something`.
"""
function is_conserved(obj, key::Symbol)::Bool
    c = conserved_qnums(obj)
    c === nothing && throw(ArgumentError(
        "`is_conserved`: this $(nameof(typeof(obj))) carries no model, so which axes are " *
        "conserved was never derived -- the answer is UNKNOWN, not `false`. Rebuild it from " *
        "a compiled model, or ask the model directly with `conserved_qnums(model)`."))
    return key in c
end

"""
    n_kpoints(spectrum) -> Int

The number of k-points (columns of `energies` = `length(spectrum.mesh)`).
"""
n_kpoints(sp::BlochSpectrum)::Int = size(sp.energies, 2)

"""
    states_matrix(spectrum, i) -> AbstractMatrix

The `n_states × n_bands` eigenvector block at k-index `i`, as a contiguous view (zero-copy;
the BLAS-3 operand of the form-factor contraction).
"""
function states_matrix(sp::BlochSpectrum, i::Integer)
    sp.states === nothing && throw(ArgumentError(
        "This `BlochSpectrum` was computed with `vectors = false` and stores no " *
        "eigenvectors. Re-run `diagonalize_over_bz` with `vectors = true`."))
    1 <= i <= n_kpoints(sp) || throw(ArgumentError(
        "`states_matrix`: k-index $i is outside 1:$(n_kpoints(sp))."))
    return view(sp.states, :, :, i)
end

function Base.show(io::IO, sp::BlochSpectrum{D}) where {D}
    print(io, "BlochSpectrum{", D, "} ", n_bands(sp), " band", n_bands(sp) == 1 ? "" : "s",
          " × ", n_kpoints(sp), " k-points (gauge :", sp.gauge, ", ",
          sp.states === nothing ? "no vectors" : "vectors stored", ")")
    return nothing
end




#############################################################
###### The grid diagonalization driver                  #####
#############################################################

"""
    diagonalize_over_bz(model, mesh; nev=nothing, which=:smallest, sigma=nothing,
                        method=:auto, window=nothing, matrix=:dense, vectors=true, kwargs...)
        -> BlochSpectrum{D}

Diagonalize `model` at every momentum of `mesh` -- the twisted-boundary-condition cluster
diagonalization: the mesh IS the boundary condition (see [`bz_mesh`](@ref)), and the model is
never modified.

Args:
-----
- `model`: any `HamiltonianModel{D}` with a Brillouin zone (`kperiodicity != :none`).
- `mesh`: a [`BZMesh`](@ref) on the model's own lattice (checked; a mesh on a different
  lattice samples wrong momenta silently, so it is refused by name).
- `nev`, `which`, `sigma`, `method`: partial-spectrum controls, forwarded through the same
  validation as [`berry_curvature`](@ref) (`_solver_kwargs`): `nev` bands, `:smallest`/
  `:largest`, or nearest `sigma` (shift-invert).
- `window`: an [`EnergyWindow`](@ref) (or `(lo, hi)` tuple) selecting the stored bands by
  ENERGY -- the multi-orbital moiré call: it composes with `nev` + `sigma`, and the selection
  is verified at every k-point (same rank everywhere, bracketed by computed states on every
  truncated side) exactly as in `berry_curvature`.
- `matrix`: `:dense` (one reusable buffer per thread chunk, `overwrite = true` -- zero per-k
  matrix allocation) or `:sparse` (one CSC workspace per chunk; requires `nev`).
- `vectors`: `false` stores energies only (`states === nothing`).
- `kwargs...`: forwarded to [`diagonalize`](@ref) (tolerances, `krylovdim`, ...).

Returns:
--------
- [`BlochSpectrum{D}`](@ref); energies ascending per column, columns in the mesh's K-ORDER,
  deterministic regardless of threading (each k-point writes its own slot).

Notes:
------
**Gauge.** The driver always assembles the ATOMIC convention (`change_gauge` is an O(1) flag
flip) and tags the result `gauge = :atomic`. Two reasons, both downstream: (i) the exact
zone-wrap unitary [`wrap_unitary`](@ref) exists for the atomic assembly only, and is
k-INDEPENDENT there -- one matrix per distinct reciprocal carry, applicable BLAS-3 -- while the
periodic view's boundary unitary is k-dependent; (ii) the form-factor contract derives the
periodic convention at ITS boundary from the stored atomic grid via [`gauge_matrix`](@ref)
(a diagonal, cheap per k), never the other way around. A consumer that wants periodic-gauge
vectors multiplies by `gauge_matrix(model.cell, k)` -- and the tag says so instead of letting
two conventions mix silently.

**Degeneracies.** Values are ascending and deterministic; the eigenvector BASIS within a
degenerate multiplet is solver-dependent. Multiband quantities of an isolated group (subspace
projectors, non-Abelian links) are covariant under that freedom; a single-band quantity AT a
degeneracy is not well-defined -- no driver option can fix that.

Example:
--------
    lat  = bravais_lattice(:triangular; a=1.0)
    m    = compile(cell, terms)
    mesh = bz_mesh(lat, [6 0; 0 6]; fluxes=[0.1, 0.0])      # 36-site cluster, twisted
    sp   = diagonalize_over_bz(m, mesh; nev=4, which=:smallest)
    sp.energies[1, k_index(mesh, (2, 3))]                    # lowest band at grid point (2,3)
"""
function diagonalize_over_bz(model::HamiltonianModel{D}, mesh::BZMesh{D};
                             nev::Union{Nothing,Int}=nothing, which::Symbol=:smallest,
                             sigma::Union{Nothing,Real}=nothing, method::Symbol=:auto,
                             window::Union{Nothing,EnergyWindow,Tuple{Real,Real}}=nothing,
                             matrix::Symbol=:dense, vectors::Bool=true,
                             kwargs...) where {D}
    _require_zone(kperiodicity(model), "`diagonalize_over_bz` tiles the Brillouin zone",
        "Evaluate on an explicit `kregion` grid (`density_of_states`/`berry_curvature` " *
        "style), or `discretize` the model.")
    same_basis(mesh.lattice, model.cell.lattice;
               atol=LATTICE_ATOL * length_scale(model.cell.lattice)) || throw(ArgumentError(
        "`diagonalize_over_bz`: the mesh discretizes a DIFFERENT lattice than the model's -- " *
        "its fractional momenta would silently mean different cartesian momenta. Build the " *
        "mesh from the model's own lattice: `bz_mesh(model, ...)`."))
    solve_kw = _solver_kwargs(matrix, method, nev, which, sigma; caller="diagonalize_over_bz")
    ewin = window isa Tuple ? EnergyWindow(window) : window

    am0 = change_gauge(model, :atomic)
    am = am0                              # fresh binding: the threaded closure must not box it
    n = n_states(am.cell)
    Nk = length(mesh)

    ### the first momentum runs serially: it fixes the stored band count (and, with a window,
    ### the reference rank every other k-point is checked against)
    probe = bloch_solver(am; matrix=matrix)(mesh.kpts[1]; vectors=vectors, solve_kw...,
                                            kwargs...)
    tlo, thi = _truncated_sides(n, nev, which, sigma)
    sel1 = ewin === nothing ? collect(1:length(probe.values)) :
           _window_selection(probe.values, ewin, tlo, thi, mesh.kpts[1];
                             caller="diagonalize_over_bz")
    nb = length(sel1)

    energies = Matrix{Float64}(undef, nb, Nk)
    states = vectors ? Array{ComplexF64,3}(undef, n, nb, Nk) : nothing
    _store_point!(energies, states, probe, sel1, 1)

    if Nk > 1
        threaded_chunks(Nk - 1) do chunk
            solve = bloch_solver(am; matrix=matrix)         # ONE buffer/workspace per chunk
            for c in chunk
                i = c + 1
                r = solve(mesh.kpts[i]; vectors=vectors, solve_kw..., kwargs...)
                sel = ewin === nothing ? sel1 :
                      _window_selection(r.values, ewin, tlo, thi, mesh.kpts[i];
                                        caller="diagonalize_over_bz")
                length(sel) == nb || throw(ArgumentError(
                    "`diagonalize_over_bz`: the energy window ($(ewin.lo), $(ewin.hi)) holds " *
                    "$(length(sel)) bands at k = $(Tuple(mesh.kpts[i])) but $nb at the first " *
                    "mesh point -- a band crosses the window edge inside the mesh, so the " *
                    "stored grid would change rank. Move the window edges into a gap, or " *
                    "select by `nev` instead."))
                _store_point!(energies, states, r, sel, i)
            end
        end
    end
    return BlochSpectrum(mesh, am0.cell, am0, energies, states, :atomic, nev, which,
                            sigma === nothing ? nothing : Float64(sigma), ewin)
end

"""
    _store_point!(energies, states, r, sel, i) -> Nothing

Write one solver result into k-column `i`: the selected values, and (when stored) the selected
eigenvector columns via per-column `copyto!` on views -- `r.vectors` may alias the solver
buffer, so everything kept is copied out before the next solve reuses it.
"""
@inline function _store_point!(energies::Matrix{Float64},
                               states::Union{Nothing,Array{ComplexF64,3}},
                               r, sel::Vector{Int}, i::Int)::Nothing
    @inbounds for (b, s) in enumerate(sel)
        energies[b, i] = r.values[s]
    end
    states === nothing && return nothing
    @inbounds for (b, s) in enumerate(sel)
        @views copyto!(states[:, b, i], r.vectors[:, s])
        @views _pin_phase!(states[:, b, i])
    end
    return nothing
end

### The per-band phase LAPACK leaves arbitrary, pinned to something every build agrees on.
###
### An eigenvector is defined only up to a phase, and `zheev` picks one however its algorithm
### happens to fall out: Accelerate on aarch64 and OpenBLAS on x86_64 return DIFFERENT phases
### for the same Hamiltonian, and two OpenBLAS versions differ from each other. That is
### invisible in every observable -- bands, densities, Berry curvature and form factors are all
### built from phase-invariant combinations -- right up until a phase CROSSES A MACHINE. A
### Wannier90 `.amn`/`.mmn` deck is written from these vectors, and the `_u.mat` the binary
### returns refers to the gauge it was given; re-importing it against freshly diagonalized
### vectors then silently mixes two conventions and reports a wrong spread. So the deck path
### needs the eigenvectors to be a FUNCTION of the Hamiltonian, not of the BLAS.
###
### The convention: rotate so that the FIRST component whose amplitude clears
### `PHASE_PIN_ATOL` is real and positive. Deliberately first-above-threshold rather than
### largest-in-modulus: symmetry routinely makes several components equal in modulus
### (graphene's `(1, ±e^{iθ})/√2` has two), and picking the largest would then decide on a
### last-bit comparison that genuinely differs between builds. A fixed threshold has no such
### knife-edge -- the only ambiguity is a component sitting exactly at `PHASE_PIN_ATOL`, and
### the vectors are normalized, so that is a component carrying no weight at all.
###
### Degenerate multiplets are NOT fixed by this: there the eigensolver chooses a basis, not
### just a phase, and no per-column rule can pin it. Quantities that need a degenerate subspace
### must stay subspace-invariant, as they already must be.
@inline function _pin_phase!(v::AbstractVector{ComplexF64})::Nothing
    @inbounds for k in eachindex(v)
        z = v[k]
        a = abs(z)
        a > PHASE_PIN_ATOL || continue
        c = conj(z) / a                # unit modulus: rotates v[k] onto the positive real axis
        for j in eachindex(v)
            v[j] *= c
        end
        return nothing
    end
    return nothing                     # an all-but-zero column has no phase worth pinning
end




#############################################################
###### The sanctioned zone-boundary wrap                #####
#############################################################

"""
    wrap_unitary(model, G) -> Matrix{ComplexF64}
    wrap_unitary(spectrum::BlochSpectrum, G)

THE sanctioned zone-boundary wrap operator: the k-independent unitary with
`H(k + G) = W' * H(k) * W`, so a stored eigenvector wraps as `ψ(k + G) ≅ W' * ψ(k)` (equal up
to the eigensolver's per-band phase; subspace-exact for degenerate multiplets). `G` is the
exact integer carry [`kshift`](@ref) returns.

This is [`bloch_periodicity`](@ref) plus the one guard it lacks: k-DEPENDENT kernels are
refused here (their argument `bᵀ(k + q)` is not BZ-periodic, so no k-independent unitary
exists) -- the same third guard `berry_curvature`'s `seam = :glue` applies. The inherited
refusals stay: `:truncated` models (H(k+G) is a DIFFERENT truncation window -- evaluate at the
unwrapped momentum instead, the seam-recompute strategy), `:none` models (no torus),
non-`:atomic` gauge, and fractional or aperiodic-direction `G`.

A `BlochSpectrum` routes to whichever owner the grid carries -- its model, or its cell alone for
a grid imported from an external band-structure code -- with the same refusals on both routes.
"""
function wrap_unitary(model::HamiltonianModel{D},
                      G::AbstractVector{<:Integer})::Matrix{ComplexF64} where {D}
    has_kdependence(model) && throw(ArgumentError(
        "`wrap_unitary`: this model carries k-dependent kernels, whose argument `bᵀ(k + q)` " *
        "is not BZ-periodic -- H(k+G) is not related to H(k) by any k-independent unitary. " *
        "Evaluate at the unwrapped momentum instead (the `berry_curvature` seam-recompute " *
        "strategy)."))
    return bloch_periodicity(model, G)
end
### a fractional G is refused by `_integer_shift_or_refuse` -- one message, one owner, shared
### with `bloch_periodicity`, but naming the function the caller actually invoked
wrap_unitary(model::HamiltonianModel{D}, G::AbstractVector{<:Real}) where {D} =
    wrap_unitary(model, _integer_shift_or_refuse(G, "wrap_unitary"))

"""
    wrap_unitary(cell::UnitCell, G) -> Matrix{ComplexF64}

The zone-wrap unitary from the CELL alone, for an eigenvector grid that has no compiled
Hamiltonian behind it -- an import from an external band-structure code.

`W` is a property of the basis, not of the Hamiltonian: it relabels `|τ, G_s; k+G⟩` as
`|τ, G_s+G; k⟩`. The guards a model method also applies are kept here in cell terms -- the
shift must vanish along aperiodic directions, and the cell must be exactly k-periodic, since on
a truncated plane-wave basis `k` and `k+G` carry different truncation windows and no unitary
relates them exactly, and `G` must be an exact integer carry. There is no gauge guard because a
cell has no assembly: the caller declares its gauge, as the spectra layer requires.
"""
function wrap_unitary(cell::UnitCell{D},
                      G::AbstractVector{<:Integer})::Matrix{ComplexF64} where {D}
    length(G) == D || throw(DimensionMismatch("G has length $(length(G)), expected $D."))
    _require_periodic_shift(cell.lattice, G, "wrap_unitary")
    _require_exact_zone(kperiodicity(cell), "`wrap_unitary`",
        "Use the label gather with accounted loss (`form_factors` does this automatically on " *
        "such cells), or evaluate at the unwrapped momentum.",
        "There is no zone boundary to wrap across.")
    return _wrap_unitary_cell(cell, G)
end
### the same fractional refusal the model method carries: a cell has fewer guards, not laxer
### ones, and without this method a float `G` reached the caller as a bare `MethodError`
wrap_unitary(cell::UnitCell{D}, G::AbstractVector{<:Real}) where {D} =
    wrap_unitary(cell, _integer_shift_or_refuse(G, "wrap_unitary"))

### The `BlochSpectrum` entry point routes to whichever owner the grid has -- the compiled model
### when there is one, the cell alone for an imported grid (`read_qe_spectrum`) -- and forwards
### `G` UNCONVERTED so both routes get identical treatment. It used to write `Int.(G)` on the
### model-less side, which silently made the import path the odd one out: `[0.5, 0.0]` died with
### `InexactError: Int64(0.5)` instead of the named refusal, and `[1.0, 0.0]` was accepted there
### while the model route refused it. Dispatch on the eltype does that job correctly for free.
wrap_unitary(sp::BlochSpectrum{D}, G::AbstractVector{<:Real}) where {D} =
    sp.model === nothing ? wrap_unitary(sp.cell, G) : wrap_unitary(sp.model, G)
