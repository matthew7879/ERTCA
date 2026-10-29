##################################################################
###### export only the following functions from this module ######
##################################################################

export bloch_solver, bz_map

### ------------------------------------------------------------------------------------
### Running a compiled model over k-points
### ------------------------------------------------------------------------------------
###
### `bloch_solver` turns a compiled model into a reusable `k -> (values, vectors, gauge)`
### closure -- one buffer, or one sparse workspace, reused across every k it is called on --
### and `bz_map` is the threaded one-call form of it. Everything they touch lives at or below
### this layer: `bloch_hamiltonian!`, the sparse workspace, `diagonalize` and `threaded_chunks`.
###
### They lived in `analysis/bandstructure/spectrum_along_line.jl` until 2026-09-16, because a
### band structure along a path was the first thing that needed them. That put the package's
### solver factory ABOVE `spectra/`, whose grid driver is the other main caller -- so
### `diagonalize_over_bz` reached up into `analysis/` for it. Being a k-loop driver is not the
### same concept as being a band structure, and this file holds only the first.



"""
    bz_map(f, model, ks; matrix=:dense, kwargs...) -> Vector

Diagonalize `model` at every momentum in `ks`, in parallel, and collect `f(k, r)` in `ks`
order. The one-call form of the [`threaded_chunks`](@ref) + [`bloch_solver`](@ref) pattern:
one solver (hence one reusable buffer) per thread, BLAS throttled for the duration, results
returned in order regardless of which thread produced them.

This is the driver for filling the Brillouin zone with any per-k quantity -- occupations, form
factors, spin textures, overlaps, interaction matrix elements -- without re-deriving the loop:

    # the flat-band Chern-like integrand on a Monkhorst-Pack grid
    ks = monkhorst_pack(model.cell.lattice, (24, 24))
    occ = bz_map(model, ks; nev=4) do k, r
        sum(exp.(-r.values ./ 0.01))
    end

    # eigenvectors: COPY what you keep -- see below
    vs = bz_map(model, ks; nev=2) do k, r
        copy(r.vectors)
    end

`kwargs` go straight to [`diagonalize`](@ref) through the solver (`nev`, `which`, `sigma`,
`vectors`, ...), and `matrix = :sparse` selects the workspace form for models where `n²`
storage is the enemy (it computes a partial spectrum by construction, so pass `nev`).

Two contracts inherited from `bloch_solver`, both load-bearing:

- **`r.vectors` may alias the solver's own buffer**, which the next `k` overwrites. Return a
  `copy` from `f` for anything that must outlive the call; `r.values` is freshly allocated.
- `r.vectors` columns are in the cell's state order and in the model's STORED gauge, echoed as
  `r.gauge` -- eigenvectors in the two conventions differ by the k-dependent translator
  `gauge_matrix`, so a position-dependent quantity must read the atomic assembly.

`f` runs on worker threads, so it must not mutate shared state; write into `ks`-indexed slots
of a preallocated array, or return a value and let `bz_map` collect it.

The element type is taken from `f` at the first momentum, so `f` must return one type for all
of `ks` (or a common supertype -- annotate with `f(k, r)::T` if inference needs help).
"""
function bz_map(f, model::HamiltonianModel{D}, ks::AbstractVector; matrix::Symbol=:dense,
                kwargs...) where {D}
    nk = length(ks)
    nk == 0 && throw(ArgumentError("`ks` is empty; there is nothing to map over."))
    ### the first momentum runs here, serially, to fix the element type before allocating
    first_val = bloch_solver(model; matrix=matrix)(ks[1]; kwargs...) |> r -> f(ks[1], r)
    out = Vector{typeof(first_val)}(undef, nk)
    out[1] = first_val
    nk == 1 && return out
    threaded_chunks(nk - 1) do chunk
        solve = bloch_solver(model; matrix=matrix)     # ONE buffer per chunk
        for c in chunk
            i = c + 1
            out[i] = f(ks[i], solve(ks[i]; kwargs...))
        end
    end
    return out
end

"""
    bloch_solver(model; matrix=:dense) -> solve(k; kwargs...) -> (values, vectors, gauge)

Build the per-k solver every batch driver in this package loops over: a closure that assembles
`H(k)` into ITS OWN reusable buffer and hands it to [`diagonalize`](@ref) (whose keywords --
`nev`, `which`, `sigma`, `vectors`, ... -- pass straight through). This is the building block
for a CUSTOM k-loop analysis (spin textures, Wilson loops, overlaps):

    threaded_chunks(length(kpts)) do chunk
        solve = bloch_solver(model)                 # one buffer per chunk -- see below
        for i in chunk
            r = solve(kpts[i]; nev=4)
            # r.values ascending; r.vectors columns in the cell's state order (STATE_ORDER),
            # in the model's stored gauge -- echoed as r.gauge, because eigenvectors in the
            # two conventions differ by the k-dependent translator `gauge_matrix`
        end
    end

- `matrix = :dense` allocates one `n × n` buffer, refills it with `bloch_hamiltonian!` and
  diagonalizes in place (`overwrite=true`) -- zero per-k-point matrix allocation. The returned
  eigenvectors may alias that buffer: copy out anything that must survive the next call.
- `matrix = :sparse` builds one `sparse_bloch_workspace` and rewrites its non-zeros per k-point
  -- the mode for large models where `n²` storage is the enemy; it computes a PARTIAL spectrum
  by construction, so pass `nev` (and the Krylov method follows `sigma` unless named).

The closure owns mutable state (the buffer / the workspace), so it is NOT thread-safe: build
one solver per chunk (as above), never one shared across threads.
"""
function bloch_solver(model::HamiltonianModel{D}; matrix::Symbol=:dense) where {D}
    matrix in (:dense, :sparse) || throw(ArgumentError(
        "`matrix` must be `:dense` or `:sparse`; got `$matrix`."))
    if matrix === :sparse
        ws = sparse_bloch_workspace(model)
        return function (k; sigma=nothing, kwargs...)
            S = sparse_bloch_hamiltonian!(ws, model, k)
            ### Krylov by default: `:auto` would densify a small sparse matrix, allocating the
            ### very n² copies this mode exists to avoid. The caller's `method` still wins.
            meth = isnothing(sigma) ? :lanczos : :shift_invert
            r = diagonalize(S; method=meth, sigma=sigma, kwargs...)
            ### the gauge tag: eigenvectors are convention-dependent (they differ by
            ### `gauge_matrix`), so the result names the convention it is expressed in
            return (values=r.values, vectors=r.vectors, gauge=model.gauge)
        end
    end
    H = zeros(ComplexF64, n_states(model.cell), n_states(model.cell))
    return function (k; kwargs...)
        bloch_hamiltonian!(H, model, k)
        r = diagonalize(H; overwrite=true, kwargs...)
        ### same gauge tag as the sparse closure above
        return (values=r.values, vectors=r.vectors, gauge=model.gauge)
    end
end
