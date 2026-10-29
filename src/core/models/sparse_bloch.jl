##################################################################
###### export only the following functions from this module ######
##################################################################

export sparse_bloch_workspace, sparse_bloch_hamiltonian, sparse_bloch_hamiltonian!




#############################################################
###### Sparse Bloch Hamiltonians                        #####
#############################################################

"""
    SparseBlochWorkspace{D}

The reusable machinery for assembling `H(k)` as a `SparseMatrixCSC`: the sparsity pattern,
computed ONCE, plus the slot map that lets every k-point refill only the non-zero values.

Purpose:
--------
The dense Bloch Hamiltonian is `O(n²)` in memory and in per-k assembly, which is what caps the
package at a few thousand states. But the compiled coupling list already IS a sparse matrix in
COO form -- `(i, j, ΔR, q, w, κ)` with k entering only through phases and kernels -- and a
ribbon, a disorder supercell, or a small-angle moiré model touches a vanishing fraction of the
`n²` entries. This workspace turns the list into CSC once; after that, one k-point costs one
pass over the non-zeros, and `diagonalize`'s Lanczos / shift-invert paths do the rest.

Fields:
-------
- `S`: the matrix, structure fixed, values rewritten per k
- `slot[e]`: index into `nonzeros(S)` receiving coupling entry `e`

Notes:
------
Entries sharing `(i, j)` -- several `ΔR` on one pair, several kernels overlapping -- share one
slot and accumulate, exactly as they would in the dense array. The list is sorted `(kernel, j,
i, ΔR)`, so slot indices are monotone within each kernel group. The pattern is built from the
model at construction and identified by the coupling list's `pattern_id` -- so the workspace
accepts exactly the models whose slots it can fill correctly: the model it was built from and
everything [`reweight`](@ref)/[`change_gauge`](@ref) derive from it (amplitudes change, slots
do not). A sweep therefore pays for the CSC pattern ONCE. Anything else refuses by name.
"""
struct SparseBlochWorkspace{D}
    S::SparseMatrixCSC{ComplexF64,Int}
    slot::Vector{Int}
    ### the sparsity-pattern fingerprint of the coupling list the slots were built from
    ### (`CouplingList.pattern_id`): a list with a DIFFERENT pattern would otherwise assemble
    ### garbage silently through stale slots
    source::UInt
end

"""
    sparse_bloch_workspace(model) -> SparseBlochWorkspace

Build the fixed sparsity pattern of `H(k)` for this model. Do it once, outside any k-loop.
"""
function sparse_bloch_workspace(model::HamiltonianModel{D})::SparseBlochWorkspace{D} where {D}
    n = n_states(model.cell)
    c = model.couplings
    isempty(c.i) && throw(ArgumentError(
        "The model has no couplings at all; a sparse H(k) would be identically zero."))

    ### `sparse` coalesces duplicates; zeros as placeholder values give the pure pattern
    S = sparse(copy(c.i), copy(c.j), zeros(ComplexF64, n_couplings(c)), n, n)

    ### slot of (i, j) = position in nzval: binary search of i in the rowval range of column j
    slotof(i, j) = begin
        r = searchsortedfirst(view(rowvals(S), nzrange(S, j)), i)
        first(nzrange(S, j)) + r - 1
    end
    slot = [slotof(c.i[e], c.j[e]) for e in eachindex(c.i)]
    return SparseBlochWorkspace{D}(S, slot, c.pattern_id)
end

"""
    sparse_bloch_hamiltonian!(ws, model, k_frac) -> SparseMatrixCSC

Refill the workspace with `H(k)` at fractional momentum `k_frac` and return its matrix.

The sparse counterpart of `bloch_hamiltonian!`: identical physics (same phases, same kernels,
gauge already baked into `model.delta`), but one pass over the non-zeros instead of an `n × n`
array. The workspace must have been built from a model with THIS sparsity pattern -- the model
itself, or any model [`reweight`](@ref)/[`change_gauge`](@ref) derived from it (the advertised
sweep pairing: one workspace for the whole phase diagram).
"""
function sparse_bloch_hamiltonian!(ws::SparseBlochWorkspace{D}, model::HamiltonianModel{D},
                                   k_frac::AbstractVector{<:Real}) where {D}
    length(k_frac) == D || throw(DimensionMismatch(
        "k_frac has length $(length(k_frac)), expected $D."))
    k = SVector{D,Float64}(k_frac)
    _check_k_periodic(model.cell.lattice, k)
    ### the pattern check runs BEFORE the workspace is touched: a refused call must not wipe
    ### the H(k) a previous successful call left in the aliased ws.S
    c = model.couplings
    ws.source == c.pattern_id || throw(ArgumentError(
        "Sparsity-pattern mismatch: this model's coupling list was not produced from the " *
        "workspace's model by `reweight`/`change_gauge`. Slot indices are pattern-specific " *
        "-- an entry-count coincidence would assemble garbage silently, so the pattern " *
        "fingerprint is checked, not the size. Build a fresh `sparse_bloch_workspace(model)`."))
    nz = nonzeros(ws.S)
    fill!(nz, zero(ComplexF64))

    bT = SMatrix{D,D,Float64}(transpose(model.cell.lattice.b))
    for κ in eachindex(c.kernels)
        isempty(c.groups[κ]) && continue
        _accumulate_sparse!(nz, ws.slot, c.kernels[κ], c, c.groups[κ], k, bT, model.delta)
    end
    ### the same single gauge exit as the dense assembly: H_per = W H_at W†, one per-nonzero
    ### phase pass (the pattern is untouched, so the workspace stays valid)
    if model.gauge === :periodic
        Ws = _gauge_phases(model.cell, k)
        rows = rowvals(ws.S)
        @inbounds for j in 1:size(ws.S, 2), idx in nzrange(ws.S, j)
            nz[idx] *= Ws[rows[idx]] * conj(Ws[j])
        end
    end
    return ws.S
end

### the sparse twins of the dense `_accumulate!`, kept separate so both stay branchless; the
### kernel function barrier works identically (dispatch once per kernel group per k-point)
@inline function _accumulate_sparse!(nz::Vector{ComplexF64}, slot::Vector{Int}, ::ConstantKernel,
                                     c::CouplingList{D}, r::UnitRange{Int}, k::SVector{D,Float64},
                                     ::SMatrix{D,D,Float64},
                                     δ::Vector{SVector{D,Float64}})::Nothing where {D}
    @inbounds for e in r
        nz[slot[e]] += c.w[e] * bloch_phase(k, c.q[e], δ[e])
    end
    return nothing
end

### the Bloch-sum fast path, matching `_accumulate!`: one scratch allocation per kernel group
@inline function _accumulate_sparse!(nz::Vector{ComplexF64}, slot::Vector{Int},
                                     kern::CallbackKernel{BlochSumEnvelope},
                                     c::CouplingList{2}, r::UnitRange{Int}, k::SVector{2,Float64},
                                     bT::SMatrix{2,2,Float64},
                                     δ::Vector{SVector{2,Float64}})::Nothing
    env = kern.f
    tx, ty = _bloch_scratch(env)
    @inbounds for e in r
        kq = k + c.q[e]
        nz[slot[e]] += c.w[e] * _bloch_value(env, bT * kq, tx, ty) * bloch_phase(kq, δ[e])
    end
    return nothing
end

@inline function _accumulate_sparse!(nz::Vector{ComplexF64}, slot::Vector{Int}, kern,
                                     c::CouplingList{D}, r::UnitRange{Int}, k::SVector{D,Float64},
                                     bT::SMatrix{D,D,Float64},
                                     δ::Vector{SVector{D,Float64}})::Nothing where {D}
    @inbounds for e in r
        p = bT * (k + c.q[e])
        nz[slot[e]] += c.w[e] * _kvalue(kern, p) * bloch_phase(k, c.q[e], δ[e])
    end
    return nothing
end

"""
    sparse_bloch_hamiltonian(model, k_frac) -> SparseMatrixCSC

One-shot sparse `H(k)`. For a k-loop, build a [`sparse_bloch_workspace`](@ref) once and call
[`sparse_bloch_hamiltonian!`](@ref) instead -- the pattern construction is the expensive part.
"""
sparse_bloch_hamiltonian(model::HamiltonianModel{D}, k_frac::AbstractVector{<:Real}) where {D} =
    sparse_bloch_hamiltonian!(sparse_bloch_workspace(model), model, k_frac)
