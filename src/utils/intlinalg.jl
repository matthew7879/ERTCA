# Exact integer linear algebra: adjugates, determinants, inverses, Hermite and Smith-like
# normal forms, coset labels. Everything here is exact -- no floats, no tolerances -- and
# everything is written once at `{D}`, never per dimension. That charter is the whole point of
# the file: a signature pinned to `SMatrix{2,2}` invites the four-character closed form instead
# of the call, and this module's contents were once copied out by hand in five places
# (`geometry/twisted_bilayer.jl`, `hamiltonians/ertca/{model,harmonics}.jl`, `interfaces/moiredb/moiredb.jl`).
# Its float counterpart is `core/unit_cell/reduction.jl` -- lattice reduction, successive
# minima, anything carrying a tolerance. The split is by charter, not by subject.
#
# Consumers: the supercell transforms (transforms/supercell.jl), the congruence duplicate check
# of folded cells (core/unit_cell/unitcell.jl), the moire archive's frame reconstruction
# (interfaces/moiredb/moiredb.jl), and the ERTCA folding (hamiltonians/ertca/).

"""
    _hnf_cols(A) -> SMatrix{D,D,Int}

The column-style Hermite normal form of an integer matrix: `H = A·V` with `V` unimodular, `H`
lower triangular with positive diagonal. Exact integer arithmetic (pairwise Euclid on columns);
the column lattice of `H` is that of `A`, which is all the coset enumeration needs.
"""
function _hnf_cols(A::SMatrix{D,D,Int})::SMatrix{D,D,Int} where {D}
    H = MMatrix{D,D,Int}(A)
    for i in 1:D
        ### pivot: some column j ≥ i with H[i, j] ≠ 0 (exists iff A is non-singular)
        piv = findfirst(j -> H[i, j] != 0, i:D)
        isnothing(piv) && throw(ArgumentError(
            "Supercell transformation matrix must be invertible."))
        j0 = piv + i - 1
        if j0 != i
            for r in 1:D
                H[r, i], H[r, j0] = H[r, j0], H[r, i]
            end
        end
        ### Euclid across the columns to the right until row i is zero there
        for j in (i + 1):D
            while H[i, j] != 0
                q = fld(H[i, j], H[i, i])
                for r in 1:D
                    H[r, j] -= q * H[r, i]
                end
                if H[i, j] != 0            # remainder became the smaller pivot: swap and continue
                    for r in 1:D
                        H[r, i], H[r, j] = H[r, j], H[r, i]
                    end
                end
            end
        end
        if H[i, i] < 0
            for r in 1:D
                H[r, i] = -H[r, i]
            end
        end
    end
    return SMatrix{D,D,Int}(H)
end

"""
    _int_adjugate(A) -> SMatrix{D,D,Int}

The adjugate of an integer matrix, exactly: `adj(A)·A = det(A)·I`, so `adj(A)·x mod |det A|` is
an exact integer label for the coset of `x` in `ℤᴰ / A ℤᴰ` -- the workhorse of every coset
lookup. Closed forms for D ≤ 3; higher dimensions have no `bravais_lattice` anyway.
"""
_int_adjugate(::SMatrix{1,1,Int}) = SMatrix{1,1,Int}(1)
_int_adjugate(A::SMatrix{2,2,Int}) = SMatrix{2,2,Int}(A[2, 2], -A[2, 1], -A[1, 2], A[1, 1])
function _int_adjugate(A::SMatrix{3,3,Int})
    c(i1, i2, j1, j2) = A[i1, j1] * A[i2, j2] - A[i1, j2] * A[i2, j1]
    ### adj[i,j] = (-1)^(i+j) det(A without row j, column i) -- the TRANSPOSED cofactors;
    ### SMatrix fills column-major, so the arguments run adj[1,1], adj[2,1], adj[3,1], adj[1,2]...
    return SMatrix{3,3,Int}(
        c(2, 3, 2, 3), -c(2, 3, 1, 3), c(2, 3, 1, 2),
        -c(1, 3, 2, 3), c(1, 3, 1, 3), -c(1, 3, 1, 2),
        c(1, 2, 2, 3), -c(1, 2, 1, 3), c(1, 2, 1, 2))
end
function _int_adjugate(A::SMatrix{D,D,Int}) where {D}
    throw(ArgumentError("`_int_adjugate` is implemented for D ≤ 3; got D = $D."))
end

"""
    _coset_label(adjA, det_abs, v) -> SVector{D,Int}

The exact integer label of the coset of `v` in `ℤᴰ / A ℤᴰ`, given `adjA = _int_adjugate(A)` and
`det_abs = |det A|`: two integer vectors are congruent mod `A ℤᴰ` iff their labels agree.
"""
@inline _coset_label(adjA::SMatrix{D,D,Int}, det_abs::Int, v::SVector{D,Int}) where {D} =
    SVector{D,Int}(mod.(adjA * v, det_abs))

"""
    _int_det(A) -> Int

The SIGNED determinant of an integer matrix, exactly, for the `D <= 3` `_int_adjugate` covers.
`LinearAlgebra.det` widens an `SMatrix{D,D,Int}` to Float64; this does not, which is what the
unimodularity checks and the rational arithmetic built on them need.

Computed as `(adj(A)*A)[1,1]`, so there is ONE closed form here rather than one per dimension --
`_int_adjugate` already owns the per-`D` cofactor expansions, and a second hand-written
expansion beside it is exactly the duplication this file exists to prevent. At `D = 2` the
adjugate is free (sign flips and a reordering), so the generic form costs what the hand-written
`a11*a22 - a12*a21` cost.

The sign is the point, and it is why this is not `_int_det_abs`: `det == -1` is what
distinguishes an in-plane reflection from a rotation, and the moire database's Cartesian frame
is reconstructed from exactly that distinction (`interfaces/moiredb/moiredb.jl`), as is the layer-swap
predicate of the symmetry-generated continuum models
(`hamiltonians/continuummodel/genericfromsymmetry.jl`).
"""
@inline function _int_det(A::SMatrix{D,D,Int})::Int where {D}
    adj = _int_adjugate(A)
    return sum(ntuple(k -> adj[1, k] * A[k, 1], Val(D)))
end

"""
    _int_det_abs(A) -> Int

`|det A|` of an integer matrix, as an integer. `LinearAlgebra.det` promotes to Float64; the
value is exact for any realistic supercell matrix, so rounding it back is safe.

Deliberately NOT `abs(_int_det(A))`, though they agree wherever both are defined. The two have
different domains on purpose: [`_int_det`](@ref) is exact but goes through `_int_adjugate`, which
has closed forms only for `D <= 3`, while this one is defined at any `D` -- which is what a
`force = true` lattice above three dimensions needs. Where the sign matters, use `_int_det`.
"""
@inline _int_det_abs(A::SMatrix{D,D,Int}) where {D} = round(Int, abs(det(SMatrix{D,D,Float64}(A))))

"""
    _int_inv(A) -> SMatrix{D,D,Int}

The exact inverse of a UNIMODULAR integer matrix: `adj(A)/det(A)` with `|det A| = 1`, so the
division is exact and the result is integral.

Refuses anything else BY NAME. A matrix with `|det| != 1` has no integer inverse, and an
integer division that silently truncates would produce a plausible matrix that is not an
inverse -- the shape of wrong answer this package ranks highest.
"""
function _int_inv(A::SMatrix{D,D,Int})::SMatrix{D,D,Int} where {D}
    d = _int_det(A)
    abs(d) == 1 || throw(ArgumentError(
        "`_int_inv`: only a unimodular matrix has an integer inverse; got |det| = $(abs(d)) " *
        "!= 1 for $A."))
    return _int_adjugate(A) .÷ d
end

"""
    _int_invT(A) -> SMatrix{D,D,Rational{Int}}

`A^-T` as exact rationals, `transpose(adj(A))//det(A)`: the action of a supercell or
point-group matrix on FRACTIONAL RECIPROCAL coordinates.

Rational where [`_int_inv`](@ref) is integral, because this one is exact for ANY non-singular
`A`, not only unimodular ones -- a folded cell's period matrix has `|det| = n > 1` and its
reciprocal action genuinely has denominators.
"""
@inline _int_invT(A::SMatrix{D,D,Int}) where {D} =
    SMatrix{D,D,Rational{Int}}(transpose(_int_adjugate(A)) .// _int_det(A))

"""
    _int_solve(adjA, detA, v) -> Union{Nothing,SVector{D,Int}}

The exact integer solution of `A*x = v`, or `nothing` when `v` is not in `A*Z^D`.

Takes the precomputed `adjA = _int_adjugate(A)` and `detA = _int_det(A)` rather than `A`,
because every caller solves against one fixed `A` for many `v`. `adj(A)*v = 0 (mod det A)`
componentwise IS the membership test, so the test and the quotient come from one product --
which is also why this cannot be phrased as a division that might fail.
"""
@inline function _int_solve(adjA::SMatrix{D,D,Int}, detA::Int,
                            v::SVector{D,Int})::Union{Nothing,SVector{D,Int}} where {D}
    num = adjA * v
    all(ntuple(m -> mod(num[m], detA) == 0, Val(D))) || return nothing
    return SVector{D,Int}(ntuple(m -> num[m] ÷ detA, Val(D)))
end

"""
    _unimodular_diagonalize(M; nblock=D) -> (L, Λ, R)

Diagonalize an integer matrix by unimodular row and column operations, exactly:
`Λ = L * M * R` with `L`, `R` unimodular (|det| = 1) and `Λ` diagonal with positive entries.

This is the Smith decomposition up to its divisor chain, which is deliberately NOT enforced:
no consumer here needs `Λ[1,1] | Λ[2,2]`, and enforcing it would mix axes gratuitously (the
true Smith form of `diag(2, 3)` is `diag(1, 6)` -- a worse mesh shape for the same momentum
set). The consumer is `bz_mesh(lattice, M)`: the cluster momenta `M⁻¹(m + φ) = Rᵀᵀ...` reduce
through this to a rectangular grid `Λ` sheared by `Rᵀ` and shifted by `L·φ`.

`nblock` restricts the pivot search to the leading `nblock × nblock` block; the trailing block
must already be the identity (asserted). This keeps a `blockdiag(M_P, I)` input -- the only
legal supercell shape on a partially periodic lattice -- block-diagonal in the output, so row
and column operations never mix periodic and aperiodic axes.
"""
function _unimodular_diagonalize(M::SMatrix{D,D,Int};
                                 nblock::Int=D)::NTuple{3,SMatrix{D,D,Int}} where {D}
    _int_det_abs(M) == 0 && throw(ArgumentError(
        "`_unimodular_diagonalize`: the matrix must be invertible; got det = 0 for $M."))
    (1 <= nblock <= D) || throw(ArgumentError("`nblock` must lie in 1:$D; got $nblock."))
    for i in (nblock + 1):D, j in 1:D
        (M[i, j] == (i == j ? 1 : 0) && M[j, i] == (i == j ? 1 : 0)) || throw(ArgumentError(
            "`_unimodular_diagonalize` with nblock = $nblock needs the trailing block to be " *
            "the identity; got $M."))
    end
    A = MMatrix{D,D,Int}(M)
    L = MMatrix{D,D,Int}(one(SMatrix{D,D,Int}))
    R = MMatrix{D,D,Int}(one(SMatrix{D,D,Int}))
    for t in 1:nblock
        while true
            ### pivot: the minimal-|entry| non-zero of the remaining block, moved to (t, t).
            ### |pivot| never increases and strictly decreases whenever a Euclid step leaves a
            ### remainder, so the loop terminates.
            bi, bj, best = 0, 0, 0
            for j in t:nblock, i in t:nblock
                a = abs(A[i, j])
                if a != 0 && (best == 0 || a < best)
                    bi, bj, best = i, j, a
                end
            end
            best == 0 && throw(ArgumentError(
                "`_unimodular_diagonalize`: the matrix must be invertible; got $M."))
            if bi != t
                for c in 1:D
                    A[t, c], A[bi, c] = A[bi, c], A[t, c]
                    L[t, c], L[bi, c] = L[bi, c], L[t, c]
                end
            end
            if bj != t
                for r in 1:D
                    A[r, t], A[r, bj] = A[r, bj], A[r, t]
                    R[r, t], R[r, bj] = R[r, bj], R[r, t]
                end
            end
            clean = true
            for i in (t + 1):nblock
                q = fld(A[i, t], A[t, t])
                if q != 0
                    for c in 1:D
                        A[i, c] -= q * A[t, c]
                        L[i, c] -= q * L[t, c]
                    end
                end
                A[i, t] == 0 || (clean = false)
            end
            for j in (t + 1):nblock
                q = fld(A[t, j], A[t, t])
                if q != 0
                    for r in 1:D
                        A[r, j] -= q * A[r, t]
                        R[r, j] -= q * R[r, t]
                    end
                end
                A[t, j] == 0 || (clean = false)
            end
            clean && break
        end
        if A[t, t] < 0
            for c in 1:D
                A[t, c] = -A[t, c]
                L[t, c] = -L[t, c]
            end
        end
    end
    return SMatrix{D,D,Int}(L), SMatrix{D,D,Int}(A), SMatrix{D,D,Int}(R)
end


#############################################################
###### The one Bloch phase                              #####
#############################################################

"""
    bloch_phase(kq, δ) -> ComplexF64
    bloch_phase(k, q, δ) -> ComplexF64

THE constant-kernel Bloch phase, `exp(2πi (k + q)·δ)`, in one place.

It is one line, and it was written out SEVEN times: three times in the dense assembly
(`bloch_hamiltonian!` -- constant, k-kernel and envelope paths), three times in the sparse one,
and once in `_apply_bilinear!`. The envelope paths hid behind a different spelling, forming
`kq = k + c.q[e]` first and passing that, which is why the two-argument method exists.

That is not a style complaint. `BilinearOperator` was built by copying `i`, `j`, `δ` and `w`
off the compiled coupling list and re-deriving this expression, and its copy silently omitted
the `+ q` term -- so on any cell whose couplings carry a momentum offset (a fold, a G-labelled
plane-wave cell, a `k0` frame) it evaluated a DIFFERENT operator, off by O(bandwidth), with
nothing downstream able to tell. Patching that one copy fixes the instance; a seventh copy
would omit something else.

`q` is the entry's OWN offset (`CouplingList.q`), not the transfer of a two-point function: a
bilinear's midpoint convention is the caller's business and arrives here already folded into
the first argument.
"""
@inline function bloch_phase(kq::SVector{D,Float64}, δ::SVector{D,Float64})::ComplexF64 where {D}
    return cispi(2 * dot(kq, δ))
end

@inline function bloch_phase(k::SVector{D,Float64}, q::SVector{D,Float64},
                             δ::SVector{D,Float64})::ComplexF64 where {D}
    return bloch_phase(k + q, δ)
end

"""
    _int_kernel(B::SMatrix{D,D,Int}) -> Vector{SVector{D,Int}}

A `Z`-basis of `ker(B) ∩ Z^D`, exactly: every integer vector `v` with `B * v = 0`, as the
generators of that sublattice, each divided by the gcd of its entries and sign-normalised so its
first non-zero entry is positive.

Unimodular COLUMN reduction `B * R = H` by pairwise Euclid across columns, accumulating `R`; the
kernel is the set of columns of `R` whose column of `H` is zero. It cannot reuse
[`_hnf_cols`](@ref), which throws on a singular pivot -- and every input here is singular by
construction, because the whole point is to find what `B` annihilates.

This is how a rotation AXIS is obtained from a point-group operation: `_int_kernel(W - I)` is the
invariant direction of `W`, exactly, with no tolerance and no search. A BOUNDED COORDINATE SEARCH
is the trap it replaces -- a `±6` scan loses real monoclinic axes, and a `±4` unimodular
enumeration on an unreduced basis loses far more, because a skew basis can put the axis at
coordinates in the dozens.
"""
function _int_kernel(B::SMatrix{D,D,Int}) where {D}
    H = MMatrix{D,D,Int}(B)
    R = MMatrix{D,D,Int}(one(SMatrix{D,D,Int}))
    lead = 0
    for r in 1:D
        c0 = lead + 1
        c0 > D && break
        while true
            ### pivot on the smallest non-zero entry of this row, so Euclid terminates fast
            cpiv, best = 0, 0
            for c in c0:D
                H[r, c] == 0 && continue
                (cpiv == 0 || abs(H[r, c]) < best) && (cpiv = c; best = abs(H[r, c]))
            end
            cpiv == 0 && break
            if cpiv != c0
                for i in 1:D
                    H[i, c0], H[i, cpiv] = H[i, cpiv], H[i, c0]
                    R[i, c0], R[i, cpiv] = R[i, cpiv], R[i, c0]
                end
            end
            leftover = false
            for c in (c0 + 1):D
                H[r, c] == 0 && continue
                q = div(H[r, c], H[r, c0])
                for i in 1:D
                    H[i, c] -= q * H[i, c0]
                    R[i, c] -= q * R[i, c0]
                end
                H[r, c] != 0 && (leftover = true)
            end
            leftover || break
        end
        H[r, c0] != 0 && (lead = c0)
    end
    out = SVector{D,Int}[]
    for c in (lead + 1):D
        v = SVector{D,Int}(ntuple(i -> R[i, c], Val(D)))
        g = foldl(gcd, abs.(v); init=0)
        g > 0 || continue
        v = v .÷ g
        ### sign-normalise: the first non-zero entry positive, so one axis has one spelling
        k = findfirst(!iszero, v)
        (k !== nothing && v[k] < 0) && (v = -v)
        push!(out, v)
    end
    return out
end

"""
    _sublattice_cosets(V::SMatrix{D,D,Int}) -> Vector{SVector{D,Rational{Int}}}

The non-zero coset representatives of the primitive lattice in the conventional one, exactly.

With `conv = V * prim` (rows are vectors, the package's convention), row `i` of `V⁻¹` gives the
conventional fractional coordinates of primitive row `i`; the ones that are not already integral
are the centring vectors. `(0, ½, ½)` and friends fall out of this with no table and no
tolerance -- the arithmetic is `Rational{Int}` throughout.

This is what separates `oI` from `oS` once the index `|det V| = 2` is known: body centring gives
`(½, ½, ½)`, one-face centring does not.

**Use [`_int_adjugate`](@ref) here, not a hand-written adjugate.** A transposed one is the trap:
it misread `oI` as `oS` on 83 of 120 scrambled cells and `oS` as `oI` on 16 of 120, with no other
type affected and no error raised anywhere -- the two are exactly the pair this function exists
to tell apart, so the mistake is invisible until it is the answer.
"""
function _sublattice_cosets(V::SMatrix{D,D,Int}) where {D}
    d = _int_det(V)
    d == 0 && throw(ArgumentError("`_sublattice_cosets`: `V` must be invertible; got det = 0."))
    adj = _int_adjugate(V)
    out = SVector{D,Rational{Int}}[]
    for i in 1:D
        v = SVector{D,Rational{Int}}(ntuple(j -> mod(adj[i, j] // d, 1), Val(D)))
        ### distinct representatives only: all three primitive rows of a body-centred cell share
        ### the coset (1/2, 1/2, 1/2), and reporting it three times would suggest three centrings
        (all(iszero, v) || v in out) && continue
        push!(out, v)
    end
    return out
end





### ------------------------------------------------------------------------------------
### Recovering an integer matrix from two float lattice bases
### ------------------------------------------------------------------------------------
###
### The float counterpart of the exact arithmetic above: given two sets of lattice vectors,
### decide whether one is an integer superlattice of the other and return the integer matrix
### if it is. It belongs here rather than with the supercell transform that used to own it --
### `spectra/bz_mesh.jl` needs the same recovery to read a cluster matrix, and reaching into
### `transforms/` for it was the only reason `spectra` referenced that layer at all.

"""
    _supercell_integer_matrix(new_a::AbstractMatrix{<:Real}, a_old::AbstractMatrix{<:Real}; atol::Float64=INT_ROUND_ATOL) -> SMatrix{D,D,Int}

Compute the integer supercell transformation matrix `M` such that
`new_a ≈ M * a_old`. This validates that `new_a` is a real-space supercell of the
original lattice within tolerance.
"""
function _supercell_integer_matrix(new_a::AbstractMatrix{<:Real}, a_old::AbstractMatrix{<:Real}; atol::Float64=INT_ROUND_ATOL)::AbstractMatrix{<:Integer}
    D = size(a_old, 1)
    size(a_old) == (D, D) || throw(DimensionMismatch("Original lattice matrix must be square."))
    size(new_a) == (D,D) || throw(DimensionMismatch("New lattice matrix must have the same shape as the original lattice."))

    M = new_a * inv(a_old)
    if !isapprox(M, round.(M); atol=atol)
        throw(DomainError("New lattice vectors are not an integer multiple of the original lattice."))
    end
    return round.(Int, M)
end
